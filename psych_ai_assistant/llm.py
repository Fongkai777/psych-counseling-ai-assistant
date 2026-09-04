import json
import logging
import http.client
import re
import socket
import time
import urllib.error
import urllib.request


LOGGER = logging.getLogger(__name__)
DEFAULT_MAX_EMBEDDING_INPUT_CHARS = 8000
DASHSCOPE_EMBEDDING_MAX_INPUT_CHARS = 8000
DASHSCOPE_EMBEDDING_BATCH_SIZE = 10


class EmbeddingAPIError(Exception):
    pass


class LLMClient:
    def __init__(self, config):
        self.api_key = config.get("LLM_API_KEY") or config.get("DASHSCOPE_API_KEY") or ""
        self.dashscope_api_key = config.get("DASHSCOPE_API_KEY") or self.api_key
        self.provider = (config.get("LLM_PROVIDER") or "openai_compatible").strip()
        self.base_url = (config.get("LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = config.get("LLM_MODEL") or "gpt-5.4-mini"
        self.temperature = float(config.get("LLM_TEMPERATURE") or 0.7)
        self.enable_thinking = str(config.get("LLM_ENABLE_THINKING", "")).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.embedding_api_key = config.get("EMBEDDING_API_KEY") or self.api_key
        self.embedding_base_url = (
            config.get("EMBEDDING_BASE_URL") or self.base_url
        ).rstrip("/")
        self.embedding_model = config.get("EMBEDDING_MODEL") or "text-embedding-3-small"
        self.embedding_batch_size = max(
            1,
            min(int(config.get("EMBEDDING_BATCH_SIZE") or 5), self._provider_embedding_batch_cap()),
        )
        configured_embedding_max = int(
            config.get("EMBEDDING_MAX_INPUT_CHARS") or DEFAULT_MAX_EMBEDDING_INPUT_CHARS
        )
        self.embedding_max_input_chars = max(
            1000,
            min(configured_embedding_max, self._provider_embedding_input_cap()),
        )
        self.rerank_enabled = str(config.get("RERANK_ENABLED", "")).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.rerank_model = config.get("RERANK_MODEL") or "qwen3-vl-rerank"
        self.last_embedding_error = ""

    def status(self):
        return {
            "mode": self.provider if self.api_key else "demo",
            "model": self.model,
            "base_url": self.base_url,
            "embedding_mode": "api" if self.embedding_api_key else "local",
            "embedding_model": self.embedding_model,
            "embedding_batch_size": self.embedding_batch_size,
            "embedding_max_input_chars": self.embedding_max_input_chars,
            "rerank_mode": "dashscope_api"
            if self.rerank_enabled and self.dashscope_api_key
            else "off",
            "rerank_model": self.rerank_model,
        }

    def chat_payload(self, prompt, enable_thinking=None):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        use_thinking = self.enable_thinking if enable_thinking is None else enable_thinking
        if use_thinking:
            payload["enable_thinking"] = True
        return payload

    def chat_curl(self, prompt):
        if self.provider == "dashscope_native":
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
            }
            return (
                "# DashScope native SDK call\n"
                "import dashscope\n"
                "dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'\n"
                "response = dashscope.MultiModalConversation.call(\n"
                "  api_key='***REDACTED***',\n"
                f"  model={self.model!r},\n"
                f"  messages={payload['messages']!r},\n"
                ")\n"
            )
        payload = self.chat_payload(prompt)
        return (
            "curl -X POST "
            f"'{self.base_url}/chat/completions' \\\n"
            "  -H 'Authorization: Bearer ***REDACTED***' \\\n"
            "  -H 'Content-Type: application/json' \\\n"
            "  -d "
            + repr(json.dumps(payload, ensure_ascii=False, indent=2))
        )

    def generate(self, prompt, enable_thinking=None, timeout=90):
        if not self.api_key:
            return self.demo_answer(prompt)
        if self.provider == "dashscope_native":
            return self.generate_dashscope_native(prompt)

        payload = self.chat_payload(prompt, enable_thinking=enable_thinking)
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except (
            urllib.error.URLError,
            socket.timeout,
            TimeoutError,
            KeyError,
            IndexError,
            json.JSONDecodeError,
        ) as exc:
            return (
                "模型 API 调用失败，以下是本地演示草稿。\n\n"
                f"失败原因：{exc}\n\n"
                + self.demo_answer(prompt)
            )

    def generate_dashscope_native(self, prompt):
        try:
            import dashscope

            dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
            response = dashscope.MultiModalConversation.call(
                api_key=self.dashscope_api_key,
                model=self.model,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
            )
            status_code = getattr(response, "status_code", None)
            if status_code and int(status_code) != 200:
                raise RuntimeError(
                    f"DashScope status={status_code}, code={getattr(response, 'code', '')}, "
                    f"message={getattr(response, 'message', '')}"
                )
            content = response.output.choices[0].message.content
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    return item["text"]
            raise RuntimeError("DashScope response has no text content")
        except Exception as exc:
            LOGGER.error("DashScope native generation failed: %s", exc)
            return (
                "模型 API 调用失败，以下是本地演示草稿。\n\n"
                f"失败原因：{exc}\n\n"
                + self.demo_answer(prompt)
            )

    def clean_document(self, title, content):
        prompt = f"""请把下面从 PDF/docx 中抽取出的混乱文本整理成适合 RAG 入库的中文 Markdown 知识卡片。

要求：
1. 修复明显的断行、页眉页脚、重复空白和目录噪音。
2. 保留原文观点，不要新增书中没有的概念、数据或作者。
3. 用小标题、要点和短段落组织。
4. 如果文本像心理学/咨询材料，保留概念定义、适用边界、风险提醒和可操作建议。
5. 输出整理后的正文，不要解释你的整理过程。

【资料标题】
{title}

【原始文本】
{content[:18000]}
"""
        return self.generate(prompt)

    def embed_texts(self, texts, batch_size=None, progress_callback=None, raise_on_error=False):
        if not self.embedding_api_key:
            return []
        texts = [self._fit_embedding_input(text) for text in texts]
        embeddings = []
        batch_size = max(1, int(batch_size or self.embedding_batch_size))
        batch_size = min(self._provider_embedding_batch_cap(), batch_size)
        total_batches = (len(texts) + batch_size - 1) // batch_size if texts else 0
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            try:
                batch_embeddings = self._embed_text_batch_resilient(batch, raise_on_error)
            except EmbeddingAPIError:
                if raise_on_error:
                    raise
                batch_embeddings = []
            if len(batch_embeddings) == len(batch):
                embeddings.extend(batch_embeddings)
            else:
                message = (
                    f"Embedding 返回数量不匹配：expected={len(batch)}, "
                    f"actual={len(batch_embeddings)}, model={self.embedding_model}"
                )
                self.last_embedding_error = message
                LOGGER.error(message)
                if raise_on_error:
                    raise EmbeddingAPIError(message)
                embeddings.extend([None] * len(batch))
            if progress_callback:
                progress_callback(min(total_batches, start // batch_size + 1), total_batches)
        return embeddings

    def _provider_embedding_input_cap(self):
        model = (self.embedding_model or "").lower()
        base_url = (self.embedding_base_url or "").lower()
        if "dashscope.aliyuncs.com" in base_url and model == "text-embedding-v4":
            return DASHSCOPE_EMBEDDING_MAX_INPUT_CHARS
        return 30000

    def _provider_embedding_batch_cap(self):
        model = (self.embedding_model or "").lower()
        base_url = (self.embedding_base_url or "").lower()
        if "dashscope.aliyuncs.com" in base_url and model == "text-embedding-v4":
            return DASHSCOPE_EMBEDDING_BATCH_SIZE
        return 20

    def _fit_embedding_input(self, text, max_chars=None):
        text = text or ""
        max_chars = int(max_chars or self.embedding_max_input_chars)
        if len(text) <= max_chars:
            return text
        head_size = max_chars // 2
        tail_size = max_chars // 3
        middle_size = max_chars - head_size - tail_size - 80
        middle_start = max(0, (len(text) - middle_size) // 2)
        fitted = (
            text[:head_size]
            + "\n\n[... embedding input truncated: middle sample ...]\n\n"
            + text[middle_start : middle_start + middle_size]
            + "\n\n[... embedding input truncated: tail sample ...]\n\n"
            + text[-tail_size:]
        )
        LOGGER.warning(
            "Embedding input truncated from %s to %s chars for model=%s",
            len(text),
            min(len(fitted), max_chars),
            self.embedding_model,
        )
        return fitted[:max_chars]

    def _parse_embedding_input_limit(self, body):
        match = re.search(r"Range of input length should be \[1,\s*(\d+)\]", body or "")
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    def _embed_text_batch_resilient(self, texts, raise_on_error=False):
        try:
            return self._embed_text_batch(texts)
        except EmbeddingAPIError:
            if len(texts) <= 1:
                raise
            midpoint = max(1, len(texts) // 2)
            LOGGER.warning(
                "Embedding batch failed; splitting batch %s into %s + %s",
                len(texts),
                midpoint,
                len(texts) - midpoint,
            )
            left = self._embed_text_batch_resilient(texts[:midpoint], raise_on_error)
            right = self._embed_text_batch_resilient(texts[midpoint:], raise_on_error)
            return left + right

    def _embed_text_batch(self, texts):
        payload = {"model": self.embedding_model, "input": texts}
        char_count = sum(len(text or "") for text in texts)
        req = urllib.request.Request(
            f"{self.embedding_base_url}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.embedding_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt in range(1, 3):
            try:
                with urllib.request.urlopen(req, timeout=90) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return [item["embedding"] for item in data["data"]]
            except http.client.IncompleteRead as exc:
                message = (
                    f"Embedding API 响应读取不完整：attempt={attempt}, "
                    f"model={self.embedding_model}, url={self.embedding_base_url}/embeddings, "
                    f"batch={len(texts)}, chars={char_count}, "
                    f"read={len(exc.partial)}, expected_more={exc.expected}"
                )
                self.last_embedding_error = message
                LOGGER.warning(message)
                if attempt < 2:
                    time.sleep(1)
                    continue
                raise EmbeddingAPIError(message) from exc
            except urllib.error.HTTPError as exc:
                try:
                    body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    body = ""
                api_limit = self._parse_embedding_input_limit(body)
                if len(texts) == 1 and api_limit and char_count > api_limit:
                    safe_limit = max(1000, min(self.embedding_max_input_chars, api_limit - 300))
                    self.embedding_max_input_chars = safe_limit
                    shortened = self._fit_embedding_input(texts[0], max_chars=safe_limit)
                    if shortened != texts[0]:
                        LOGGER.warning(
                            "Embedding API reported input limit %s; retrying with %s chars",
                            api_limit,
                            len(shortened),
                        )
                        return self._embed_text_batch([shortened])
                message = (
                    f"Embedding API HTTP {exc.code}: model={self.embedding_model}, "
                    f"url={self.embedding_base_url}/embeddings, batch={len(texts)}, "
                    f"chars={char_count}, body={body[:1200]}"
                )
                self.last_embedding_error = message
                LOGGER.error(message)
                raise EmbeddingAPIError(message) from exc
            except (
                urllib.error.URLError,
                socket.timeout,
                TimeoutError,
                KeyError,
                json.JSONDecodeError,
            ) as exc:
                message = (
                    f"Embedding API 调用失败：attempt={attempt}, {type(exc).__name__}: {exc}; "
                    f"model={self.embedding_model}, url={self.embedding_base_url}/embeddings, "
                    f"batch={len(texts)}, chars={char_count}"
                )
                self.last_embedding_error = message
                LOGGER.warning(message)
                if attempt < 2:
                    time.sleep(1)
                    continue
                raise EmbeddingAPIError(message) from exc

    def rerank(self, query, candidates, top_n=5):
        if not self.api_key or not self.rerank_enabled or not candidates:
            return candidates[:top_n]
        payload = []
        for index, item in enumerate(candidates[:20], start=1):
            payload.append(
                {
                    "index": index,
                    "title": item.get("title", ""),
                    "mode": item.get("mode", ""),
                    "score": item.get("score", 0),
                    "content": (item.get("content") or "")[:900],
                }
            )
        prompt = f"""请根据用户问题，对候选 RAG 片段按“是否有助于回答问题”重新排序。

要求：
1. 只输出 JSON。
2. ranked_indexes 里放候选 index，最相关的排最前。
3. 最多返回 {top_n} 个 index。
4. 不要新增候选之外的内容。

【用户问题】
{query}

【候选片段】
{json.dumps(payload, ensure_ascii=False, indent=2)}

输出格式：
{{"ranked_indexes": [1, 3, 2], "reason": "简短说明"}}
"""
        raw = self.generate(prompt)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            data = {}
            if start >= 0 and end > start:
                try:
                    data = json.loads(raw[start : end + 1])
                except json.JSONDecodeError:
                    data = {}
        ranked = []
        seen = set()
        for value in data.get("ranked_indexes", []):
            try:
                idx = int(value) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(candidates) and idx not in seen:
                item = dict(candidates[idx])
                item["rerank_rank"] = len(ranked) + 1
                item["rerank_reason"] = data.get("reason", "")
                ranked.append(item)
                seen.add(idx)
            if len(ranked) >= top_n:
                break
        for idx, item in enumerate(candidates):
            if len(ranked) >= top_n:
                break
            if idx not in seen:
                ranked.append(item)
        return ranked[:top_n]

    def rerank_texts(self, query, documents, top_n=5, timeout=90):
        if not self.dashscope_api_key or not self.rerank_enabled or not documents:
            return []
        documents = [(doc or "")[:6000] for doc in documents if (doc or "").strip()]
        if not documents:
            return []
        payload = {
            "model": self.rerank_model,
            "input": {
                "query": query,
                "documents": documents[:50],
            },
            "parameters": {
                "return_documents": True,
                "top_n": max(1, min(int(top_n or 5), len(documents))),
            },
        }
        req = urllib.request.Request(
            "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.dashscope_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            self.last_embedding_error = (
                f"Rerank API HTTP {exc.code}: model={self.rerank_model}, body={body[:800]}"
            )
            LOGGER.warning(self.last_embedding_error)
            return []
        except (
            urllib.error.URLError,
            socket.timeout,
            TimeoutError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            self.last_embedding_error = f"Rerank API 调用失败：{type(exc).__name__}: {exc}"
            LOGGER.warning(self.last_embedding_error)
            return []

        raw_results = (
            data.get("output", {}).get("results")
            or data.get("output", {}).get("result")
            or data.get("results")
            or []
        )
        results = []
        used_indexes = set()
        for item in raw_results:
            index = None
            document = item.get("document") or item.get("text") or item.get("content") or ""
            if isinstance(document, dict):
                document = document.get("text") or document.get("content") or ""
            if document:
                for candidate_index, candidate in enumerate(documents):
                    if candidate_index not in used_indexes and candidate == document:
                        index = candidate_index
                        break
            if index is None:
                index = item.get("index")
            try:
                index = int(index)
            except (TypeError, ValueError):
                continue
            if not 0 <= index < len(documents) and 1 <= index <= len(documents):
                index = index - 1
            if not 0 <= index < len(documents):
                continue
            used_indexes.add(index)
            score = (
                item.get("relevance_score")
                if item.get("relevance_score") is not None
                else item.get("score", 0)
            )
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.0
            results.append(
                {
                    "index": index,
                    "score": score,
                    "document": document or documents[index],
                }
            )
        return results[:top_n]

    def demo_answer(self, prompt):
        title = "这个问题背后，可能不是简单的“不够努力”。"
        for line in prompt.splitlines():
            if line.strip() and "【" not in line and len(line.strip()) > 8:
                title = line.strip()
                break

        return f"""我会先把这个问题理解成：你已经意识到自己卡住了，但单靠“提醒自己应该改变”并没有真的让事情变容易。

这类困扰通常不只是意志力问题。很多时候，人会在行动前遇到三层阻力：第一层是任务太大，大到一想到就想逃；第二层是对失败或评价的担心；第三层是长期形成的自我保护方式。它们不一定合理，但往往有来处。

可以先试试这样处理：

1. 先把问题缩小到一个可观察的场景  
不要先问“我为什么总是这样”，而是问：“我最近一次卡住，是在什么时候、面对什么任务、脑子里闪过了什么念头？”

2. 把建议降到足够小  
如果目标是“写一篇完整回答”，第一步可以只是打开文档，写下三个关键词。能开始，比一开始就完美更重要。

3. 区分情绪和事实  
“我做不到”是一种感受，不一定是事实。可以把它改写成：“我现在对这件事有压力，所以我倾向于逃开。”

4. 给自己留一个求助入口  
如果这种状态已经持续影响睡眠、学习、关系，或者伴随强烈绝望感，建议找可信的人聊聊，必要时寻求专业心理咨询支持。

我不太建议把它简单归结为懒。更准确地说，也许是你还没有找到一个足够安全、足够小的开始方式。

选题备注：{title}
"""
