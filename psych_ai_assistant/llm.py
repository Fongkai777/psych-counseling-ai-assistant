import json
import urllib.error
import urllib.request


class LLMClient:
    def __init__(self, config):
        self.api_key = config.get("LLM_API_KEY", "")
        self.base_url = (config.get("LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = config.get("LLM_MODEL") or "gpt-5.4-mini"
        self.temperature = float(config.get("LLM_TEMPERATURE") or 0.7)
        self.embedding_api_key = config.get("EMBEDDING_API_KEY") or self.api_key
        self.embedding_base_url = (
            config.get("EMBEDDING_BASE_URL") or self.base_url
        ).rstrip("/")
        self.embedding_model = config.get("EMBEDDING_MODEL") or "text-embedding-3-small"

    def status(self):
        return {
            "mode": "api" if self.api_key else "demo",
            "model": self.model,
            "base_url": self.base_url,
            "embedding_mode": "api" if self.embedding_api_key else "local",
            "embedding_model": self.embedding_model,
        }

    def chat_payload(self, prompt):
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }

    def chat_curl(self, prompt):
        payload = self.chat_payload(prompt)
        return (
            "curl -X POST "
            f"'{self.base_url}/chat/completions' \\\n"
            "  -H 'Authorization: Bearer ***REDACTED***' \\\n"
            "  -H 'Content-Type: application/json' \\\n"
            "  -d "
            + repr(json.dumps(payload, ensure_ascii=False, indent=2))
        )

    def generate(self, prompt):
        if not self.api_key:
            return self.demo_answer(prompt)

        payload = self.chat_payload(prompt)
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
            with urllib.request.urlopen(req, timeout=90) as response:
                data = json.loads(response.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as exc:
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

    def embed_texts(self, texts):
        if not self.embedding_api_key:
            return []
        embeddings = []
        batch_size = 8
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            batch_embeddings = self._embed_text_batch(batch)
            if len(batch_embeddings) == len(batch):
                embeddings.extend(batch_embeddings)
            else:
                embeddings.extend([None] * len(batch))
        return embeddings

    def _embed_text_batch(self, texts):
        payload = {"model": self.embedding_model, "input": texts}
        req = urllib.request.Request(
            f"{self.embedding_base_url}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.embedding_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                data = json.loads(response.read().decode("utf-8"))
            return [item["embedding"] for item in data["data"]]
        except (urllib.error.URLError, KeyError, json.JSONDecodeError):
            return []

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
