import hashlib
import json
import math
import re
from pathlib import Path

from psych_ai_assistant.prompts import (
    build_expression_emotion_prompt,
    build_quote_picker_prompt,
)


class ExpressionRetriever:
    """Retrieve reusable personal expression snippets for the current answer."""

    def __init__(self, library_path, cache_path=None):
        self.library_path = Path(library_path)
        self.cache_path = Path(cache_path) if cache_path else None

    def retrieve(self, question, intent, llm, top_k=8, final_limit=3):
        snippets = self.load_snippets()
        if not snippets:
            return {"emotion_summary": "", "candidates": [], "selected": []}

        emotion_prompt = build_expression_emotion_prompt(question, intent)
        emotion_summary = llm.generate(emotion_prompt, enable_thinking=False, timeout=90)
        candidates = self.retrieve_candidates(
            emotion_summary,
            snippets,
            llm,
            top_k=top_k,
        )
        selected = self.pick_quotes(
            question,
            intent,
            emotion_summary,
            candidates,
            llm,
            final_limit=final_limit,
        )
        return {
            "emotion_summary": emotion_summary,
            "candidates": candidates,
            "selected": selected,
        }

    def load_snippets(self):
        if not self.library_path.exists():
            return []
        text = self.library_path.read_text(encoding="utf-8")
        sections = re.split(r"\n(?=## \d+\.)", text)
        snippets = []
        for section in sections:
            title_match = re.search(r"^##\s+(\d+)\.\s+(.+)$", section, re.MULTILINE)
            quote_match = re.search(r"^>\s*(.+(?:\n> .+)*)", section, re.MULTILINE)
            if not title_match or not quote_match:
                continue
            quote = re.sub(r"\n>\s*", "\n", quote_match.group(1)).strip()
            item = {
                "id": title_match.group(1),
                "title": title_match.group(2).strip(),
                "themes": self.field(section, "主题"),
                "usage": self.field(section, "适用场景"),
                "reuse_mode": self.field(section, "复用方式"),
                "weight": self.field(section, "权重"),
                "source": self.field(section, "来源"),
                "text": quote,
            }
            item["profile_text"] = "\n".join(
                part
                for part in [
                    f"标题：{item['title']}",
                    f"主题：{item['themes']}",
                    f"适用场景：{item['usage']}",
                    f"复用方式：{item['reuse_mode']}",
                    f"片段：{item['text']}",
                ]
                if part.strip()
            )
            snippets.append(item)
        return snippets

    def field(self, section, name):
        match = re.search(rf"^{re.escape(name)}：(.+)$", section, re.MULTILINE)
        return match.group(1).strip() if match else ""

    def retrieve_candidates(self, emotion_summary, snippets, llm, top_k=8):
        if not getattr(llm, "embedding_api_key", ""):
            return snippets[:top_k]

        cache = self.load_cache(llm)
        embeddings = cache.get("embeddings", {})
        missing = [item for item in snippets if item["id"] not in embeddings]
        if missing:
            texts = [item["profile_text"] for item in missing]
            vectors = llm.embed_texts(texts, raise_on_error=False)
            for item, vector in zip(missing, vectors):
                if vector:
                    embeddings[item["id"]] = vector
            cache["embeddings"] = embeddings
            self.write_cache(cache, llm)

        query_vector = llm.embed_texts([emotion_summary], raise_on_error=False)
        if not query_vector or not query_vector[0]:
            return snippets[:top_k]

        scored = []
        for item in snippets:
            vector = embeddings.get(item["id"])
            if not vector:
                continue
            scored.append(
                {
                    **item,
                    "score": round(self.cosine(query_vector[0], vector), 4),
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k] or snippets[:top_k]

    def pick_quotes(self, question, intent, emotion_summary, candidates, llm, final_limit=3):
        if not candidates:
            return []
        if not getattr(llm, "api_key", ""):
            return candidates[:final_limit]

        prompt = build_quote_picker_prompt(
            question,
            intent,
            emotion_summary,
            candidates,
            final_limit=final_limit,
        )
        raw = llm.generate(prompt, enable_thinking=False, timeout=90)
        data = self.parse_json(raw)
        selected_ids = data.get("selected_ids") or []
        selected = []
        by_id = {item["id"]: item for item in candidates}
        for value in selected_ids:
            item_id = str(value).strip()
            if item_id in by_id:
                item = dict(by_id[item_id])
                item["picker_reason"] = data.get("reason", "")
                selected.append(item)
            if len(selected) >= final_limit:
                break
        return selected

    def parse_json(self, raw):
        try:
            return json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            pass
        raw = raw or ""
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return {}
        return {}

    def load_cache(self, llm):
        if not self.cache_path or not self.cache_path.exists():
            return self.empty_cache(llm)
        try:
            cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self.empty_cache(llm)
        if (
            cache.get("library_sha1") != self.library_sha1()
            or cache.get("embedding_model") != getattr(llm, "embedding_model", "")
        ):
            return self.empty_cache(llm)
        return cache

    def write_cache(self, cache, llm):
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache["library_sha1"] = self.library_sha1()
        cache["embedding_model"] = getattr(llm, "embedding_model", "")
        self.cache_path.write_text(
            json.dumps(cache, ensure_ascii=False),
            encoding="utf-8",
        )

    def empty_cache(self, llm):
        return {
            "library_sha1": self.library_sha1(),
            "embedding_model": getattr(llm, "embedding_model", ""),
            "embeddings": {},
        }

    def library_sha1(self):
        if not self.library_path.exists():
            return ""
        return hashlib.sha1(self.library_path.read_bytes()).hexdigest()

    def cosine(self, left, right):
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if not left_norm or not right_norm:
            return 0.0
        return dot / (left_norm * right_norm)
