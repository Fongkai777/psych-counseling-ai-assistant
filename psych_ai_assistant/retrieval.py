import math
import re


def tokenize(text):
    text = text.lower()
    words = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text)
    bigrams = [text[i : i + 2] for i in range(max(0, len(text) - 1)) if "\n" not in text[i : i + 2]]
    return words + bigrams


def chunks(text, size=520, overlap=80):
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(text) <= size:
        return [text]
    output = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        output.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return output


class Retriever:
    def __init__(self, db, llm=None):
        self.db = db
        self.llm = llm

    def search(self, query, limit=5):
        if self.llm:
            semantic_results = self.semantic_search(query, limit)
            if semantic_results:
                return semantic_results
        return self.lexical_search(query, limit)

    def semantic_search(self, query, limit):
        query_embedding = self.llm.embed_texts([query])
        if not query_embedding:
            return []
        query_vector = query_embedding[0]
        candidates = []
        for chunk in self.db.list_document_chunks():
            embedding = chunk.get("embedding")
            if not embedding:
                continue
            score = self.vector_cosine(query_vector, embedding)
            candidates.append(
                {
                    "document_id": chunk["document_id"],
                    "title": chunk["title"],
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                    "score": round(score, 4),
                    "content": chunk["content"],
                    "mode": "embedding",
                }
            )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:limit]

    def lexical_search(self, query, limit=5):
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        query_counts = self.count(query_tokens)
        candidates = []
        chunk_rows = self.db.list_document_chunks()
        if chunk_rows:
            for chunk in chunk_rows:
                score = self.cosine(query_counts, self.count(tokenize(chunk["content"])))
                if score > 0:
                    candidates.append(
                        {
                            "document_id": chunk["document_id"],
                            "title": chunk["title"],
                            "source": chunk["source"],
                            "chunk_index": chunk["chunk_index"],
                            "score": round(score, 4),
                            "content": chunk["content"],
                            "mode": "local",
                        }
                    )
        else:
            for document in self.db.get_documents():
                for index, chunk in enumerate(chunks(document["content"])):
                    score = self.cosine(query_counts, self.count(tokenize(chunk)))
                    if score > 0:
                        candidates.append(
                            {
                                "document_id": document["id"],
                                "title": document["title"],
                                "source": document["source"],
                                "chunk_index": index,
                                "score": round(score, 4),
                                "content": chunk,
                                "mode": "local",
                            }
                        )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:limit]

    def count(self, tokens):
        counts = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        return counts

    def cosine(self, left, right):
        shared = set(left) & set(right)
        dot = sum(left[token] * right[token] for token in shared)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if left_norm == 0 or right_norm == 0:
            return 0
        return dot / (left_norm * right_norm)

    def vector_cosine(self, left, right):
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0
        return dot / (left_norm * right_norm)


def build_chunk_items(content, llm=None):
    text_chunks = chunks(content, size=900, overlap=120)
    embeddings = llm.embed_texts(text_chunks) if llm else []
    items = []
    for index, chunk in enumerate(text_chunks):
        embedding = embeddings[index] if index < len(embeddings) else None
        items.append(
            {
                "chunk_index": index,
                "content": chunk,
                "embedding": embedding,
                "embedding_model": llm.embedding_model if embedding is not None and llm else None,
            }
        )
    return items
