import math
import re

DEFAULT_MIN_SEMANTIC_SCORE = 0.56
DEFAULT_MIN_LEXICAL_SCORE = 0.12
DEFAULT_MIN_SCORE = 0.0
INDEX_CHUNK_SIZE = 900
INDEX_CHUNK_OVERLAP = 120
DEFAULT_SENTENCE_WINDOW_SIZE = 3
DEFAULT_HIERARCHY_CHUNK_SIZES = [200, 800, 2400]
SUMMARY_LOCAL_CHUNK_LIMIT = 3
SUMMARY_DIRECT_CHUNK_LIMIT = 10
SUMMARY_MAX_SECTIONS = 12
HYBRID_SEMANTIC_WEIGHT = 0.55
HYBRID_LEXICAL_WEIGHT = 0.35
HYBRID_EXACT_WEIGHT = 0.10
SUMMARY_SEMANTIC_WEIGHT = 0.65
SUMMARY_LEXICAL_WEIGHT = 0.25
SUMMARY_EXACT_WEIGHT = 0.10
RETRIEVAL_MODES = {
    "hybrid": "标准混合检索",
    "sentence_window": "Sentence Window",
    "auto_merging": "Auto Merging",
}


def tokenize(text):
    text = text.lower()
    words = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text)
    bigrams = [text[i : i + 2] for i in range(max(0, len(text) - 1)) if "\n" not in text[i : i + 2]]
    return words + bigrams


def split_sentences(text):
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    parts = re.split(r"(?<=[。！？!?；;])\s*|(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def chunks(text, size=520, overlap=80):
    size = max(100, int(size or INDEX_CHUNK_SIZE))
    overlap = max(0, int(overlap or 0))
    overlap = min(overlap, size - 1)
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


def chunks_with_spans(text, size=520, overlap=0):
    size = max(100, int(size or INDEX_CHUNK_SIZE))
    overlap = max(0, int(overlap or 0))
    overlap = min(overlap, size - 1)
    text = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    output = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        output.append({"content": text[start:end], "start": start, "end": end})
        if end == len(text):
            break
        start = max(0, end - overlap)
    return output


class Retriever:
    def __init__(self, db, llm=None):
        self.db = db
        self.llm = llm

    def search(
        self,
        query,
        limit=5,
        min_semantic_score=DEFAULT_MIN_SEMANTIC_SCORE,
        min_lexical_score=DEFAULT_MIN_LEXICAL_SCORE,
        min_score=DEFAULT_MIN_SCORE,
        use_summary_index=True,
        summary_limit=6,
        retrieval_mode="hybrid",
        sentence_window_size=2,
        auto_merge_group_size=3,
        auto_merge_threshold=0.5,
    ):
        summary_routes = []
        document_ids = None
        if use_summary_index:
            summary_routes = self.summary_search(query, summary_limit)
            document_ids = [item["document_id"] for item in summary_routes]
        if retrieval_mode == "sentence_window":
            results = self.sentence_window_search(
                query,
                limit * 4,
                document_ids=document_ids,
                summary_routes=summary_routes,
                min_semantic_score=min_semantic_score,
                min_lexical_score=min_lexical_score,
                min_score=min_score,
            )
            if not results and document_ids:
                results = self.sentence_window_search(
                    query,
                    limit * 4,
                    min_semantic_score=min_semantic_score,
                    min_lexical_score=min_lexical_score,
                    min_score=min_score,
                )
            if results:
                return self.rerank_if_enabled(query, results, limit)
        if retrieval_mode == "auto_merging":
            results = self.auto_merging_search(
                query,
                limit * 4,
                document_ids=document_ids,
                summary_routes=summary_routes,
                min_semantic_score=min_semantic_score,
                min_lexical_score=min_lexical_score,
                min_score=min_score,
                merge_threshold=auto_merge_threshold,
            )
            if not results and document_ids:
                results = self.auto_merging_search(
                    query,
                    limit * 4,
                    min_semantic_score=min_semantic_score,
                    min_lexical_score=min_lexical_score,
                    min_score=min_score,
                    merge_threshold=auto_merge_threshold,
                )
            if results:
                return self.rerank_if_enabled(query, results, limit)
        if self.llm:
            raw_limit = limit * 4 if retrieval_mode in {"sentence_window", "auto_merging"} else limit
            hybrid_results = self.hybrid_search(
                query,
                raw_limit,
                min_semantic_score=min_semantic_score,
                min_lexical_score=min_lexical_score,
                min_score=min_score,
                document_ids=document_ids,
                summary_routes=summary_routes,
            )
            if hybrid_results:
                return self.rerank_if_enabled(query, self.apply_retrieval_mode(
                    query,
                    hybrid_results,
                    limit,
                    retrieval_mode=retrieval_mode,
                    sentence_window_size=sentence_window_size,
                    auto_merge_group_size=auto_merge_group_size,
                    auto_merge_threshold=auto_merge_threshold,
                ), limit)
            if document_ids:
                hybrid_results = self.hybrid_search(
                    query,
                    raw_limit,
                    min_semantic_score=min_semantic_score,
                    min_lexical_score=min_lexical_score,
                    min_score=min_score,
                )
                if hybrid_results:
                    return self.rerank_if_enabled(query, self.apply_retrieval_mode(
                        query,
                        hybrid_results,
                        limit,
                        retrieval_mode=retrieval_mode,
                        sentence_window_size=sentence_window_size,
                        auto_merge_group_size=auto_merge_group_size,
                        auto_merge_threshold=auto_merge_threshold,
                    ), limit)
        lexical_results = [
            item
            for item in self.lexical_search(query, limit * 8, document_ids=document_ids)
            if item.get("score", 0) >= min_lexical_score
            and item.get("score", 0) >= min_score
        ][: limit * 4 if retrieval_mode in {"sentence_window", "auto_merging"} else limit] or [
            item
            for item in self.lexical_search(query, limit * 8)
            if item.get("score", 0) >= min_lexical_score
            and item.get("score", 0) >= min_score
        ][: limit * 4 if retrieval_mode in {"sentence_window", "auto_merging"} else limit]
        return self.rerank_if_enabled(query, self.apply_retrieval_mode(
            query,
            lexical_results,
            limit,
            retrieval_mode=retrieval_mode,
            sentence_window_size=sentence_window_size,
            auto_merge_group_size=auto_merge_group_size,
            auto_merge_threshold=auto_merge_threshold,
        ), limit)

    def rerank_if_enabled(self, query, results, limit):
        if not results:
            return []
        if self.llm and getattr(self.llm, "rerank_enabled", False):
            return self.llm.rerank(query, results, limit)
        return results[:limit]

    def apply_retrieval_mode(
        self,
        query,
        results,
        limit,
        retrieval_mode="hybrid",
        sentence_window_size=2,
        auto_merge_group_size=3,
        auto_merge_threshold=0.5,
    ):
        if retrieval_mode == "sentence_window":
            return [
                self.sentence_window_result(query, item, sentence_window_size)
                for item in results[:limit]
            ]
        if retrieval_mode == "auto_merging":
            return self.auto_merge_results(
                results,
                limit,
                group_size=auto_merge_group_size,
                threshold=auto_merge_threshold,
            )
        return results[:limit]

    def hybrid_search(
        self,
        query,
        limit,
        min_semantic_score=DEFAULT_MIN_SEMANTIC_SCORE,
        min_lexical_score=DEFAULT_MIN_LEXICAL_SCORE,
        min_score=DEFAULT_MIN_SCORE,
        document_ids=None,
        summary_routes=None,
    ):
        semantic_results = self.semantic_search(query, limit * 8, document_ids=document_ids)
        lexical_results = self.lexical_search(query, limit * 8, document_ids=document_ids)
        route_map = {
            item["document_id"]: item for item in (summary_routes or [])
        }
        if not semantic_results:
            return [
                self.with_route(item, route_map)
                for item in lexical_results
                if item.get("score", 0) >= min_lexical_score
                and item.get("score", 0) >= min_score
            ][:limit]
        if not lexical_results:
            return [
                self.with_route(item, route_map)
                for item in semantic_results
                if item.get("score", 0) >= min_semantic_score
                and item.get("score", 0) >= min_score
            ][:limit]

        merged = {}
        for item in semantic_results:
            key = (item["document_id"], item["chunk_index"])
            merged[key] = dict(item)
            merged[key]["semantic_score"] = item["score"]
            merged[key]["lexical_score"] = 0
        for item in lexical_results:
            key = (item["document_id"], item["chunk_index"])
            if key not in merged:
                merged[key] = dict(item)
                merged[key]["semantic_score"] = 0
            merged[key]["lexical_score"] = item["score"]

        semantic_max = max((item.get("semantic_score", 0) for item in merged.values()), default=0)
        lexical_max = max((item.get("lexical_score", 0) for item in merged.values()), default=0)
        query_terms = self.query_terms(query)

        for item in merged.values():
            semantic_norm = item.get("semantic_score", 0) / semantic_max if semantic_max else 0
            lexical_norm = item.get("lexical_score", 0) / lexical_max if lexical_max else 0
            exact_bonus = self.exact_term_bonus(query_terms, item["content"])
            score = (
                semantic_norm * HYBRID_SEMANTIC_WEIGHT
                + lexical_norm * HYBRID_LEXICAL_WEIGHT
                + exact_bonus * HYBRID_EXACT_WEIGHT
            )
            item["score"] = round(score, 4)
            item["semantic_norm"] = round(semantic_norm, 4)
            item["lexical_norm"] = round(lexical_norm, 4)
            item["exact_bonus"] = round(exact_bonus, 4)
            item["mode"] = "hybrid"
            self.with_route(item, route_map)

        results = [
            item
            for item in merged.values()
            if item.get("semantic_score", 0) >= min_semantic_score
            or item.get("lexical_score", 0) >= min_lexical_score
        ]
        results = [item for item in results if item.get("score", 0) >= min_score]
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:limit]

    def legacy_search(self, query, limit=5):
        if self.llm:
            semantic_results = self.semantic_search(query, limit)
            if semantic_results:
                return semantic_results
        return self.lexical_search(query, limit)

    def semantic_search(self, query, limit, document_ids=None):
        query_embedding = self.llm.embed_texts([query])
        if not query_embedding:
            return []
        query_vector = query_embedding[0]
        if not query_vector:
            return []
        candidates = []
        document_id_set = set(document_ids or [])
        for chunk in self.db.list_document_chunks():
            if document_id_set and chunk["document_id"] not in document_id_set:
                continue
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
                    "semantic_score": round(score, 4),
                    "lexical_score": 0,
                    "content": chunk["content"],
                    "mode": "embedding",
                }
            )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:limit]

    def lexical_search(self, query, limit=5, document_ids=None):
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        query_counts = self.count(query_tokens)
        candidates = []
        document_id_set = set(document_ids or [])
        chunk_rows = self.db.list_document_chunks()
        if chunk_rows:
            for chunk in chunk_rows:
                if document_id_set and chunk["document_id"] not in document_id_set:
                    continue
                score = self.cosine(query_counts, self.count(tokenize(chunk["content"])))
                if score > 0:
                    candidates.append(
                        {
                            "document_id": chunk["document_id"],
                            "title": chunk["title"],
                            "source": chunk["source"],
                            "chunk_index": chunk["chunk_index"],
                            "score": round(score, 4),
                            "semantic_score": 0,
                            "lexical_score": round(score, 4),
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
                            "semantic_score": 0,
                            "lexical_score": round(score, 4),
                            "content": chunk,
                            "mode": "local",
                        }
                        )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:limit]

    def summary_search(self, query, limit=6):
        summaries = self.db.list_document_summaries()
        if not summaries:
            return []
        query_tokens = tokenize(query)
        query_counts = self.count(query_tokens) if query_tokens else {}
        query_embedding = []
        if self.llm:
            embeddings = self.llm.embed_texts([query])
            query_embedding = embeddings[0] if embeddings else []
            if not query_embedding:
                query_embedding = []

        candidates = []
        for item in summaries:
            semantic_score = 0
            if query_embedding and item.get("embedding"):
                semantic_score = self.vector_cosine(query_embedding, item["embedding"])
            lexical_score = 0
            if query_counts:
                lexical_score = self.cosine(query_counts, self.count(tokenize(item["summary"])))
            if semantic_score <= 0 and lexical_score <= 0:
                continue
            candidates.append(
                {
                    "document_id": item["document_id"],
                    "title": item["title"],
                    "source": item["source"],
                    "folder": item.get("folder", "默认"),
                    "summary": item["summary"],
                    "semantic_score": round(semantic_score, 4),
                    "lexical_score": round(lexical_score, 4),
                }
            )

        semantic_max = max((item["semantic_score"] for item in candidates), default=0)
        lexical_max = max((item["lexical_score"] for item in candidates), default=0)
        query_terms = self.query_terms(query)
        for item in candidates:
            semantic_norm = item["semantic_score"] / semantic_max if semantic_max else 0
            lexical_norm = item["lexical_score"] / lexical_max if lexical_max else 0
            exact_bonus = self.exact_term_bonus(query_terms, item["summary"])
            item["score"] = round(
                semantic_norm * SUMMARY_SEMANTIC_WEIGHT
                + lexical_norm * SUMMARY_LEXICAL_WEIGHT
                + exact_bonus * SUMMARY_EXACT_WEIGHT,
                4,
            )
            item["mode"] = "summary_route"

        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[: max(1, int(limit or 1))]

    def score_text_items(
        self,
        query,
        rows,
        text_key,
        limit,
        min_semantic_score=DEFAULT_MIN_SEMANTIC_SCORE,
        min_lexical_score=DEFAULT_MIN_LEXICAL_SCORE,
        min_score=DEFAULT_MIN_SCORE,
    ):
        query_tokens = tokenize(query)
        query_counts = self.count(query_tokens) if query_tokens else {}
        query_vector = []
        if self.llm:
            embeddings = self.llm.embed_texts([query])
            query_vector = embeddings[0] if embeddings else []
            if not query_vector:
                query_vector = []

        merged = []
        for row in rows:
            semantic_score = 0
            if query_vector and row.get("embedding"):
                semantic_score = self.vector_cosine(query_vector, row["embedding"])
            lexical_score = 0
            if query_counts:
                lexical_score = self.cosine(query_counts, self.count(tokenize(row.get(text_key, ""))))
            if semantic_score <= 0 and lexical_score <= 0:
                continue
            item = dict(row)
            item["semantic_score"] = round(semantic_score, 4)
            item["lexical_score"] = round(lexical_score, 4)
            merged.append(item)

        semantic_max = max((item.get("semantic_score", 0) for item in merged), default=0)
        lexical_max = max((item.get("lexical_score", 0) for item in merged), default=0)
        query_terms = self.query_terms(query)
        for item in merged:
            semantic_norm = item.get("semantic_score", 0) / semantic_max if semantic_max else 0
            lexical_norm = item.get("lexical_score", 0) / lexical_max if lexical_max else 0
            exact_bonus = self.exact_term_bonus(query_terms, item.get(text_key, ""))
            item["score"] = round(
                semantic_norm * HYBRID_SEMANTIC_WEIGHT
                + lexical_norm * HYBRID_LEXICAL_WEIGHT
                + exact_bonus * HYBRID_EXACT_WEIGHT,
                4,
            )
            item["semantic_norm"] = round(semantic_norm, 4)
            item["lexical_norm"] = round(lexical_norm, 4)
            item["exact_bonus"] = round(exact_bonus, 4)

        results = [
            item
            for item in merged
            if item.get("semantic_score", 0) >= min_semantic_score
            or item.get("lexical_score", 0) >= min_lexical_score
        ]
        results = [item for item in results if item.get("score", 0) >= min_score]
        results.sort(key=lambda item: item.get("score", 0), reverse=True)
        return results[:limit]

    def sentence_window_search(
        self,
        query,
        limit,
        document_ids=None,
        summary_routes=None,
        min_semantic_score=DEFAULT_MIN_SEMANTIC_SCORE,
        min_lexical_score=DEFAULT_MIN_LEXICAL_SCORE,
        min_score=DEFAULT_MIN_SCORE,
    ):
        rows = self.db.list_sentence_nodes()
        document_id_set = set(document_ids or [])
        if document_id_set:
            rows = [row for row in rows if row["document_id"] in document_id_set]
        if not rows:
            return []
        route_map = {item["document_id"]: item for item in (summary_routes or [])}
        scored = self.score_text_items(
            query,
            rows,
            "sentence",
            limit,
            min_semantic_score=min_semantic_score,
            min_lexical_score=min_lexical_score,
            min_score=min_score,
        )
        results = []
        for item in scored:
            output = {
                "document_id": item["document_id"],
                "title": item["title"],
                "source": item["source"],
                "chunk_index": item["sentence_index"],
                "score": item["score"],
                "semantic_score": item.get("semantic_score", 0),
                "lexical_score": item.get("lexical_score", 0),
                "content": item["window"],
                "hit_sentence": item["sentence"],
                "mode": "sentence_window_chunk",
                "window_size": item.get("window_size"),
            }
            output = self.sentence_window_result(
                query,
                output,
                item.get("window_size") or DEFAULT_SENTENCE_WINDOW_SIZE,
            )
            output["mode"] = "sentence_window_chunk"
            self.with_route(output, route_map)
            results.append(output)
        return results

    def auto_merging_search(
        self,
        query,
        limit,
        document_ids=None,
        summary_routes=None,
        min_semantic_score=DEFAULT_MIN_SEMANTIC_SCORE,
        min_lexical_score=DEFAULT_MIN_LEXICAL_SCORE,
        min_score=DEFAULT_MIN_SCORE,
        merge_threshold=0.5,
    ):
        rows = self.db.list_hierarchy_nodes(level=0)
        document_id_set = set(document_ids or [])
        if document_id_set:
            rows = [row for row in rows if row["document_id"] in document_id_set]
        if not rows:
            return []
        route_map = {item["document_id"]: item for item in (summary_routes or [])}
        leaf_hits = self.score_text_items(
            query,
            rows,
            "content",
            limit,
            min_semantic_score=min_semantic_score,
            min_lexical_score=min_lexical_score,
            min_score=min_score,
        )
        if not leaf_hits:
            return []
        merged = self.merge_hierarchy_nodes(leaf_hits, merge_threshold)
        results = []
        for item in merged:
            output = {
                "document_id": item["document_id"],
                "title": item["title"],
                "source": item["source"],
                "chunk_index": item["chunk_index"],
                "score": item["score"],
                "semantic_score": item.get("semantic_score", 0),
                "lexical_score": item.get("lexical_score", 0),
                "content": item["content"],
                "mode": item.get("mode", "auto_merging_full"),
                "node_key": item.get("node_key"),
                "node_level": item.get("level"),
                "parent_key": item.get("parent_key"),
                "merged_chunk_count": item.get("merged_chunk_count"),
                "merge_hit_count": item.get("merge_hit_count"),
                "merge_threshold": item.get("merge_threshold"),
            }
            self.with_route(output, route_map)
            results.append(output)
        results.sort(key=lambda item: item.get("score", 0), reverse=True)
        return results[:limit]

    def merge_hierarchy_nodes(self, leaf_hits, threshold=0.5):
        threshold = max(0.1, min(1.0, float(threshold or 0.5)))
        selected = {
            (item["document_id"], item["node_key"]): dict(item, mode="auto_merging_leaf")
            for item in leaf_hits
        }
        changed = True
        while changed:
            changed = False
            by_parent = {}
            for key, item in list(selected.items()):
                parent_key = item.get("parent_key")
                if not parent_key:
                    continue
                by_parent.setdefault((item["document_id"], parent_key), []).append(key)
            for (document_id, parent_key), child_keys in by_parent.items():
                total_children = self.db.hierarchy_child_count(document_id, parent_key)
                if total_children <= 0:
                    continue
                ratio = len(child_keys) / total_children
                if len(child_keys) < 2 or ratio < threshold:
                    continue
                parent = self.db.get_hierarchy_node(document_id, parent_key)
                if not parent:
                    continue
                hit_items = [selected[key] for key in child_keys if key in selected]
                best = max(hit_items, key=lambda item: item.get("score", 0))
                for key in child_keys:
                    selected.pop(key, None)
                parent_item = dict(parent)
                parent_item["score"] = best.get("score", 0)
                parent_item["semantic_score"] = max(
                    (item.get("semantic_score", 0) for item in hit_items), default=0
                )
                parent_item["lexical_score"] = max(
                    (item.get("lexical_score", 0) for item in hit_items), default=0
                )
                parent_item["mode"] = "auto_merging_full"
                parent_item["merged_chunk_count"] = total_children
                parent_item["merge_hit_count"] = len(hit_items)
                parent_item["merge_threshold"] = threshold
                selected[(document_id, parent_key)] = parent_item
                changed = True
        output = list(selected.values())
        output.sort(key=lambda item: item.get("score", 0), reverse=True)
        return output

    def sentence_window_result(self, query, item, window_size=2):
        sentences = split_sentences(item.get("content", ""))
        if not sentences:
            return item
        query_counts = self.count(tokenize(query))
        best_index = 0
        best_score = -1
        for index, sentence in enumerate(sentences):
            score = self.cosine(query_counts, self.count(tokenize(sentence)))
            if score > best_score:
                best_index = index
                best_score = score
        window_size = max(0, int(window_size or 0))
        start = max(0, best_index - window_size)
        end = min(len(sentences), best_index + window_size + 1)
        output = dict(item)
        output["original_content"] = item.get("content", "")
        output["hit_sentence"] = sentences[best_index]
        output["content"] = "".join(sentences[start:end])
        output["mode"] = "sentence_window"
        output["window_size"] = window_size
        return output

    def auto_merge_results(self, results, limit, group_size=3, threshold=0.5):
        group_size = max(2, int(group_size or 3))
        threshold = max(0.1, min(1.0, float(threshold or 0.5)))
        groups = {}
        for item in results:
            key = (item["document_id"], item["chunk_index"] // group_size)
            groups.setdefault(key, []).append(item)

        merged = []
        used_keys = set()
        for key, hits in groups.items():
            document_id, group_index = key
            all_chunks = [
                chunk
                for chunk in self.db.list_document_chunks(document_id)
                if chunk["chunk_index"] // group_size == group_index
            ]
            all_chunks.sort(key=lambda chunk: chunk["chunk_index"])
            ratio = len({hit["chunk_index"] for hit in hits}) / max(1, len(all_chunks))
            if len(hits) >= 2 and ratio >= threshold and all_chunks:
                best = max(hits, key=lambda item: item.get("score", 0))
                output = dict(best)
                output["chunk_index"] = all_chunks[0]["chunk_index"]
                output["content"] = "\n\n".join(chunk["content"] for chunk in all_chunks)
                output["mode"] = "auto_merging"
                output["merged_chunk_count"] = len(all_chunks)
                output["merge_hit_count"] = len(hits)
                output["merge_threshold"] = threshold
                output["merge_group_size"] = group_size
                merged.append(output)
                used_keys.add(key)

        for item in results:
            key = (item["document_id"], item["chunk_index"] // group_size)
            if key not in used_keys:
                merged.append(item)

        deduped = {}
        for item in merged:
            key = (item["document_id"], item["chunk_index"], item.get("mode", ""))
            if key not in deduped or item.get("score", 0) > deduped[key].get("score", 0):
                deduped[key] = item
        output = list(deduped.values())
        output.sort(key=lambda item: item.get("score", 0), reverse=True)
        return output[:limit]

    def with_route(self, item, route_map):
        route = route_map.get(item["document_id"]) if route_map else None
        if route:
            item["route_score"] = route.get("score", 0)
            item["route_summary"] = route.get("summary", "")
            item["route_title"] = route.get("title", "")
        return item

    def count(self, tokens):
        counts = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        return counts

    def query_terms(self, query):
        stop_terms = {
            "这个",
            "问题",
            "回答",
            "什么",
            "怎么",
            "应该",
            "可以",
            "需要",
            "不要",
            "不是",
            "没有",
            "以及",
            "对于",
            "如果",
            "时候",
            "读者",
            "情绪",
            "角度",
            "检索",
            "知识",
            "片段",
            "内容",
        }
        terms = set()
        for token in tokenize(query):
            if len(token) < 2:
                continue
            if token.isdigit() or token in stop_terms:
                continue
            terms.add(token)
        return terms

    def exact_term_bonus(self, query_terms, content):
        if not query_terms:
            return 0
        content = content.lower()
        hits = sum(1 for term in query_terms if term in content)
        return min(1, hits / 8)

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


def build_chunk_items(
    content,
    llm=None,
    chunk_size=INDEX_CHUNK_SIZE,
    chunk_overlap=INDEX_CHUNK_OVERLAP,
    progress_callback=None,
):
    text_chunks = chunks(content, size=chunk_size, overlap=chunk_overlap)
    embeddings = (
        llm.embed_texts(
            text_chunks,
            progress_callback=progress_callback,
            raise_on_error=True,
        )
        if llm
        else []
    )
    items = []
    for index, chunk in enumerate(text_chunks):
        embedding = embeddings[index] if index < len(embeddings) else None
        items.append(
            {
                "chunk_index": index,
                "content": chunk,
                "embedding": embedding,
                "embedding_model": llm.embedding_model if embedding is not None and llm else None,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }
        )
    return items


def extractive_summary(title, content, max_chars=900):
    text = re.sub(r"\s+", " ", (content or "").strip())
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return f"{title}\n\n{head}\n...\n{tail}"


def sampled_summary_sections(sections, max_sections=SUMMARY_MAX_SECTIONS):
    if len(sections) <= max_sections:
        return sections
    if max_sections <= 1:
        return [sections[0]]
    indexes = sorted(
        {
            round(i * (len(sections) - 1) / (max_sections - 1))
            for i in range(max_sections)
        }
    )
    return [sections[index] for index in indexes]


def llm_document_summary(
    title,
    content,
    llm,
    source_chunks=None,
    progress_callback=None,
):
    text = re.sub(r"\s+", " ", (content or "").strip())
    if not text:
        return ""
    source_chunks = source_chunks or [text]
    if len(source_chunks) <= SUMMARY_DIRECT_CHUNK_LIMIT:
        from psych_ai_assistant.prompts import build_document_summary_prompt

        if progress_callback:
            progress_callback(1, 1)
        return llm.generate(
            build_document_summary_prompt(title, "\n\n".join(source_chunks)),
            enable_thinking=False,
            timeout=45,
        ).strip()

    from psych_ai_assistant.prompts import (
        build_document_section_summary_prompt,
        build_document_summary_merge_prompt,
    )

    sections = sampled_summary_sections(source_chunks)
    section_summaries = []
    total_steps = len(sections) + 1
    for index, section in enumerate(sections, start=1):
        if progress_callback:
            progress_callback(index, total_steps)
        summary = llm.generate(
            build_document_section_summary_prompt(title, index, len(sections), section),
            enable_thinking=False,
            timeout=35,
        ).strip()
        if summary.startswith("模型 API 调用失败"):
            summary = extractive_summary(f"{title} 第{index}段", section, max_chars=360)
        section_summaries.append(f"第 {index} 段：{summary}")

    if progress_callback:
        progress_callback(total_steps, total_steps)
    merged = llm.generate(
        build_document_summary_merge_prompt(title, "\n\n".join(section_summaries)),
        enable_thinking=False,
        timeout=45,
    ).strip()
    if merged.startswith("模型 API 调用失败"):
        return extractive_summary(title, "\n\n".join(section_summaries), max_chars=900)
    return merged


def build_summary_item(title, content, llm=None, chunk_size=None, chunk_overlap=None, progress_callback=None):
    summary = ""
    source_chunks = chunks(
        content or "",
        size=chunk_size or INDEX_CHUNK_SIZE,
        overlap=chunk_overlap or INDEX_CHUNK_OVERLAP,
    )
    if len(source_chunks) <= SUMMARY_LOCAL_CHUNK_LIMIT:
        summary = extractive_summary(title, content)
    if llm and getattr(llm, "api_key", ""):
        if not summary:
            try:
                summary = llm_document_summary(
                    title,
                    content,
                    llm,
                    source_chunks=source_chunks,
                    progress_callback=progress_callback,
                )
            except Exception:
                summary = ""
            if summary.startswith("模型 API 调用失败"):
                summary = ""
    if not summary:
        summary = extractive_summary(title, content)

    embeddings = (
        llm.embed_texts(
            [summary],
            progress_callback=progress_callback,
            raise_on_error=True,
        )
        if llm
        else []
    )
    embedding = embeddings[0] if embeddings else None
    return {
        "summary": summary,
        "embedding": embedding,
        "embedding_model": llm.embedding_model if embedding is not None and llm else None,
        "summary_model": llm.model if llm else "local",
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }


def build_sentence_window_items(
    content,
    llm=None,
    window_size=DEFAULT_SENTENCE_WINDOW_SIZE,
    chunk_size=INDEX_CHUNK_SIZE,
    chunk_overlap=INDEX_CHUNK_OVERLAP,
    progress_callback=None,
):
    source_chunks = chunks(content or "", size=chunk_size, overlap=chunk_overlap)
    window_size = max(0, int(window_size or 0))
    embeddings = (
        llm.embed_texts(
            source_chunks,
            progress_callback=progress_callback,
            raise_on_error=True,
        )
        if llm and source_chunks
        else []
    )
    items = []
    for index, chunk in enumerate(source_chunks):
        embedding = embeddings[index] if index < len(embeddings) else None
        items.append(
            {
                "sentence_index": index,
                "sentence": chunk,
                "window": chunk,
                "embedding": embedding,
                "embedding_model": llm.embedding_model if embedding is not None and llm else None,
                "window_size": window_size,
            }
        )
    return items


def parse_hierarchy_chunk_sizes(value):
    if isinstance(value, (list, tuple)):
        sizes = [int(item) for item in value if int(item) > 0]
    else:
        sizes = []
        for item in re.split(r"[,，/、\\s]+", str(value or "")):
            item = item.strip()
            if item.isdigit():
                sizes.append(int(item))
    sizes = sorted(set(size for size in sizes if size >= 100))
    return sizes or DEFAULT_HIERARCHY_CHUNK_SIZES


def build_hierarchy_items(content, llm=None, chunk_sizes=None, progress_callback=None):
    sizes = parse_hierarchy_chunk_sizes(chunk_sizes or DEFAULT_HIERARCHY_CHUNK_SIZES)
    levels = []
    for level, size in enumerate(sizes):
        spans = chunks_with_spans(content, size=size, overlap=0)
        nodes = []
        for index, span in enumerate(spans):
            nodes.append(
                {
                    "node_key": f"L{level}:{index}",
                    "level": level,
                    "chunk_index": index,
                    "parent_key": None,
                    "content": span["content"],
                    "embedding": None,
                    "embedding_model": None,
                    "chunk_size": size,
                    "start_char": span["start"],
                    "end_char": span["end"],
                }
            )
        levels.append(nodes)

    for level, nodes in enumerate(levels[:-1]):
        parents = levels[level + 1]
        for node in nodes:
            midpoint = (node["start_char"] + node["end_char"]) / 2
            parent = next(
                (
                    item
                    for item in parents
                    if item["start_char"] <= midpoint < item["end_char"]
                ),
                parents[-1] if parents else None,
            )
            if parent:
                node["parent_key"] = parent["node_key"]

    leaf_nodes = levels[0] if levels else []
    embeddings = (
        llm.embed_texts(
            [node["content"] for node in leaf_nodes],
            progress_callback=progress_callback,
            raise_on_error=True,
        )
        if llm
        else []
    )
    for index, node in enumerate(leaf_nodes):
        embedding = embeddings[index] if index < len(embeddings) else None
        node["embedding"] = embedding
        node["embedding_model"] = llm.embedding_model if embedding is not None and llm else None

    return [node for nodes in levels for node in nodes]
