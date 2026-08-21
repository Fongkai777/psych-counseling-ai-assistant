import hashlib
import json
import logging
import re
import shutil
import time
import warnings
from pathlib import Path

import jieba
from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import CompletionResponse, CustomLLM, LLMMetadata
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    SentenceSplitter,
    SentenceWindowNodeParser,
    get_leaf_nodes,
)
from llama_index.core.postprocessor import LLMRerank, MetadataReplacementPostProcessor
from llama_index.core.retrievers import AutoMergingRetriever, QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever

from psych_ai_assistant.retrieval import (
    DEFAULT_MIN_LEXICAL_SCORE,
    DEFAULT_MIN_SCORE,
    DEFAULT_MIN_SEMANTIC_SCORE,
    INDEX_CHUNK_OVERLAP,
    INDEX_CHUNK_SIZE,
    Retriever,
)


LOGGER = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message="The tokenizer parameter is deprecated.*")


class DashScopeEmbeddingAdapter(BaseEmbedding):
    """LlamaIndex embedding adapter backed by the app's existing LLMClient."""

    _client = PrivateAttr()
    _progress_callback = PrivateAttr(default=None)
    _progress_label = PrivateAttr(default="")

    def __init__(self, client, **kwargs):
        super().__init__(
            model_name=getattr(client, "embedding_model", "dashscope-embedding"),
            **kwargs,
        )
        self._client = client
        self._progress_callback = None
        self._progress_label = ""

    @classmethod
    def class_name(cls):
        return "DashScopeEmbeddingAdapter"

    def _get_query_embedding(self, query):
        embeddings = self._client.embed_texts([query], raise_on_error=True)
        if not embeddings or not embeddings[0]:
            raise ValueError("Embedding API did not return a query vector")
        return embeddings[0]

    async def _aget_query_embedding(self, query):
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text):
        embeddings = self._client.embed_texts([text], raise_on_error=True)
        if not embeddings or not embeddings[0]:
            raise ValueError("Embedding API did not return a text vector")
        return embeddings[0]

    def _get_text_embeddings(self, texts):
        def report_progress(done, total):
            if self._progress_callback:
                self._progress_callback(
                    done,
                    total,
                    self._progress_label,
                    len(texts),
                )

        embeddings = self._client.embed_texts(
            texts,
            raise_on_error=True,
            progress_callback=report_progress if self._progress_callback else None,
        )
        if len(embeddings) != len(texts):
            raise ValueError("Embedding API returned an unexpected number of vectors")
        return embeddings

    def set_progress_callback(self, callback=None, label=""):
        self._progress_callback = callback
        self._progress_label = label or ""


class DashScopeLLMAdapter(CustomLLM):
    """LlamaIndex LLM adapter backed by the app's existing generation client."""

    _client = PrivateAttr()

    def __init__(self, client, **kwargs):
        super().__init__(**kwargs)
        self._client = client

    @property
    def metadata(self):
        return LLMMetadata(
            context_window=8192,
            num_output=1024,
            model_name=getattr(self._client, "model", "dashscope-qwen"),
        )

    def complete(self, prompt, formatted=False, **kwargs):
        text = self._client.generate(prompt, enable_thinking=False, timeout=90)
        return CompletionResponse(text=text)

    def stream_complete(self, prompt, formatted=False, **kwargs):
        yield self.complete(prompt, formatted=formatted, **kwargs)


def chinese_tokenizer(text):
    return [
        token.strip().lower()
        for token in jieba.lcut(re.sub(r"\s+", " ", text or ""))
        if token.strip()
    ]


class LlamaIndexRetriever:
    """LlamaIndex-based retriever for hybrid, sentence-window, auto-merge, and rerank."""

    STORAGE_VERSION = "llamaindex-native-v3"

    def __init__(self, db, llm=None, storage_dir=None):
        self.db = db
        self.llm = llm
        self.storage_dir = Path(storage_dir or "data/llamaindex_storage")
        self.legacy = Retriever(db, llm)
        self.embed_model = DashScopeEmbeddingAdapter(llm) if llm else None
        self.llama_llm = DashScopeLLMAdapter(llm) if llm else None
        self._cache = {}

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
        folder_filter="全部",
        source_filter="全部",
    ):
        started_at = time.perf_counter()
        settings = {
            "limit": limit,
            "min_semantic_score": min_semantic_score,
            "min_lexical_score": min_lexical_score,
            "min_score": min_score,
            "use_summary_index": use_summary_index,
            "summary_limit": summary_limit,
            "retrieval_mode": retrieval_mode,
            "sentence_window_size": sentence_window_size,
            "auto_merge_group_size": auto_merge_group_size,
            "auto_merge_threshold": auto_merge_threshold,
            "folder_filter": folder_filter,
            "source_filter": source_filter,
        }
        metadata_filter = self.metadata_filter(folder_filter, source_filter)
        if not self.llm or not getattr(self.llm, "embedding_api_key", ""):
            results = self.legacy.search(
                query,
                limit=limit,
                min_semantic_score=min_semantic_score,
                min_lexical_score=min_lexical_score,
                min_score=min_score,
                use_summary_index=use_summary_index,
                summary_limit=summary_limit,
                retrieval_mode=retrieval_mode,
                sentence_window_size=sentence_window_size,
                auto_merge_group_size=auto_merge_group_size,
                auto_merge_threshold=auto_merge_threshold,
            )
            self.record_trace(
                query,
                "legacy_local",
                settings,
                [],
                [],
                results,
                started_at,
            )
            return results

        summary_routes = []
        document_ids = None
        if use_summary_index:
            summary_routes = self.summary_route(
                query,
                summary_limit,
                metadata_filter=metadata_filter,
            )
            document_ids = [item["document_id"] for item in summary_routes]

        raw_limit = max(limit * 8, 20)
        try:
            if retrieval_mode == "sentence_window":
                hits = self.sentence_window_retrieve(
                    query,
                    raw_limit,
                    document_ids=document_ids,
                    window_size=sentence_window_size,
                    metadata_filter=metadata_filter,
                )
            elif retrieval_mode == "auto_merging":
                hits = self.auto_merging_retrieve(
                    query,
                    raw_limit,
                    document_ids=document_ids,
                    merge_threshold=auto_merge_threshold,
                    metadata_filter=metadata_filter,
                )
            else:
                hits = self.hybrid_retrieve(
                    query,
                    raw_limit,
                    document_ids=document_ids,
                    metadata_filter=metadata_filter,
                )
        except Exception as exc:
            LOGGER.exception("LlamaIndex retrieval failed; falling back to legacy search")
            if self.llm:
                self.llm.last_embedding_error = f"LlamaIndex 检索失败，已回退旧检索：{exc}"
            results = self.legacy.search(
                query,
                limit=limit,
                min_semantic_score=min_semantic_score,
                min_lexical_score=min_lexical_score,
                min_score=min_score,
                use_summary_index=use_summary_index,
                summary_limit=summary_limit,
                retrieval_mode=retrieval_mode,
                sentence_window_size=sentence_window_size,
                auto_merge_group_size=auto_merge_group_size,
                auto_merge_threshold=auto_merge_threshold,
            )
            self.record_trace(
                query,
                "legacy_fallback",
                settings,
                summary_routes,
                [],
                results,
                started_at,
            )
            return results

        if not hits and document_ids:
            if retrieval_mode == "sentence_window":
                hits = self.sentence_window_retrieve(
                    query,
                    raw_limit,
                    window_size=sentence_window_size,
                    metadata_filter=metadata_filter,
                )
            elif retrieval_mode == "auto_merging":
                hits = self.auto_merging_retrieve(
                    query,
                    raw_limit,
                    merge_threshold=auto_merge_threshold,
                    metadata_filter=metadata_filter,
                )
            else:
                hits = self.hybrid_retrieve(query, raw_limit, metadata_filter=metadata_filter)

        candidate_trace = [self.node_to_trace_item(hit) for hit in hits[:30]]
        hits = self.filter_nodes(
            hits,
            min_semantic_score=min_semantic_score,
            min_lexical_score=min_lexical_score,
            min_score=min_score,
        )
        hits = self.llamaindex_rerank(query, hits, limit)
        results = [
            self.node_to_result(
                hit,
                summary_routes=summary_routes,
                mode=self.result_mode(retrieval_mode),
            )
            for hit in hits[:limit]
        ]
        self.record_trace(
            query,
            self.result_mode(retrieval_mode),
            settings,
            summary_routes,
            candidate_trace,
            results,
            started_at,
        )
        return results

    def hybrid_retrieve(self, query, limit, document_ids=None, metadata_filter=None):
        index, nodes = self.ensure_vector_index()
        if not index:
            return []
        return self.query_fusion_retrieve(
            query,
            index,
            nodes,
            limit,
            document_ids=document_ids,
            metadata_filter=metadata_filter,
            retrieval_stage="llamaindex_query_fusion",
        )

    def summary_route(self, query, limit=6, metadata_filter=None):
        index, nodes = self.ensure_summary_index()
        if not index:
            return []
        hits = self.query_fusion_retrieve(
            query,
            index,
            nodes,
            max(1, int(limit or 1)),
            metadata_filter=metadata_filter,
            retrieval_stage="llamaindex_summary_route",
        )
        routes = []
        seen = set()
        for hit in hits:
            metadata = hit.node.metadata or {}
            document_id = int(metadata.get("document_id") or 0)
            if not document_id or document_id in seen:
                continue
            seen.add(document_id)
            routes.append(
                {
                    "document_id": document_id,
                    "title": metadata.get("title", ""),
                    "source": metadata.get("source", ""),
                    "folder": metadata.get("folder", "默认"),
                    "summary": hit.node.get_content(metadata_mode="none") or "",
                    "score": round(float(hit.score or 0), 4),
                    "semantic_score": round(float(metadata.get("semantic_score") or 0), 4),
                    "lexical_score": round(float(metadata.get("lexical_score") or 0), 4),
                    "mode": "llamaindex_summary_route",
                }
            )
            if len(routes) >= limit:
                break
        return routes

    def sentence_window_retrieve(
        self,
        query,
        limit,
        document_ids=None,
        window_size=2,
        metadata_filter=None,
    ):
        index, nodes = self.ensure_sentence_window_index(window_size)
        if not index:
            return []
        hits = self.query_fusion_retrieve(
            query,
            index,
            nodes,
            limit,
            document_ids=document_ids,
            metadata_filter=metadata_filter,
            retrieval_stage="llamaindex_sentence_window",
        )
        processor = MetadataReplacementPostProcessor(target_metadata_key="window")
        return processor.postprocess_nodes(hits, query_str=query)

    def auto_merging_retrieve(
        self,
        query,
        limit,
        document_ids=None,
        merge_threshold=0.5,
        metadata_filter=None,
    ):
        index, storage_context = self.ensure_auto_merging_index()
        if not index or not storage_context:
            return []
        vector_retriever = index.as_retriever(similarity_top_k=max(1, int(limit or 1)))
        vector_hits = vector_retriever.retrieve(query)
        vector_scores = self.normalize_scores(
            {hit.node.node_id: float(hit.score or 0) for hit in vector_hits}
        )
        retriever = AutoMergingRetriever(
            vector_retriever,
            storage_context,
            simple_ratio_thresh=max(0.1, min(1.0, float(merge_threshold or 0.5))),
        )
        hits = retriever.retrieve(query)
        return self.annotate_and_filter_by_document(
            hits,
            vector_scores=vector_scores,
            bm25_scores={},
            document_ids=document_ids,
            metadata_filter=metadata_filter,
            retrieval_stage="llamaindex_auto_merging",
        )

    def query_fusion_retrieve(
        self,
        query,
        index,
        nodes,
        limit,
        document_ids=None,
        metadata_filter=None,
        retrieval_stage="llamaindex_query_fusion",
    ):
        top_k = max(1, min(int(limit or 1), len(nodes)))
        vector_retriever = index.as_retriever(similarity_top_k=top_k)
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=top_k,
            tokenizer=chinese_tokenizer,
        )
        vector_hits = vector_retriever.retrieve(query)
        bm25_hits = bm25_retriever.retrieve(query)
        vector_scores = self.normalize_scores(
            {hit.node.node_id: float(hit.score or 0) for hit in vector_hits}
        )
        bm25_scores = self.normalize_scores(
            {hit.node.node_id: float(hit.score or 0) for hit in bm25_hits}
        )
        retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            llm=self.llama_llm,
            mode=FUSION_MODES.RELATIVE_SCORE,
            similarity_top_k=top_k,
            num_queries=1,
            use_async=False,
            retriever_weights=[0.65, 0.35],
        )
        hits = retriever.retrieve(query)
        return self.annotate_and_filter_by_document(
            hits,
            vector_scores=vector_scores,
            bm25_scores=bm25_scores,
            document_ids=document_ids,
            metadata_filter=metadata_filter,
            retrieval_stage=retrieval_stage,
        )

    def ensure_vector_index(self):
        chunk_size, chunk_overlap, _, hierarchy_sizes = self.index_settings()
        signature = self.source_signature("vector", chunk_size, chunk_overlap)
        cached = self._cache.get("vector")
        if cached and cached["signature"] == signature:
            return cached["index"], cached["nodes"]

        nodes = self.build_vector_nodes(chunk_size, chunk_overlap)
        index = self.load_or_build_index("vector", signature, nodes)
        self._cache["vector"] = {"signature": signature, "index": index, "nodes": nodes}
        return index, nodes

    def ensure_summary_index(self):
        signature = self.source_signature("summary_route")
        cached = self._cache.get("summary_route")
        if cached and cached["signature"] == signature:
            return cached["index"], cached["nodes"]

        nodes = self.build_summary_nodes()
        index = self.load_or_build_index("summary_route", signature, nodes)
        self._cache["summary_route"] = {
            "signature": signature,
            "index": index,
            "nodes": nodes,
        }
        return index, nodes

    def ensure_sentence_window_index(self, window_size):
        window_size = max(0, min(8, int(window_size or 0)))
        signature = self.source_signature("sentence_window", window_size)
        cache_key = f"sentence_window:{window_size}"
        cached = self._cache.get(cache_key)
        if cached and cached["signature"] == signature:
            return cached["index"], cached["nodes"]

        nodes = self.build_sentence_window_nodes(window_size)
        index = self.load_or_build_index(cache_key, signature, nodes)
        self._cache[cache_key] = {"signature": signature, "index": index, "nodes": nodes}
        return index, nodes

    def ensure_auto_merging_index(self):
        chunk_size, chunk_overlap, _, hierarchy_sizes = self.index_settings()
        signature = self.source_signature("auto_merging", hierarchy_sizes, chunk_overlap)
        cached = self._cache.get("auto_merging")
        if cached and cached["signature"] == signature:
            return cached["index"], cached["storage_context"]

        nodes = self.build_hierarchy_nodes(hierarchy_sizes, chunk_overlap)
        leaf_nodes = get_leaf_nodes(nodes) if nodes else []
        if not leaf_nodes:
            return None, None
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)
        index = self.load_or_build_index(
            "auto_merging",
            signature,
            leaf_nodes,
            storage_context=storage_context,
        )
        self._cache["auto_merging"] = {
            "signature": signature,
            "index": index,
            "storage_context": storage_context,
        }
        return index, storage_context

    def load_or_build_index(self, kind, signature, nodes, storage_context=None):
        if not nodes:
            return None
        persist_dir = self.persist_dir(kind)
        manifest_path = persist_dir / "manifest.json"
        if self.can_load_persisted(manifest_path, signature):
            try:
                storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
                return load_index_from_storage(
                    storage_context,
                    embed_model=self.embed_model,
                )
            except Exception:
                LOGGER.exception("Failed to load persisted LlamaIndex %s index; rebuilding", kind)
                shutil.rmtree(persist_dir, ignore_errors=True)

        persist_dir.mkdir(parents=True, exist_ok=True)
        if persist_dir.exists():
            shutil.rmtree(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        storage_context = storage_context or StorageContext.from_defaults()
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=self.embed_model,
        )
        index.storage_context.persist(persist_dir=str(persist_dir))
        self.write_manifest(manifest_path, signature, len(nodes))
        return index

    def set_embedding_progress_callback(self, callback=None, label=""):
        if self.embed_model and hasattr(self.embed_model, "set_progress_callback"):
            self.embed_model.set_progress_callback(callback, label)

    def clear_storage(self):
        self._cache = {}
        shutil.rmtree(self.storage_dir, ignore_errors=True)

    def rebuild_all(self, progress_callback=None):
        self.clear_storage()
        steps = [
            ("Summary Route", self.ensure_summary_index),
            ("Vector + BM25 Fusion", self.ensure_vector_index),
            (
                "Sentence Window",
                lambda: self.ensure_sentence_window_index(self.index_settings()[2]),
            ),
            ("Auto Merging", self.ensure_auto_merging_index),
        ]
        for index, (label, builder) in enumerate(steps, start=1):
            if progress_callback:
                progress_callback(index, len(steps), label)
            def embedding_progress(done, batch_total, progress_label, node_count):
                if progress_callback:
                    progress_callback(
                        index,
                        len(steps),
                        (
                            f"{progress_label or label} · "
                            f"embedding {done}/{batch_total} 批 · "
                            f"{node_count} nodes"
                        ),
                    )

            self.set_embedding_progress_callback(embedding_progress, label)
            try:
                builder()
            finally:
                self.set_embedding_progress_callback(None, "")
        if progress_callback:
            progress_callback(len(steps), len(steps), "完成")
        return self.storage_stats()

    def build_vector_nodes(self, chunk_size, chunk_overlap):
        parser = SentenceSplitter.from_defaults(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        nodes = parser.get_nodes_from_documents(self.llama_documents())
        self.prepare_nodes(nodes, retrieval_stage="llamaindex_query_fusion")
        return nodes

    def build_summary_nodes(self):
        nodes = []
        for doc in self.db.get_documents():
            summary = self.extractive_document_summary(doc)
            metadata = {
                "document_id": doc["id"],
                "title": doc.get("title", ""),
                "source": doc.get("source", ""),
                "folder": doc.get("folder", "默认"),
                "chunk_index": 0,
                "retrieval_stage": "llamaindex_summary_route",
            }
            node = TextNode(text=summary, id_=f"summary-{doc['id']}", metadata=metadata)
            node.excluded_embed_metadata_keys = list(metadata.keys())
            node.excluded_llm_metadata_keys = list(metadata.keys())
            nodes.append(node)
        return nodes

    def build_sentence_window_nodes(self, window_size):
        parser = SentenceWindowNodeParser.from_defaults(window_size=window_size)
        nodes = parser.get_nodes_from_documents(self.llama_documents())
        self.prepare_nodes(
            nodes,
            retrieval_stage="llamaindex_sentence_window",
            extra_metadata={"window_size": window_size},
        )
        return nodes

    def build_hierarchy_nodes(self, hierarchy_sizes, chunk_overlap):
        parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=hierarchy_sizes,
            chunk_overlap=chunk_overlap,
        )
        nodes = parser.get_nodes_from_documents(self.llama_documents())
        self.prepare_nodes(nodes, retrieval_stage="llamaindex_auto_merging")
        return nodes

    def llama_documents(self):
        documents = []
        for doc in self.db.get_documents():
            metadata = {
                "document_id": doc["id"],
                "title": doc.get("title", ""),
                "source": doc.get("source", ""),
                "folder": doc.get("folder", "默认"),
            }
            documents.append(Document(text=doc.get("content") or "", metadata=metadata))
        return documents

    def extractive_document_summary(self, doc, max_chars=1200):
        title = doc.get("title", "")
        folder = doc.get("folder", "默认")
        content = re.sub(r"\s+", " ", doc.get("content") or "").strip()
        if len(content) <= max_chars:
            body = content
        else:
            part = max_chars // 3
            middle_start = max(0, len(content) // 2 - part // 2)
            body = "\n...\n".join(
                [
                    content[:part],
                    content[middle_start : middle_start + part],
                    content[-part:],
                ]
            )
        return f"标题：{title}\n文件夹：{folder}\n摘要候选文本：{body}"

    def prepare_nodes(self, nodes, retrieval_stage, extra_metadata=None):
        counters = {}
        for node in nodes:
            metadata = dict(node.metadata or {})
            document_id = metadata.get("document_id")
            counters[document_id] = counters.get(document_id, 0) + 1
            metadata.setdefault("chunk_index", counters[document_id] - 1)
            metadata["retrieval_stage"] = retrieval_stage
            if extra_metadata:
                metadata.update(extra_metadata)
            node.metadata = metadata
            node.excluded_embed_metadata_keys = list(metadata.keys())
            node.excluded_llm_metadata_keys = list(metadata.keys())

    def annotate_and_filter_by_document(
        self,
        hits,
        vector_scores,
        bm25_scores,
        document_ids=None,
        metadata_filter=None,
        retrieval_stage="llamaindex",
    ):
        document_id_set = set(document_ids or [])
        output = []
        for hit in hits:
            metadata = dict(hit.node.metadata or {})
            document_id = int(metadata.get("document_id") or 0)
            if document_id_set and document_id not in document_id_set:
                continue
            if not self.metadata_matches(metadata, metadata_filter):
                continue
            metadata["semantic_score"] = vector_scores.get(hit.node.node_id, float(hit.score or 0))
            metadata["lexical_score"] = bm25_scores.get(hit.node.node_id, 0)
            metadata["retrieval_stage"] = retrieval_stage
            hit.node.metadata = metadata
            output.append(hit)
        return output

    def metadata_filter(self, folder_filter="全部", source_filter="全部"):
        return {
            "folder_filter": folder_filter or "全部",
            "source_filter": source_filter or "全部",
        }

    def metadata_matches(self, metadata, metadata_filter=None):
        if not metadata_filter:
            return True
        folder_filter = metadata_filter.get("folder_filter") or "全部"
        source_filter = metadata_filter.get("source_filter") or "全部"
        folder = metadata.get("folder") or "默认"
        source = metadata.get("source") or ""
        if folder_filter != "全部" and folder != folder_filter:
            return False
        if source_filter == "我的旧回答" and not str(source).startswith("past_answer:"):
            return False
        if source_filter == "上传文件" and str(source).startswith("past_answer:"):
            return False
        return True

    def filter_nodes(
        self,
        hits,
        min_semantic_score=DEFAULT_MIN_SEMANTIC_SCORE,
        min_lexical_score=DEFAULT_MIN_LEXICAL_SCORE,
        min_score=DEFAULT_MIN_SCORE,
    ):
        filtered = []
        for hit in hits:
            metadata = hit.node.metadata or {}
            semantic_score = float(metadata.get("semantic_score") or 0)
            lexical_score = float(metadata.get("lexical_score") or 0)
            score = float(hit.score or 0)
            if (
                semantic_score >= min_semantic_score
                or lexical_score >= min_lexical_score
                or not (semantic_score or lexical_score)
            ) and score >= min_score:
                filtered.append(hit)
        return filtered

    def llamaindex_rerank(self, query, hits, limit):
        if not hits:
            return []
        if not (
            self.llm
            and self.llama_llm
            and getattr(self.llm, "rerank_enabled", False)
            and (getattr(self.llm, "dashscope_api_key", "") or getattr(self.llm, "api_key", ""))
        ):
            return hits[:limit]
        if hasattr(self.llm, "rerank_texts"):
            try:
                documents = [hit.node.get_content(metadata_mode="none") or "" for hit in hits[:50]]
                reranked = self.llm.rerank_texts(query, documents, top_n=limit)
                if reranked:
                    ranked_hits = []
                    used = set()
                    for item in reranked:
                        index = item.get("index")
                        if index is None or index in used or not 0 <= index < len(hits):
                            continue
                        hit = hits[index]
                        metadata = dict(hit.node.metadata or {})
                        metadata["rerank_score"] = round(float(item.get("score") or 0), 4)
                        metadata["rerank_model"] = getattr(self.llm, "rerank_model", "")
                        metadata["rerank_stage"] = "dashscope_text_rerank"
                        hit.node.metadata = metadata
                        hit.score = item.get("score") or hit.score
                        ranked_hits.append(hit)
                        used.add(index)
                    for index, hit in enumerate(hits):
                        if len(ranked_hits) >= limit:
                            break
                        if index not in used:
                            ranked_hits.append(hit)
                    return ranked_hits[:limit]
            except Exception as exc:
                LOGGER.exception("DashScope text rerank failed; trying LlamaIndex LLMRerank")
                if self.llm:
                    self.llm.last_embedding_error = f"DashScope rerank 失败，尝试 LLMRerank：{exc}"
        try:
            reranker = LLMRerank(
                llm=self.llama_llm,
                top_n=limit,
                choice_batch_size=min(8, max(1, len(hits))),
            )
            return reranker.postprocess_nodes(hits, query_str=query)
        except Exception as exc:
            LOGGER.exception("LlamaIndex LLMRerank failed; keeping fusion order")
            if self.llm:
                self.llm.last_embedding_error = f"LlamaIndex rerank 失败，已保留融合排序：{exc}"
            return hits[:limit]

    def node_to_result(self, hit, summary_routes=None, mode="llamaindex_query_fusion"):
        metadata = hit.node.metadata or {}
        route_map = {item["document_id"]: item for item in (summary_routes or [])}
        item = {
            "document_id": int(metadata.get("document_id") or 0),
            "title": metadata.get("title", ""),
            "source": metadata.get("source", ""),
            "folder": metadata.get("folder", "默认"),
            "chunk_index": int(metadata.get("chunk_index") or 0),
            "score": round(float(hit.score or 0), 4),
            "semantic_score": round(float(metadata.get("semantic_score") or 0), 4),
            "lexical_score": round(float(metadata.get("lexical_score") or 0), 4),
            "content": hit.node.get_content(metadata_mode="none") or "",
            "mode": mode,
            "retrieval_stage": metadata.get("retrieval_stage", mode),
        }
        if metadata.get("rerank_score") is not None:
            item["rerank_score"] = metadata.get("rerank_score")
        if metadata.get("rerank_model"):
            item["rerank_model"] = metadata.get("rerank_model")
        if metadata.get("rerank_stage"):
            item["rerank_stage"] = metadata.get("rerank_stage")
        if metadata.get("window_size") is not None:
            item["window_size"] = metadata.get("window_size")
        return self.legacy.with_route(item, route_map)

    def node_to_trace_item(self, hit):
        metadata = hit.node.metadata or {}
        content = hit.node.get_content(metadata_mode="none") or ""
        return {
            "document_id": int(metadata.get("document_id") or 0),
            "title": metadata.get("title", ""),
            "source": metadata.get("source", ""),
            "folder": metadata.get("folder", "默认"),
            "chunk_index": int(metadata.get("chunk_index") or 0),
            "score": round(float(hit.score or 0), 4),
            "semantic_score": round(float(metadata.get("semantic_score") or 0), 4),
            "lexical_score": round(float(metadata.get("lexical_score") or 0), 4),
            "retrieval_stage": metadata.get("retrieval_stage", ""),
            "content_preview": content[:500],
        }

    def record_trace(
        self,
        query,
        retrieval_mode,
        settings,
        summary_routes,
        candidates,
        final,
        started_at,
    ):
        if not hasattr(self.db, "add_retrieval_trace"):
            return
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        try:
            self.db.add_retrieval_trace(
                query=query,
                retrieval_mode=retrieval_mode,
                settings=settings,
                summary_routes=summary_routes,
                candidates=candidates,
                final=[
                    {
                        **{key: value for key, value in item.items() if key != "content"},
                        "content_preview": (item.get("content") or "")[:500],
                    }
                    for item in (final or [])
                ],
                elapsed_ms=elapsed_ms,
            )
        except Exception:
            LOGGER.exception("Failed to record retrieval trace")

    def result_mode(self, retrieval_mode):
        return {
            "hybrid": "llamaindex_query_fusion",
            "sentence_window": "llamaindex_sentence_window",
            "auto_merging": "llamaindex_auto_merging",
        }.get(retrieval_mode, "llamaindex_query_fusion")

    def index_settings(self):
        chunk_size = int(self.db.get_setting("rag_chunk_size", str(INDEX_CHUNK_SIZE)))
        chunk_overlap = int(
            self.db.get_setting("rag_chunk_overlap", str(INDEX_CHUNK_OVERLAP))
        )
        chunk_size = max(300, min(4000, chunk_size))
        chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))
        sentence_window_size = int(self.db.get_setting("rag_sentence_window_size", "3"))
        hierarchy_sizes = [chunk_size, chunk_size * 3, chunk_size * 9]
        return chunk_size, chunk_overlap, sentence_window_size, hierarchy_sizes

    def source_signature(self, kind, *settings):
        source = [
            {
                "id": doc.get("id"),
                "title": doc.get("title"),
                "source": doc.get("source"),
                "folder": doc.get("folder", "默认"),
                "content_len": len(doc.get("content") or ""),
                "content_sha1": hashlib.sha1(
                    (doc.get("content") or "").encode("utf-8")
                ).hexdigest(),
            }
            for doc in self.db.get_documents()
        ]
        payload = {
            "version": self.STORAGE_VERSION,
            "kind": kind,
            "settings": settings,
            "embedding_model": getattr(self.llm, "embedding_model", ""),
            "source": source,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def normalize_scores(self, scores):
        max_score = max(scores.values(), default=0)
        if max_score <= 0:
            return {key: 0 for key in scores}
        return {key: value / max_score for key, value in scores.items()}

    def persist_dir(self, kind):
        return self.storage_dir / re.sub(r"[^a-zA-Z0-9_.:-]+", "_", kind)

    def expected_index_specs(self):
        chunk_size, chunk_overlap, sentence_window_size, hierarchy_sizes = self.index_settings()
        return [
            (
                "summary_route",
                "Summary Route",
                self.source_signature("summary_route"),
            ),
            (
                "vector",
                "Vector + BM25 Fusion",
                self.source_signature("vector", chunk_size, chunk_overlap),
            ),
            (
                f"sentence_window:{sentence_window_size}",
                "Sentence Window",
                self.source_signature("sentence_window", sentence_window_size),
            ),
            (
                "auto_merging",
                "Auto Merging",
                self.source_signature("auto_merging", hierarchy_sizes, chunk_overlap),
            ),
        ]

    def index_status(self):
        statuses = []
        for kind, label, signature in self.expected_index_specs():
            persist_dir = self.persist_dir(kind)
            manifest_path = persist_dir / "manifest.json"
            valid = self.can_load_persisted(manifest_path, signature)
            node_count = 0
            stored_model = ""
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    node_count = int(manifest.get("node_count") or 0)
                    stored_model = manifest.get("embedding_model") or ""
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    pass
            statuses.append(
                {
                    "kind": kind,
                    "label": label,
                    "valid": valid,
                    "exists": manifest_path.exists(),
                    "node_count": node_count,
                    "embedding_model": stored_model,
                    "path": str(persist_dir),
                }
            )
        return statuses

    def can_load_persisted(self, manifest_path, signature):
        if not manifest_path.exists():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return (
            manifest.get("version") == self.STORAGE_VERSION
            and manifest.get("signature") == signature
            and manifest.get("embedding_model") == getattr(self.llm, "embedding_model", "")
        )

    def write_manifest(self, manifest_path, signature, node_count):
        manifest_path.write_text(
            json.dumps(
                {
                    "version": self.STORAGE_VERSION,
                    "signature": signature,
                    "embedding_model": getattr(self.llm, "embedding_model", ""),
                    "node_count": node_count,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def storage_stats(self):
        if not self.storage_dir.exists():
            statuses = self.index_status()
            return {
                "path": str(self.storage_dir),
                "size_bytes": 0,
                "node_count": 0,
                "indexes": statuses,
                "stale_kinds": [item["label"] for item in statuses if not item["valid"]],
            }
        size = sum(path.stat().st_size for path in self.storage_dir.rglob("*") if path.is_file())
        node_count = 0
        for manifest_path in self.storage_dir.rglob("manifest.json"):
            try:
                node_count += int(
                    json.loads(manifest_path.read_text(encoding="utf-8")).get("node_count", 0)
                )
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
        statuses = self.index_status()
        return {
            "path": str(self.storage_dir),
            "size_bytes": size,
            "node_count": node_count,
            "indexes": statuses,
            "stale_kinds": [item["label"] for item in statuses if not item["valid"]],
        }
