import json
import logging
from pathlib import Path
from datetime import datetime

import streamlit as st

from psych_ai_assistant.config import load_config
from psych_ai_assistant.db import Database
from psych_ai_assistant.document_loader import extract_uploaded_text
from psych_ai_assistant.feedback import analyze_edit
from psych_ai_assistant.llm import LLMClient
from psych_ai_assistant.prompts import (
    build_answer_prompt,
    build_intent_prompt,
    build_rag_triad_eval_prompt,
    build_revision_feedback_memory_prompt,
    default_persona,
)
from psych_ai_assistant.retrieval import (
    INDEX_CHUNK_OVERLAP,
    INDEX_CHUNK_SIZE,
    Retriever,
    build_hierarchy_items,
    build_chunk_items,
    build_sentence_window_items,
    build_summary_item,
)
from psych_ai_assistant.zhihu import import_question_from_html


ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "assistant.sqlite3"
LOG_PATH = ROOT / "logs" / "app.log"
STYLE_PROFILE_PATH = ROOT / "data" / "style_voice_profile.md"
LOG_PATH.parent.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
RECOMMENDED_RAG_SETTINGS = {
    "rag_chunk_size": 1800,
    "rag_chunk_overlap": 240,
    "rag_embedding_batch_size": 5,
    "rag_build_summary_index": True,
    "rag_build_sentence_index": True,
    "rag_sentence_window_size": 3,
    "rag_build_hierarchy_index": True,
    "retrieval_use_summary_index": True,
    "retrieval_summary_limit": 6,
    "retrieval_limit": 5,
    "retrieval_min_score": 0.0,
    "retrieval_min_semantic_score": 0.50,
    "retrieval_min_lexical_score": 0.08,
    "retrieval_use_rerank": True,
}


def services():
    config = load_config(ROOT / ".env")
    database = Database(DB_PATH)
    seed_if_empty(database)
    client = LLMClient(config)
    return {
        "db": database,
        "retriever": Retriever(database, client),
        "llm": client,
    }


def seed_if_empty(db):
    db.delete_documents_by_source("sample")
    if db.list_questions():
        return
    db.add_question(
        title="为什么我明明知道应该开始，却总是拖延？",
        source_url="https://www.zhihu.com/question/example",
        description="",
        tags="拖延,自我成长,情绪",
        heat=82,
    )
    db.add_question(
        title="原生家庭带来的不安全感，长大后真的能改变吗？",
        source_url="https://www.zhihu.com/question/example-family",
        description="",
        tags="原生家庭,亲密关系,安全感",
        heat=76,
    )


svc = services()
db = svc["db"]
retriever = svc["retriever"]
llm = svc["llm"]

st.set_page_config(page_title="心理内容 AI 运营助手", layout="wide")

st.markdown(
    """
	    <style>
	    .block-container { padding-top: 1.4rem; }
	    h1 {
	        font-size: 1.75rem !important;
	        line-height: 1.25 !important;
	        margin-bottom: 0.35rem !important;
	    }
	    div[data-testid="stSegmentedControl"] button {
	        font-size: 1.05rem !important;
	        font-weight: 650 !important;
	        padding: 0.45rem 0.9rem !important;
	        border-radius: 0.35rem !important;
	    }
	    textarea { line-height: 1.6 !important; }
    .small-muted { color: #6b7280; font-size: 13px; }
    .pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        background: #eef2ff;
        color: #3b63f4;
        font-size: 12px;
        margin-right: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state():
    questions = db.list_questions()
    defaults = {
        "selected_question_id": questions[0]["id"] if questions else None,
        "answer_id": None,
        "draft_text": "",
        "answer_edit_text": "",
        "context": [],
        "last_retrieval_query": "",
        "review_text": "",
        "rag_current_folder": "默认",
        "rag_move_doc_id": None,
        "global_answer_prompt_editor": db.get_setting("global_answer_prompt", default_persona()),
        "generation_run_id": None,
        "last_generation_prompt": "",
        "pending_answer_edit_text": None,
        "last_revision_instruction": "",
        "revision_instruction_history": [],
        "pending_revision_instruction_clear": False,
        "workflow_intent": "",
        "intent_feedback_history": [],
        "retrieval_feedback_history": [],
        "retrieval_limit": 5,
        "retrieval_min_score": 0.0,
        "retrieval_min_semantic_score": 0.56,
        "retrieval_min_lexical_score": 0.12,
        "retrieval_use_summary_index": db.get_setting("retrieval_use_summary_index", "1")
        == "1",
        "retrieval_summary_limit": int(db.get_setting("retrieval_summary_limit", "6")),
        "retrieval_mode": db.get_setting("retrieval_mode", "hybrid"),
        "sentence_window_size": int(db.get_setting("sentence_window_size", "2")),
        "auto_merge_group_size": int(db.get_setting("auto_merge_group_size", "3")),
        "auto_merge_threshold": float(db.get_setting("auto_merge_threshold", "0.5")),
        "rag_chunk_size": int(db.get_setting("rag_chunk_size", str(INDEX_CHUNK_SIZE))),
        "rag_chunk_overlap": int(
            db.get_setting("rag_chunk_overlap", str(INDEX_CHUNK_OVERLAP))
        ),
        "rag_embedding_batch_size": max(
            1,
            min(
                20,
                int(db.get_setting("rag_embedding_batch_size", str(llm.embedding_batch_size))),
            ),
        ),
        "rag_build_summary_index": db.get_setting("rag_build_summary_index", "1") == "1",
        "rag_build_sentence_index": db.get_setting("rag_build_sentence_index", "1")
        == "1",
        "rag_sentence_window_size": int(db.get_setting("rag_sentence_window_size", "3")),
        "rag_build_hierarchy_index": db.get_setting("rag_build_hierarchy_index", "1")
        == "1",
        "retrieval_use_rerank": db.get_setting("retrieval_use_rerank", "0") == "1",
        "pending_intent_feedback_clear": False,
        "pending_retrieval_feedback_clear": False,
        "answer_status_label": "",
        "answer_status_time": "",
        "last_revision_memory_prompt": "",
        "last_revision_memory_text": "",
        "last_revision_memory_count": 0,
        "last_rag_eval": {},
        "last_rag_eval_prompt": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def rerun():
    st.rerun()


class RagTaskCancelled(Exception):
    pass


def request_rag_task_cancel():
    db.set_setting("rag_task_cancel_requested", "1")


def clear_rag_task_cancel():
    db.set_setting("rag_task_cancel_requested", "0")


def rag_task_cancel_requested():
    return db.get_setting("rag_task_cancel_requested", "0") == "1"


def check_rag_task_cancelled():
    if rag_task_cancel_requested():
        raise RagTaskCancelled("用户暂停了当前 RAG 任务")


def status_label(status):
    return {"new": "待回答", "drafted": "已生成", "edited": "已编辑"}.get(status, status)


def current_question():
    if not st.session_state.selected_question_id:
        return None
    return db.get_question(st.session_state.selected_question_id)


def select_question(question_id):
    st.session_state.selected_question_id = question_id
    st.session_state.answer_id = None
    st.session_state.draft_text = ""
    st.session_state.answer_edit_text = ""
    st.session_state.context = []
    st.session_state.last_retrieval_query = ""
    st.session_state.review_text = ""
    st.session_state.last_generation_prompt = ""
    st.session_state.pending_answer_edit_text = None
    st.session_state.last_revision_instruction = ""
    st.session_state.revision_instruction_history = []
    st.session_state.pending_revision_instruction_clear = False
    st.session_state.workflow_intent = ""
    st.session_state.intent_feedback_history = []
    st.session_state.retrieval_feedback_history = []
    st.session_state.pending_intent_feedback_clear = False
    st.session_state.pending_retrieval_feedback_clear = False
    st.session_state.answer_status_label = ""
    st.session_state.answer_status_time = ""
    st.session_state.last_revision_memory_prompt = ""
    st.session_state.last_revision_memory_text = ""
    st.session_state.last_revision_memory_count = 0
    st.session_state.last_rag_eval = {}
    st.session_state.last_rag_eval_prompt = ""
    st.session_state.last_rag_eval_prompt = ""


def restore_answer_workspace(question):
    if not question:
        return
    if st.session_state.answer_id or st.session_state.draft_text or st.session_state.context:
        return
    answer = db.get_latest_answer_for_question(question["id"])
    if answer:
        text = answer.get("edited") or answer.get("draft") or ""
        st.session_state.answer_id = answer["id"]
        st.session_state.draft_text = text
        st.session_state.answer_edit_text = text
        st.session_state.context = answer.get("context") or []
    run = db.get_latest_generation_run_for_question(question["id"])
    if run:
        st.session_state.generation_run_id = run["id"]
        st.session_state.last_generation_prompt = run.get("prompt") or ""
        intent = extract_workflow_intent(run.get("prompt") or "")
        if intent:
            st.session_state.workflow_intent = intent
        if run.get("run_type") == "rewrite":
            instruction = extract_revision_instruction(run.get("prompt") or "")
            st.session_state.last_revision_instruction = instruction
            if instruction and not st.session_state.revision_instruction_history:
                st.session_state.revision_instruction_history = [instruction]
        st.session_state.answer_status_label = (
            "按意见重写稿" if run.get("run_type") == "rewrite" else "AI 初稿"
        )
        st.session_state.answer_status_time = run.get("updated_at") or ""
        if not st.session_state.context:
            st.session_state.context = run.get("context") or []
        feedback = run.get("feedback") or {}
        st.session_state.last_rag_eval = feedback.get("rag_triad") or {}


def now_label():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def extract_revision_instruction(prompt):
    if "最高优先级生成意见：" in prompt or "生成意见：" in prompt:
        marker = "最高优先级生成意见：" if "最高优先级生成意见：" in prompt else "生成意见："
        text = prompt.split(marker, 1)[1]
        for marker in ["当前稿件：", "【通用回答 Prompt】", "【个人语气文档】"]:
            if marker in text:
                text = text.split(marker, 1)[0]
        return text.strip()
    start = "【人工意见】"
    end = "【检索依据】"
    if start not in prompt:
        return ""
    text = prompt.split(start, 1)[1]
    if end in text:
        text = text.split(end, 1)[0]
    return text.strip()


def extract_workflow_intent(prompt):
    if "意图识别结果：" not in prompt:
        return ""
    text = prompt.split("意图识别结果：", 1)[1]
    for marker in ["最高优先级生成意见：", "生成意见：", "当前稿件：", "【通用回答 Prompt】", "【个人语气文档】"]:
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


def workflow_guidance(intent, retrieval_feedback="", draft_feedback="", current_draft=""):
    parts = []
    if intent.strip():
        parts.append(f"意图识别结果：\n{intent.strip()}")
    if retrieval_feedback.strip():
        parts.append(f"检索补充要求：\n{retrieval_feedback.strip()}")
    if draft_feedback.strip():
        parts.append(f"最高优先级生成意见：\n{draft_feedback.strip()}")
    if current_draft.strip() and draft_feedback.strip():
        parts.append(f"当前稿件：\n{current_draft.strip()}")
    return "\n\n".join(parts)


def workflow_search_query(question, intent, retrieval_feedback=""):
    return "\n\n".join(
        part
        for part in [
            question_query_text(question),
            intent or "",
            retrieval_feedback or "",
        ]
        if part.strip()
    )


def retrieval_settings():
    return {
        "limit": int(st.session_state.get("retrieval_limit", 5)),
        "min_score": float(st.session_state.get("retrieval_min_score", 0.0)),
        "min_semantic_score": float(
            st.session_state.get("retrieval_min_semantic_score", 0.56)
        ),
        "min_lexical_score": float(
            st.session_state.get("retrieval_min_lexical_score", 0.12)
        ),
        "use_summary_index": bool(st.session_state.get("retrieval_use_summary_index", True)),
        "summary_limit": int(st.session_state.get("retrieval_summary_limit", 6)),
        "retrieval_mode": st.session_state.get("retrieval_mode", "hybrid"),
        "sentence_window_size": int(st.session_state.get("sentence_window_size", 2)),
        "auto_merge_group_size": int(st.session_state.get("auto_merge_group_size", 3)),
        "auto_merge_threshold": float(st.session_state.get("auto_merge_threshold", 0.5)),
    }


def rag_index_settings():
    chunk_size = int(st.session_state.get("rag_chunk_size", INDEX_CHUNK_SIZE))
    chunk_overlap = int(st.session_state.get("rag_chunk_overlap", INDEX_CHUNK_OVERLAP))
    embedding_batch_size = int(
        st.session_state.get("rag_embedding_batch_size", llm.embedding_batch_size)
    )
    chunk_size = max(100, min(4000, chunk_size))
    chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))
    embedding_batch_size = max(1, min(20, embedding_batch_size))
    build_summary_index = bool(st.session_state.get("rag_build_summary_index", True))
    build_sentence_index = bool(st.session_state.get("rag_build_sentence_index", True))
    sentence_window_size = int(st.session_state.get("rag_sentence_window_size", 3))
    sentence_window_size = max(0, min(8, sentence_window_size))
    build_hierarchy_index = bool(st.session_state.get("rag_build_hierarchy_index", True))
    hierarchy_sizes = [chunk_size, chunk_size * 3, chunk_size * 9]
    hierarchy_chunk_sizes = ",".join(str(size) for size in hierarchy_sizes)
    db.set_setting("rag_chunk_size", str(chunk_size))
    db.set_setting("rag_chunk_overlap", str(chunk_overlap))
    db.set_setting("rag_embedding_batch_size", str(embedding_batch_size))
    db.set_setting("rag_build_summary_index", "1" if build_summary_index else "0")
    db.set_setting("rag_build_sentence_index", "1" if build_sentence_index else "0")
    db.set_setting("rag_sentence_window_size", str(sentence_window_size))
    db.set_setting("rag_build_hierarchy_index", "1" if build_hierarchy_index else "0")
    db.set_setting("rag_hierarchy_chunk_sizes", hierarchy_chunk_sizes)
    llm.embedding_batch_size = embedding_batch_size
    return (
        chunk_size,
        chunk_overlap,
        build_summary_index,
        build_sentence_index,
        sentence_window_size,
        build_hierarchy_index,
        hierarchy_chunk_sizes,
    )


def apply_recommended_rag_settings():
    for key, value in RECOMMENDED_RAG_SETTINGS.items():
        st.session_state[key] = value
        if isinstance(value, bool):
            db.set_setting(key, "1" if value else "0")
        else:
            db.set_setting(key, str(value))


def set_retrieval_preset(limit, min_score, min_semantic_score, min_lexical_score):
    st.session_state.retrieval_limit = limit
    st.session_state.retrieval_min_score = min_score
    st.session_state.retrieval_min_semantic_score = min_semantic_score
    st.session_state.retrieval_min_lexical_score = min_lexical_score
    rerun()


def render_workflow_visual():
    workflow_image = ROOT / "assets" / "answer_workflow.png"
    if workflow_image.exists():
        st.image(str(workflow_image), width="stretch")
    else:
        st.warning("未找到流程图图片，请把流程图保存到 assets/answer_workflow.png。")


def render_feedback_history(title, items):
    if not items:
        return
    st.caption(title)
    for index, item in enumerate(items, start=1):
        st.caption(f"{index}. {item}")


def render_flow_steps():
    steps = [
        ("1", "意图识别", bool(st.session_state.workflow_intent)),
        ("2", "检索依据", bool(st.session_state.context)),
        ("3", "生成 AI 初稿", bool(st.session_state.draft_text)),
    ]
    cols = st.columns(3)
    for col, (number, title, done) in zip(cols, steps):
        with col:
            with st.container(border=True):
                st.markdown(f"**{number}. {title}**")
                st.caption("已完成" if done else "等待运行")


def render_answer_status():
    if not st.session_state.answer_id and not st.session_state.draft_text:
        st.caption("当前稿件：尚未生成")
        return
    label = st.session_state.answer_status_label or "当前编辑稿"
    time_text = f" · {st.session_state.answer_status_time}" if st.session_state.answer_status_time else ""
    if label in {"按意见重写稿", "按意见生成稿"} and st.session_state.last_revision_instruction:
        st.info(f"当前稿件：本次已提意见：{st.session_state.last_revision_instruction}{time_text}")
        return
    st.info(f"当前稿件：{label}{time_text}")


def question_query_text(question):
    return question["title"] + "\n" + (question.get("description") or "")


def render_question_picker(key_prefix):
    questions = db.list_questions()
    if not questions:
        st.info("还没有选题。先导入一个知乎问题链接。")
        return

    options = {f"{item['title']} · {status_label(item['status'])}": item["id"] for item in questions}
    current = st.session_state.selected_question_id
    labels = list(options.keys())
    current_label = next((label for label, value in options.items() if value == current), labels[0])
    picked = st.selectbox(
        "当前选题",
        labels,
        index=labels.index(current_label),
        key=f"{key_prefix}_question_picker",
    )
    if options[picked] != current:
        select_question(options[picked])
        rerun()


def render_context(results):
    if not results:
        st.caption("没有检索到相关知识片段。")
        return
    for item in results:
        with st.expander(f"{item['title']} · 相关度 {item['score']}", expanded=False):
            mode = item.get("mode", "local")
            detail = f"{item['source'] or '未注明来源'} · {mode}"
            if item.get("semantic_score") is not None or item.get("lexical_score") is not None:
                detail += (
                    f" · embedding {item.get('semantic_score', 0):.4f}"
                    f" · 关键词 {item.get('lexical_score', 0):.4f}"
                )
            if item.get("route_score") is not None:
                detail += f" · 摘要路由 {item.get('route_score', 0):.4f}"
            if item.get("window_size") is not None:
                detail += f" · window ±{item.get('window_size')}"
            if item.get("node_key"):
                detail += f" · node {item.get('node_key')}"
            if item.get("node_level") is not None:
                detail += f" · level {item.get('node_level')}"
            if item.get("merged_chunk_count") is not None:
                detail += (
                    f" · 合并 {item.get('merge_hit_count', 0)}/"
                    f"{item.get('merged_chunk_count', 0)} chunks"
                )
            if item.get("rerank_rank") is not None:
                detail += f" · rerank #{item.get('rerank_rank')}"
            st.caption(detail)
            if item.get("route_summary"):
                with st.expander("命中的文档摘要", expanded=False):
                    st.write(item["route_summary"][:1600])
            if item.get("hit_sentence"):
                st.caption(f"命中句：{item['hit_sentence'][:260]}")
            if item.get("rerank_reason"):
                st.caption(f"重排理由：{item['rerank_reason'][:260]}")
            st.write(item["content"])


def parse_json_object(text):
    text = (text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {"raw": text}


def rag_triad_score(eval_result, key):
    if not isinstance(eval_result, dict):
        return None
    item = eval_result.get(key, {})
    if not isinstance(item, dict):
        return None
    try:
        return float(item.get("score"))
    except (TypeError, ValueError):
        return None


def rag_triad_next_actions(eval_result):
    if not isinstance(eval_result, dict):
        return []

    model_actions = eval_result.get("next_actions") or []
    actions = []
    if isinstance(model_actions, list):
        for item in model_actions:
            if not isinstance(item, dict):
                continue
            instruction = (item.get("instruction") or "").strip()
            if instruction:
                actions.append(
                    {
                        "type": item.get("type") or "next_step",
                        "priority": item.get("priority") or "medium",
                        "reason": item.get("reason") or "",
                        "instruction": instruction,
                    }
                )
    if actions:
        return actions

    answer_score = rag_triad_score(eval_result, "answer_relevance")
    context_score = rag_triad_score(eval_result, "context_relevance")
    grounded_score = rag_triad_score(eval_result, "groundedness")

    if answer_score is not None and answer_score < 0.65:
        actions.append(
            {
                "type": "rerun_intent",
                "priority": "high" if answer_score < 0.45 else "medium",
                "reason": "回答切题分偏低，说明稿件可能没有抓住问题真正要问的点。",
                "instruction": "回到意图识别，重新明确这个问题的核心冲突、回答对象和需要避开的跑题方向。",
            }
        )
    if context_score is not None and context_score < 0.65:
        actions.append(
            {
                "type": "improve_retrieval",
                "priority": "high" if context_score < 0.45 else "medium",
                "reason": "资料相关分偏低，说明当前检索片段不够贴近问题。",
                "instruction": "补充更直接的检索关键词，优先寻找与问题核心概念、人群、场景和心理机制相关的片段，再重新检索。",
            }
        )
    if grounded_score is not None and grounded_score < 0.65:
        actions.append(
            {
                "type": "revise_answer",
                "priority": "high" if grounded_score < 0.45 else "medium",
                "reason": "回答有据分偏低，说明稿件里有些判断没有被检索片段支撑。",
                "instruction": "重写时减少未引用判断；无法从检索片段推出的内容标注为一般建议；增加观点与检索片段的对应关系。",
            }
        )
    return actions


def render_rag_triad(eval_result, show_empty=False):
    if not eval_result and not show_empty:
        return
    with st.expander("RAG Triad 评价结果", expanded=True):
        metrics = [
            ("answer_relevance", "回答切题"),
            ("context_relevance", "资料相关"),
            ("groundedness", "回答有据"),
        ]
        cols = st.columns(3)
        for col, (key, label) in zip(cols, metrics):
            item = (eval_result or {}).get(key, {})
            score = item.get("score") if isinstance(item, dict) else None
            with col:
                st.metric(label, f"{float(score):.2f}" if score is not None else "待评价")
                if isinstance(item, dict) and item.get("reason"):
                    st.caption(item["reason"])
        if not eval_result:
            st.caption("点击“评价当前稿件”后，会生成每项分数、原因和建议下一步动作。")
            return
        suggestion = eval_result.get("suggestion")
        if suggestion:
            st.info(f"改进建议：{suggestion}")
        actions = rag_triad_next_actions(eval_result)
        if actions:
            st.markdown("**建议下一步动作**")
            for index, action in enumerate(actions, start=1):
                priority = action.get("priority") or "medium"
                action_type = action.get("type") or "next_step"
                st.write(f"{index}. `{priority}` · `{action_type}`：{action.get('instruction')}")
                if action.get("reason"):
                    st.caption(action["reason"])
        if eval_result.get("raw"):
            st.text_area("原始评价输出", value=eval_result["raw"], height=220, disabled=True)


def render_retrieval_config_panel():
    with st.expander("检索配置", expanded=True):
        st.markdown("**1. 召回范围：先找哪些文档**")
        st.checkbox(
            "先用 Summary Index 筛文档",
            key="retrieval_use_summary_index",
            help="开启后先从文档摘要里找候选资料，再进入这些资料的 chunk/句子/层级索引检索。",
        )
        st.number_input(
            "候选文档数",
            min_value=1,
            max_value=20,
            step=1,
            key="retrieval_summary_limit",
            help="摘要索引先保留多少份候选文档。",
        )
        db.set_setting(
            "retrieval_use_summary_index",
            "1" if st.session_state.retrieval_use_summary_index else "0",
        )
        db.set_setting(
            "retrieval_summary_limit",
            str(int(st.session_state.retrieval_summary_limit)),
        )

        st.divider()
        st.markdown("**2. 检索策略：怎么找片段**")
        mode_options = {
            "标准混合检索": "hybrid",
            "Sentence Window": "sentence_window",
            "Auto Merging": "auto_merging",
        }
        current_mode = st.session_state.get("retrieval_mode", "hybrid")
        current_mode_label = next(
            (label for label, value in mode_options.items() if value == current_mode),
            "标准混合检索",
        )
        mode_label = st.selectbox(
            "检索方式",
            list(mode_options.keys()),
            index=list(mode_options.keys()).index(current_mode_label),
            key="retrieval_mode_label",
            help="标准混合检索走 chunk；Sentence Window 走句子窗口；Auto Merging 走层级合并。",
        )
        st.session_state.retrieval_mode = mode_options[mode_label]
        db.set_setting("retrieval_mode", st.session_state.retrieval_mode)
        if st.session_state.retrieval_mode == "sentence_window":
            st.caption("当前模式会优先使用入库时生成的 Sentence Window 索引。")
            st.number_input(
                "旧索引回退窗口",
                min_value=0,
                max_value=8,
                step=1,
                key="sentence_window_size",
                help="旧资料没建 sentence index 时，才从普通 chunk 中临时取前后句。",
            )
            db.set_setting(
                "sentence_window_size",
                str(int(st.session_state.sentence_window_size)),
            )
        if st.session_state.retrieval_mode == "auto_merging":
            st.caption("当前模式会优先使用入库时生成的 hierarchy node 索引。")
            st.number_input(
                "旧索引回退组大小",
                min_value=2,
                max_value=8,
                step=1,
                key="auto_merge_group_size",
                help="旧资料没建 hierarchy index 时，才按连续 chunk 分组回退。",
            )
            st.slider(
                "合并触发比例",
                min_value=0.1,
                max_value=1.0,
                step=0.05,
                key="auto_merge_threshold",
                help="同一 parent 下命中的 child 比例达到这个值，就返回 parent。",
            )
            db.set_setting(
                "auto_merge_group_size",
                str(int(st.session_state.auto_merge_group_size)),
            )
            db.set_setting(
                "auto_merge_threshold",
                str(float(st.session_state.auto_merge_threshold)),
            )

        st.divider()
        st.markdown("**3. 过滤与重排：留下哪些片段**")
        preset_col1, preset_col2, preset_col3 = st.columns(3)
        with preset_col1:
            if st.button("宽松", use_container_width=True, key="retrieval_preset_loose"):
                set_retrieval_preset(8, 0.0, 0.50, 0.08)
        with preset_col2:
            if st.button("均衡", use_container_width=True, key="retrieval_preset_balanced"):
                set_retrieval_preset(5, 0.20, 0.56, 0.12)
        with preset_col3:
            if st.button("严格", use_container_width=True, key="retrieval_preset_strict"):
                set_retrieval_preset(4, 0.35, 0.62, 0.18)
        st.number_input(
            "返回 Top K",
            min_value=1,
            max_value=12,
            step=1,
            key="retrieval_limit",
            help="最多返回多少条检索片段。阈值过滤后可能少于这个数量。",
        )
        st.number_input(
            "关键词阈值",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            format="%.2f",
            key="retrieval_min_lexical_score",
            help="本地关键词相似度最低分。适合控制是否命中关键概念。",
        )
        st.slider(
            "相关度阈值",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="retrieval_min_score",
            help="综合排序分的最低要求。建议先用 0.20 左右；如果结果太杂，再逐步拉高到 0.35。",
        )
        st.slider(
            "Embedding 阈值",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="retrieval_min_semantic_score",
            help="向量相似度最低分。片段满足 embedding 阈值或关键词阈值任一条件，就会被保留。",
        )
        st.checkbox(
            "API 重排候选片段",
            key="retrieval_use_rerank",
            help="先召回更多候选，再调用当前大模型按问题相关性重排。会额外消耗一次模型调用。",
        )
        db.set_setting(
            "retrieval_use_rerank",
            "1" if st.session_state.retrieval_use_rerank else "0",
        )
        llm.rerank_enabled = bool(st.session_state.retrieval_use_rerank)


def render_global_prompt_panel():
    current = db.get_setting("global_answer_prompt", default_persona())
    with st.expander("通用回答 Prompt", expanded=False):
        prompt = st.text_area(
            "适用于所有回答",
            height=260,
            key="global_answer_prompt_editor",
            help="这里写长期稳定的角色、风格、边界和输出要求。",
        )
        if st.button("保存 Prompt", use_container_width=True, key="save_global_answer_prompt"):
            db.set_setting("global_answer_prompt", prompt)
            st.success("已保存 Prompt")
            rerun()
        if prompt != current:
            st.caption("有未保存修改。保存后刷新页面不会丢。")
    prompt = st.session_state.get("global_answer_prompt_editor", current)
    return prompt


def render_deposited_prompt_memories(limit=12):
    memories = [
        item
        for item in db.list_style_memories(limit=80)
        if item.get("source_type") == "revision_feedback_prompt"
    ][:limit]
    if not memories:
        return
    with st.expander("已沉淀 Prompt", expanded=False):
        st.caption("这些规则已经写入长期风格记忆，后续生成回答时会自动带给模型。")
        for index, memory in enumerate(memories, start=1):
            st.markdown(f"**{index}.** {memory['content']}")
            st.caption(f"{memory['created_at']} · weight {memory['weight']}")


def load_style_profile():
    if STYLE_PROFILE_PATH.exists():
        return STYLE_PROFILE_PATH.read_text(encoding="utf-8").strip()
    return ""


def style_memory_text(limit=12):
    memories = db.list_style_memories(limit=limit)
    if not memories:
        return ""
    return "\n".join(f"- {item['content']}" for item in memories)


def extract_feedback_deposit_memories(text):
    memories = []
    section = ""
    skip_section = False
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("【") and line.endswith("】"):
            section = line.strip("【】")
            skip_section = "不应沉淀" in section or "私有信息" in section
            continue
        if skip_section:
            continue
        if line.startswith(("-", "•")):
            line = line[1:].strip()
        elif len(line) < 8:
            continue
        if not line or line in {"无", "暂无", "没有"}:
            continue
        memory = f"{section}：{line}" if section else line
        memories.append(memory)
    return memories


def save_style_memories_from_deposit(
    answer_id,
    deposit_text,
    source_type="llm_feedback_deposit",
    weight=1.2,
):
    memories = extract_feedback_deposit_memories(deposit_text)
    for memory in memories:
        db.add_style_memory(
            memory,
            source_type=source_type,
            source_id=answer_id,
            weight=weight,
        )
    return len(memories)


def save_revision_feedback_memory(answer_id, question, instruction, current_draft):
    instruction = (instruction or "").strip()
    if not instruction:
        return "", "", 0
    db.add_style_memory(
        f"用户原始回答评价：{instruction}",
        source_type="revision_instruction_raw",
        source_id=answer_id,
        weight=0.9,
    )
    prompt = build_revision_feedback_memory_prompt(question, instruction, current_draft)
    memory_text = llm.generate(prompt)
    memory_count = save_style_memories_from_deposit(
        answer_id,
        memory_text,
        source_type="revision_feedback_prompt",
        weight=1.2,
    )
    return prompt, memory_text, memory_count + 1


def store_document(
    title,
    source,
    content,
    folder="默认",
    skip_if_source_exists=False,
    status_callback=None,
    cancelable=False,
):
    if skip_if_source_exists and db.document_exists_by_source(source):
        return None
    settings = rag_index_settings()
    document = db.add_document(title=title, source=source, content=content, folder=folder)
    document["content"] = content
    try:
        index_document_with_settings(document, settings, status_callback, cancelable=cancelable)
    except RagTaskCancelled:
        db.delete_document(document["id"])
        raise
    return document


def index_document_with_settings(document, settings, status_callback=None, cancelable=False):
    (
        chunk_size,
        chunk_overlap,
        build_summary_index,
        build_sentence_index,
        sentence_window_size,
        build_hierarchy_index,
        hierarchy_chunk_sizes,
    ) = settings
    if cancelable:
        check_rag_task_cancelled()
    if status_callback:
        status_callback("Vector Index")
    def vector_progress(done, total):
        if cancelable:
            check_rag_task_cancelled()
        if status_callback:
            status_callback(f"Vector Index · embedding {done}/{total} 批")

    db.replace_document_chunks(
        document["id"],
        build_chunk_items(
            document["content"],
            llm,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            progress_callback=vector_progress,
        ),
    )
    if cancelable:
        check_rag_task_cancelled()
    if build_summary_index:
        if status_callback:
            status_callback("Summary Index")
        def summary_progress(done, total):
            if cancelable:
                check_rag_task_cancelled()
            if status_callback:
                status_callback(f"Summary Index · 摘要 {done}/{total} 段")

        db.replace_document_summary(
            document["id"],
            build_summary_item(
                document["title"],
                document["content"],
                llm,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                progress_callback=summary_progress,
            ),
        )
    if cancelable:
        check_rag_task_cancelled()
    if build_sentence_index:
        if status_callback:
            status_callback("Sentence Window Index")
        def sentence_progress(done, total):
            if cancelable:
                check_rag_task_cancelled()
            if status_callback:
                status_callback(f"Sentence Window Index · chunk embedding {done}/{total} 批")

        db.replace_sentence_nodes(
            document["id"],
            build_sentence_window_items(
                document["content"],
                llm,
                window_size=sentence_window_size,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                progress_callback=sentence_progress,
            ),
        )
    if cancelable:
        check_rag_task_cancelled()
    if build_hierarchy_index:
        if status_callback:
            status_callback("Auto Merging Hierarchy Index")
        def hierarchy_progress(done, total):
            if cancelable:
                check_rag_task_cancelled()
            if status_callback:
                status_callback(f"Auto Merging Hierarchy Index · leaf embedding {done}/{total} 批")

        db.replace_hierarchy_nodes(
            document["id"],
            build_hierarchy_items(
                document["content"],
                llm,
                chunk_sizes=hierarchy_chunk_sizes,
                progress_callback=hierarchy_progress,
            ),
        )


def rebuild_rag_index(progress_callback=None):
    settings = rag_index_settings()
    documents = db.get_documents()
    total = len(documents)
    for index, document in enumerate(documents, start=1):
        check_rag_task_cancelled()
        def step_callback(step):
            check_rag_task_cancelled()
            if progress_callback:
                progress_callback(index, total, document, step)

        index_document_with_settings(document, settings, step_callback, cancelable=True)
        if progress_callback:
            progress_callback(index, total, document, "完成")
    return total


def rag_health():
    documents = db.document_index_status()
    return {
        "documents": documents,
        "missing_index": [doc for doc in documents if not doc["indexed"]],
        "missing_embeddings": [
            doc for doc in documents if doc["indexed"] and not doc["fully_embedded"]
        ],
        "missing_summaries": [doc for doc in documents if not doc["summary_indexed"]],
        "missing_summary_embeddings": [
            doc
            for doc in documents
            if doc["summary_indexed"] and not doc["summary_embedded"]
        ],
        "missing_sentence_index": [doc for doc in documents if not doc["sentence_indexed"]],
        "missing_sentence_embeddings": [
            doc
            for doc in documents
            if doc["sentence_indexed"] and not doc["sentence_fully_embedded"]
        ],
        "missing_hierarchy_index": [doc for doc in documents if not doc["hierarchy_indexed"]],
        "stale_model": [
            doc
            for doc in documents
            if doc["embedding_models"] and llm.embedding_model not in doc["embedding_models"]
        ],
    }


def format_bytes(size):
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024


def render_rag_index_config():
    with st.expander("入库索引配置", expanded=False):
        st.caption(
            "这里决定资料入库时预先生成哪些索引；回答工作台只是在检索阶段选择使用哪一种。"
        )
        rec_col, note_col = st.columns([0.32, 0.68])
        with rec_col:
            if st.button("使用推荐配置", use_container_width=True):
                apply_recommended_rag_settings()
                st.success("已填入推荐配置。旧资料需要重建索引。")
                rerun()
        with note_col:
            st.caption(
                "推荐适合你的心理书籍/知乎回答场景：先用摘要路由缩小文档，再用向量、句子窗口或层级合并找片段。"
            )
        current_chunk_size = int(st.session_state.get("rag_chunk_size", INDEX_CHUNK_SIZE))
        current_chunk_overlap = int(
            st.session_state.get("rag_chunk_overlap", INDEX_CHUNK_OVERLAP)
        )
        if current_chunk_overlap >= current_chunk_size:
            st.session_state.rag_chunk_overlap = max(0, current_chunk_size - 1)

        with st.container(border=True):
            st.markdown("**1. Vector Index：原文片段索引**")
            st.caption("最基础的 RAG 索引。把资料切成 chunk，给每个 chunk 生成 embedding。")
            st.number_input(
                "Chunk 长度",
                min_value=300,
                max_value=4000,
                step=100,
                key="rag_chunk_size",
                help="每个原文片段的最大字符数。短一点更精确，长一点上下文更完整。",
            )
            st.number_input(
                "Chunk 重叠",
                min_value=0,
                max_value=max(0, int(st.session_state.rag_chunk_size) - 1),
                step=20,
                key="rag_chunk_overlap",
                help="相邻片段重复多少字符，避免一句话被切断后丢上下文。",
            )
            st.number_input(
                "Embedding batch size",
                min_value=1,
                max_value=20,
                step=1,
                key="rag_embedding_batch_size",
                help="每次向 embedding 接口提交多少个片段。阿里 text-embedding-v4 上限是 20。",
            )

        with st.container(border=True):
            st.markdown("**2. Summary Index：文档路由索引**")
            st.caption("先用每篇资料的摘要筛文档，再进入具体片段检索。它不是最终回答依据。")
            st.checkbox(
                "生成文档摘要索引",
                key="rag_build_summary_index",
                help="每篇资料额外生成一段摘要和摘要向量，用来先判断应该检索哪份文档。",
            )
            st.caption("推荐：开启。摘要策略跟随当前 chunk：≤3 块本地摘要，4-10 块模型全文摘要，>10 块分段摘要后合并。")

        with st.container(border=True):
            st.markdown("**3A. Sentence Window：chunk 召回 + 句子窗口**")
            st.caption("入库时按当前 chunk 生成 embedding；检索命中 chunk 后，再在 chunk 内定位相关句子并返回前后窗口。")
            st.checkbox(
                "生成 Sentence Window 索引",
                key="rag_build_sentence_index",
                help="入库时复用当前 chunk 粒度建索引；检索时在命中的 chunk 里抽取句子窗口，速度比每句 embedding 快很多。",
            )
            st.number_input(
                "窗口前后句数",
                min_value=0,
                max_value=8,
                step=1,
                key="rag_sentence_window_size",
                help="chunk 命中后，在 chunk 内定位最相关句子，并返回前后多少句作为 window。",
            )

        with st.container(border=True):
            st.markdown("**3B. Auto Merging：层级合并索引**")
            auto_sizes = [
                int(st.session_state.rag_chunk_size),
                int(st.session_state.rag_chunk_size) * 3,
                int(st.session_state.rag_chunk_size) * 9,
            ]
            st.caption(
                "检索时先命中当前 chunk，多个 chunk 命中同一上层区域时返回更完整上下文。"
            )
            st.checkbox(
                "生成 Auto Merging 层级索引",
                key="rag_build_hierarchy_index",
                help="入库时按当前 chunk 自动构建 child -> parent -> grandparent；检索时命中多个 child 后合并 parent。",
            )
            st.caption(
                f"当前自动层级：{auto_sizes[0]},{auto_sizes[1]},{auto_sizes[2]}。"
                "含义是 child / parent / grandparent。"
            )

        (
            chunk_size,
            chunk_overlap,
            build_summary_index,
            build_sentence_index,
            sentence_window_size,
            build_hierarchy_index,
            hierarchy_chunk_sizes,
        ) = rag_index_settings()
        st.caption(
            f"当前入库配置：Vector chunk {chunk_size}/overlap {chunk_overlap} · "
            f"embedding batch {st.session_state.rag_embedding_batch_size} · "
            f"Summary {'开' if build_summary_index else '关'} · "
            f"Sentence Window {'开' if build_sentence_index else '关'}(±{sentence_window_size}) · "
            f"Auto Merging {'开' if build_hierarchy_index else '关'}({hierarchy_chunk_sizes})。"
        )
        st.warning("修改这里以后，只影响新导入资料；旧资料需要点击下面按钮重建索引。")
        task_col, cancel_col = st.columns([0.68, 0.32])
        with cancel_col:
            if st.button("暂停当前任务", use_container_width=True, key="cancel_rag_rebuild"):
                request_rag_task_cancel()
                st.warning("已请求暂停。正在进行中的 API 调用会在返回或超时后停止。")
        with task_col:
            start_rebuild = st.button(
                "按当前参数重建全部 RAG 索引",
                use_container_width=True,
            )
        if start_rebuild:
            clear_rag_task_cancel()
            progress_bar = st.progress(0)
            status_box = st.empty()
            detail_box = st.empty()

            def show_rebuild_progress(index, total, document, step):
                ratio = index / total if total else 1
                progress_bar.progress(ratio)
                status_box.info(
                    f"正在重建 {index}/{total}：{document['title']} · {step}"
                )
                detail_box.caption(
                    f"当前资料约 {len(document.get('content') or ''):,} 字。"
                )

            try:
                total = rebuild_rag_index(show_rebuild_progress)
                progress_bar.progress(1.0)
                status_box.success(f"RAG 索引已重建完成，共处理 {total} 份资料。")
                detail_box.caption("已生成 Vector、Summary、Sentence Window 和 Auto Merging 相关索引。")
            except RagTaskCancelled:
                status_box.warning("RAG 索引重建已暂停。已经完成的文件会保留，未完成的文件下次可重新重建。")
                detail_box.caption("再次点击重建时，会自动清除暂停标记并从头按当前参数重建。")


init_state()
llm.embedding_batch_size = max(
    1, min(20, int(st.session_state.get("rag_embedding_batch_size", llm.embedding_batch_size)))
)

st.title("心理内容 AI 运营助手")
model_status = llm.status()
st.caption(
    f"本机工作台 · 回答模型：{model_status['mode']} / {model_status['model']} · "
    f"RAG：{model_status['embedding_mode']} / {model_status['embedding_model']}"
)

MAIN_TABS = ["选题池", "RAG 维护", "回答工作台", "回答管理", "系统状态"]
query_tab = st.query_params.get("tab", "选题池")
if isinstance(query_tab, list):
    query_tab = query_tab[0] if query_tab else "选题池"
if query_tab not in MAIN_TABS:
    query_tab = "选题池"
active_tab = st.segmented_control(
    "功能区",
    MAIN_TABS,
    default=query_tab,
    label_visibility="collapsed",
    key="main_tab",
    width="stretch",
)
active_tab = active_tab or query_tab
if st.query_params.get("tab") != active_tab:
    st.query_params["tab"] = active_tab

if active_tab == "选题池":
    left, right = st.columns([0.9, 1.1], gap="large")

    with left:
        st.subheader("粘贴知乎页面 HTML")
        st.caption("从 DevTools 复制 `<head>`、整页 HTML，或者直接粘贴页面标题文本。系统会抽取标题、描述和关键词。")
        html_text = st.text_area(
            "HTML / 标题文本",
            height=420,
            placeholder="粘贴知乎页面 <head>...</head>、ariaTipText 片段，或直接粘贴标题。",
            label_visibility="collapsed",
            key="zhihu_html_import_text",
        )
        if st.button(
            "抽取并加入选题池",
            type="primary",
            use_container_width=True,
            key="import_zhihu_html_button",
        ):
            if not html_text.strip():
                st.warning("先粘贴 HTML 或标题文本，再点抽取。")
            else:
                try:
                    payload = import_question_from_html(html_text)
                    item = db.add_question(**payload)
                    st.success(f"已加入问题池：{item['title']}")
                except Exception as exc:
                    st.error(f"导入失败：{exc}")

    with right:
        st.subheader("问题列表")
        questions = db.list_questions()
        if not questions:
            st.caption("还没有问题。")
        for question in questions:
            with st.container(border=True):
                st.markdown(f"**{question['title']}**")
                st.markdown(
                    f"<span class='pill'>{status_label(question['status'])}</span>"
                    f"<span class='small-muted'>热度 {question['heat']} · {question['tags'] or '未标注'}</span>",
                    unsafe_allow_html=True,
                )
                if question["description"]:
                    st.caption(question["description"])
                if question["source_url"]:
                    st.caption(question["source_url"])
                if st.button("删除这条选题", key=f"delete_question_{question['id']}"):
                    db.delete_question(question["id"])
                    if st.session_state.selected_question_id == question["id"]:
                        remaining = db.list_questions()
                        if remaining:
                            select_question(remaining[0]["id"])
                        else:
                            st.session_state.selected_question_id = None
                    st.success("已删除选题")
                    rerun()

if active_tab == "RAG 维护":
    folders = db.list_document_folders()
    if st.session_state.rag_current_folder not in folders:
        st.session_state.rag_current_folder = folders[0] if folders else "默认"
    current_folder = st.session_state.rag_current_folder

    left, right = st.columns([1.12, 0.88], gap="large")

    with left:
        st.subheader("文件夹")
        with st.form("create_rag_folder"):
            new_folder = st.text_input("新建文件夹", placeholder="例如：哲学 / 心理咨询 / 我的旧回答")
            if st.form_submit_button("新建文件夹", use_container_width=True):
                try:
                    folder = db.add_document_folder(new_folder)
                    st.success(f"已新建「{folder}」")
                    rerun()
                except Exception as exc:
                    st.error(f"新建失败：{exc}")

        folders = db.list_document_folders()
        for folder in folders:
            count = len(db.list_documents(folder))
            open_col, delete_col = st.columns([0.78, 0.22])
            with open_col:
                label = f"{'● ' if folder == current_folder else ''}{folder}（{count}）"
                if st.button(label, key=f"open_folder_{folder}", use_container_width=True):
                    st.session_state.rag_current_folder = folder
                    rerun()
            with delete_col:
                if count == 0:
                    if st.button("删", key=f"delete_folder_{folder}", use_container_width=True):
                        try:
                            db.delete_document_folder(folder)
                            if current_folder == folder:
                                remaining_folders = db.list_document_folders()
                                st.session_state.rag_current_folder = (
                                    remaining_folders[0] if remaining_folders else ""
                                )
                            st.success(f"已删除空文件夹「{folder}」。")
                            rerun()
                        except Exception as exc:
                            st.error(f"删除失败：{exc}")

        render_rag_index_config()

    with right:
        docs = db.list_documents(current_folder)

        title_col, action_col = st.columns([0.68, 0.32])
        with title_col:
            st.subheader(f"「{current_folder}」")
            st.caption("当前文件夹中的资料会作为回答时可检索的知识来源。")
        with action_col:
            st.metric("资料数", len(docs))

        with st.expander("上传文件到当前文件夹", expanded=True):
            st.caption("支持 PDF、docx、doc、txt、md；导入时会自动清洗、分块并生成 RAG 索引。")
            uploads = st.file_uploader(
                "选择知识库文件",
                type=["pdf", "docx", "doc", "txt", "md"],
                accept_multiple_files=True,
                key=f"rag_uploads_{current_folder}",
            )
            import_col, cancel_import_col = st.columns([0.68, 0.32])
            with cancel_import_col:
                if st.button(
                    "暂停当前导入",
                    use_container_width=True,
                    key=f"cancel_rag_import_{current_folder}",
                ):
                    request_rag_task_cancel()
                    st.warning("已请求暂停。当前 API 调用会在返回或超时后停止。")
            with import_col:
                start_import = st.button(
                    f"导入到「{current_folder}」",
                    type="primary",
                    use_container_width=True,
                    key=f"rag_import_files_{current_folder}",
                )
            if start_import:
                if not uploads:
                    st.warning("请先选择文件。")
                else:
                    clear_rag_task_cancel()
                    imported = 0
                    skipped = 0
                    cancelled = False
                    import_progress = st.progress(0)
                    import_status = st.empty()
                    import_detail = st.empty()
                    total_uploads = len(uploads)
                    for upload_index, uploaded in enumerate(uploads, start=1):
                        try:
                            check_rag_task_cancelled()
                            import_progress.progress((upload_index - 1) / total_uploads)
                            import_status.info(
                                f"正在导入 {upload_index}/{total_uploads}：{uploaded.name}"
                            )
                            title = uploaded.name
                            if db.document_exists_by_title(title):
                                skipped += 1
                                import_detail.caption(f"{title} 已存在，已跳过。")
                                continue
                            title, content = extract_uploaded_text(uploaded)
                            if not content.strip():
                                import_detail.caption(f"{title} 没有提取到文字，已跳过。")
                                continue

                            def show_import_step(step):
                                check_rag_task_cancelled()
                                import_status.info(
                                    f"正在导入 {upload_index}/{total_uploads}：{title} · {step}"
                                )
                                import_detail.caption(f"当前资料约 {len(content):,} 字。")

                            store_document(
                                title=title,
                                source="上传文件",
                                content=content,
                                folder=current_folder,
                                status_callback=show_import_step,
                                cancelable=True,
                            )
                            imported += 1
                            import_progress.progress(upload_index / total_uploads)
                            import_detail.caption(f"{title} 已完成入库和索引。")
                        except RagTaskCancelled:
                            cancelled = True
                            import_status.warning("导入已暂停。当前未完成的新文件已清理，下次可以重新导入。")
                            import_detail.caption(
                                f"已导入 {imported} 个文件，跳过 {skipped} 个同名文件。"
                            )
                            break
                        except Exception as exc:
                            st.error(f"{uploaded.name} 导入失败：{exc}")
                    if not cancelled:
                        import_progress.progress(1.0)
                    if imported and not cancelled:
                        import_status.success(
                            f"导入完成：新增 {imported} 个文件，跳过 {skipped} 个同名文件。"
                        )
                        st.success(
                            f"已导入 {imported} 个文件到「{current_folder}」；"
                            f"跳过 {skipped} 个同名文件。"
                        )
                        rerun()
                    elif skipped and not cancelled:
                        st.info(f"没有新文件导入，已跳过 {skipped} 个同名文件。")

        if not docs:
            st.info("这个文件夹还没有资料。展开上面的上传区，放入第一份 PDF、docx 或文本文件。")
        folders_for_move = db.list_document_folders()
        for doc in docs[:50]:
            with st.container(border=True):
                info_col, action_col = st.columns([0.58, 0.42], vertical_alignment="center")
                with info_col:
                    st.markdown(f"**{doc['title']}**")
                    st.caption(f"{doc['source'] or '未注明来源'} · {doc['size']} 字")
                with action_col:
                    target_folders = [folder for folder in folders_for_move if folder != current_folder]
                    move_btn_col, delete_col = st.columns(2)
                    with move_btn_col:
                        if st.button(
                            "移动",
                            key=f"show_move_doc_{doc['id']}",
                            use_container_width=True,
                            disabled=not target_folders,
                        ):
                            if st.session_state.rag_move_doc_id == doc["id"]:
                                st.session_state.rag_move_doc_id = None
                            else:
                                st.session_state.rag_move_doc_id = doc["id"]
                            rerun()
                    with delete_col:
                        if st.button("删除", key=f"delete_doc_{doc['id']}", use_container_width=True):
                            db.delete_document(doc["id"])
                            if st.session_state.rag_move_doc_id == doc["id"]:
                                st.session_state.rag_move_doc_id = None
                            st.success("已删除")
                            rerun()
                if st.session_state.rag_move_doc_id == doc["id"]:
                    move_select_col, confirm_col, cancel_col = st.columns([0.62, 0.19, 0.19])
                    with move_select_col:
                        target_folder = st.selectbox(
                            "移动到",
                            target_folders,
                            key=f"move_doc_target_{doc['id']}",
                            disabled=not target_folders,
                            label_visibility="collapsed",
                        )
                    with confirm_col:
                        if st.button("确认", key=f"move_doc_{doc['id']}", use_container_width=True):
                            db.move_document(doc["id"], target_folder)
                            st.session_state.rag_move_doc_id = None
                            st.success(f"已移动到「{target_folder}」。")
                            rerun()
                    with cancel_col:
                        if st.button("取消", key=f"cancel_move_doc_{doc['id']}", use_container_width=True):
                            st.session_state.rag_move_doc_id = None
                            rerun()
                with st.expander("预览与 chunks"):
                    full_doc = db.get_document(doc["id"])
                    content = (full_doc or {}).get("content", "")
                    st.text_area(
                        "入库正文",
                        value=content[:8000],
                        height=260,
                        key=f"preview_doc_{doc['id']}",
                        disabled=True,
                    )
                    summaries = db.list_document_summaries(doc["id"])
                    if summaries:
                        st.text_area(
                            "摘要索引",
                            value=summaries[0]["summary"][:3000],
                            height=180,
                            key=f"preview_summary_{doc['id']}",
                            disabled=True,
                        )
                    chunks = db.list_document_chunks(doc["id"])
                    sentence_nodes = db.list_sentence_nodes(doc["id"])
                    hierarchy_nodes = db.list_hierarchy_nodes(doc["id"])
                    embedded = sum(1 for chunk in chunks if chunk.get("embedding"))
                    embedded_sentences = sum(1 for node in sentence_nodes if node.get("embedding"))
                    embedded_leaf_nodes = sum(
                        1
                        for node in hierarchy_nodes
                        if node.get("level") == 0 and node.get("embedding")
                    )
                    leaf_nodes = sum(1 for node in hierarchy_nodes if node.get("level") == 0)
                    summary_status = "1 summary" if summaries else "无摘要索引"
                    st.caption(
                        f"{len(chunks)} chunks · {embedded} chunk embeddings · {summary_status} · "
                        f"{len(sentence_nodes)} sentence nodes / {embedded_sentences} embeddings · "
                        f"{len(hierarchy_nodes)} hierarchy nodes / leaf {embedded_leaf_nodes}/{leaf_nodes}"
                    )
                    if chunks:
                        with st.expander("查看 chunks"):
                            for chunk in chunks[:8]:
                                st.markdown(f"**Chunk {chunk['chunk_index']}**")
                                st.write(chunk["content"][:1000])
                    if sentence_nodes:
                        with st.expander("查看 sentence window nodes"):
                            for node in sentence_nodes[:8]:
                                st.markdown(f"**Sentence {node['sentence_index']}**")
                                st.caption(f"命中句：{node['sentence'][:220]}")
                                st.write(node["window"][:1000])
                    if hierarchy_nodes:
                        with st.expander("查看 hierarchy nodes"):
                            for node in hierarchy_nodes[:12]:
                                st.markdown(
                                    f"**{node['node_key']} · level {node['level']} · parent {node.get('parent_key') or '无'}**"
                                )
                                st.write(node["content"][:800])

if active_tab == "回答工作台":
    answer_left, answer_main = st.columns([0.28, 0.72], gap="large")

    with answer_left:
        global_prompt = render_global_prompt_panel()
        style_profile = load_style_profile()
        style_memories = style_memory_text()
        with st.expander("个人语气文档", expanded=False):
            if style_profile:
                st.caption(f"已加载：{STYLE_PROFILE_PATH.name}，约 {len(style_profile)} 字")
                st.text(style_profile[:3000])
            else:
                st.caption("还没有生成个人语气文档。")
        with st.expander("长期风格记忆", expanded=False):
            memory_count = len(db.list_style_memories(limit=999))
            st.caption(f"已沉淀 {memory_count} 条人工修改偏好。")
            if style_memories:
                st.text(style_memories)

    with answer_main:
        render_question_picker("answer")
        question = current_question()

        if question:
            restore_answer_workspace(question)
            st.header(question["title"])
            st.caption(question["source_url"] or "未填写来源链接")
            if question.get("description"):
                st.write(question["description"])

            with st.container(border=True):
                st.markdown("### 1. 意图识别")
                intent_result_col, intent_action_col = st.columns([0.58, 0.42], gap="large")
                with intent_result_col:
                    if st.session_state.workflow_intent:
                        st.text_area(
                            "意图识别结果",
                            value=st.session_state.workflow_intent,
                            height=260,
                            key=f"workflow_intent_{question['id']}_{st.session_state.generation_run_id or 'empty'}",
                            disabled=True,
                            label_visibility="collapsed",
                        )
                    else:
                        st.caption("先运行意图识别，系统会把问题拆成回答角度、读者处境和检索关键词。")
                    render_feedback_history("已提意图意见", st.session_state.intent_feedback_history)
                with intent_action_col:
                    if st.session_state.pending_intent_feedback_clear:
                        st.session_state.intent_feedback_input = ""
                        st.session_state.pending_intent_feedback_clear = False
                    intent_feedback = st.text_area(
                        "对意图识别提意见",
                        height=140,
                        key="intent_feedback_input",
                        placeholder="比如：重点不是亲子矛盾，而是被长期规训后的自我要求和羞耻感。",
                    )
                    if st.button(
                        "运行意图识别 / 按意见重识别",
                        type="primary",
                        use_container_width=True,
                        key="run_intent_step",
                    ):
                        if intent_feedback.strip():
                            st.session_state.intent_feedback_history.append(intent_feedback.strip())
                        with st.spinner("正在识别问题意图..."):
                            intent_prompt = build_intent_prompt(
                                question,
                                "\n".join(st.session_state.intent_feedback_history),
                            )
                            st.session_state.workflow_intent = llm.generate(intent_prompt)
                        st.session_state.context = []
                        st.session_state.last_retrieval_query = ""
                        st.session_state.pending_intent_feedback_clear = True
                        st.success("意图识别已更新，下一步可以重新检索。")
                        rerun()

            with st.container(border=True):
                st.markdown("### 2. 检索片段")
                retrieval_config_col, retrieval_result_col = st.columns([0.42, 0.58], gap="large")
                with retrieval_config_col:
                    render_retrieval_config_panel()
                    if st.session_state.pending_retrieval_feedback_clear:
                        st.session_state.retrieval_feedback_input = ""
                        st.session_state.pending_retrieval_feedback_clear = False
                    retrieval_feedback = st.text_area(
                        "对检索片段提意见",
                        height=140,
                        key="retrieval_feedback_input",
                        placeholder="比如：不要只找共情片段，多找书里关于依恋、羞耻感、自我要求的论述。",
                    )
                    if st.button(
                        "检索知识库 / 按意见重检索",
                        type="primary",
                        use_container_width=True,
                        key="run_retrieval_step",
                    ):
                        if retrieval_feedback.strip():
                            st.session_state.retrieval_feedback_history.append(retrieval_feedback.strip())
                        with st.spinner("正在检索知识库..."):
                            if not st.session_state.workflow_intent:
                                intent_prompt = build_intent_prompt(
                                    question,
                                    "\n".join(st.session_state.intent_feedback_history),
                                )
                                st.session_state.workflow_intent = llm.generate(intent_prompt)
                            retrieval_guidance = "\n".join(st.session_state.retrieval_feedback_history)
                            search_query = workflow_search_query(
                                question,
                                st.session_state.workflow_intent,
                                retrieval_guidance,
                            )
                            st.session_state.last_retrieval_query = search_query
                            search_settings = retrieval_settings()
                            st.session_state.context = retriever.search(
                                search_query,
                                **search_settings,
                            )
                        st.session_state.pending_retrieval_feedback_clear = True
                        st.success("检索片段已更新。")
                        rerun()
                with retrieval_result_col:
                    if st.session_state.last_retrieval_query:
                        with st.expander("本次检索 Query", expanded=False):
                            st.text_area(
                                "Query",
                                value=st.session_state.last_retrieval_query,
                                height=180,
                                disabled=True,
                                label_visibility="collapsed",
                                key=f"retrieval_query_{question['id']}_{st.session_state.generation_run_id or 'empty'}",
                            )
                    render_context(st.session_state.context)
                    render_feedback_history("已提检索意见", st.session_state.retrieval_feedback_history)

            with st.container(border=True):
                st.markdown("### 3. 生成回答")
                generation_editor_col, generation_action_col = st.columns([0.64, 0.36], gap="large")
                with generation_action_col:
                    if st.session_state.pending_revision_instruction_clear:
                        st.session_state.answer_revision_instruction = ""
                        st.session_state.pending_revision_instruction_clear = False
                    draft_feedback = st.text_area(
                        "回答评价 / 重写要求",
                        height=160,
                        key="answer_revision_instruction",
                        placeholder="比如：不要瞎编例子；不是我真正经历过的事情就不要写；这段太空洞。",
                    )
                    if st.button(
                        "生成回答 / 按评价重写",
                        type="primary",
                        use_container_width=True,
                        key="run_generation_step",
                    ):
                        if draft_feedback.strip():
                            st.session_state.revision_instruction_history.append(draft_feedback.strip())
                            st.session_state.last_revision_instruction = draft_feedback.strip()
                        else:
                            st.session_state.last_revision_instruction = ""

                        with st.spinner("正在生成回答..."):
                            if not st.session_state.workflow_intent:
                                intent_prompt = build_intent_prompt(
                                    question,
                                    "\n".join(st.session_state.intent_feedback_history),
                                )
                                st.session_state.workflow_intent = llm.generate(intent_prompt)
                            if not st.session_state.context:
                                retrieval_guidance = "\n".join(st.session_state.retrieval_feedback_history)
                                search_query = workflow_search_query(
                                    question,
                                    st.session_state.workflow_intent,
                                    retrieval_guidance,
                                )
                                st.session_state.last_retrieval_query = search_query
                                search_settings = retrieval_settings()
                                st.session_state.context = retriever.search(
                                    search_query,
                                    **search_settings,
                                )
                            current_draft = st.session_state.get("answer_edit_text") or st.session_state.draft_text
                            revision_memory_text = ""
                            if draft_feedback.strip():
                                (
                                    revision_memory_prompt,
                                    revision_memory_text,
                                    revision_memory_count,
                                ) = save_revision_feedback_memory(
                                    st.session_state.get("answer_id"),
                                    question,
                                    draft_feedback,
                                    current_draft,
                                )
                                st.session_state.last_revision_memory_prompt = revision_memory_prompt
                                st.session_state.last_revision_memory_text = revision_memory_text
                                st.session_state.last_revision_memory_count = revision_memory_count
                            feedback_history_text = "\n".join(st.session_state.revision_instruction_history)
                            guidance = workflow_guidance(
                                st.session_state.workflow_intent,
                                "\n".join(st.session_state.retrieval_feedback_history),
                                feedback_history_text,
                                current_draft,
                            )
                            style_memories_for_prompt = style_memories
                            if revision_memory_text.strip():
                                style_memories_for_prompt = "\n".join(
                                    part
                                    for part in [
                                        style_memories,
                                        revision_memory_text,
                                    ]
                                    if part.strip()
                                )
                            prompt = build_answer_prompt(
                                question,
                                st.session_state.context,
                                guidance,
                                global_prompt,
                                style_profile,
                                style_memories_for_prompt,
                            )
                            draft = llm.generate(prompt)

                        answer = db.save_answer(question["id"], draft, st.session_state.context)
                        run = db.add_generation_run(
                            question_id=question["id"],
                            run_type="rewrite" if draft_feedback.strip() else "draft",
                            model=model_status["model"],
                            prompt=prompt,
                            curl=llm.chat_curl(prompt),
                            context=st.session_state.context,
                            style_memories_text=style_memories_for_prompt,
                            style_profile_text=style_profile,
                            global_prompt=global_prompt,
                            guidance=guidance,
                            draft=draft,
                            answer_id=answer["id"],
                        )
                        db.update_question_status(question["id"], "drafted")
                        st.session_state.answer_id = answer["id"]
                        st.session_state.generation_run_id = run["id"]
                        st.session_state.draft_text = draft
                        st.session_state.answer_edit_text = draft
                        st.session_state.review_text = ""
                        st.session_state.last_generation_prompt = prompt
                        st.session_state.answer_status_label = "按评价重写稿" if draft_feedback.strip() else "AI 初稿"
                        st.session_state.answer_status_time = now_label()
                        st.session_state.pending_revision_instruction_clear = True
                        st.success("回答已生成。")
                        rerun()

                    render_feedback_history("已提回答评价", st.session_state.revision_instruction_history)
                    if st.session_state.get("last_revision_memory_text"):
                        with st.expander("本次评价已沉淀为长期 Prompt", expanded=False):
                            st.caption(
                                f"已写入 {st.session_state.get('last_revision_memory_count', 0)} 条长期风格记忆。"
                            )
                            st.text_area(
                                "沉淀结果",
                                value=st.session_state.last_revision_memory_text,
                                height=160,
                                key=f"revision_memory_text_{question['id']}_{st.session_state.generation_run_id or 'empty'}",
                                disabled=True,
                                label_visibility="collapsed",
                            )
                            st.caption("沉淀 Prompt")
                            st.text_area(
                                "沉淀 Prompt",
                                value=st.session_state.last_revision_memory_prompt,
                                height=180,
                                key=f"revision_memory_prompt_{question['id']}_{st.session_state.generation_run_id or 'empty'}",
                                disabled=True,
                                label_visibility="collapsed",
                            )
                    render_deposited_prompt_memories()

                with generation_editor_col:
                    render_answer_status()
                    if st.session_state.pending_answer_edit_text is not None:
                        st.session_state.answer_edit_text = st.session_state.pending_answer_edit_text
                        st.session_state.pending_answer_edit_text = None
                    edited = st.text_area(
                        "回答正文",
                        height=420,
                        label_visibility="collapsed",
                        key="answer_edit_text",
                        placeholder="生成回答后在这里直接编辑，满意后保存为人工终稿。",
                    )
                    st.session_state.draft_text = edited

            with st.container(border=True):
                st.markdown("### 4. RAG Triad 评价")
                st.caption("单独检查当前稿件是否切题、检索资料是否相关、回答判断是否有依据。")
                eval_col1, eval_col2 = st.columns([0.36, 0.64])
                with eval_col1:
                    if st.button(
                        "评价当前稿件",
                        type="primary",
                        use_container_width=True,
                        key="run_rag_triad_eval",
                    ):
                        edited_for_eval = st.session_state.get("answer_edit_text", "")
                        if not edited_for_eval.strip():
                            st.warning("当前稿件为空，不能评价。")
                        elif not st.session_state.context:
                            st.warning("当前没有检索依据，先检索或生成一版回答。")
                        else:
                            with st.spinner("正在评价回答、检索资料和证据支撑..."):
                                eval_prompt = build_rag_triad_eval_prompt(
                                    question,
                                    edited_for_eval,
                                    st.session_state.context,
                                )
                                eval_text = llm.generate(eval_prompt)
                                eval_result = parse_json_object(eval_text)
                                st.session_state.last_rag_eval_prompt = eval_prompt
                                st.session_state.last_rag_eval = eval_result
                                if st.session_state.generation_run_id:
                                    db.update_generation_run_feedback(
                                        st.session_state.generation_run_id,
                                        {"rag_triad": eval_result},
                                    )
                            st.success("RAG Triad 评价完成。")
                with eval_col2:
                    st.caption("评价完成后，下方会给出分数、原因和建议下一步动作。")
                render_rag_triad(st.session_state.get("last_rag_eval"), show_empty=True)

            with st.container(border=True):
                st.markdown("### 5. 人工终稿与沉淀")
                st.caption("你在第 3 步写过的回答评价，会在重写时自动沉淀进长期风格记忆；这里负责保存最终版本。")
                if st.button(
                    "保存人工终稿",
                    type="primary",
                    use_container_width=True,
                    key="answer_save_final",
                ):
                    edited = st.session_state.get("answer_edit_text", "")
                    if not st.session_state.answer_id:
                        st.warning("请先生成一版 AI 初稿，再保存。")
                    else:
                        answer = db.get_answer(st.session_state.answer_id)
                        feedback = analyze_edit(answer["draft"], edited)
                        db.save_edited_answer(st.session_state.answer_id, edited, feedback)
                        db.complete_generation_run(st.session_state.answer_id, edited, feedback)
                        store_document(
                            title=f"我的人工终稿：{question['title']}",
                            source=f"past_answer:{st.session_state.answer_id}",
                            content=edited,
                            folder="我的旧回答",
                            skip_if_source_exists=True,
                        )
                        db.update_question_status(question["id"], "edited")
                        st.session_state.answer_status_label = "人工终稿已保存"
                        st.session_state.answer_status_time = now_label()
                        st.success("终稿已保存，并已自动加入知识库。")

if active_tab == "回答管理":
    trace_tab, memory_tab, feedback_tab = st.tabs(["生成轨迹", "风格记忆", "人工编辑反馈"])

    with trace_tab:
        st.subheader("生成轨迹与满意样本")
        runs = db.list_generation_runs(limit=80)
        if not runs:
            st.caption("生成初稿或重写后，这里会记录 prompt、curl、检索依据、初稿和人工终稿。")
        for run in runs:
            with st.container(border=True):
                st.markdown(f"**{run['question_title']}**")
                st.caption(
                    f"{run['run_type']} · {run['status']} · {run['model']} · "
                    f"{run['updated_at']} · {'满意样本' if run['is_satisfied'] else '未标记'}"
                )
                col_a, col_b, col_c = st.columns([0.24, 0.24, 0.52])
                with col_a:
                    rating = st.selectbox(
                        "评分",
                        [None, 1, 2, 3, 4, 5],
                        index=0 if run.get("rating") is None else [None, 1, 2, 3, 4, 5].index(run["rating"]),
                        key=f"run_rating_{run['id']}",
                    )
                with col_b:
                    satisfied = st.checkbox(
                        "满意样本",
                        value=bool(run["is_satisfied"]),
                        key=f"run_satisfied_{run['id']}",
                    )
                with col_c:
                    note = st.text_input(
                        "备注",
                        value=run.get("note") or "",
                        key=f"run_note_{run['id']}",
                        placeholder="比如：风格接近，可作为 few-shot；或：内容好但太像报告。",
                    )
                if st.button("保存样本标记", key=f"save_run_rating_{run['id']}"):
                    db.update_generation_run_rating(run["id"], rating, satisfied, note)
                    st.success("已保存")
                    rerun()
                rag_triad = (run.get("feedback") or {}).get("rag_triad")
                if rag_triad:
                    render_rag_triad(rag_triad)
                with st.expander("查看输入与输出"):
                    st.caption("Prompt")
                    st.text_area(
                        "Prompt",
                        value=run["prompt"],
                        height=220,
                        key=f"run_prompt_{run['id']}",
                        disabled=True,
                        label_visibility="collapsed",
                    )
                    st.caption("Curl")
                    st.code(run["curl"] or "", language="bash")
                    st.caption("AI 初稿")
                    st.text_area(
                        "AI 初稿",
                        value=run.get("draft") or "",
                        height=180,
                        key=f"run_draft_{run['id']}",
                        disabled=True,
                        label_visibility="collapsed",
                    )
                    if run.get("final"):
                        st.caption("人工终稿")
                        st.text_area(
                            "人工终稿",
                            value=run["final"],
                            height=180,
                            key=f"run_final_{run['id']}",
                            disabled=True,
                            label_visibility="collapsed",
                        )

    with memory_tab:
        st.subheader("长期风格记忆")
        memories = db.list_style_memories(limit=80)
        if not memories:
            st.caption("保存人工终稿后，这里会自动沉淀你的修改偏好。")
        for memory in memories:
            with st.container(border=True):
                st.write(memory["content"])
                st.caption(
                    f"{memory['source_type']} · weight {memory['weight']} · {memory['created_at']}"
                )
                if st.button("删除这条记忆", key=f"delete_style_memory_{memory['id']}"):
                    db.delete_style_memory(memory["id"])
                    st.success("已删除")
                    rerun()

    with feedback_tab:
        st.subheader("人工编辑反馈")
        feedback_items = db.list_feedback(limit=30)
        if not feedback_items:
            st.caption("保存人工终稿后，这里会记录 AI 初稿和人工终稿的差异。")
        for item in feedback_items:
            feedback = item["feedback"]
            with st.container(border=True):
                st.caption(f"问题 #{item['question_id']} · 修改幅度 {feedback.get('change_ratio')}")
                for note in feedback.get("style_notes", []):
                    st.write(f"- {note}")
                added = feedback.get("added_sentences") or []
                if added:
                    st.caption("人工新增表达")
                    for sentence in added[:3]:
                        st.write(f"“{sentence}”")

if active_tab == "系统状态":
    st.subheader("数据库与 RAG 状态")
    stats = db.storage_stats()
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("SQLite 文件", format_bytes(stats["db_size_bytes"]))
    col2.metric("知识库原文", f"{stats['document_chars']:,} 字")
    col3.metric("RAG chunks", stats["tables"]["document_chunks"])
    col4.metric("已向量化 chunks", stats["embedded_chunks"])
    col5.metric("摘要索引", stats["tables"].get("document_summaries", 0))
    col6.metric("句子节点", stats["tables"].get("sentence_nodes", 0))

    st.subheader("表数据量")
    st.json(stats["tables"])

    st.subheader("当前模型配置")
    st.write(f"回答模型：`{model_status['model']}`")
    st.write(f"回答接口：`{model_status['base_url']}`")
    st.write(f"Embedding 模式：`{model_status['embedding_mode']}`")
    st.write(f"Embedding 模型：`{model_status['embedding_model']}`")
    st.write(f"Embedding batch size：`{model_status.get('embedding_batch_size', 20)}`")
    st.write(
        f"API 重排：`{'开启' if st.session_state.get('retrieval_use_rerank') else '关闭'}`"
    )
    st.caption("如果 embedding_mode 是 local，说明没有配置 embedding API，RAG 会自动使用本地词面检索。")
    if stats["embedding_models"]:
        st.write(f"索引中出现过的 embedding 模型：`{', '.join(stats['embedding_models'])}`")
    else:
        st.write("索引中还没有 embedding 向量。")
    st.write(
        f"当前 chunk 参数：`{st.session_state.rag_chunk_size}` 字 / "
        f"重叠 `{st.session_state.rag_chunk_overlap}` 字"
    )
    st.write(
        f"摘要索引占用：`{stats['summary_chars']:,}` 字；"
        f"已向量化摘要：`{stats['embedded_summaries']}`"
    )
    st.write(
        f"Sentence Window：`{stats['tables'].get('sentence_nodes', 0)}` 个 chunk 窗口节点；"
        f"已向量化：`{stats['embedded_sentences']}`"
    )
    st.write(
        f"Auto Merging 层级节点：`{stats['tables'].get('hierarchy_nodes', 0)}`；"
        f"leaf 已向量化：`{stats['embedded_hierarchy_leaf']}`"
    )
    test_col, log_col = st.columns([0.35, 0.65])
    with test_col:
        if st.button("测试 Embedding API", use_container_width=True):
            try:
                result = llm.embed_texts(
                    ["这是一次 embedding 连通性测试。"],
                    batch_size=1,
                    raise_on_error=True,
                )
                if result and result[0]:
                    st.success(f"Embedding API 正常，向量维度：{len(result[0])}")
                else:
                    st.error("Embedding API 没有返回有效向量。")
            except Exception as exc:
                st.error(f"Embedding API 测试失败：{exc}")
    with log_col:
        st.caption(f"日志文件：`{LOG_PATH}`")
        if getattr(llm, "last_embedding_error", ""):
            st.error(llm.last_embedding_error)

    st.subheader("知识库占用明细")
    health = rag_health()
    if (
        health["missing_index"]
        or health["missing_embeddings"]
        or health["missing_summaries"]
        or health["missing_summary_embeddings"]
        or health["missing_sentence_index"]
        or health["missing_sentence_embeddings"]
        or health["missing_hierarchy_index"]
    ):
        st.warning(
            f"需要关注：未生成索引 {len(health['missing_index'])} 份，"
            f"未生成 chunk 向量 {len(health['missing_embeddings'])} 份，"
            f"未生成摘要索引 {len(health['missing_summaries'])} 份，"
            f"未生成摘要向量 {len(health['missing_summary_embeddings'])} 份，"
            f"未生成句子索引 {len(health['missing_sentence_index'])} 份，"
            f"句子向量不完整 {len(health['missing_sentence_embeddings'])} 份，"
            f"未生成层级索引 {len(health['missing_hierarchy_index'])} 份。"
        )
    else:
        st.success("RAG 索引状态正常。")
    for doc in health["documents"]:
        models = ", ".join(doc["embedding_models"]) if doc["embedding_models"] else "无"
        chunk_sizes = ", ".join(str(value) for value in doc["chunk_sizes"]) or "未知"
        chunk_overlaps = ", ".join(str(value) for value in doc["chunk_overlaps"]) or "未知"
        summary_status = "有摘要索引" if doc["summary_indexed"] else "无摘要索引"
        if doc["summary_indexed"] and not doc["summary_embedded"]:
            summary_status += "，无摘要向量"
        sentence_status = (
            f"sentence {doc['sentence_embedded_count']}/{doc['sentence_node_count']}"
            if doc["sentence_indexed"]
            else "无句子索引"
        )
        hierarchy_status = (
            f"hierarchy {doc['hierarchy_node_count']} nodes · leaf "
            f"{doc['hierarchy_leaf_embedded_count']}/{doc['hierarchy_leaf_count']}"
            if doc["hierarchy_indexed"]
            else "无层级索引"
        )
        with st.container(border=True):
            st.markdown(f"**{doc['title']}**")
            st.caption(
                f"{doc.get('folder', '默认')} · {doc['source'] or '未注明来源'} · {doc['size']} 字 · "
                f"{doc['chunk_count']} chunks · {doc['embedded_count']} embeddings · "
                f"模型：{models} · chunk {chunk_sizes}/overlap {chunk_overlaps} · "
                f"{summary_status} · {sentence_status} · {hierarchy_status}"
            )
            if doc["chunk_count"] and doc["embedded_count"] < doc["chunk_count"]:
                st.error(
                    f"Chunk 向量不完整：{doc['embedded_count']}/{doc['chunk_count']}。"
                    "需要修复 embedding API 后重建索引。"
                )
            if doc["sentence_node_count"] and doc["sentence_embedded_count"] < doc["sentence_node_count"]:
                st.warning(
                    f"Sentence Window 向量不完整："
                    f"{doc['sentence_embedded_count']}/{doc['sentence_node_count']}。"
                )
            if doc["hierarchy_leaf_count"] and doc["hierarchy_leaf_embedded_count"] < doc["hierarchy_leaf_count"]:
                st.warning(
                    f"Auto Merging leaf 向量不完整："
                    f"{doc['hierarchy_leaf_embedded_count']}/{doc['hierarchy_leaf_count']}。"
                )

    st.subheader("最近日志")
    if LOG_PATH.exists():
        lines = LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        st.code("\n".join(lines) or "暂无日志", language="text")
    else:
        st.caption("暂无日志文件。")
