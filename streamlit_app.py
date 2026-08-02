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
    default_persona,
)
from psych_ai_assistant.retrieval import Retriever, build_chunk_items
from psych_ai_assistant.zhihu import import_question_from_html


ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "assistant.sqlite3"
STYLE_PROFILE_PATH = ROOT / "data" / "style_voice_profile.md"


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
        "pending_intent_feedback_clear": False,
        "answer_status_label": "",
        "answer_status_time": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def rerun():
    st.rerun()


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
    st.session_state.review_text = ""
    st.session_state.last_generation_prompt = ""
    st.session_state.pending_answer_edit_text = None
    st.session_state.last_revision_instruction = ""
    st.session_state.revision_instruction_history = []
    st.session_state.pending_revision_instruction_clear = False
    st.session_state.workflow_intent = ""
    st.session_state.intent_feedback_history = []
    st.session_state.pending_intent_feedback_clear = False
    st.session_state.answer_status_label = ""
    st.session_state.answer_status_time = ""


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


def now_label():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def extract_revision_instruction(prompt):
    if "生成意见：" in prompt:
        text = prompt.split("生成意见：", 1)[1]
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
    for marker in ["生成意见：", "当前稿件：", "【通用回答 Prompt】", "【个人语气文档】"]:
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


def workflow_guidance(intent, draft_feedback="", current_draft=""):
    parts = []
    if intent.strip():
        parts.append(f"意图识别结果：\n{intent.strip()}")
    if draft_feedback.strip():
        parts.append(f"生成意见：\n{draft_feedback.strip()}")
    if current_draft.strip() and draft_feedback.strip():
        parts.append(f"当前稿件：\n{current_draft.strip()}")
    return "\n\n".join(parts)


def workflow_search_query(question, intent):
    return question_query_text(question) + "\n\n" + (intent or "")


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
            st.caption(f"{item['source'] or '未注明来源'} · {mode}")
            st.write(item["content"])


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


def load_style_profile():
    if STYLE_PROFILE_PATH.exists():
        return STYLE_PROFILE_PATH.read_text(encoding="utf-8").strip()
    return ""


def style_memory_text(limit=12):
    memories = db.list_style_memories(limit=limit)
    if not memories:
        return ""
    return "\n".join(f"- {item['content']}" for item in memories)


def save_style_memories_from_feedback(answer_id, feedback):
    for note in feedback.get("style_notes", []):
        db.add_style_memory(note, source_type="manual_edit", source_id=answer_id, weight=1.0)
    for sentence in feedback.get("added_sentences", [])[:3]:
        db.add_style_memory(
            f"人工终稿常补充这种表达：{sentence}",
            source_type="manual_added_sentence",
            source_id=answer_id,
            weight=0.8,
        )


def store_document(title, source, content, folder="默认", skip_if_source_exists=False):
    if skip_if_source_exists and db.document_exists_by_source(source):
        return None
    document = db.add_document(title=title, source=source, content=content, folder=folder)
    db.replace_document_chunks(document["id"], build_chunk_items(content, llm))
    return document


def rebuild_rag_index():
    for document in db.get_documents():
        db.replace_document_chunks(document["id"], build_chunk_items(document["content"], llm))


def rag_health():
    documents = db.document_index_status()
    return {
        "documents": documents,
        "missing_index": [doc for doc in documents if not doc["indexed"]],
        "missing_embeddings": [
            doc for doc in documents if doc["indexed"] and not doc["fully_embedded"]
        ],
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


init_state()

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
        with st.form("import_zhihu_html"):
            html_text = st.text_area(
                "HTML / 标题文本",
                height=420,
                placeholder="粘贴知乎页面 <head>...</head>、ariaTipText 片段，或直接粘贴标题。",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("抽取并加入选题池", use_container_width=True)
            if submitted:
                try:
                    payload = import_question_from_html(html_text)
                    item = db.add_question(**payload)
                    st.success(f"已加入问题池：{item['title']}")
                    rerun()
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

    left, right = st.columns([0.72, 1.28], gap="large")

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
                                st.session_state.rag_current_folder = "默认"
                            st.success(f"已删除空文件夹「{folder}」。")
                            rerun()
                        except Exception as exc:
                            st.error(f"删除失败：{exc}")

    with right:
        docs = db.list_documents(current_folder)

        title_col, action_col = st.columns([0.68, 0.32])
        with title_col:
            st.subheader(f"「{current_folder}」")
            st.caption("当前文件夹中的资料会作为回答时可检索的知识来源。")
        with action_col:
            st.metric("资料数", len(docs))

        with st.expander("上传文件到当前文件夹", expanded=True):
            st.caption("支持 PDF、docx、doc、txt、md；导入时会自动清洗、分块并生成向量索引。")
            uploads = st.file_uploader(
                "选择知识库文件",
                type=["pdf", "docx", "doc", "txt", "md"],
                accept_multiple_files=True,
                key=f"rag_uploads_{current_folder}",
            )
            if st.button(
                f"导入到「{current_folder}」",
                type="primary",
                use_container_width=True,
                key=f"rag_import_files_{current_folder}",
            ):
                if not uploads:
                    st.warning("请先选择文件。")
                else:
                    imported = 0
                    skipped = 0
                    for uploaded in uploads:
                        try:
                            title = uploaded.name
                            if db.document_exists_by_title(title):
                                skipped += 1
                                st.info(f"{title} 已存在，已跳过。")
                                continue
                            title, content = extract_uploaded_text(uploaded)
                            if not content.strip():
                                st.warning(f"{title} 没有提取到文字。")
                                continue
                            st.caption(f"{title} 提取到 {len(content):,} 字。")
                            store_document(
                                title=title,
                                source="上传文件",
                                content=content,
                                folder=current_folder,
                            )
                            imported += 1
                        except Exception as exc:
                            st.error(f"{uploaded.name} 导入失败：{exc}")
                    if imported:
                        st.success(
                            f"已导入 {imported} 个文件到「{current_folder}」；"
                            f"跳过 {skipped} 个同名文件。"
                        )
                        rerun()
                    elif skipped:
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
                    chunks = db.list_document_chunks(doc["id"])
                    embedded = sum(1 for chunk in chunks if chunk.get("embedding"))
                    st.caption(f"{len(chunks)} chunks · {embedded} embeddings")
                    if chunks:
                        with st.expander("查看 chunks"):
                            for chunk in chunks[:8]:
                                st.markdown(f"**Chunk {chunk['chunk_index']}**")
                                st.write(chunk["content"][:1000])

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

            st.subheader("回答流程")
            render_flow_steps()

            with st.container(border=True):
                st.markdown("**流程控制**")
                control_col, feedback_col = st.columns([0.38, 0.62], gap="large")
                with control_col:
                    run_from = st.radio(
                        "回退到哪一步",
                        ["意图识别", "生成 AI 初稿"],
                        horizontal=True,
                        key="workflow_run_from",
                    )
                    st.caption("检索不准时，选“意图识别”并写清楚检索方向；系统会重新识别、重新检索、重新生成。")
                    run_workflow = st.button(
                        "运行流程 / 按意见重跑",
                        type="primary",
                        use_container_width=True,
                        key="run_answer_workflow",
                    )
                with feedback_col:
                    if st.session_state.pending_intent_feedback_clear:
                        st.session_state.intent_feedback_input = ""
                        st.session_state.pending_intent_feedback_clear = False
                    intent_feedback = st.text_area(
                        "对意图识别提意见",
                        height=95,
                        key="intent_feedback_input",
                        placeholder="比如：不要把问题理解成亲子矛盾，重点是被亚洲父母规训后的自我要求。",
                    )
                    if st.session_state.pending_revision_instruction_clear:
                        st.session_state.answer_revision_instruction = ""
                        st.session_state.pending_revision_instruction_clear = False
                    draft_feedback = st.text_area(
                        "对 AI 初稿提意见",
                        height=95,
                        key="answer_revision_instruction",
                        placeholder="比如：不用瞎编例子；不是我真正经历过的事情就不要写；减少空洞共情。",
                    )

                if run_workflow:
                    run_intent = run_from == "意图识别" or not st.session_state.workflow_intent
                    if run_intent and intent_feedback.strip():
                        st.session_state.intent_feedback_history.append(intent_feedback.strip())
                    if draft_feedback.strip():
                        st.session_state.revision_instruction_history.append(draft_feedback.strip())
                        st.session_state.last_revision_instruction = draft_feedback.strip()
                    else:
                        st.session_state.last_revision_instruction = ""

                    with st.spinner("正在运行回答流程..."):
                        if run_intent:
                            intent_prompt = build_intent_prompt(
                                question,
                                "\n".join(st.session_state.intent_feedback_history),
                            )
                            st.session_state.workflow_intent = llm.generate(intent_prompt)
                            st.session_state.context = retriever.search(
                                workflow_search_query(question, st.session_state.workflow_intent),
                                limit=5,
                            )
                        elif not st.session_state.context:
                            st.session_state.context = retriever.search(
                                workflow_search_query(question, st.session_state.workflow_intent),
                                limit=5,
                            )

                        current_draft = st.session_state.get("answer_edit_text") or st.session_state.draft_text
                        guidance = workflow_guidance(
                            st.session_state.workflow_intent,
                            draft_feedback,
                            current_draft,
                        )
                        prompt = build_answer_prompt(
                            question,
                            st.session_state.context,
                            guidance,
                            global_prompt,
                            style_profile,
                            style_memories,
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
                        style_memories_text=style_memories,
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
                    st.session_state.answer_status_label = "按意见生成稿" if draft_feedback.strip() else "AI 初稿"
                    st.session_state.answer_status_time = now_label()
                    st.session_state.pending_intent_feedback_clear = True
                    st.session_state.pending_revision_instruction_clear = True
                    st.success("流程已完成")
                    rerun()

            step1, step2 = st.columns(2)
            with step1:
                st.subheader("意图识别")
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
                    st.caption("运行流程后显示意图识别结果。")
                if st.session_state.intent_feedback_history:
                    st.caption("已提意图意见")
                    for index, item in enumerate(st.session_state.intent_feedback_history, start=1):
                        st.caption(f"{index}. {item}")

            with step2:
                st.subheader("检索依据")
                render_context(st.session_state.context)

            st.subheader("生成 AI 初稿")
            render_answer_status()
            if st.session_state.pending_answer_edit_text is not None:
                st.session_state.answer_edit_text = st.session_state.pending_answer_edit_text
                st.session_state.pending_answer_edit_text = None
            edited = st.text_area(
                "回答正文",
                height=420,
                label_visibility="collapsed",
                key="answer_edit_text",
                placeholder="运行流程后在这里编辑 AI 初稿。",
            )
            st.session_state.draft_text = edited
            if st.session_state.revision_instruction_history:
                st.caption("已提生成意见")
                for index, item in enumerate(st.session_state.revision_instruction_history, start=1):
                    st.caption(f"{index}. {item}")
            if st.button("保存人工终稿并记录反馈", use_container_width=True, key="answer_save_final"):
                if not st.session_state.answer_id:
                    st.warning("请先运行流程生成一版 AI 初稿，再保存反馈。")
                else:
                    answer = db.get_answer(st.session_state.answer_id)
                    feedback = analyze_edit(answer["draft"], edited)
                    db.save_edited_answer(st.session_state.answer_id, edited, feedback)
                    db.complete_generation_run(st.session_state.answer_id, edited, feedback)
                    save_style_memories_from_feedback(st.session_state.answer_id, feedback)
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
                    st.success("终稿已保存，反馈已记录，并已自动加入知识库。")

            if st.session_state.last_generation_prompt:
                st.subheader("本次 Prompt")
                st.text_area(
                    "Prompt",
                    value=st.session_state.last_generation_prompt,
                    height=280,
                    key=f"preview_prompt_{question['id']}_{st.session_state.generation_run_id or 'empty'}",
                    disabled=True,
                    label_visibility="collapsed",
                )

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
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("SQLite 文件", format_bytes(stats["db_size_bytes"]))
    col2.metric("知识库原文", f"{stats['document_chars']:,} 字")
    col3.metric("RAG chunks", stats["tables"]["document_chunks"])
    col4.metric("已向量化 chunks", stats["embedded_chunks"])

    st.subheader("表数据量")
    st.json(stats["tables"])

    st.subheader("当前模型配置")
    st.write(f"回答模型：`{model_status['model']}`")
    st.write(f"回答接口：`{model_status['base_url']}`")
    st.write(f"Embedding 模式：`{model_status['embedding_mode']}`")
    st.write(f"Embedding 模型：`{model_status['embedding_model']}`")
    st.caption("如果 embedding_mode 是 local，说明没有配置 embedding API，RAG 会自动使用本地词面检索。")
    if stats["embedding_models"]:
        st.write(f"索引中出现过的 embedding 模型：`{', '.join(stats['embedding_models'])}`")
    else:
        st.write("索引中还没有 embedding 向量。")

    st.subheader("知识库占用明细")
    health = rag_health()
    if health["missing_index"] or health["missing_embeddings"]:
        st.warning(
            f"需要关注：未生成索引 {len(health['missing_index'])} 份，"
            f"未生成向量 {len(health['missing_embeddings'])} 份。"
        )
    else:
        st.success("RAG 索引状态正常。")
    for doc in health["documents"]:
        models = ", ".join(doc["embedding_models"]) if doc["embedding_models"] else "无"
        with st.container(border=True):
            st.markdown(f"**{doc['title']}**")
            st.caption(
                f"{doc.get('folder', '默认')} · {doc['source'] or '未注明来源'} · {doc['size']} 字 · "
                f"{doc['chunk_count']} chunks · {doc['embedded_count']} embeddings · "
                f"模型：{models}"
            )
