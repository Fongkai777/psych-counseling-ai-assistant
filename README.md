# 心理类内容 AI 运营助手

一个本机运行的 AI 内容运营工作台，用来把「知乎选题池、私有知识库、意图识别、RAG 检索、回答生成、人工修改、风格反馈」串成完整流程。

项目定位不是简单套壳聊天机器人，而是面向内容运营场景的 AI 辅助决策工具：先理解题目，再检索可信依据，最后生成可人工编辑的回答初稿。

## 功能概览

- 选题池：从粘贴的知乎页面 HTML 中解析问题标题，维护待回答选题。
- RAG 维护：上传 PDF、docx、doc、txt、md 文件，按文件夹管理知识库。
- 文档解析：优先提取原文文本，识别扫描版 PDF 和低质量文本层。
- 意图识别：先分析问题真正想问什么、适合的回答角度和检索关键词。
- 检索依据：基于意图识别结果检索知识库片段，支持 embedding 检索和本地词面检索回退。
- 生成初稿：调用 OpenAI 兼容 Chat Completions API 生成中文回答。
- 人工反馈：可以对意图识别或 AI 初稿分别提意见，并从对应步骤回退重跑。
- 风格沉淀：保存人工终稿后，自动记录编辑反馈，并把终稿加入“我的旧回答”知识库。
- 状态面板：查看数据库、文档、chunk、embedding 和模型配置状态。

## 技术栈

- Python
- Streamlit
- SQLite
- OpenAI-compatible Chat Completions API
- OpenAI-compatible Embeddings API
- pdfplumber / pypdf / python-docx

## 启动

建议使用 Python 3.9+。

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
.venv/bin/streamlit run streamlit_app.py
```

打开：

```text
http://127.0.0.1:8501
```

如果没有配置 API key，应用会进入本地演示/回退模式，方便先查看流程。没有配置 embedding key 时，RAG 会退回本地词面检索。

## 配置模型

复制 `.env.example` 到 `.env` 后填写：

```bash
LLM_API_KEY=
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=

EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=
```

只要服务兼容 `/chat/completions` 和 `/embeddings`，也可以换成其他模型平台。

## 使用流程

1. 在「选题池」粘贴知乎问题页面 HTML，加入候选选题。
2. 在「RAG 维护」上传本地资料，按文件夹整理知识库。
3. 在「回答工作台」选择问题。
4. 点击「运行流程 / 按意见重跑」：
   - 意图识别
   - 检索依据
   - 生成 AI 初稿
5. 如果检索方向不对，在「对意图识别提意见」里说明，并回退到「意图识别」重跑。
6. 如果回答写法不对，在「对 AI 初稿提意见」里说明，并从「生成 AI 初稿」重跑。
7. 人工编辑终稿后保存，系统会记录反馈并自动加入知识库。

## 隐私与安全

默认不会提交以下运行时数据：

- `.env`
- `.venv/`
- `.idea/`
- `data/`
- `.streamlit/secrets.toml`

`data/` 中通常包含 SQLite 数据库、知识库内容、个人语气文档和人工终稿，不建议公开上传。

如果准备把项目发布到 GitHub，请只提交代码、配置样例和说明文档，不要提交真实 API key、数据库、个人文档或知乎回答原文。

## 项目亮点

这个项目适合在简历中描述为：

> 基于 RAG、意图识别和人工反馈闭环的心理类内容运营 AI 工作台。系统支持私有知识库维护、问题意图分析、检索依据追踪、回答初稿生成和人工终稿沉淀，用于探索 AI 在内容运营决策与个性化写作中的应用。

## 目录结构

```text
.
├── streamlit_app.py        # Streamlit 主应用
├── app.py                  # 备用本地 Web 服务入口
├── psych_ai_assistant/     # 核心业务逻辑
├── scripts/                # 文档重建、语气画像等脚本
├── web/                    # 备用前端页面
├── data/                   # 运行时数据库和知识库，默认不提交
├── requirements.txt
└── .env.example
```
