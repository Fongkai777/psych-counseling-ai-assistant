# 心理内容 AI 运营助手：简历项目经历

> 适用方向：AI 产品运营、发行运营 AI 方向、AIGC 应用实习、LLM/RAG 应用开发实习、内容平台策略实习。

## 中文简历版

**心理内容 AI 运营助手 | 个人项目 | Python / Streamlit / SQLite / RAG / DashScope Qwen**

- 设计并实现一套面向知乎心理类内容运营的本地 AI 工作台，将选题池、私有知识库、意图识别、RAG 检索、回答生成、人工编辑和风格反馈沉淀串成闭环流程，避免停留在单纯 ChatBot 套壳。
- 搭建私有知识库管理模块，支持 PDF、docx、txt、md 等资料上传、文件夹分类、文本抽取、chunk 分块、向量化入库和索引重建；当前本地测试库约 12 份资料、260 万字、1678 个 chunks。
- 实现多阶段 RAG 检索流程：先通过 Summary Index 进行文档路由，再结合向量相似度、关键词匹配、Sentence Window 和 Auto Merging 层级索引召回证据片段，并支持 Top K、相关度阈值、Embedding 阈值、关键词阈值等参数调节。
- 接入阿里云百炼 / DashScope 模型服务，使用 Qwen 生成回答与评估反馈，使用 text-embedding-v4 生成向量；封装 API 调用、错误日志、batch size 限制、超时重试与本地回退机制。
- 设计“意图识别 → 检索依据 → 生成初稿 → RAG Triad 评价 → 人工终稿沉淀”的工作流，允许用户对意图、检索和回答分别提意见并从对应步骤重跑，提高内容生成的可控性。
- 实现 RAG Triad 评价模块，从回答切题性、资料相关性、回答有据性三个维度评估稿件，并将低分项自动转化为下一步动作建议，例如 groundedness 低时要求减少未引用判断、标注一般建议并增加证据对应关系。
- 建立人工反馈沉淀机制，将用户对 AI 初稿的修改意见提炼为长期写作偏好 Prompt，并将人工终稿自动加入“我的旧回答”知识库，用于后续模仿个人表达风格。
- 使用 SQLite 管理问题、文档、chunk、摘要索引、句子窗口、层级节点、生成记录、人工反馈和风格记忆，并在系统状态页展示数据库占用、索引完整性、embedding 模型和异常日志。

## 精简版

**心理内容 AI 运营助手 | Python / Streamlit / SQLite / RAG / Qwen**

- 开发本地 AI 内容运营工作台，支持知乎选题解析、私有知识库维护、意图识别、RAG 检索、回答生成、人工编辑和风格反馈沉淀。
- 实现 Summary Index、Vector Index、Sentence Window、Auto Merging 等多路检索策略，并支持检索阈值、Top K、重排和索引重建参数调节。
- 接入 DashScope Qwen 与 text-embedding-v4，封装模型调用、embedding batch、失败重试、日志记录和本地回退。
- 构建 RAG Triad 评价流程，从切题性、资料相关性和 groundedness 评估回答，并将评价结果转化为可执行的重写/重检索建议。

## 英文简历版

**AI Content Operations Assistant | Personal Project | Python, Streamlit, SQLite, RAG, DashScope Qwen**

- Built a local AI-assisted content operations workspace for psychology-related Zhihu topics, covering topic intake, private knowledge base management, intent analysis, RAG retrieval, draft generation, human editing, and feedback-based style refinement.
- Implemented a multi-stage RAG pipeline with Summary Index routing, vector retrieval, keyword matching, Sentence Window retrieval, and Auto Merging hierarchical retrieval; exposed configurable Top K, similarity thresholds, keyword thresholds, and reranking options.
- Integrated DashScope Qwen for answer generation and evaluation, and text-embedding-v4 for semantic retrieval; added API abstraction, embedding batch control, timeout handling, error logging, and local fallback.
- Designed a controllable workflow of intent analysis, evidence retrieval, draft generation, RAG Triad evaluation, and final-answer memory, enabling users to revise intent, retrieval direction, or answer style from the corresponding step.
- Added a RAG Triad evaluation module to assess answer relevance, context relevance, and groundedness, converting weak scores into concrete next-step suggestions such as reducing unsupported claims or improving retrieval queries.

## 面试讲法

这个项目不是一个简单的“把大模型接到前端”的应用。我想解决的是内容运营里更真实的问题：运营人员不是只需要一段回答，而是需要一个可控的工作流。它要先判断问题到底在问什么，再从自己的资料库里找证据，然后生成初稿，最后让人工评价能够反过来影响下一轮生成。

我把流程拆成了五步：意图识别、检索依据、生成回答、RAG Triad 评价、人工终稿沉淀。这样做的好处是，每一步都可以被人工干预。比如检索片段不对，就不是盲目重写回答，而是回到检索步骤补充关键词；如果回答太空洞，就把评价沉淀成长期 Prompt，下次生成时自动带上。

技术上，我做了几层 RAG：Summary Index 先筛文档，Vector Index 做基础语义召回，Sentence Window 保留局部上下文，Auto Merging 在多个相邻 chunk 命中时返回更完整的上层片段。最后再用 RAG Triad 检查回答是否切题、资料是否相关、判断是否被资料支持。

这个项目对我来说最重要的不是模型本身，而是把业务需求拆成可执行 AI 工作流的能力：哪些步骤适合 LLM，哪些步骤应该保留人工控制，哪些反馈应该沉淀成长期记忆，哪些结果需要可解释和可回退。

## 可放在作品集 README 的一句话

基于 RAG、意图识别和人工反馈闭环的心理类内容运营 AI 工作台，用于探索大模型在内容选题、知识库检索、回答生成、证据校验和个性化风格学习中的应用。

## 不建议夸大的点

- 不要写“心理咨询 AI”或“提供心理诊断”，更建议写“心理类内容运营助手”。
- 不要写“微调大模型”，目前主要是 RAG、Prompt、反馈记忆和工作流编排，不是模型参数微调。
- 不要写“上线服务大量用户使用”，目前是本地个人项目。
- 可以写“本地测试库约 260 万字”，但建议面试时说明数据来自个人整理资料和测试样本。

## 更适合投递 AI 运营岗的标题

- AI 内容运营工作台
- RAG-based Content Operations Assistant
- 私有知识库驱动的知乎回答生成与评估系统
- 面向心理类内容创作的 AI 辅助决策工具

