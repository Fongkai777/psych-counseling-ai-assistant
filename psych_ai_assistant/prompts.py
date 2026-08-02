def default_guardrails():
    return """你是一个心理类内容创作者的 AI 写作助手，不扮演治疗师，不做诊断。

表达风格：
- 先承认对方处境，不急着说教。
- 多用“也许、可能、我会先把它理解为”这类留有余地的表达。
- 结构清楚，但不要像教科书。
- 给出 2-4 个可执行的小步骤。
- 避免夸大承诺，例如“你一定会好起来”。

安全边界：
- 不诊断疾病，不替代心理咨询或医疗建议。
- 遇到自伤、伤人、严重危机时，建议尽快联系身边可信的人和专业机构。
- 对知乎回答要适合公开发布，避免暴露隐私和攻击性表达。
"""


def default_persona():
    return default_guardrails()


def build_intent_prompt(question, feedback=""):
    return f"""请先分析这个知乎问题的回答意图，为后续 RAG 检索和回答生成做准备。

【问题标题】
{question['title']}

【问题描述】
{question.get('description') or '无'}

【人工意见】
{feedback or '无'}

请输出：
1. 这个问题真正想问什么。
2. 读者可能处在什么情绪或关系处境里。
3. 回答时应该采用什么角度。
4. 适合检索知识库的关键词或概念。
5. 哪些内容不要写，避免空洞、编造经历或越界。

要求：简洁，但要足够指导检索和写作。
"""


def build_answer_prompt(
    question,
    context,
    guidance="",
    global_prompt="",
    style_profile="",
    style_memories="",
):
    context_text = "\n\n".join(
        f"[{idx + 1}] 来源：{item['title']}，相关度：{item['score']}\n{item['content']}"
        for idx, item in enumerate(context)
    )
    if not context_text:
        context_text = "暂无匹配知识片段。请明确说明回答主要基于一般性理解，避免编造来源。"

    return f"""请为一个知乎心理类问题生成回答初稿。

【问题标题】
{question['title']}

【问题描述】
{question.get('description') or '无'}

【标签】
{question.get('tags') or '无'}

【意图识别和人工意见，可为空】
{guidance or question.get('guidance') or '无'}

【通用回答 Prompt】
{global_prompt or '无'}

【个人语气文档】
{style_profile or '无'}

【长期风格记忆】
{style_memories or '无'}

【可引用知识片段】
{context_text}

【任务】
请根据通用回答 Prompt、个人语气文档、长期风格记忆、意图识别和人工意见，以及可引用知识片段，输出一版完整回答初稿。
"""


def build_revision_prompt(
    question,
    context,
    draft,
    instruction,
    guidance="",
    global_prompt="",
    style_profile="",
    style_memories="",
):
    return f"""请根据人工意见重写一版知乎回答。

【问题】
{question['title']}

【本题引导词】
{guidance or question.get('guidance') or '无'}

【通用回答 Prompt】
{global_prompt or '无'}

【个人语气文档】
{style_profile or '无'}

【长期风格记忆】
{style_memories or '无'}

【当前初稿】
{draft}

【人工意见】
{instruction}

【检索依据】
{format_context(context)}

【要求】
请优先服从人工意见，并结合通用回答 Prompt、个人语气文档、长期风格记忆、本题引导词和检索依据，输出一版完整重写稿。
"""


def build_editor_suggestions_prompt(question, context, answer):
    return f"""请像一个中文内容编辑一样，给下面这篇知乎回答提供可执行的人工修改建议。

【问题】
{question['title']}

【回答】
{answer}

【检索依据】
{format_context(context)}

请输出：
1. 全文结构建议：哪里应该提前、删掉、拆开或补充。
2. 逐段修改建议：指出原文片段，并说明怎么改。
3. 语气与风格建议：哪里太像 AI、太说教、太硬。
4. 可直接替换的表达：给出 3-6 组“原句 -> 建议改写”。
5. 保留意见：哪些句子可以保留，不必过度修改。

不要输出发布风险打分，不要做审稿式评价，重点服务人工编辑。
"""


def format_context(context):
    if not context:
        return "暂无检索依据。"
    return "\n\n".join(
        f"[{idx + 1}] {item['title']}，相关度 {item['score']}\n{item['content']}"
        for idx, item in enumerate(context)
    )
