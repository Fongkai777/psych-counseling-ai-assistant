import json


def json_dumps(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


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
    expression_snippets=None,
):
    context_text = "\n\n".join(
        f"[{idx + 1}] 来源：{item['title']}，相关度：{item['score']}\n{item['content']}"
        for idx, item in enumerate(context)
    )
    if not context_text:
        context_text = "暂无匹配知识片段。请明确说明回答主要基于一般性理解，避免编造来源。"
    expression_text = format_expression_snippets(expression_snippets or [])

    return f"""请为一个知乎心理类问题生成回答初稿。

【优先级】
1. 最高优先级：意图识别和人工意见、长期风格记忆。
2. 其次：通用回答 Prompt。
3. 最后：个人语气文档和可引用知识片段。
如果不同部分互相冲突，必须优先服从人工意见和长期风格记忆。

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

【本题匹配到的个人表达片段】
{expression_text}

【可引用知识片段】
{context_text}

【反矫情写作约束】
如果人工意见或长期风格记忆中出现“太矫情、太空洞、太 AI、太抒情、不要编例子、不要瞎编”等意思，必须执行：
- 不使用诗化比喻、感官化抒情、排比式设问、强行共情收尾。
- 不用“轻轻一沉、像什么落进胸口、你有没有过这种时候”这类句式。
- 优先使用清楚的判断、概念解释、场景对比和具体行动。
- 可以有温度，但表达要平实、克制、像一个人在认真说明问题。
- 不编造研究者、年份、数据或个人经历；除非可引用知识片段中明确提供。
- 个人表达片段只用于语气、意象和可复用表达。只有复用方式允许“直接引用”且与本题自然贴合时，才可以原句使用；否则只吸收表达方式，不要硬塞。

【任务】
请根据上述材料输出一版完整回答初稿。不要解释你遵守了哪些规则，直接输出正文。
"""


def build_expression_emotion_prompt(question, intent=""):
    return f"""请为下面这个知乎问题抽取“情绪结构摘要”，用于匹配个人表达片段。

不要回答问题，只分析这个问题背后的情绪处境。

【问题标题】
{question['title']}

【问题描述】
{question.get('description') or '无'}

【已有意图识别】
{intent or '无'}

请输出 120-220 字中文，覆盖：
1. 读者可能处在什么情绪里。
2. 这个问题背后的核心冲突是什么。
3. 适合调用什么类型的个人表达。

只输出摘要正文，不要输出编号。
"""


def build_quote_picker_prompt(question, intent, emotion_summary, candidates, final_limit=3):
    payload = [
        {
            "id": item.get("id"),
            "title": item.get("title"),
            "themes": item.get("themes"),
            "usage": item.get("usage"),
            "reuse_mode": item.get("reuse_mode"),
            "score": item.get("score"),
            "text": item.get("text"),
        }
        for item in candidates
    ]
    return f"""请从候选个人表达片段中，为当前知乎回答选择最多 {final_limit} 条真正适合使用的片段。

判断标准：
1. 情绪结构是否贴合当前问题。
2. 是否能让回答更像一个真实的人在思考，而不是硬塞漂亮句子。
3. 是否适合直接引用；不适合直接引用时，也可以作为“改写/吸收语气”的素材。
4. 如果都不合适，可以返回空数组。

【问题标题】
{question['title']}

【问题描述】
{question.get('description') or '无'}

【意图识别】
{intent or '无'}

【本题情绪结构摘要】
{emotion_summary or '无'}

【候选表达片段】
{json_dumps(payload)}

请只输出 JSON，不要输出 Markdown，不要解释 JSON 之外的内容。

格式：
{{"selected_ids": ["001", "007"], "reason": "简短说明为什么选这些；如果没选，说明原因"}}
"""


def format_expression_snippets(snippets):
    if not snippets:
        return "无。不要为了模仿风格而硬编私人经历或硬塞金句。"
    lines = []
    for index, item in enumerate(snippets, start=1):
        lines.append(
            "\n".join(
                part
                for part in [
                    f"[{index}] {item.get('title', '')}",
                    f"主题：{item.get('themes', '')}",
                    f"适用场景：{item.get('usage', '')}",
                    f"复用方式：{item.get('reuse_mode', '')}",
                    f"片段：{item.get('text', '')}",
                ]
                if part.strip()
            )
        )
    return "\n\n".join(lines)


def build_revision_feedback_memory_prompt(question, instruction, current_draft=""):
    return f"""请把用户这次对 AI 回答的评价，提炼成以后可长期复用的回答 Prompt 规则。

【问题标题】
{question['title']}

【用户本次评价】
{instruction or '无'}

【被评价的当前稿件】
{current_draft or '无'}

【任务】
请输出 1-3 条可以长期加入 Prompt 的写作规则，用于以后生成同类知乎心理类回答。

要求：
1. 保留用户评价里的真实偏好，不要过度扩写。
2. 把具体问题里的局部意见，抽象成可复用规则。
3. 不记录私人经历、人名、地点、账号或联系方式。
4. 不要输出解释，只输出规则。

格式：

【长期回答 Prompt】
- ...
"""


def build_document_summary_prompt(title, content):
    return f"""请为下面这份知识库资料生成一段用于 RAG 路由检索的中文摘要。

这段摘要不是最终回答，它的用途是帮助系统判断“用户问题应该优先检索哪份资料”。

要求：
1. 保留资料的核心主题、关键概念、适用问题、重要人物/书名/理论名。
2. 如果资料较长，请按主题概括，不要逐页复述。
3. 不要新增原文没有的观点、案例、数据。
4. 输出 300-800 字中文摘要；如果原文很短，可以更短。
5. 只输出摘要正文，不要解释你的处理过程。

【资料标题】
{title}

【资料正文】
{content}
"""


def build_document_section_summary_prompt(title, section_index, section_total, content):
    return f"""请为下面这份长文档的一个片段生成局部摘要，用于后续合成全文 RAG 路由摘要。

要求：
1. 只概括这个片段里真实出现的主题、概念、人物、理论、案例。
2. 不要补充片段外的知识。
3. 输出 120-220 字中文摘要。
4. 只输出摘要正文。

【资料标题】
{title}

【片段位置】
第 {section_index} / {section_total} 段

【片段正文】
{content}
"""


def build_document_summary_merge_prompt(title, section_summaries):
    return f"""请把下面这些局部摘要合并成一段用于 RAG 路由检索的中文文档摘要。

这段摘要的用途是帮助系统判断“用户问题应该优先检索哪份资料”，不是最终回答依据。

要求：
1. 覆盖不同片段中反复出现或重要的主题。
2. 保留关键概念、适用问题、重要人物/书名/理论名。
3. 不要新增局部摘要里没有的信息。
4. 输出 300-900 字中文摘要。
5. 只输出摘要正文，不要解释过程。

【资料标题】
{title}

【局部摘要】
{section_summaries}
"""


def build_rag_triad_eval_prompt(question, answer, context):
    context_text = format_context(context)
    return f"""请作为 RAG 评估器，评估下面这次回答。

请使用 RAG Triad 三个指标：
1. answer_relevance：回答是否切题，是否真正回应用户问题。
2. context_relevance：检索片段是否和用户问题相关。
3. groundedness：回答中的关键判断是否被检索片段支持。

评分范围为 0 到 1，1 表示很好，0 表示很差。

请同时给出 next_actions。它不是泛泛建议，而是下一步工作流动作：
- answer_relevance 低：建议回到意图识别，重新明确问题真正要回答什么。
- context_relevance 低：建议补充检索 query、扩大候选文档数、切换检索方式，重新检索。
- groundedness 低：建议重写回答，减少未被检索支持的判断；把无法从片段推出的内容标注为一般建议；增加回答观点与检索片段的对应关系。

【问题标题】
{question['title']}

【问题描述】
{question.get('description') or '无'}

【回答】
{answer or '无'}

【检索片段】
{context_text}

请只输出 JSON，不要输出 Markdown，不要解释 JSON 之外的内容。

格式：
{{
  "answer_relevance": {{"score": 0.0, "reason": "..."}},
  "context_relevance": {{"score": 0.0, "reason": "..."}},
  "groundedness": {{"score": 0.0, "reason": "..."}},
  "suggestion": "综合改进建议",
  "next_actions": [
    {{"type": "revise_answer", "priority": "high", "reason": "...", "instruction": "可直接复制到重写意见里的具体要求"}},
    {{"type": "improve_retrieval", "priority": "medium", "reason": "...", "instruction": "下一轮检索应该补充的关键词或方向"}},
    {{"type": "rerun_intent", "priority": "low", "reason": "...", "instruction": "意图识别需要重新明确的点"}}
  ]
}}
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
