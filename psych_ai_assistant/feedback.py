import difflib
import re


def analyze_edit(draft, edited):
    draft_len = len(draft)
    edited_len = len(edited)
    ratio = difflib.SequenceMatcher(None, draft, edited).ratio()
    changed = 1 - ratio

    additions = extract_added_sentences(draft, edited)
    style_notes = []
    if edited_len < draft_len * 0.85:
        style_notes.append("人工终稿明显更克制，后续生成应减少铺陈和重复。")
    if edited_len > draft_len * 1.15:
        style_notes.append("人工终稿补充了更多解释，后续生成可以增加背景推理和例子。")
    if "也许" in edited or "可能" in edited:
        style_notes.append("人工终稿保留不确定性表达，后续应避免绝对判断。")
    if "建议" in edited and "专业" in edited:
        style_notes.append("人工终稿强调专业求助边界，后续心理类回答需保留安全提示。")
    if not style_notes:
        style_notes.append("人工修改幅度较小，可把终稿作为相近风格样本。")

    return {
        "change_ratio": round(changed, 3),
        "draft_length": draft_len,
        "edited_length": edited_len,
        "added_sentences": additions[:5],
        "style_notes": style_notes,
    }


def extract_added_sentences(draft, edited):
    draft_parts = set(split_sentences(draft))
    added = []
    for sentence in split_sentences(edited):
        if sentence and sentence not in draft_parts and len(sentence) >= 8:
            added.append(sentence)
    return added


def split_sentences(text):
    return [part.strip() for part in re.split(r"[。！？!?]\s*", text) if part.strip()]

