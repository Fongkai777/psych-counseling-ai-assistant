import html
import json
import re

import requests
from bs4 import BeautifulSoup


QUESTION_RE = re.compile(r"zhihu\.com/question/(\d+)")
ANSWER_RE = re.compile(r"zhihu\.com/question/\d+/answer/(\d+)")


def import_question_from_url(url):
    url = url.strip()
    if not url:
        raise ValueError("请先填写知乎问题链接。")

    question_id = extract_question_id(url)
    answer_id = extract_answer_id(url)
    title = ""
    description = ""
    parse_note = ""

    if answer_id:
        try:
            title, description = fetch_answer_api(answer_id)
        except requests.RequestException as exc:
            parse_note = parse_error_note(exc)
    if is_bad_title(title) and question_id:
        try:
            title, description = fetch_question_api(question_id)
        except requests.RequestException as exc:
            parse_note = parse_error_note(exc)
    if is_bad_title(title):
        try:
            title, description = fetch_page_metadata(url, question_id)
        except requests.RequestException as exc:
            parse_note = parse_error_note(exc)

    if is_bad_title(title):
        title = f"待补标题：知乎问题 {question_id}" if question_id else url

    return {
        "title": clean_title(title),
        "source_url": url,
        "description": description or parse_note,
        "tags": "知乎",
        "heat": 50,
    }


def import_question_from_url_and_hint(url, title_hint=""):
    payload = import_question_from_url(url)
    hint = metadata_from_hint(title_hint)
    if hint.get("title"):
        payload["title"] = hint["title"]
        if payload["description"].startswith("自动解析失败"):
            payload["description"] = ""
    if hint.get("description"):
        payload["description"] = hint["description"]
    if hint.get("tags"):
        payload["tags"] = hint["tags"]
    return payload


def import_question_from_html(html_text):
    metadata = metadata_from_hint(html_text)
    title = metadata.get("title") or "未命名知乎选题"
    return {
        "title": title,
        "source_url": metadata.get("url", ""),
        "description": metadata.get("description", ""),
        "tags": metadata.get("tags", "知乎"),
        "heat": 50,
    }


def title_from_hint(text):
    return metadata_from_hint(text).get("title", "")


def metadata_from_hint(text):
    text = (text or "").strip()
    if not text:
        return {}
    soup = BeautifulSoup(text, "html.parser")
    title = (
        aria_tip_title(soup)
        or question_header_title(soup)
        or meta_content(soup, "og:title")
        or html_title(soup)
    )
    description = (
        meta_content(soup, "og:description")
        or meta_name(soup, "description")
        or meta_content(soup, "description")
    )
    tags = meta_name(soup, "keywords")
    url = (
        meta_content(soup, "og:url")
        or canonical_url(soup)
        or regex_url(text)
    )

    if title:
        title = clean_title(title)
    if "<" in text and ">" in text:
        if not title:
            text = soup.get_text(" ", strip=True)
    elif not title:
        title = clean_title(text)

    return {
        "title": title,
        "description": strip_html(description),
        "tags": clean_tags(tags),
        "url": url.strip(),
    }


def html_title(soup):
    if not soup.title or not soup.title.string:
        return ""
    return soup.title.string


def canonical_url(soup):
    tag = soup.find("link", rel=lambda value: value and "canonical" in value)
    return tag.get("href", "").strip() if tag else ""


def regex_url(text):
    match = re.search(r"https://www\.zhihu\.com/question/\d+(?:/answer/\d+)?", text)
    return match.group(0) if match else ""


def extract_question_id(url):
    match = QUESTION_RE.search(url)
    return match.group(1) if match else ""


def extract_answer_id(url):
    match = ANSWER_RE.search(url)
    return match.group(1) if match else ""


def fetch_answer_api(answer_id):
    response = requests.get(
        f"https://www.zhihu.com/api/v4/answers/{answer_id}",
        params={"include": "question,excerpt"},
        headers=request_headers(),
        timeout=12,
    )
    response.raise_for_status()
    data = response.json()
    question = data.get("question") or {}
    title = question.get("title") or question.get("name") or data.get("question_title") or ""
    description = question.get("detail") or data.get("excerpt") or ""
    return title, strip_html(description)


def fetch_question_api(question_id):
    response = requests.get(
        f"https://www.zhihu.com/api/v4/questions/{question_id}",
        params={"include": "detail"},
        headers=request_headers(),
        timeout=12,
    )
    response.raise_for_status()
    data = response.json()
    title = data.get("title") or data.get("name") or ""
    description = data.get("detail") or data.get("excerpt") or ""
    return title, strip_html(description)


def fetch_page_metadata(url, question_id=""):
    response = requests.get(
        url,
        headers=request_headers(),
        timeout=12,
    )
    html_text = response.text
    soup = BeautifulSoup(response.text, "html.parser")
    title = (
        question_header_title(soup)
        or aria_tip_title(soup)
        or initial_state_title(soup, question_id)
        or json_ld_title(soup)
        or meta_content(soup, "og:title")
        or regex_title(html_text)
        or (soup.title.string if soup.title else "")
    )
    description = meta_content(soup, "og:description") or meta_name(soup, "description")
    if is_bad_title(title):
        response.raise_for_status()
    return title or "", description or ""


def parse_error_note(exc):
    if isinstance(exc, requests.HTTPError):
        status = exc.response.status_code if exc.response is not None else ""
        if status == 403:
            return "自动解析失败：知乎返回 403 安全验证页，需要手动补充中文标题。"
        return f"自动解析失败：知乎返回 HTTP {status}。"
    return f"自动解析失败：{exc.__class__.__name__}。"


def request_headers():
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.7",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
        "Referer": "https://www.zhihu.com/",
    }


def question_header_title(soup):
    tag = soup.find(["h1", "div"], class_=re.compile("QuestionHeader-title"))
    return tag.get_text(" ", strip=True) if tag else ""


def aria_tip_title(soup):
    tag = soup.find(id="ariaTipText")
    if not tag:
        return ""
    label = tag.get("aria-label", "")
    if not label:
        return ""
    label = html.unescape(label)
    label = re.sub(r"^欢迎进入\s*", "", label)
    label = re.split(r"\s*-\s*知乎[，,]", label, maxsplit=1)[0]
    label = re.split(r"盲人用户使用操作智能引导", label, maxsplit=1)[0]
    return label.strip(" ，,。")


def json_ld_title(soup):
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "{}")
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            for key in ("name", "headline", "title"):
                if data.get(key):
                    return str(data[key])
    return ""


def initial_state_title(soup, question_id):
    candidates = []
    for tag in soup.find_all("script"):
        text = tag.string or tag.get_text() or ""
        if "initialState" in text or "entities" in text or "QuestionHeader-title" in text:
            candidates.append(text)

    for text in candidates:
        parsed = parse_jsonish_script(text)
        title = find_question_title(parsed, question_id)
        if title:
            return title
    return ""


def parse_jsonish_script(text):
    text = html.unescape(text.strip())
    if "=" in text and not text.startswith("{"):
        text = text.split("=", 1)[1].strip().rstrip(";")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def find_question_title(value, question_id):
    if isinstance(value, dict):
        if question_id and question_id in value:
            title = find_question_title(value[question_id], "")
            if title:
                return title
        if value.get("title") and not is_bad_title(str(value["title"])):
            return str(value["title"])
        if value.get("name") and not is_bad_title(str(value["name"])):
            return str(value["name"])
        for child in value.values():
            title = find_question_title(child, question_id)
            if title:
                return title
    elif isinstance(value, list):
        for child in value:
            title = find_question_title(child, question_id)
            if title:
                return title
    return ""


def regex_title(html_text):
    patterns = [
        r'"QuestionHeader-title"[^>]*>(.*?)</',
        r'"title"\s*:\s*"([^"]{6,120})"',
        r'"name"\s*:\s*"([^"]{6,120})"',
    ]
    for pattern in patterns:
        match = re.search(pattern, html_text)
        if match:
            raw = re.sub(r"<.*?>", "", match.group(1))
            return html.unescape(raw.encode("utf-8").decode("unicode_escape", errors="ignore"))
    return ""


def meta_content(soup, prop):
    tag = soup.find("meta", property=prop)
    return tag.get("content", "").strip() if tag else ""


def meta_name(soup, name):
    tag = soup.find("meta", attrs={"name": name})
    return tag.get("content", "").strip() if tag else ""


def clean_title(title):
    title = strip_html(title)
    title = re.sub(r"^\([^)]*(私信|消息)[^)]*\)\s*", "", title)
    title = re.sub(r"\s*-\s*知乎.*$", "", title.strip())
    title = re.sub(r"\s+", " ", title)
    return title


def clean_tags(tags):
    tags = strip_html(tags)
    tags = re.sub(r"\s+", "", tags)
    return tags


def strip_html(text):
    text = re.sub(r"<[^>]+>", "", text or "")
    return html.unescape(text).strip()


def is_bad_title(title):
    if not title:
        return True
    bad_parts = ("安全验证", "知乎 - 有问题", "知乎，中文互联网", "403", "验证码")
    return any(part in title for part in bad_parts)
