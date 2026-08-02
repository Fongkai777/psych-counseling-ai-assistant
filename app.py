from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import json
import os
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from psych_ai_assistant.config import load_config
from psych_ai_assistant.db import Database
from psych_ai_assistant.feedback import analyze_edit
from psych_ai_assistant.llm import LLMClient
from psych_ai_assistant.prompts import build_answer_prompt, default_persona
from psych_ai_assistant.retrieval import Retriever


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
DB_PATH = ROOT / "data" / "assistant.sqlite3"

config = load_config(ROOT / ".env")
db = Database(DB_PATH)
retriever = Retriever(db)
llm = LLMClient(config)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def log_message(self, fmt, *args):
        print("[server]", fmt % args)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self.handle_api_get(parsed)
            return
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            self.send_error(404)
            return
        body = self.read_json_body()
        self.handle_api_post(parsed, body)

    def read_json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def send_json(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def handle_api_get(self, parsed):
        query = parse_qs(parsed.query)
        if parsed.path == "/api/bootstrap":
            self.send_json(
                {
                    "questions": db.list_questions(),
                    "documents": db.list_documents(),
                    "persona": db.get_setting("persona", default_persona()),
                    "feedback": db.list_feedback(limit=8),
                    "llm": llm.status(),
                }
            )
        elif parsed.path == "/api/search":
            text = query.get("q", [""])[0]
            self.send_json({"results": retriever.search(text, limit=5)})
        else:
            self.send_error(404)

    def handle_api_post(self, parsed, body):
        if parsed.path == "/api/questions":
            item = db.add_question(
                title=body.get("title", "").strip(),
                source_url=body.get("source_url", "").strip(),
                description=body.get("description", "").strip(),
                tags=body.get("tags", "").strip(),
                heat=int(body.get("heat") or 0),
            )
            self.send_json({"question": item})
        elif parsed.path == "/api/documents":
            item = db.add_document(
                title=body.get("title", "").strip(),
                source=body.get("source", "").strip(),
                content=body.get("content", "").strip(),
            )
            self.send_json({"document": item})
        elif parsed.path == "/api/persona":
            persona = body.get("persona", "").strip()
            db.set_setting("persona", persona or default_persona())
            self.send_json({"persona": db.get_setting("persona", default_persona())})
        elif parsed.path == "/api/generate":
            question_id = int(body["question_id"])
            question = db.get_question(question_id)
            if not question:
                self.send_json({"error": "Question not found"}, status=404)
                return
            persona = db.get_setting("persona", default_persona())
            context = retriever.search(question["title"] + "\n" + question["description"], limit=5)
            prompt = build_answer_prompt(question, context, persona)
            draft = llm.generate(prompt)
            answer = db.save_answer(question_id, draft=draft, context=context)
            db.update_question_status(question_id, "drafted")
            self.send_json({"answer": answer, "context": context, "questions": db.list_questions()})
        elif parsed.path == "/api/answers/edit":
            answer_id = int(body["answer_id"])
            edited = body.get("edited", "").strip()
            answer = db.get_answer(answer_id)
            if not answer:
                self.send_json({"error": "Answer not found"}, status=404)
                return
            feedback = analyze_edit(answer["draft"], edited)
            updated = db.save_edited_answer(answer_id, edited, feedback)
            db.update_question_status(updated["question_id"], "edited")
            self.send_json(
                {
                    "answer": updated,
                    "feedback": feedback,
                    "feedback_items": db.list_feedback(limit=8),
                    "questions": db.list_questions(),
                }
            )
        else:
            self.send_error(404)


def seed_if_empty():
    if db.list_questions():
        return
    sample_knowledge = ROOT / "samples" / "knowledge" / "心理回答边界示例.md"
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
    if sample_knowledge.exists():
        db.add_document(
            title="心理回答边界示例",
            source="sample",
            content=sample_knowledge.read_text(encoding="utf-8"),
        )
    db.set_setting("persona", default_persona())


if __name__ == "__main__":
    seed_if_empty()
    host = os.environ.get("APP_HOST") or config.get("APP_HOST") or "127.0.0.1"
    port = int(os.environ.get("APP_PORT") or config.get("APP_PORT") or 8000)
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"AI content assistant running at http://{host}:{port}")
    server.serve_forever()
