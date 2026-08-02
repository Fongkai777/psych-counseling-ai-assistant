from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from psych_ai_assistant.config import load_config
from psych_ai_assistant.db import Database
from psych_ai_assistant.document_loader import extract_pdf
from psych_ai_assistant.llm import LLMClient
from psych_ai_assistant.retrieval import build_chunk_items


def index_pdfs(source_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in sorted(source_dir.rglob("*.pdf")):
        paths.setdefault(path.name, path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "assistant.sqlite3")
    args = parser.parse_args()

    db = Database(args.db)
    llm = LLMClient(load_config(ROOT / ".env"))
    pdf_paths = index_pdfs(args.source_dir)

    docs = [doc for doc in db.list_documents() if doc["title"].lower().endswith(".pdf")]
    for doc in docs:
        path = pdf_paths.get(doc["title"])
        if not path:
            print(f"MISS\t{doc['id']}\t{doc['title']}\told={doc['size']}")
            continue

        data = path.read_bytes()
        content = extract_pdf(data)
        if not content.strip():
            print(f"EMPTY\t{doc['id']}\t{doc['title']}\tpath={path}")
            continue

        db.update_document_content(doc["id"], content)
        chunks = build_chunk_items(content, llm)
        db.replace_document_chunks(doc["id"], chunks)
        embedded = sum(1 for item in chunks if item.get("embedding") is not None)
        print(
            "OK\t"
            f"{doc['id']}\t{doc['title']}\told={doc['size']}\t"
            f"new={len(content)}\tchunks={len(chunks)}\tembedded={embedded}\tpath={path}"
        )


if __name__ == "__main__":
    main()
