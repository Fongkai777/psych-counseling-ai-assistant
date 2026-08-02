import json
import sqlite3
from pathlib import Path


class Database:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.init_schema()

    def connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self):
        with self.connect() as conn:
            conn.executescript(
                """
                create table if not exists questions (
                    id integer primary key autoincrement,
                    title text not null,
                    source_url text,
                    description text,
                    tags text,
                    heat integer default 0,
                    status text default 'new',
                    created_at datetime default current_timestamp
                );

                create table if not exists documents (
                    id integer primary key autoincrement,
                    title text not null,
                    source text,
                    content text not null,
                    created_at datetime default current_timestamp
                );

                create table if not exists knowledge_folders (
                    name text primary key,
                    created_at datetime default current_timestamp
                );

                create table if not exists document_chunks (
                    id integer primary key autoincrement,
                    document_id integer not null,
                    chunk_index integer not null,
                    content text not null,
                    embedding_json text,
                    embedding_model text,
                    created_at datetime default current_timestamp
                );

                create table if not exists answers (
                    id integer primary key autoincrement,
                    question_id integer not null,
                    draft text not null,
                    edited text,
                    context_json text,
                    feedback_json text,
                    created_at datetime default current_timestamp,
                    updated_at datetime default current_timestamp
                );

                create table if not exists answer_notes (
                    id integer primary key autoincrement,
                    answer_id integer not null,
                    question_id integer not null,
                    note_type text not null,
                    quote text not null,
                    comment text,
                    rewrite text,
                    created_at datetime default current_timestamp
                );

                create table if not exists style_memories (
                    id integer primary key autoincrement,
                    source_type text not null,
                    source_id integer,
                    content text not null,
                    weight real default 1.0,
                    created_at datetime default current_timestamp
                );

                create table if not exists generation_runs (
                    id integer primary key autoincrement,
                    question_id integer not null,
                    answer_id integer,
                    run_type text not null,
                    status text default 'drafted',
                    model text,
                    prompt text not null,
                    curl text,
                    context_json text,
                    style_memories_text text,
                    style_profile_text text,
                    global_prompt text,
                    guidance text,
                    draft text,
                    final text,
                    feedback_json text,
                    rating integer,
                    is_satisfied integer default 0,
                    note text,
                    created_at datetime default current_timestamp,
                    updated_at datetime default current_timestamp
                );

                create table if not exists settings (
                    key text primary key,
                    value text not null
                );
                """
            )
            self.migrate(conn)

    def migrate(self, conn):
        question_columns = {
            row["name"] for row in conn.execute("pragma table_info(questions)").fetchall()
        }
        if "guidance" not in question_columns:
            conn.execute("alter table questions add column guidance text")
        document_columns = {
            row["name"] for row in conn.execute("pragma table_info(documents)").fetchall()
        }
        if "folder" not in document_columns:
            conn.execute("alter table documents add column folder text default '默认'")
        conn.execute(
            "insert or ignore into knowledge_folders (name) values (?)",
            ("默认",),
        )
        for row in conn.execute(
            "select distinct coalesce(folder, '默认') as folder from documents"
        ).fetchall():
            if row["folder"]:
                conn.execute(
                    "insert or ignore into knowledge_folders (name) values (?)",
                    (row["folder"],),
                )

    def row_to_dict(self, row):
        return dict(row) if row else None

    def list_questions(self):
        with self.connect() as conn:
            rows = conn.execute(
                "select * from questions order by status = 'new' desc, heat desc, id desc"
            ).fetchall()
        return [self.row_to_dict(row) for row in rows]

    def get_question(self, question_id):
        with self.connect() as conn:
            row = conn.execute("select * from questions where id = ?", (question_id,)).fetchone()
        return self.row_to_dict(row)

    def delete_question(self, question_id):
        with self.connect() as conn:
            answer_ids = [
                row["id"]
                for row in conn.execute(
                    "select id from answers where question_id = ?", (question_id,)
                ).fetchall()
            ]
            for answer_id in answer_ids:
                conn.execute("delete from answer_notes where answer_id = ?", (answer_id,))
                conn.execute("delete from generation_runs where answer_id = ?", (answer_id,))
                conn.execute("delete from answers where id = ?", (answer_id,))
            conn.execute("delete from questions where id = ?", (question_id,))

    def add_question(self, title, source_url, description="", tags="", heat=50, guidance=""):
        if not title:
            raise ValueError("title is required")
        with self.connect() as conn:
            cur = conn.execute(
                """
                insert into questions (title, source_url, description, tags, heat, guidance)
                values (?, ?, ?, ?, ?, ?)
                """,
                (title, source_url, description, tags, heat, guidance),
            )
            question_id = cur.lastrowid
        return self.get_question(question_id)

    def update_question_guidance(self, question_id, guidance):
        with self.connect() as conn:
            conn.execute(
                "update questions set guidance = ? where id = ?",
                (guidance, question_id),
            )

    def update_question_status(self, question_id, status):
        with self.connect() as conn:
            conn.execute("update questions set status = ? where id = ?", (status, question_id))

    def update_question(self, question_id, title, source_url, description, tags, heat):
        if not title:
            raise ValueError("title is required")
        with self.connect() as conn:
            conn.execute(
                """
                update questions
                set title = ?, source_url = ?, description = ?, tags = ?, heat = ?
                where id = ?
                """,
                (title, source_url, description, tags, heat, question_id),
            )
        return self.get_question(question_id)

    def list_documents(self, folder=None):
        where = ""
        params = []
        if folder and folder != "全部":
            where = "where coalesce(folder, '默认') = ?"
            params.append(folder)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                select
                    id,
                    title,
                    source,
                    coalesce(folder, '默认') as folder,
                    length(content) as size,
                    created_at
                from documents
                {where}
                order by id desc
                """,
                params,
            ).fetchall()
        return [self.row_to_dict(row) for row in rows]

    def get_documents(self):
        with self.connect() as conn:
            rows = conn.execute("select * from documents order by id desc").fetchall()
        return [self.row_to_dict(row) for row in rows]

    def list_document_folders(self):
        with self.connect() as conn:
            conn.execute("insert or ignore into knowledge_folders (name) values (?)", ("默认",))
            rows = conn.execute(
                """
                select name as folder from knowledge_folders
                union
                select distinct coalesce(folder, '默认') as folder from documents
                order by folder
                """
            ).fetchall()
        folders = [row["folder"] for row in rows if row["folder"]]
        return folders or ["默认"]

    def add_document_folder(self, name):
        name = (name or "").strip()
        if not name:
            raise ValueError("folder name is required")
        with self.connect() as conn:
            conn.execute(
                "insert or ignore into knowledge_folders (name) values (?)",
                (name,),
            )
        return name

    def delete_document_folder(self, name):
        name = (name or "").strip()
        if not name:
            raise ValueError("folder name is required")
        with self.connect() as conn:
            count = conn.execute(
                """
                select count(*) as count
                from documents
                where coalesce(folder, '默认') = ?
                """,
                (name,),
            ).fetchone()["count"]
            if count:
                raise ValueError("文件夹里还有资料，不能删除")
            conn.execute("delete from knowledge_folders where name = ?", (name,))
            conn.execute(
                "insert or ignore into knowledge_folders (name) values (?)",
                ("默认",),
            )
        return 0

    def add_document(self, title, source, content, folder="默认"):
        if not title:
            raise ValueError("title is required")
        if not content:
            raise ValueError("content is required")
        folder = (folder or "默认").strip() or "默认"
        with self.connect() as conn:
            cur = conn.execute(
                "insert into documents (title, source, content, folder) values (?, ?, ?, ?)",
                (title, source, content, folder),
            )
            document_id = cur.lastrowid
        with self.connect() as conn:
            row = conn.execute(
                """
                select id, title, source, coalesce(folder, '默认') as folder,
                       length(content) as size, created_at
                from documents
                where id = ?
                """,
                (document_id,),
            ).fetchone()
        return self.row_to_dict(row)

    def document_exists_by_source(self, source):
        with self.connect() as conn:
            row = conn.execute(
                "select id from documents where source = ? limit 1", (source,)
            ).fetchone()
        return row is not None

    def document_exists_by_title(self, title, folder=None):
        title = (title or "").strip()
        if not title:
            return False
        sql = "select id from documents where title = ?"
        params = [title]
        if folder and folder != "全部":
            sql += " and coalesce(folder, '默认') = ?"
            params.append(folder)
        sql += " limit 1"
        with self.connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return row is not None

    def get_document(self, document_id):
        with self.connect() as conn:
            row = conn.execute("select * from documents where id = ?", (document_id,)).fetchone()
        return self.row_to_dict(row)

    def update_document_content(self, document_id, content):
        if not content:
            raise ValueError("content is required")
        with self.connect() as conn:
            conn.execute(
                "update documents set content = ? where id = ?",
                (content, document_id),
            )
        return self.get_document(document_id)

    def move_document(self, document_id, folder):
        folder = (folder or "默认").strip() or "默认"
        with self.connect() as conn:
            conn.execute(
                "insert or ignore into knowledge_folders (name) values (?)",
                (folder,),
            )
            conn.execute(
                "update documents set folder = ? where id = ?",
                (folder, document_id),
            )

    def delete_document(self, document_id):
        with self.connect() as conn:
            conn.execute("delete from document_chunks where document_id = ?", (document_id,))
            conn.execute("delete from documents where id = ?", (document_id,))

    def delete_documents_by_source(self, source):
        with self.connect() as conn:
            ids = [
                row["id"]
                for row in conn.execute("select id from documents where source = ?", (source,)).fetchall()
            ]
            for document_id in ids:
                conn.execute("delete from document_chunks where document_id = ?", (document_id,))
            conn.execute("delete from documents where source = ?", (source,))

    def replace_document_chunks(self, document_id, chunk_items):
        with self.connect() as conn:
            conn.execute("delete from document_chunks where document_id = ?", (document_id,))
            conn.executemany(
                """
                insert into document_chunks
                    (document_id, chunk_index, content, embedding_json, embedding_model)
                values (?, ?, ?, ?, ?)
                """,
                [
                    (
                        document_id,
                        item["chunk_index"],
                        item["content"],
                        json.dumps(item.get("embedding"), ensure_ascii=False)
                        if item.get("embedding") is not None
                        else None,
                        item.get("embedding_model"),
                    )
                    for item in chunk_items
                ],
            )

    def list_document_chunks(self, document_id=None):
        sql = """
            select
                document_chunks.*,
                documents.title,
                documents.source
            from document_chunks
            join documents on documents.id = document_chunks.document_id
        """
        params = []
        if document_id is not None:
            sql += " where document_chunks.document_id = ?"
            params.append(document_id)
        sql += " order by documents.id desc, document_chunks.chunk_index asc"
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        items = []
        for row in rows:
            item = self.row_to_dict(row)
            item["embedding"] = (
                json.loads(item["embedding_json"]) if item.get("embedding_json") else None
            )
            items.append(item)
        return items

    def document_index_status(self):
        statuses = []
        for doc in self.list_documents():
            chunks = self.list_document_chunks(doc["id"])
            embedded = [chunk for chunk in chunks if chunk.get("embedding")]
            models = sorted(
                {
                    chunk.get("embedding_model")
                    for chunk in chunks
                    if chunk.get("embedding_model")
                }
            )
            statuses.append(
                {
                    **doc,
                    "chunk_count": len(chunks),
                    "embedded_count": len(embedded),
                    "embedding_models": models,
                    "indexed": bool(chunks),
                    "fully_embedded": bool(chunks) and len(chunks) == len(embedded),
                }
            )
        return statuses

    def storage_stats(self):
        with self.connect() as conn:
            tables = {}
            for table in (
                "questions",
                "knowledge_folders",
                "documents",
                "document_chunks",
                "answers",
                "generation_runs",
                "style_memories",
                "settings",
            ):
                tables[table] = conn.execute(f"select count(*) as count from {table}").fetchone()[
                    "count"
                ]
            doc_chars = conn.execute(
                "select coalesce(sum(length(content)), 0) as size from documents"
            ).fetchone()["size"]
            chunk_chars = conn.execute(
                "select coalesce(sum(length(content)), 0) as size from document_chunks"
            ).fetchone()["size"]
            embedded_chunks = conn.execute(
                "select count(*) as count from document_chunks where embedding_json is not null"
            ).fetchone()["count"]
            embedding_models = [
                row["embedding_model"]
                for row in conn.execute(
                    """
                    select distinct embedding_model
                    from document_chunks
                    where embedding_model is not null
                    order by embedding_model
                    """
                ).fetchall()
            ]
        return {
            "db_path": str(self.path),
            "db_size_bytes": self.path.stat().st_size if self.path.exists() else 0,
            "tables": tables,
            "document_chars": doc_chars,
            "chunk_chars": chunk_chars,
            "embedded_chunks": embedded_chunks,
            "embedding_models": embedding_models,
        }

    def save_answer(self, question_id, draft, context):
        with self.connect() as conn:
            cur = conn.execute(
                "insert into answers (question_id, draft, context_json) values (?, ?, ?)",
                (question_id, draft, json.dumps(context, ensure_ascii=False)),
            )
            answer_id = cur.lastrowid
        return self.get_answer(answer_id)

    def add_generation_run(
        self,
        question_id,
        run_type,
        model,
        prompt,
        curl,
        context,
        style_memories_text="",
        style_profile_text="",
        global_prompt="",
        guidance="",
        draft="",
        answer_id=None,
    ):
        with self.connect() as conn:
            cur = conn.execute(
                """
                insert into generation_runs (
                    question_id, answer_id, run_type, model, prompt, curl,
                    context_json, style_memories_text, style_profile_text,
                    global_prompt, guidance, draft
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    question_id,
                    answer_id,
                    run_type,
                    model,
                    prompt,
                    curl,
                    json.dumps(context or [], ensure_ascii=False),
                    style_memories_text,
                    style_profile_text,
                    global_prompt,
                    guidance,
                    draft,
                ),
            )
            run_id = cur.lastrowid
        return self.get_generation_run(run_id)

    def link_generation_run_answer(self, run_id, answer_id):
        if not run_id:
            return
        with self.connect() as conn:
            conn.execute(
                """
                update generation_runs
                set answer_id = ?, updated_at = current_timestamp
                where id = ?
                """,
                (answer_id, run_id),
            )

    def complete_generation_run(self, answer_id, final, feedback):
        with self.connect() as conn:
            conn.execute(
                """
                update generation_runs
                set final = ?, feedback_json = ?, status = 'final_saved',
                    updated_at = current_timestamp
                where answer_id = ?
                """,
                (final, json.dumps(feedback, ensure_ascii=False), answer_id),
            )

    def get_generation_run(self, run_id):
        with self.connect() as conn:
            row = conn.execute("select * from generation_runs where id = ?", (run_id,)).fetchone()
        item = self.row_to_dict(row)
        if item:
            item["context"] = json.loads(item.pop("context_json") or "[]")
            item["feedback"] = json.loads(item.pop("feedback_json") or "{}")
        return item

    def get_latest_generation_run_for_question(self, question_id):
        with self.connect() as conn:
            row = conn.execute(
                """
                select * from generation_runs
                where question_id = ?
                order by updated_at desc, id desc
                limit 1
                """,
                (question_id,),
            ).fetchone()
        item = self.row_to_dict(row)
        if item:
            item["context"] = json.loads(item.pop("context_json") or "[]")
            item["feedback"] = json.loads(item.pop("feedback_json") or "{}")
        return item

    def list_generation_runs(self, limit=50, satisfied_only=False):
        where = "where is_satisfied = 1" if satisfied_only else ""
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                select
                    generation_runs.*,
                    questions.title as question_title
                from generation_runs
                join questions on questions.id = generation_runs.question_id
                {where}
                order by generation_runs.updated_at desc, generation_runs.id desc
                limit ?
                """,
                (limit,),
            ).fetchall()
        items = []
        for row in rows:
            item = self.row_to_dict(row)
            item["context"] = json.loads(item.pop("context_json") or "[]")
            item["feedback"] = json.loads(item.pop("feedback_json") or "{}")
            items.append(item)
        return items

    def update_generation_run_rating(self, run_id, rating=None, is_satisfied=False, note=""):
        with self.connect() as conn:
            conn.execute(
                """
                update generation_runs
                set rating = ?, is_satisfied = ?, note = ?, updated_at = current_timestamp
                where id = ?
                """,
                (rating, 1 if is_satisfied else 0, note, run_id),
            )

    def get_answer(self, answer_id):
        with self.connect() as conn:
            row = conn.execute("select * from answers where id = ?", (answer_id,)).fetchone()
        item = self.row_to_dict(row)
        if item:
            item["context"] = json.loads(item.pop("context_json") or "[]")
            item["feedback"] = json.loads(item.pop("feedback_json") or "{}")
        return item

    def get_latest_answer_for_question(self, question_id):
        with self.connect() as conn:
            row = conn.execute(
                """
                select * from answers
                where question_id = ?
                order by updated_at desc, id desc
                limit 1
                """,
                (question_id,),
            ).fetchone()
        item = self.row_to_dict(row)
        if item:
            item["context"] = json.loads(item.pop("context_json") or "[]")
            item["feedback"] = json.loads(item.pop("feedback_json") or "{}")
        return item

    def save_edited_answer(self, answer_id, edited, feedback):
        with self.connect() as conn:
            conn.execute(
                """
                update answers
                set edited = ?, feedback_json = ?, updated_at = current_timestamp
                where id = ?
                """,
                (edited, json.dumps(feedback, ensure_ascii=False), answer_id),
            )
        return self.get_answer(answer_id)

    def list_feedback(self, limit=8):
        with self.connect() as conn:
            rows = conn.execute(
                """
                select id, question_id, feedback_json, updated_at
                from answers
                where feedback_json is not null
                order by updated_at desc
                limit ?
                """,
                (limit,),
            ).fetchall()
        items = []
        for row in rows:
            item = self.row_to_dict(row)
            item["feedback"] = json.loads(item.pop("feedback_json") or "{}")
            items.append(item)
        return items

    def add_style_memory(self, content, source_type="manual_edit", source_id=None, weight=1.0):
        content = (content or "").strip()
        if not content:
            return None
        with self.connect() as conn:
            existing = conn.execute(
                """
                select id from style_memories
                where source_type = ? and coalesce(source_id, -1) = coalesce(?, -1)
                  and content = ?
                limit 1
                """,
                (source_type, source_id, content),
            ).fetchone()
            if existing:
                return self.get_style_memory(existing["id"])
            cur = conn.execute(
                """
                insert into style_memories (source_type, source_id, content, weight)
                values (?, ?, ?, ?)
                """,
                (source_type, source_id, content, weight),
            )
            memory_id = cur.lastrowid
        return self.get_style_memory(memory_id)

    def get_style_memory(self, memory_id):
        with self.connect() as conn:
            row = conn.execute("select * from style_memories where id = ?", (memory_id,)).fetchone()
        return self.row_to_dict(row)

    def list_style_memories(self, limit=20):
        with self.connect() as conn:
            rows = conn.execute(
                """
                select * from style_memories
                order by weight desc, created_at desc, id desc
                limit ?
                """,
                (limit,),
            ).fetchall()
        return [self.row_to_dict(row) for row in rows]

    def delete_style_memory(self, memory_id):
        with self.connect() as conn:
            conn.execute("delete from style_memories where id = ?", (memory_id,))

    def add_answer_note(self, answer_id, question_id, note_type, quote, comment="", rewrite=""):
        if not quote:
            raise ValueError("quote is required")
        with self.connect() as conn:
            cur = conn.execute(
                """
                insert into answer_notes
                    (answer_id, question_id, note_type, quote, comment, rewrite)
                values (?, ?, ?, ?, ?, ?)
                """,
                (answer_id, question_id, note_type, quote, comment, rewrite),
            )
            note_id = cur.lastrowid
        return self.get_answer_note(note_id)

    def get_answer_note(self, note_id):
        with self.connect() as conn:
            row = conn.execute("select * from answer_notes where id = ?", (note_id,)).fetchone()
        return self.row_to_dict(row)

    def list_answer_notes(self, answer_id=None, question_id=None, limit=50):
        sql = "select * from answer_notes"
        params = []
        clauses = []
        if answer_id is not None:
            clauses.append("answer_id = ?")
            params.append(answer_id)
        if question_id is not None:
            clauses.append("question_id = ?")
            params.append(question_id)
        if clauses:
            sql += " where " + " and ".join(clauses)
        sql += " order by created_at desc, id desc limit ?"
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self.row_to_dict(row) for row in rows]

    def delete_answer_note(self, note_id):
        with self.connect() as conn:
            conn.execute("delete from answer_notes where id = ?", (note_id,))

    def list_edited_answers(self, limit=30):
        with self.connect() as conn:
            rows = conn.execute(
                """
                select
                    answers.id,
                    answers.question_id,
                    answers.edited,
                    answers.updated_at,
                    questions.title as question_title
                from answers
                join questions on questions.id = answers.question_id
                where answers.edited is not null and length(answers.edited) > 0
                order by answers.updated_at desc
                limit ?
                """,
                (limit,),
            ).fetchall()
        return [self.row_to_dict(row) for row in rows]

    def get_setting(self, key, default=""):
        with self.connect() as conn:
            row = conn.execute("select value from settings where key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_setting(self, key, value):
        with self.connect() as conn:
            conn.execute(
                """
                insert into settings (key, value) values (?, ?)
                on conflict(key) do update set value = excluded.value
                """,
                (key, value),
            )
