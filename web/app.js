let state = {
  questions: [],
  documents: [],
  feedback: [],
  persona: "",
  selectedQuestion: null,
  currentAnswer: null,
};

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.classList.remove("hidden");
  setTimeout(() => el.classList.add("hidden"), 2600);
}

async function bootstrap() {
  const data = await api("/api/bootstrap");
  state.questions = data.questions;
  state.documents = data.documents;
  state.feedback = data.feedback;
  state.persona = data.persona;
  $("personaBox").value = data.persona;
  $("modelStatus").textContent = `${data.llm.mode === "api" ? "API" : "演示"} · ${data.llm.model}`;
  renderAll();
}

function renderAll() {
  renderQuestions();
  renderDocuments();
  renderFeedback();
}

function renderQuestions() {
  const list = $("questionList");
  list.innerHTML = "";
  state.questions.forEach((q) => {
    const card = document.createElement("div");
    card.className = "question-card" + (state.selectedQuestion?.id === q.id ? " active" : "");
    card.innerHTML = `
      <h3>${escapeHtml(q.title)}</h3>
      <div class="meta">
        <span>热度 ${q.heat}</span>
        <span class="status">${statusLabel(q.status)}</span>
        <span>${escapeHtml(q.tags || "未标注")}</span>
      </div>
    `;
    card.onclick = () => selectQuestion(q.id);
    list.appendChild(card);
  });
}

function renderDocuments() {
  const list = $("documentList");
  list.innerHTML = "";
  state.documents.forEach((doc) => {
    const card = document.createElement("div");
    card.className = "doc-card";
    card.innerHTML = `
      <strong>${escapeHtml(doc.title)}</strong>
      <div class="meta">
        <span>${escapeHtml(doc.source || "未注明来源")}</span>
        <span>${doc.size} 字</span>
      </div>
    `;
    list.appendChild(card);
  });
}

function renderFeedback() {
  const list = $("feedbackList");
  list.innerHTML = "";
  if (!state.feedback.length) {
    list.innerHTML = `<p class="muted">保存人工终稿后，会在这里看到风格学习记录。</p>`;
    return;
  }
  state.feedback.forEach((item) => {
    const card = document.createElement("div");
    card.className = "feedback-card";
    const notes = item.feedback.style_notes || [];
    card.innerHTML = `
      <p>修改幅度：${item.feedback.change_ratio}</p>
      <p>${notes.map(escapeHtml).join("<br>")}</p>
    `;
    list.appendChild(card);
  });
}

function selectQuestion(id) {
  state.selectedQuestion = state.questions.find((q) => q.id === id);
  state.currentAnswer = null;
  $("emptyState").classList.add("hidden");
  $("workspace").classList.remove("hidden");
  $("selectedTags").textContent = state.selectedQuestion.tags || "未标注";
  $("selectedTitle").textContent = state.selectedQuestion.title;
  $("selectedDesc").textContent = state.selectedQuestion.description || "暂无描述";
  $("selectedUrl").textContent = state.selectedQuestion.source_url || "";
  $("selectedUrl").href = state.selectedQuestion.source_url || "#";
  $("answerBox").value = "";
  $("contextList").innerHTML = "";
  renderQuestions();
}

async function addQuestion() {
  const data = await api("/api/questions", {
    method: "POST",
    body: JSON.stringify({
      title: $("qTitle").value,
      source_url: $("qUrl").value,
      description: $("qDesc").value,
      tags: $("qTags").value,
      heat: $("qHeat").value,
    }),
  });
  state.questions.unshift(data.question);
  renderQuestions();
  toast("问题已加入池子");
}

async function addDocument() {
  const data = await api("/api/documents", {
    method: "POST",
    body: JSON.stringify({
      title: $("docTitle").value,
      source: $("docSource").value,
      content: $("docContent").value,
    }),
  });
  state.documents.unshift(data.document);
  renderDocuments();
  toast("知识库已更新");
}

async function searchKnowledge() {
  if (!state.selectedQuestion) return;
  const query = encodeURIComponent(`${state.selectedQuestion.title}\n${state.selectedQuestion.description || ""}`);
  const data = await api(`/api/search?q=${query}`);
  renderContext(data.results);
}

async function generateAnswer() {
  if (!state.selectedQuestion) return;
  $("generateBtn").disabled = true;
  $("generateBtn").textContent = "生成中...";
  try {
    const data = await api("/api/generate", {
      method: "POST",
      body: JSON.stringify({ question_id: state.selectedQuestion.id }),
    });
    state.currentAnswer = data.answer;
    state.questions = data.questions;
    $("answerBox").value = data.answer.draft;
    renderContext(data.context);
    renderQuestions();
    toast("初稿已生成");
  } finally {
    $("generateBtn").disabled = false;
    $("generateBtn").textContent = "生成 AI 初稿";
  }
}

async function saveEdit() {
  if (!state.currentAnswer) {
    toast("请先生成一版 AI 初稿");
    return;
  }
  const data = await api("/api/answers/edit", {
    method: "POST",
    body: JSON.stringify({
      answer_id: state.currentAnswer.id,
      edited: $("answerBox").value,
    }),
  });
  state.currentAnswer = data.answer;
  state.feedback = data.feedback_items;
  state.questions = data.questions;
  renderFeedback();
  renderQuestions();
  toast("终稿已保存，风格反馈已记录");
}

async function savePersona() {
  const data = await api("/api/persona", {
    method: "POST",
    body: JSON.stringify({ persona: $("personaBox").value }),
  });
  state.persona = data.persona;
  toast("风格画像已保存");
}

function renderContext(results) {
  const list = $("contextList");
  if (!results.length) {
    list.innerHTML = `<p class="muted">没有检索到相关知识片段，可以先补充知识库。</p>`;
    return;
  }
  list.innerHTML = "";
  results.forEach((item) => {
    const card = document.createElement("div");
    card.className = "context-card";
    card.innerHTML = `
      <div class="score">相关度 ${item.score} · ${escapeHtml(item.title)}</div>
      <p>${escapeHtml(item.content)}</p>
    `;
    list.appendChild(card);
  });
}

function statusLabel(status) {
  return {
    new: "待回答",
    drafted: "已生成",
    edited: "已编辑",
  }[status] || status;
}

function escapeHtml(value) {
  return String(value || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

$("refreshBtn").onclick = bootstrap;
$("addQuestionBtn").onclick = () => addQuestion().catch((err) => toast(err.message));
$("addDocBtn").onclick = () => addDocument().catch((err) => toast(err.message));
$("savePersonaBtn").onclick = () => savePersona().catch((err) => toast(err.message));
$("searchBtn").onclick = () => searchKnowledge().catch((err) => toast(err.message));
$("generateBtn").onclick = () => generateAnswer().catch((err) => toast(err.message));
$("saveEditBtn").onclick = () => saveEdit().catch((err) => toast(err.message));

bootstrap().catch((err) => toast(err.message));

