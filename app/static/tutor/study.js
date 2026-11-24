"use strict";
const el = id => document.getElementById(id);
const curriculum = {id: "", filename: "Digital literacy"};
let books = [], active = curriculum, turns = [], busy = false;

function element(tag, text) {
  const item = document.createElement(tag);
  if (text !== undefined) item.textContent = text;
  return item;
}

async function request(path, options) {
  const response = await fetch("/tutor/" + path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : "Request failed (" + response.status + "). Please retry.");
  }
  return response.json();
}

function lock(value) {
  busy = value;
  for (const button of document.querySelectorAll("main button")) button.disabled = value;
  el("question").readOnly = value;
}

function renderLibrary() {
  const nav = el("library");
  nav.replaceChildren();
  for (const source of [curriculum, ...books]) {
    const button = element("button", source.filename);
    button.append(element("span", source.id ? source.pages_indexed + " readable pages" : "Built-in course"));
    button.setAttribute("aria-pressed", String(source.id === active.id));
    button.onclick = () => { if (source.id !== active.id) select(source); };
    nav.append(button);
  }
}

function select(source) {
  active = source;
  turns = [];
  el("title").textContent = source.filename;
  el("question").value = "";
  el("ask-status").textContent = "";
  el("conversation").replaceChildren();
  el("welcome").hidden = false;
  el("reset").hidden = true;
  renderLibrary();
  const starters = source.id
    ? ["Explain a key idea in this reading.", "Give me a practice question about this PDF."]
    : ["How can I check whether a source is reliable?", "How can I spot a phishing email?"];
  el("starters").replaceChildren(...starters.map(text => {
    const button = element("button", text);
    button.onclick = () => { el("question").value = text; el("question").focus(); };
    return button;
  }));
}

function message(role, text) {
  const item = element("article");
  item.className = "message " + role;
  item.append(element("h3", role === "user" ? "You" : "Course tutor"), element("p", text));
  el("conversation").append(item);
  return item;
}

function showSources(item, sources) {
  sources.forEach((source, index) => {
    const details = element("details");
    const name = source.source.startsWith("builtin://digital_literacy/")
      ? source.source.split("/").pop().replaceAll("_", " ")
      : source.source;
    const location = source.page ? " · page " + source.page : "";
    details.append(element("summary", "[" + (index + 1) + "] " + name + location),
                   element("blockquote", source.text));
    item.append(details);
  });
}

el("reset").onclick = () => { select(active); el("question").focus(); };
el("upload").onclick = () => el("file").click();
el("file").onchange = async () => {
  const file = el("file").files[0];
  el("file").value = "";
  if (!file || busy) return;
  if (file.size > 10 * 1024 * 1024) {
    el("file-status").textContent = "Choose a PDF of 10 MiB or less.";
    return;
  }
  lock(true);
  el("file-status").textContent = "Reading and indexing " + file.name + "…";
  try {
    const book = await request("documents?filename=" + encodeURIComponent(file.name), {
      method: "POST", headers: {"Content-Type": "application/pdf"}, body: file
    });
    books = [...books.filter(b => b.id !== book.id), book].sort((a, b) => a.filename.localeCompare(b.filename));
    select(book);
    el("file-status").textContent = "Ready to study · " + book.pages_indexed + " of " + book.pages + " pages indexed.";
    el("question").focus();
  } catch (error) { el("file-status").textContent = error.message; }
  finally { lock(false); }
};

el("ask").onsubmit = async event => {
  event.preventDefault();
  const question = el("question").value.trim();
  if (busy || !question) return;
  lock(true);
  el("welcome").hidden = true;
  const sent = message("user", question);
  el("ask-status").textContent = "Reading your source…";
  try {
    const reply = await request("ask", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question, document_id: active.id || null, history: turns.slice(-3)})
    });
    showSources(message("assistant", reply.answer), reply.sources);
    turns.push({question, answer: reply.answer.slice(0, 4000)});
    el("question").value = "";
    el("ask-status").textContent = "";
    el("reset").hidden = false;
    el("ask").scrollIntoView({block: "nearest"});
  } catch (error) {
    sent.remove();
    el("welcome").hidden = turns.length > 0;
    el("ask-status").textContent = error.message;
  } finally { lock(false); el("question").focus(); }
};
el("question").onkeydown = event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    el("ask").requestSubmit();
  }
};
select(curriculum);
lock(true);
request("documents").then(result => { books = result; renderLibrary(); })
  .catch(error => { el("file-status").textContent = "Unable to load the library. " + error.message; })
  .finally(() => lock(false));
