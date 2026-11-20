import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env", override=True)

app = FastAPI(title="AI Troubleshooting Assistant")


# ---------- Config ----------
def env(name, default=""):
    """Read an env var and strip spaces and stray quotes."""
    value = os.getenv(name, default) or ""
    return value.strip().strip('"').strip("'").strip()


GROQ_API_KEY = env("GROQ_API_KEY")
GROQ_MODEL = env("GROQ_MODEL")
OPENROUTER_API_KEY = env("OPENROUTER_API_KEY")
OPENROUTER_MODEL = env("OPENROUTER_MODEL")
GEMINI_API_KEY = env("GEMINI_API_KEY")
GEMINI_MODEL = env("GEMINI_MODEL", "gemini-3.5-flash-lite")
OLLAMA_MODEL = env("OLLAMA_MODEL")
OLLAMA_URL = env("OLLAMA_URL", "http://localhost:11434")
FORCE_PROVIDER = env("PROVIDER").lower()  # groq, openrouter, gemini, ollama

MIN_SCORE = 0.12  # below this, we say the KB is insufficient
TOP_K = 2

SECRETS = [k for k in (GROQ_API_KEY, OPENROUTER_API_KEY, GEMINI_API_KEY) if k]


def redact(text):
    """Make sure API keys never appear in messages."""
    text = str(text)
    for s in SECRETS:
        text = text.replace(s, "***")
    return text


# ---------- Load knowledge base ----------
with open(BASE_DIR / "kb.json", "r", encoding="utf8") as f:
    KB = json.load(f)


def entry_text(e):
    return f"{e['title']} {e['symptoms']} {' '.join(e['checks'])}"


vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
KB_MATRIX = vectorizer.fit_transform([entry_text(e) for e in KB])


# ---------- Retrieval ----------
def retrieve(query):
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, KB_MATRIX)[0]
    ranked = sorted(range(len(KB)), key=lambda i: scores[i], reverse=True)
    return [(KB[i], float(scores[i])) for i in ranked[:TOP_K]]


# ---------- Prompt ----------
SYSTEM_PROMPT = """You are an L1/L2 technical support assistant.
Answer ONLY using the knowledge base entries given in the user message.
Rules:
1. Give numbered troubleshooting checks in a sensible order.
2. Name the team to escalate to, taken from the entry.
3. Mention the entry id(s) you used, like [KB001].
4. If the entries do not clearly match the issue, say the knowledge base is
   insufficient and ask for more details (error code, service name, time, logs).
5. Never invent a resolution that is not in the entries.
"""


def build_context(hits):
    parts = []
    for e, s in hits:
        parts.append(
            f"[{e['id']}] {e['title']}\n"
            f"Symptoms: {e['symptoms']}\n"
            f"Checks: {'; '.join(e['checks'])}\n"
            f"Resolution: {e['resolution']}\n"
            f"Escalate to: {e['team']}"
        )
    return "\n\n".join(parts)


def user_message(query, hits):
    return f"Knowledge base entries:\n{build_context(hits)}\n\nUser issue: {query}"


# ---------- HTTP helper ----------
def post_json(url, headers, body, timeout):
    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:250]}")
    return r.json()


def clean_reply(text):
    if not text or not str(text).strip():
        raise RuntimeError("model returned an empty reply")
    return str(text).strip()


# ---------- LLM providers ----------
def chat_completions(url, key, model, query, hits):
    """Works for Groq and OpenRouter, they share the same request format."""
    data = post_json(
        url,
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        {
            "model": model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message(query, hits)},
            ],
        },
        60,
    )
    return clean_reply(data["choices"][0]["message"]["content"])


def ask_groq(query, hits):
    return chat_completions(
        "https://api.groq.com/openai/v1/chat/completions",
        GROQ_API_KEY, GROQ_MODEL, query, hits,
    )


def ask_openrouter(query, hits):
    return chat_completions(
        "https://openrouter.ai/api/v1/chat/completions",
        OPENROUTER_API_KEY, OPENROUTER_MODEL, query, hits,
    )


def ask_gemini(query, hits):
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )
    prompt = f"{SYSTEM_PROMPT}\n\n{user_message(query, hits)}"
    data = post_json(
        url,
        {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
        {"contents": [{"parts": [{"text": prompt}]}]},
        60,
    )
    return clean_reply(data["candidates"][0]["content"]["parts"][0]["text"])


def ask_ollama(query, hits):
    data = post_json(
        f"{OLLAMA_URL}/api/chat",
        {"Content-Type": "application/json"},
        {
            "model": OLLAMA_MODEL,
            "stream": False,
            "options": {"temperature": 0.2},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message(query, hits)},
            ],
        },
        180,
    )
    return clean_reply(data["message"]["content"])


# name, is configured, function
PROVIDERS = [
    ("groq", bool(GROQ_API_KEY and GROQ_MODEL), ask_groq),
    ("openrouter", bool(OPENROUTER_API_KEY and OPENROUTER_MODEL), ask_openrouter),
    ("gemini", bool(GEMINI_API_KEY), ask_gemini),
    ("ollama", bool(OLLAMA_MODEL), ask_ollama),
]


def ask_llm(query, hits):
    """Try configured providers in order. Returns (answer, provider, errors)."""
    errors = []
    if FORCE_PROVIDER and FORCE_PROVIDER not in [p[0] for p in PROVIDERS]:
        errors.append(
            f"PROVIDER='{FORCE_PROVIDER}' is not valid, "
            "use groq, openrouter, gemini or ollama"
        )
    for name, configured, fn in PROVIDERS:
        if FORCE_PROVIDER and name != FORCE_PROVIDER:
            continue
        if not configured:
            if FORCE_PROVIDER == name:
                errors.append(f"{name}: not configured, check key and model in .env")
            continue
        try:
            return fn(query, hits), name, errors
        except Exception as ex:
            msg = f"{name}: {redact(ex)}"
            print(f"[LLM error] {msg}")
            errors.append(msg)
    return None, "fallback", errors


def fallback_answer(hits):
    """No LLM available: build the answer straight from the best KB entry."""
    e = hits[0][0]
    steps = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(e["checks"]))
    return (
        f"Most relevant guidance: {e['title']} [{e['id']}]\n\n"
        f"Troubleshooting checks:\n{steps}\n\n"
        f"Known resolution: {e['resolution']}\n\n"
        f"Escalate to: {e['team']}"
    )


# ---------- API ----------
class Query(BaseModel):
    message: str


@app.post("/ask")
def ask(q: Query):
    hits = retrieve(q.message)
    best_score = hits[0][1]

    # Confidence gate: do not call the LLM when nothing matches well
    if best_score < MIN_SCORE:
        return {
            "answer": (
                "I could not find a matching entry in the knowledge base, so I "
                "do not want to guess. Please share more details: the error "
                "message or code, the affected service, when it started, and "
                "any recent changes. If this is a new issue, escalate to L2."
            ),
            "sources": [],
            "mode": "insufficient",
            "errors": [],
        }

    answer, mode, errors = ask_llm(q.message, hits)
    if answer is None:
        answer = fallback_answer(hits)
        if errors:
            answer += "\n\n(LLM calls failed)\n" + "\n".join(errors)

    sources = [
        {"id": e["id"], "title": e["title"], "score": round(s, 2),
         "resolution": e["resolution"], "team": e["team"]}
        for e, s in hits if s >= MIN_SCORE
    ]
    return {"answer": answer, "sources": sources, "mode": mode, "errors": errors}


@app.get("/status")
def status():
    """Shows which providers are configured and which models they use."""
    return {
        "groq": {"configured": PROVIDERS[0][1], "model": GROQ_MODEL},
        "openrouter": {"configured": PROVIDERS[1][1], "model": OPENROUTER_MODEL},
        "gemini": {"configured": PROVIDERS[2][1], "model": GEMINI_MODEL},
        "ollama": {"configured": PROVIDERS[3][1], "model": OLLAMA_MODEL},
        "forced_provider": FORCE_PROVIDER or None,
    }


@app.get("/diagnose")
def diagnose():
    """Pings every configured provider once and reports ok or the error."""
    hits = retrieve("payment api returning 500 error")
    report = {}
    for name, configured, fn in PROVIDERS:
        if not configured:
            report[name] = "not configured"
            continue
        try:
            reply = fn("The payment API is returning a 500 error", hits)
            report[name] = "OK: " + reply[:80].replace("\n", " ")
        except Exception as ex:
            report[name] = "FAILED: " + redact(ex)
    return report


# ---------- Simple UI ----------
PAGE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>AI Troubleshooting Assistant</title>
<style>
  body { font-family: Arial, sans-serif; max-width: 900px; margin: 30px auto; padding: 0 16px; }
  textarea { width: 100%; height: 80px; padding: 8px; font-size: 15px; }
  button { padding: 10px 20px; font-size: 15px; margin-top: 8px; cursor: pointer; }
  .box { border: 1px solid #ccc; border-radius: 8px; padding: 14px; margin-top: 16px; white-space: pre-wrap; }
  .src { background: #f5f7fa; }
  .tag { font-size: 12px; color: #555; margin-top: 8px; }
</style>
</head>
<body>
<h2>AI Troubleshooting Assistant (L1/L2)</h2>
<textarea id="msg" placeholder="Describe the issue, e.g. The payment API is returning a 500 error"></textarea>
<br><button onclick="send()">Get help</button>
<div id="out"></div>
<script>
async function send() {
  const msg = document.getElementById("msg").value;
  const out = document.getElementById("out");
  out.innerHTML = "<div class='box'>Thinking...</div>";
  const res = await fetch("/ask", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({message: msg})
  });
  const d = await res.json();
  let html = "<div class='box'>" + d.answer.replace(/</g, "&lt;") +
             "<div class='tag'>mode: " + d.mode + "</div></div>";
  if (d.sources.length) {
    html += "<h4>Knowledge base entries used</h4>";
    d.sources.forEach(s => {
      html += "<div class='box src'><b>" + s.id + ": " + s.title + "</b> (match " + s.score + ")" +
              "\\nResolution: " + s.resolution + "\\nEscalate to: " + s.team + "</div>";
    });
  }
  out.innerHTML = html;
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE