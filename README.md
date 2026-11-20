# AI Troubleshooting Assistant for L1/L2 Support

A small retrieval augmented assistant that helps support engineers investigate
technical issues. The user describes a problem in plain language. The app finds
the most relevant entries in a sample knowledge base, asks an LLM to answer using
only those entries, suggests troubleshooting checks, names the escalation team,
and shows which knowledge base entry was used.

All data is simulated. No production systems are accessed.

## How it works

1. **Knowledge base** (`kb.json`): 15 simulated entries, each with symptoms,
   diagnostic checks, a known resolution and an owning team.
2. **Retrieval**: TF IDF vectors (scikit-learn) with cosine similarity. The top 2
   entries are returned.
3. **Confidence gate**: if the best similarity score is below 0.12, the LLM is not
   called. The app says the knowledge base is insufficient and asks for more
   details (error code, service, time, recent changes).
4. **Generation**: the matched entries and the question go to an LLM with a strict
   prompt: answer only from the entries, give numbered checks, name the team, cite
   the entry id, and never invent a resolution.
5. **Provider failover**: Groq first, then Gemini. If every provider fails or is
   rate limited, the app builds the answer directly from the best matching entry,
   so the user still gets grounded guidance.
6. **UI**: a simple FastAPI web page. Each answer shows the source entries,
   match scores, and which mode produced it (groq, gemini, fallback or insufficient).

## AI tools and models used

- Runtime: Groq (`openai/gpt-oss-20b`) as the main provider, Gemini
  (`gemini-3.5-flash-lite`) as backup.
- Also integrated and tested, but not used in the demo: OpenRouter (free Gemma
  model) and Ollama (`llama3.2:3b`).
- Development: I used Claude to help plan the architecture, write, and debug the
  code. I ran and tested the prototype myself, compared the four LLM providers, and chose the final setup from the results.

## Setup

1. Install Python 3.10 or newer.
2. In the project folder:
   pip install -r requirements.txt
3. Copy `.env.example` to `.env` and fill in at least one API key and model.
   Without any key, the app still runs and answers directly from the knowledge base.
4. Start the server:
   python -m uvicorn app:app --reload
5. Open http://127.0.0.1:8000
6. Helpful pages: `/status` shows configured providers, `/diagnose` tests each one.

## Testing

`run_tests.py` sends 18 queries to the running server: 15 in scope (one per
knowledge base entry) and 3 out of scope. For each, it checks that the correct
entry was retrieved, or that the app refused to answer.

Retrieval results: 18/18 correct, including all 3 out of scope queries refused
without calling an LLM.

| Provider | LLM answered | Typical time | Notes |
|---|---|---|---|
| None (KB fallback) | n/a | about 1s | Answer built from the best entry |
| Groq | 14/15 (15/15 on a rerun) | 1 to 6s | Hit a tokens per minute limit on back to back requests |
| Gemini | 14/15 | 2 to 13s | One 60s timeout, handled by fallback |
| OpenRouter (free model) | 0/15 | 1 to 5s | Rate limited upstream on every call |
| Ollama (3B model) | 14/15 | 50 to 180s | Works offline but too slow on an 8 GB laptop without a GPU |

Failover was observed in practice: when Groq returned a rate limit error, Gemini
answered the same query.


## Handling unknown issues

Queries with no good match (for example "my laptop screen is flickering") never
reach the LLM. The app replies that the knowledge base is insufficient and asks
for more details. The prompt also tells the LLM to say so if the retrieved
entries do not fit, and never to invent a fix.

## Limitations

- TF IDF matches keywords, not meaning. Paraphrased issues with different words
  may miss the right entry. Sentence embeddings would fix much of this.
- The knowledge base is small and simulated.
- The match score threshold (0.12) was tuned by hand on a small set of queries.
- Free API tiers have rate limits, so a burst of requests can fall back to the
  plain knowledge base answer.
- Small local models can be slow, and they follow the "only use the entries" rule
  less reliably than larger hosted models.
- LLM output is not verified automatically. A person should confirm steps before
  acting on them in production.

## Possible improvements

Embedding based or hybrid search, a larger and versioned knowledge base, a
feedback button to improve entries, automatic answer grounding checks, and
access control and audit logging for production use.

## Files

- `app.py`: backend, retrieval, provider logic and UI
- `kb.json`: simulated knowledge base
- `run_tests.py`: automated test script
- `.env.example`: configuration template (never commit the real `.env`)
