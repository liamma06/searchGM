# signal

A multi-agent retrieval harness that answers precise, citable questions over large financial research
corpora served through an MCP tool. Built for Hack the North 2026 (RBC *Signal in the Noise*, Elastic
*Find the Signal*, Rox *Best AI Agent*, Huawei openJiuwen multi-agent).

## How it works

```
question -> Planner -> Retriever (x N, parallel) -> Verifier -> [follow-up round] -> Writer -> cited answer
```

| Agent | Job |
|---|---|
| **Planner** | Splits the question into sub-queries using the corpus overview (files, sections, entities), so it adapts to unseen datasets. |
| **Retriever** | Hybrid search (BM25 + dense vectors, fused with RRF), entity-filtered and unfiltered, then LLM rerank. |
| **Verifier** | Audits evidence before writing: extracts facts, detects conflicts (same metric + entity + period, different values), flags gaps, asks for follow-up retrieval. |
| **Writer** | Answers only from evidence with `[n]` citations; states conflicts and missing data instead of guessing. |

Control flow is plain Python (`app/orchestrator.py`); the LLMs do the reasoning. Every run streams events
to the UI (live agent trace, sources with exact highlighted lines, grounding badges).

Auditability checks that don't rely on the LLM: chunk ids map to exact source line ranges, citation
numbers are validated against the evidence, and every figure (`%`, `$`, decimals) in an answer is
checked against the cited text ("ungrounded" figures are flagged).

**Phase 2 (new dataset on judging day):** paste the new MCP server URL in the header and press
*Load dataset*. The MCP client calls every no-argument tool, the chunker is schema-agnostic (headings,
tables, `<details>` blocks), and the planner is given the new corpus overview. No code changes needed.

## Interface

- **Home screen** (shown first): the React Bits *Shape Waves* WebGPU background with the word "signal" carved out of it, plus sign-in / create-account or *open signal* buttons. Click "signal" in the app header to return to it. Browsers without WebGPU get a static fallback. The effect is built from `frontend-src/shapewaves/` into `frontend/vendor/` (see `frontend/vendor/README.md` for licenses).

- **Chat** with rendered markdown and clickable `[n]` citations that open the exact source passage (rendered, with a
  *raw lines* toggle for the numbered original).
- **Tracker** (right, collapsed by default): live pipeline steps with timings, sources cited, conflicts, gaps and
  whether every figure is grounded. Expands into the full panel: **graph**, sources, agent trace, data quality.
- **Graph**: an evidence graph that builds as the agents finish: question -> queries or clues -> (companies) ->
  retrieved passages -> cited passages -> answer. Retrieved-but-unused passages show as grey dots, cited ones as
  labelled nodes, supporting links as white lines and contradicting ones as red dashed lines. Hover to isolate a
  path, click a cited passage to open it. It is drawn from events the server already streams, so it adds no latency.
- **Deep team analysis** toggle and a **dataset loader** for a new MCP URL.

## Send to other apps

Under every answer, **send to…** opens a panel to post it to **Slack**, save it as a **Notion** page, or create a
**Google Doc** or **Google Slides** deck (`COMPOSIO_API_KEY`, via [Composio](https://composio.dev)). You pick the
destination and the channel or parent page, choose what to include (question, answer, sources, caveats such as
conflicts and missing data), and edit the exact text before it is sent. Each app keeps its own draft. An
**ask AI** box rewrites the draft from a plain instruction ("shorter", "only the ROE part", "bullet points for Slack")
with an undo; it keeps numbers and citations as written and never sends anything.

Sign-in to those apps is Composio-managed OAuth: each user connects an app once on a Composio-hosted page, Composio
stores and refreshes the tokens, and this app only holds the Composio key. Calls are made as the signed-in user, so
nobody can act with someone else's connected accounts. Nothing is ever sent automatically: an action happens only on
a click, on content the user can read and edit first, so text inside a document can't trigger one.

## Accounts and saved chats

Email + password accounts (argon2id hashes, signed HttpOnly session cookie, login throttling, CSRF header on
every unsafe request). Chats are stored in MongoDB (`MONGODB_URI`; without it an in-memory store is used and
nothing survives a restart). Each run keeps its full event stream, so opening a past chat **replays** the graph,
trace and sources exactly as they were. A chat can be shared as a read-only link (`/?share=<token>`) that works
without signing in and can be revoked. The first account to register is the admin; only admins can load a new
dataset or run the GPTZero scan. Questions are rate-limited per user (`RATE_LIMIT_PER_HOUR`). Set `AUTH_REQUIRED=0`
to run open as a single local user.

## Identification questions and the analyst team

The RBC prize questions are riddles ("which bank did X while also Y?"). The planner recognises them and
splits them into clues. Two ways to answer, chosen per request (`IDENTIFY_FLOW`, or the UI's
*Deep team analysis* toggle):

| Flow | How | Measured on the 30 RBC questions |
|---|---|---|
| `lookup` (default) | retrieve for the clues, verify, write | median 7 s, p90 14 s |
| `team` | one analyst agent per company grades every clue in parallel; period analysts check each quarter for the top candidates; an independent verifier on a different model family (`VERIFIER_MODEL`, via OpenRouter) audits the matrix; the writer answers | median 16 s, p90 43 s |
| `auto` | lookup first, escalate to the team if the verifier is unsure | median 7 s, p90 56 s |

All three named the same company on all 30 questions, so lookup is the default. The team is more
conservative: it reports which clues are unsupported instead of asserting a full match, and it shows the
agent collaboration in the trace. The verifier runs with reasoning **off**: with it on, the same call took
about 95 s versus about 7 s. Companies are recognised from several heading styles (ticker first, ticker in
parentheses, or any heading repeated across sections), so a differently formatted dataset still gets a team.

## Optional integrations (off unless configured in `.env`)

**Sentry** (`SENTRY_DSN`): Tracing (a `gen_ai.invoke_agent` span per agent, nested under the FastAPI request),
AI agent monitoring (OpenAI chat/embedding spans), Logs (a structured audit log per answer: passages,
conflicts, gaps, ungrounded figures), and Profiling. Answers containing figures or citations not backed by
evidence raise a warning event, so grounding failures show up in Sentry rather than only in the UI.

**GPTZero** (`GPTZERO_API_KEY`): each cited passage gets an AI-written-text badge, and the *Data quality* tab
scans every long passage in the corpus for machine-generated writing (one request per passage, so it asks
before running). Errors from the API never break an answer.

## Run

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
cp .env.example .env        # fill in OPENAI_API_KEY (+ ELASTIC_URL / ELASTIC_API_KEY; OPENROUTER_API_KEY for the team verifier)
.venv/Scripts/python -m uvicorn app.main:app --port 8000
# open http://localhost:8000
```

Without `ELASTIC_URL` the app uses an in-process BM25 + vector store with the same interface; with it,
the index is created and searched in Elasticsearch (BM25 + kNN, fused client-side with RRF).

```bash
.venv/Scripts/python -m pytest tests                 # unit tests
.venv/Scripts/python -m scripts.ask "question"       # CLI, prints the full agent trace
.venv/Scripts/python -m scripts.eval                 # regression eval (eval/questions.json)
.venv/Scripts/python -m scripts.compare_flows        # lookup vs team vs auto on the 30 RBC questions
```

To run it in Docker instead (no Python setup), see [DEPLOY.md](DEPLOY.md): `docker compose up --build`.
