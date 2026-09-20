# signal: prize tracks and how we used each sponsor's technology

signal is a multi-agent retrieval harness. A planner splits a question into sub-queries, parallel retrievers
search a financial-research corpus, a verifier audits the evidence, and a writer answers with `[n]` citations
that map to exact source lines. Below: the Hack the North 2026 tracks it targets, and for each one what we
actually built with the sponsor's technology. Where we did not use a sponsor's technology, it says so.

## Reading the evidence graph

Every answer draws a graph from events the server already streams:

- **left**: the question, then the sub-queries the planner made (each shows how many passages it retrieved)
- **grey squares**: passages that were retrieved but not used
- **blue boxes** `[n] file Lline`: passages cited in the answer, linked to it with solid lines
- **red dashed boxes**: conflicts the verifier found (same metric, company and period, different values)
- **answer node**: shows the conflict count, e.g. "2 conflicts"

It exists so a judge can see *why* an answer says what it says, and which sources disagree.

---

## RBC: Signal in the Noise

**Tech: MCP tool + real-world financial data.** `app/mcp_client.py` is a schema-agnostic streamable-HTTP MCP
client: it calls every tool that needs no arguments and normalises whatever documents come back. The chunker
handles headings, tables and `<details>` blocks, and the planner is fed a corpus overview, so pasting a new
MCP URL loads a new dataset with no code changes. Citations are precise: chunk ids map to exact line ranges,
citation numbers are validated against the evidence, and every `%`, `$` and decimal in an answer is checked
against the cited text. We measured the pipeline on the 30 RBC prize questions (`eval/rbc_prize_questions.json`).

## Elastic: Find the Signal

**Tech: Elasticsearch.** `ElasticStore` (`app/store.py`) indexes chunks with dense vectors and searches them
with BM25 and kNN, then fuses the two rankings with reciprocal rank fusion. Hybrid search finds both exact
tokens (tickers, figures) and paraphrases, which a financial corpus needs. If Elastic is unreachable the app
falls back to an in-process store with the same interface, so a demo can't die on a network problem. Reranking
is done by an LLM, not by Elastic, and we did not use ES|QL or Workflows.

## Rox: Best AI Agent

**Tech: LLMs only. Rox requires no sponsor product, and we used none.** We built to the track's criteria of
messy data, conflicting sources and robust error handling. The verifier detects conflicts (same metric,
company, period, different values) and normalises values so `$2.73` vs `$2.73` or `18` vs `18.0%` is not
flagged as a conflict. It separates different *measures* (adjusted vs reported) from real conflicts, flags gaps,
and triggers a follow-up retrieval round. The writer states conflicts and missing data instead of guessing.
Errors degrade gracefully: Elastic falls back to a local store, the cross-model verifier falls back to OpenAI,
and GPTZero errors never break an answer. Answers can then be sent on to Slack, Notion, Google Docs or Google
Slides (see Composio).

## Huawei: openJiuwen Multi-Agent Challenge

**Tech: we did NOT use JiuwenSwarm or WorkSwarm.** The track makes the framework optional, and this repo has no
openJiuwen code. We built the collaboration pattern the track judges directly, in plain Python. For riddle-style
"which company did X while also Y?" questions, a leader splits the question into clues, and one analyst agent
per company grades every clue against that company's own passages in parallel. Period analysts then check each
quarter for the top candidates, and an independent verifier on a different model family (DeepSeek via
OpenRouter) audits the result before the writer answers. The trace shows the agents collaborating.

## OpenAI: API Prizes

**Tech: OpenAI API.** `gpt-4.1` runs the planner, verifier and writer, `gpt-4.1-mini` handles the many small calls
(reranking and the per-company analyst grading), and `text-embedding-3-small` produces the dense vectors for retrieval. Structured JSON output
(`chat_json`) is what lets the agents hand each other typed facts, conflicts and gaps instead of prose.

> TODO(team): this track also asks how **Codex** helped development. We don't have that from the code, so add
> one or two honest sentences here, or drop this track.

## Sentry: Best Use of Sentry

**Tech: Sentry, four products beyond error monitoring** (the track needs two or more). Tracing gives one
`gen_ai.invoke_agent` span per pipeline agent, nested under the FastAPI request. AI agent monitoring instruments
the OpenAI calls. Logs carry a structured audit line per answer (passages, conflicts, gaps, ungrounded figures).
Profiling is tied to traces. Answers containing a figure or citation the evidence doesn't back raise a warning
event, so grounding failures surface in Sentry and not just in the UI (`app/observability.py`).

## GPTZero: Best Use of GPTZero API

**Tech: GPTZero AI-detection API** (`app/gptzero.py`). Each cited passage gets an AI-written-text badge, so a
data-quality signal sits next to every citation. The *Data quality* tab scans every long passage in the corpus
for machine-generated writing. Requests are cached, concurrency-capped, and only run after the user confirms,
because each passage costs credits.

## Composio: 6 months free + $10K credits

**Tech: Composio managed OAuth + tool execution** (`app/integrations.py`). This is the "apply the information
to something" step: under every answer, *send to…* posts it to Slack, saves it as a Notion page, or creates a
Google Doc or Google Slides deck. Composio holds and refreshes the OAuth tokens, so the app only stores an API
key, and calls are made as the signed-in user. Nothing is sent automatically: the user picks the destination,
chooses what to include (question, answer, sources, caveats) and edits the text first.

## MLH: Best Use of MongoDB Atlas

**Tech: MongoDB Atlas** (`app/db.py`). It stores accounts, chats and runs, with indexes on user email, per-user
chat recency and share tokens. Each run keeps its full event stream, so opening a past chat replays the graph,
trace and sources exactly as they were, and a chat can be shared as a revocable read-only link. Without a
connection string it falls back to an in-memory store so local runs still work.

---

## Not entered

The remaining tracks (Cloudflare, Browserbase, Backboard, Zip, Linq, Baseten, Huawei OMNI and the rest) need a
sponsor product we don't use.
