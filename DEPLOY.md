# Running signal in Docker

This runs the whole app in one container on your own machine. It does not need a domain or a hosting account.

## What you need

- Docker Desktop, running.
- A `.env` file next to this one. Copy `.env.example` and fill it in. Nothing in `.env` is copied into the image; Compose passes it in when the container starts.

| Variable | Needed? | Notes |
|---|---|---|
| `OPENAI_API_KEY` | yes | answers, planning, embeddings |
| `ELASTIC_URL`, `ELASTIC_API_KEY` | yes | hybrid search (falls back to a slower local index without them) |
| `MONGODB_URI` | yes | accounts and saved chats. In Atlas, Network Access must allow the machine running Docker |
| `SESSION_SECRET` | yes | a long random string. If it changes, everyone is signed out |
| `COMPOSIO_API_KEY` | optional | Slack, Notion, Google Docs and Slides |
| `OPENROUTER_API_KEY` | optional | independent verifier model in Deep mode (falls back to OpenAI) |
| `SENTRY_DSN` | optional | error and trace monitoring |
| `GPTZERO_API_KEY` | leave blank | switches the AI-detection UI on |
| `ADMIN_USERS` | optional | emails that are admins. Otherwise the first account created is |

`COOKIE_SECURE` is set to `0` by `docker-compose.yml` because you will use plain `http://localhost`.

## Start it

```
docker compose up --build
```

Open http://localhost:8000. Stop any other copy running on port 8000 first.

The first start indexes the dataset into Elastic, which takes 15 to 30 seconds. The header shows "indexing…" until it is done. Create your account first, since the first account is the admin (only admins can load a new dataset).

Useful commands:

```
docker compose up -d --build   # run in the background
docker compose logs -f         # watch it
docker compose down            # stop it (data and accounts are kept)
```

## Demo-day checklist

1. Start the container 5 minutes early and ask one question so everything is warm.
2. Phase 2 (the new dataset): sign in as the admin, paste the new MCP URL in the header box and press "Load dataset".
3. If you connect Slack, Notion or Google in the apps panel, do it before the demo. The approval page is hosted by Composio, so it works without a domain.

## Limits of running it this way

- **Only your machine can open it.** Share links (`?share=...`) point at `localhost`, so they won't work for anyone else.
- **Other devices on your network** (a phone at `http://192.168.x.x:8000`) work, but the animated home screen falls back to a static dot grid, because WebGPU needs HTTPS or localhost. Everything else works.
- **One container only.** Rate limits and the loaded dataset live in memory, so do not scale it to several copies.

## If you later put it on the internet

Use the same image on any host that runs containers (Render, Railway, Fly.io). Set the variables from the table in the host's settings, set `COOKIE_SECURE=1` (the host provides HTTPS), pick an always-on plan so it does not cold-start, and in Atlas allow the host's address. Rotate every key that has been pasted into chats or screenshots before you do.
