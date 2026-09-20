"""Tests must never talk to real services: no Sentry events, no real MongoDB, no paid APIs.
These are set before the app is imported (config is read at import time; explicit empty values win over .env)."""
import os

os.environ["SENTRY_DSN"] = ""
os.environ["MONGODB_URI"] = ""
