# ADR 0001: LLM provider is Claude; hosting is a single EC2 instance, not a split Vercel/Render stack

**Status:** Accepted
**Date:** 2026-07-11

## Context

The original project sketch (drafted as a CV bullet before any code existed) specified GPT-4 as the reasoning layer and implied a modern split frontend/backend deployment. Two constraints changed that:

1. The project owner holds an active Anthropic subscription and would rather spend against that than open a new OpenAI billing relationship. (Note: a Claude.ai Pro subscription does not include API access — a separate pay-as-you-go API console account is required; see `PLAN.md` §00.)
2. The project owner knows AWS (EC2, and RDS as a managed Postgres option) but not Next.js, Vercel, or Render. The project's stated goal is to learn as much as possible while building something deployable — that's better served by depth on a stack the owner can actually extend and debug, than breadth across unfamiliar tools that a Data Scientist role won't examine closely anyway.

## Decision

- **LLM provider: Claude** (Anthropic API), Haiku for the bulk explanation pass, Sonnet for a small evaluation sample. Interface is kept provider-agnostic in `llm/` (a thin client wrapper) so swapping providers later is a config change, not a rewrite.
- **Hosting: a single AWS EC2 instance** running Docker Compose (FastAPI + Streamlit + self-hosted Postgres), fronted by Nginx with Let's Encrypt TLS. No Vercel, no Render, no managed RDS by default.
- **Database: self-hosted Postgres in a container**, not RDS, to avoid RDS's always-on cost for a low-traffic portfolio demo. RDS remains a documented, easy upgrade path (swap the connection string) if ever wanted.

## Consequences

- CV language changes from "GPT-4" to "Claude" throughout (see `PLAN.md` §15).
- One box is a single point of failure with no redundancy — acceptable for a portfolio demo, called out explicitly in the risk register (`PLAN.md` §12) rather than silently accepted.
- Slightly less "trendy" frontend stack, but the full request lifecycle (Nginx → FastAPI → Postgres, Docker Compose orchestration, CI-driven SSH deploy) is more representative of what a Data Scientist is actually asked to debug in production than a Next.js app would be.
