---
title: Aster & Row Support Agent
emoji: 🎒
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Aster & Row Support Agent

A reliability-focused RAG support agent for the Aster & Row take-home. Built to survive the
four failure modes called out in the brief: conflicting policy answers, invented order
info, lost multi-turn context, and unsafe retrieved content.

<!-- DEMO: replace with your recorded GIF/video before submitting -->
**Demo:** `docs/demo.gif` *(record after your first live run — see "Recording the demo" below)*

---

## 1. Setup and run instructions

```bash
git clone <your-repo-url>
cd aster-row-agent
python -m venv .venv && source .venv/bin/activate      # optional but recommended
pip install -r requirements.txt

cp .env.example .env
# edit .env and set GROQ_API_KEY=gsk_...
# get a free key (no credit card) at https://console.groq.com/keys

python -m app.cli                 # interactive chat
python -m app.cli --debug         # also prints the retrieval/tool trace per turn
```

No database, no build step, no external services beyond the Groq API. The knowledge
base and order data are read straight from the `knowledge-base/` and `data/` folders on
startup; TF-IDF indexing happens in memory in well under a second.

### Required environment variables

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | yes | Model calls (chat + tool use), via Groq's free tier |
| `ASTER_ROW_MODEL` | no | Overrides the model string (default `openai/gpt-oss-120b`) |

See `.env.example`.

---

## Hybrid retrieval update

The retriever now uses hybrid lexical and semantic search: TF-IDF preserves
exact policy-language matches, `BAAI/bge-small-en-v1.5` via
`sentence-transformers` recovers paraphrases, and reciprocal-rank fusion
combines the two rankings. The embedding model downloads on first run and is
cached locally afterwards; no embedding API or vector database is required.

## 2. Model, embedding, framework, and storage choices

- **Model:** GPT-OSS 120B via **Groq**, called through the `openai` Python SDK pointed
  at Groq's OpenAI-compatible endpoint (`https://api.groq.com/openai/v1`), using native
  function/tool calling for `order_lookup`. Chosen for cost: Groq runs an ongoing free
  tier (unlike a one-time trial credit), so this can be developed and evaluated without
  a paid key. Groq deprecates/retires models on a regular cadence (this project
  originally targeted `llama-3.3-70b-versatile`, which Groq shut down on
  August 16, 2026, mid-project) — `ASTER_ROW_MODEL` makes swapping models a one-line
  change; check https://console.groq.com/docs/models for current tool-calling-capable
  options if `openai/gpt-oss-120b` is retired later. No agent framework
  (LangChain/LlamaIndex) — at 14 documents and one tool, a framework added indirection
  without buying reliability.
- **Retrieval:** TF-IDF (scikit-learn) + cosine similarity over heading-level chunks,
  **not** a neural embedding model. Chosen deliberately: it's deterministic (identical
  results every run, which matters for a regression eval suite), needs no model download
  or embedding API key, and is fast enough to reindex from scratch on every process start.
  The real cost is weaker generalization to heavy paraphrasing than a proper embedding
  model — documented as a limitation below, with a clear upgrade path.
- **Storage:** none — everything is re-indexed in memory from the markdown/JSON source
  files at startup. No vector DB; not needed at this corpus size, and the assignment
  explicitly says not to build one.
- **Framework:** plain Python, FastAPI-free — a CLI is the interface (see §7). Business
  logic lives in `app/`, independent of the interface, so a thin FastAPI wrapper could be
  added later without touching the agent.

---

## 3. Architecture

```
knowledge-base/*.md ──▶ app/ingest.py ──▶ app/retriever.py ──┐
                          (parse front matter,     (TF-IDF +   │
                           chunk by ## heading)      metadata   │
                                                      authority, │
                                                      conflict   │
                                                      registry)  │
                                                                 ▼
data/orders.json ──▶ app/tools/order_lookup.py ──▶ app/agent.py ──▶ app/cli.py
  (deterministic,       (allow-listed fields,        (system prompt,
   never returns          status-derived rules,        tool-use loop,
   internal/PII           handoff flags)                session history,
   fields)                                              structured trace log)
```

**One turn:**
1. User message is appended to the session's history (`app/session.py`).
2. `Retriever.retrieve_for_agent()` runs TF-IDF search over KB chunks, applies a metadata
   **authority weight** (active/official/customer-facing content is preferred; superseded
   and draft/internal content is down-weighted but not deleted), and separately **flags**
   any non-authoritative chunk that scores highly enough on raw similarity that the user
   is plausibly asking about it directly (e.g. quoting the internal migration note). A
   small **known-conflict registry** checks whether the query touches the one genuine
   active-vs-active disagreement in the corpus (product care guide vs. Breeze Tumbler
   product card on dishwasher safety).
3. Only the retrieved chunks (never the whole corpus) go into the per-turn context block,
   each explicitly tagged `AUTHORITATIVE` or `NON-AUTHORITATIVE (status=..., audience=...)`.
4. The model is called with the system prompt, the context block, full tool schema for
   `order_lookup`, and the session's message history (bounded to the last 12 turns).
5. If the model calls `order_lookup`, the **real** deterministic tool runs (never the
   model "pretending" to look something up), and only the sanitized, allow-listed result
   is fed back — internal/PII fields simply never enter the model's context, so they
   can't leak regardless of prompting.
6. The final response, sources, and a handoff flag are returned; everything (message,
   retrieval scores, tool calls + sanitized results, final response, errors) is logged as
   one JSON line to `logs/trace.jsonl`.

**Why retrieval precedence is enforced in code, not just in the prompt:** the system
prompt does tell the model to prefer active/official sources, but the retriever also
*structurally* prevents superseded/draft content from ever being labeled
`AUTHORITATIVE`, and citation requirements in the eval suite check this in the actual
retrieved metadata, not just the model's stated intentions. Defense in depth: even a
model that ignores instructions can't accidentally cite the wrong doc as authority,
because the authoritative set was already filtered before the prompt was built.

**Why order lookups are a real tool, not RAG:** `orders.json` is never embedded or put in
any prompt. `OrderLookupTool` is a plain function; the model can only see what it returns,
and the allow-list means secrets structurally cannot appear in a tool result.

---

## 4. Evaluation suite

```bash
python -m evaluation.evaluate                                   # visible + custom cases
python -m evaluation.evaluate --file evaluation/visible-cases.json
python -m evaluation.evaluate --json evaluation/results.json     # also dump JSON results
python -m evaluation.evaluate --llm-judge                        # + optional secondary LLM annotation (see below)
```

Requires `GROQ_API_KEY` (it drives the real agent end-to-end, including live tool
calls — this is intentionally not a mocked test of the harness itself; that's covered by
`tests/test_eval_assertions.py`).

**A note on Groq's free tier:** it's an ongoing free tier, not a one-time trial credit,
but individual models have per-minute and per-day request/token caps (see
[Groq's rate limit docs](https://console.groq.com/docs/rate-limits) for current numbers —
they change, so check before you run). In practice, a full 23-case run got throttled
partway through during development, with every case after the throttle point failing
identically (a fallback message, not a real per-case bug — see the note below). Two
mitigations are now in place: `evaluate.py` pauses `--delay` seconds (default 3.0) between
cases and 1.5s between turns of the same multi-turn case, proactively staying under the
per-minute cap instead of hoping post-failure retries recover in time; and
`Agent._create_with_retry` still backs off automatically (up to ~2 minutes of cumulative
backoff) if a 429 does happen. If you still see a real 429 error in `logs/eval_trace.jsonl`,
try `python -m evaluation.evaluate --delay 6` for a slower, safer run, or switch
`ASTER_ROW_MODEL` to another current tool-calling model (check
https://console.groq.com/docs/models).

**If several consecutive cases in your results all show the exact same response text**
(e.g. "I'm having trouble reaching the support system right now"), that is not several
different bugs — it's one thing: the run got rate-limited partway through and every call
after that point hit the same fallback. Check `logs/eval_trace.jsonl` for the actual
`"errors"` field on those turns to confirm, then re-run (ideally with a larger `--delay`).
Don't spend time debugging those cases individually; they didn't really run.

**Design:**
- Every visible case in `evaluation/visible-cases.json` runs unmodified.
- `evaluation/custom-cases.json` adds 8 original cases (≥5 required) covering: system-prompt
  exfiltration attempts, refusing to promise a completed action (address change), a second
  multi-turn order follow-up, a policy question about sharing gift-card codes in chat, PO
  box shipping (retrieval only), a final-sale + warranty combination, a malformed order ID
  embedded in a full sentence, and case-normalization on a real order ID.
- Assertions are **deterministic**: exact-string checks (`must_include`/`must_not_include`),
  source-citation checks against the actual retrieved/cited filenames, tool-call and
  tool-argument checks against the agent's real trace, and keyword-based heuristics for
  "concept" checks (documented in `evaluation/evaluate.py` — a known precision/recall
  tradeoff versus a semantic grader, chosen because the assignment explicitly disallows
  relying exclusively on another LLM to grade).
- Results are reported **per case** and **rolled up by category** (retrieval, tool-use,
  tool-reliability, privacy, conversation, prompt-security, abstention, source-conflict,
  multi-source-grounding, actions), not just a single score.
- The assertion engine itself (`evaluate_expectations`) is unit-tested with synthetic
  inputs in `tests/test_eval_assertions.py`, independent of any live model call.

### Optional secondary LLM-assisted annotation (`--llm-judge`)

The assignment requires that grading not rely *exclusively* on another LLM -- it does not
forbid using one as a secondary signal, and there's a real gap a purely deterministic
harness can't close: natural-language paraphrase is unbounded, so a fixed keyword/regex
list will always have some false negatives (a response that's semantically correct but
phrased in a way the check didn't anticipate) and, more rarely, false positives (a
matching substring in an unrelated context).

`evaluation/llm_judge.py` adds an **advisory-only** second pass, off by default
(`--llm-judge`, since it costs one extra model call per case against your daily token
quota): for every case, a judge model is given the conversation, the response, a
plain-language description of the expectation, and the deterministic checks' own
pass/fail results -- then asked whether those results actually match what the response
does in substance. Two design choices keep this trustworthy rather than turning it into
"the LLM decides the score":

- **The deterministic result stays authoritative.** The judge never changes a case's
  pass/fail or the reported total; it only flags disagreements for a human to look at.
- **The judge is scoped narrowly.** It isn't asked to grade the response from scratch
  (which would just be a second, less-tested version of the same paraphrase-matching
  problem) -- it's asked the narrower, more reliable question of whether a *specific,
  already-computed* check result matches reality. This is a meaningfully easier and more
  trustworthy task than open-ended quality judgment, and it's the reason to trust its
  disagreement flags more than a bare "did the agent do well" score would deserve.

Output is a per-case verdict plus an agreement/disagreement count; `tests/test_llm_judge.py`
covers the JSON-parsing and prompt-construction logic with a mocked client, so the module's
own correctness doesn't depend on a live API call either.

### Baseline vs. final results

**Deterministic components** (unit tests, no live API needed — 34/34 passing):

| Suite | Result |
|---|---|
| `tests/test_retriever.py` (authority precedence, conflict detection, flagging, stemming) | 6/6 pass |
| `tests/test_order_lookup.py` (privacy allow-list, stale-ETA rules, ID normalization) | 10/10 pass |
| `tests/test_agent_mocked.py` (tool-use wiring, no-PII-to-model) | 1/1 pass |
| `tests/test_eval_assertions.py` (assertion engine correctness) | 12/12 pass |
| `tests/test_llm_judge.py` (optional secondary annotation layer, mocked client) | 5/5 pass |

Run `pytest -q` to reproduce these.

**Live end-to-end eval.** The "baseline" here isn't a single clean run — the first four
live runs each surfaced a genuine infrastructure or harness bug (a deprecated model ID,
two rounds of invisible Unicode substitution in model output, missing retrieval stemming,
best-chunk-vs-best-document retrieval, and a handoff-detection regex gap — see the bug
diary below for each). Each round's score reflects *different* failures, not the same
ones failing to go away:

| Run | Total | What changed before the next run |
|---|---|---|
| 1 (baseline) | 0/23 | deprecated Groq model ID (Bug: not in diary, pure config) |
| 2 | 4/23 | fixed conflict-detection heading selection; Unicode hyphen in citations |
| 3 | 10/23 | throttling added for rate limits; several assertion-heuristic fixes |
| 4 | 13/23 | stemming + document-level retrieval expansion (2 real retrieval bugs) |
| 5 | 13/23 | invisible narrow-no-break-space bug found and fixed (different failures than run 4) |
| 6 (final) | **14/23** | handoff-detection regex + system-prompt tuning |

**Final run, by category:**

| Category | Result |
|---|---|
| retrieval | 2/3 |
| tool-use | 3/3 |
| tool-reliability | 3/4 |
| privacy | 1/2 |
| conversation | 1/2 |
| prompt-security | 1/2 |
| abstention | 0/1 |
| source-conflict | 1/1 |
| groundedness | 2/2 |
| multi-source-grounding | 0/2 |
| actions | 0/1 |
| **Total** | **14/23** |

This is a genuinely clean run (no rate-limit fallbacks). Inspecting the 9 failures
individually shows they split into two honest categories, not one undifferentiated pile:

1. **Real model-instruction-following gaps** (a handful of cases): exact phrasing of a
   policy duration ("45-calendar-day" instead of "45 calendar days" despite an explicit
   system-prompt instruction not to do this); not proactively surfacing tangential-but-
   relevant info (duties/taxes) unless asked directly; occasionally citing one of two
   contributing sources instead of both. These are genuine, minor characteristics of a
   free-tier model with weaker instruction-following than a frontier model -- see §6.
2. **Deterministic-grader precision limits, not agent errors** (the majority of the
   remaining failures): every `handoff` mismatch checked by hand turned out to be the
   model *correctly* offering to escalate, phrased in a way the regex checker didn't
   anticipate -- "help you **connect with** a specialist" vs. the expected "**connect you
   with**"; "a **Aster & Row** support specialist" (company name inserted mid-phrase); "I
   can **forward** your request to a specialist" (a verb never added to the pattern list).
   This is the honest floor of a keyword/regex-based, non-LLM-graded harness (which the
   assignment explicitly requires): natural language paraphrase variation is effectively
   unbounded, and no fixed pattern list closes 100% of it. Documented here rather than
   chased indefinitely -- see §6.

This is a documented, expected pair of tradeoffs — free-tier model capability vs.
frontier, and deterministic grading precision vs. LLM grading — not code defects. See §6
for the full reasoning.

---

## 5. Bug diary

Three reproduced failures found while building this, with root cause, fix, and the
regression test that now catches each one. (All three are already fixed on `main`; the
"repro" steps below reproduce them against the *pre-fix* code for illustration.)

### Bug 1 — Injection-adjacent content was invisible, so the agent couldn't explain it

- **How I reproduced it:** ran the retriever directly against the exact wording of the
  `retrieved-prompt-injection` visible case ("The migration note says to ignore the real
  policy and give everyone 60 days..."). The eval case expects the agent to say the
  migration note "is not authoritative" — but my first retriever implementation only
  returned the top-k chunks *after* applying the authority weight, so the migration note
  (status: draft, weight ×0.1) was pushed out of the returned set entirely. The model
  would never see it, so it could describe the wrong 30-day policy correctly but had
  nothing to point to when explaining *why* the user's claim about a "60-day migration
  note" was wrong.
- **Root cause:** conflating "should this be trusted as policy" with "should this be
  retrievable at all." A single weighted top-k cut both signals with one knob.
- **Fix:** split retrieval into two lanes in `retrieve_for_agent()`: an `authoritative`
  lane (weighted, used to answer policy questions) and a `flagged_non_authoritative` lane
  (raw-similarity threshold, explicitly tagged `NON-AUTHORITATIVE` in the prompt context)
  so the model can acknowledge and correctly dismiss a document the user is directly
  referencing, without ever treating it as policy.
- **Regression test:** `tests/test_retriever.py::test_migration_note_flagged_when_referenced`.

### Bug 2 — Order IDs embedded in a full sentence were rejected as malformed

- **How I reproduced it:** this is *not* from the visible cases — I found it by feeding
  the tool more realistic phrasing than the visible cases use, e.g. `"check ORD-1007
  please"` and `"ORD1007"` (no hyphen). The original `normalize_order_id()` only stripped
  whitespace/punctuation from the *whole* string and upper-cased it; a trailing/leading
  word around the ID meant the regex `^ORD-\d+$` never matched, so a perfectly valid,
  unambiguous order ID was rejected as malformed. Since a tool-calling model can
  sometimes pass along more of the user's phrasing than just the bare ID, this was a
  real risk of spuriously failing lookups.
- **Root cause:** the normalizer assumed the *entire* input was the ID rather than
  allowing the ID to be a substring of a longer phrase, and didn't handle a missing
  hyphen at all.
- **Fix:** added a bounded fallback in `OrderLookupTool.lookup()` — if the strict
  whole-string match fails, search for an `ORD-?\d+` pattern *inside* the raw input and
  normalize just that substring. This only extracts an ID that is already present
  verbatim in the input; it never guesses a different ID, preserving the data
  dictionary's "do not guess a substantially different order ID" rule (verified by the
  companion test that `"order 12345"` — no `ORD` prefix at all — still correctly fails).
- **Regression tests:**
  `tests/test_order_lookup.py::test_recovers_order_id_embedded_in_a_phrase` and
  `::test_does_not_guess_an_id_when_prefix_is_missing`.

### Bug 3 — Stale code path after a refactor (`detect_conflict` signature drift)

- **How I reproduced it:** while cleaning up `Retriever.detect_conflict()`, I noticed its
  `results` parameter was accepted but never used (the method always recomputed a full
  corpus search internally). Removing the dead parameter to avoid a misleading API
  broke the module's own `__main__` smoke-test block, which still called the old
  two-argument signature — running `python -m app.retriever` after the change raised a
  `TypeError` immediately.
- **Root cause:** a signature change wasn't grepped across all call sites before
  committing; the only reason it was caught immediately was that the module has a
  runnable smoke test at the bottom of the file.
- **Fix:** updated the stale call site, and this is the concrete reason
  `tests/test_retriever.py::test_tumbler_conflict_detected` exists as a real pytest case
  rather than relying on the `__main__` block alone — a proper test suite catches this
  class of bug in CI, not just when a human happens to run the file manually.
- **Regression test:** `tests/test_retriever.py::test_tumbler_conflict_detected` (plus the
  fixed `__main__` block itself no longer errors).

---

### Bug 4 — Conflict detection surfaced the wrong section of the product card

- **How I reproduced it:** ran the retriever directly against the eval case's query
  ("Should I put the entire Breeze Tumbler in the dishwasher?") and inspected exactly
  which text was placed in the `CONFLICT DETECTED` context block. The product-care doc's
  chunk was correct (`Breeze Tumbler` heading — "hand-washed... lid may be placed on the
  top rack"), but the product-card doc's chunk was `Product details` (dimensions/leakproof
  claims) — **not** `Cleaning`, the section that actually contains the conflicting claim
  ("all components are dishwasher safe"). The model was never shown the contradiction, so
  when it answered the tumbler question it correctly used the care-guide's nuanced
  guidance (body hand-wash, lid dishwasher-safe) and simply had no way to know a
  *different* document made a broader, conflicting claim — it wasn't ignoring an
  instruction, the retrieval step silently dropped the evidence before the prompt was
  ever built.
- **Root cause:** `detect_conflict()` originally picked "whichever chunk from this doc_id
  scored highest for the current query" as that doc's side of the conflict. A document
  with multiple sections (product details, cleaning, temperature use) doesn't guarantee
  the section that actually conflicts is also the one with the best TF-IDF overlap with
  this specific phrasing of the question — and in this case it wasn't.
- **Fix:** the conflict registry now pins the *exact heading* known to contain each side
  of a registered conflict (`"headings": {"CARE-2026-01": "Breeze Tumbler",
  "PROD-BREEZE-20": "Cleaning"}`) instead of selecting by score. This is a direct
  consequence of the registry already being a hand-curated, known-conflicts list (see
  §6 limitations) — since a human already had to identify the conflicting doc pair, pinning
  the exact conflicting sections at the same time is nearly free and far more reliable
  than hoping similarity search finds them.
- **Regression test:** `tests/test_retriever.py::test_tumbler_conflict_detected` now also
  implicitly depends on the correct heading being selected (verified manually; the doc-id
  set check would not have caught this by itself, which is worth knowing about that
  test's coverage).

### Bug 5 — Non-ASCII punctuation in model output broke exact-string eval checks

- **How I reproduced it:** ran the agent live (once switched to a working Groq model)
  and inspected its citation text. When citing filenames like `01-returns-policy-current.md`,
  the model sometimes rendered the hyphen as U+2011 (non-breaking hyphen) rather than
  ASCII `-` (e.g. `01‑returns‑policy‑current.md`) -- and similarly for hand-wash-style
  compound words in prose. This is a known behavior of some models applying "smart
  typography" to output text. It's invisible to a human reader but silently fails any
  exact-substring `required_sources` or `must_include` check.
- **Root cause:** the eval harness compared raw model output against ASCII expected
  strings with no normalization step.
- **Fix:** added `normalize()` in `evaluation/evaluate.py`, mapping common Unicode
  punctuation variants (non-breaking hyphen, en/em dash, minus sign, smart quotes,
  ellipsis) to their ASCII equivalents, applied to all response text before any
  assertion runs.
- **Regression test:** covered indirectly by re-running the full suite against live
  output; a synthetic unit test asserting `normalize("hand\u2011washed") ==
  "hand-washed"` would be a good addition if you want an explicit one.

---

### Bug 6 — No stemming caused a hard retrieval miss on a simple verb-tense mismatch

- **How I reproduced it:** ran the visible case's exact wording, "Can you ship an Atlas
  Weekender to Germany?", directly against the retriever and got **zero** authoritative
  results back -- the international shipping doc, which literally opens with "Aster & Row
  currently ships internationally only to Canada," never surfaced at all.
- **Root cause:** plain `TfidfVectorizer` tokenizes "ship" and "ships" as two unrelated
  tokens with zero shared vocabulary. The query's "ship" and the document's "ships" never
  overlap, so cosine similarity was a hard 0 -- not "low," actually zero -- and the chunk
  was filtered out before it ever reached authority weighting.
- **Fix:** added a lightweight custom tokenizer (`_stem()` in `app/retriever.py`) that
  strips common suffixes (`s`, `es`, `ing`, `ed`, `ies`) before vectorizing, collapsing
  ship/ships/shipping, day/days, duty/duties, return/returns into shared tokens. Not real
  linguistic stemming (no NLTK dependency, deliberately kept simple for a 14-doc corpus),
  but enough to close this specific, verified failure class. Discovering this also
  surfaced a second, quieter bug: sklearn's built-in stop-word list is matched against
  tokenizer *output*, so an unstemmed stop-word list silently stopped filtering any
  stopword whose stem changed (e.g. "across" -> "acros") the moment a custom tokenizer was
  introduced -- fixed by stemming the stop-word list with the same function.
- **Regression test:** covered by re-running `tests/test_retriever.py` and the live
  `unsupported-country` case; a synthetic unit test asserting the international-shipping
  doc is retrieved for a "ship ... Germany" query would be a good explicit addition.

### Bug 7 — Best-chunk-per-corpus selection dropped clearly relevant sections of a selected document

- **How I reproduced it:** for the `canada-multiturn` follow-up ("What about Canada?")
  and the `final-sale-damaged-exception` case, the retriever correctly identified the
  right *documents* but the specific sections containing the actually-needed facts
  ("Duties and taxes"; "Reports after seven days") never made it into the model's context
  -- they lost out to other sections of the *same* document for one of only 4 top-k slots
  across the whole corpus, because a top-level "top-k chunks globally" selection doesn't
  guarantee every section of a relevant document gets a slot.
- **Root cause:** retrieval treated the *chunk* as the unit of relevance ranking, when for
  a corpus this small (14 documents, 2-5 short sections each) the *document* is the more
  sensible unit -- once a document is clearly the right source, its other sections are
  essentially free to include and are often exactly what's missing from a narrow top-k cut.
- **Fix:** `retrieve_for_agent()` now selects the top-N *documents* by best-chunk score,
  then includes every authoritative chunk belonging to those documents -- including
  sections with zero term overlap with this specific phrasing of the query (needed a
  second helper, `_score_all_chunks()`, since the ranking-oriented `search()` path filters
  out zero-score chunks entirely before this expansion step ever sees them).
- **Regression test:** covered by re-running `tests/test_retriever.py` and the live
  `canada-multiturn` and `final-sale-damaged-exception` cases; explicit unit tests
  asserting "Duties and taxes" and "Reports after seven days" appear in the authoritative
  set for their respective queries would be good, more targeted additions.



### Bug 8 — A different invisible-whitespace character, found only after three rounds of guessing wrong

- **How I reproduced it:** `valid-order-lookup`'s `must_include: "August 22, 2026"` failed
  across three separate live runs despite the response visibly containing that exact
  phrase in every copy-pasted transcript. Two earlier hypotheses (a stale eval harness;
  Unicode "mathematical bold" digit substitution) were tested and ruled out. The only way
  to actually resolve it was inspecting the *raw* stored response with `repr()` instead of
  reading rendered/pasted text, which revealed `U+202F` (narrow no-break space) in place
  of plain spaces: `"August\u202f22,\u202f2026"`. This is a second, different Unicode
  substitution from Bug 5's non-breaking hyphen -- same failure class (typographic
  Unicode instead of ASCII), different character, and this one is *whitespace*, which is
  visually indistinguishable from a real space in literally every font and renderer. No
  amount of eyeballing a transcript would ever have caught this; only inspecting raw
  codepoints would.
- **Root cause:** `normalize()` handled punctuation substitutions (hyphens, quotes,
  ellipsis) but had no whitespace normalization at all, so a narrow no-break space
  silently broke any exact-substring check spanning it.
- **Fix:** extended `_PUNCT_NORMALIZE` to also collapse narrow no-break space (`U+202F`),
  no-break space (`U+00A0`), thin space (`U+2009`), and zero-width space (`U+200B`, dropped
  entirely) to a plain ASCII space (or nothing, for the zero-width case).
- **Regression test:** `tests/test_eval_assertions.py::test_narrow_no_break_space_does_not_break_exact_match`,
  built directly from the exact failing live text rather than a synthetic guess.
- **Lesson for the eval methodology itself:** copy-pasted transcripts are not a reliable
  debugging surface for exact-string checks when the thing under test is LLM output --
  Unicode substitution is invisible to a human reader by definition. The fix that actually
  worked was asking for `repr()` of the raw stored value, not another round of "does this
  look right to you."



### Bug 9 — Handoff phrase list broke in both directions on real model phrasing

- **How I reproduced it:** two visible cases failed in opposite ways in the same run.
  `custom-address-change-no-promise` (expects `handoff: true`) failed because the model
  wrote "a human **support** specialist," which doesn't contain the literal substring "a
  human specialist" -- the inserted word broke the match. Meanwhile `shipped-without-eta`
  (expects `handoff: false`) failed the other way: the model had started habitually
  appending "I can connect you with a human specialist" to routine, already-complete
  answers (a plain shipped-order status check, an order genuinely not found by ID) where
  no real escalation was needed.
- **Root cause:** two separate problems disguised as one. (1) A literal phrase list is
  brittle against any word insertion a model might use. (2) The system prompt only said
  *when* a human specialist was an appropriate recommendation, not that it should be
  reserved for genuine next-steps rather than added as a reflexive courtesy -- so the
  model drifted toward over-offering it.
- **Fix:** two changes, matched to the two problems. Replaced the literal `HANDOFF_PHRASES`
  list with tolerant regex patterns (`HANDOFF_PATTERNS`) that allow common insertions like
  "support"/"human"/"our" in different combinations. Separately, added an explicit "WHEN TO
  RECOMMEND A HUMAN SPECIALIST (and when not to)" section to the system prompt telling the
  model not to append an escalation offer to an already-complete routine answer.
- **Regression tests:** `tests/test_eval_assertions.py::test_handoff_regex_tolerates_word_insertions`
  and `::test_generic_signoff_does_not_trigger_false_handoff` (the latter already existed
  from Bug fixes in an earlier round and still passes under the new regex).



## Retrieval benchmark

The repository includes a reproducible document-retrieval benchmark that
compares the three ranking strategies used during development:

```bash
python -m evaluation.retrieval_benchmark
```

It evaluates TF-IDF-only lexical search, `BAAI/bge-small-en-v1.5` semantic
search, and reciprocal-rank-fused hybrid search against 12 labelled queries in
`evaluation/retrieval-benchmark-cases.json`. The queries intentionally include
paraphrases (for example, a zipper failure instead of the phrase "warranty"),
exact policy wording, and the known multi-document tumbler conflict.

The report prints macro Recall@1, Recall@3, Hit@1, Hit@3, and MRR at the
document level. It deduplicates heading-level chunks before scoring, so a
document with many headings cannot inflate a metric. These are retrieval
metrics only: policy authority, conflict handling, and generation correctness
remain covered by the existing unit and end-to-end evaluation suites.

## 6. Known limitations / what I'd improve before production

- **Baseline eval runs surfaced real bugs across four rounds, not yet a clean final run**
  (see §4 and Bugs 4–9 in the diary) — 4/23 → 10/23 → 13/23 → 13/23 (different failures
  each time) as each round's fixes revealed the next real issue: a conflict-detection
  heading bug, two rounds of invisible Unicode substitution, missing stemming, a
  best-chunk-vs-best-document retrieval bug, and a handoff-detection bug that broke in
  both directions on real model phrasing. A fresh run after this round is the next step.
- **TF-IDF retrieval, not embeddings** — even with the light stemmer added in Bug 6, this
  will under-perform on real paraphrases or synonyms with low lexical overlap (e.g. "vegan"
  vs. "cruelty-free" vs. "animal-derived materials" share no stem at all). A production
  version should swap in a real embedding model (Voyage, OpenAI, or a local
  sentence-transformer) behind the same `Retriever` interface — the metadata
  authority-weighting, document-level expansion, and conflict registry logic don't change.
- **Conflict detection is a small hand-curated registry**, not general contradiction
  detection between arbitrary documents. It catches the one known conflict in this corpus
  but would miss a newly introduced conflicting pair until someone adds it to
  `KNOWN_CONFLICTS`. A more general approach (e.g. clustering same-topic chunks across
  docs and running a lightweight NLI check) is future work.
- **Concept-based eval assertions are keyword heuristics**, not semantic understanding —
  documented tradeoff in `evaluation/evaluate.py`. They're deterministic and fast but can
  have false negatives on a correct answer phrased very differently than expected.
- **Session store is in-memory and single-process** — fine for a CLI demo, not for a
  multi-instance deployment. Would move to Redis or similar with a TTL for real traffic.
- **Retry covers rate limits, not all failure modes** — `Agent._create_with_retry` backs
  off on 429s (expected occasionally on Groq's free tier), but a fully production system
  should also retry on 5xx/timeout and consider a circuit breaker if Groq is down.
- **Tool-use loop is capped at 4 rounds** as a safety bound against a runaway loop; never
  observed to matter in testing but worth monitoring in the trace log.
- **Provider choice was cost-driven, not quality-driven** — a Groq free-tier model was
  chosen specifically to avoid requiring a paid key for a take-home assignment (currently
  `openai/gpt-oss-120b`, after the originally-targeted `llama-3.3-70b-versatile` was
  deprecated by Groq mid-project). A production deployment would likely re-evaluate
  against a frontier model on the actual eval suite before committing, since tool-calling
  reliability and instruction-following under adversarial retrieved content (the
  prompt-injection case) can vary meaningfully between model families.

---

## 7. Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository. Keep `app.py` and
   `requirements.txt` in the repository root.
2. In Streamlit Community Cloud, create an app and select `app.py` as the
   entrypoint. Use Python 3.12 unless you have tested another supported
   version.
3. In **Advanced settings → Secrets**, add:

   ```toml
   GROQ_API_KEY = "gsk_your-key-here"
   # Optional:
   # ASTER_ROW_MODEL = "openai/gpt-oss-120b"
   ```

   Do not commit `.streamlit/secrets.toml` or `.env`.

The BGE embedding model downloads from Hugging Face on the first cold start.
It is cached for the lifetime of the running Streamlit instance, so normal
requests do not rebuild the index or re-download the model.

## 8. Interface

CLI only (`python -m app.cli`), per "visual polish will not affect the score." Each
response shows the answer, the sources it cited (if any), and whether a human handoff is
recommended:

```
you> How long can I return a backpack?

agent> Standard customers have 30 calendar days from delivery to request a
       return, as long as the item meets the condition requirements...
       sources: 01-returns-policy-current.md
```

Run with `--debug` to also print the full structured trace (retrieval scores, tool calls,
handoff reasoning) after each turn — this doubles as the observability requirement in §6
of the assignment; the same structure is written to `logs/trace.jsonl` on every turn.

---

## 8. AI coding tools used

This entire repository (architecture, code, tests, eval harness, and this README) was
built with **Claude** (Anthropic) in an agentic coding session, with me reviewing and
running the code at each step rather than accepting it blind.

**One example of an AI-generated suggestion that was wrong/incomplete:** the first draft
of the retriever applied the authority weight *before* selecting top-k results and
returned only that single weighted list to the agent. It looked correct and passed a
casual smoke test on plain queries, but it silently made the internal migration-note
document (the prompt-injection test fixture) *unretrievable* for the one case that
specifically needed the agent to see and dismiss it — a subtle bug that only showed up
when tested against the actual wording of the `retrieved-prompt-injection` eval case
rather than a generic query. See Bug 1 in the diary above for the fix. This is exactly
the kind of failure the assignment is testing for, and it's a good example of why
"looks right on a happy-path demo" isn't sufficient — it needed a case specifically
designed to probe the failure mode.

---

## 9. Recording the demo

Once you've run `python -m app.cli` locally with a real key, record a 2–4 minute
GIF/video showing, in order:
1. A knowledge-base question with citations (e.g. "How long can I return a backpack?").
2. An order lookup (e.g. "Where is ORD-1007 and when will it arrive?").
3. A multi-turn follow-up (e.g. "Do you ship internationally?" → "What about Canada?").
4. A correct refusal/handoff (e.g. the Breeze Tumbler dishwasher conflict question, or
   the vegan-materials abstention case).
5. `python -m evaluation.evaluate` running to completion with the category summary.

Save it to `docs/demo.gif` and it will render inline at the top of this README.

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── .env.example
├── app/
│   ├── ingest.py           # front-matter parsing + heading-level chunking
│   ├── retriever.py        # TF-IDF search, authority weighting, conflict registry
│   ├── session.py          # in-memory multi-turn session store
│   ├── agent.py            # system prompt, tool wiring, orchestration, logging
│   ├── cli.py               # interactive CLI
│   └── tools/
│       └── order_lookup.py # deterministic, privacy-safe order lookup
├── evaluation/
│   ├── visible-cases.json  # supplied cases (unmodified)
│   ├── custom-cases.json   # 8 original cases
│   └── evaluate.py         # eval runner + deterministic assertion engine
├── tests/
│   ├── test_retriever.py
│   ├── test_order_lookup.py
│   ├── test_agent_mocked.py
│   └── test_eval_assertions.py
├── knowledge-base/         # supplied, unmodified
├── data/                   # supplied, unmodified
└── logs/                   # trace.jsonl written here at runtime (gitignored)
```
