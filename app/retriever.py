"""
Retriever over the knowledge-base chunks.

Design choices (documented in README too):
- TF-IDF + cosine similarity instead of a neural embedding model. With only
  14 short documents, TF-IDF is deterministic, needs no API key or GPU/model
  download, and is easy to reason about in an eval suite. The tradeoff is
  weaker generalization to heavy paraphrases than a real embedding model
  would give -- documented as a known limitation.
- A light suffix-stripping "stemmer" is applied before vectorizing. Plain
  TF-IDF treats "ship" (in a question) and "ships"/"shipping" (in a doc) as
  unrelated tokens, which is a real, verified retrieval miss: "Can you ship
  an Atlas Weekender to Germany?" scored a hard zero against the shipping
  doc without this (see README bug diary). This isn't full linguistic
  stemming (no NLTK dependency), just enough suffix normalization to close
  the most common verb/noun mismatches in this domain.
- Retrieval selects top-scoring *documents*, then includes ALL of that
  document's authoritative chunks (not just the single best-scoring
  section). A 14-document corpus with 2-5 short sections each makes
  whole-document inclusion cheap, and picking only the single best chunk
  per document was a verified real bug: relevant sections like "Duties and
  taxes" or "Reports after seven days" lost out to other sections of the
  *same* document for a top-k slot and were silently never shown to the
  model, even though the document as a whole was clearly the right source.
- A metadata "authority weight" is applied on top of the raw similarity
  score so that active/official/customer-facing content is preferred over
  superseded or draft/internal content, per the assignment's precedence
  requirement. Superseded and draft content can still be *retrieved* (e.g.
  to explain that a migration note exists and is not authoritative) but is
  never marked citable-as-authority.
- A small, explicit "known conflict" registry flags the one genuine
  active-vs-active conflict in the corpus (product care guide vs. the
  Breeze Tumbler product card on dishwasher safety). General-purpose
  automatic contradiction detection between arbitrary policy documents is
  out of scope for this system; this is a documented limitation.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

from app.ingest import Chunk, load_chunks

AUTHORITY_WEIGHT = {
    "active": 1.0,
    "superseded": 0.30,
    "draft": 0.10,
}

_TOKEN_RE = re.compile(r"[a-zA-Z]+")
_SUFFIXES = ("ing", "edly", "ed", "ies", "es", "s")


def _stem(word: str) -> str:
    """Very light suffix stripping -- not real linguistic stemming, just
    enough to collapse ship/ships/shipping, day/days, return/returns,
    duty/duties into the same token so exact-morphology mismatches between
    a question and a document don't cause a hard retrieval miss."""
    w = word.lower()
    if len(w) <= 4:
        return w
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            if suf == "ies":
                return w[: -len(suf)] + "y"
            return w[: -len(suf)]
    return w


def _stemming_tokenizer(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text)]


# sklearn's built-in stop-word list is matched against tokenizer *output*.
# Since our tokenizer stems words, an unstemmed list silently fails to
# filter any stopword whose stem changed (e.g. "across" -> "acros"), which
# is a real, functional bug (verified: it let common-word noise skew which
# documents scored highest), not just the cosmetic warning sklearn raises
# for it. Stem the stop-word list itself so filtering happens consistently
# in the same token space as the corpus.
_STOP_WORDS = frozenset(_stem(w) for w in ENGLISH_STOP_WORDS)


# Only chunks meeting all three of these are eligible to be *cited as
# policy authority* to a customer, regardless of similarity score.
def is_authoritative(c: Chunk) -> bool:
    return (
        c.status == "active"
        and c.policy_authority == "official"
        and c.audience == "customer"
    )


# Registry of known genuine conflicts between active, official documents
# where neither supersedes the other. Keyed by a topic hint used to decide
# whether a query is "about" this conflict. `headings` pins the *exact*
# section per doc that contains the conflicting claim -- picking "whatever
# chunk scores highest for this doc" (the original approach) is unreliable:
# a doc can have several sections, and the top-scoring one for a given
# query is not necessarily the one that actually conflicts (verified via a
# real miss -- see README bug diary).
KNOWN_CONFLICTS = [
    {
        "keywords": ["dishwasher", "breeze tumbler", "tumbler dishwasher", "hand-wash", "hand wash"],
        "headings": {"CARE-2026-01": "Breeze Tumbler", "PROD-BREEZE-20": "Cleaning"},
    }
]


@dataclass
class RetrievedChunk:
    chunk: Chunk
    raw_score: float
    weighted_score: float
    authoritative: bool


class Retriever:
    def __init__(self, kb_dir: str):
        self.chunks: list[Chunk] = load_chunks(kb_dir)
        corpus = [f"{c.title} {c.heading} {c.text}" for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words=list(_STOP_WORDS), tokenizer=_stemming_tokenizer, token_pattern=None)
        with warnings.catch_warnings():
            # sklearn flags 2 residual stemmed-stopword edge cases (e.g.
            # "across" -> "acro" vs. our stemmer's "acros"); harmless, not
            # worth a heavier stemmer for 2 words. See module docstring.
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        results = []
        for chunk, raw in zip(self.chunks, sims):
            if raw <= 0:
                continue
            weight = AUTHORITY_WEIGHT.get(chunk.status, 0.5)
            weighted = float(raw) * weight
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    raw_score=float(raw),
                    weighted_score=weighted,
                    authoritative=is_authoritative(chunk),
                )
            )
        results.sort(key=lambda r: r.weighted_score, reverse=True)
        return results[:top_k]

    def detect_conflict(self, query: str) -> dict | None:
        """Return conflict info if the query matches a known conflict topic
        AND both sides of that conflict are present among authoritative
        chunks anywhere in the corpus. Uses the registry's pinned heading
        per doc (not "best-scoring chunk for that doc") so the *actual*
        conflicting text is what gets surfaced, regardless of how TF-IDF
        happens to rank that doc's other sections for this query."""
        ql = query.lower()
        for conflict in KNOWN_CONFLICTS:
            if not any(kw in ql for kw in conflict["keywords"]):
                continue
            by_chunk_id = {c.chunk_id: c for c in self.chunks}
            selected = []
            all_present = True
            for doc_id, heading in conflict["headings"].items():
                match = next(
                    (c for c in self.chunks if c.doc_id == doc_id and c.heading == heading),
                    None,
                )
                if match is None or not is_authoritative(match):
                    all_present = False
                    break
                # wrap in a RetrievedChunk so downstream formatting code
                # (which expects .chunk/.raw_score) doesn't need two shapes
                selected.append(RetrievedChunk(chunk=match, raw_score=1.0, weighted_score=1.0, authoritative=True))
            if all_present:
                return {"topic": conflict["keywords"][0], "chunks": selected}
        return None

    def _search_all(self, query: str) -> list[RetrievedChunk]:
        return self.search(query, top_k=len(self.chunks))

    def _score_all_chunks(self, query: str) -> list[RetrievedChunk]:
        """Score every chunk against the query, including zero-overlap
        chunks (raw_score == 0). Needed for document-level expansion: once a
        document is selected as relevant, a section with no term overlap
        with this specific phrasing (e.g. "Duties and taxes" sharing no
        stemmed tokens with "What about Canada?") should still be included
        because it belongs to the relevant document -- `search()`/
        `_search_all()` filter these out for ranking purposes, which is
        correct there but wrong for expansion."""
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        out = []
        for chunk, raw in zip(self.chunks, sims):
            weight = AUTHORITY_WEIGHT.get(chunk.status, 0.5)
            out.append(
                RetrievedChunk(
                    chunk=chunk,
                    raw_score=float(raw),
                    weighted_score=float(raw) * weight,
                    authoritative=is_authoritative(chunk),
                )
            )
        return out

    def retrieve_for_agent(
        self, query: str, top_docs: int = 3, raw_flag_threshold: float = 0.18, max_chunks: int = 10
    ) -> dict:
        """Assemble everything the agent needs for one turn's retrieval:
        - authoritative: ALL authoritative chunks belonging to the top-N
          *documents* by best-chunk score (not just the single best chunk
          per document -- see module docstring for why that was a real bug).
          Once a document is selected, every one of its sections is
          included even if that specific section has zero term overlap
          with this exact phrasing of the query (see `_score_all_chunks`).
        - flagged: non-authoritative chunks (superseded/draft/internal) that
          scored high enough on raw similarity that the user is plausibly
          asking about them directly (e.g. quoting the migration note, or
          asking about the old policy). These are surfaced so the agent can
          acknowledge/explain them, but are explicitly marked untrusted and
          non-authoritative so they are never used to answer policy questions.
        - conflict: known-conflict info, if the query matches one.
        """
        all_results = self._search_all(query)
        auth_results = [r for r in all_results if r.authoritative]

        best_score_by_doc: dict[str, float] = {}
        for r in auth_results:
            best_score_by_doc[r.chunk.doc_id] = max(
                best_score_by_doc.get(r.chunk.doc_id, 0.0), r.weighted_score
            )
        top_doc_ids = set(sorted(best_score_by_doc, key=lambda d: best_score_by_doc[d], reverse=True)[:top_docs])

        full_scored = self._score_all_chunks(query)
        authoritative = [
            r for r in full_scored if r.authoritative and r.chunk.doc_id in top_doc_ids
        ]
        authoritative.sort(key=lambda r: r.weighted_score, reverse=True)
        authoritative = authoritative[:max_chunks]

        flagged = [
            r
            for r in all_results
            if not r.authoritative and r.raw_score >= raw_flag_threshold
        ][:3]
        conflict = self.detect_conflict(query)
        return {
            "authoritative": authoritative,
            "flagged_non_authoritative": flagged,
            "conflict": conflict,
        }


if __name__ == "__main__":
    import os

    r = Retriever(os.path.join(os.path.dirname(__file__), "..", "knowledge-base"))
    for q in [
        "How long can I return a backpack?",
        "Can I put the entire Breeze Tumbler in the dishwasher?",
        "The migration note says give everyone 60 days",
    ]:
        print("Q:", q)
        for res in r.search(q, top_k=3):
            print(
                f"  {res.chunk.filename} [{res.chunk.heading}] raw={res.raw_score:.3f} "
                f"weighted={res.weighted_score:.3f} auth={res.authoritative}"
            )
        conflict = r.detect_conflict(q)
        print("  conflict:", conflict["topic"] if conflict else None)
