#!/usr/bin/env python3
"""
Conflict Resolution in RAG
A hard retrieval problem: user asks about "my sister" but mentions appear
across multiple topic checkpoints with contradictory context.

Resolver pipeline:
  1. Retrieve all chunks matching the query.
  2. Score each chunk by recency + emotional weight.
  3. Detect contradictions via sentiment/opposite-keyword analysis.
  4. Merge into a coherent, temporally-aware answer.
"""

import json
import math
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
from datetime import datetime


@dataclass
class Chunk:
    id: str
    text: str
    topic: str
    checkpoint: str
    date: str          # ISO date string
    emotional_weight: float   # 0.0 = neutral, 1.0 = highly emotional
    sentiment: float         # -1.0 negative → +1.0 positive


class RAGConflictResolver:
    """
    Resolves contradictions across retrieved chunks.
    """

    # Keywords that signal polar-opposite claims
    CONTRADICTION_PAIRS = [
        ("love", "hate"),
        ("good", "bad"),
        ("great", "terrible"),
        ("happy", "sad"),
        ("close", "distant"),
        ("talk", "ignore"),
        ("support", "betray"),
        ("trust", "distrust"),
        ("together", "apart"),
        ("help", "hurt"),
        ("like", "dislike"),
        ("friend", "enemy"),
        ("fair", "unfair"),
        ("wonderful", "awful"),
        ("excited", "dread"),
    ]

    def __init__(self, chunks: List[Chunk], decay_days: float = 7.0):
        self.chunks = chunks
        self.decay_days = decay_days
        self.today = datetime.utcnow().date()

    def _parse_date(self, d: str) -> datetime:
        return datetime.strptime(d, "%Y-%m-%d").date()

    def _recency_score(self, chunk: Chunk) -> float:
        days_old = (self.today - self._parse_date(chunk.date)).days
        return math.exp(-days_old / self.decay_days)

    def _rank_chunks(self, query: str) -> List[Chunk]:
        """Rank by combined recency + emotional weight + lexical overlap."""
        q_tokens = set(re.findall(r"\b\w+\b", query.lower()))
        scored = []
        for c in self.chunks:
            overlap = len(q_tokens & set(re.findall(r"\b\w+\b", c.text.lower())))
            lexical = overlap / max(len(q_tokens), 1)
            score = (
                0.35 * self._recency_score(c) +
                0.35 * c.emotional_weight +
                0.30 * lexical
            )
            scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored]

    def _detect_contradictions(self, top_chunks: List[Chunk]) -> List[Dict[str, Any]]:
        """
        Flag pairs of chunks that contain semantically opposite claims.
        Uses keyword polarity + sentiment inversion as heuristics.
        """
        flags = []
        n = len(top_chunks)
        for i in range(n):
            for j in range(i + 1, n):
                a = top_chunks[i]
                b = top_chunks[j]
                contradictions = []

                # Keyword polarity check
                a_lower = a.text.lower()
                b_lower = b.text.lower()
                for w1, w2 in self.CONTRADICTION_PAIRS:
                    if (w1 in a_lower and w2 in b_lower) or (w2 in a_lower and w1 in b_lower):
                        contradictions.append(f"'{w1}' vs '{w2}'")

                # Sentiment inversion check
                if a.sentiment * b.sentiment < -0.2 and abs(a.sentiment - b.sentiment) > 0.8:
                    contradictions.append("sentiment inversion")

                if contradictions:
                    flags.append({
                        "chunk_a": a.id,
                        "chunk_b": b.id,
                        "reasons": contradictions,
                        "recommendation": "resolve_by_recency" if self._recency_score(a) > self._recency_score(b) else "resolve_by_recency",
                    })
        return flags

    def _merge_answer(self, top_chunks: List[Chunk], flags: List[Dict[str, Any]]) -> str:
        """
        Build a temporally-aware merged narrative.
        - Sort by date ascending.
        - For flagged contradictions, prefer the more recent chunk.
        """
        # Identify suppressed chunks from contradictions
        suppressed = set()
        for f in flags:
            # Prefer the more recent chunk, suppress the older one
            a = next(c for c in top_chunks if c.id == f["chunk_a"])
            b = next(c for c in top_chunks if c.id == f["chunk_b"])
            older = a if self._parse_date(a.date) < self._parse_date(b.date) else b
            suppressed.add(older.id)

        accepted = [c for c in top_chunks if c.id not in suppressed]
        accepted.sort(key=lambda c: self._parse_date(c.date))

        if not accepted:
            return "No consistent information available."

        parts = []
        parts.append(f"You mentioned your sister across {len(top_chunks)} topic checkpoints. "
                     f"Here is the coherent narrative, resolved for contradictions:")
        parts.append("")

        for c in accepted:
            date_str = c.date
            sentiment_word = "positive" if c.sentiment > 0.2 else "negative" if c.sentiment < -0.2 else "neutral"
            parts.append(f"  • {date_str} ({c.topic}): {c.text} "
                         f"[sentiment: {sentiment_word}, emotional weight: {c.emotional_weight:.1f}]")

        if suppressed:
            parts.append("")
            parts.append("Suppressed contradictory entries:")
            for sid in suppressed:
                sc = next(c for c in top_chunks if c.id == sid)
                parts.append(f"  – {sc.date}: {sc.text}")

        return "\n".join(parts)

    def resolve(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        ranked = self._rank_chunks(query)
        top = ranked[:top_k]
        flags = self._detect_contradictions(top)
        merged = self._merge_answer(top, flags)

        return {
            "query": query,
            "chunks_considered": len(self.chunks),
            "chunks_used": len(top),
            "ranking": [
                {
                    "id": c.id,
                    "topic": c.topic,
                    "date": c.date,
                    "recency_score": round(self._recency_score(c), 3),
                    "emotional_weight": c.emotional_weight,
                    "sentiment": c.sentiment,
                    "text": c.text,
                }
                for c in top
            ],
            "contradiction_flags": flags,
            "merged_answer": merged,
        }


def sample_sister_chunks() -> List[Chunk]:
    """Synthetic retrieval result for the 'sister' query with contradictions."""
    return [
        Chunk(
            id="chk-01",
            text="My sister and I are super close; she's my best friend.",
            topic="family_relationships",
            checkpoint="early_week",
            date="2026-06-17",
            emotional_weight=0.8,
            sentiment=0.9,
        ),
        Chunk(
            id="chk-02",
            text="Had a terrible argument with my sister. I feel betrayed.",
            topic="conflict",
            checkpoint="mid_week",
            date="2026-06-19",
            emotional_weight=0.9,
            sentiment=-0.8,
        ),
        Chunk(
            id="chk-03",
            text="My sister helped me move apartments. She's wonderful.",
            topic="life_events",
            checkpoint="late_week",
            date="2026-06-22",
            emotional_weight=0.7,
            sentiment=0.85,
        ),
        Chunk(
            id="chk-04",
            text="Sometimes I feel distant from my sister lately.",
            topic="reflection",
            checkpoint="end_week",
            date="2026-06-23",
            emotional_weight=0.5,
            sentiment=-0.3,
        ),
        Chunk(
            id="chk-05",
            text="My sister ignored my texts all day. It hurts.",
            topic="conflict",
            checkpoint="end_week",
            date="2026-06-23",
            emotional_weight=0.85,
            sentiment=-0.75,
        ),
    ]


def main():
    print("=" * 70)
    print("RAG CONFLICT RESOLVER")
    print("=" * 70)

    chunks = sample_sister_chunks()
    resolver = RAGConflictResolver(chunks, decay_days=7.0)
    result = resolver.resolve("Did I mention anything about my sister?", top_k=5)

    print(f"Query       : {result['query']}")
    print(f"Considered  : {result['chunks_considered']} chunks")
    print(f"Top-K used  : {result['chunks_used']}")
    print()

    print("RANKED CHUNKS")
    print("-" * 70)
    for r in result["ranking"]:
        print(f"  [{r['id']}] {r['date']} | topic={r['topic']} | "
              f"recency={r['recency_score']} | emotion={r['emotional_weight']} | "
              f"sentiment={r['sentiment']}")
        print(f"      → {r['text']}")

    print()
    print("CONTRADICTION FLAGS")
    print("-" * 70)
    if result["contradiction_flags"]:
        for f in result["contradiction_flags"]:
            print(f"  {f['chunk_a']} <-> {f['chunk_b']}: {', '.join(f['reasons'])}")
    else:
        print("  No contradictions detected.")

    print()
    print("MERGED ANSWER")
    print("-" * 70)
    print(result["merged_answer"])
    print()

    print("FULL JSON")
    print("-" * 70)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
