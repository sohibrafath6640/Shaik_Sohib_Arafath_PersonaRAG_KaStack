#!/usr/bin/env python3
"""
Adaptive Persona Engine — Persona Drift Detector
Tracks how a user's mood and tone evolve across days, detects triggers,
and outputs a drift timeline.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import datetime


@dataclass
class DailyPersona:
    day: int
    date: str
    mood: str
    tone: str
    topics: List[str]
    events: List[str]
    people: List[str]
    notes: str


class PersonaDriftDetector:
    """Detects shifts in mood/tone across a user's conversation history."""

    MOOD_VALENCE = {
        "curious": 0.6,
        "frustrated": -0.7,
        "playful": 0.5,
        "formal": 0.1,
        "casual": 0.0,
        "anxious": -0.6,
        "excited": 0.8,
        "neutral": 0.0,
        "sad": -0.8,
        "angry": -0.9,
        "happy": 0.9,
        "hopeful": 0.7,
    }

    TONE_VALENCE = {
        "formal": 0.1,
        "casual": 0.0,
        "playful": 0.5,
        "serious": -0.2,
        "assertive": 0.2,
        "submissive": -0.3,
        "sarcastic": -0.4,
        "warm": 0.7,
        "cold": -0.6,
    }

    def __init__(self, history: List[DailyPersona]):
        self.history = sorted(history, key=lambda x: x.day)
        self.drifts: List[Dict[str, Any]] = []

    def _valence(self, mood: str, tone: str) -> float:
        return self.MOOD_VALENCE.get(mood, 0.0) + self.TONE_VALENCE.get(tone, 0.0)

    def _detect_trigger(self, prev: DailyPersona, curr: DailyPersona) -> Dict[str, Any]:
        """Heuristic trigger detection based on newly introduced topics/events/people."""
        triggers = []

        new_topics = set(curr.topics) - set(prev.topics)
        new_events = set(curr.events) - set(prev.events)
        new_people = set(curr.people) - set(prev.people)

        for t in new_topics:
            triggers.append({"type": "topic", "value": t})
        for e in new_events:
            triggers.append({"type": "event", "value": e})
        for p in new_people:
            triggers.append({"type": "person", "value": p})

        # Fallback: if no explicit new entity, derive from notes keywords
        if not triggers:
            keywords = ["fight", "argue", "promotion", "breakup", "birthday",
                        "deadline", "success", "failure", "trip", "accident",
                        "rejection", "approval", "sick", "recover"]
            lowered = curr.notes.lower()
            for kw in keywords:
                if kw in lowered:
                    triggers.append({"type": "event", "value": kw})
                    break

        return {
            "new_topics": list(new_topics),
            "new_events": list(new_events),
            "new_people": list(new_people),
            "triggers": triggers if triggers else [{"type": "unknown", "value": "gradual shift"}]
        }

    def detect(self) -> List[Dict[str, Any]]:
        """Compare each consecutive day and record significant drifts."""
        for i in range(1, len(self.history)):
            prev = self.history[i - 1]
            curr = self.history[i]

            prev_v = self._valence(prev.mood, prev.tone)
            curr_v = self._valence(curr.mood, curr.tone)
            delta = curr_v - prev_v

            # Drift threshold: any shift >= 0.6 in combined valence
            if abs(delta) >= 0.6:
                trigger_info = self._detect_trigger(prev, curr)
                self.drifts.append({
                    "from_day": prev.day,
                    "to_day": curr.day,
                    "from_state": {"mood": prev.mood, "tone": prev.tone},
                    "to_state": {"mood": curr.mood, "tone": curr.tone},
                    "delta": round(delta, 2),
                    "triggers": trigger_info["triggers"],
                    "new_entities": {
                        "topics": trigger_info["new_topics"],
                        "events": trigger_info["new_events"],
                        "people": trigger_info["new_people"],
                    }
                })

        return self.drifts

    def timeline(self) -> List[Dict[str, Any]]:
        """Return a flat timeline of persona state per day with drift markers."""
        drift_days = {d["to_day"] for d in self.drifts}
        tl = []
        for entry in self.history:
            tl.append({
                "day": entry.day,
                "date": entry.date,
                "mood": entry.mood,
                "tone": entry.tone,
                "valence": round(self._valence(entry.mood, entry.tone), 2),
                "drift_detected": entry.day in drift_days,
                "topics": entry.topics,
                "events": entry.events,
                "people": entry.people,
            })
        return tl

    def summary(self) -> Dict[str, Any]:
        """High-level summary of persona evolution."""
        if not self.drifts:
            return {
                "total_days": len(self.history),
                "drift_count": 0,
                "overall_arc": "stable",
                "timeline": self.timeline()
            }

        first = self.history[0]
        last = self.history[-1]
        overall_delta = self._valence(last.mood, last.tone) - self._valence(first.mood, first.tone)

        arc = "stable"
        if overall_delta > 1.0:
            arc = "warming"
        elif overall_delta < -1.0:
            arc = "cooling"
        elif len(self.drifts) >= 2:
            arc = "volatile"

        return {
            "total_days": len(self.history),
            "drift_count": len(self.drifts),
            "overall_arc": arc,
            "overall_delta": round(overall_delta, 2),
            "drifts": self.drifts,
            "timeline": self.timeline()
        }


def sample_persona_json() -> List[Dict[str, Any]]:
    """Return a sample Round-1 persona JSON for demonstration."""
    return [
        {
            "day": 1,
            "date": "2026-06-17",
            "mood": "curious",
            "tone": "formal",
            "topics": ["productivity", "career"],
            "events": [],
            "people": [],
            "notes": "Asked about time management frameworks and how to structure deep work blocks."
        },
        {
            "day": 2,
            "date": "2026-06-18",
            "mood": "curious",
            "tone": "formal",
            "topics": ["productivity", "career"],
            "events": [],
            "people": ["manager"],
            "notes": "Follow-up on OKRs and quarterly review prep."
        },
        {
            "day": 3,
            "date": "2026-06-19",
            "mood": "neutral",
            "tone": "formal",
            "topics": ["productivity", "career"],
            "events": ["quarterly review"],
            "people": ["manager"],
            "notes": "Quarterly review happened; feedback was mixed."
        },
        {
            "day": 4,
            "date": "2026-06-20",
            "mood": "frustrated",
            "tone": "casual",
            "topics": ["career", "conflict"],
            "events": ["quarterly review"],
            "people": ["manager", "sister"],
            "notes": "Venting about the review. Called my sister and she agreed the feedback was unfair."
        },
        {
            "day": 5,
            "date": "2026-06-21",
            "mood": "anxious",
            "tone": "casual",
            "topics": ["conflict", "health"],
            "events": ["quarterly review"],
            "people": ["sister"],
            "notes": "Couldn't sleep. Talking to my sister helps but I'm still worried."
        },
        {
            "day": 6,
            "date": "2026-06-22",
            "mood": "neutral",
            "tone": "casual",
            "topics": ["health", "hobbies"],
            "events": [],
            "people": [],
            "notes": "Tried a new running route. Feeling a bit better."
        },
        {
            "day": 7,
            "date": "2026-06-23",
            "mood": "happy",
            "tone": "playful",
            "topics": ["hobbies", "social"],
            "events": ["game night"],
            "people": ["sister", "friends"],
            "notes": "Game night with friends and my sister! So much fun, laughed a lot."
        },
    ]


def main():
    raw = sample_persona_json()
    history = [DailyPersona(**item) for item in raw]
    detector = PersonaDriftDetector(history)
    detector.detect()
    result = detector.summary()

    print("=" * 60)
    print("PERSONA DRIFT DETECTION REPORT")
    print("=" * 60)
    print(f"Total days tracked : {result['total_days']}")
    print(f"Significant drifts  : {result['drift_count']}")
    print(f"Overall arc        : {result['overall_arc']}")
    print(f"Overall delta      : {result.get('overall_delta', 0)}")
    print()

    print("DRIFT TIMELINE")
    print("-" * 60)
    for entry in result["timeline"]:
        marker = "🔺" if entry["drift_detected"] else " "
        print(f"Day {entry['day']:>2} | {entry['date']} {marker} "
              f"mood={entry['mood']:>10}  tone={entry['tone']:>10}  "
              f"valence={entry['valence']:>5.2f}")

    if result["drift_count"] > 0:
        print()
        print("DETECTED DRIFTS & TRIGGERS")
        print("-" * 60)
        for drift in result["drifts"]:
            print(f"  Day {drift['from_day']} → Day {drift['to_day']}: "
                  f"{drift['from_state']['mood']} & {drift['from_state']['tone']} "
                  f"→ {drift['to_state']['mood']} & {drift['to_state']['tone']} "
                  f"(Δ {drift['delta']})")
            for t in drift["triggers"]:
                print(f"      Trigger ({t['type']}): {t['value']}")
            if drift["new_entities"]["topics"]:
                print(f"      New topics: {drift['new_entities']['topics']}")
            if drift["new_entities"]["people"]:
                print(f"      New people: {drift['new_entities']['people']}")

    print()
    print("FULL JSON OUTPUT")
    print("-" * 60)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
