# Sync Architecture Design Document

## 1. Overview

This document outlines the on-device sync architecture for the Adaptive AI Companion.
The system must keep user data (persona, memories, messages) consistent across
multiple devices while respecting privacy, bandwidth, and battery constraints.

---

## 2. On-Device Storage

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Hot Cache** | In-memory LRU (Redis-compatible) | Active session context, last 50 messages |
| **Structured Store** | SQLite (WAL mode) | Persona vectors, intent labels, metadata indexes |
| **Vector Store** | sqlite-vec / faiss-lite | Embedding indexes for RAG retrieval |
| **Blob Store** | Encrypted flat files (AES-256-GCM) | Attachments, voice memos, model weights |
| **Journal** | Append-only log | Immutable event stream for conflict resolution |

### What Stays Local (Never Syncs)
- Raw voice memos & large attachments (only hash + metadata sync)
- Model weights & classifier binaries (device-specific builds)
- Session hot-cache (rebuilt on launch)
- Temporary inference embeddings

### What Syncs
- **Persona checkpoints** — daily mood/tone snapshots (small JSON, ~2 KB)
- **Message metadata** — sender, timestamp, intent label, emotional weight
- **Conversation summaries** — condensed RAG chunks (not full transcripts)
- **Intent classifier outputs** — cached predictions (re-validated on-device)
- **User preferences** — opt-in flags, reminder schedules

---

## 3. Sync Rules & Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLOUD GATEWAY                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Sync Broker  │  │ Conflict     │  │ Backup Vault         │  │
│  │ (MQTT/WebSkt)│  │ Resolver     │  │ (encrypted S3 equiv) │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────────────────┘  │
└─────────┼─────────────────┼───────────────────────────────────────┘
          │                 │
          ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DEVICE MESH                                 │
│  ┌────────────┐        ┌────────────┐        ┌────────────┐    │
│  │  Phone     │◄─────►│  Tablet    │◄─────►│  Desktop   │    │
│  │  SQLite    │  BLE   │  SQLite    │  LAN   │  SQLite    │    │
│  │  Journal   │        │  Journal   │        │  Journal   │    │
│  └─────┬──────┘        └─────┬──────┘        └─────┬──────┘    │
│        │                     │                     │            │
│        └─────────────────────┴─────────────────────┘            │
│                         │                                       │
│                         ▼                                       │
│                 ┌───────────────┐                                 │
│                 │   Sync Engine │  (delta sync, CRDT-lite)      │
│                 │   · Gossip    │                                 │
│                 │   · Bloom     │                                 │
│                 │   · Merkle    │                                 │
│                 └───────────────┘                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Sync Frequency
| Data Type | Trigger | Max Latency |
|-----------|---------|-------------|
| Persona drift | Every new checkpoint | 5 min (push) |
| Message metadata | After each turn | Real-time (push) |
| Summaries | Batch nightly | 24 h (pull) |
| Preferences | On change | Immediate (push) |
| Full backup | Weekly + on logout | On-demand |

---

## 4. Conflict Resolution

Conflicts arise when two devices edit the same persona checkpoint or
message label independently.

### Resolution Strategy Matrix

| Conflict Type | Rule | Winner |
|---------------|------|--------|
| **Persona mood** | Higher emotional weight wins | Emotional peak overrides bland re-label |
| **Tone label** | Most recent explicit user correction wins | User feedback > model inference |
| **Intent prediction** | Confidence-weighted merge; flag if <0.6 | Promote to "unknown" if low |
| **Duplicate message** | Hash + timestamp dedup | First arrival (idempotent) |
| **Contradictory RAG chunk** | Recency × emotional weight ranking | See `rag_conflict_resolver.py` |

### CRDT-Lite for Counters & Sets
- **Persona topics** are stored as a Grow-Only Set (G-Set): union of all devices.
- **Emotional weight** uses a Last-Writer-Wins (LWW) register with vector clocks.
- **Daily valence** is an PN-Counter (positive/negative increments merge by summation).

### Fallback: Cloud Arbiter
If vector-clock divergence exceeds 3 hops, the Cloud Conflict Resolver:
1. Replays both device journals.
2. Re-ranks by recency + emotional weight.
3. Publishes a canonical `resolution_event` to all devices.

---

## 5. Security & Privacy

- **End-to-end encryption**: All sync payloads are encrypted with device-derived keys (X3DH).
- **Zero-knowledge gateway**: Cloud broker sees only encrypted blobs and Merkle roots.
- **Local-first**: App is fully functional offline; sync is opportunistic, not mandatory.

---

## 6. Failure Modes

| Scenario | Mitigation |
|----------|------------|
| Device offline > 7 days | Stale flag; full delta computed on reconnect |
| Large vector index mismatch | Rebuild from journal; incremental faiss-lite merge |
| Model version skew | Version gate; incompatible devices sync metadata only |
| Journal corruption | Trim to last valid checksum; recover from backup vault |

---

*Document version 1.0 — 2026-06-24*
