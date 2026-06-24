# Shaik_Sohib_Arafath_PersonaRAG_KaStack
# Requirements Document

## 1. Application Overview

**Application Name**: Multi-Module Persona & Intent Analysis Platform

**Description**: A single-page web application providing four core modules: Adaptive Persona Engine for tracking user behavior drift, Offline Intent Classifier for message categorization, RAG Conflict Resolver for knowledge base query resolution, and System Design Visualization for architecture display. The application uses tab navigation to switch between modules and includes pre-loaded demo data for immediate use without external API dependencies.

## 2. Users and Usage Scenarios

**Target Users**: Developers, product managers, and researchers working on conversational AI systems, persona analysis, or knowledge management.

**Core Usage Scenarios**:
- Analyze how user personas evolve over time through conversation history
- Quickly classify user messages into predefined intent categories
- Resolve conflicting information from multiple knowledge sources
- Understand system architecture for data synchronization and conflict resolution

## 3. Page Structure and Functional Description

### Page Structure
```
Main Application
├── Tab Navigation Bar
├── Part 1: Adaptive Persona Engine
├── Part 2: Offline Intent Classifier
├── Part 3: RAG Conflict Resolver
└── Part 4: System Design Visualization
```

### 3.1 Main Application Page

**Tab Navigation Bar**
- Display four tabs: \"Adaptive Persona Engine\", \"Offline Intent Classifier\", \"RAG Conflict Resolver\", \"System Design Visualization\"
- Switch between modules by clicking tabs
- Highlight active tab

### 3.2 Part 1: Adaptive Persona Engine

**Input Section**
- Accept persona JSON input containing user profile and conversation history across multiple days
- JSON structure includes: user ID, conversation records with timestamps, messages, and metadata

**Persona Drift Detection**
- Analyze conversation history to identify mood and tone changes across days
- Track changes at daily granularity (not overall aggregation)
- Identify triggers causing each drift (topic, event, or person mentioned)

**Visual Timeline Display**
- Show timeline with day markers (e.g., Day 1, Day 4, Day 7)
- Display mood/tone characteristics for each day (e.g., curious & formal, casual & frustrated, playful)
- Highlight detected triggers for each drift point
- Use visual indicators (colors, icons) to represent different mood/tone states

**Demo Data**
- Pre-load sample persona JSON with multi-day conversation history
- Include varied mood/tone patterns and identifiable triggers

### 3.3 Part 2: Offline Intent Classifier

**Message Input**
- Provide text input field for user to enter messages
- Support real-time input

**Intent Classification**
- Classify messages into categories: reminder, emotional-support, action-item, small-talk, unknown
- Use rule-based and keyword-based classification logic
- No external API calls or machine learning model training
- Display classification result instantly after input

**Result Display**
- Show classified intent category
- Display matching keywords or rules that triggered the classification

**Demo Data**
- Provide sample messages demonstrating each intent category

### 3.4 Part 3: RAG Conflict Resolver

**Knowledge Base Setup**
- Simulate knowledge base with multiple topic checkpoints
- Each checkpoint contains text chunks with metadata (topic, timestamp, emotional weight)

**Query Input**
- Accept user questions (e.g., \"Did I mention anything about my sister?\")
- Support natural language queries

**Chunk Retrieval and Ranking**
- Find relevant chunks across all topics
- Rank chunks by recency and emotional weight
- Identify contradictions between chunks

**Result Display**
- Show retrieved chunks with ranking scores
- Flag contradictory information
- Display merged coherent answer resolving conflicts
- Explain resolution logic

**Demo Data**
- Pre-load knowledge base with sample topics and checkpoints
- Include contradictory information for demonstration

### 3.5 Part 4: System Design Visualization

**Architecture Diagram**
- Display interactive visual diagram of sync architecture
- Show components: on-device storage, sync mechanisms, conflict resolution logic
- Indicate what data syncs vs stays local
- Use nodes and arrows to represent data flow and relationships

**Interaction**
- Allow users to click nodes for detailed explanations
- Highlight related components when hovering over elements

**Demo Data**
- Pre-configured architecture diagram with all components

## 4. Business Rules and Logic

### 4.1 Persona Drift Detection Logic
- Analyze conversation messages day by day
- Extract mood/tone indicators from message content and metadata
- Compare consecutive days to identify significant changes
- Identify triggers by analyzing topics, events, or persons mentioned around drift points

### 4.2 Intent Classification Rules
- **Reminder**: Keywords include \"remind\", \"don't forget\", \"remember to\", time expressions
- **Emotional-support**: Keywords include \"feel\", \"sad\", \"happy\", \"worried\", emotional expressions
- **Action-item**: Keywords include \"need to\", \"must\", \"should\", \"task\", action verbs
- **Small-talk**: Keywords include \"how are you\", \"weather\", \"hello\", casual greetings
- **Unknown**: Messages not matching any above categories
- Apply rules in priority order, assign first matching category

### 4.3 RAG Conflict Resolution Logic
- Retrieve chunks matching query keywords
- Calculate ranking score: (recency_score * 0.6) + (emotional_weight * 0.4)
- Detect contradictions by comparing semantic content of top-ranked chunks
- Merge information by prioritizing more recent and emotionally weighted chunks
- Generate coherent answer explaining resolved context

### 4.4 Data Handling
- All modules use pre-loaded demo data stored locally
- No external API calls or backend services required
- Data persists only during current session

## 5. Exceptions and Edge Cases

| Scenario | Handling |
|----------|----------|
| Invalid persona JSON format | Display error message, prompt user to check JSON structure |
| Empty message input in Intent Classifier | Disable classification, show prompt to enter message |
| No matching chunks in RAG Resolver | Display \"No relevant information found\" message |
| Query with no contradictions | Show retrieved chunks without conflict flags |
| Tab switching during data processing | Preserve current state, allow seamless switching |

## 6. Acceptance Criteria

1. User opens the application and sees the tab navigation bar with four tabs
2. User clicks \"Adaptive Persona Engine\" tab and views pre-loaded persona timeline showing mood/tone changes across days with identified triggers
3. User clicks \"Offline Intent Classifier\" tab, enters a message, and sees the classified intent category displayed instantly
4. User clicks \"RAG Conflict Resolver\" tab, enters a query, and views retrieved chunks with conflict flags and merged answer
5. User clicks \"System Design Visualization\" tab and views the interactive architecture diagram showing sync components and data flow

## 7. Out of Scope for This Release

- User authentication and login system
- Backend data persistence or database integration
- External API integration (OpenAI, Gemini, or other AI services)
- Machine learning model training or fine-tuning
- Multi-language support beyond English
- Export functionality for analysis results
- Customizable classification rules or persona analysis parameters
- Real-time collaboration features
- Mobile app version
- Advanced visualization customization options
