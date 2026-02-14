"""
Chat Engine — RAG-powered conversational AI for gym intelligence.

Pipeline:  user message → retrieve context from KB → augment prompt → LLM → response

Supports:
  - Single-turn and multi-turn conversations
  - Streaming (SSE) responses
  - Automatic context retrieval from gym data
  - Conversation memory (last N turns)
"""
import json
import time
from pathlib import Path
from typing import Generator, Optional

import yaml

from serving.llm_client import LLMClient, create_client_from_config
from serving.knowledge_base import KnowledgeBase


ROOT = Path(__file__).resolve().parent.parent

SYSTEM_PROMPT = """\
You are **FitFlex AI**, an expert fitness and gym management assistant powered by real gym data.

Your capabilities:
- Personalized workout recommendations based on actual member demographics and performance data
- Calorie burn predictions grounded in real exercise data
- Progress analysis using body performance benchmarks
- Attendance and churn risk insights from membership patterns
- Nutrition and recovery advice backed by sports science

Guidelines:
- Always be encouraging, data-driven, and specific
- When you reference statistics, mention they come from gym data
- Give actionable advice with numbers (sets, reps, durations, calories)
- If asked about a specific member, use the data context provided
- For medical concerns, recommend consulting a healthcare professional
- Keep responses concise but thorough — aim for 2-4 paragraphs
- Use formatting (bold, bullets) for readability
"""


class ChatEngine:
    """RAG-powered chat with conversation memory."""

    def __init__(self, llm: LLMClient, kb: Optional[KnowledgeBase] = None):
        self.llm = llm
        self.kb = kb
        self.conversations: dict[str, list[dict]] = {}  # session_id → messages
        self.max_history = 10  # keep last N turns

    def _retrieve_context(self, query: str, top_k: int = 6) -> str:
        """Retrieve relevant chunks from the knowledge base."""
        if self.kb is None:
            return ""
        results = self.kb.query(query, top_k=top_k)
        if not results:
            return ""
        chunks = []
        for r in results:
            chunks.append(f"[{r['metadata'].get('category', 'data')}] {r['text']}")
        return "\n\n".join(chunks)

    def _build_prompt(
        self,
        message: str,
        context: str,
        history: list[dict] = None,
    ) -> str:
        """Build the full prompt with context and history."""
        parts = []

        if context:
            parts.append("━━━ GYM DATA CONTEXT ━━━")
            parts.append(context)
            parts.append("━━━ END CONTEXT ━━━\n")

        if history:
            parts.append("━━━ CONVERSATION HISTORY ━━━")
            for msg in history[-self.max_history:]:
                role = "User" if msg["role"] == "user" else "FitFlex AI"
                parts.append(f"{role}: {msg['content']}")
            parts.append("━━━ END HISTORY ━━━\n")

        parts.append(f"User: {message}")
        parts.append("\nRespond helpfully based on the gym data context above:")
        return "\n".join(parts)

    def chat(
        self,
        message: str,
        session_id: str = "default",
    ) -> dict:
        """Single response (non-streaming)."""
        t0 = time.perf_counter()

        # Retrieve context
        context = self._retrieve_context(message)

        # Get conversation history
        history = self.conversations.get(session_id, [])

        # Build prompt
        prompt = self._build_prompt(message, context, history)

        # Generate response
        try:
            response = self.llm.generate(prompt, system=SYSTEM_PROMPT)
        except ConnectionError as e:
            response = str(e)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Store in history
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        self.conversations[session_id].append({"role": "user", "content": message})
        self.conversations[session_id].append({"role": "assistant", "content": response})

        # Trim history
        if len(self.conversations[session_id]) > self.max_history * 2:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history * 2:]

        return {
            "response": response,
            "context_used": len(context) > 0,
            "chunks_retrieved": len(self.kb.query(message)) if self.kb else 0,
            "inference_ms": round(elapsed_ms, 1),
            "model": self.llm.model,
            "provider": self.llm.provider,
        }

    def stream_chat(
        self,
        message: str,
        session_id: str = "default",
    ) -> Generator[str, None, None]:
        """Stream tokens via SSE — yields 'data: {...}\n\n' strings."""
        t0 = time.perf_counter()

        context = self._retrieve_context(message)
        history = self.conversations.get(session_id, [])
        prompt = self._build_prompt(message, context, history)

        full_response = []

        try:
            for token in self.llm.stream(prompt, system=SYSTEM_PROMPT):
                full_response.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"
        except ConnectionError as e:
            error_msg = str(e)
            yield f"data: {json.dumps({'token': error_msg})}\n\n"
            full_response.append(error_msg)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Store in history
        response_text = "".join(full_response)
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        self.conversations[session_id].append({"role": "user", "content": message})
        self.conversations[session_id].append({"role": "assistant", "content": response_text})

        # Trim
        if len(self.conversations[session_id]) > self.max_history * 2:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history * 2:]

        # Final metadata event
        yield f"data: {json.dumps({'done': True, 'inference_ms': round(elapsed_ms, 1), 'model': self.llm.model})}\n\n"

    def clear_history(self, session_id: str = "default"):
        self.conversations.pop(session_id, None)

    def get_suggestions(self) -> list[str]:
        """Return contextual conversation starters."""
        return [
            "What workout should a beginner do on their first day?",
            "How many calories does a typical HIIT session burn?",
            "What's the average member profile at this gym?",
            "Which members are at risk of leaving the gym?",
            "Give me a 4-week workout plan for muscle building",
            "How does heart rate affect calorie burn?",
            "What are the top performance benchmarks in this gym?",
            "Compare yoga vs strength training for weight loss",
        ]


def create_chat_engine(config: dict = None) -> ChatEngine:
    """Factory — build engine from config.yaml."""
    if config is None:
        with open(ROOT / "configs" / "config.yaml") as f:
            config = yaml.safe_load(f)

    llm = create_client_from_config(config)

    # Try to load knowledge base
    kb = None
    kb_path = ROOT / "models" / "artifacts" / "knowledge_base"
    if (kb_path / "documents.json").exists():
        try:
            kb = KnowledgeBase.load(kb_path)
            print(f"[CHAT] Knowledge base loaded: {len(kb.documents)} chunks")
        except Exception as e:
            print(f"[CHAT] KB load failed: {e}")
    else:
        print("[CHAT] No knowledge base found — run `python train.py` first")

    return ChatEngine(llm, kb)
