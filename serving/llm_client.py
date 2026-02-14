"""
Multi-provider LLM client — Ollama (local GPU) | Groq (free cloud) | OpenAI.

Ollama is the default: free, private, runs on your RTX 2060.
Groq provides a generous free tier with fast inference.
OpenAI is the premium fallback.

Usage:
    client = LLMClient(provider="ollama", model="llama3.2:3b")
    response = client.generate("What workout should I do today?", system="You are a fitness coach.")
    for token in client.stream("Tell me about HIIT", system="You are a fitness coach."):
        print(token, end="", flush=True)
"""
import json
import os
import requests
from typing import Generator, Optional


OLLAMA_DEFAULT_URL = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "llama3.2:3b"


class LLMClient:
    """Unified LLM interface with provider abstraction."""

    def __init__(
        self,
        provider: str = "ollama",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.provider = provider.lower()

        if self.provider == "ollama":
            # Env vars should be able to override config.yaml defaults in production.
            env_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_URL")
            env_model = os.getenv("OLLAMA_MODEL")
            self.base_url = env_url or base_url or OLLAMA_DEFAULT_URL
            self.model = env_model or model or OLLAMA_DEFAULT_MODEL
            self.api_key = None
        elif self.provider == "groq":
            self.base_url = base_url or "https://api.groq.com/openai/v1"
            self.model = model or "llama-3.1-8b-instant"
            self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        elif self.provider == "openai":
            self.base_url = base_url or "https://api.openai.com/v1"
            self.model = model or "gpt-4o-mini"
            self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    # ── Availability ──────────────────────────────────────────
    def is_available(self) -> bool:
        try:
            if self.provider == "ollama":
                r = requests.get(f"{self.base_url}/api/tags", timeout=5)
                return r.status_code == 200
            return bool(self.api_key)
        except Exception:
            return False

    def list_models(self) -> list[str]:
        if self.provider != "ollama":
            return [self.model]
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    def ensure_model(self) -> bool:
        """Pull the configured model if not already present (Ollama only)."""
        if self.provider != "ollama":
            return True
        models = self.list_models()
        # check if model or model without tag exists
        if any(self.model in m for m in models):
            return True
        print(f"[LLM] Pulling {self.model} … this may take a few minutes on first run.")
        try:
            r = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model, "stream": False},
                timeout=600,
            )
            return r.status_code == 200
        except Exception as e:
            print(f"[LLM] Pull failed: {e}")
            return False

    # ── Synchronous generation ────────────────────────────────
    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        if self.provider == "ollama":
            return self._ollama_generate(prompt, system, temperature)
        return self._openai_generate(prompt, system, temperature, max_tokens)

    def _ollama_generate(self, prompt: str, system: str, temperature: float) -> str:
        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": system,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": 1024},
                },
                timeout=120,
            )
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except requests.ConnectionError:
            raise ConnectionError(
                "Ollama is not running. Start it:\n"
                "  1. Install: https://ollama.com/download\n"
                "  2. Run: ollama serve\n"
                "  3. Pull model: ollama pull llama3.2:3b"
            )

    def _openai_generate(
        self, prompt: str, system: str, temperature: float, max_tokens: int
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    # ── Streaming generation ──────────────────────────────────
    def stream(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """Yield tokens one by one — powers the SSE chat endpoint."""
        if self.provider == "ollama":
            yield from self._ollama_stream(prompt, system, temperature)
        else:
            yield from self._openai_stream(prompt, system, temperature)

    def _ollama_stream(
        self, prompt: str, system: str, temperature: float
    ) -> Generator[str, None, None]:
        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": system,
                    "stream": True,
                    "options": {"temperature": temperature, "num_predict": 1024},
                },
                stream=True,
                timeout=120,
            )
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    data = json.loads(line)
                    if token := data.get("response"):
                        yield token
                    if data.get("done"):
                        break
        except requests.ConnectionError:
            yield "[Error] Ollama is not running. Install from https://ollama.com and run: ollama serve"

    def _openai_stream(
        self, prompt: str, system: str, temperature: float
    ) -> Generator[str, None, None]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 1024,
                    "stream": True,
                },
                stream=True,
                timeout=60,
            )
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                text = line.decode("utf-8", errors="ignore")
                if text.startswith("data: "):
                    text = text[6:]
                if text.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(text)
                    delta = chunk["choices"][0].get("delta", {})
                    if token := delta.get("content"):
                        yield token
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
        except Exception as e:
            yield f"[Error] {e}"

    # ── Chat with history ─────────────────────────────────────
    def chat(
        self,
        messages: list[dict],
        system: str = "",
        temperature: float = 0.7,
    ) -> str:
        """Multi-turn chat. Messages: [{"role":"user|assistant","content":"..."}]"""
        if self.provider == "ollama":
            chat_msgs = []
            if system:
                chat_msgs.append({"role": "system", "content": system})
            chat_msgs.extend(messages)
            try:
                r = requests.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": chat_msgs,
                        "stream": False,
                        "options": {"temperature": temperature},
                    },
                    timeout=120,
                )
                r.raise_for_status()
                return r.json().get("message", {}).get("content", "").strip()
            except requests.ConnectionError:
                raise ConnectionError("Ollama is not running.")
        else:
            # OpenAI-compatible
            api_msgs = []
            if system:
                api_msgs.append({"role": "system", "content": system})
            api_msgs.extend(messages)
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": api_msgs,
                    "temperature": temperature,
                    "max_tokens": 1024,
                },
                timeout=60,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()


def create_client_from_config(config: dict) -> LLMClient:
    """Factory — reads llm section from config.yaml."""
    llm_cfg = config.get("llm", {})
    return LLMClient(
        provider=llm_cfg.get("provider", "ollama"),
        model=llm_cfg.get("model"),
        base_url=llm_cfg.get("base_url"),
        api_key=llm_cfg.get("api_key"),
    )
