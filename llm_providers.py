#!/usr/bin/env python3
"""
METATRON - llm_providers.py
Abstraction layer for multiple LLM providers (Strategy Pattern)
Supports: Ollama (local), OpenAI (GPT-4), Anthropic (Claude), Azure OpenAI (Copilot)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import os
import re
import requests
import json
from datetime import datetime
import time

import httpx
import openai
from openai import OpenAI


MODEL_CONTEXT_WINDOWS = {
    # OpenAI
    "gpt-3.5-turbo": 16385,
    "gpt-3.5-turbo-16k": 16385,
    "gpt-4": 8192,
    "gpt-4-0613": 8192,
    "gpt-4-turbo": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    # Anthropic
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    "claude-3-5-sonnet-20240620": 200000,
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-5-sonnet-latest": 200000,
    "claude-3-5-haiku-20241022": 200000,
    "claude-3-5-haiku-latest": 200000,
    # Ollama / local models used by this project
    "metatron-qwen": 16384,
    "huihui_ai/qwen3.5-abliterated:9b": 16384,
    "huihui_ai/qwen3.5-abliterated:4b": 16384,
}

MODEL_CONTEXT_WINDOW_ALIASES = {
    "gpt-4o-mini": 128000,
    "gpt-4o": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
    "claude-3-5-haiku": 200000,
    "claude-3-5-sonnet": 200000,
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
    "claude": 200000,
    "metatron-qwen": 16384,
    "huihui_ai/qwen3.5-abliterated": 16384,
    "qwen3.5": 16384,
}

# Backward compatibility for older references in the codebase.
OPENAI_CONTEXT_WINDOWS = MODEL_CONTEXT_WINDOWS


def infer_model_context_window(model: str, default: int = 8192) -> int:
    """Infer a safe context window for supported model names and deployments."""
    normalized_model = (model or "").strip().lower()
    if not normalized_model:
        return default

    if normalized_model in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[normalized_model]

    for prefix, context_window in MODEL_CONTEXT_WINDOW_ALIASES.items():
        if normalized_model.startswith(prefix):
            return context_window

    return default


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are METATRON, an elite AI penetration testing assistant running on Parrot OS.
You are precise, technical, and direct. No fluff.

You have access to real tools. To use them, write tags in your response:

  [TOOL: nmap -sV 192.168.1.1]       → runs nmap or any CLI tool
  [SEARCH: CVE-2021-44228 exploit]   → searches the web via DuckDuckGo

Rules:
- Always analyze scan data thoroughly before suggesting exploits
- List vulnerabilities with: name, severity (critical/high/medium/low), port, service
- For each vulnerability, suggest a concrete fix
- If you need more information, use [SEARCH:] or [TOOL:]
- Format vulnerabilities clearly so they can be saved to a database
- Be specific about CVE IDs when you know them
- Always give a final risk rating: CRITICAL / HIGH / MEDIUM / LOW

Output format for vulnerabilities (use this exactly):
VULN: <name> | SEVERITY: <level> | PORT: <port> | SERVICE: <service>
DESC: <description>
FIX: <fix recommendation>

Output format for exploits:
EXPLOIT: <name> | TOOL: <tool> | PAYLOAD: <payload or description>
RESULT: <expected result>
NOTES: <any notes>

End your analysis with:
RISK_LEVEL: <CRITICAL|HIGH|MEDIUM|LOW>
SUMMARY: <2-3 sentence overall summary>
IMPORTANT: Never use markdown bold (**text**) or headers (## text). Plain text only. No exceptions.
IMPORTANT RULES FOR ACCURACY:
- nmap filtered or no-response means INCONCLUSIVE not vulnerable
- Never assert a server version without seeing it in scan output
- Never infer CVEs from guessed versions
- curl timeouts and HTTP_CODE=000 mean the host is unreachable not exploitable
- ab and stress tools are not Slowloris unless confirmed
- Only assign CRITICAL if there is direct evidence of exploitability
- If evidence is weak mark severity as LOW with note: unconfirmed"""


# ─────────────────────────────────────────────
# BASE PROVIDER CLASS
# ─────────────────────────────────────────────

class LLMProvider(ABC):
    """Abstract base class for all LLM providers"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_output_tokens = int(config.get("max_output_tokens", 2000))
        self.context_window = int(
            config.get(
                "context_window",
                infer_model_context_window(self._get_context_model_name())
            )
        )

    def _get_context_model_name(self) -> str:
        """Return the model or deployment name used to infer context window."""
        return str(
            self.config.get("model")
            or self.config.get("deployment")
            or ""
        )

    def _estimate_message_tokens(self, messages: List[Dict]) -> int:
        """Rough token estimate without adding a tokenizer dependency."""
        total_chars = 0
        for message in messages:
            content = str(message.get("content", ""))
            total_chars += len(content)

        return max(1, total_chars // 4) + (len(messages) * 12)

    def _truncate_text(self, text: str, max_chars: int) -> str:
        """Keep the start and end of long content because both are often informative."""
        if len(text) <= max_chars:
            return text

        half = max_chars // 2
        return (
            text[:half]
            + "\n\n[... content truncated to fit model context ...]\n\n"
            + text[-half:]
        )

    def _prepare_messages(self, messages: List[Dict]) -> List[Dict]:
        """Compact message history to stay within the model's input budget."""
        prepared = [
            {
                "role": message.get("role", "user"),
                "content": str(message.get("content", "")),
            }
            for message in messages
        ]

        target_input_budget = max(1024, self.context_window - self.max_output_tokens - 512)
        if self._estimate_message_tokens(prepared) <= target_input_budget:
            return prepared

        for index in range(1, len(prepared)):
            if self._estimate_message_tokens(prepared) <= target_input_budget:
                break

            content = prepared[index]["content"]
            if len(content) > 2400:
                prepared[index]["content"] = self._truncate_text(content, 2400)

        if self._estimate_message_tokens(prepared) <= target_input_budget:
            return prepared

        if len(prepared) > 4:
            prepared = [prepared[0]] + prepared[-3:]

        if self._estimate_message_tokens(prepared) <= target_input_budget:
            return prepared

        for index in range(1, len(prepared)):
            prepared[index]["content"] = self._truncate_text(prepared[index]["content"], 1200)

        return prepared

    def _calculate_response_tokens(self, messages: List[Dict]) -> int:
        """Reserve output tokens based on estimated input size and context window."""
        input_tokens = self._estimate_message_tokens(messages)
        available = self.context_window - input_tokens - 256
        return max(256, min(self.max_output_tokens, available))
    
    @abstractmethod
    def ask(self, messages: List[Dict]) -> str:
        """Send messages and get response"""
        pass
    
    @abstractmethod
    def validate(self) -> tuple[bool, str]:
        """Validate credentials/connection. Returns (is_valid, message)"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict:
        """Return provider info"""
        pass


# ─────────────────────────────────────────────
# OLLAMA PROVIDER (Local)
# ─────────────────────────────────────────────

class OllamaProvider(LLMProvider):
    """Local Ollama provider - free and privacy-first"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.url = config.get("url", "http://localhost:11434/api/chat")
        self.model = config.get("model", "metatron-qwen")
        self.timeout = config.get("timeout", 600)

        # For new/custom Ollama models, auto-detect context from Ollama API.
        if "context_window" not in config:
            self.context_window = self._resolve_ollama_context_window()

    def _ollama_base_url(self) -> str:
        """Return Ollama server base URL from a chat endpoint URL."""
        if "/api/" in self.url:
            return self.url.split("/api/")[0].rstrip("/")
        return self.url.rstrip("/")

    def _resolve_ollama_context_window(self) -> int:
        """Resolve context window for local models using Ollama metadata when available."""
        inferred_default = infer_model_context_window(self.model, 16384)
        base_url = self._ollama_base_url()

        try:
            response = requests.post(
                f"{base_url}/api/show",
                json={"name": self.model},
                timeout=min(10, self.timeout),
            )
            response.raise_for_status()
            data = response.json()

            model_info = data.get("model_info", {}) or {}
            for key, value in model_info.items():
                if "context_length" in str(key):
                    try:
                        context_window = int(value)
                        if context_window > 0:
                            return context_window
                    except (TypeError, ValueError):
                        continue

            combined_text = "\n".join([
                str(data.get("parameters", "")),
                str(data.get("modelfile", "")),
            ])
            match = re.search(r"num_ctx\s+(\d+)", combined_text)
            if match:
                context_window = int(match.group(1))
                if context_window > 0:
                    return context_window
        except Exception:
            pass

        return inferred_default
    
    def ask(self, messages: List[Dict]) -> str:
        try:
            prepared_messages = self._prepare_messages(messages)
            response_tokens = self._calculate_response_tokens(prepared_messages)
            payload = {
                "model": self.model,
                "messages": prepared_messages,
                "stream": False,
                "options": {
                    "num_predict": response_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            }
            resp = requests.post(self.url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            response = data.get("message", {}).get("content", "").strip()
            return response if response else "[!] Model returned empty response."
        except requests.exceptions.ConnectionError:
            return "[!] Cannot connect to Ollama. Is it running? Try: ollama serve"
        except requests.exceptions.Timeout:
            return "[!] Ollama timed out. Model may be loading, try again."
        except requests.exceptions.HTTPError as e:
            return f"[!] Ollama HTTP error: {e}"
        except Exception as e:
            return f"[!] Unexpected error: {e}"
    
    def validate(self) -> tuple[bool, str]:
        try:
            resp = requests.get(self.url.replace("/api/chat", "/api/tags"), timeout=5)
            resp.raise_for_status()
            return (True, "Ollama connection successful")
        except Exception as e:
            return (False, f"Ollama connection failed: {e}")
    
    def get_model_info(self) -> Dict:
        return {
            "provider": "Ollama",
            "model": self.model,
            "cost": "Free",
            "location": "Local"
        }


# ─────────────────────────────────────────────
# OPENAI PROVIDER (GPT-4)
# ─────────────────────────────────────────────

class OpenAIProvider(LLMProvider):
    """OpenAI provider - GPT-4, GPT-3.5-turbo"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.model = config.get("model", "gpt-4")
        self.temperature = config.get("temperature", 0.7)

    def _infer_context_window(self, model: str) -> int:
        """Infer a safe context window when the model is known."""
        return infer_model_context_window(model, 8192)

    def _get_client(self):
        """Create an OpenAI client using the configured API key."""
        if not self.api_key:
            raise ValueError("OpenAI API key not configured")

        try:
            return OpenAI(api_key=self.api_key)
        except TypeError as e:
            # Some openai/httpx combinations may pass `proxies` internally, while
            # newer httpx versions expect `proxy`.
            if "proxies" in str(e).lower():
                return OpenAI(api_key=self.api_key, http_client=httpx.Client())
            raise

    def _is_legacy_client(self, client) -> bool:
        return not hasattr(client, "chat") or not hasattr(client.chat, "completions")

    def ask(self, messages: List[Dict]) -> str:
        try:
            client = self._get_client()
            prepared_messages = self._prepare_messages(messages)
            response_tokens = self._calculate_response_tokens(prepared_messages)

            if self._is_legacy_client(client):
                response = client.ChatCompletion.create(
                    model=self.model,
                    messages=prepared_messages,
                    temperature=self.temperature,
                    max_tokens=response_tokens,
                    timeout=60
                )
                content = response["choices"][0]["message"]["content"]
            else:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=prepared_messages,
                    temperature=self.temperature,
                    max_tokens=response_tokens,
                    timeout=60
                )
                content = response.choices[0].message.content

            return content.strip() if content else "[!] OpenAI returned empty response."
        except ImportError:
            return "[!] openai package not installed. Install with: pip install openai"
        except Exception as e:
            return f"[!] OpenAI error: {e}"
    
    def validate(self) -> tuple[bool, str]:
        try:
            client = self._get_client()
            if hasattr(client, "models"):
                client.models.list()
            elif hasattr(client, "Model"):
                client.Model.list()
            else:
                return (False, "OpenAI client is not compatible with this code path")
            return (True, "OpenAI API key valid")
        except ImportError:
            return (False, "openai package not installed")
        except ValueError as e:
            return (False, str(e))
        except Exception as e:
            return (False, f"OpenAI validation failed: {str(e)[:100]}")
    
    def get_model_info(self) -> Dict:
        return {
            "provider": "OpenAI",
            "model": self.model,
            "cost": "$0.03-0.06 per 1M tokens",
            "location": "Cloud"
        }


# ─────────────────────────────────────────────
# ANTHROPIC PROVIDER (Claude)
# ─────────────────────────────────────────────

class AnthropicProvider(LLMProvider):
    """Anthropic provider - Claude-3 models"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.api_key = config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        self.model = config.get("model", "claude-3-opus-20240229")
        self.temperature = config.get("temperature", 0.7)
    
    def ask(self, messages: List[Dict]) -> str:
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.api_key)
            prepared_messages = self._prepare_messages(messages)
            response_tokens = self._calculate_response_tokens(prepared_messages)
            
            # Claude doesn't use role="system", extract it separately
            system_msg = SYSTEM_PROMPT
            for message in prepared_messages:
                if message.get("role") == "system":
                    system_msg = str(message.get("content", SYSTEM_PROMPT))
                    break

            messages_filtered = [m for m in prepared_messages if m.get("role") != "system"]
            
            response = client.messages.create(
                model=self.model,
                max_tokens=response_tokens,
                system=system_msg,
                messages=messages_filtered,
                temperature=self.temperature
            )
            return response.content[0].text.strip()
        except ImportError:
            return "[!] anthropic package not installed. Install with: pip install anthropic"
        except Exception as e:
            return f"[!] Anthropic error: {e}"
    
    def validate(self) -> tuple[bool, str]:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            # Test with a simple message
            response = client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return (True, "Anthropic API key valid")
        except ImportError:
            return (False, "anthropic package not installed")
        except Exception as e:
            return (False, f"Anthropic validation failed: {str(e)[:100]}")
    
    def get_model_info(self) -> Dict:
        return {
            "provider": "Anthropic",
            "model": self.model,
            "cost": "$0.003-0.024 per 1M tokens",
            "location": "Cloud"
        }


# ─────────────────────────────────────────────
# AZURE OPENAI PROVIDER (Copilot)
# ─────────────────────────────────────────────

class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI provider - GPT-4, Copilot integration"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.api_key = config.get("api_key") or os.getenv("AZURE_OPENAI_API_KEY")
        self.resource = config.get("resource") or os.getenv("AZURE_OPENAI_RESOURCE")
        self.deployment = config.get("deployment") or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.api_version = config.get("api_version", "2024-02-15-preview")
        self.temperature = config.get("temperature", 0.7)

    def _get_context_model_name(self) -> str:
        """Use deployment name first because Azure often hides the base model name."""
        return str(
            self.config.get("deployment")
            or self.config.get("model")
            or os.getenv("AZURE_OPENAI_DEPLOYMENT")
            or ""
        )
    
    def ask(self, messages: List[Dict]) -> str:
        try:
            import openai
            from openai import AzureOpenAI
            prepared_messages = self._prepare_messages(messages)
            response_tokens = self._calculate_response_tokens(prepared_messages)
            
            client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=f"https://{self.resource}.openai.azure.com"
            )
            
            response = client.chat.completions.create(
                model=self.deployment,
                messages=prepared_messages,
                temperature=self.temperature,
                max_tokens=response_tokens
            )
            return response.choices[0].message.content.strip()
        except ImportError:
            return "[!] openai package not installed. Install with: pip install openai"
        except Exception as e:
            return f"[!] Azure OpenAI error: {e}"
    
    def validate(self) -> tuple[bool, str]:
        try:
            if not all([self.api_key, self.resource, self.deployment]):
                return (False, "Missing Azure credentials (api_key, resource, deployment)")
            
            from openai import AzureOpenAI
            client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=f"https://{self.resource}.openai.azure.com"
            )
            # Test with a simple call
            response = client.chat.completions.create(
                model=self.deployment,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=10
            )
            return (True, "Azure OpenAI credentials valid")
        except ImportError:
            return (False, "openai package not installed")
        except Exception as e:
            return (False, f"Azure validation failed: {str(e)[:100]}")
    
    def get_model_info(self) -> Dict:
        return {
            "provider": "Azure OpenAI",
            "model": self.deployment,
            "cost": "Depends on Azure plan",
            "location": "Azure Cloud"
        }


# ─────────────────────────────────────────────
# PROVIDER FACTORY
# ─────────────────────────────────────────────

class LLMProviderFactory:
    """Factory for creating LLM provider instances"""
    
    _providers = {
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "azure": AzureOpenAIProvider,
    }
    
    @staticmethod
    def create(provider_name: str, config: Dict) -> Optional[LLMProvider]:
        """Create a provider instance"""
        if provider_name not in LLMProviderFactory._providers:
            return None
        
        provider_class = LLMProviderFactory._providers[provider_name]
        return provider_class(config)
    
    @staticmethod
    def get_providers() -> List[str]:
        """List available providers"""
        return list(LLMProviderFactory._providers.keys())
    
    @staticmethod
    def get_provider_names() -> Dict[str, str]:
        """Get provider names with descriptions"""
        return {
            "ollama": "Ollama (Local - Free)",
            "openai": "OpenAI (GPT-4 - Cloud)",
            "anthropic": "Anthropic (Claude - Cloud)",
            "azure": "Azure OpenAI (Copilot - Cloud)",
        }
