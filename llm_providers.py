#!/usr/bin/env python3
"""
METATRON - llm_providers.py
Abstraction layer for multiple LLM providers (Strategy Pattern)
Supports: Ollama (local), OpenAI (GPT-4), Anthropic (Claude), Azure OpenAI (Copilot)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import os
import requests
import json
from datetime import datetime
import time


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
    
    def ask(self, messages: List[Dict]) -> str:
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_predict": 8192,
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
    
    def ask(self, messages: List[Dict]) -> str:
        try:
            import openai
            openai.api_key = self.api_key
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=8192,
                timeout=60
            )
            return response.choices[0].message.content.strip()
        except ImportError:
            return "[!] openai package not installed. Install with: pip install openai"
        except Exception as e:
            return f"[!] OpenAI error: {e}"
    
    def validate(self) -> tuple[bool, str]:
        try:
            import openai
            openai.api_key = self.api_key
            openai.Model.list()
            return (True, "OpenAI API key valid")
        except ImportError:
            return (False, "openai package not installed")
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
            
            # Claude doesn't use role="system", extract it separately
            system_msg = SYSTEM_PROMPT
            messages_filtered = [m for m in messages if m.get("role") != "system"]
            
            response = client.messages.create(
                model=self.model,
                max_tokens=8192,
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
    
    def ask(self, messages: List[Dict]) -> str:
        try:
            import openai
            from openai import AzureOpenAI
            
            client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=f"https://{self.resource}.openai.azure.com"
            )
            
            response = client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=self.temperature,
                max_tokens=8192
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
