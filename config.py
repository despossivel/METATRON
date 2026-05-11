#!/usr/bin/env python3
"""
METATRON - config.py
Configuration manager for LLM providers
Handles config file, environment variables, and credential validation
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import yaml

CONFIG_DIR = Path.home() / ".metatron"
CONFIG_FILE = CONFIG_DIR / "config.yml"


def ensure_config_dir():
    """Create config directory if it doesn't exist"""
    CONFIG_DIR.mkdir(exist_ok=True, mode=0o700)


def load_config() -> Dict:
    """Load configuration from file or create defaults"""
    ensure_config_dir()
    
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = yaml.safe_load(f) or {}
                return config
        except Exception as e:
            print(f"[!] Error reading config: {e}")
            return get_default_config()
    else:
        return get_default_config()


def get_default_config() -> Dict:
    """Return default configuration"""
    return {
        "active_provider": "ollama",
        "providers": {
            "ollama": {
                "url": "http://localhost:11434",
                "model": "metatron-qwen",
                "timeout": 600
            },
            "openai": {
                "api_key": "",
                "model": "gpt-4o",
                "temperature": 0.7,
                "max_output_tokens": 2000,
                "context_window": 128000
            },
            "anthropic": {
                "api_key": "",
                "model": "claude-3-opus-20240229",
                "temperature": 0.7
            },
            "azure": {
                "api_key": "",
                "resource": "",
                "deployment": "",
                "api_version": "2024-02-15-preview",
                "temperature": 0.7
            }
        }
    }


def save_config(config: Dict):
    """Save configuration to file"""
    ensure_config_dir()
    try:
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        # Protect config file (600 permissions)
        os.chmod(CONFIG_FILE, 0o600)
    except Exception as e:
        print(f"[!] Error saving config: {e}")


def get_active_provider() -> str:
    """Get the active provider name"""
    # Priority: env var > config file > default
    if "METATRON_PROVIDER" in os.environ:
        return os.environ["METATRON_PROVIDER"]
    
    config = load_config()
    return config.get("active_provider", "ollama")


def set_active_provider(provider_name: str):
    """Set the active provider"""
    config = load_config()
    config["active_provider"] = provider_name
    save_config(config)


def get_provider_config(provider_name: str) -> Dict:
    """Get configuration for a specific provider"""
    config = load_config()
    return config.get("providers", {}).get(provider_name, {})


def update_provider_config(provider_name: str, provider_config: Dict):
    """Update configuration for a specific provider"""
    config = load_config()
    if "providers" not in config:
        config["providers"] = {}
    config["providers"][provider_name] = provider_config
    save_config(config)


def resolve_env_vars(config_str: str) -> str:
    """Resolve ${VAR_NAME} to environment variables"""
    if not isinstance(config_str, str):
        return config_str
    
    import re
    def replace_var(match):
        var_name = match.group(1)
        return os.getenv(var_name, match.group(0))
    
    return re.sub(r'\$\{(\w+)\}', replace_var, config_str)


def get_resolved_provider_config(provider_name: str) -> Dict:
    """Get provider config with environment variables resolved"""
    config = get_provider_config(provider_name)
    resolved = {}
    
    for key, value in config.items():
        if isinstance(value, str):
            resolved[key] = resolve_env_vars(value)
        else:
            resolved[key] = value
    
    return resolved


def is_provider_configured(provider_name: str) -> bool:
    """Check if provider has necessary credentials"""
    from llm_providers import LLMProviderFactory
    
    provider_config = get_resolved_provider_config(provider_name)
    provider = LLMProviderFactory.create(provider_name, provider_config)
    
    if provider is None:
        return False
    
    is_valid, _ = provider.validate()
    return is_valid


def validate_all_providers() -> Dict[str, Tuple[bool, str]]:
    """Validate all configured providers"""
    from llm_providers import LLMProviderFactory
    
    results = {}
    for provider_name in LLMProviderFactory.get_providers():
        provider_config = get_resolved_provider_config(provider_name)
        provider = LLMProviderFactory.create(provider_name, provider_config)
        
        if provider is None:
            results[provider_name] = (False, "Provider not found")
        else:
            results[provider_name] = provider.validate()
    
    return results


def prompt_provider_selection() -> Optional[str]:
    """Interactive menu for provider selection"""
    from llm_providers import LLMProviderFactory
    
    providers = LLMProviderFactory.get_provider_names()
    
    print(f"\n\033[33m{'─' * 50}\033[0m")
    print("\033[92m  Select LLM Provider:\033[0m")
    print(f"\033[33m{'─' * 50}\033[0m\n")
    
    for idx, (key, name) in enumerate(providers.items(), 1):
        print(f"  \033[92m[{idx}]\033[0m  {name}")
    
    print()
    choice = input("\033[36mEnter choice (1-4): \033[0m").strip()
    
    provider_list = list(providers.keys())
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(provider_list):
            return provider_list[idx]
    except ValueError:
        pass
    
    return None


def prompt_provider_credentials(provider_name: str) -> Dict:
    """Interactive menu for entering provider credentials"""
    print(f"\n\033[33m{'─' * 50}\033[0m")
    print(f"\033[92m  Configure {provider_name.upper()}:\033[0m")
    print(f"\033[33m{'─' * 50}\033[0m\n")
    
    config = get_provider_config(provider_name)
    new_config = config.copy()
    
    if provider_name == "ollama":
        new_config["url"] = input(f"Ollama URL [{config.get('url', 'http://localhost:11434')}]: ").strip() or config.get('url')
        new_config["model"] = input(f"Model name [{config.get('model', 'metatron-qwen')}]: ").strip() or config.get('model')
    
    elif provider_name == "openai":
        key = input("OpenAI API Key (leave empty to use env OPENAI_API_KEY): ").strip()
        if key:
            new_config["api_key"] = key
        new_config["model"] = input(f"Model [{config.get('model', 'gpt-4o')}]: ").strip() or config.get('model', 'gpt-4o')
        max_output_tokens = input(
            f"Max output tokens [{config.get('max_output_tokens', 2000)}]: "
        ).strip()
        new_config["max_output_tokens"] = int(max_output_tokens) if max_output_tokens else config.get('max_output_tokens', 2000)
    
    elif provider_name == "anthropic":
        key = input("Anthropic API Key (leave empty to use env ANTHROPIC_API_KEY): ").strip()
        if key:
            new_config["api_key"] = key
        new_config["model"] = input(f"Model [{config.get('model', 'claude-3-opus-20240229')}]: ").strip() or config.get('model')
    
    elif provider_name == "azure":
        key = input("Azure OpenAI API Key (leave empty to use env AZURE_OPENAI_API_KEY): ").strip()
        if key:
            new_config["api_key"] = key
        new_config["resource"] = input("Azure Resource Name (leave empty to use env AZURE_OPENAI_RESOURCE): ").strip() or config.get('resource', '')
        new_config["deployment"] = input("Deployment Name (leave empty to use env AZURE_OPENAI_DEPLOYMENT): ").strip() or config.get('deployment', '')
    
    return new_config


def show_current_provider():
    """Display current active provider"""
    from llm_providers import LLMProviderFactory
    
    active = get_active_provider()
    provider_config = get_resolved_provider_config(active)
    provider = LLMProviderFactory.create(active, provider_config)
    
    if provider:
        info = provider.get_model_info()
        print(f"\n\033[92mCurrent Provider: {info['provider']} ({info['model']})\033[0m")
        print(f"   Cost: {info['cost']}")
        print(f"   Location: {info['location']}\n")
    else:
        print(f"\n\033[91m[!] Provider not found: {active}\033[0m\n")


def show_provider_status():
    """Show status of all providers"""
    from llm_providers import LLMProviderFactory
    
    print(f"\n\033[33m{'─' * 50}\033[0m")
    print("\033[92m  Provider Status:\033[0m")
    print(f"\033[33m{'─' * 50}\033[0m\n")
    
    for provider_name in LLMProviderFactory.get_providers():
        provider_config = get_resolved_provider_config(provider_name)
        provider = LLMProviderFactory.create(provider_name, provider_config)
        
        if provider:
            is_valid, msg = provider.validate()
            status = "\033[92mOK\033[0m" if is_valid else "\033[91mERR\033[0m"
            info = provider.get_model_info()
            print(f"  {status} {info['provider']} ({info['model']})")
            print(f"     {msg}\n")
