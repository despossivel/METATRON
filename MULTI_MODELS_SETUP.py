#!/usr/bin/env python3
"""
METATRON - Multi-Model Provider Setup
Quick setup guide for cloud models (OpenAI, Anthropic, Azure OpenAI)
"""

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                   METATRON - Multi-Model Setup Guide                       ║
╚════════════════════════════════════════════════════════════════════════════╝

📋 AVAILABLE PROVIDERS:

  1️⃣  OLLAMA (Local - Default)
      ✓ Free and private
      ✓ No internet needed
      ✓ No API keys required
      ✗ Slower than cloud models
      Setup: Just use as is (requires Ollama running locally)

  2️⃣  OPENAI (GPT-4)
      ✓ Most advanced model
      ✓ Excellent for complex analysis
      💰 Cost: ~$0.03-0.06 per 1M tokens
      Setup: 
        1. Get API key from https://platform.openai.com/api-keys
        2. Start METATRON and go to Provider Settings
        3. Configure with your API key

  3️⃣  ANTHROPIC (Claude)
      ✓ Great reasoning capabilities
      ✓ 200K context window (largest)
      💰 Cost: ~$0.003-0.024 per 1M tokens
      Setup:
        1. Get API key from https://console.anthropic.com/
        2. Start METATRON and go to Provider Settings
        3. Configure with your API key

  4️⃣  AZURE OPENAI (Copilot)
      ✓ Enterprise integration
      ✓ Microsoft ecosystem compatibility
      💰 Cost: Depends on Azure plan
      Setup:
        1. Create Azure OpenAI resource
        2. Get: API Key, Resource Name, Deployment Name
        3. Start METATRON and go to Provider Settings
        4. Configure all three values


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 QUICK START:

METHOD 1: Via Environment Variables (Fastest)
──────────────────────────────────────────────
  export METATRON_PROVIDER=openai
  python metatron.py
  # Will use OpenAI with API key from OPENAI_API_KEY env var

  Supported env vars:
  - METATRON_PROVIDER: provider name (ollama|openai|anthropic|azure)
  - OPENAI_API_KEY: for OpenAI
  - ANTHROPIC_API_KEY: for Anthropic
  - AZURE_OPENAI_API_KEY: for Azure
  - AZURE_OPENAI_RESOURCE: for Azure
  - AZURE_OPENAI_DEPLOYMENT: for Azure


METHOD 2: Via Config File
──────────────────────────
  Edit: ~/.metatron/config.yml
  
  Example for OpenAI:
    active_provider: openai
    providers:
      openai:
        api_key: sk-... (your API key here)
        model: gpt-4
        temperature: 0.7

  Then: python metatron.py
  # Will automatically use OpenAI with saved credentials


METHOD 3: Interactive Menu (Recommended)
───────────────────────────────────────
  python metatron.py
  # Main Menu → [3] Provider Settings
  # Choose option to:
    1. Change Provider
    2. Configure Credentials
    3. View Provider Status
  # Credentials are saved to ~/.metatron/config.yml


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔐 SECURITY NOTES:

  ⚠️  API Keys are sensitive!
  
  Best practices:
  ✓ Store API keys in environment variables (OPENAI_API_KEY, etc)
  ✓ Don't commit ~/.metatron/config.yml to git (add to .gitignore)
  ✓ Config file is protected with 600 permissions (user-only readable)
  ✓ Never hardcode API keys in code
  
  If using config file:
  $ chmod 600 ~/.metatron/config.yml
  

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 USAGE EXAMPLES:

  # Switch between providers in same session
  metatron> [1] New Scan → Scan runs
  metatron> [3] Provider Settings → Change to Claude
  metatron> [1] New Scan → Scan runs with Claude
  
  # Quick switch via env var
  export METATRON_PROVIDER=anthropic
  python metatron.py
  
  # Check provider status
  metatron> [3] Provider Settings → [3] View All Providers Status


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 COST COMPARISON (for 100K tokens):

  Ollama:        $0.00 (Free, local)
  OpenAI GPT-4:  $3-6 USD
  Claude-3:      $1.50-3 USD
  Azure OpenAI:  Variable (enterprise plan)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❓ TROUBLESHOOTING:

  Provider not found?
  → Check ~/.metatron/config.yml exists
  → Verify provider name is lowercase (ollama|openai|anthropic|azure)

  API key validation failed?
  → Check key is correct in config or env var
  → Make sure environment variable is set: echo $OPENAI_API_KEY
  → For Azure: need all three: api_key, resource, deployment

  Ollama not responding?
  → Start Ollama: ollama serve
  → Check it's running: curl http://localhost:11434/api/tags

  Can't import modules?
  → Install dependencies: pip install -r requirements.txt
  → Ensure pyyaml installed: pip install pyyaml


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📝 FILE STRUCTURE:

  ~/.metatron/config.yml          ← Your config file (created automatically)
  /METATRON/llm_providers.py      ← Provider implementations
  /METATRON/config.py             ← Config management
  /METATRON/llm.py                ← Updated to use providers
  /METATRON/metatron.py           ← Updated with Provider Settings menu


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 NEXT STEPS:

  1. Run: pip install -r requirements.txt
  2. Run: python metatron.py
  3. Go to: Provider Settings
  4. Configure your preferred provider
  5. Start scanning with any provider!

""")
