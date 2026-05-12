#!/usr/bin/env python3
"""
METATRON - llm.py
LLM interface for multiple providers (Ollama, OpenAI, Anthropic, Azure).
Builds prompts, handles AI responses, runs tool dispatch loop.
"""

import re
import requests
import json
from tools import run_tool_by_command, run_nmap, run_curl_headers
from search import handle_search_dispatch

# Try importing provider classes (optional for backward compatibility)
try:
    from llm_providers import LLMProvider, LLMProviderFactory
except ImportError:
    LLMProvider = None
    LLMProviderFactory = None

OLLAMA_URL  = "http://localhost:11434/api/chat"
MODEL_NAME  = "metatron-qwen"
MAX_OUTPUT_TOKENS = 1024
MAX_TOOL_LOOPS = 6   # max times AI can call tools per session
OLLAMA_TIMEOUT = 600
MAX_RAW_SCAN_CHARS = 10000
MAX_TOOL_OUTPUT_SUMMARY_CHARS = 4000
MAX_TOOL_RESULTS_CHARS = 3200
MAX_CHAT_HISTORY_CHARS = 14000

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
IMPORTANT: Never use markdown bold (**text**) or 
headers (## text). Plain text only. No exceptions.
IMPORTANT RULES FOR ACCURACY:
- nmap filtered or no-response means INCONCLUSIVE not vulnerable
- Never assert a server version without seeing it in scan output
- Never infer CVEs from guessed versions
- curl timeouts and HTTP_CODE=000 mean the host is unreachable not exploitable
- ab and stress tools are not Slowloris unless confirmed
- Only assign CRITICAL if there is direct evidence of exploitability
- If evidence is weak mark severity as LOW with note: unconfirmed"""


# ─────────────────────────────────────────────
# OLLAMA API CALL
# ─────────────────────────────────────────────

def ask_ollama(messages: list) -> str:
    try:
        payload = {
            "model":  MODEL_NAME,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": MAX_OUTPUT_TOKENS,
                "temperature": 0.7,
                "top_p": 0.9,
            }
        }
        print(f"\n[*] Sending to {MODEL_NAME}...")
        resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        response = data.get("message", {}).get("content", "").strip()
        if not response:
            return "[!] Model returned empty response."
        return response
    except requests.exceptions.ConnectionError:
        return "[!] Cannot connect to Ollama. Is it running? Try: ollama serve"
    except requests.exceptions.Timeout:
        return "[!] Ollama timed out. Model may be loading, try again."
    except requests.exceptions.HTTPError as e:
        return f"[!] Ollama HTTP error: {e}"
    except Exception as e:
        return f"[!] Unexpected error: {e}"


def _truncate_text_for_prompt(text: str, max_chars: int) -> str:
    """Truncate long model inputs while preserving head and tail context."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return (
        text[:half]
        + "\n\n[... truncated for performance ...]\n\n"
        + text[-half:]
    )


def _simple_tool_summary(raw_output: str) -> str:
    """Create a lightweight summary of tool output using heuristics, not extra model calls."""
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    if not lines:
        return raw_output

    if len(raw_output) < MAX_TOOL_OUTPUT_SUMMARY_CHARS:
        return raw_output

    selected = []
    keywords = [
        "open", "closed", "filtered", "tcp", "udp", "ssl", "tls",
        "http", "server", "version", "vuln", "vulnerability", "error",
        "warning", "certificate", "expired", "protocol", "service",
        "host", "banner", "title", "status", "x-frame-options",
        "strict-transport-security", "content-security-policy"
    ]

    seen = set()
    for line in lines[:40]:
        if line not in seen:
            selected.append(line)
            seen.add(line)
    for line in lines[-20:]:
        if line not in seen:
            selected.append(line)
            seen.add(line)
    for line in lines:
        low = line.lower()
        if any(k in low for k in keywords) and line not in seen:
            selected.append(line)
            seen.add(line)
            if len(selected) >= 80:
                break

    if len(selected) < 20:
        selected = lines[:min(60, len(lines))]

    return "\n".join(selected[:80]) + "\n[... tool output truncated ...]"


def _summarize_nmap(raw_output: str) -> str:
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    relevant = []
    for line in lines:
        if re.match(r'^\d+/(tcp|udp)\s+', line):
            relevant.append(line)
        elif re.search(r'(service info|service detection|version|traceroute|filtered|open|closed|running)', line, re.IGNORECASE):
            relevant.append(line)
        elif line.startswith('PORT'):
            relevant.append(line)
        elif 'Nmap done' in line:
            relevant.append(line)
    if not relevant:
        return _simple_tool_summary(raw_output)
    return "\n".join(dict.fromkeys(relevant)) + "\n[... nmap output summarized ...]"


def _summarize_curl(raw_output: str) -> str:
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    relevant = []
    headers = [
        'http/', 'server:', 'x-', 'strict-transport-security:', 'content-security-policy:',
        'x-frame-options:', 'x-content-type-options:', 'set-cookie:', 'location:', 'content-type:'
    ]
    for line in lines:
        low = line.lower()
        if any(h in low for h in headers) or re.match(r'^http/\d', low):
            relevant.append(line)
        elif 'warning' in low or 'error' in low:
            relevant.append(line)
    if not relevant:
        return _simple_tool_summary(raw_output)
    return "\n".join(dict.fromkeys(relevant[:60])) + "\n[... curl headers summarized ...]"


def _summarize_dig(raw_output: str) -> str:
    lines = [line.strip() for line in raw_output.splitlines() if line.strip() and not line.startswith(';;')]
    records = []
    for line in lines:
        if re.search(r'\b(A|MX|NS|TXT|CNAME|AAAA)\b', line) or re.match(r'^[^;]+\s+IN\s+', line):
            records.append(line)
    if not records:
        return _simple_tool_summary(raw_output)
    return "\n".join(dict.fromkeys(records[:80])) + "\n[... dig records summarized ...]"


def _summarize_whois(raw_output: str) -> str:
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    relevant = []
    keys = [
        'domain name', 'registrar', 'creation date', 'expiry date', 'expiration date',
        'updated date', 'name server', 'status', 'registrant', 'dnssec', 'email', 'tech contact'
    ]
    for line in lines:
        low = line.lower()
        if any(k in low for k in keys):
            relevant.append(line)
    if not relevant:
        return _simple_tool_summary(raw_output)
    return "\n".join(dict.fromkeys(relevant[:80])) + "\n[... whois summarized ...]"


def _summarize_whatweb(raw_output: str) -> str:
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    relevant = []
    for line in lines:
        low = line.lower()
        if re.match(r'^(http|https)://', low) or ':' in line:
            relevant.append(line)
        elif 'title' in low or 'server' in low or 'cms' in low or 'x-powered-by' in low:
            relevant.append(line)
    if not relevant:
        return _simple_tool_summary(raw_output)
    return "\n".join(dict.fromkeys(relevant[:80])) + "\n[... whatweb summarized ...]"


def _summarize_nikto(raw_output: str) -> str:
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    relevant = []
    for line in lines:
        if re.search(r'(OSVDB|CVE|Warning|Server|title|cookie|header|ssl|tls|vulnerable|dangerous)', line, re.IGNORECASE):
            relevant.append(line)
        elif re.match(r'^\+\s+', line):
            relevant.append(line)
    if not relevant:
        return _simple_tool_summary(raw_output)
    return "\n".join(dict.fromkeys(relevant[:80])) + "\n[... nikto summarized ...]"


def _summarize_search(raw_output: str) -> str:
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    if not lines:
        return raw_output
    return "\n".join(lines[:40]) + "\n[... search results truncated ...]"


def _extract_tool_name(call_content: str) -> str:
    parts = call_content.strip().split()
    if not parts:
        return "unknown"
    return parts[0].lower().split('/')[-1]


def summarize_tool_output(raw_output: str, tool_name: str | None = None) -> str:
    """Compress raw tool output before adding it to the LLM context."""
    if len(raw_output) <= MAX_TOOL_OUTPUT_SUMMARY_CHARS:
        return raw_output

    tool_name = (tool_name or "").lower()
    if tool_name == 'nmap':
        return _summarize_nmap(raw_output)
    if tool_name == 'curl':
        return _summarize_curl(raw_output)
    if tool_name == 'dig':
        return _summarize_dig(raw_output)
    if tool_name == 'whois':
        return _summarize_whois(raw_output)
    if tool_name == 'whatweb':
        return _summarize_whatweb(raw_output)
    if tool_name == 'nikto':
        return _summarize_nikto(raw_output)
    if tool_name == 'search':
        return _summarize_search(raw_output)

    return _simple_tool_summary(raw_output)


def _truncate_tool_results(raw_output: str) -> str:
    return _truncate_text_for_prompt(raw_output, MAX_TOOL_RESULTS_CHARS)


def _compact_messages(messages: list) -> list:
    """Keep only essential message history for the next model call."""
    if len(messages) <= 4:
        return _prune_history_messages(messages)
    system = messages[0]
    initial = messages[1]
    tail = messages[-2:]
    compacted = [system, initial] + tail
    return _prune_history_messages(compacted)


def _prune_history_messages(messages: list) -> list:
    """Trim messages if the total history exceeds the allowed character budget."""
    total = sum(len(str(m.get("content", ""))) for m in messages)
    if total <= MAX_CHAT_HISTORY_CHARS:
        return messages

    # always preserve system prompt and the two most recent exchanges
    system = messages[0]
    recent = messages[-2:] if len(messages) > 2 else messages[1:]
    retained = [system] + recent

    # keep the first user prompt if present, truncated
    if len(messages) > 1:
        first = messages[1]
        retained.insert(1, {
            "role": first["role"],
            "content": _truncate_text_for_prompt(first["content"], MAX_CHAT_HISTORY_CHARS // 4)
        })

    # trim all retained messages to fit the budget
    while sum(len(str(m.get("content", ""))) for m in retained) > MAX_CHAT_HISTORY_CHARS:
        for msg in retained[1:]:
            msg["content"] = _truncate_text_for_prompt(msg["content"], max(256, MAX_CHAT_HISTORY_CHARS // len(retained)))
        if all(len(msg["content"]) <= 512 for msg in retained[1:]):
            break

    return retained


# ─────────────────────────────────────────────
# TOOL DISPATCH
# ─────────────────────────────────────────────

def extract_tool_calls(response: str) -> list:
    """
    Extract all [TOOL: ...] and [SEARCH: ...] tags from AI response.
    Returns list of tuples: [("TOOL", "nmap -sV x.x.x.x"), ("SEARCH", "CVE...")]
    """
    calls = []

    tool_matches   = re.findall(r'\[TOOL:\s*(.+?)\]',   response)
    search_matches = re.findall(r'\[SEARCH:\s*(.+?)\]', response)

    for m in tool_matches:
        calls.append(("TOOL", m.strip()))
    for m in search_matches:
        calls.append(("SEARCH", m.strip()))

    return calls

def run_tool_calls(calls: list) -> str:
    """
    Execute all tool/search calls and return combined results string.
    """
    if not calls:
        return ""

    results = ""
    for call_type, call_content in calls:
        print(f"\n  [DISPATCH] {call_type}: {call_content}")

        if call_type == "TOOL":
            output = run_tool_by_command(call_content)
            tool_name = _extract_tool_name(call_content)
        elif call_type == "SEARCH":
            output = handle_search_dispatch(call_content)
            tool_name = "search"
        else:
            output = f"[!] Unknown call type: {call_type}"
            tool_name = None

        compressed = summarize_tool_output(output.strip(), tool_name=tool_name)
        results += f"\n[{call_type} RESULT: {call_content}]\n"
        results += "─" * 40 + "\n"
        results += compressed + "\n"

    return results


# ─────────────────────────────────────────────
# PARSER — extract structured data from AI output
# ─────────────────────────────────────────────
def _clean(line: str) -> str:
    return re.sub(r'\*+', '', line).strip()
def parse_vulnerabilities(response: str) -> list:
    """
    Parse VULN: lines from AI response into dicts.
    Returns list of vulnerability dicts ready for db.save_vulnerability()
    """
    vulns = []
    lines = response.splitlines()

    i = 0
    while i < len(lines):
        line = _clean(lines[i])
        if line.startswith("VULN:"):
            vuln = {
                "vuln_name":   "",
                "severity":    "medium",
                "port":        "",
                "service":     "",
                "description": "",
                "fix":         ""
            }

            # parse header line: VULN: name | SEVERITY: x | PORT: x | SERVICE: x
            parts = line.split("|")
            for part in parts:
                part = part.strip()
                if part.startswith("VULN:"):
                    vuln["vuln_name"] = part.replace("VULN:", "").strip()
                elif part.startswith("SEVERITY:"):
                    vuln["severity"] = part.replace("SEVERITY:", "").strip().lower()
                elif part.startswith("PORT:"):
                    vuln["port"] = part.replace("PORT:", "").strip()
                elif part.startswith("SERVICE:"):
                    vuln["service"] = part.replace("SERVICE:", "").strip()

            # look ahead for DESC: and FIX: lines
            j = i + 1
            while j < len(lines) and j <= i + 5:
                next_line = _clean(lines[j])
                if next_line.startswith(("VULN:", "EXPLOIT:", "RISK_LEVEL:", "SUMMARY:")):
                    break
                if next_line.startswith("DESC:"):
                    vuln["description"] = next_line.replace("DESC:", "").strip()
                elif next_line.startswith("FIX:"):
                    vuln["fix"] = next_line.replace("FIX:", "").strip()
                j += 1

            if vuln["vuln_name"]:
                vulns.append(vuln)

        i += 1

    return vulns


def parse_exploits(response: str) -> list:
    """
    Parse EXPLOIT: lines from AI response into dicts.
    Returns list of exploit dicts ready for db.save_exploit()
    """
    exploits = []
    lines = response.splitlines()

    i = 0
    while i < len(lines):
        line = _clean(lines[i])
        if line.startswith("EXPLOIT:"):
            exploit = {
                "exploit_name": "",
                "tool_used":    "",
                "payload":      "",
                "result":       "unknown",
                "notes":        ""
            }

            parts = line.split("|")
            for part in parts:
                part = part.strip()
                if part.startswith("EXPLOIT:"):
                    exploit["exploit_name"] = part.replace("EXPLOIT:", "").strip()
                elif part.startswith("TOOL:"):
                    exploit["tool_used"] = part.replace("TOOL:", "").strip()
                elif part.startswith("PAYLOAD:"):
                    exploit["payload"] = part.replace("PAYLOAD:", "").strip()

            j = i + 1
            while j < len(lines) and j <= i + 4:
                next_line = _clean(lines[j])
                if next_line.startswith(("VULN:", "EXPLOIT:", "RISK_LEVEL:", "SUMMARY:")):
                    break
                if next_line.startswith("RESULT:"):
                    exploit["result"] = next_line.replace("RESULT:", "").strip()
                elif next_line.startswith("NOTES:"):
                    exploit["notes"] = next_line.replace("NOTES:", "").strip()
                j += 1

            if exploit["exploit_name"]:
                exploits.append(exploit)

        i += 1

    return exploits


def parse_risk_level(response: str) -> str:
    """Extract RISK_LEVEL from AI response."""
    match = re.search(r'RISK_LEVEL:\s*(CRITICAL|HIGH|MEDIUM|LOW)', response, re.IGNORECASE)
    return match.group(1).upper() if match else "UNKNOWN"


def parse_summary(response: str) -> str:
    match = re.search(r'SUMMARY:\s*(.+)', response, re.IGNORECASE)
    return match.group(1).strip() if match else ""


# ─────────────────────────────────────────────
# MAIN ANALYSIS FUNCTION
# ─────────────────────────────────────────────

def analyse_target(target: str, raw_scan: str) -> dict:
    raw_scan = _truncate_text_for_prompt(raw_scan, MAX_RAW_SCAN_CHARS)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"""TARGET: {target}

RECON DATA:
{raw_scan}

Analyze this target completely. Use [TOOL:] or [SEARCH:] if you need more information.
List all vulnerabilities, fixes, and suggest exploits where applicable."""
        }
    ]

    final_response = ""

    for loop in range(MAX_TOOL_LOOPS):
        response = ask_ollama(messages)

        print(f"\n{'─'*60}")
        print(f"[METATRON - Round {loop + 1}]")
        print(f"{'─'*60}")
        print(response)

        final_response = response

        tool_calls = extract_tool_calls(response)
        if not tool_calls:
            print("\n[*] No tool calls. Analysis complete.")
            break

        tool_results = run_tool_calls(tool_calls)
        tool_results = _truncate_tool_results(tool_results)

        # add assistant response and tool results as new messages
        messages.append({
            "role": "assistant",
            "content": response
        })
        messages.append({
            "role": "user",
            "content": f"""[TOOL RESULTS]
{tool_results}

Continue your analysis with this new information.
If analysis is complete, give the final RISK_LEVEL and SUMMARY."""
        })
        messages = _compact_messages(messages)

    vulnerabilities = parse_vulnerabilities(final_response)
    exploits        = parse_exploits(final_response)
    risk_level      = parse_risk_level(final_response)
    summary         = parse_summary(final_response)

    print(f"\n[+] Parsed: {len(vulnerabilities)} vulns, {len(exploits)} exploits | Risk: {risk_level}")

    return {
        "full_response":   final_response,
        "vulnerabilities": vulnerabilities,
        "exploits":        exploits,
        "risk_level":      risk_level,
        "summary":         summary,
        "raw_scan":        raw_scan
    }


# ─────────────────────────────────────────────
# PROVIDER-BASED ANALYSIS (Multi-Model)
# ─────────────────────────────────────────────

def analyse_target_with_provider(provider: 'LLMProvider', target: str, raw_scan: str) -> dict:
    """
    Analyze target using a dynamic LLM provider.
    Replaces ask_ollama() with provider.ask() for multi-model support.
    """
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"""TARGET: {target}

RECON DATA:
{raw_scan}

Analyze this target completely. Use [TOOL:] or [SEARCH:] if you need more information.
List all vulnerabilities, fixes, and suggest exploits where applicable."""
        }
    ]

    final_response = ""

    for loop in range(MAX_TOOL_LOOPS):
        # Use provider abstraction instead of ask_ollama()
        response = provider.ask(messages)

        print(f"\n{'─'*60}")
        print(f"[METATRON - Round {loop + 1}]")
        print(f"{'─'*60}")
        print(response)

        final_response = response

        tool_calls = extract_tool_calls(response)
        if not tool_calls:
            print("\n[*] No tool calls. Analysis complete.")
            break

        tool_results = run_tool_calls(tool_calls)

        # add assistant response and tool results as new messages
        messages.append({
            "role": "assistant",
            "content": response
        })
        messages.append({
            "role": "user",
            "content": f"""[TOOL RESULTS]
{tool_results}

Continue your analysis with this new information.
If analysis is complete, give the final RISK_LEVEL and SUMMARY."""
        })

    vulnerabilities = parse_vulnerabilities(final_response)
    exploits        = parse_exploits(final_response)
    risk_level      = parse_risk_level(final_response)
    summary         = parse_summary(final_response)

    print(f"\n[+] Parsed: {len(vulnerabilities)} vulns, {len(exploits)} exploits | Risk: {risk_level}")

    return {
        "full_response":   final_response,
        "vulnerabilities": vulnerabilities,
        "exploits":        exploits,
        "risk_level":      risk_level,
        "summary":         summary,
        "raw_scan":        raw_scan
    }


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("[ llm.py test — direct AI query ]\n")

    # test if ollama is reachable
    try:
        r = requests.get("http://localhost:11434", timeout=5)
        print("[+] Ollama is running.")
    except Exception:
        print("[!] Ollama not reachable. Run: ollama serve")
        exit(1)

    target = input("Test target: ").strip()
    test_scan = f"Test recon for {target} — nmap and whois data would appear here."
    result = analyse_target(target, test_scan)

    print(f"\nRisk Level : {result['risk_level']}")
    print(f"Summary    : {result['summary']}")
    print(f"Vulns found: {len(result['vulnerabilities'])}")
    print(f"Exploits   : {len(result['exploits'])}")
