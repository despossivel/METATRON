#!/usr/bin/env python3
"""
METATRON - metatron.py
Main CLI entry point. Wires db.py + tools.py + search.py + llm.py together.
Multi-model support for local and cloud LLM providers.
Run with: python metatron.py
"""
from export import export_menu
from llm_providers import LLMProviderFactory
from config import (
    load_config, get_active_provider, set_active_provider,
    prompt_provider_selection, prompt_provider_credentials,
    update_provider_config, get_resolved_provider_config,
    show_current_provider, show_provider_status
)
from llm import analyse_target_with_provider
import os
import sys
from db import (
    get_connection,
    initialize_schema,
    create_session,
    save_vulnerability,
    save_fix,
    save_exploit,
    update_summary,
    get_all_history,
    get_latest_history_for_target,
    get_session,
    get_vulnerabilities,
    get_fixes,
    get_exploits,
    edit_vulnerability,
    edit_fix,
    edit_exploit,
    edit_history,
    edit_summary_risk,
    delete_vulnerability,
    delete_exploit,
    delete_fix,
    delete_full_session,
    print_history,
    print_session
)
from datetime import datetime
from tools import interactive_tool_run, format_recon_for_llm, run_default_recon


# ─────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────

def banner():
    os.system("clear")
    print("""
\033[91m
    ███╗   ███╗███████╗████████╗ █████╗ ████████╗██████╗  ██████╗ ███╗   ██╗
    ████╗ ████║██╔════╝╚══██╔══╝██╔══██╗╚══██╔══╝██╔══██╗██╔═══██╗████╗  ██║
    ██╔████╔██║█████╗     ██║   ███████║   ██║   ██████╔╝██║   ██║██╔██╗ ██║
    ██║╚██╔╝██║██╔══╝     ██║   ██╔══██║   ██║   ██╔══██╗██║   ██║██║╚██╗██║
    ██║ ╚═╝ ██║███████╗   ██║   ██║  ██║   ██║   ██║  ██║╚██████╔╝██║ ╚████║
    ╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
\033[0m
    \033[90mAI Penetration Testing Assistant  |  Parrot OS\033[0m
""")
    
    # Show current provider
    try:
        active_provider = get_active_provider()
        config = get_resolved_provider_config(active_provider)
        provider = LLMProviderFactory.create(active_provider, config)
        if provider:
            info = provider.get_model_info()
            print(f"    \033[92mUsing: {info['provider']} ({info['model']})\033[0m")
    except Exception:
        pass
    
    print("    \033[90m─────────────────────────────────────────────────────────────────────\033[0m")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def divider(label=""):
    if label:
        print(f"\n\033[33m{'─'*20} {label} {'─'*20}\033[0m")
    else:
        print(f"\033[90m{'─'*60}\033[0m")


def prompt(text):
    return input(f"\033[36m{text}\033[0m").strip()


def success(text):
    print(f"\033[92m[+] {text}\033[0m")


def warn(text):
    print(f"\033[93m[!] {text}\033[0m")


def error(text):
    print(f"\033[91m[x] {text}\033[0m")


def info(text):
    print(f"\033[94m[*] {text}\033[0m")


def confirm(question: str) -> bool:
    ans = prompt(f"{question} [y/N]: ").lower()
    return ans == "y"


# ─────────────────────────────────────────────
# NEW SCAN
# ─────────────────────────────────────────────

def new_scan():
    divider("NEW SCAN")
    target = prompt("[?] Enter target IP or domain: ")
    if not target:
        warn("No target entered.")
        return

    # check if target was scanned before
    history = get_all_history()
    past = [row for row in history if row[1] == target]
    sl_no = None
    existing_raw_scan = None

    if past:
        warn(f"Target '{target}' has been scanned before ({len(past)} time(s)).")
        if not confirm("Continue with this target and append new scan results to the existing session?"):
            return

        existing = get_latest_history_for_target(target)
        sl_no = existing[0]
        session_data = get_session(sl_no)
        summary_row = session_data.get("summary")
        if summary_row and summary_row[2]:
            existing_raw_scan = summary_row[2]
        success(f"Resuming session — SL# {sl_no}")

    if not sl_no:
        sl_no = create_session(target)
        success(f"Session created — SL# {sl_no}")

    # run recon tools
    divider("RECON")
    info("Choose recon tools to run:")
    new_scan_data = interactive_tool_run(target)

    if not new_scan_data.strip():
        warn("No scan data collected. Aborting.")
        if not past:
            delete_full_session(sl_no)
        return

    if existing_raw_scan:
        raw_scan = (
            f"{existing_raw_scan}\n\n"
            f"{'='*60}\n"
            f"[ADDED SCAN - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n"
            f"{'='*60}\n"
            f"{new_scan_data}"
        )
    else:
        raw_scan = new_scan_data

    # send to AI
    divider("AI ANALYSIS")
    try:
        active_provider = get_active_provider()
        provider_config = get_resolved_provider_config(active_provider)
        provider = LLMProviderFactory.create(active_provider, provider_config)
        
        if not provider:
            error(f"Provider '{active_provider}' not found!")
            delete_full_session(sl_no)
            return
        
        # Validate provider
        is_valid, msg = provider.validate()
        if not is_valid:
            error(f"Provider validation failed: {msg}")
            delete_full_session(sl_no)
            return
        
        info(f"Using {provider.get_model_info()['provider']}")
        
        # Use provider-based analysis
        result = analyse_target_with_provider(provider, target, raw_scan)
    except Exception as e:
        error(f"Analysis error: {e}")
        delete_full_session(sl_no)
        return

    # ── save everything to DB ──────────────────
    divider("SAVING TO DATABASE")

    # save vulnerabilities and their fixes
    for vuln in result["vulnerabilities"]:
        vuln_id = save_vulnerability(
            sl_no,
            vuln["vuln_name"],
            vuln["severity"],
            vuln["port"],
            vuln["service"],
            vuln["description"]
        )
        if vuln.get("fix"):
            save_fix(sl_no, vuln_id, vuln["fix"], source="ai")
        success(f"Saved vuln: {vuln['vuln_name']} [{vuln['severity']}]")

    # save exploits
    for exp in result["exploits"]:
        save_exploit(
            sl_no,
            exp["exploit_name"],
            exp["tool_used"],
            exp["payload"],
            exp["result"],
            exp["notes"]
        )
        success(f"Saved exploit: {exp['exploit_name']}")

    # save or update summary
    update_summary(
        sl_no,
        result["raw_scan"],
        result["full_response"],
        result["risk_level"]
    )

    success(f"All data saved. SL# {sl_no} | Risk: {result['risk_level']}")
    divider()

    # show results and offer edit/delete
    data = get_session(sl_no)
    print_session(data)

    if confirm("Edit or delete anything in this session?"):
        edit_delete_menu(sl_no)


def append_scan_to_session(sl_no: int, target: str):
    divider("APPEND SCAN")
    info("Choose additional scanner tools to run:")
    new_scan_data = interactive_tool_run(target)

    if not new_scan_data.strip():
        warn("No scan data collected. Aborting.")
        return

    session_data = get_session(sl_no)
    summary_row = session_data.get("summary")
    existing_raw_scan = summary_row[2] if summary_row and summary_row[2] else ""

    raw_scan = (
        f"{existing_raw_scan}\n\n"
        f"{'='*60}\n"
        f"[ADDED SCAN - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n"
        f"{'='*60}\n"
        f"{new_scan_data}"
    ) if existing_raw_scan else new_scan_data

    divider("AI ANALYSIS")
    result = analyse_target(target, raw_scan)

    divider("SAVING TO DATABASE")
    for vuln in result["vulnerabilities"]:
        vuln_id = save_vulnerability(
            sl_no,
            vuln["vuln_name"],
            vuln["severity"],
            vuln["port"],
            vuln["service"],
            vuln["description"]
        )
        if vuln.get("fix"):
            save_fix(sl_no, vuln_id, vuln["fix"], source="ai")
        success(f"Saved vuln: {vuln['vuln_name']} [{vuln['severity']}]")

    for exp in result["exploits"]:
        save_exploit(
            sl_no,
            exp["exploit_name"],
            exp["tool_used"],
            exp["payload"],
            exp["result"],
            exp["notes"]
        )
        success(f"Saved exploit: {exp['exploit_name']}")

    update_summary(
        sl_no,
        result["raw_scan"],
        result["full_response"],
        result["risk_level"]
    )

    success(f"Additional scan saved. SL# {sl_no} | Risk: {result['risk_level']}")
    divider()

    data = get_session(sl_no)
    print_session(data)


# ─────────────────────────────────────────────
# VIEW HISTORY
# ─────────────────────────────────────────────

def view_history():
    while True:
        divider("SCAN HISTORY")
        rows = get_all_history()

        if not rows:
            warn("No scans in database yet.")
            return

        print_history(rows)
        print("  [1] View session details")
        print("  [2] Edit history record")
        print("  [3] Delete full session")
        print("  [4] Back")
        divider()

        action = prompt("History action: ")

        if action == "4" or action == "":
            return

        if action not in ("1", "2", "3"):
            warn("Invalid choice.")
            continue

        sl_no_str = prompt("Enter SL#: ")
        if not sl_no_str.isdigit():
            error("Invalid SL#.")
            continue

        sl_no = int(sl_no_str)
        data = get_session(sl_no)
        if not data["history"]:
            error(f"SL# {sl_no} not found.")
            continue

        if action == "1":
            print_session(data)

            if confirm("Run another scanner on this target and append results to this session?"):
                append_scan_to_session(sl_no, data["history"][1])
                data = get_session(sl_no)

            if confirm("Export this session?"):
                export_menu(data)

            if confirm("Edit or delete anything in this session?"):
                edit_delete_menu(sl_no)

        elif action == "2":
            print("  Fields: target / status / scan_date")
            field = prompt("Field to edit: ").strip()
            value = prompt(f"New value for '{field}': ")
            edit_history(sl_no, field, value)

        elif action == "3":
            if confirm(f"\n\033[91mPermanently delete ENTIRE session SL# {sl_no} from all tables?\033[0m"):
                delete_full_session(sl_no)
                success(f"Session SL# {sl_no} wiped.")


# ─────────────────────────────────────────────
# PROVIDER SETTINGS MENU
# ─────────────────────────────────────────────

def provider_settings_menu():
    while True:
        divider("PROVIDER SETTINGS")
        
        # Show current provider
        show_current_provider()
        
        print("  \033[92m[1]\033[0m  Change Provider")
        print("  \033[92m[2]\033[0m  Configure Credentials")
        print("  \033[92m[3]\033[0m  View All Providers Status")
        print("  \033[92m[4]\033[0m  Back to Main Menu")
        divider()
        
        choice = prompt("settings> ")
        
        if choice == "1":
            # Change provider
            provider_name = prompt_provider_selection()
            if provider_name:
                set_active_provider(provider_name)
                success(f"Provider set to: {provider_name}")
                input("\n\033[90mPress Enter to continue...\033[0m")
        
        elif choice == "2":
            # Configure credentials
            provider_name = prompt_provider_selection()
            if provider_name:
                new_config = prompt_provider_credentials(provider_name)
                update_provider_config(provider_name, new_config)
                
                # Validate
                provider_config = get_resolved_provider_config(provider_name)
                provider = LLMProviderFactory.create(provider_name, provider_config)
                if provider:
                    is_valid, msg = provider.validate()
                    if is_valid:
                        success(msg)
                        set_active_provider(provider_name)
                    else:
                        error(msg)
                
                input("\n\033[90mPress Enter to continue...\033[0m")
        
        elif choice == "3":
            # Show all providers status
            show_provider_status()
            input("\n\033[90mPress Enter to continue...\033[0m")
        
        elif choice == "4":
            # Back to main
            return
        
        else:
            warn("Invalid choice.")


# ─────────────────────────────────────────────
# EDIT / DELETE MENU
# ─────────────────────────────────────────────

def edit_delete_menu(sl_no: int):
    while True:
        divider(f"EDIT / DELETE — SL# {sl_no}")
        print("  [1] Edit a vulnerability")
        print("  [2] Edit a fix")
        print("  [3] Edit an exploit")
        print("  [4] Edit risk level")
        print("  [5] Delete a vulnerability")
        print("  [6] Delete a fix")
        print("  [7] Delete an exploit")
        print("  [8] Delete FULL session (all tables)")
        print("  [9] Back")
        divider()

        choice = prompt("Choice: ")

        # ── EDIT VULNERABILITY ─────────────────
        if choice == "1":
            vulns = get_vulnerabilities(sl_no)
            if not vulns:
                warn("No vulnerabilities recorded for this session.")
                continue

            print("\n[ VULNERABILITIES ]")
            for v in vulns:
                print(f"  id={v[0]} | {v[2]} | {v[3]} | port {v[4]} | {v[5]}")

            vid = prompt("Enter vulnerability id to edit: ")
            if not vid.isdigit():
                error("Invalid id.")
                continue

            print("  Fields: vuln_name / severity / port / service / description")
            field = prompt("Field to edit: ").strip()
            value = prompt(f"New value for '{field}': ")
            edit_vulnerability(int(vid), field, value)

        # ── EDIT FIX ──────────────────────────
        elif choice == "2":
            fixes = get_fixes(sl_no)
            if not fixes:
                warn("No fixes recorded for this session.")
                continue

            print("\n[ FIXES ]")
            for f in fixes:
                print(f"  id={f[0]} | vuln_id={f[2]} | {f[3][:80]}")

            fid = prompt("Enter fix id to edit: ")
            if not fid.isdigit():
                error("Invalid id.")
                continue

            new_text = prompt("New fix text: ")
            edit_fix(int(fid), new_text)

        # ── EDIT EXPLOIT ──────────────────────
        elif choice == "3":
            exploits = get_exploits(sl_no)
            if not exploits:
                warn("No exploits recorded for this session.")
                continue

            print("\n[ EXPLOITS ]")
            for e in exploits:
                print(f"  id={e[0]} | {e[2]} | tool: {e[3]} | result: {e[5]}")

            eid = prompt("Enter exploit id to edit: ")
            if not eid.isdigit():
                error("Invalid id.")
                continue

            print("  Fields: exploit_name / tool_used / payload / result / notes")
            field = prompt("Field to edit: ").strip()
            value = prompt(f"New value for '{field}': ")
            edit_exploit(int(eid), field, value)

        # ── EDIT RISK LEVEL ───────────────────
        elif choice == "4":
            print("  Options: CRITICAL / HIGH / MEDIUM / LOW")
            risk = prompt("New risk level: ").upper()
            if risk not in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
                error("Invalid risk level.")
                continue
            edit_summary_risk(sl_no, risk)

        # ── DELETE VULNERABILITY ──────────────
        elif choice == "5":
            vulns = get_vulnerabilities(sl_no)
            if not vulns:
                warn("No vulnerabilities to delete.")
                continue

            print("\n[ VULNERABILITIES ]")
            for v in vulns:
                print(f"  id={v[0]} | {v[2]} | {v[3]}")

            vid = prompt("Enter vulnerability id to delete: ")
            if not vid.isdigit():
                error("Invalid id.")
                continue

            if confirm(f"Delete vulnerability id={vid} and its linked fixes?"):
                delete_vulnerability(int(vid))

        # ── DELETE FIX ────────────────────────
        elif choice == "6":
            fixes = get_fixes(sl_no)
            if not fixes:
                warn("No fixes to delete.")
                continue

            print("\n[ FIXES ]")
            for f in fixes:
                print(f"  id={f[0]} | vuln_id={f[2]} | {f[3][:80]}")

            fid = prompt("Enter fix id to delete: ")
            if not fid.isdigit():
                error("Invalid id.")
                continue

            if confirm(f"Delete fix id={fid}?"):
                delete_fix(int(fid))

        # ── DELETE EXPLOIT ────────────────────
        elif choice == "7":
            exploits = get_exploits(sl_no)
            if not exploits:
                warn("No exploits to delete.")
                continue

            print("\n[ EXPLOITS ]")
            for e in exploits:
                print(f"  id={e[0]} | {e[2]} | result: {e[5]}")

            eid = prompt("Enter exploit id to delete: ")
            if not eid.isdigit():
                error("Invalid id.")
                continue

            if confirm(f"Delete exploit id={eid}?"):
                delete_exploit(int(eid))

        # ── DELETE FULL SESSION ───────────────
        elif choice == "8":
            if confirm(f"\n\033[91mPermanently delete ENTIRE session SL# {sl_no} from all tables?\033[0m"):
                delete_full_session(sl_no)
                success(f"Session SL# {sl_no} wiped.")
                return   # go back to main menu

        # ── BACK ──────────────────────────────
        elif choice == "9":
            break

        else:
            warn("Invalid choice.")


# ─────────────────────────────────────────────
# DB CONNECTION CHECK
# ─────────────────────────────────────────────

def check_db():
    try:
        conn = get_connection()
        conn.close()
        initialize_schema()
        return True
    except Exception as e:
        error(f"MariaDB connection failed: {e}")
        error("Make sure MariaDB is running: sudo systemctl start mariadb")
        return False


# ─────────────────────────────────────────────
# MAIN MENU
# ─────────────────────────────────────────────

def main_menu():
    while True:
        banner()
        print("  \033[92m[1]\033[0m  New Scan")
        print("  \033[92m[2]\033[0m  View History")
        print("  \033[92m[3]\033[0m  Provider Settings")
        print("  \033[92m[4]\033[0m  Exit")
        divider()

        choice = prompt("metatron> ")

        if choice == "1":
            new_scan()
            input("\n\033[90mPress Enter to continue...\033[0m")

        elif choice == "2":
            view_history()
            input("\n\033[90mPress Enter to continue...\033[0m")

        elif choice == "3":
            provider_settings_menu()

        elif choice == "4":
            print("\n\033[91m[*] Shutting down Metatron. Stay legal.\033[0m\n")
            sys.exit(0)

        else:
            warn("Invalid choice.")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if not check_db():
        sys.exit(1)
    
    # Initialize config and validate provider
    try:
        from config import ensure_config_dir, load_config, get_active_provider
        ensure_config_dir()
        load_config()
        
        active_provider = get_active_provider()
        provider_config = get_resolved_provider_config(active_provider)
        provider = LLMProviderFactory.create(active_provider, provider_config)
        
        if not provider:
            warn(f"Warning: Provider '{active_provider}' not found. Please configure in Provider Settings.")
        else:
            is_valid, msg = provider.validate()
            if not is_valid:
                warn(f"Warning: Provider '{active_provider}' validation failed: {msg}")
                warn("Please reconfigure in Provider Settings.")
    except Exception as e:
        warn(f"Warning: Could not initialize provider config: {e}")
    
    main_menu()
