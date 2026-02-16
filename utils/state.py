# utils/state.py
import streamlit as st

def init_session_state():
    """セッション状態を初期化"""
    defaults = {
        "site_scenarios": {},
        "active_site": None,
        "maint_flags": {},
        "live_result": None,
        "verification_result": None,
        "generated_report": None,
        "remediation_plan": None,
        "verification_log": None,
        "messages": [],
        "chat_session": None,
        "chat_quick_text": "",
        "trigger_analysis": False,
        "logic_engines": {},
        "balloons_shown": False,
        "recovered_devices": {},
        "recovered_scenario_map": {},
        "report_cache": {},
        "injected_weak_signal": None,
        "tuning_report": None, # v45.0
    }
    
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default
