# ui/sidebar.py
import streamlit as st
import os
from registry import list_sites, get_display_name, load_topology, get_paths
from utils.const import SCENARIO_MAP
from utils.llm_helper import get_rate_limiter, GENAI_AVAILABLE

def render_sidebar():
    with st.sidebar:
        st.header("⚡ 拠点シナリオ設定")
        sites = list_sites()
        
        for site_id in sites:
            display_name = get_display_name(site_id)
            with st.expander(f"📍 {display_name}", expanded=True):
                category = st.selectbox("カテゴリ", list(SCENARIO_MAP.keys()), key=f"cat_{site_id}")
                scenarios = SCENARIO_MAP[category]
                current = st.session_state.site_scenarios.get(site_id, "正常稼働")
                
                idx = 0
                for i, s in enumerate(scenarios):
                    if s == current or current in s: idx = i; break
                
                selected = st.radio("シナリオ", scenarios, index=idx, key=f"scenario_{site_id}")
                
                if selected != current:
                    st.session_state.site_scenarios[site_id] = selected
                    # Clear site-specific cache
                    keys = [k for k in list(st.session_state.report_cache.keys()) if site_id in k]
                    for k in keys: del st.session_state.report_cache[k]
                    if st.session_state.active_site == site_id:
                        st.session_state.generated_report = None
                        st.session_state.remediation_plan = None
                        st.session_state.messages = []
        
        st.divider()
        with st.expander("🛠️ メンテナンス設定"):
            for site_id in sites:
                st.session_state.maint_flags[site_id] = st.checkbox(
                    get_display_name(site_id), 
                    value=st.session_state.maint_flags.get(site_id, False),
                    key=f"maint_{site_id}"
                )
        
        st.divider()
        _render_weak_signal_injection()
        
        return _render_api_key_input()

def _render_weak_signal_injection():
    with st.expander("🔮 予兆シミュレーション", expanded=True):
        active = st.session_state.get("active_site")
        # (Simplified device listing logic for brevity, assuming similar to original)
        target_device = st.text_input("対象デバイスID", value="WAN_ROUTER_01", key="pred_target_input")
        scenario_type = st.selectbox("劣化シナリオ", ["Optical Decay", "Microburst", "Route Instability"], key="pred_scenario")
        level = st.slider("劣化度", 0, 5, 0, key="pred_level")
        
        if level > 0:
            msg = f"Simulated {scenario_type} degradation level {level}"
            st.session_state["injected_weak_signal"] = {
                "device_id": target_device, "messages": [msg], "message": msg,
                "level": level, "scenario": scenario_type
            }
            st.info(f"💉 Signal Injected: {msg}")
        else:
            st.session_state["injected_weak_signal"] = None

def _render_api_key_input():
    api_key = None
    if GENAI_AVAILABLE:
        if "GOOGLE_API_KEY" in st.secrets: api_key = st.secrets["GOOGLE_API_KEY"]
        else: api_key = os.environ.get("GOOGLE_API_KEY")
        
        if api_key:
            st.success("✅ API 接続済み")
        else:
            st.warning("⚠️ API Key未設定")
            api_key = st.text_input("Google API Key", type="password")
    return api_key
