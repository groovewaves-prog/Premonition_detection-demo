# app.py (Refactored Entry Point)
import streamlit as st
from utils.state import init_session_state
from ui.sidebar import render_sidebar
from ui.dashboard import render_site_status_board, render_triage_center
from ui.cockpit import render_incident_cockpit
from ui.tuning import render_tuning_dashboard

st.set_page_config(page_title="AIOps Cockpit", page_icon="🛡️", layout="wide")

def main():
    init_session_state()
    api_key = render_sidebar()
    
    st.title("🛡️ AIOps インシデント・コックピット")
    
    active_site = st.session_state.get("active_site")
    
    if active_site:
        tab_ops, tab_tune = st.tabs(["🚀 Incident Cockpit", "🔧 Digital Twin Tuning"])
        with tab_ops:
            render_incident_cockpit(active_site, api_key)
        with tab_tune:
            render_tuning_dashboard(active_site)
    else:
        tab1, tab2 = st.tabs(["📊 拠点状態ボード", "🚨 トリアージ"])
        with tab1: render_site_status_board()
        with tab2: render_triage_center()

if __name__ == "__main__":
    main()
