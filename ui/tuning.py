# ui/tuning.py
import streamlit as st
import pandas as pd
import sqlite3
import os

def render_tuning_dashboard(site_id: str):
    st.subheader("🔧 Digital Twin Tuning & Audit")
    
    engine_key = f"engine_{site_id}"
    if engine_key not in st.session_state.logic_engines:
        st.warning("Engine not initialized.")
        return
    
    dt_engine = st.session_state.logic_engines[engine_key].digital_twin
    if not dt_engine:
        st.error("Digital Twin Engine unavailable.")
        return

    tab1, tab2, tab3 = st.tabs(["⚡ Auto-Tuning", "📜 Audit Log", "🛑 Maintenance"])
    
    with tab1:
        col1, col2 = st.columns([1, 3])
        if col1.button("🔄 提案生成"):
            with st.spinner("Analyzing..."):
                report = dt_engine.generate_tuning_report(days=30)
                st.session_state["tuning_report"] = report
        
        report = st.session_state.get("tuning_report")
        if report and report.get("tuning_proposals"):
            for p in report["tuning_proposals"]:
                with st.expander(f"{p['rule_pattern']} ({p['apply_recommendation']['apply_mode']})", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Recall", p["current_stats"]["recall"])
                    c2.metric("New Threshold", p["proposal"]["paging_threshold"])
                    c3.metric("Impact", f"FP -{p['expected_impact']['fp_reduction']*100:.0f}%")
                    
                    if p['apply_recommendation']['apply_mode'] == 'auto':
                        st.success("✅ Auto-Eligible")
                    
                    if st.button(f"Approve & Apply", key=f"ap_{p['rule_pattern']}"):
                        res = dt_engine.apply_tuning_proposals_if_auto([p])
                        st.write(res)
        else:
            st.info("No active proposals.")

    with tab2:
        db_path = dt_engine.storage.paths["sqlite_db"]
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            df = pd.read_sql("SELECT timestamp, event_type, rule_pattern, status FROM audit_log ORDER BY timestamp DESC LIMIT 50", conn)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            st.dataframe(df, use_container_width=True)
            conn.close()

    with tab3:
        if st.button("🚑 DB Repair (Self-Healing)"):
            if dt_engine.repair_db_from_rules_json():
                st.success("Repaired!")
            else:
                st.error("Failed.")
