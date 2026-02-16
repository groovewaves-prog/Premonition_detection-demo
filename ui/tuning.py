import streamlit as st
import pandas as pd
import sqlite3
import os
import json

def render_tuning_dashboard(site_id: str):
    """
    Digital Twin チューニング & 監査画面
    エンジンの初期化状態をチェックし、以前のUXを提供します。
    """
    st.subheader("🔧 Digital Twin Tuning & Audit")
    
    # --- 1. エンジンの初期化状態チェック (Fix: image_8a20c3.png) ---
    engine_key = f"engine_{site_id}"
    if engine_key not in st.session_state.logic_engines:
        st.error("分析エンジンが準備できていません。コックピット画面を一度開いてください。")
        return
    
    # LogicalRCA 内部の digital_twin プロパティにアクセス
    engine = st.session_state.logic_engines[engine_key]
    dt_engine = getattr(engine, "digital_twin", None)
    
    if not dt_engine:
        st.error("Digital Twin Engine unavailable. (エンジンモジュールがロードされていません)")
        return

    # --- 2. タブ構成の描画 ---
    tab1, tab2, tab3 = st.tabs(["⚡ Auto-Tuning", "📜 Audit Log", "🛑 Maintenance"])
    
    # --- Tab 1: Auto-Tuning (Shadow Mode 提案) ---
    with tab1:
        st.caption("AIによる閾値自動調整の提案を確認し、適用します。")
        
        col1, col2 = st.columns([1, 3])
        if col1.button("🔄 提案を生成 (Generate)"):
            with st.spinner("Analyzing prediction history..."):
                # 過去30日のデータに基づきレポート生成
                report = dt_engine.generate_tuning_report(days=30)
                st.session_state["tuning_report"] = report
        
        report = st.session_state.get("tuning_report")
        if report and report.get("tuning_proposals"):
            for p in report["tuning_proposals"]:
                rule_pattern = p['rule_pattern']
                rec = p['apply_recommendation']
                
                with st.expander(f"📦 {rule_pattern} ({rec['apply_mode']})", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Recall (再現率)", f"{p['current_stats']['recall']:.2f}")
                    c2.metric("New Threshold", f"{p['proposal']['paging_threshold']:.2f}")
                    c3.metric("FP Reduction", f"-{p['expected_impact']['fp_reduction']*100:.0f}%", delta_color="normal")
                    
                    st.markdown(f"**理由:** {rec.get('shadow_note')}")
                    
                    if rec['apply_mode'] == 'auto':
                        st.success("✅ Auto-Eligible (推奨)")
                    
                    # 承認ボタン（以前のUX通り、その場で適用）
                    if st.button(f"承認して適用 (Apply)", key=f"ap_{rule_pattern}"):
                        res = dt_engine.apply_tuning_proposals_if_auto([p])
                        if res['applied']:
                            st.success(f"適用完了: {res['applied']}")
                        else:
                            st.error(f"適用失敗/スキップ: {res['skipped']}")
        else:
            st.info("現在、適用すべき新しい提案はありません。")

    # --- Tab 2: Audit Log (ローカルSQLite監査ログ) ---
    with tab2:
        st.caption("システムに加えられた変更の監査ログを表示します。")
        
        db_path = dt_engine.storage.paths["sqlite_db"]
        if os.path.exists(db_path):
            try:
                # 以前のクエリ形式を維持
                conn = sqlite3.connect(db_path)
                df = pd.read_sql(
                    "SELECT timestamp, event_type, actor, rule_pattern, status FROM audit_log ORDER BY timestamp DESC LIMIT 50", 
                    conn
                )
                conn.close()
                
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("監査ログはまだありません。")
            except Exception as e:
                st.error(f"ログ読み込みエラー: {e}")
        else:
            st.warning("監査データベースが見つかりません。")

    # --- Tab 3: Maintenance (自己修復機能) ---
    with tab3:
        st.markdown("#### System Maintenance")
        st.caption("エンジンの内部状態を正常化するためのコンソールです。")
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            if st.button("🚑 DB Repair (Self-Healing)"):
                if dt_engine.repair_db_from_rules_json():
                    st.success("データベースを rules.json から正常に復元しました。")
                else:
                    st.error("復元に失敗しました。")
        
        with col_m2:
            if st.button("🧹 Cache Clear"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("キャッシュをクリアしました。")

        st.divider()
        st.markdown("##### 📂 エンジン構成情報")
        st.json({
            "Tenant ID": dt_engine.tenant_id,
            "Rules SOT": dt_engine._rules_sot,
            "DB Path": dt_engine.storage.paths["sqlite_db"]
        })
