import streamlit as st
import pandas as pd
import json
import time
import hashlib
from typing import Optional, List, Dict, Any

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from registry import get_paths, load_topology, get_display_name
from alarm_generator import generate_alarms_for_scenario, Alarm
from inference_engine import LogicalRCA
from network_ops import (
    generate_analyst_report_streaming, 
    generate_remediation_commands_streaming, 
    run_remediation_parallel_v2, 
    RemediationEnvironment
)
from utils.helpers import get_status_from_alarms, get_status_icon, load_config_by_id
from utils.llm_helper import get_rate_limiter, generate_content_with_retry
from verifier import verify_log_content
from .graph import render_topology_graph

# =====================================================
# 復元された以前のロジック
# =====================================================
def _hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]

def run_diagnostic_simulation_no_llm(scenario: str, target_node) -> dict:
    dev_id = getattr(target_node, "id", "UNKNOWN") if target_node else "UNKNOWN"
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"[PROBE] ts={ts}", f"[PROBE] scenario={scenario}", f"[PROBE] target_device={dev_id}", ""]
    if "WAN" in scenario: lines += ["show ip interface brief", "GigabitEthernet0/0 down down", "Neighbor 203.0.113.2 Idle"]
    elif "FW" in scenario: lines += ["show chassis cluster status", "Redundancy group 0: degraded", "control link: down"]
    else: lines += ["show system alarms", "No active alarms"]
    return {"status": "SUCCESS", "sanitized_log": "\n".join(lines), "device_id": dev_id}

# =====================================================
# メイン描画関数 (image_6c9089.png のUXを完全再現)
# =====================================================
def render_incident_cockpit(site_id: str, api_key: Optional[str]):
    scenario = st.session_state.site_scenarios.get(site_id, "正常稼働")
    
    # 以前のヘッダー
    col_h1, col_h2 = st.columns([4, 1])
    with col_h1: st.markdown(f"### 🛡️ AIOps インシデント・コックピット")
    with col_h2:
        if st.button("🔙 一覧に戻る", key="back_to_list"):
            st.session_state.active_site = None
            st.rerun()

    # データ読み込み
    paths = get_paths(site_id)
    topology = load_topology(paths.topology_path)
    alarms = generate_alarms_for_scenario(topology, scenario)
    status = get_status_from_alarms(scenario, alarms)
    
    # 分析エンジンの初期化
    engine_key = f"engine_{site_id}"
    if engine_key not in st.session_state.logic_engines:
        st.session_state.logic_engines[engine_key] = LogicalRCA(topology)
    engine = st.session_state.logic_engines[engine_key]
    results = engine.analyze(alarms) if alarms else []

    # KPIメトリクス (image_885725.png の完全再現)
    st.markdown("---")
    k1, k2, k3 = st.columns(3)
    k1.metric("🚨 ステータス", f"{get_status_icon(status)} {status}")
    k2.metric("📊 アラーム数", f"{len(alarms)}件")
    k3.metric("🎯 被疑箇所", f"{len([r for r in results if r.get('prob', 0) > 0.5])}件")
    st.markdown("---")

    # =====================================================
    # 根本原因の絞り込みと分離ロジック (image_6c9089.png の再現)
    # =====================================================
    root_ids = {a.device_id for a in alarms if a.is_root_cause}
    ds_ids = {a.device_id for a in alarms if not a.is_root_cause}
    
    # 1台の根本原因と、複数の下流デバイスを厳密に分ける
    rc_list = [r for r in results if r['id'] in root_ids]
    ds_list = [r for r in results if r['id'] in ds_ids and r['id'] not in root_ids]

    # 青帯バナーの復元
    if rc_list and ds_list:
        st.info(f"📍 **根本原因**: {rc_list[0]['id']} → 影響範囲: 配下 {len(ds_list)} 機器")

    # 🎯 根本原因候補テーブル (🔴 危険(根本原因)を表示)
    if rc_list:
        st.markdown("#### 🎯 根本原因候補")
        df_rc = pd.DataFrame([{
            "順位": i+1,
            "ステータス": "🔴 危険 (根本原因)",
            "デバイス": x['id'],
            "原因": x.get('label'),
            "確信度": f"{x['prob']*100:.0f}%",
            "推奨アクション": "🚀 自動修復が可能",
            "_obj": x
        } for i, x in enumerate(rc_list)])
        
        event = st.dataframe(df_rc.drop(columns=["_obj"]), use_container_width=True, hide_index=True, selection_mode="single-row", on_select="rerun")
        if event.selection and len(event.selection.rows) > 0:
            st.session_state.selected_candidate = df_rc.iloc[event.selection.rows[0]]["_obj"]
        elif not st.session_state.get("selected_candidate"):
            st.session_state.selected_candidate = rc_list[0]

    # ▼ 影響を受けている機器リスト (image_8840a6.jpg の再現)
    if ds_list:
        with st.expander(f"▼ 影響を受けている機器 ({len(ds_list)}台) - 上流復旧待ち", expanded=False):
            st.dataframe(pd.DataFrame([{"No": i+1, "デバイス": d['id'], "状態": "⚫ 応答なし", "備考": "上流復旧待ち"} for i, d in enumerate(ds_list)]), use_container_width=True, hide_index=True)

    # ========================================
    # 以前の2カラムレイアウト (image_88b505.png)
    # ========================================
    col_l, col_r = st.columns([1.2, 1])
    
    with col_l:
        st.subheader("🌐 Network Topology")
        st.graphviz_chart(render_topology_graph(topology, alarms, results), use_container_width=True)
        st.markdown("---")
        st.subheader("🛠️ Auto-Diagnostics")
        if st.button("🚀 診断実行 (Run Diagnostics)", type="primary"):
            with st.status("エージェント稼働中..."):
                res = run_diagnostic_simulation_no_llm(scenario, st.session_state.get("selected_candidate"))
                st.session_state.live_result = res
                st.session_state.verification_result = verify_log_content(res.get('sanitized_log', ""))
            st.rerun()
        if st.session_state.get("live_result"):
            st.code(st.session_state.live_result.get("sanitized_log"), language="text")

    with col_r:
        st.subheader("📝 AI Analyst & Chat")
        cand = st.session_state.get("selected_candidate")
        if cand:
            # ターゲットバナー (image_88b505.png)
            st.info(f"Target: **{cand['id']}** {cand.get('label','')}")
            
            tab_rpt, tab_chat = st.tabs(["📝 レポート", "💬 チャット"])
            with tab_rpt:
                c1, c2 = st.columns(2)
                if c1.button("📝 詳細レポートを作成", use_container_width=True):
                    placeholder = st.empty()
                    full_text = ""
                    for chunk in generate_analyst_report_streaming(scenario, topology.get(cand['id']), {"id": cand['id']}, load_config_by_id(cand['id']), "", api_key):
                        full_text += chunk
                        placeholder.markdown(full_text + "▌")
                    st.session_state.generated_report = full_text
                    placeholder.markdown(full_text)
                
                if c2.button("✨ 復旧プランを作成", use_container_width=True):
                    if not st.session_state.get("generated_report"): st.warning("先に詳細レポートを作成してください")
                    else:
                        placeholder = st.empty()
                        full_text = ""
                        for chunk in generate_remediation_commands_streaming(scenario, st.session_state.generated_report, topology.get(cand['id']), api_key):
                            full_text += chunk
                            placeholder.markdown(full_text + "▌")
                        st.session_state.remediation_plan = full_text
                        placeholder.markdown(full_text)

                if st.session_state.generated_report: st.markdown(st.session_state.generated_report)

            with tab_chat:
                if not st.session_state.get("chat_session") and api_key:
                    genai.configure(api_key=api_key)
                    st.session_state.chat_session = genai.GenerativeModel("gemma-3-12b-it").start_chat(history=[])
                # チャット履歴と入力ロジック (以前の app.py 通り)
                prompt = st.chat_input("AIに質問...")
