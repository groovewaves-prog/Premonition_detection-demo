import streamlit as st
import pandas as pd
import json
import time
import re
import hashlib
from typing import Optional, List, Dict, Any

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from registry import get_paths, load_topology, get_display_name
from alarm_generator import generate_alarms_for_scenario, Alarm, get_alarm_summary
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

# ヘルパー関数（以前のapp.pyより）
def _pick_first(mapping: dict, keys: list, default: str = "") -> str:
    for k in keys:
        try: v = mapping.get(k)
        except: v = None
        if v: return str(v)
    return default

def _build_ci_context_for_chat(topology: dict, target_node_id: str) -> dict:
    node = topology.get(target_node_id)
    md = node.get('metadata', {}) if node and isinstance(node, dict) else (getattr(node, 'metadata', {}) if node else {})
    ci = {
        "device_id": target_node_id or "",
        "hostname": _pick_first(md, ["hostname", "host"], default=target_node_id or ""),
        "vendor": _pick_first(md, ["vendor"], default=""),
        "model": _pick_first(md, ["model"], default=""),
    }
    try:
        conf = load_config_by_id(target_node_id)
        if conf: ci["config_excerpt"] = conf[:1000]
    except: pass
    return ci

def render_incident_cockpit(site_id: str, api_key: Optional[str]):
    display_name = get_display_name(site_id)
    scenario = st.session_state.site_scenarios.get(site_id, "正常稼働")
    
    # 以前のヘッダー
    col_header = st.columns([4, 1])
    with col_header[0]: st.markdown(f"### 🛡️ AIOps インシデント・コックピット")
    with col_header[1]:
        if st.button("🔙 一覧に戻る", key="back_btn"):
            st.session_state.active_site = None
            st.rerun()

    # データ構築
    paths = get_paths(site_id)
    topology = load_topology(paths.topology_path)
    alarms = generate_alarms_for_scenario(topology, scenario)
    
    injected = st.session_state.get("injected_weak_signal")
    if injected and injected["device_id"] in topology:
        for m in injected.get("messages", []):
            alarms.append(Alarm(injected["device_id"], m, "INFO", False))

    engine_key = f"engine_{site_id}"
    if engine_key not in st.session_state.logic_engines:
        st.session_state.logic_engines[engine_key] = LogicalRCA(topology)
    engine = st.session_state.logic_engines[engine_key]
    
    analysis_results = engine.analyze(alarms) if alarms else []
    status = get_status_from_alarms(scenario, alarms)

    # 以前のKPIメトリクス
    st.markdown("---")
    k1, k2, k3 = st.columns(3)
    k1.metric("🚨 ステータス", f"{get_status_icon(status)} {status}")
    k2.metric("📊 アラーム数", f"{len(alarms)}件")
    k3.metric("🎯 被疑箇所", f"{len([r for r in analysis_results if r.get('prob', 0) > 0.5])}件")
    st.markdown("---")

    # 分離ロジックの復元
    root_ids = {a.device_id for a in alarms if a.is_root_cause}
    ds_ids = {a.device_id for a in alarms if not a.is_root_cause}
    
    rc_cands = [r for r in analysis_results if r.get('is_prediction') or r['id'] in root_ids or r.get('prob', 0) > 0.5]
    ds_devs = [r for r in analysis_results if r['id'] in ds_ids]

    # 青帯バナーの復元
    if rc_cands and ds_devs:
        st.info(f"📍 **根本原因**: {rc_cands[0]['id']} → 影響範囲: 配下 {len(ds_devs)} 機器")

    # 未来予知（Future Radar） - ここだけ追加
    preds = [r for r in rc_cands if r.get('is_prediction')]
    if preds:
        with st.container(border=True):
            st.markdown("##### 🔮 AIOps Future Radar (Precognition)")
            for p in preds:
                time_to = p.get('prediction_time_to_critical_min', 60)
                st.warning(f"⚠️ **{p['id']}**: あと約{time_to}分で深刻な障害へ進展する恐れがあります。")
                st.caption(f"推奨アクション: {p.get('recommended_actions', [{'title': '詳細調査'}])[0]['title']}")

    # 根本原因テーブルの復元
    if rc_cands:
        st.markdown("#### 🎯 根本原因候補")
        df_rc = pd.DataFrame([{
            "ステータス": "🔮 予兆" if x.get('is_prediction') else "🔴 危険" if x['prob']>=0.9 else "🟡 警告",
            "デバイス": x['id'], "原因": x['label'], "確信度": f"{x['prob']*100:.0f}%",
            "_obj": x
        } for x in rc_cands])
        
        event = st.dataframe(df_rc.drop(columns=["_obj"]), use_container_width=True, hide_index=True, selection_mode="single-row", on_select="rerun")
        if event.selection.rows:
            st.session_state.selected_candidate = df_rc.iloc[event.selection.rows[0]]["_obj"]
        elif rc_cands and not st.session_state.get("selected_candidate"):
            st.session_state.selected_candidate = rc_cands[0]

    # 影響を受けている機器リストの復元
    if ds_devs:
        with st.expander(f"▼ 影響を受けている機器 ({len(ds_devs)}台) - 上流復旧待ち", expanded=False):
            st.dataframe(pd.DataFrame([{"No": i+1, "デバイス": d['id'], "状態": "⚫ 応答なし", "備考": "上流復旧待ち"} for i, d in enumerate(ds_devs)]), use_container_width=True, hide_index=True)

    # 以前の2カラムレイアウト
    col_l, col_r = st.columns([1.2, 1])
    
    with col_l:
        st.subheader("🌐 Network Topology")
        st.graphviz_chart(render_topology_graph(topology, alarms, analysis_results), use_container_width=True)
        
        st.markdown("---")
        st.subheader("🛠️ Auto-Diagnostics")
        if st.button("🚀 診断実行 (Run Diagnostics)", type="primary"):
            with st.status("診断中..."):
                time.sleep(1) # シミュレーション
                st.session_state.live_result = {"sanitized_log": "[PROBE] Diagnostics successful."}
            st.rerun()
        if st.session_state.get("live_result"):
            st.code(st.session_state.live_result.get("sanitized_log"), language="text")

    with col_r:
        # インシデントワークスペース
        st.subheader("📝 AI Analyst & Chat")
        cand = st.session_state.get("selected_candidate")
        if cand:
            st.info(f"Target: **{cand['id']}**\n{cand.get('label')}")
            
            # 以前のタブ構成
            t_rpt, t_chat = st.tabs(["📝 レポート", "💬 チャット"])
            with t_rpt:
                if st.button("詳細レポート作成"):
                    st.write("🤖 レポート生成中...")
            with t_chat:
                st.chat_input("AIに質問...")
