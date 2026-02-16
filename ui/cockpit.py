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

# =====================================================
# 復元されたヘルパー関数
# =====================================================
def _pick_first(mapping: dict, keys: list, default: str = "") -> str:
    """マッピングから最初の非空値を取得"""
    for k in keys:
        try:
            v = mapping.get(k)
            if v: return str(v)
        except: pass
    return default

def _build_ci_context_for_chat(topology: dict, target_node_id: str) -> dict:
    """チャット用のCIコンテキストを構築"""
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

def run_diagnostic_simulation_no_llm(scenario: str, target_node) -> dict:
    """診断シミュレーションの疑似実行"""
    dev_id = getattr(target_node, "id", "UNKNOWN") if target_node else "UNKNOWN"
    lines = [f"[PROBE] scenario={scenario}", f"target={dev_id}", ""]
    if "WAN" in scenario:
        lines += ["show ip int brief", "Gi0/0/0 UP", "BGP State: Established"]
    else:
        lines += ["show system alarms", "No active alarms"]
    return {"status": "SUCCESS", "sanitized_log": "\n".join(lines), "device_id": dev_id}

# =====================================================
# メイン描画関数
# =====================================================
def render_incident_cockpit(site_id: str, api_key: Optional[str]):
    display_name = get_display_name(site_id)
    scenario = st.session_state.site_scenarios.get(site_id, "正常稼働")
    
    # 復元されたヘッダーと「戻る」ボタン
    col_header = st.columns([4, 1])
    with col_header[0]:
        st.markdown(f"### 🛡️ AIOps インシデント・コックピット")
    with col_header[1]:
        st.markdown('<span id="back-btn-marker"></span>', unsafe_allow_html=True)
        st.markdown("""
        <style>
        #back-btn-marker + div button {
            background-color: #d32f2f !important;
            color: white !important;
            font-weight: bold !important;
        }
        </style>
        """, unsafe_allow_html=True)
        if st.button("🔙 一覧に戻る", key="back_btn"):
            st.session_state.active_site = None
            st.rerun()

    # データ構築
    paths = get_paths(site_id)
    topology = load_topology(paths.topology_path)
    alarms = generate_alarms_for_scenario(topology, scenario)
    
    # 予兆シグナル注入
    injected = st.session_state.get("injected_weak_signal")
    if injected and injected["device_id"] in topology:
        for m in injected.get("messages", []):
            alarms.append(Alarm(injected["device_id"], m, "INFO", False))

    engine_key = f"engine_{site_id}"
    if engine_key not in st.session_state.logic_engines:
        st.session_state.logic_engines[engine_key] = LogicalRCA(topology)
    engine = st.session_state.logic_engines[engine_key]
    
    results = engine.analyze(alarms) if alarms else []
    status = get_status_from_alarms(scenario, alarms)

    # 復元されたKPIメトリクス表示
    st.markdown("---")
    k1, k2, k3 = st.columns(3)
    k1.metric("🚨 ステータス", f"{get_status_icon(status)} {status}")
    k2.metric("📊 アラーム数", f"{len(alarms)}件")
    k3.metric("🎯 被疑箇所", f"{len([r for r in results if r.get('prob', 0) > 0.5])}件")
    st.markdown("---")

    # =====================================================
    # 根本原因と影響範囲の厳密な分離
    # =====================================================
    root_ids = {a.device_id for a in alarms if a.is_root_cause}
    ds_ids = {a.device_id for a in alarms if not a.is_root_cause}
    
    # 根本原因候補: アラームで根本原因判定されたもの、または予兆
    rc_list = [r for r in results if r.get('is_prediction') or r['id'] in root_ids]
    # 影響デバイス: 根本原因以外のアラームが出ているもの
    ds_list = [r for r in results if r['id'] in ds_ids and r['id'] not in root_ids]

    # 青帯バナーの復元
    if rc_list and ds_list:
        st.info(f"📍 **根本原因**: {rc_list[0]['id']} → 影響範囲: 配下 {len(ds_list)} 機器")

    # Future Radar (予兆がある場合のみ表示)
    preds = [r for r in rc_list if r.get('is_prediction')]
    if preds:
        with st.container(border=True):
            st.markdown("##### 🔮 AIOps Future Radar (Precognition)")
            for p in preds:
                st.warning(f"⚠️ **{p['id']}**: 深刻な障害へ進展する恐れがあります。急性期まで残り約{p.get('prediction_time_to_critical_min', 60)}分")
                # 推奨アクションの即時提示
                rec_actions = p.get("recommended_actions", [])
                if rec_actions:
                    st.markdown(f"👉 **まずやるべきこと:** {rec_actions[0]['title']} ({rec_actions[0]['effect']})")

    # 根本原因候補テーブルの描画
    if rc_list:
        st.markdown("#### 🎯 根本原因候補")
        df_rc = pd.DataFrame([{
            "順位": i+1,
            "ステータス": "🔮 予兆" if x.get('is_prediction') else "🔴 危険 (根本原因)" if x['prob']>=0.9 else "🟡 警告",
            "デバイス": x['id'],
            "原因": x.get('label'),
            "確信度": f"{x['prob']*100:.0f}%",
            "推奨アクション": "🚀 自動修復が可能" if x['prob']>=0.8 else "🔍 詳細調査",
            "_obj": x
        } for i, x in enumerate(rc_list)])
        
        event = st.dataframe(
            df_rc.drop(columns=["_obj"]), 
            use_container_width=True, 
            hide_index=True, 
            selection_mode="single-row", 
            on_select="rerun"
        )
        
        if event.selection and len(event.selection.rows) > 0:
            st.session_state.selected_candidate = df_rc.iloc[event.selection.rows[0]]["_obj"]
        elif rc_list and not st.session_state.get("selected_candidate"):
            st.session_state.selected_candidate = rc_list[0]

    # 影響を受けている機器リストの復元
    if ds_list:
        with st.expander(f"▼ 影響を受けている機器 ({len(ds_list)}台) - 上流復旧待ち", expanded=False):
            st.dataframe(
                pd.DataFrame([{"No": i+1, "デバイス": d['id'], "状態": "⚫ 応答なし", "備考": "上流復旧待ち"} for i, d in enumerate(ds_list)]), 
                use_container_width=True, 
                hide_index=True
            )

    # 以前の2カラムレイアウトの維持
    col_l, col_r = st.columns([1.2, 1])
    
    # === 左カラム: トポロジー & 診断 ===
    with col_l:
        st.subheader("🌐 Network Topology")
        st.graphviz_chart(render_topology_graph(topology, alarms, results), use_container_width=True)
        
        st.markdown("---")
        st.subheader("🛠️ Auto-Diagnostics")
        if st.button("🚀 診断実行 (Run Diagnostics)", type="primary"):
            with st.status("エージェントが診断ログを収集中..."):
                res = run_diagnostic_simulation_no_llm(scenario, st.session_state.get("selected_candidate"))
                st.session_state.live_result = res
            st.rerun()
        
        if st.session_state.get("live_result"):
            res = st.session_state.live_result
            st.markdown("#### 📄 Diagnostic Results")
            st.code(res.get("sanitized_log"), language="text")

    # === 右カラム: AI分析 & チャット ===
    with col_r:
        st.subheader("📝 AI Analyst & Chat")
        cand = st.session_state.get("selected_candidate")
        if cand:
            st.info(f"Target: **{cand['id']}**\n{cand.get('label')}")
            
            tab_rpt, tab_chat = st.tabs(["📝 レポート", "💬 チャット"])
            with tab_rpt:
                if st.button("📝 詳細レポートを作成"):
                    with st.spinner("AIがトポロジーとログを分析中..."):
                        # レポート生成ロジックを呼び出し
                        time.sleep(1) # シミュレーション
                        st.session_state.generated_report = f"### 分析レポート: {cand['id']}\nデジタルツインの推論に基づき、..."
                
                if st.session_state.generated_report:
                    st.markdown(st.session_state.generated_report)
            
            with tab_chat:
                if not st.session_state.get("chat_session") and api_key:
                    genai.configure(api_key=api_key)
                    st.session_state.chat_session = genai.GenerativeModel("gemma-3-12b-it").start_chat(history=[])
                
                chat_cont = st.container(height=300)
                with chat_cont:
                    for msg in st.session_state.get("messages", []):
                        st.markdown(f"**{'🤖' if msg['role']=='assistant' else '👤'}**: {msg['content']}")
                
                prompt = st.chat_input("AIに質問...")
                if prompt:
                    st.session_state.setdefault("messages", []).append({"role": "user", "content": prompt})
                    # LLM呼び出し処理
                    st.rerun()
