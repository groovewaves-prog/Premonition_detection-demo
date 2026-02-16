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
def _hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]

def _pick_first(mapping: dict, keys: list, default: str = "") -> str:
    for k in keys:
        try:
            v = mapping.get(k)
            if v: return str(v)
        except: pass
    return default

def _build_ci_context_for_chat(topology: dict, target_node_id: str) -> dict:
    node = topology.get(target_node_id)
    md = node.metadata if node and hasattr(node, 'metadata') else (node.get('metadata', {}) if isinstance(node, dict) else {})
    ci = {
        "device_id": target_node_id or "",
        "hostname": _pick_first(md, ["hostname", "host", "name"], default=(target_node_id or "")),
        "vendor": _pick_first(md, ["vendor", "manufacturer"], default=""),
        "os": _pick_first(md, ["os", "platform"], default=""),
        "model": _pick_first(md, ["model", "hw_model"], default=""),
        "role": _pick_first(md, ["role", "type"], default=""),
    }
    try:
        conf = load_config_by_id(target_node_id) if target_node_id else ""
        if conf: ci["config_excerpt"] = conf[:1500]
    except Exception: pass
    return ci

def run_diagnostic_simulation_no_llm(scenario: str, target_node) -> dict:
    dev_id = getattr(target_node, "id", "UNKNOWN") if target_node else "UNKNOWN"
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"[PROBE] ts={ts}", f"[PROBE] scenario={scenario}", f"[PROBE] target_device={dev_id}", ""]
    if "WAN" in scenario: lines += ["show ip interface brief", "GigabitEthernet0/0 down down", "Neighbor 203.0.113.2 Idle"]
    elif "FW" in scenario: lines += ["show chassis cluster status", "Redundancy group 0: degraded", "control link: down"]
    else: lines += ["show system alarms", "No active alarms"]
    return {"status": "SUCCESS", "sanitized_log": "\n".join(lines), "device_id": dev_id}

# =====================================================
# メイン描画関数
# =====================================================
def render_incident_cockpit(site_id: str, api_key: Optional[str]):
    display_name = get_display_name(site_id)
    scenario = st.session_state.site_scenarios.get(site_id, "正常稼働")
    
    # 1. ヘッダー
    col_header = st.columns([4, 1])
    with col_header[0]:
        st.markdown(f"### 🛡️ AIOps インシデント・コックピット")
    with col_header[1]:
        if st.button("🔙 一覧に戻る", key="back_to_list"):
            st.session_state.active_site = None
            st.rerun()

    # データ構築
    paths = get_paths(site_id)
    topology = load_topology(paths.topology_path)
    alarms = generate_alarms_for_scenario(topology, scenario)
    status = get_status_from_alarms(scenario, alarms)
    
    # 予兆注入
    injected = st.session_state.get("injected_weak_signal")
    if injected and injected["device_id"] in topology:
        for m in injected.get("messages", []):
            alarms.append(Alarm(injected["device_id"], m, "INFO", False))

    engine_key = f"engine_{site_id}"
    if engine_key not in st.session_state.logic_engines:
        st.session_state.logic_engines[engine_key] = LogicalRCA(topology)
    engine = st.session_state.logic_engines[engine_key]
    results = engine.analyze(alarms) if alarms else []

    # 2. KPIメトリクス
    st.markdown("---")
    k1, k2, k3 = st.columns(3)
    k1.metric("🚨 ステータス", f"{get_status_icon(status)} {status}")
    k2.metric("📊 アラーム数", f"{len(alarms)}件")
    k3.metric("🎯 被疑箇所", f"{len([r for r in results if r.get('prob', 0) > 0.5])}件")
    st.markdown("---")

    # =====================================================
    # 🔮 【新規追加】予兆判断を助けるUI (Future Radar)
    # =====================================================
    preds = [r for r in results if r.get('is_prediction')]
    if preds:
        st.markdown("### 🔮 AIOps Future Radar (Precognition)")
        for p in preds:
            with st.container():
                # 「いつ・どうする・影響範囲」を現場の言葉で表示
                st.markdown(f"""
                <div style="border: 2px solid #E1BEE7; border-left: 10px solid #9C27B0; background-color: #F3E5F5; padding: 15px; border-radius: 8px; margin-bottom: 10px;">
                    <div style="display:flex; justify-content:space-between;">
                        <span style="font-weight:bold; color:#4A148C; font-size:1.1em;">📍 {p['id']} : {p.get('label', '').replace('🔮 [予兆] ', '')}</span>
                        <span style="color:#880E4F; font-weight:bold;">確信度 {p.get('prob', 0)*100:.0f}%</span>
                    </div>
                    <div style="margin-top:10px; display:grid; grid-template-columns: 1fr 1.2fr 1fr; gap:15px;">
                        <div style="background:white; padding:8px; border-radius:4px;">
                            <small style="color:#666;">急性期(Critical)まで</small><br><b>あと約 {p.get('prediction_time_to_critical_min', 0)} 分</b>
                        </div>
                        <div style="background:white; padding:8px; border-radius:4px;">
                            <small style="color:#666;">影響の広がり</small><br>配下 <b>{p.get('prediction_affected_count', 0)} 台</b> のリスク
                        </div>
                        <div style="background:white; padding:8px; border-radius:4px;">
                            <small style="color:#666;">早期捕捉</small><br><b>{p.get('prediction_early_warning_hours', 0)}時間前</b> に検知
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # 推奨アクション (Primary Actions)
                rec_actions = p.get("recommended_actions", [])
                if rec_actions:
                    cols_act = st.columns(len(rec_actions))
                    for idx, act in enumerate(rec_actions):
                        with cols_act[idx]:
                            st.info(f"👉 **{act['title']}**\n\n{act['effect']}")
        st.markdown("---")

    # =====================================================
    # 根本原因と影響範囲の分離表示
    # =====================================================
    root_ids = {a.device_id for a in alarms if a.is_root_cause}
    ds_ids = {a.device_id for a in alarms if not a.is_root_cause}
    rc_list = [r for r in results if r.get('is_prediction') or r['id'] in root_ids or r.get('prob', 0) > 0.8]
    ds_list = [r for r in results if r['id'] in ds_ids and r['id'] not in root_ids]

    if rc_list and ds_list:
        st.info(f"📍 **根本原因**: {rc_list[0]['id']} → 影響範囲: 配下 {len(ds_list)} 機器")

    # 根本原因候補テーブル
    if rc_list:
        st.markdown("#### 🎯 根本原因候補")
        df_rc = pd.DataFrame([{
            "順位": i+1,
            "ステータス": "🔮 予兆" if x.get('is_prediction') else "🔴 危険 (根本原因)" if x['prob']>=0.9 else "🟡 警告",
            "デバイス": x['id'], "原因": x.get('label'), "確信度": f"{x['prob']*100:.0f}%",
            "推奨アクション": "🚀 自動修復が可能" if x['prob']>=0.8 else "🔍 詳細調査",
            "_obj": x
        } for i, x in enumerate(rc_list)])
        
        event = st.dataframe(df_rc.drop(columns=["_obj"]), use_container_width=True, hide_index=True, selection_mode="single-row", on_select="rerun")
        if event.selection and len(event.selection.rows) > 0:
            st.session_state.selected_candidate = df_rc.iloc[event.selection.rows[0]]["_obj"]
        elif not st.session_state.get("selected_candidate") and rc_list:
            st.session_state.selected_candidate = rc_list[0]

    # 上流復旧待ちリスト
    if ds_list:
        with st.expander(f"▼ 影響を受けている機器 ({len(ds_list)}台) - 上流復旧待ち", expanded=False):
            st.dataframe(pd.DataFrame([{"No": i+1, "デバイス": d['id'], "状態": "⚫ 応答なし", "備考": "上流復旧待ち"} for i, d in enumerate(ds_list)]), use_container_width=True, hide_index=True)

    # 3. 2カラムレイアウト (トポロジー / 分析)
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
            res = st.session_state.live_result
            with st.container(border=True):
                if st.session_state.get("verification_result"):
                    v = st.session_state.verification_result
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Ping", v.get('ping_status'))
                    c2.metric("Intf", v.get('interface_status'))
                    c3.metric("HW", v.get('hardware_status'))
                st.divider()
                st.code(res.get("sanitized_log"), language="text")

    with col_r:
        st.subheader("📝 AI Analyst & Chat")
        cand = st.session_state.get("selected_candidate")
        if cand:
            st.info(f"Target: **{cand['id']}**\n{cand.get('label')}")
            tab_rpt, tab_chat = st.tabs(["📝 レポート", "💬 チャット"])
            
            with tab_rpt:
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("📝 詳細レポートを作成 (Generate Report)", use_container_width=True):
                        st.session_state.generated_report = ""
                        placeholder = st.empty()
                        target_conf = load_config_by_id(cand['id'])
                        for chunk in generate_analyst_report_streaming(scenario, topology.get(cand['id']), {"id": cand['id']}, target_conf, "", api_key):
                            st.session_state.generated_report += chunk
                            placeholder.markdown(st.session_state.generated_report + "▌")
                        placeholder.markdown(st.session_state.generated_report)
                
                with col_btn2:
                    if st.button("✨ 復旧プランを作成 (Generate Fix)", use_container_width=True):
                        if not st.session_state.get("generated_report"):
                            st.warning("先に「詳細レポート」を作成してください。")
                        else:
                            st.session_state.remediation_plan = ""
                            placeholder = st.empty()
                            for chunk in generate_remediation_commands_streaming(scenario, st.session_state.generated_report, topology.get(cand['id']), api_key):
                                st.session_state.remediation_plan += chunk
                                placeholder.markdown(st.session_state.remediation_plan + "▌")
                            placeholder.markdown(st.session_state.remediation_plan)

                if st.session_state.generated_report:
                    with st.container(height=400, border=True): st.markdown(st.session_state.generated_report)
                if st.session_state.get("remediation_plan"):
                    st.success("復旧プランが作成されました")
                    st.markdown(st.session_state.remediation_plan)

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
                    st.rerun()
