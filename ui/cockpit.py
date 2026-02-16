# ui/cockpit.py (Full UX Restoration + Enhanced Precognition)
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

# --- 復元されたヘルパー関数 ---
def _hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]

def _pick_first(mapping: dict, keys: list, default: str = "") -> str:
    for k in keys:
        try:
            v = mapping.get(k, None)
        except Exception: v = None
        if v is None: continue
        if isinstance(v, (int, float, bool)):
            s = str(v)
            if s: return s
        elif isinstance(v, str):
            if v.strip(): return v.strip()
    return default

def _build_ci_context_for_chat(topology: dict, target_node_id: str) -> dict:
    node = topology.get(target_node_id)
    if node:
        md = node.metadata if hasattr(node, 'metadata') else node.get('metadata', {})
    else: md = {}
    ci = {
        "device_id": target_node_id or "",
        "hostname": _pick_first(md, ["hostname", "host", "name"], default=(target_node_id or "")),
        "vendor": _pick_first(md, ["vendor", "manufacturer"], default=""),
        "model": _pick_first(md, ["model", "hw_model"], default=""),
        "role": _pick_first(md, ["role", "type"], default=""),
        "site": _pick_first(md, ["site", "location"], default=""),
    }
    try:
        conf = load_config_by_id(target_node_id) if target_node_id else ""
        if conf: ci["config_excerpt"] = conf[:1500]
    except Exception: pass
    return ci

def run_diagnostic_simulation_no_llm(selected_scenario: str, target_node_obj) -> dict:
    device_id = getattr(target_node_obj, "id", "UNKNOWN") if target_node_obj else "UNKNOWN"
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"[PROBE] ts={ts}", f"[PROBE] scenario={selected_scenario}", f"[PROBE] target_device={device_id}", ""]
    recovered_devices = st.session_state.get("recovered_devices") or {}
    if recovered_devices.get(device_id):
        lines += ["show chassis cluster status", "Redundancy group 0: healthy", "control link: up"]
    else:
        if "WAN" in selected_scenario: lines += ["show ip interface brief", "GigabitEthernet0/0 down down", "Neighbor 203.0.113.2 Idle"]
        elif "FW" in selected_scenario: lines += ["show chassis cluster status", "Redundancy group 0: degraded", "control link: down"]
        else: lines += ["show system alarms", "No active alarms"]
    return {"status": "SUCCESS", "sanitized_log": "\n".join(lines), "device_id": device_id}

# --- メイン描画 ---
def render_incident_cockpit(site_id: str, api_key: Optional[str]):
    display_name = get_display_name(site_id)
    scenario = st.session_state.site_scenarios.get(site_id, "正常稼働")
    
    # 復元されたヘッダー
    col_header = st.columns([4, 1])
    with col_header[0]: st.markdown(f"### 🛡️ AIOps インシデント・コックピット")
    with col_header[1]:
        st.markdown('<span id="back-btn-marker"></span>', unsafe_allow_html=True)
        st.markdown("<style>#back-btn-marker + div button { background-color: #d32f2f !important; color: white !important; border: 2px solid #b71c1c !important; font-weight: bold !important; }</style>", unsafe_allow_html=True)
        if st.button("🔙 一覧に戻る", key="back_to_list"):
            st.session_state.active_site = None
            st.rerun()
    
    # データ構築
    paths = get_paths(site_id)
    topology = load_topology(paths.topology_path)
    if not topology:
        st.error("トポロジー読み込みエラー")
        return

    alarms = generate_alarms_for_scenario(topology, scenario)
    status = get_status_from_alarms(scenario, alarms)
    
    injected = st.session_state.get("injected_weak_signal")
    if injected and injected["device_id"] in topology:
        for m in injected.get("messages", []):
            alarms.append(Alarm(injected["device_id"], m, "INFO", False))

    engine_key = f"engine_{site_id}"
    if engine_key not in st.session_state.logic_engines:
        st.session_state.logic_engines[engine_key] = LogicalRCA(topology)
    engine = st.session_state.logic_engines[engine_key]
    
    results = engine.analyze(alarms) if alarms else []
    if not results: results = [{"id": "SYSTEM", "label": "正常稼働", "prob": 0.0, "type": "Normal"}]

    # KPI表示 (完全復元)
    preds = [r for r in results if r.get('is_prediction')]
    st.markdown("---")
    k1, k2, k3 = st.columns(3)
    k1.metric("🚨 ステータス", f"{get_status_icon(status)} {status}")
    k2.metric("📊 アラーム数", f"{len(alarms)}件")
    k3.metric("🎯 被疑箇所", f"{len([r for r in results if r.get('prob', 0) > 0.5])}件", delta=f"うち🔮予兆 {len(preds)}件" if preds else None, delta_color="off")
    
    st.markdown("---")

    # =====================================================
    # 🔮 新規: 予兆情報特化UX (Future Radar & Actions)
    # =====================================================
    if preds:
        st.markdown("### 🔮 AIOps Future Radar")
        st.caption("現在のアラートから、デジタルツインが将来の障害を予測しました。")
        for p in preds:
            with st.container():
                # カード形式で予兆を表示
                st.markdown(f"""
                <div style="border: 2px solid #E1BEE7; border-left: 10px solid #9C27B0; background-color: #F3E5F5; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h4 style="margin:0; color:#4A148C;">📍 {p['id']} : {p.get('label', '').replace('🔮 [予兆] ', '')}</h4>
                        <span style="background-color:#9C27B0; color:white; padding:5px 15px; border-radius:20px; font-weight:bold;">確信度 {p.get('prob', 0)*100:.0f}%</span>
                    </div>
                    <div style="margin-top:15px; display:grid; grid-template-columns: 1fr 1fr 1fr; gap:20px;">
                        <div style="background:white; padding:10px; border-radius:5px;">
                            <span style="font-size:0.8em; color:#666;">早期検知</span><br>
                            <b>{p.get('prediction_early_warning_hours', 0)}時間前</b> に兆候を捕捉
                        </div>
                        <div style="background:white; padding:10px; border-radius:5px;">
                            <span style="font-size:0.8em; color:#666;">急性期(Critical)まで</span><br>
                            <b>あと約 {p.get('prediction_time_to_critical_min', 0)} 分</b> で深刻化
                        </div>
                        <div style="background:white; padding:10px; border-radius:5px;">
                            <span style="font-size:0.8em; color:#666;">影響の広がり</span><br>
                            配下 <b>{p.get('prediction_affected_count', 0)} 台</b> が通信断の恐れ
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # 推奨アクション (だからどうする) を明示
                rec_actions = p.get("recommended_actions", [])
                if rec_actions:
                    st.markdown("##### ⚡ 予防的推奨アクション (Primary Actions)")
                    cols_act = st.columns(len(rec_actions))
                    for idx, act in enumerate(rec_actions):
                        with cols_act[idx]:
                            with st.container(border=True):
                                st.markdown(f"**👉 {act['title']}**")
                                st.caption(f"効果: {act['effect']}")

                if st.button(f"🔍 {p['id']} の予兆を詳細分析する", key=f"btn_future_{p['id']}"):
                    st.session_state.selected_candidate = p
        st.markdown("---")

    # =====================================================
    # 復元されたインシデント管理セクション
    # =====================================================
    root_cause_ids = {a.device_id for a in alarms if a.is_root_cause}
    downstream_ids = {a.device_id for a in alarms if not a.is_root_cause}
    rc_list = [r for r in results if r.get('is_prediction') or r['id'] in root_cause_ids or r.get('prob', 0) > 0.5]
    ds_list = [r for r in results if r['id'] in downstream_ids]

    # 青いインフォバーの復元
    if rc_list and ds_list:
        st.info(f"📍 **根本原因**: {rc_list[0]['id']} → 影響範囲: 配下 {len(ds_list)} 機器")

    st.markdown("#### 🎯 根本原因候補")
    df_data = []
    for i, c in enumerate(rc_list, 1):
        p = c.get('prob', 0)
        status_txt = "🔮 予兆" if c.get('is_prediction') else "🔴 危険" if p > 0.9 else "🟡 警告"
        df_data.append({"順位": i, "ステータス": status_txt, "デバイス": c['id'], "原因": c.get('label'), "確信度": f"{p*100:.0f}%", "_obj": c})
    
    df = pd.DataFrame(df_data)
    sel = st.dataframe(df.drop(columns=["_obj"]), use_container_width=True, hide_index=True, selection_mode="single-row", on_select="rerun")
    
    if sel.selection.rows:
        st.session_state.selected_candidate = df.iloc[sel.selection.rows[0]]["_obj"]
    elif rc_list and not st.session_state.get("selected_candidate"):
        st.session_state.selected_candidate = rc_list[0]

    if ds_list:
        with st.expander(f"▼ 影響を受けている機器 ({len(ds_list)}台) - 上流復旧待ち", expanded=False):
            st.dataframe(pd.DataFrame([{"Device": d['id'], "Status": "Unreachable"} for d in ds_list]), use_container_width=True, hide_index=True)

    # 以前の2カラムレイアウトの復元
    col_l, col_r = st.columns([1.2, 1])
    
    with col_l:
        st.subheader("🌐 Network Topology")
        st.graphviz_chart(render_topology_graph(topology, alarms, results), use_container_width=True)
        st.markdown("---")
        st.subheader("🛠️ Auto-Diagnostics")
        if st.button("🚀 診断実行 (Run Diagnostics)", type="primary"):
            with st.status("Agent Operating...", expanded=True) as status:
                res = run_diagnostic_simulation_no_llm(scenario, st.session_state.selected_candidate)
                st.session_state.live_result = res
                st.session_state.verification_result = verify_log_content(res.get('sanitized_log', ""))
                status.update(label="Diagnostics Complete!", state="complete", expanded=False)
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
        st.subheader("📝 AI Analyst Report")
        cand = st.session_state.get("selected_candidate")
        if cand:
            if st.session_state.generated_report is None:
                st.info(f"Target: **{cand['id']}**")
                btn_lbl = "🔮 予兆分析レポート作成" if cand.get('is_prediction') else "📝 詳細レポート作成"
                if st.button(btn_lbl):
                    full_text = ""
                    placeholder = st.empty()
                    for chunk in generate_analyst_report_streaming(scenario, topology.get(cand['id']), {"id": cand['id']}, load_config_by_id(cand['id']), "", api_key):
                        full_text += chunk
                        placeholder.markdown(full_text)
                    st.session_state.generated_report = full_text
            else:
                with st.container(height=400, border=True): st.markdown(st.session_state.generated_report)
                if st.button("🔄 再作成"): st.session_state.generated_report = None; st.rerun()
        
        st.markdown("---")
        st.subheader("🤖 Remediation & Chat")
        with st.expander("💬 Chat with AI Agent", expanded=False):
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
                ci = _build_ci_context_for_chat(topology, cand['id'] if cand else None)
                resp = generate_content_with_retry(st.session_state.chat_session.model, f"Context: {json.dumps(ci)}\nQuestion: {prompt}", stream=False)
                st.session_state.messages.append({"role": "assistant", "content": resp.text})
                st.rerun()
