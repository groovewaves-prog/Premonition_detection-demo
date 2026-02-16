import streamlit as st
import pandas as pd
import json
import time
import re
import hashlib
from typing import Optional, List, Dict, Any

# Google Generative AI
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
    RemediationEnvironment,
    sanitize_output
)
from utils.helpers import get_status_from_alarms, get_status_icon, load_config_by_id
from utils.llm_helper import get_rate_limiter, generate_content_with_retry
from verifier import verify_log_content
from .graph import render_topology_graph

# =====================================================
# ローカルヘルパー関数
# =====================================================
def _hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]

def _pick_first(mapping: dict, keys: list, default: str = "") -> str:
    for k in keys:
        try:
            v = mapping.get(k, None)
        except Exception:
            v = None
        if v is None: continue
        if isinstance(v, (int, float, bool)):
            s = str(v)
            if s: return s
        elif isinstance(v, str):
            if v.strip(): return v.strip()
    return default

def _build_ci_context_for_chat(topology: dict, target_node_id: str) -> dict:
    node = topology.get(target_node_id) if target_node_id else None
    if node:
        if hasattr(node, 'metadata'):
            md = node.metadata or {}
        else:
            md = node.get('metadata', {}) if isinstance(node, dict) else {}
    else:
        md = {}

    ci = {
        "device_id": target_node_id or "",
        "hostname": _pick_first(md, ["hostname", "host", "name"], default=(target_node_id or "")),
        "vendor": _pick_first(md, ["vendor", "manufacturer"], default=""),
        "os": _pick_first(md, ["os", "platform"], default=""),
        "model": _pick_first(md, ["model", "hw_model"], default=""),
        "role": _pick_first(md, ["role", "type"], default=""),
        "layer": _pick_first(md, ["layer", "level"], default=""),
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
    recovered_map = st.session_state.get("recovered_scenario_map") or {}

    if recovered_devices.get(device_id) and recovered_map.get(device_id) == selected_scenario:
        if "FW" in selected_scenario:
            lines += ["show chassis cluster status", "Redundancy group 0: healthy", "control link: up"]
        elif "WAN" in selected_scenario:
            lines += ["show ip interface brief", "GigabitEthernet0/0 up up", "show ip bgp summary", "Neighbor 203.0.113.2 Established"]
        elif "L2SW" in selected_scenario:
            lines += ["show environment", "Fan: OK", "Temperature: OK", "show interface status", "Uplink: up"]
        else:
            lines += ["show system alarms", "No active alarms", "ping 8.8.8.8 repeat 5", "Success rate is 100 percent"]
        return {"status": "SUCCESS", "sanitized_log": "\n".join(lines), "device_id": device_id}

    if "WAN全回線断" in selected_scenario or "[WAN]" in selected_scenario:
        lines += ["show ip interface brief", "GigabitEthernet0/0 down down", "Neighbor 203.0.113.2 Idle"]
    elif "FW片系障害" in selected_scenario or "[FW]" in selected_scenario:
        lines += ["show chassis cluster status", "Redundancy group 0: degraded", "control link: down"]
    elif "L2SW" in selected_scenario:
        lines += ["show environment", "Fan: FAIL", "Temperature: HIGH", "show interface status", "Uplink: flapping"]
    else:
        lines += ["show system alarms", "No active alarms"]

    return {"status": "SUCCESS", "sanitized_log": "\n".join(lines), "device_id": device_id}

# =====================================================
# メイン描画関数
# =====================================================
def render_incident_cockpit(site_id: str, api_key: Optional[str]):
    display_name = get_display_name(site_id)
    scenario = st.session_state.site_scenarios.get(site_id, "正常稼働")
    
    # ヘッダー & 戻るボタン
    col_header = st.columns([4, 1])
    with col_header[0]:
        st.markdown(f"### 🛡️ AIOps インシデント・コックピット")
    with col_header[1]:
        st.markdown('<span id="back-btn-marker"></span>', unsafe_allow_html=True)
        st.markdown("""
        <style>
        #back-btn-marker + div button,
        #back-btn-marker ~ div[data-testid="stButton"] button {
            background-color: #d32f2f !important;
            color: white !important;
            border: 2px solid #b71c1c !important;
            font-weight: bold !important;
            border-radius: 8px !important;
        }
        #back-btn-marker + div button:hover,
        #back-btn-marker ~ div[data-testid="stButton"] button:hover {
            background-color: #b71c1c !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        if st.button("🔙 一覧に戻る", key="back_to_list"):
            st.session_state.active_site = None
            st.rerun()
    
    # データ読み込み
    paths = get_paths(site_id)
    topology = load_topology(paths.topology_path)
    if not topology:
        st.error("トポロジーが読み込めませんでした。")
        return
    
    alarms = generate_alarms_for_scenario(topology, scenario)
    status = get_status_from_alarms(scenario, alarms)
    
    injected = st.session_state.get("injected_weak_signal")
    if injected and injected["device_id"] in topology:
        for m in injected.get("messages", []):
            alarms.append(Alarm(injected["device_id"], m, "INFO", False))

    # 分析エンジン
    engine_key = f"engine_{site_id}"
    if engine_key not in st.session_state.logic_engines:
        st.session_state.logic_engines[engine_key] = LogicalRCA(topology)
    engine = st.session_state.logic_engines[engine_key]
    
    analysis_results = engine.analyze(alarms) if alarms else [{
        "id": "SYSTEM", "label": "正常稼働", "prob": 0.0, "type": "Normal", "tier": 3, "reason": "アラームなし"
    }]
    
    # KPIメトリクス
    root_cause_alarms = [a for a in alarms if a.is_root_cause]
    total_alarms = len(alarms)
    noise_reduction = ((total_alarms - len(root_cause_alarms)) / total_alarms * 100) if total_alarms > 0 else 0.0
    action_required = len(set(a.device_id for a in root_cause_alarms))
    prediction_count = len([r for r in analysis_results if r.get('is_prediction')])
    
    st.markdown("---")
    cols = st.columns(3)
    cols[0].metric("🚨 ステータス", f"{get_status_icon(status)} {status}")
    cols[1].metric("📊 アラーム数", f"{len(alarms)}件")
    suspect_count = len([r for r in analysis_results if r.get('prob', 0) > 0.5])
    cols[2].metric("🎯 被疑箇所", f"{suspect_count}件", delta=f"うち🔮予兆 {prediction_count}件" if prediction_count else None, delta_color="off")
    
    kpi_cols = st.columns(3)
    with kpi_cols[0]:
        st.metric("📉 ノイズ削減率", f"{noise_reduction:.1f}%", delta="↑ 高効率" if noise_reduction > 90 else "通常")
    with kpi_cols[1]:
        st.metric("🔮 予兆検知", f"{prediction_count}件", delta="⚡ 要注意" if prediction_count > 0 else "問題なし", delta_color="inverse")
    with kpi_cols[2]:
        st.metric("🚨 要対応インシデント", f"{action_required}件", delta="↑ 対処必要" if action_required > 0 else "問題なし", delta_color="inverse")
    
    st.markdown("---")
    
    # Future Radar
    preds = [c for c in analysis_results if c.get('is_prediction')]
    if preds:
        st.markdown("### 🔮 AIOps Future Radar")
        with st.container(border=True):
            injected_info = st.session_state.get("injected_weak_signal")
            scenario_lbl = f"（劣化シナリオ: {injected_info.get('scenario')}）" if injected_info else ""
            st.info(f"⚠️ **予兆検知**: 将来の障害リスクを検出しました。{scenario_lbl}")
            
            radar_cols = st.columns(min(len(preds), 3))
            for idx, item in enumerate(preds[:3]):
                with radar_cols[idx]:
                    prob_pct = f"{item.get('prob',0)*100:.0f}%"
                    st.error(f"**📍 {item['id']}**")
                    st.markdown(f"<div style='text-align:center;'><span style='font-size:36px;font-weight:bold;color:#d32f2f;'>{prob_pct}</span><br>発生確率</div>", unsafe_allow_html=True)
                    st.divider()
                    st.markdown(f"**予測障害:** {item.get('label','').replace('🔮 [予兆] ', '')}")
                    st.markdown(f"**急性期:** {item.get('prediction_timeline','不明')}")
                    with st.expander("🔍 検知詳細"):
                        st.text(item.get('reason', ''))
        st.markdown("---")

    # インシデント候補 & 影響範囲リスト
    root_cause_ids = {a.device_id for a in alarms if a.is_root_cause}
    downstream_ids = {a.device_id for a in alarms if not a.is_root_cause}
    
    rc_candidates = []
    ds_devices = []
    
    for c in analysis_results:
        did = c.get('id')
        if c.get('is_prediction') or did in root_cause_ids or c.get('prob', 0) > 0.5:
            rc_candidates.append(c)
        elif did in downstream_ids:
            ds_devices.append(c)
            
    if not rc_candidates and not alarms:
        rc_candidates = [{"id": "SYSTEM", "label": "正常稼働", "prob": 0.0, "type": "Normal"}]

    selected_cand = None
    target_dev_id = None

    if rc_candidates:
        df_data = []
        for i, c in enumerate(rc_candidates, 1):
            prob = c.get('prob', 0)
            act = "⚡ 予兆対応" if c.get('is_prediction') else "🚀 自動修復" if prob > 0.8 else "🔍 調査"
            status_txt = "🔮 予兆" if c.get('is_prediction') else "🔴 危険" if prob > 0.9 else "🟡 警告"
            
            df_data.append({
                "順位": i, "ステータス": status_txt, "デバイス": c['id'], 
                "原因": c.get('label'), "確信度": f"{prob*100:.0f}%", "推奨アクション": act,
                "_id": c['id']
            })
        
        st.markdown("#### 🎯 根本原因候補")
        df = pd.DataFrame(df_data)
        event = st.dataframe(df.drop(columns=["_id"]), use_container_width=True, hide_index=True, selection_mode="single-row", on_select="rerun")
        
        if event.selection and event.selection.rows:
            sel_row = df.iloc[event.selection.rows[0]]
            for c in rc_candidates:
                if c['id'] == sel_row['_id']:
                    selected_cand = c
                    target_dev_id = c['id']
                    break
        elif rc_candidates:
            selected_cand = rc_candidates[0]
            target_dev_id = rc_candidates[0]['id']

        # ★ ここが復活箇所: 影響を受けている機器（上流復旧待ち）リスト
        if ds_devices:
            with st.expander(f"▼ 影響を受けている機器 ({len(ds_devices)}台) - 上流復旧待ち", expanded=False):
                dd_df = pd.DataFrame([
                    {"No": i+1, "デバイス": d['id'], "状態": "⚫ 応答なし", "備考": "上流復旧待ち"}
                    for i, d in enumerate(ds_devices)
                ])
                if len(ds_devices) >= 10:
                    with st.container(height=300):
                        st.dataframe(dd_df, use_container_width=True, hide_index=True)
                else:
                    st.dataframe(dd_df, use_container_width=True, hide_index=True)

    # 2カラムレイアウト
    col_map, col_chat = st.columns([1.2, 1])
    
    with col_map:
        st.subheader("🌐 Network Topology")
        st.graphviz_chart(render_topology_graph(topology, alarms, analysis_results), use_container_width=True)
        
        st.markdown("---")
        st.subheader("🛠️ Auto-Diagnostics")
        if st.button("🚀 診断実行 (Run Diagnostics)", type="primary"):
            if not api_key:
                st.error("API Key Required")
            else:
                with st.status("Agent Operating...", expanded=True) as status:
                    target_node = topology.get(target_dev_id)
                    res = run_diagnostic_simulation_no_llm(scenario, target_node)
                    st.session_state.live_result = res
                    if res["status"] == "SUCCESS":
                        status.update(label="Diagnostics Complete!", state="complete", expanded=False)
                        st.session_state.verification_result = verify_log_content(res.get('sanitized_log', ""))
                    else:
                        st.write("❌ Failed.")
                        status.update(label="Failed", state="error")
                st.rerun()
        
        if st.session_state.live_result:
            res = st.session_state.live_result
            st.markdown("#### 📄 Diagnostic Results")
            with st.container(border=True):
                if st.session_state.verification_result:
                    v = st.session_state.verification_result
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Ping", v.get('ping_status'))
                    c2.metric("Interface", v.get('interface_status'))
                    c3.metric("HW", v.get('hardware_status'))
                st.divider()
                st.code(res.get("sanitized_log"), language="text")

    with col_chat:
        st.subheader("📝 AI Analyst Report")
        if selected_cand:
            if st.session_state.generated_report is None:
                st.info(f"Target: **{selected_cand['id']}**")
                if api_key and (scenario != "正常稼働" or selected_cand.get('is_prediction')):
                    btn_label = "🔮 予兆分析レポート" if selected_cand.get('is_prediction') else "📝 詳細レポート作成"
                    if st.button(btn_label):
                        cont = st.empty()
                        t_node = topology.get(selected_cand['id'])
                        topology_context = {"id": selected_cand['id']}
                        target_conf = load_config_by_id(selected_cand['id'])
                        
                        cache_key = _hash_text(f"{site_id}|{scenario}|{selected_cand['id']}")
                        if cache_key in st.session_state.report_cache:
                            full_text = st.session_state.report_cache[cache_key]
                            cont.markdown(full_text)
                        else:
                            cont.write("🤖 AI 分析中...")
                            full_text = ""
                            try:
                                for chunk in generate_analyst_report_streaming(
                                    scenario, t_node, topology_context, target_conf, "", api_key
                                ):
                                    full_text += chunk
                                    cont.markdown(full_text)
                                st.session_state.report_cache[cache_key] = full_text
                            except Exception as e:
                                cont.error(f"Error: {e}")
                        st.session_state.generated_report = full_text
            else:
                with st.container(height=400, border=True):
                    st.markdown(st.session_state.generated_report)
                if st.button("🔄 再作成"):
                    st.session_state.generated_report = None
                    st.rerun()
        
        st.markdown("---")
        st.subheader("🤖 Remediation & Chat")
        
        if selected_cand and selected_cand['prob'] > 0.6:
            if st.session_state.remediation_plan is None:
                if st.button("✨ 修復プラン作成"):
                    if not st.session_state.generated_report:
                        st.warning("先にレポートを作成してください")
                    else:
                        cont = st.empty()
                        t_node = topology.get(selected_cand['id'])
                        rem_text = ""
                        for chunk in generate_remediation_commands_streaming(
                            scenario, st.session_state.generated_report, t_node, api_key
                        ):
                            rem_text += chunk
                            cont.markdown(rem_text)
                        st.session_state.remediation_plan = rem_text
                        st.rerun()
            
            if st.session_state.remediation_plan:
                with st.container(height=300, border=True):
                    st.info("AI Remediation Plan")
                    st.markdown(st.session_state.remediation_plan)
                
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("🚀 修復実行", type="primary"):
                        with st.status("Executing Fix...", expanded=True):
                            t_node = topology.get(selected_cand['id'])
                            dev_info = t_node.metadata if t_node and hasattr(t_node, 'metadata') else {}
                            res = run_remediation_parallel_v2(selected_cand['id'], dev_info, scenario)
                            st.write("✅ Done.")
                            st.session_state.recovered_devices[selected_cand['id']] = True
                            st.balloons()
                with c2:
                    if st.button("キャンセル"):
                        st.session_state.remediation_plan = None
                        st.rerun()

        with st.expander("💬 Chat with AI Agent", expanded=False):
            if not st.session_state.chat_session and api_key and GENAI_AVAILABLE:
                genai.configure(api_key=api_key)
                st.session_state.chat_session = genai.GenerativeModel("gemma-3-12b-it").start_chat(history=[])
            
            tab_chat, tab_hist = st.tabs(["💬 会話", "📝 履歴"])
            with tab_chat:
                if st.session_state.messages:
                    last = st.session_state.messages[-1]
                    if last["role"] == "assistant":
                        st.info("🤖 " + last["content"])
                
                prompt = st.text_area("質問:", height=70, key="chat_in")
                if st.button("送信", type="primary") and prompt:
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    if st.session_state.chat_session:
                        ci = _build_ci_context_for_chat(topology, target_dev_id)
                        full_p = f"Context: {json.dumps(ci)}\nQuestion: {prompt}"
                        with st.spinner("AI thinking..."):
                            resp = generate_content_with_retry(st.session_state.chat_session.model, full_p, stream=False)
                            if resp:
                                st.session_state.messages.append({"role": "assistant", "content": resp.text})
                    st.rerun()
