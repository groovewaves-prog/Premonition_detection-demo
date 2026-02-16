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
# Helper Functions
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
    md = node.get('metadata', {}) if node and isinstance(node, dict) else (node.metadata if node else {})
    ci = {
        "device_id": target_node_id or "",
        "hostname": _pick_first(md, ["hostname", "host"], default=target_node_id or ""),
        "vendor": _pick_first(md, ["vendor"], default=""),
        "model": _pick_first(md, ["model"], default=""),
        "os": _pick_first(md, ["os"], default=""),
        "site": _pick_first(md, ["site", "location"], default="")
    }
    try:
        conf = load_config_by_id(target_node_id)
        if conf: ci["config_excerpt"] = conf[:1000]
    except: pass
    return ci

def run_diagnostic_simulation_no_llm(scenario: str, target_node) -> dict:
    dev_id = getattr(target_node, "id", "UNKNOWN") if target_node else "UNKNOWN"
    lines = [f"[PROBE] scenario={scenario}", f"target={dev_id}", ""]
    if "WAN" in scenario:
        lines += ["show ip int brief", "Gi0/0/0 UP", "BGP State: Established"]
    else:
        lines += ["show system alarms", "No active alarms"]
    return {"status": "SUCCESS", "sanitized_log": "\n".join(lines), "device_id": dev_id}

# =====================================================
# Main Render Function
# =====================================================
def render_incident_cockpit(site_id: str, api_key: Optional[str]):
    display_name = get_display_name(site_id)
    scenario = st.session_state.site_scenarios.get(site_id, "正常稼働")
    
    # --- Header ---
    c1, c2 = st.columns([4, 1])
    c1.markdown(f"### 🛡️ AIOps インシデント・コックピット")
    
    st.markdown("""
    <style>
    div[data-testid="stButton"] button { border-radius: 6px; }
    .st-emotion-cache-1r6slb0 { border: 1px solid #ddd; border-radius: 8px; padding: 15px; }
    </style>
    """, unsafe_allow_html=True)
    
    if c2.button("🔙 一覧に戻る", key="back_btn"):
        st.session_state.active_site = None
        st.rerun()

    # --- Load Data ---
    paths = get_paths(site_id)
    topology = load_topology(paths.topology_path)
    if not topology:
        st.error("トポロジー読み込みエラー")
        return

    alarms = generate_alarms_for_scenario(topology, scenario)
    status = get_status_from_alarms(scenario, alarms)
    
    # Injection
    injected = st.session_state.get("injected_weak_signal")
    if injected and injected["device_id"] in topology:
        for m in injected.get("messages", []):
            alarms.append(Alarm(injected["device_id"], m, "INFO", False))

    # Analysis
    engine_key = f"engine_{site_id}"
    if engine_key not in st.session_state.logic_engines:
        st.session_state.logic_engines[engine_key] = LogicalRCA(topology)
    engine = st.session_state.logic_engines[engine_key]
    
    # ★ 修正箇所: 変数名を results に統一
    results = engine.analyze(alarms) if alarms else []
    if not results: results = [{"id": "SYSTEM", "label": "正常稼働", "prob": 0.0, "type": "Normal"}]

    # --- KPI & Precognition ---
    preds = [r for r in results if r.get('is_prediction')]
    pred_count = len(preds)
    
    st.markdown("---")
    k1, k2, k3 = st.columns(3)
    k1.metric("🚨 ステータス", f"{get_status_icon(status)} {status}")
    k2.metric("📊 アラーム数", len(alarms))
    
    delta_color = "inverse" if pred_count > 0 else "off"
    delta_msg = "⚡ 将来のリスクを検知" if pred_count > 0 else "問題なし"
    k3.metric("🔮 予兆検知 (Precognition)", f"{pred_count}件", delta=delta_msg, delta_color=delta_color)
    
    st.markdown("---")

    # --- Future Radar (Precognition) ---
    if preds:
        st.markdown("### 🔮 AIOps Future Radar")
        st.caption("AIが予測する未来の障害イベント。クリックで詳細分析へジャンプします。")
        
        st.markdown("""
        <style>
        .future-card {
            border-left: 6px solid #9C27B0;
            background-color: #F3E5F5;
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

        for p in preds:
            # タイムライン計算（数値として取得）
            time_min = p.get('prediction_time_to_critical_min', 60)
            
            with st.container():
                st.markdown(f"""
                <div class="future-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="font-size:1.1em; font-weight:bold; color:#4A148C;">📍 {p['id']}</span>
                            <br><span style="color:#6A1B9A;">{p.get('label', '').replace('🔮 [予兆] ', '')}</span>
                        </div>
                        <div style="text-align:right;">
                            <span style="font-size:1.2em; font-weight:bold; color:#880E4F;">{p.get('prob', 0)*100:.0f}%</span>
                            <br><span style="font-size:0.8em; color:#666;">発生確率</span>
                        </div>
                    </div>
                    <div style="margin-top:8px; font-size:0.9em; color:#555;">
                        影響範囲: 配下 {p.get('prediction_affected_count', 0)} 台 / 早期検知: {p.get('prediction_early_warning_hours', 0)}時間前
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                c_time, c_act = st.columns([3, 1])
                with c_time:
                    # 時間の逼迫度に応じて色を変える
                    bar_val = max(0, min(100, 100 - time_min))
                    st.progress(bar_val, text=f"🔥 障害発生まで: あと約 {time_min} 分")
                with c_act:
                    if st.button("詳細対応", key=f"btn_future_{p['id']}", type="primary", use_container_width=True):
                        st.session_state.selected_candidate = p

        st.markdown("---")

    # --- Candidates & Selection ---
    rc_list = []
    ds_list = []
    root_cause_ids = {a.device_id for a in alarms if a.is_root_cause}
    
    for r in results:
        if r.get('is_prediction'): rc_list.append(r)
        elif r['id'] in root_cause_ids: rc_list.append(r)
        elif r.get('prob', 0) > 0.5: rc_list.append(r)
        elif r['id'] != 'SYSTEM': ds_list.append(r)
            
    if not rc_list and not alarms: rc_list = [{"id": "SYSTEM", "label": "正常稼働", "prob": 0.0}]

    sel_cand = st.session_state.get("selected_candidate")
    
    # === Main Layout ===
    col_l, col_r = st.columns([1.1, 1.2]) # 右側を少し広く（ワークスペース用）
    
    # Left: Visualization
    with col_l:
        st.subheader("🌐 Network Topology")
        st.graphviz_chart(render_topology_graph(topology, alarms, results), use_container_width=True)
        
        st.markdown("#### 🎯 インシデント候補")
        df = pd.DataFrame([{
            "Type": "🔮" if x.get('is_prediction') else "🔴" if x.get('prob')>=0.9 else "🟡",
            "Device": x['id'],
            "Cause": x.get('label'),
            "Conf": f"{x.get('prob',0)*100:.0f}%",
            "_obj": x
        } for x in rc_list])
        
        event = st.dataframe(
            df.drop(columns=["_obj"]), 
            use_container_width=True, 
            hide_index=True, 
            selection_mode="single-row", 
            on_select="rerun"
        )
        
        if event.selection.rows:
            sel_cand = df.iloc[event.selection.rows[0]]["_obj"]
            st.session_state.selected_candidate = sel_cand

        if ds_list:
            with st.expander(f"▼ 影響デバイス ({len(ds_list)}台)", expanded=False):
                st.dataframe(pd.DataFrame([{"Device": d['id'], "Status": "Unreachable"} for d in ds_list]), use_container_width=True, hide_index=True)

    # Right: Incident Workspace (One-Stop Operation)
    with col_r:
        if sel_cand:
            # 1. Header Card
            bg = "#F3E5F5" if sel_cand.get('is_prediction') else "#FFEBEE"
            bd = "#9C27B0" if sel_cand.get('is_prediction') else "#D32F2F"
            title_icon = "🔮" if sel_cand.get('is_prediction') else "🚨"
            
            st.markdown(f"""
            <div style="background-color: {bg}; border-left: 6px solid {bd}; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h3 style="margin:0; color:#333;">{title_icon} {sel_cand['id']}</h3>
                <p style="margin:5px 0 0 0; font-weight:bold; color:#555;">{sel_cand.get('label')}</p>
                <p style="font-size:0.9em; color:#666;">信頼度: {sel_cand.get('prob',0)*100:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 2. 推奨アクション (Primary Actions) - AIに聞く前に表示
            rec_actions = sel_cand.get("recommended_actions", [])
            if rec_actions and sel_cand.get('is_prediction'):
                st.markdown("#### ⚡ 推奨アクション (Primary Actions)")
                for act in rec_actions:
                    with st.container():
                        st.markdown(f"**👉 {act['title']}**")
                        st.caption(f"効果: {act['effect']}")
                st.divider()

            # 3. Workflow Tabs (Action Center)
            tab_act, tab_chat, tab_rpt = st.tabs(["🛠️ 対応アクション", "💬 AIチャット", "📝 レポート"])
            
            with tab_act:
                # Step 1: Analyze
                with st.expander("🔍 Step 1: 原因・影響の確認", expanded=True):
                    if sel_cand.get('reason'):
                        st.info(sel_cand.get('reason'))
                    
                    st.checkbox("ログを確認した", key=f"chk_log_{sel_cand['id']}")
                    st.checkbox("影響範囲を確認した", key=f"chk_imp_{sel_cand['id']}")

                # Step 2: Remediate
                with st.expander("🛠️ Step 2: 修復対応", expanded=True):
                    if st.session_state.remediation_plan is None:
                        if st.button("✨ 手順書を生成 (AI)", use_container_width=True):
                            with st.spinner("Generating Plan..."):
                                time.sleep(1) # Mock
                                st.session_state.remediation_plan = """
                                **AI推奨手順:**
                                1. `show interface status` で物理層を確認
                                2. ポートリセット (`shutdown` -> `no shutdown`)
                                3. 疎通確認 (`ping`)
                                """
                                st.rerun()
                    
                    if st.session_state.remediation_plan:
                        st.markdown(st.session_state.remediation_plan)
                        col_run, col_clr = st.columns([2, 1])
                        if col_run.button("🚀 Playbook実行", type="primary", use_container_width=True):
                            st.toast("Playbookを実行中...")
                            time.sleep(1)
                            st.session_state.recovered_devices[sel_cand['id']] = True
                            st.success("実行完了: 正常性を確認しました")
                            st.balloons()
                        if col_clr.button("クリア", use_container_width=True):
                            st.session_state.remediation_plan = None
                            st.rerun()

                # Step 3: Resolve (One-Click Feedback)
                st.markdown("#### ✅ Step 3: 結果登録 (Feedback)")
                c_ok, c_fp, c_mute = st.columns(3)
                if c_ok.button("解決 (Resolved)", type="primary", use_container_width=True):
                    st.toast(f"インシデント {sel_cand['id']} を解決済として記録しました。")
                if c_fp.button("誤検知 (FP)", use_container_width=True):
                    st.toast("誤検知として報告しました。学習データに反映されます。")
                if c_mute.button("静観 (Mute)", use_container_width=True):
                    st.toast("このアラートを24時間ミュートします。")

            with tab_chat:
                if not st.session_state.chat_session and api_key and GENAI_AVAILABLE:
                    genai.configure(api_key=api_key)
                    st.session_state.chat_session = genai.GenerativeModel("gemma-3-12b-it").start_chat(history=[])
                
                chat_cont = st.container(height=300)
                with chat_cont:
                    for msg in st.session_state.messages[-10:]:
                        icon = "🤖" if msg["role"] == "assistant" else "👤"
                        st.markdown(f"**{icon}** {msg['content']}")

                prompt = st.chat_input("AIエージェントに質問...")
                if prompt:
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    if st.session_state.chat_session:
                        ci = _build_ci_context_for_chat(topology, sel_cand['id'])
                        full_p = f"Context: {json.dumps(ci)}\nQuestion: {prompt}"
                        with st.spinner("Thinking..."):
                            resp = generate_content_with_retry(st.session_state.chat_session.model, full_p, stream=False)
                            if resp:
                                st.session_state.messages.append({"role": "assistant", "content": resp.text})
                    st.rerun()

            with tab_rpt:
                if st.button("📝 報告書作成 (PDFプレビュー)", use_container_width=True):
                    st.success("レポートを生成しました（モック）")
                    st.markdown("""
                    **障害報告書ドラフト**
                    * 発生時刻: 2026-02-17 10:00
                    * 対象: WAN_ROUTER_01
                    * 原因: マイクロバーストによるパケットドロップ
                    * 対応: QoSポリシー調整により解消
                    """)

        else:
            # 待機画面
            st.info("👈 左側のリストから、対応するインシデントまたは予兆を選択してください。")
            st.markdown("""
            **オペレーターガイド:**
            1. **🔮 予兆** は優先的に確認し、予防措置を検討してください。
            2. **🔴 障害** は影響範囲を確認し、直ちに自動修復を実行してください。
            3. **🟡 警告** は静観またはチケット起票を行ってください。
            """)
