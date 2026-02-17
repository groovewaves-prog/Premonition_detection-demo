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
# ヘルパー関数
# =====================================================
def _hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _pick_first(mapping: dict, keys: list, default: str = "") -> str:
    for k in keys:
        try:
            v = mapping.get(k)
            if v:
                return str(v)
        except:
            pass
    return default


def _build_ci_context_for_chat(topology: dict, target_node_id: str) -> dict:
    node = topology.get(target_node_id)
    if node and hasattr(node, 'metadata'):
        md = node.metadata or {}
    elif isinstance(node, dict):
        md = node.get('metadata', {})
    else:
        md = {}
    ci = {
        "device_id": target_node_id or "",
        "hostname": _pick_first(md, ["hostname", "host", "name"], default=(target_node_id or "")),
        "vendor": _pick_first(md, ["vendor", "manufacturer", "maker", "brand"], default=""),
        "os": _pick_first(md, ["os", "platform", "os_name"], default=""),
        "model": _pick_first(md, ["model", "hw_model", "product"], default=""),
        "role": _pick_first(md, ["role", "type", "device_role"], default=""),
    }
    try:
        conf = load_config_by_id(target_node_id) if target_node_id else ""
        if conf:
            ci["config_excerpt"] = conf[:1500]
    except Exception:
        pass
    return ci


def run_diagnostic_simulation_no_llm(scenario: str, target_node_obj) -> dict:
    """LLMを呼ばない疑似診断"""
    device_id = getattr(target_node_obj, "id", "UNKNOWN") if target_node_obj else "UNKNOWN"
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"[PROBE] ts={ts}",
        f"[PROBE] scenario={scenario}",
        f"[PROBE] target_device={device_id}",
        "",
    ]

    recovered_devices = st.session_state.get("recovered_devices") or {}
    recovered_map = st.session_state.get("recovered_scenario_map") or {}

    if recovered_devices.get(device_id) and recovered_map.get(device_id) == scenario:
        if "FW" in scenario:
            lines += ["show chassis cluster status", "Redundancy group 0: healthy", "control link: up", "fabric link: up"]
        elif "WAN" in scenario:
            lines += ["show ip interface brief", "GigabitEthernet0/0 up up", "Neighbor 203.0.113.2 Established",
                      "ping 203.0.113.2 repeat 5", "Success rate is 100 percent (5/5)"]
        elif "L2SW" in scenario:
            lines += ["show environment", "Fan: OK", "Temperature: OK", "show interface status", "Uplink: up"]
        else:
            lines += ["show system alarms", "No active alarms", "ping 8.8.8.8 repeat 5", "Success rate is 100 percent (5/5)"]
        return {"status": "SUCCESS", "sanitized_log": "\n".join(lines), "device_id": device_id}

    if "WAN全回線断" in scenario or "[WAN]" in scenario:
        lines += ["show ip interface brief", "GigabitEthernet0/0 down down", "show ip bgp summary",
                  "Neighbor 203.0.113.2 Idle", "ping 203.0.113.2 repeat 5", "Success rate is 0 percent (0/5)"]
    elif "FW片系障害" in scenario or "[FW]" in scenario:
        lines += ["show chassis cluster status", "Redundancy group 0: degraded", "control link: down", "fabric link: up"]
    elif "L2SW" in scenario:
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

    # ヘッダー
    col_header = st.columns([4, 1])
    with col_header[0]:
        st.markdown(f"### 🛡️ AIOps インシデント・コックピット")
    with col_header[1]:
        if st.button("🔙 一覧に戻る", key="back_to_list"):
            st.session_state.active_site = None
            st.rerun()

    # トポロジー読み込み
    paths = get_paths(site_id)
    topology = load_topology(paths.topology_path)

    if not topology:
        st.error("トポロジーが読み込めませんでした。")
        return

    # アラーム生成
    alarms = generate_alarms_for_scenario(topology, scenario)
    status = get_status_from_alarms(scenario, alarms)

    # 予兆シグナル注入
    injected = st.session_state.get("injected_weak_signal")
    if injected and injected["device_id"] in topology:
        messages = injected.get("messages", [injected.get("message", "")])
        for msg in messages:
            if msg:
                alarms.append(Alarm(
                    device_id=injected["device_id"],
                    message=msg,
                    severity="INFO",
                    is_root_cause=False
                ))

    # LogicalRCA エンジン
    engine_key = f"engine_{site_id}"
    if engine_key not in st.session_state.logic_engines:
        st.session_state.logic_engines[engine_key] = LogicalRCA(topology)
    engine = st.session_state.logic_engines[engine_key]

    if alarms:
        analysis_results = engine.analyze(alarms)
    else:
        analysis_results = [{
            "id": "SYSTEM",
            "label": "正常稼働",
            "prob": 0.0,
            "type": "Normal",
            "tier": 3,
            "reason": "アラームなし"
        }]

    # =====================================================
    # KPIメトリクス
    # =====================================================
    root_cause_alarms = [a for a in alarms if a.is_root_cause]
    total_alarms = len(alarms)
    noise_reduction = ((total_alarms - len(root_cause_alarms)) / total_alarms * 100) if total_alarms > 0 else 0.0
    action_required = len(set(a.device_id for a in root_cause_alarms))
    prediction_results = [r for r in analysis_results if r.get('is_prediction')]
    prediction_count = len(prediction_results)

    st.markdown("---")
    cols = st.columns(3)
    cols[0].metric("🚨 ステータス", f"{get_status_icon(status)} {status}")
    cols[1].metric("📊 アラーム数", f"{total_alarms}件")
    suspect_count = len([r for r in analysis_results if r.get('prob', 0) > 0.5])
    if prediction_count > 0:
        cols[2].metric("🎯 被疑箇所", f"{suspect_count}件",
                       delta=f"うち🔮予兆 {prediction_count}件", delta_color="off")
    else:
        cols[2].metric("🎯 被疑箇所", f"{suspect_count}件")

    kpi_cols = st.columns(3)
    with kpi_cols[0]:
        delta_text = "↑ 高効率稼働中" if noise_reduction > 90 else ("→ 通常稼働" if noise_reduction > 50 else "↓ 要確認")
        delta_color = "normal" if noise_reduction > 90 else ("off" if noise_reduction > 50 else "inverse")
        kpi_cols[0].metric("📉 ノイズ削減率", f"{noise_reduction:.1f}%", delta=delta_text, delta_color=delta_color)
    with kpi_cols[1]:
        kpi_cols[1].metric("🔮 予兆検知", f"{prediction_count}件",
                           delta="⚡ 要注意" if prediction_count > 0 else "問題なし",
                           delta_color="inverse" if prediction_count > 0 else "normal")
    with kpi_cols[2]:
        kpi_cols[2].metric("🚨 要対応インシデント", f"{action_required}件",
                           delta="↑ 対処が必要" if action_required > 0 else "問題なし",
                           delta_color="inverse" if action_required > 0 else "normal")

    st.markdown("---")

    # =====================================================
    # 根本原因候補とダウンストリームの分離
    # =====================================================
    root_cause_device_ids = set(a.device_id for a in alarms if a.is_root_cause)
    downstream_device_ids = set(a.device_id for a in alarms if not a.is_root_cause)

    root_cause_candidates = []
    downstream_devices = []

    for cand in analysis_results:
        device_id = cand.get('id', '')
        if cand.get('is_prediction'):
            root_cause_candidates.append(cand)
        elif device_id in root_cause_device_ids:
            root_cause_candidates.append(cand)
        elif device_id in downstream_device_ids:
            downstream_devices.append(cand)
        elif cand.get('prob', 0) > 0.5:
            root_cause_candidates.append(cand)

    if not root_cause_candidates and not alarms:
        root_cause_candidates = [{
            "id": "SYSTEM", "label": "正常稼働", "prob": 0.0,
            "type": "Normal", "tier": 3, "reason": "アラームなし"
        }]

    if root_cause_candidates and downstream_devices:
        st.info(f"📍 **根本原因**: {root_cause_candidates[0]['id']} → 影響範囲: 配下 {len(downstream_devices)} 機器")

    # =====================================================
    # 🔮 AIOps Future Radar（予兆専用表示エリア）
    # =====================================================
    prediction_candidates = [c for c in root_cause_candidates if c.get('is_prediction')]

    if prediction_candidates:
        st.markdown("### 🔮 AIOps Future Radar")
        with st.container(border=True):
            injected_info = st.session_state.get("injected_weak_signal")
            if injected_info:
                level = injected_info.get("level", 0)
                scenario_name = injected_info.get("scenario", "")
                st.info(
                    f"⚠️ **予兆検知**: 現在のネットワーク状態は「正常」ですが、"
                    f"AIが微細なシグナルから将来の障害リスクを検出しました。"
                    f"（劣化シナリオ: {scenario_name} / レベル: {level}/5）"
                )
            else:
                st.info("⚠️ **予兆検知**: AIが将来の障害リスクを検出しました。")

            radar_cols = st.columns(min(len(prediction_candidates), 3))
            for idx, pred_item in enumerate(prediction_candidates[:3]):
                with radar_cols[idx]:
                    prob_pct = f"{pred_item.get('prob', 0)*100:.0f}%"
                    pred_timeline = pred_item.get('prediction_timeline', '不明')
                    pred_affected = pred_item.get('prediction_affected_count', 0)
                    pred_label = pred_item.get('label', '').replace('🔮 [予兆] ', '')
                    pred_early_hours = pred_item.get('prediction_early_warning_hours', 0)

                    st.error(f"**📍 {pred_item['id']}**")
                    st.markdown(
                        f"<div style='text-align:center;'>"
                        f"<span style='font-size:36px; font-weight:bold; color:#d32f2f;'>{prob_pct}</span>"
                        f"<br><span style='color:#666;'>発生確率（急性期: {pred_timeline}）</span>"
                        f"</div>", unsafe_allow_html=True
                    )
                    st.divider()
                    st.markdown(f"**予測障害:** {pred_label}")
                    if pred_early_hours >= 24:
                        early_display = f"最大 **{pred_early_hours // 24}日前** から検知可能"
                    elif pred_early_hours > 0:
                        early_display = f"最大 **{pred_early_hours}時間前** から検知可能"
                    else:
                        early_display = "不明"
                    st.markdown(f"**早期予兆:** {early_display}")
                    st.markdown(f"**急性期:** 発症後 **{pred_timeline}** に深刻化")
                    st.markdown(f"**影響範囲:** 配下 **{pred_affected}台** が通信断の恐れ")

                    with st.expander("🔍 検知された予兆 (Weak Signal)"):
                        st.text(pred_item.get('reason', ''))
                        factors = pred_item.get('prediction_confidence_factors', {})
                        if factors:
                            st.caption(
                                f"ベース信頼度: {factors.get('base', 0):.2f} / "
                                f"マッチ品質: {factors.get('match_quality', 0):.2f} / "
                                f"SPOF: {'Yes' if factors.get('is_spof') else 'No'} / "
                                f"冗長性: {'Yes' if factors.get('has_redundancy') else 'No'}"
                            )
        st.markdown("---")

    # =====================================================
    # 🎯 根本原因候補テーブル
    # ★★★ 修正①: alarm_info_mapを使ったseverity基準の判定に戻す ★★★
    # =====================================================
    selected_incident_candidate = None
    target_device_id = None

    if root_cause_candidates:
        # アラームのseverityとsilentフラグをデバイスIDでマッピング
        alarm_info_map = {}
        for a in alarms:
            if a.device_id not in alarm_info_map:
                alarm_info_map[a.device_id] = {'severity': 'INFO', 'is_silent': False}
            if a.severity == 'CRITICAL':
                alarm_info_map[a.device_id]['severity'] = 'CRITICAL'
            elif a.severity == 'WARNING' and alarm_info_map[a.device_id]['severity'] != 'CRITICAL':
                alarm_info_map[a.device_id]['severity'] = 'WARNING'
            if hasattr(a, 'is_silent_suspect') and a.is_silent_suspect:
                alarm_info_map[a.device_id]['is_silent'] = True

        df_data = []
        for rank, cand in enumerate(root_cause_candidates, 1):
            prob = cand.get('prob', 0)
            cand_type = cand.get('type', 'UNKNOWN')
            device_id = cand['id']
            alarm_info = alarm_info_map.get(device_id, {'severity': 'INFO', 'is_silent': False})

            # ★ 旧app.pyと同じ判定ロジック（severity基準）
            if cand.get('is_prediction'):
                status_text = "🔮 予兆検知"
                timeline = cand.get('prediction_timeline', '')
                affected = cand.get('prediction_affected_count', 0)
                early_hours = cand.get('prediction_early_warning_hours', 0)
                early_str = (f"(予兆: {early_hours // 24}日前〜)" if early_hours >= 24
                             else (f"(予兆: {early_hours}時間前〜)" if early_hours > 0 else ""))
                if timeline and affected:
                    action = f"⚡ 急性期{timeline}以内 {early_str} ({affected}台影響)"
                else:
                    action = f"⚡ 予防的対処を推奨 {early_str}"
            elif alarm_info['is_silent'] or "Silent" in cand_type:
                status_text = "🟣 サイレント疑い"
                action = "🔍 上位確認"
            elif alarm_info['severity'] == 'CRITICAL':
                # ★ ここが修正ポイント: prob閾値ではなくCRITICAL severity で判定
                status_text = "🔴 危険 (根本原因)"
                action = "🚀 自動修復が可能"
            elif alarm_info['severity'] == 'WARNING':
                status_text = "🟡 警告"
                action = "🔍 詳細調査"
            elif prob > 0.6:
                status_text = "🟡 被疑箇所"
                action = "🔍 詳細調査"
            else:
                status_text = "⚪ 監視中"
                action = "👁️ 静観"

            df_data.append({
                "順位": rank,
                "ステータス": status_text,
                "デバイス": device_id,
                "原因": cand.get('label', ''),
                "確信度": f"{prob*100:.0f}%",
                "推奨アクション": action,
                "_id": device_id,
                "_prob": prob
            })

        df = pd.DataFrame(df_data)

        st.markdown("#### 🎯 根本原因候補")
        event = st.dataframe(
            df[["順位", "ステータス", "デバイス", "原因", "確信度", "推奨アクション"]],
            use_container_width=True,
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun"
        )

        if event.selection and len(event.selection.rows) > 0:
            sel_row = df.iloc[event.selection.rows[0]]
            for cand in root_cause_candidates:
                if cand['id'] == sel_row['_id']:
                    selected_incident_candidate = cand
                    target_device_id = cand['id']
                    break
        elif root_cause_candidates:
            selected_incident_candidate = root_cause_candidates[0]
            target_device_id = root_cause_candidates[0]['id']

        # 影響デバイス（下流）一覧
        if downstream_devices:
            with st.expander(f"▼ 影響を受けている機器 ({len(downstream_devices)}台) - 上流復旧待ち", expanded=False):
                dd_df = pd.DataFrame([
                    {"No": i+1, "デバイス": d['id'], "状態": "⚫ 応答なし", "備考": "上流復旧待ち"}
                    for i, d in enumerate(downstream_devices)
                ])
                if len(downstream_devices) >= 10:
                    with st.container(height=300):
                        st.dataframe(dd_df, use_container_width=True, hide_index=True)
                else:
                    st.dataframe(dd_df, use_container_width=True, hide_index=True)

    # =====================================================
    # 2カラムレイアウト
    # =====================================================
    col_map, col_chat = st.columns([1.2, 1])

    # === 左カラム: トポロジー & Auto-Diagnostics ===
    with col_map:
        st.subheader("🌐 Network Topology")
        graph = render_topology_graph(topology, alarms, analysis_results)
        st.graphviz_chart(graph, use_container_width=True)

        st.markdown("---")
        st.subheader("🛠️ Auto-Diagnostics")

        if st.button("🚀 診断実行 (Run Diagnostics)", type="primary"):
            if not api_key:
                st.error("API Key Required")
            else:
                with st.status("Agent Operating...", expanded=True) as status_widget:
                    st.write("🔌 Connecting to device...")
                    target_node_obj = topology.get(target_device_id) if target_device_id else None
                    res = run_diagnostic_simulation_no_llm(scenario, target_node_obj)
                    st.session_state.live_result = res
                    if res["status"] == "SUCCESS":
                        st.write("✅ Log Acquired & Sanitized.")
                        status_widget.update(label="Diagnostics Complete!", state="complete", expanded=False)
                        log_content = res.get('sanitized_log', "")
                        st.session_state.verification_result = verify_log_content(log_content)
                        st.session_state.trigger_analysis = True
                    else:
                        st.write("❌ Connection Failed.")
                        status_widget.update(label="Diagnostics Failed", state="error")
                st.rerun()

        if st.session_state.live_result:
            res = st.session_state.live_result
            if res["status"] == "SUCCESS":
                st.markdown("#### 📄 Diagnostic Results")
                with st.container(border=True):
                    if st.session_state.verification_result:
                        v = st.session_state.verification_result
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Ping Status", v.get('ping_status'))
                        c2.metric("Interface", v.get('interface_status'))
                        c3.metric("Hardware", v.get('hardware_status'))
                    st.divider()
                    st.caption("🔒 Raw Logs (Sanitized)")
                    st.code(res["sanitized_log"], language="text")

    # =====================================================
    # === 右カラム: AI Analyst Report & Remediation & Chat ===
    # =====================================================
    with col_chat:
        # ============================================
        # A. AI Analyst Report
        # ============================================
        st.subheader("📝 AI Analyst Report")

        if selected_incident_candidate:
            cand = selected_incident_candidate

            if st.session_state.generated_report is None:
                st.info(f"インシデント選択中: **{cand['id']}** ({cand.get('label', '')})")

                if api_key and (scenario != "正常稼働" or cand.get('is_prediction')):
                    is_pred = cand.get('is_prediction')
                    btn_label = ("🔮 予兆の確認手順を生成 (Predictive Analysis)"
                                 if is_pred else "📝 詳細レポートを作成 (Generate Report)")

                    if st.button(btn_label):
                        report_container = st.empty()
                        target_conf = load_config_by_id(cand['id'])
                        verification_context = cand.get("verification_log", "特になし")

                        t_node = topology.get(cand["id"])
                        t_node_dict = {
                            "id": getattr(t_node, "id", None) if t_node else None,
                            "type": getattr(t_node, "type", None) if t_node else None,
                            "layer": getattr(t_node, "layer", None) if t_node else None,
                            "metadata": (getattr(t_node, "metadata", {}) or {}) if t_node else {},
                        }
                        parent_id = t_node.parent_id if t_node and hasattr(t_node, 'parent_id') else None
                        children_ids = [
                            nid for nid, n in topology.items()
                            if (getattr(n, "parent_id", None) if hasattr(n, 'parent_id')
                                else n.get('parent_id')) == cand["id"]
                        ]
                        topology_context = {
                            "node": t_node_dict,
                            "parent_id": parent_id,
                            "children_ids": children_ids
                        }

                        cache_key_analyst = "|".join([
                            "analyst", site_id, scenario,
                            str(cand.get("id")),
                            _hash_text(verification_context),
                        ])

                        if cache_key_analyst in st.session_state.report_cache:
                            full_text = st.session_state.report_cache[cache_key_analyst]
                            report_container.markdown(full_text)
                        else:
                            try:
                                report_container.write("🤖 AIレポート生成中...")
                                placeholder = report_container.empty()
                                full_text = ""

                                for chunk in generate_analyst_report_streaming(
                                    scenario=scenario,
                                    target_node=t_node,
                                    analysis_result={"node": t_node_dict, "topology": topology_context},
                                    target_conf=target_conf,
                                    verification_context=verification_context,
                                    api_key=api_key,
                                    max_retries=2,
                                    backoff=3
                                ):
                                    full_text += chunk
                                    placeholder.markdown(full_text)

                                if not full_text or full_text.startswith("Error"):
                                    full_text = f"⚠️ 分析レポート生成に失敗しました: {full_text}"
                                    placeholder.markdown(full_text)

                                st.session_state.report_cache[cache_key_analyst] = full_text
                            except Exception as e:
                                full_text = f"⚠️ 分析レポート生成に失敗しました: {type(e).__name__}: {e}"
                                report_container.markdown(full_text)

                        st.session_state.generated_report = full_text
            else:
                # レポートを表示（height=400のスクロールコンテナ）
                with st.container(height=400, border=True):
                    st.markdown(st.session_state.generated_report)
                if st.button("🔄 レポート再作成"):
                    st.session_state.generated_report = None
                    st.rerun()

        # ============================================
        # ★★★ B. 復旧手順 + 修復実行(Execute)ボタン
        #         「AI Analyst Report」と「Remediation & Chat」の間に配置
        # ============================================
        if selected_incident_candidate and selected_incident_candidate["prob"] > 0.6:
            is_pred_rem = selected_incident_candidate.get('is_prediction')

            if st.session_state.remediation_plan is None:
                # Generate Fix ボタン（レポートセクション直下、Remediation & Chat の前）
                fix_label = ("🔮 予防措置プランを生成" if is_pred_rem
                             else "✨ 修復プランを作成 (Generate Fix)")
                report_prereq = ("「🔮 予兆の確認手順を生成」" if is_pred_rem
                                 else "「📝 詳細レポートを作成 (Generate Report)」")

                if st.button(fix_label):
                    if st.session_state.generated_report is None:
                        st.warning(f"先に{report_prereq}を実行してください。")
                    else:
                        remediation_container = st.empty()
                        t_node = topology.get(selected_incident_candidate["id"])

                        rem_scenario = scenario
                        if is_pred_rem:
                            pred_timeline = selected_incident_candidate.get('prediction_timeline', '不明')
                            pred_affected = selected_incident_candidate.get('prediction_affected_count', 0)
                            pred_early_hours = selected_incident_candidate.get('prediction_early_warning_hours', 0)
                            pred_time_critical = selected_incident_candidate.get('prediction_time_to_critical_min', 0)
                            early_ctx = (f"最大{pred_early_hours // 24}日前から検知可能" if pred_early_hours >= 24
                                         else (f"最大{pred_early_hours}時間前から検知可能" if pred_early_hours > 0
                                               else "早期検知パターン"))
                            rem_scenario = (
                                f"[予兆対応 - Predictive Maintenance] {selected_incident_candidate['id']}で障害の前兆を検出。\n"
                                f"・早期予兆: {early_ctx}\n"
                                f"・急性期: 発症後{pred_time_critical}分に深刻化の恐れ（影響{pred_affected}台）\n\n"
                                f"「復旧」ではなく「予防措置」として手順を提示してください。"
                            )

                        cache_key_remediation = "|".join([
                            "remediation", site_id, scenario,
                            str(selected_incident_candidate.get("id")),
                            _hash_text(st.session_state.generated_report or ""),
                        ])

                        if cache_key_remediation in st.session_state.report_cache:
                            remediation_text = st.session_state.report_cache[cache_key_remediation]
                            remediation_container.markdown(remediation_text)
                        else:
                            try:
                                loading_msg = ("🔮 予防措置プラン生成中..." if is_pred_rem
                                               else "🤖 復旧プラン生成中...")
                                remediation_container.write(loading_msg)
                                placeholder = remediation_container.empty()
                                remediation_text = ""

                                for chunk in generate_remediation_commands_streaming(
                                    scenario=rem_scenario,
                                    analysis_result=st.session_state.generated_report or "",
                                    target_node=t_node,
                                    api_key=api_key,
                                    max_retries=2,
                                    backoff=3
                                ):
                                    remediation_text += chunk
                                    placeholder.markdown(remediation_text)

                                if not remediation_text or remediation_text.startswith("Error"):
                                    remediation_text = f"⚠️ 復旧プラン生成に失敗しました: {remediation_text}"
                                    placeholder.markdown(remediation_text)

                                st.session_state.report_cache[cache_key_remediation] = remediation_text
                            except Exception as e:
                                remediation_text = f"⚠️ 復旧プラン生成に失敗しました: {type(e).__name__}: {e}"
                                remediation_container.markdown(remediation_text)

                        st.session_state.remediation_plan = remediation_text
                        st.rerun()

            if st.session_state.remediation_plan is not None:
                # 復旧手順コンテナ
                with st.container(height=400, border=True):
                    st.info("AI Generated Recovery Procedure（復旧手順）")
                    st.markdown(st.session_state.remediation_plan)

                # ★ 修復実行(Execute) / キャンセル ボタン
                #   → AI Analyst Report と Remediation & Chat の「間」に表示される
                col_exec1, col_exec2 = st.columns(2)
                with col_exec1:
                    exec_clicked = st.button("🚀 修復実行 (Execute)", type="primary")
                with col_exec2:
                    cancel_clicked = st.button("キャンセル")

                if cancel_clicked:
                    st.session_state.remediation_plan = None
                    st.session_state.verification_log = None
                    st.rerun()

                if exec_clicked:
                    if not api_key:
                        st.error("API Key Required")
                    else:
                        with st.status("🔧 修復処理実行中...", expanded=True) as status_widget:
                            target_node_obj = topology.get(selected_incident_candidate["id"])
                            device_info = (target_node_obj.metadata
                                           if target_node_obj and hasattr(target_node_obj, 'metadata')
                                           else {})

                            st.write("🔄 Executing remediation steps in parallel...")

                            results_rem = run_remediation_parallel_v2(
                                device_id=selected_incident_candidate["id"],
                                device_info=device_info,
                                scenario=scenario,
                                environment=RemediationEnvironment.DEMO,
                                timeout_per_step=30
                            )

                            st.write("📋 Remediation steps result:")
                            all_success = True
                            remediation_summary = []

                            for step_name in ["Backup", "Apply", "Verify"]:
                                result = results_rem.get(step_name)
                                if result:
                                    st.write(str(result))
                                    remediation_summary.append(str(result))
                                    if result.status != "success":
                                        all_success = False

                            verification_log = "\n".join(remediation_summary)
                            st.session_state.verification_log = verification_log

                            if all_success:
                                st.write("✅ All remediation steps completed successfully.")
                                status_widget.update(label="Process Finished", state="complete", expanded=False)
                                st.session_state.recovered_devices[selected_incident_candidate["id"]] = True
                                st.session_state.recovered_scenario_map[selected_incident_candidate["id"]] = scenario
                                if not st.session_state.balloons_shown:
                                    st.balloons()
                                    st.session_state.balloons_shown = True
                                st.success("✅ System Recovered Successfully!")
                            else:
                                st.write("⚠️ Some remediation steps failed. Please review.")
                                status_widget.update(label="Process Finished - With Errors", state="error", expanded=True)

                if st.session_state.get("verification_log"):
                    st.markdown("#### 🔎 Post-Fix Verification Logs")
                    st.code(st.session_state.verification_log, language="text")

        # ============================================
        # C. Remediation & Chat（AIステータス表示 + Chat）
        # ============================================
        st.markdown("---")
        st.subheader("🤖 Remediation & Chat")

        if selected_incident_candidate and selected_incident_candidate["prob"] > 0.6:
            is_pred_rem = selected_incident_candidate.get('is_prediction')
            if is_pred_rem:
                timeline = selected_incident_candidate.get('prediction_timeline', '不明')
                affected = selected_incident_candidate.get('prediction_affected_count', 0)
                early_hours = selected_incident_candidate.get('prediction_early_warning_hours', 0)
                early_display = (f"最大 <b>{early_hours // 24}日前</b> から検知可能" if early_hours >= 24
                                 else (f"最大 <b>{early_hours}時間前</b> から検知可能" if early_hours > 0
                                       else "不明"))
                st.markdown(f"""
                <div style="background-color:#fff3e0;padding:10px;border-radius:5px;border:1px solid #ff9800;color:#e65100;margin-bottom:10px;">
                    <strong>🔮 Digital Twin 未来予測 (Predictive Maintenance)</strong><br>
                    <b>{selected_incident_candidate['id']}</b> で障害の兆候を検出しました。<br>
                    ・早期予兆: {early_display}<br>
                    ・急性期進行: 発症後 <b>{timeline}</b> に深刻化の恐れ<br>
                    ・影響範囲: <b>{affected}台</b> のデバイスに影響の可能性<br>
                    ・推奨: メンテナンスウィンドウでの予防交換/対応<br>
                    (信頼度: <span style="font-size:1.2em;font-weight:bold;">{selected_incident_candidate['prob']*100:.0f}%</span>)
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color:#e8f5e9;padding:10px;border-radius:5px;border:1px solid #4caf50;color:#2e7d32;margin-bottom:10px;">
                    <strong>✅ AI Analysis Completed</strong><br>
                    特定された原因 <b>{selected_incident_candidate['id']}</b> に対する復旧手順が利用可能です。<br>
                    (リスクスコア: <span style="font-size:1.2em;font-weight:bold;">{selected_incident_candidate['prob']*100:.0f}</span>)
                </div>
                """, unsafe_allow_html=True)

        else:
            if selected_incident_candidate:
                device_id = selected_incident_candidate.get('id', '')
                score = selected_incident_candidate['prob'] * 100
                if device_id == "SYSTEM" and score == 0:
                    st.markdown("""
                    <div style="background-color:#e8f5e9;padding:10px;border-radius:5px;border:1px solid #4caf50;color:#2e7d32;margin-bottom:10px;">
                        <strong>✅ 正常稼働中</strong><br>
                        現在、ネットワークは正常に稼働しています。対応が必要なインシデントはありません。
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color:#fff3e0;padding:10px;border-radius:5px;border:1px solid #ff9800;color:#e65100;margin-bottom:10px;">
                        <strong>⚠️ 監視中</strong><br>
                        対象: <b>{device_id}</b><br>
                        (リスクスコア: {score:.0f} - 60以上で自動修復を推奨)
                    </div>
                    """, unsafe_allow_html=True)

        # ★★★ 修正③: Chat with AI Agent (旧UIのExpander形式を完全復元) ★★★
        with st.expander("💬 Chat with AI Agent", expanded=False):
            _chat_target_id = ""
            if selected_incident_candidate:
                _chat_target_id = selected_incident_candidate.get("id", "") or ""
            if not _chat_target_id and target_device_id:
                _chat_target_id = target_device_id

            _chat_ci = _build_ci_context_for_chat(topology, _chat_target_id) if _chat_target_id else {}
            if _chat_ci:
                _vendor = _chat_ci.get("vendor", "") or "Unknown"
                _os = _chat_ci.get("os", "") or "Unknown"
                _model_name = _chat_ci.get("model", "") or "Unknown"
                st.caption(f"対象機器: {_chat_target_id}   Vendor: {_vendor}   OS: {_os}   Model: {_model_name}")

            # クイック質問ボタン
            q1, q2, q3 = st.columns(3)
            with q1:
                if st.button("設定バックアップ", use_container_width=True):
                    st.session_state.chat_quick_text = "この機器で、現在の設定を安全にバックアップする手順とコマンド例を教えてください。"
            with q2:
                if st.button("ロールバック", use_container_width=True):
                    st.session_state.chat_quick_text = "この機器で、変更をロールバックする代表的な手順（候補）と注意点を教えてください。"
            with q3:
                if st.button("確認コマンド", use_container_width=True):
                    st.session_state.chat_quick_text = "今回の症状を切り分けるために、まず実行すべき確認コマンド（show/diagnostic）を優先度順に教えてください。"

            if st.session_state.chat_quick_text:
                st.info("クイック質問（コピーして貼り付け）")
                st.code(st.session_state.chat_quick_text)

            if st.session_state.chat_session is None and api_key and GENAI_AVAILABLE:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemma-3-12b-it")
                st.session_state.chat_session = model.start_chat(history=[])

            # タブでレイアウト: 会話 / 履歴
            tab1, tab2 = st.tabs(["💬 会話", "📝 履歴"])

            with tab1:
                if st.session_state.messages:
                    last_msg = st.session_state.messages[-1]
                    if last_msg["role"] == "assistant":
                        st.info("🤖 最新の回答")
                        with st.container(height=300):
                            st.markdown(last_msg["content"])

                st.markdown("---")
                prompt = st.text_area(
                    "質問を入力してください:",
                    height=70,
                    placeholder="Ctrl+Enter または 送信ボタンで送信",
                    key="chat_textarea"
                )

                col1, col2, col3 = st.columns([3, 1, 1])
                with col2:
                    send_button = st.button("送信", type="primary", use_container_width=True)
                with col3:
                    if st.button("クリア"):
                        st.session_state.messages = []
                        st.rerun()

                if send_button and prompt:
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    if st.session_state.chat_session:
                        ci = _build_ci_context_for_chat(topology, _chat_target_id) if _chat_target_id else {}
                        ci_prompt = f"""あなたはネットワーク運用（NOC/SRE）の実務者です。
次の CI 情報と Config 抜粋を必ず参照して、具体的に回答してください。

【CI (JSON)】
{json.dumps(ci, ensure_ascii=False, indent=2)}

【ユーザーの質問】
{prompt}

回答ルール:
- CI/Config に基づく具体手順・コマンド例を提示する
- 追加確認が必要なら、質問は最小限（1〜2点）に絞る
- 不明な前提は推測せず「CIに無いので確認が必要」と明記する
"""
                        with st.spinner("AI が回答を生成中..."):
                            try:
                                response = generate_content_with_retry(
                                    st.session_state.chat_session.model, ci_prompt, stream=False
                                )
                                if response:
                                    full_response = response.text if hasattr(response, "text") else str(response)
                                    if not full_response.strip():
                                        full_response = "AI応答が空でした。"
                                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                                else:
                                    st.error("AIからの応答がありませんでした。")
                            except Exception as e:
                                st.error(f"エラーが発生しました: {e}")
                    st.rerun()

            with tab2:
                if st.session_state.messages:
                    history_container = st.container(height=400)
                    with history_container:
                        for i, msg in enumerate(st.session_state.messages):
                            icon = "🤖" if msg["role"] == "assistant" else "👤"
                            with st.container(border=True):
                                st.markdown(f"{icon} **{msg['role'].upper()}** (メッセージ {i+1})")
                                st.markdown(msg["content"])
                else:
                    st.info("会話履歴はまだありません。")
