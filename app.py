# -*- coding: utf-8 -*-
"""
AIOps Incident Cockpit - Multi-Site Edition
=============================================
複数拠点対応版 AIOps インシデント・コックピット

主な機能:
- 拠点状態ボード: 全拠点の状態を一覧表示
- 拠点別シナリオ: 各拠点で異なるシナリオを設定可能
- トリアージ・コマンドセンター: 優先度順の対応管理
- インシデント・コックピット: 詳細分析と復旧支援
"""

import streamlit as st
import graphviz
import os
import time
import json
import re
import hashlib
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from google.api_core import exceptions as google_exceptions

# Google Generative AI
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# モジュール群のインポート
from registry import (
    SiteRegistry,
    list_sites,
    list_networks,
    get_paths,
    load_topology,
    get_display_name,
    NetworkNode,
)
from alarm_generator import generate_alarms_for_scenario, get_alarm_summary, Alarm
from inference_engine import LogicalRCA
from network_ops import (
    run_diagnostic_simulation,
    generate_remediation_commands,
    sanitize_output,
)
from verifier import verify_log_content, format_verification_report
from rate_limiter import GlobalRateLimiter, RateLimitConfig

# --- ページ設定 ---
st.set_page_config(
    page_title="AIOps Incident Cockpit",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 定数定義
# =====================================================
class ImpactLevel:
    """影響度レベル定義"""
    COMPLETE_OUTAGE = 100
    CRITICAL = 90
    DEGRADED_HIGH = 80
    DEGRADED_MID = 70
    DOWNSTREAM = 50
    LOW_PRIORITY = 20

# シナリオと影響度のマッピング
SCENARIO_IMPACT_MAP = {
    "正常稼働": 0,
    "WAN全回線断": ImpactLevel.COMPLETE_OUTAGE,
    "[WAN] 電源障害：両系": ImpactLevel.COMPLETE_OUTAGE,
    "[L2SW] 電源障害：両系": ImpactLevel.COMPLETE_OUTAGE,
    "[Core] 両系故障": ImpactLevel.CRITICAL,
    "[FW] 電源障害：両系": ImpactLevel.CRITICAL,
    "[FW] 電源障害：片系": ImpactLevel.DEGRADED_HIGH,
    "FW片系障害": ImpactLevel.DEGRADED_HIGH,
    "[WAN] 電源障害：片系": ImpactLevel.DEGRADED_MID,
    "[L2SW] 電源障害：片系": ImpactLevel.DEGRADED_MID,
    "L2SWサイレント障害": ImpactLevel.DEGRADED_HIGH,
    "[WAN] BGPルートフラッピング": ImpactLevel.DEGRADED_HIGH,
    "[WAN] FAN故障": ImpactLevel.DEGRADED_MID,
    "[FW] FAN故障": ImpactLevel.DEGRADED_MID,
    "[L2SW] FAN故障": ImpactLevel.DEGRADED_MID,
    "[WAN] メモリリーク": ImpactLevel.DEGRADED_MID,
    "[FW] メモリリーク": ImpactLevel.DEGRADED_MID,
    "[L2SW] メモリリーク": ImpactLevel.DEGRADED_MID,
    "[WAN] 複合障害：電源＆FAN": ImpactLevel.DEGRADED_HIGH,
    "[Complex] 同時多発：FW & AP": ImpactLevel.DEGRADED_HIGH,
}

# シナリオカテゴリ
SCENARIO_MAP = {
    "基本・広域障害": [
        "正常稼働",
        "1. WAN全回線断",
        "2. FW片系障害",
        "3. L2SWサイレント障害"
    ],
    "WAN Router": [
        "4. [WAN] 電源障害：片系",
        "5. [WAN] 電源障害：両系",
        "6. [WAN] BGPルートフラッピング",
        "7. [WAN] FAN故障",
        "8. [WAN] メモリリーク"
    ],
    "Firewall": [
        "9. [FW] 電源障害：片系",
        "10. [FW] 電源障害：両系",
        "11. [FW] FAN故障",
        "12. [FW] メモリリーク"
    ],
    "L2 Switch": [
        "13. [L2SW] 電源障害：片系",
        "14. [L2SW] 電源障害：両系",
        "15. [L2SW] FAN故障",
        "16. [L2SW] メモリリーク"
    ],
    "複合・その他": [
        "17. [WAN] 複合障害：電源＆FAN",
        "18. [Complex] 同時多発：FW & AP"
    ]
}


# =====================================================
# ユーティリティ関数
# =====================================================
def get_scenario_impact_level(scenario: str) -> int:
    """シナリオの影響度を取得"""
    for key, value in SCENARIO_IMPACT_MAP.items():
        if key in scenario:
            return value
    return ImpactLevel.DEGRADED_MID


def get_status_from_alarms(scenario: str, alarms: List[Alarm]) -> str:
    """アラームからステータスを判定"""
    if not alarms:
        return "正常"
    
    impact = get_scenario_impact_level(scenario)
    
    if impact >= ImpactLevel.COMPLETE_OUTAGE:
        return "停止"
    elif impact >= ImpactLevel.DEGRADED_HIGH:
        return "要対応"
    elif impact >= ImpactLevel.DEGRADED_MID:
        # CRITICALアラームがあれば格上げ
        if any(a.severity == "CRITICAL" for a in alarms):
            return "要対応"
        return "注意"
    elif impact >= ImpactLevel.DOWNSTREAM:
        return "注意"
    else:
        return "正常"


def get_status_color(status: str) -> str:
    """ステータスに対応する色を取得"""
    return {
        "停止": "#d32f2f",
        "要対応": "#f57c00",
        "注意": "#fbc02d",
        "正常": "#4caf50"
    }.get(status, "#9e9e9e")


def get_status_icon(status: str) -> str:
    """ステータスに対応するアイコンを取得"""
    return {
        "停止": "🔴",
        "要対応": "🟠",
        "注意": "🟡",
        "正常": "🟢"
    }.get(status, "⚪")


@st.cache_resource
def get_rate_limiter():
    """レートリミッターのシングルトン"""
    return GlobalRateLimiter(RateLimitConfig(rpm=30, rpd=14400, safety_margin=0.9))


def generate_content_with_retry(model, prompt, stream=True, retries=3):
    """リトライ付きコンテンツ生成"""
    limiter = get_rate_limiter()
    for i in range(retries):
        try:
            if not limiter.wait_for_slot(timeout=60):
                raise RuntimeError("Rate limit timeout")
            limiter.record_request()
            return model.generate_content(prompt, stream=stream)
        except google_exceptions.ServiceUnavailable:
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))
    return None


# =====================================================
# セッション状態の初期化
# =====================================================
def init_session_state():
    """セッション状態を初期化"""
    defaults = {
        # 拠点別シナリオ
        "site_scenarios": {},
        # 選択中の拠点
        "active_site": None,
        # メンテナンスフラグ
        "maint_flags": {},
        # 分析結果
        "live_result": None,
        "verification_result": None,
        "generated_report": None,
        "remediation_plan": None,
        # チャット
        "messages": [],
        "chat_session": None,
        # その他
        "trigger_analysis": False,
        "logic_engines": {},
        "balloons_shown": False,
    }
    
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


init_session_state()


# =====================================================
# 拠点状態データの構築
# =====================================================
@dataclass
class SiteStatus:
    """拠点の状態情報"""
    site_id: str
    display_name: str
    scenario: str
    status: str
    alarm_count: int
    critical_count: int
    warning_count: int
    affected_devices: List[str]
    is_maintenance: bool
    mttr_estimate: str


def build_site_statuses() -> List[SiteStatus]:
    """全拠点の状態を構築"""
    sites = list_sites()
    statuses = []
    
    for site_id in sites:
        # 拠点のシナリオを取得
        scenario = st.session_state.site_scenarios.get(site_id, "正常稼働")
        
        # トポロジーを読み込み
        paths = get_paths(site_id)
        topology = load_topology(paths.topology_path)
        
        # アラームを生成
        alarms = generate_alarms_for_scenario(topology, scenario)
        summary = get_alarm_summary(alarms)
        
        # ステータスを判定
        status = get_status_from_alarms(scenario, alarms)
        
        # メンテナンスフラグ
        is_maint = st.session_state.maint_flags.get(site_id, False)
        
        # MTTR推定
        if status in ["停止", "要対応"]:
            mttr = f"{30 + summary['total'] * 5}分"
        else:
            mttr = "-"
        
        statuses.append(SiteStatus(
            site_id=site_id,
            display_name=get_display_name(site_id),
            scenario=scenario,
            status=status,
            alarm_count=summary['total'],
            critical_count=summary['critical'],
            warning_count=summary['warning'],
            affected_devices=summary['devices'],
            is_maintenance=is_maint,
            mttr_estimate=mttr
        ))
    
    # 優先度順にソート（停止 > 要対応 > 注意 > 正常）
    priority = {"停止": 0, "要対応": 1, "注意": 2, "正常": 3}
    statuses.sort(key=lambda s: (priority.get(s.status, 4), -s.alarm_count))
    
    return statuses


# =====================================================
# 拠点状態ボードの描画
# =====================================================
def render_site_status_board():
    """拠点状態ボードを描画"""
    st.subheader("🏢 拠点状態ボード")
    
    statuses = build_site_statuses()
    
    # KPIメトリクス
    count_stop = sum(1 for s in statuses if s.status == "停止")
    count_action = sum(1 for s in statuses if s.status == "要対応")
    count_warn = sum(1 for s in statuses if s.status == "注意")
    count_normal = sum(1 for s in statuses if s.status == "正常")
    
    cols = st.columns(4)
    cols[0].metric("🔴 障害発生", f"{count_stop}拠点", help="サービス停止レベル")
    cols[1].metric("🟠 要対応", f"{count_action}拠点", help="冗長性喪失")
    cols[2].metric("🟡 注意", f"{count_warn}拠点", help="軽微なアラート")
    cols[3].metric("🟢 正常", f"{count_normal}拠点", help="問題なし")
    
    st.divider()
    
    # 拠点カード表示
    if not statuses:
        st.info("拠点が登録されていません。")
        return
    
    # 2列レイアウト
    cols_per_row = 2
    for i in range(0, len(statuses), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(statuses):
                site = statuses[i + j]
                render_site_card(col, site)


def render_site_card(col, site: SiteStatus):
    """拠点カードを描画"""
    with col:
        icon = get_status_icon(site.status)
        color = get_status_color(site.status)
        
        # カード全体のコンテナ
        with st.container(border=True):
            # ヘッダー行
            header_cols = st.columns([3, 1])
            with header_cols[0]:
                st.markdown(f"### {icon} {site.display_name}")
            with header_cols[1]:
                if st.button("詳細", key=f"detail_{site.site_id}", type="primary"):
                    st.session_state.active_site = site.site_id
                    st.rerun()
            
            # メンテナンス表示
            if site.is_maintenance:
                st.caption("🛠️ メンテナンス中")
            
            # シナリオ表示
            scenario_display = site.scenario.split(". ", 1)[-1] if ". " in site.scenario else site.scenario
            st.caption(f"📋 {scenario_display}")
            
            # メトリクス行
            m_cols = st.columns(3)
            m_cols[0].metric("ステータス", site.status)
            m_cols[1].metric("アラーム", f"{site.alarm_count}件")
            m_cols[2].metric("MTTR", site.mttr_estimate)
            
            # 深刻度バー
            if site.alarm_count > 0:
                severity = min(100, 50 + site.alarm_count * 10)
                st.progress(severity / 100, text=f"深刻度: {severity}%")
            
            # 影響デバイス
            if site.affected_devices:
                st.caption(f"影響機器: {', '.join(site.affected_devices[:3])}")


# =====================================================
# トリアージ・コマンドセンター
# =====================================================
def render_triage_center():
    """トリアージ・コマンドセンターを描画"""
    st.subheader("🚨 トリアージ・コマンドセンター")
    
    statuses = build_site_statuses()
    
    # フィルタ
    col1, col2 = st.columns(2)
    with col1:
        filter_status = st.multiselect(
            "ステータスでフィルタ",
            ["停止", "要対応", "注意", "正常"],
            default=["停止", "要対応"],
            key="triage_filter"
        )
    with col2:
        show_maint = st.checkbox("メンテナンス中を含む", value=False, key="triage_maint")
    
    # フィルタ適用
    filtered = [
        s for s in statuses
        if s.status in filter_status
        and (show_maint or not s.is_maintenance)
    ]
    
    if not filtered:
        st.info("フィルタ条件に該当する拠点はありません。")
        return
    
    # テーブル形式で表示
    for site in filtered:
        with st.container(border=True):
            cols = st.columns([0.5, 2, 1.5, 1, 1.5])
            
            # アイコン
            with cols[0]:
                st.markdown(f"## {get_status_icon(site.status)}")
            
            # 拠点名
            with cols[1]:
                st.markdown(f"**{site.display_name}**")
                scenario_short = site.scenario.split(". ", 1)[-1][:30]
                st.caption(scenario_short)
            
            # アラーム数
            with cols[2]:
                if site.critical_count > 0:
                    st.error(f"🔴 {site.critical_count} CRITICAL")
                if site.warning_count > 0:
                    st.warning(f"🟡 {site.warning_count} WARNING")
            
            # MTTR
            with cols[3]:
                st.metric("MTTR", site.mttr_estimate, label_visibility="collapsed")
            
            # アクションボタン（詳細のみに統一）
            with cols[4]:
                # 停止/要対応の場合はprimaryボタン、それ以外は通常ボタン
                btn_type = "primary" if site.status in ["停止", "要対応"] else "secondary"
                if st.button("📋 詳細を確認", key=f"triage_detail_{site.site_id}", type=btn_type):
                    st.session_state.active_site = site.site_id
                    st.rerun()


# =====================================================
# サイドバー
# =====================================================
def render_sidebar():
    """サイドバーを描画"""
    with st.sidebar:
        st.header("⚡ 拠点シナリオ設定")
        st.caption("各拠点で発生させるシナリオを選択")
        
        sites = list_sites()
        
        for site_id in sites:
            display_name = get_display_name(site_id)
            
            with st.expander(f"📍 {display_name}", expanded=True):
                # カテゴリ選択
                category = st.selectbox(
                    "カテゴリ",
                    list(SCENARIO_MAP.keys()),
                    key=f"cat_{site_id}",
                    label_visibility="collapsed"
                )
                
                # シナリオ選択
                scenarios = SCENARIO_MAP[category]
                current = st.session_state.site_scenarios.get(site_id, "正常稼働")
                
                # 現在のシナリオがカテゴリ内にあればそれを選択
                default_idx = 0
                for idx, s in enumerate(scenarios):
                    if s == current or current in s:
                        default_idx = idx
                        break
                
                selected = st.radio(
                    "シナリオ",
                    scenarios,
                    index=default_idx,
                    key=f"scenario_{site_id}",
                    label_visibility="collapsed"
                )
                
                # シナリオを保存
                st.session_state.site_scenarios[site_id] = selected
        
        st.divider()
        
        # メンテナンス設定
        with st.expander("🛠️ メンテナンス設定", expanded=False):
            for site_id in sites:
                display_name = get_display_name(site_id)
                is_maint = st.checkbox(
                    display_name,
                    value=st.session_state.maint_flags.get(site_id, False),
                    key=f"maint_{site_id}"
                )
                st.session_state.maint_flags[site_id] = is_maint
        
        st.divider()
        
        # API Key設定
        api_key = None
        if GENAI_AVAILABLE:
            if "GOOGLE_API_KEY" in st.secrets:
                api_key = st.secrets["GOOGLE_API_KEY"]
            else:
                api_key = os.environ.get("GOOGLE_API_KEY")
            
            if api_key:
                st.success("✅ API 接続済み")
            else:
                st.warning("⚠️ API Key未設定")
                user_key = st.text_input("Google API Key", type="password")
                if user_key:
                    api_key = user_key
        
        return api_key


# =====================================================
# インシデント・コックピット
# =====================================================
def render_incident_cockpit(site_id: str, api_key: Optional[str]):
    """インシデント・コックピットを描画"""
    display_name = get_display_name(site_id)
    scenario = st.session_state.site_scenarios.get(site_id, "正常稼働")
    
    # ヘッダー
    col_header = st.columns([4, 1])
    with col_header[0]:
        st.markdown(f"### 🛡️ インシデント・コックピット: **{display_name}**")
    with col_header[1]:
        if st.button("← 一覧に戻る"):
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
    
    # LogicalRCA エンジン
    engine_key = f"engine_{site_id}"
    if engine_key not in st.session_state.logic_engines:
        st.session_state.logic_engines[engine_key] = LogicalRCA(topology)
    engine = st.session_state.logic_engines[engine_key]
    
    # 分析実行
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
    
    # KPIメトリクス
    st.markdown("---")
    cols = st.columns(4)
    cols[0].metric("📋 シナリオ", scenario.split(". ", 1)[-1][:20])
    cols[1].metric("🚨 ステータス", f"{get_status_icon(status)} {status}")
    cols[2].metric("📊 アラーム数", f"{len(alarms)}件")
    cols[3].metric("🎯 被疑箇所", f"{len([r for r in analysis_results if r.get('prob', 0) > 0.5])}件")
    
    st.markdown("---")
    
    # 根本原因候補とダウンストリーム機器の分離
    root_cause_candidates = []
    downstream_devices = []
    
    for cand in analysis_results:
        cand_type = cand.get('type', '')
        if "Unreachable" in cand_type or "Secondary" in cand_type:
            downstream_devices.append(cand)
        else:
            root_cause_candidates.append(cand)
    
    if root_cause_candidates and downstream_devices:
        st.info(f"📍 **根本原因**: {root_cause_candidates[0]['id']} → 影響範囲: 配下 {len(downstream_devices)} 機器")
    
    # 候補テーブル
    if root_cause_candidates:
        df_data = []
        for rank, cand in enumerate(root_cause_candidates, 1):
            prob = cand.get('prob', 0)
            cand_type = cand.get('type', 'UNKNOWN')
            
            # ステータス判定
            if "Silent" in cand_type:
                status_text = "🟣 サイレント疑い"
                action = "🔍 上位確認"
            elif prob > 0.8:
                status_text = "🔴 根本原因"
                action = "🚀 自動修復可能"
            elif prob > 0.6:
                status_text = "🟡 被疑箇所"
                action = "🔍 詳細調査"
            else:
                status_text = "⚪ 監視中"
                action = "👁️ 静観"
            
            df_data.append({
                "順位": rank,
                "ステータス": status_text,
                "デバイス": cand['id'],
                "原因": cand.get('label', ''),
                "確信度": f"{prob*100:.0f}%",
                "推奨アクション": action,
                "_id": cand['id'],
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
        
        # 選択された候補
        selected_candidate = None
        if event.selection and len(event.selection.rows) > 0:
            sel_row = df.iloc[event.selection.rows[0]]
            for cand in root_cause_candidates:
                if cand['id'] == sel_row['_id']:
                    selected_candidate = cand
                    break
        elif root_cause_candidates:
            selected_candidate = root_cause_candidates[0]
        
        # 下流デバイス
        if downstream_devices:
            with st.expander(f"▼ 影響を受けている機器 ({len(downstream_devices)}台)", expanded=False):
                dd_df = pd.DataFrame([
                    {"No": i+1, "デバイス": d['id'], "状態": "⚫ 応答なし", "備考": "上流復旧待ち"}
                    for i, d in enumerate(downstream_devices)
                ])
                st.dataframe(dd_df, use_container_width=True, hide_index=True)
        
        # 2カラムレイアウト
        col_map, col_detail = st.columns([1.2, 1])
        
        with col_map:
            st.markdown("#### 🌐 ネットワークトポロジー")
            graph = render_topology_graph(topology, alarms, analysis_results)
            st.graphviz_chart(graph, use_container_width=True)
        
        with col_detail:
            if selected_candidate:
                render_detail_panel(selected_candidate, scenario, topology, api_key)
    else:
        st.success("✅ 現在、対応が必要なインシデントはありません。")


def render_topology_graph(topology: dict, alarms: List[Alarm], analysis_results: List[dict]):
    """トポロジーグラフを生成"""
    graph = graphviz.Digraph()
    graph.attr(rankdir='TB')
    graph.attr('node', shape='box', style='rounded,filled', fontname='Helvetica')
    
    # アラームマップ
    alarm_ids = set(a.device_id for a in alarms)
    status_map = {r['id']: r.get('type', '') for r in analysis_results}
    
    for node_id, node in topology.items():
        # ノード情報取得
        if hasattr(node, 'type'):
            node_type = node.type
            metadata = node.metadata if hasattr(node, 'metadata') else {}
        else:
            node_type = node.get('type', 'UNKNOWN')
            metadata = node.get('metadata', {})
        
        color = "#e8f5e9"
        penwidth = "1"
        label = f"{node_id}\n({node_type})"
        
        # ベンダー表示
        vendor = metadata.get("vendor", "")
        if vendor:
            label += f"\n[{vendor}]"
        
        # ステータスに基づく色
        status_type = status_map.get(node_id, "")
        
        if "Silent" in status_type:
            color = "#fff3e0"
            penwidth = "4"
            label += "\n[サイレント疑い]"
        elif "Physical" in status_type or "Critical" in status_type:
            color = "#ffcdd2"
            penwidth = "3"
            label += "\n[ROOT CAUSE]"
        elif "Unreachable" in status_type:
            color = "#cfd8dc"
            label += "\n[Unreachable]"
        elif node_id in alarm_ids:
            color = "#fff9c4"
        
        graph.node(node_id, label=label, fillcolor=color, penwidth=penwidth)
    
    # エッジ
    for node_id, node in topology.items():
        if hasattr(node, 'parent_id'):
            parent_id = node.parent_id
        else:
            parent_id = node.get('parent_id')
        
        if parent_id:
            graph.edge(parent_id, node_id)
    
    return graph


def render_detail_panel(candidate: dict, scenario: str, topology: dict, api_key: Optional[str]):
    """詳細パネルを描画"""
    st.markdown("#### 📝 詳細分析")
    
    device_id = candidate['id']
    prob = candidate.get('prob', 0)
    
    with st.container(border=True):
        st.markdown(f"**対象デバイス**: `{device_id}`")
        st.markdown(f"**原因**: {candidate.get('label', 'N/A')}")
        st.markdown(f"**確信度**: {prob*100:.0f}%")
        st.markdown(f"**理由**: {candidate.get('reason', 'N/A')}")
    
    # 修復プラン生成
    if prob > 0.6 and api_key and GENAI_AVAILABLE:
        st.markdown("#### 🛠️ 復旧支援")
        
        if "remediation_plan" not in st.session_state or st.session_state.remediation_plan is None:
            if st.button("✨ 修復プランを生成"):
                with st.spinner("生成中..."):
                    node = topology.get(device_id)
                    if node:
                        plan = generate_remediation_commands(
                            scenario,
                            f"Root Cause: {candidate['label']}",
                            node,
                            api_key
                        )
                        st.session_state.remediation_plan = plan
                        st.rerun()
        else:
            with st.container(border=True):
                st.markdown(st.session_state.remediation_plan)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 実行", type="primary"):
                    st.success("修復を実行しました。（デモ）")
            with col2:
                if st.button("❌ クリア"):
                    st.session_state.remediation_plan = None
                    st.rerun()


# =====================================================
# メイン
# =====================================================
def main():
    """メインエントリーポイント"""
    # サイドバー描画
    api_key = render_sidebar()
    
    # タイトル
    st.title("🛡️ AIOps インシデント・コックピット")
    st.caption("複数拠点のネットワーク障害を統合管理・分析")
    
    # アクティブな拠点があればコックピット表示
    active_site = st.session_state.get("active_site")
    
    if active_site:
        render_incident_cockpit(active_site, api_key)
    else:
        # タブ切り替え
        tab1, tab2 = st.tabs(["📊 拠点状態ボード", "🚨 トリアージ・コマンドセンター"])
        
        with tab1:
            render_site_status_board()
        
        with tab2:
            render_triage_center()


if __name__ == "__main__":
    main()
