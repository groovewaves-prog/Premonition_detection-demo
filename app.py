# -*- coding: utf-8 -*-
"""
AIOps Incident Cockpit - Multi-Site Edition
=============================================
複数拠点対応版 AIOps インシデント・コックピット
[修正] 重複していた予兆検知バナーを削除
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
from alarm_generator import generate_alarms_for_scenario, get_alarm_summary, Alarm, NodeColor
from inference_engine import LogicalRCA
from network_ops import (
    run_diagnostic_simulation,
    generate_remediation_commands,
    generate_analyst_report_streaming,
    generate_remediation_commands_streaming,
    run_remediation_parallel_v2,
    RemediationEnvironment,
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


def _hash_text(text: str) -> str:
    """テキストのハッシュ値を計算"""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def load_config_by_id(device_id: str) -> str:
    """configsフォルダから設定ファイルを読み込む"""
    possible_paths = [f"configs/{device_id}.txt", f"{device_id}.txt"]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass
    return "Config file not found."


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


def _pick_first(mapping: dict, keys: list, default: str = "") -> str:
    """マッピングから最初の非空値を取得"""
    for k in keys:
        try:
            v = mapping.get(k, None)
        except Exception:
            v = None
        if v is None:
            continue
        if isinstance(v, (int, float, bool)):
            s = str(v)
            if s:
                return s
        elif isinstance(v, str):
            if v.strip():
                return v.strip()
    return default


def _build_ci_context_for_chat(topology: dict, target_node_id: str) -> dict:
    """チャット用のCIコンテキストを構築"""
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
        "vendor": _pick_first(md, ["vendor", "manufacturer", "maker", "brand"], default=""),
        "os": _pick_first(md, ["os", "platform", "os_name", "software", "sw"], default=""),
        "model": _pick_first(md, ["model", "hw_model", "product", "sku"], default=""),
        "role": _pick_first(md, ["role", "type", "device_role"], default=""),
        "layer": _pick_first(md, ["layer", "level", "network_layer"], default=""),
        "site": _pick_first(md, ["site", "dc", "datacenter", "location"], default=""),
    }

    try:
        conf = load_config_by_id(target_node_id) if target_node_id else ""
        if conf:
            ci["config_excerpt"] = conf[:1500]
    except Exception:
        pass

    return ci


def run_diagnostic_simulation_no_llm(selected_scenario: str, target_node_obj) -> dict:
    """LLMを呼ばない疑似診断"""
    device_id = getattr(target_node_obj, "id", "UNKNOWN") if target_node_obj else "UNKNOWN"
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"[PROBE] ts={ts}",
        f"[PROBE] scenario={selected_scenario}",
        f"[PROBE] target_device={device_id}",
        "",
    ]

    # 復旧成功フラグ（デモ用）
    recovered_devices = st.session_state.get("recovered_devices") or {}
    recovered_map = st.session_state.get("recovered_scenario_map") or {}

    if recovered_devices.get(device_id) and recovered_map.get(device_id) == selected_scenario:
        # 復旧後の疑似ログ
        if "FW" in selected_scenario:
            lines += [
                "show chassis cluster status",
                "Redundancy group 0: healthy",
                "control link: up",
                "fabric link: up",
            ]
        elif "WAN" in selected_scenario or "WAN全回線断" in selected_scenario:
            lines += [
                "show ip interface brief",
                "GigabitEthernet0/0 up up",
                "show ip bgp summary",
                "Neighbor 203.0.113.2 Established",
                "ping 203.0.113.2 repeat 5",
                "Success rate is 100 percent (5/5)",
            ]
        elif "L2SW" in selected_scenario:
            lines += [
                "show environment",
                "Fan: OK",
                "Temperature: OK",
                "show interface status",
                "Uplink: up",
            ]
        else:
            lines += [
                "show system alarms",
                "No active alarms",
                "ping 8.8.8.8 repeat 5",
                "Success rate is 100 percent (5/5)",
            ]
        return {
            "status": "SUCCESS",
            "sanitized_log": "\n".join(lines),
            "device_id": device_id,
        }

    # 障害中の疑似ログ
    if "WAN全回線断" in selected_scenario or "[WAN]" in selected_scenario:
        lines += [
            "show ip interface brief",
            "GigabitEthernet0/0 down down",
            "show ip bgp summary",
            "Neighbor 203.0.113.2 Idle",
            "ping 203.0.113.2 repeat 5",
            "Success rate is 0 percent (0/5)",
        ]
    elif "FW片系障害" in selected_scenario or "[FW]" in selected_scenario:
        lines += [
            "show chassis cluster status",
            "Redundancy group 0: degraded",
            "control link: down",
            "fabric link: up",
        ]
    elif "L2SW" in selected_scenario:
        lines += [
            "show environment",
            "Fan: FAIL",
            "Temperature: HIGH",
            "show interface status",
            "Uplink: flapping",
        ]
    else:
        lines += [
            "show system alarms",
            "No active alarms",
        ]

    return {
        "status": "SUCCESS",
        "sanitized_log": "\n".join(lines),
        "device_id": device_id,
    }


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
        "verification_log": None,
        # チャット
        "messages": [],
        "chat_session": None,
        "chat_quick_text": "",
        # その他
        "trigger_analysis": False,
        "logic_engines": {},
        "balloons_shown": False,
        "recovered_devices": {},
        "recovered_scenario_map": {},
        "report_cache": {},
        "injected_weak_signal": None,
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
        scenario = st.session_state.site_scenarios.get(site_id, "正常稼働")
        paths = get_paths(site_id)
        topology = load_topology(paths.topology_path)
        alarms = generate_alarms_for_scenario(topology, scenario)
        summary = get_alarm_summary(alarms)
        status = get_status_from_alarms(scenario, alarms)
        is_maint = st.session_state.maint_flags.get(site_id, False)
        
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
    
    if not statuses:
        st.info("拠点が登録されていません。")
        return
    
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
        
        with st.container(border=True):
            header_cols = st.columns([3, 1])
            with header_cols[0]:
                st.markdown(f"### {icon} {site.display_name}")
            with header_cols[1]:
                if st.button("詳細", key=f"detail_{site.site_id}", type="primary"):
                    st.session_state.active_site = site.site_id
                    # セッション状態をリセット
                    st.session_state.live_result = None
                    st.session_state.verification_result = None
                    st.session_state.generated_report = None
                    st.session_state.remediation_plan = None
                    st.session_state.messages = []
                    st.session_state.chat_session = None
                    st.rerun()
            
            if site.is_maintenance:
                st.caption("🛠️ メンテナンス中")
            
            scenario_display = site.scenario.split(". ", 1)[-1] if ". " in site.scenario else site.scenario
            st.caption(f"📋 {scenario_display}")
            
            m_cols = st.columns(3)
            m_cols[0].metric("ステータス", site.status)
            m_cols[1].metric("アラーム", f"{site.alarm_count}件")
            m_cols[2].metric("MTTR", site.mttr_estimate)
            
            if site.alarm_count > 0:
                # 深刻度 = CRITICAL × 30 + WARNING × 10（最大100%）
                severity = min(100, site.critical_count * 30 + site.warning_count * 10)
                st.progress(severity / 100, text=f"深刻度: {severity}%")
            
            if site.affected_devices:
                st.caption(f"影響機器: {', '.join(site.affected_devices[:3])}")


# =====================================================
# トリアージ・コマンドセンター
# =====================================================
def render_triage_center():
    """トリアージ・コマンドセンターを描画"""
    st.subheader("🚨 トリアージ・コマンドセンター")
    
    statuses = build_site_statuses()
    
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
    
    filtered = [
        s for s in statuses
        if s.status in filter_status
        and (show_maint or not s.is_maintenance)
    ]
    
    if not filtered:
        st.info("フィルタ条件に該当する拠点はありません。")
        return
    
    for site in filtered:
        with st.container(border=True):
            cols = st.columns([0.5, 2, 1.5, 1, 1.5])
            
            with cols[0]:
                st.markdown(f"## {get_status_icon(site.status)}")
            
            with cols[1]:
                st.markdown(f"**{site.display_name}**")
                scenario_short = site.scenario.split(". ", 1)[-1][:30]
                st.caption(scenario_short)
            
            with cols[2]:
                if site.critical_count > 0:
                    st.error(f"🔴 {site.critical_count} CRITICAL")
                if site.warning_count > 0:
                    st.warning(f"🟡 {site.warning_count} WARNING")
            
            with cols[3]:
                st.metric("MTTR", site.mttr_estimate, label_visibility="collapsed")
            
            with cols[4]:
                btn_type = "primary" if site.status in ["停止", "要対応"] else "secondary"
                if st.button("📋 詳細を確認", key=f"triage_detail_{site.site_id}", type=btn_type):
                    st.session_state.active_site = site.site_id
                    st.session_state.live_result = None
                    st.session_state.verification_result = None
                    st.session_state.generated_report = None
                    st.session_state.remediation_plan = None
                    st.session_state.messages = []
                    st.session_state.chat_session = None
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
                category = st.selectbox(
                    "カテゴリ",
                    list(SCENARIO_MAP.keys()),
                    key=f"cat_{site_id}",
                    label_visibility="collapsed"
                )
                
                scenarios = SCENARIO_MAP[category]
                current = st.session_state.site_scenarios.get(site_id, "正常稼働")
                
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
                
                # シナリオが変更された場合、該当サイトのセッション状態のみリセット
                if selected != current:
                    st.session_state.site_scenarios[site_id] = selected
                    # 該当サイトのレポート関連キャッシュをクリア
                    keys_to_remove = [k for k in list(st.session_state.report_cache.keys()) if site_id in k]
                    for k in keys_to_remove:
                        del st.session_state.report_cache[k]
                    # アクティブサイトが変更されたサイトの場合のみ、レポート状態をリセット
                    if st.session_state.active_site == site_id:
                        st.session_state.generated_report = None
                        st.session_state.remediation_plan = None
                        st.session_state.messages = []
                        st.session_state.chat_session = None
                        st.session_state.live_result = None
                        st.session_state.verification_result = None
                else:
                    st.session_state.site_scenarios[site_id] = selected
        
        st.divider()
        
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
        
        # ★★★ 予兆シミュレーション（Weak Signal Injection） ★★★
        with st.expander("🔮 予兆シミュレーション", expanded=True):
            st.caption("正常稼働中の機器に微細なシグナルを注入し、AIによる予兆検知をデモします。")
            
            # ★ 改善1: トポロジーから動的にデバイスリスト生成
            # 配下を持つ機器のみ（AP等の末端は影響伝搬しないため除外）
            active = st.session_state.get("active_site")
            device_options = []
            try:
                site_for_list = active if active else list_sites()[0] if list_sites() else None
                if site_for_list:
                    paths = get_paths(site_for_list)
                    topo = load_topology(paths.topology_path)
                    if topo:
                        # 各デバイスの配下数を計算
                        child_count = {}
                        for dev_id, info in topo.items():
                            pid = info.parent_id if hasattr(info, 'parent_id') else info.get('parent_id')
                            if pid:
                                child_count[pid] = child_count.get(pid, 0) + 1
                        # 配下を持つデバイスのみリスト（影響伝搬する機器）
                        for dev_id, info in topo.items():
                            if child_count.get(dev_id, 0) > 0:
                                dtype = info.type if hasattr(info, 'type') else info.get('type', '')
                                layer = info.layer if hasattr(info, 'layer') else info.get('layer', 0)
                                rg = info.redundancy_group if hasattr(info, 'redundancy_group') else info.get('redundancy_group')
                                n_children = child_count.get(dev_id, 0)
                                tag = "⚠SPOF" if not rg else "HA"
                                device_options.append((dev_id, f"L{layer} {dev_id} ({dtype}) [{tag}, 配下{n_children}台]"))
                        # layer 順 → SPOF 優先でソート
                        device_options.sort(key=lambda x: x[1])
            except Exception:
                pass
            
            if not device_options:
                device_options = [("WAN_ROUTER_01", "WAN_ROUTER_01")]
            
            target_device = st.selectbox(
                "対象デバイス",
                [d[0] for d in device_options],
                format_func=lambda x: next((d[1] for d in device_options if d[0] == x), x),
                key="pred_target"
            )
            
            scenario_type = st.selectbox(
                "劣化シナリオ",
                ["Optical Decay (光減衰)", "Microburst (パケット破棄)", "Route Instability (経路揺らぎ)"],
                key="pred_scenario"
            )
            
            degradation_level = st.slider(
                "劣化進行度",
                min_value=0, max_value=5, value=0,
                help="0:正常 → 5:障害発生直前。レベルが上がると相関シグナルが増加し、予測精度が向上します。",
                key="pred_level"
            )
            
            # ★ 改善2: 多段シグナル注入（レベルに応じて相関ログが増加）
            # 論文の設計: 複数の微弱シグナルの相関から障害確率を高める
            log_messages = []
            if degradation_level > 0:
                if "Optical" in scenario_type:
                    dbm = -23.0 - (degradation_level * 0.4)
                    # レベル1: 光減衰の初期兆候
                    log_messages.append(
                        f"%TRANSCEIVER-4-THRESHOLD_VIOLATION: Rx Power {dbm:.1f} dBm (Threshold -25.0 dBm). Signal degrading."
                    )
                    if degradation_level >= 2:
                        # レベル2: CRCエラー増加（光劣化の二次症状）
                        crc = degradation_level * 150
                        log_messages.append(
                            f"%LINK-3-ERROR: CRC errors increasing on Gi0/0/0 (Count: {crc}/min). Input queue drops detected."
                        )
                    if degradation_level >= 4:
                        # レベル4: 隣接機器からの keepalive 遅延
                        log_messages.append(
                            "%OSPF-4-ADJCHANGE: Neighbor keepalive delayed (3 consecutive misses). Stability warning."
                        )
                
                elif "Microburst" in scenario_type:
                    drops = degradation_level * 200
                    log_messages.append(
                        f"%HARDWARE-3-ASIC_ERROR: Input queue drops detected (Count: {drops}). Burst traffic."
                    )
                    if degradation_level >= 2:
                        log_messages.append(
                            f"%QOS-4-POLICER: Traffic exceeding CIR on interface ge-0/0/1. Buffer overflow risk."
                        )
                    if degradation_level >= 4:
                        retrans = degradation_level * 50
                        log_messages.append(
                            f"%TCP-5-RETRANSMIT: Retransmission rate {retrans}/sec on monitored flows. Route updates increasing."
                        )
                
                elif "Route" in scenario_type:
                    updates = degradation_level * 500
                    log_messages.append(
                        f"BGP-5-ADJCHANGE: Route updates {updates}/min. Stability warning."
                    )
                    if degradation_level >= 2:
                        log_messages.append(
                            f"%BGP-4-MAXPFX: Prefix count approaching limit (92%). Route oscillation detected."
                        )
                    if degradation_level >= 4:
                        log_messages.append(
                            "%ROUTING-3-CONVERGENCE: RIB convergence delayed. Prefix withdrawal detected on multiple peers."
                        )
            
            if log_messages:
                st.session_state["injected_weak_signal"] = {
                    "device_id": target_device,
                    "messages": log_messages,  # ★ 複数メッセージに変更
                    "message": log_messages[0],  # 後方互換
                    "level": degradation_level,
                    "scenario": scenario_type,
                }
                # 注入中のシグナル表示
                st.info(f"💉 **{len(log_messages)}件のシグナル注入中** (Level {degradation_level}/5)")
                for i, msg in enumerate(log_messages, 1):
                    st.caption(f"  {i}. `{msg[:70]}...`" if len(msg) > 70 else f"  {i}. `{msg}`")
            else:
                st.session_state["injected_weak_signal"] = None
        
        api_key = None
        if GENAI_AVAILABLE:
            if "GOOGLE_API_KEY" in st.secrets:
                api_key = st.secrets["GOOGLE_API_KEY"]
            else:
                api_key = os.environ.get("GOOGLE_API_KEY")
            
            if api_key:
                st.success("✅ API 接続済み")
                stats = get_rate_limiter().get_stats()
                st.caption(f"📊 API: {stats['requests_last_minute']}/{stats['rpm_limit']} RPM")
            else:
                st.warning("⚠️ API Key未設定")
                user_key = st.text_input("Google API Key", type="password")
                if user_key:
                    api_key = user_key
        
        return api_key


# =====================================================
# トポロジー描画
# =====================================================
def render_topology_graph(topology: dict, alarms: List[Alarm], analysis_results: List[dict]):
    """
    トポロジーグラフを生成
    """
    graph = graphviz.Digraph()
    graph.attr(rankdir='TB')
    graph.attr('node', shape='box', style='rounded,filled', fontname='Helvetica')
    
    # アラーム情報をデバイスIDでマッピング
    alarm_map = {}
    for a in alarms:
        if a.device_id not in alarm_map:
            alarm_map[a.device_id] = {
                'is_root_cause': False,
                'is_silent_suspect': False,
                'max_severity': 'INFO',
                'messages': []
            }
        info = alarm_map[a.device_id]
        info['messages'].append(a.message)
        if a.is_root_cause:
            info['is_root_cause'] = True
        if a.is_silent_suspect:
            info['is_silent_suspect'] = True
        # 最大severity を更新
        severity_order = {'CRITICAL': 3, 'WARNING': 2, 'INFO': 1}
        if severity_order.get(a.severity, 0) > severity_order.get(info['max_severity'], 0):
            info['max_severity'] = a.severity
    
    for node_id, node in topology.items():
        if hasattr(node, 'type'):
            node_type = node.type
            metadata = node.metadata if hasattr(node, 'metadata') else {}
        else:
            node_type = node.get('type', 'UNKNOWN')
            metadata = node.get('metadata', {})
        
        # デフォルト: 正常（グリーン）
        color = NodeColor.NORMAL
        penwidth = "1"
        fontcolor = "black"
        label = f"{node_id}\n({node_type})"
        status_label = ""
        
        red_type = metadata.get("redundancy_type")
        if red_type:
            label += f"\n[{red_type} Redundancy]"
        vendor = metadata.get("vendor")
        if vendor:
            label += f"\n[{vendor}]"
        
        # アラーム情報に基づいて色を決定
        if node_id in alarm_map:
            info = alarm_map[node_id]
            
            if info['is_root_cause']:
                # 根本原因
                if info['is_silent_suspect']:
                    # サイレント障害疑い（薄紫色）
                    color = NodeColor.SILENT_FAILURE
                    penwidth = "3"
                    status_label = "\n[SILENT SUSPECT]"
                elif info['max_severity'] == 'CRITICAL':
                    # サービス停止レベル（赤色）
                    color = NodeColor.ROOT_CAUSE_CRITICAL
                    penwidth = "3"
                    status_label = "\n[ROOT CAUSE]"
                else:
                    # 冗長性低下レベル（黄色）
                    color = NodeColor.ROOT_CAUSE_WARNING
                    penwidth = "2"
                    status_label = "\n[WARNING]"
            else:
                # 影響デバイス（グレー）
                color = NodeColor.UNREACHABLE
                fontcolor = "#546e7a"
                status_label = "\n[Unreachable]"
        
        label += status_label
        
        graph.node(node_id, label=label, fillcolor=color, color='black', penwidth=penwidth, fontcolor=fontcolor)
    
    for node_id, node in topology.items():
        if hasattr(node, 'parent_id'):
            parent_id = node.parent_id
        else:
            parent_id = node.get('parent_id')
        
        if parent_id:
            graph.edge(parent_id, node_id)
            parent_node = topology.get(parent_id)
            if parent_node:
                if hasattr(parent_node, 'redundancy_group'):
                    rg = parent_node.redundancy_group
                else:
                    rg = parent_node.get('redundancy_group')
                if rg:
                    for nid, n in topology.items():
                        n_rg = n.redundancy_group if hasattr(n, 'redundancy_group') else n.get('redundancy_group')
                        if n_rg == rg and nid != parent_id:
                            graph.edge(nid, node_id)
    
    return graph


# =====================================================
# インシデント・コックピット（前回のUXを完全復元）
# =====================================================
def render_incident_cockpit(site_id: str, api_key: Optional[str]):
    """インシデント・コックピットを描画（前回のUXを完全復元）"""
    display_name = get_display_name(site_id)
    scenario = st.session_state.site_scenarios.get(site_id, "正常稼働")
    
    # ヘッダー
    col_header = st.columns([4, 1])
    with col_header[0]:
        st.markdown(f"### 🛡️ AIOps インシデント・コックピット")
    with col_header[1]:
        # 戻るボタン専用の赤色スタイル（マーカーで特定）
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
    
    # トポロジー読み込み
    paths = get_paths(site_id)
    topology = load_topology(paths.topology_path)
    
    if not topology:
        st.error("トポロジーが読み込めませんでした。")
        return
    
    # アラーム生成
    alarms = generate_alarms_for_scenario(topology, scenario)
    status = get_status_from_alarms(scenario, alarms)
    
    # ★★★ 予兆シグナル注入（Weak Signal Injection） ★★★
    injected = st.session_state.get("injected_weak_signal")
    if injected and injected["device_id"] in topology:
        # ★ 複数シグナル注入（レベルに応じてログが増加）
        messages = injected.get("messages", [injected.get("message", "")])
        for msg in messages:
            if msg:
                weak_alarm = Alarm(
                    device_id=injected["device_id"],
                    message=msg,
                    severity="INFO",
                    is_root_cause=False
                )
                alarms.append(weak_alarm)
    
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
    
    # =====================================================
    # KPIメトリクス（元の情報 + 新しいメトリクス）
    # =====================================================
    # 根本原因候補の数を計算
    root_cause_alarms = [a for a in alarms if a.is_root_cause]
    downstream_alarms = [a for a in alarms if not a.is_root_cause]
    
    # ノイズ削減率の計算
    total_alarms = len(alarms)
    if total_alarms > 0:
        noise_reduction = ((total_alarms - len(root_cause_alarms)) / total_alarms) * 100
    else:
        noise_reduction = 0.0
    
    # 要対応インシデント数（根本原因の数）
    action_required = len(set(a.device_id for a in root_cause_alarms))
    
    # ★ Digital Twin 予兆検知数
    prediction_results = [r for r in analysis_results if r.get('is_prediction')]
    prediction_count = len(prediction_results)
    
    # --- 元のKPIメトリクス表示（上段） ---
    st.markdown("---")
    cols = st.columns(3)
    cols[0].metric("🚨 ステータス", f"{get_status_icon(status)} {status}")
    cols[1].metric("📊 アラーム数", f"{len(alarms)}件")
    suspect_count = len([r for r in analysis_results if r.get('prob', 0) > 0.5])
    if prediction_count > 0:
        cols[2].metric("🎯 被疑箇所", f"{suspect_count}件",
                       delta=f"うち🔮予兆 {prediction_count}件", delta_color="off")
    else:
        cols[2].metric("🎯 被疑箇所", f"{suspect_count}件")
    
    # --- 新しいKPIメトリクス表示（下段） ---
    kpi_cols = st.columns(3)
    with kpi_cols[0]:
        if noise_reduction > 90:
            delta_text = "↑ 高効率稼働中"
            delta_color = "normal"
        elif noise_reduction > 50:
            delta_text = "→ 通常稼働"
            delta_color = "off"
        else:
            delta_text = "↓ 要確認"
            delta_color = "inverse"
        st.metric(
            "📉 ノイズ削減率",
            f"{noise_reduction:.1f}%",
            delta=delta_text,
            delta_color=delta_color
        )
    
    with kpi_cols[1]:
        if prediction_count > 0:
            delta_text = "⚡ 要注意"
            delta_color = "inverse"
        else:
            delta_text = "問題なし"
            delta_color = "normal"
        st.metric(
            "🔮 予兆検知",
            f"{prediction_count}件",
            delta=delta_text,
            delta_color=delta_color
        )
    
    with kpi_cols[2]:
        if action_required > 0:
            delta_text = "↑ 対処が必要"
            delta_color = "inverse"
        else:
            delta_text = "問題なし"
            delta_color = "normal"
        st.metric(
            "🚨 要対応インシデント",
            f"{action_required}件",
            delta=delta_text,
            delta_color=delta_color
        )
    
    st.markdown("---")
    
    # =====================================================
    # 根本原因候補とダウンストリーム機器の分離
    # =====================================================
    # アラーム情報を使って根本原因と影響デバイスを分離
    root_cause_device_ids = set(a.device_id for a in alarms if a.is_root_cause)
    downstream_device_ids = set(a.device_id for a in alarms if not a.is_root_cause)
    
    root_cause_candidates = []
    downstream_devices = []
    
    for cand in analysis_results:
        device_id = cand.get('id', '')
        # ★ 予兆検知は常に根本原因候補として扱う
        if cand.get('is_prediction'):
            root_cause_candidates.append(cand)
        elif device_id in root_cause_device_ids:
            root_cause_candidates.append(cand)
        elif device_id in downstream_device_ids:
            downstream_devices.append(cand)
        elif cand.get('prob', 0) > 0.5:
            # 分析結果からも根本原因候補を抽出
            root_cause_candidates.append(cand)
    
    # 正常稼働時のデフォルト
    if not root_cause_candidates and not alarms:
        root_cause_candidates = [{
            "id": "SYSTEM",
            "label": "正常稼働",
            "prob": 0.0,
            "type": "Normal",
            "tier": 3,
            "reason": "アラームなし"
        }]
    
    if root_cause_candidates and downstream_devices:
        st.info(f"📍 **根本原因**: {root_cause_candidates[0]['id']} → 影響範囲: 配下 {len(downstream_devices)} 機器")
    
    # ★ Digital Twin 予兆検知バナー表示の削除箇所
    # (ここに以前あった st.warning(...) ブロックを削除しました)
    # 2025-02-12: ユーザー要望により削除（Future Radarと重複するため）
    
    # =====================================================
    # 🔮 AIOps Future Radar（予兆専用表示エリア）
    # =====================================================
    prediction_candidates = [c for c in root_cause_candidates if c.get('is_prediction')]
    
    if prediction_candidates:
        st.markdown("### 🔮 AIOps Future Radar")
        
        with st.container(border=True):
            # ヘッダー
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
                st.info(
                    "⚠️ **予兆検知**: 現在のネットワーク状態は「正常」ですが、"
                    "AIが将来の障害リスクを検出しました。"
                )
            
            # 各予兆のカード表示
            radar_cols = st.columns(min(len(prediction_candidates), 3))
            for idx, pred_item in enumerate(prediction_candidates[:3]):
                with radar_cols[idx]:
                    prob_val = pred_item.get('prob', 0)
                    prob_pct = f"{prob_val*100:.0f}%"
                    pred_timeline = pred_item.get('prediction_timeline', '不明')
                    pred_affected = pred_item.get('prediction_affected_count', 0)
                    pred_label = pred_item.get('label', '').replace('🔮 [予兆] ', '')
                    pred_early_hours = pred_item.get('prediction_early_warning_hours', 0)
                    
                    st.error(f"**📍 {pred_item['id']}**")
                    
                    st.markdown(
                        f"<div style='text-align:center;'>"
                        f"<span style='font-size:36px; font-weight:bold; color:#d32f2f;'>{prob_pct}</span>"
                        f"<br><span style='color:#666;'>発生確率（急性期: {pred_timeline}）</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    
                    st.divider()
                    st.markdown(f"**予測障害:** {pred_label}")
                    # ★ 2軸表示: 早期予兆 + 急性期
                    if pred_early_hours >= 24:
                        early_display = f"最大 **{pred_early_hours // 24}日前** から検知可能"
                    elif pred_early_hours > 0:
                        early_display = f"最大 **{pred_early_hours}時間前** から検知可能"
                    else:
                        early_display = "不明"
                    st.markdown(f"**早期予兆:** {early_display}")
                    st.markdown(f"**急性期:** 発症後 **{pred_timeline}** に深刻化")
                    st.markdown(f"**影響範囲:** 配下 **{pred_affected}台** が通信断の恐れ")
                    
                    # 検知された Weak Signal の詳細
                    with st.expander("🔍 検知された予兆 (Weak Signal)"):
                        reason = pred_item.get('reason', '')
                        st.text(reason)
                        
                        factors = pred_item.get('prediction_confidence_factors', {})
                        if factors:
                            st.caption(
                                f"ベース信頼度: {factors.get('base', 0):.2f} / "
                                f"マッチ品質: {factors.get('match_quality', 0):.2f} / "
                                f"SPOF: {'Yes' if factors.get('is_spof') else 'No'} / "
                                f"冗長性: {'Yes' if factors.get('has_redundancy') else 'No'}"
                            )
        
        st.markdown("---")
    
    # 候補テーブル
    selected_incident_candidate = None
    target_device_id = None
    
    if root_cause_candidates:
        # アラームからseverityとis_silent_suspectを取得するマップを作成
        alarm_info_map = {}
        for a in alarms:
            if a.device_id not in alarm_info_map:
                alarm_info_map[a.device_id] = {'severity': 'INFO', 'is_silent': False}
            if a.severity == 'CRITICAL':
                alarm_info_map[a.device_id]['severity'] = 'CRITICAL'
            elif a.severity == 'WARNING' and alarm_info_map[a.device_id]['severity'] != 'CRITICAL':
                alarm_info_map[a.device_id]['severity'] = 'WARNING'
            if a.is_silent_suspect:
                alarm_info_map[a.device_id]['is_silent'] = True
        
        df_data = []
        for rank, cand in enumerate(root_cause_candidates, 1):
            prob = cand.get('prob', 0)
            cand_type = cand.get('type', 'UNKNOWN')
            device_id = cand['id']
            
            # アラーム情報からステータスを判定
            alarm_info = alarm_info_map.get(device_id, {'severity': 'INFO', 'is_silent': False})
            
            # ★ Digital Twin 予兆検知 (is_prediction を最優先でチェック)
            if cand.get('is_prediction'):
                status_text = "🔮 予兆検知"
                timeline = cand.get('prediction_timeline', '')
                affected = cand.get('prediction_affected_count', 0)
                early_hours = cand.get('prediction_early_warning_hours', 0)
                if early_hours >= 24:
                    early_str = f"(予兆: {early_hours // 24}日前〜)"
                elif early_hours > 0:
                    early_str = f"(予兆: {early_hours}時間前〜)"
                else:
                    early_str = ""
                if timeline and affected:
                    action = f"⚡ 急性期{timeline}以内 {early_str} ({affected}台影響)"
                else:
                    action = f"⚡ 予防的対処を推奨 {early_str}"
            elif alarm_info['is_silent'] or "Silent" in cand_type:
                status_text = "🟣 サイレント疑い"
                action = "🔍 上位確認"
            elif alarm_info['severity'] == 'CRITICAL':
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
        
        # 下流デバイス（10台以上の場合はスクロール表示）
        if downstream_devices:
            with st.expander(f"▼ 影響を受けている機器 ({len(downstream_devices)}台) - 上流復旧待ち", expanded=False):
                dd_df = pd.DataFrame([
                    {"No": i+1, "デバイス": d['id'], "状態": "⚫ 応答なし", "備考": "上流復旧待ち"}
                    for i, d in enumerate(downstream_devices)
                ])
                # 10台以上の場合はスクロール可能なコンテナ内に表示
                if len(downstream_devices) >= 10:
                    with st.container(height=300):
                        st.dataframe(dd_df, use_container_width=True, hide_index=True)
                else:
                    st.dataframe(dd_df, use_container_width=True, hide_index=True)
    
    # ========================================
    # 2カラムレイアウト（前回のUXを復元）
    # ========================================
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
                        verification = verify_log_content(log_content)
                        st.session_state.verification_result = verification
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
    
    # === 右カラム: AI Analyst Report & Remediation & Chat ===
    with col_chat:
        st.subheader("📝 AI Analyst Report")
        
        if selected_incident_candidate:
            cand = selected_incident_candidate
            
            # --- A. 原因分析レポート ---
            if st.session_state.generated_report is None:
                st.info(f"インシデント選択中: **{cand['id']}** ({cand['label']})")
                
                if api_key and (scenario != "正常稼働" or cand.get('is_prediction')):
                    # ★ 予兆の場合はボタンラベルとシナリオコンテキストを変更
                    is_pred = cand.get('is_prediction')
                    btn_label = "🔮 予兆の確認手順を生成 (Predictive Analysis)" if is_pred else "📝 詳細レポートを作成 (Generate Report)"
                    
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
                            if (getattr(n, "parent_id", None) if hasattr(n, 'parent_id') else n.get('parent_id')) == cand["id"]
                        ]
                        topology_context = {"node": t_node_dict, "parent_id": parent_id, "children_ids": children_ids}
                        
                        cache_key_analyst = "|".join([
                            "analyst",
                            site_id,
                            scenario,
                            str(cand.get("id")),
                            _hash_text(json.dumps(topology_context, ensure_ascii=False, sort_keys=True)),
                        ])
                        
                        if cache_key_analyst in st.session_state.report_cache:
                            full_text = st.session_state.report_cache[cache_key_analyst]
                            report_container.markdown(full_text)
                        else:
                            try:
                                report_container.write("🤖 AI 分析中...")
                                placeholder = report_container.empty()
                                full_text = ""
                                
                                # ★ 予兆の場合、AI に予兆コンテキストを渡す
                                report_scenario = scenario
                                if is_pred:
                                    pred_reason = cand.get('reason', '')
                                    pred_timeline = cand.get('prediction_timeline', '不明')
                                    pred_affected = cand.get('prediction_affected_count', 0)
                                    signal_count = cand.get('prediction_signal_count', 1)
                                    pred_early_hours = cand.get('prediction_early_warning_hours', 0)
                                    pred_time_critical = cand.get('prediction_time_to_critical_min', 0)
                                    if pred_early_hours >= 24:
                                        early_ctx = f"{pred_early_hours // 24}日前から検知可能なパターン"
                                    elif pred_early_hours > 0:
                                        early_ctx = f"{pred_early_hours}時間前から検知可能なパターン"
                                    else:
                                        early_ctx = "早期検知パターン"
                                    report_scenario = (
                                        f"[予兆検知 - Predictive Maintenance] Digital Twinが{cand['id']}で障害の前兆を検出。\n"
                                        f"現在のステータスは「正常」だが、{signal_count}件の微弱シグナルを検出。\n"
                                        f"・早期予兆: {early_ctx}\n"
                                        f"・急性期進行: 発症後{pred_time_critical}分で深刻化の恐れ\n"
                                        f"・影響範囲: 配下{pred_affected}台に影響の恐れ\n\n"
                                        f"検出されたシグナル:\n{pred_reason}\n\n"
                                        f"以下の構成でレポートを作成してください:\n"
                                        f"1. 検出された予兆パターンの解説（何が起きているか）\n"
                                        f"2. 手動確認手順（show コマンド等、実機で確認すべき項目）\n"
                                        f"3. 予兆が障害に発展するか判定するための基準\n"
                                        f"4. 推奨される予防措置とメンテナンス計画\n"
                                        f"5. エスカレーション基準（どの段階で上位報告すべきか）"
                                    )
                                
                                for chunk in generate_analyst_report_streaming(
                                    scenario=report_scenario,
                                    target_node=t_node,
                                    topology_context=topology_context,
                                    target_conf=target_conf or "なし",
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
                with st.container(height=400, border=True):
                    st.markdown(st.session_state.generated_report)
                if st.button("🔄 レポート再作成"):
                    st.session_state.generated_report = None
                    st.rerun()
        
        # --- B. 自動修復 & チャット ---
        st.markdown("---")
        st.subheader("🤖 Remediation & Chat")
        
        if selected_incident_candidate and selected_incident_candidate["prob"] > 0.6:
            # ★ 予兆と確定インシデントで表示を分ける
            if selected_incident_candidate.get('is_prediction'):
                timeline = selected_incident_candidate.get('prediction_timeline', '不明')
                affected = selected_incident_candidate.get('prediction_affected_count', 0)
                early_hours = selected_incident_candidate.get('prediction_early_warning_hours', 0)
                if early_hours >= 24:
                    early_display = f"最大 <b>{early_hours // 24}日前</b> から検知可能"
                elif early_hours > 0:
                    early_display = f"最大 <b>{early_hours}時間前</b> から検知可能"
                else:
                    early_display = "不明"
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
            
            if st.session_state.remediation_plan is None:
                # ★ 予兆時はボタンラベルを変更
                is_pred_rem = selected_incident_candidate.get('is_prediction')
                fix_label = "🔮 予防措置プランを生成" if is_pred_rem else "✨ 修復プランを作成 (Generate Fix)"
                report_prereq = "「🔮 予兆の確認手順を生成」" if is_pred_rem else "「📝 詳細レポートを作成 (Generate Report)」"
                
                if st.button(fix_label):
                    if st.session_state.generated_report is None:
                        st.warning(f"先に{report_prereq}を実行してください。")
                    else:
                        remediation_container = st.empty()
                        t_node = topology.get(selected_incident_candidate["id"])
                        
                        # ★ 予兆時のシナリオコンテキスト
                        rem_scenario = scenario
                        if is_pred_rem:
                            pred_timeline = selected_incident_candidate.get('prediction_timeline', '不明')
                            pred_affected = selected_incident_candidate.get('prediction_affected_count', 0)
                            pred_early_hours = selected_incident_candidate.get('prediction_early_warning_hours', 0)
                            pred_time_critical = selected_incident_candidate.get('prediction_time_to_critical_min', 0)
                            if pred_early_hours >= 24:
                                early_ctx = f"最大{pred_early_hours // 24}日前から検知可能"
                            elif pred_early_hours > 0:
                                early_ctx = f"最大{pred_early_hours}時間前から検知可能"
                            else:
                                early_ctx = "早期検知パターン"
                            rem_scenario = (
                                f"[予兆対応 - Predictive Maintenance] {selected_incident_candidate['id']}で障害の前兆を検出。\n"
                                f"・早期予兆: {early_ctx}\n"
                                f"・急性期: 発症後{pred_time_critical}分に深刻化の恐れ（影響{pred_affected}台）\n\n"
                                f"「復旧」ではなく「予防措置」として、障害を未然に防ぐための手順を提示してください:\n"
                                f"1. 予防的確認コマンド（show系コマンドで現状把握）\n"
                                f"2. 予防措置の実施手順（部品交換、設定変更等）\n"
                                f"3. メンテナンスウィンドウでの作業計画\n"
                                f"4. 切り戻し手順（予防措置で問題が発生した場合）\n"
                                f"5. 正常性確認コマンド（対応完了後の確認項目）\n"
                                f"6. エスカレーション基準（対応が間に合わない場合の判断基準）"
                            )
                        
                        cache_key_remediation = "|".join([
                            "remediation",
                            site_id,
                            scenario,
                            str(selected_incident_candidate.get("id")),
                            _hash_text(st.session_state.generated_report or ""),
                        ])
                        
                        if cache_key_remediation in st.session_state.report_cache:
                            remediation_text = st.session_state.report_cache[cache_key_remediation]
                            remediation_container.markdown(remediation_text)
                        else:
                            try:
                                loading_msg = "🔮 予防措置プラン生成中..." if is_pred_rem else "🤖 復旧プラン生成中..."
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
                with st.container(height=400, border=True):
                    st.info("AI Generated Recovery Procedure（復旧手順）")
                    st.markdown(st.session_state.remediation_plan)
                
                col_exec1, col_exec2 = st.columns(2)
                
                with col_exec1:
                    if st.button("🚀 修復実行 (Execute)", type="primary"):
                        if not api_key:
                            st.error("API Key Required")
                        else:
                            with st.status("Autonomic Remediation in progress...", expanded=True) as status_widget:
                                target_node_obj = topology.get(selected_incident_candidate["id"])
                                device_info = target_node_obj.metadata if target_node_obj and hasattr(target_node_obj, 'metadata') else {}
                                
                                st.write("🔄 Executing remediation steps in parallel...")
                                
                                results = run_remediation_parallel_v2(
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
                                    result = results.get(step_name)
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
                                    
                                    # 復旧成功フラグを設定
                                    st.session_state.recovered_devices[selected_incident_candidate["id"]] = True
                                    st.session_state.recovered_scenario_map[selected_incident_candidate["id"]] = scenario
                                    
                                    if not st.session_state.balloons_shown:
                                        st.balloons()
                                        st.session_state.balloons_shown = True
                                    
                                    st.success("✅ System Recovered Successfully!")
                                else:
                                    st.write("⚠️ Some remediation steps failed. Please review.")
                                    status_widget.update(label="Process Finished - With Errors", state="error", expanded=True)
                
                with col_exec2:
                    if st.button("キャンセル"):
                        st.session_state.remediation_plan = None
                        st.session_state.verification_log = None
                        st.rerun()
                
                if st.session_state.get("verification_log"):
                    st.markdown("#### 🔎 Post-Fix Verification Logs")
                    st.code(st.session_state.verification_log, language="text")
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
        
        # --- C. Chat with AI Agent ---
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
                _model = _chat_ci.get("model", "") or "Unknown"
                st.caption(f"対象機器: {_chat_target_id}   Vendor: {_vendor}   OS: {_os}   Model: {_model}")
            
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
            
            # タブでレイアウト
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
次の CI 情報と Config 抜粋を必ず参照して、具体的に回答してください。一般論だけで終わらせないでください。

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
                                response = generate_content_with_retry(st.session_state.chat_session.model, ci_prompt, stream=False)
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


# =====================================================
# メイン
# =====================================================
def main():
    """メインエントリーポイント"""
    api_key = render_sidebar()
    
    st.title("🛡️ AIOps インシデント・コックピット")
    st.caption("複数拠点のネットワーク障害を統合管理・分析")
    
    active_site = st.session_state.get("active_site")
    
    if active_site:
        render_incident_cockpit(active_site, api_key)
    else:
        tab1, tab2 = st.tabs(["📊 拠点状態ボード", "🚨 トリアージ・コマンドセンター"])
        
        with tab1:
            render_site_status_board()
        
        with tab2:
            render_triage_center()


if __name__ == "__main__":
    main()
