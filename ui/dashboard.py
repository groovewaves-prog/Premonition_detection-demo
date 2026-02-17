import streamlit as st
import pandas as pd
from typing import List
from dataclasses import dataclass

from registry import list_sites, get_paths, load_topology, get_display_name
from alarm_generator import generate_alarms_for_scenario, get_alarm_summary
from utils.helpers import get_status_from_alarms, get_status_icon


@dataclass
class SiteStatus:
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

        mttr = f"{30 + summary['total'] * 5}分" if status in ["停止", "要対応"] else "-"

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


def render_site_status_board():
    """拠点状態ボード"""
    st.subheader("🏢 拠点状態ボード")
    statuses = build_site_statuses()

    cols = st.columns(4)
    cols[0].metric("🔴 障害発生", f"{sum(1 for s in statuses if s.status == '停止')}拠点")
    cols[1].metric("🟠 要対応",   f"{sum(1 for s in statuses if s.status == '要対応')}拠点")
    cols[2].metric("🟡 注意",     f"{sum(1 for s in statuses if s.status == '注意')}拠点")
    cols[3].metric("🟢 正常",     f"{sum(1 for s in statuses if s.status == '正常')}拠点")

    st.divider()

    cols_per_row = 2
    for i in range(0, len(statuses), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for j, col in enumerate(row_cols):
            if i + j < len(statuses):
                site = statuses[i + j]
                with col.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    c1.markdown(f"### {get_status_icon(site.status)} {site.display_name}")
                    if c2.button("詳細", key=f"board_det_{site.site_id}", type="primary"):
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

                    m1, m2, m3 = st.columns(3)
                    m1.metric("ステータス", site.status)
                    m2.metric("アラーム", f"{site.alarm_count}件")
                    m3.metric("MTTR", site.mttr_estimate)

                    if site.alarm_count > 0:
                        severity = min(100, site.critical_count * 30 + site.warning_count * 10)
                        st.progress(severity / 100, text=f"深刻度: {severity}%")

                    if site.affected_devices:
                        st.caption(f"影響機器: {', '.join(site.affected_devices[:3])}")


def render_triage_center():
    """
    トリアージ・コマンドセンター（旧UIを完全復元）
    ─ フィルタ（ステータス multiselect + メンテナンス中チェック）
    ─ 各拠点を border付きコンテナ + 5カラムレイアウトで表示
      [アイコン | 拠点名/シナリオ | CRITICAL/WARNING件数 | MTTR | 詳細を確認ボタン]
    """
    st.subheader("🚨 トリアージ・コマンドセンター")

    statuses = build_site_statuses()

    # ── フィルタ行（旧UIと同じ2カラム構成）──
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

    # ── 各拠点カード（旧UIと同じ5カラムレイアウト）──
    for site in filtered:
        with st.container(border=True):
            cols = st.columns([0.5, 2, 1.5, 1, 1.5])

            # col[0]: ステータスアイコン（大）
            with cols[0]:
                st.markdown(f"## {get_status_icon(site.status)}")

            # col[1]: 拠点名 + シナリオ
            with cols[1]:
                st.markdown(f"**{site.display_name}**")
                scenario_short = site.scenario.split(". ", 1)[-1][:30]
                st.caption(scenario_short)

            # col[2]: CRITICAL / WARNING 件数
            with cols[2]:
                if site.critical_count > 0:
                    st.error(f"🔴 {site.critical_count} CRITICAL")
                if site.warning_count > 0:
                    st.warning(f"🟡 {site.warning_count} WARNING")

            # col[3]: MTTR（ラベル非表示）
            with cols[3]:
                st.metric("MTTR", site.mttr_estimate, label_visibility="collapsed")

            # col[4]: 詳細を確認ボタン
            with cols[4]:
                btn_type = "primary" if site.status in ["停止", "要対応"] else "secondary"
                if st.button("📋 詳細を確認", key=f"triage_detail_{site.site_id}", type=btn_type):
                    st.session_state.active_site = site.site_id
                    # セッション状態をリセット
                    st.session_state.live_result = None
                    st.session_state.verification_result = None
                    st.session_state.generated_report = None
                    st.session_state.remediation_plan = None
                    st.session_state.messages = []
                    st.session_state.chat_session = None
                    st.rerun()
