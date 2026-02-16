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
    """以前の拠点状態ボードUXを復元"""
    st.subheader("🏢 拠点状態ボード")
    statuses = build_site_statuses()
    
    cols = st.columns(4)
    cols[0].metric("🔴 障害発生", f"{sum(1 for s in statuses if s.status == '停止')}拠点")
    cols[1].metric("🟠 要対応", f"{sum(1 for s in statuses if s.status == '要対応')}拠点")
    cols[2].metric("🟡 注意", f"{sum(1 for s in statuses if s.status == '注意')}拠点")
    cols[3].metric("🟢 正常", f"{sum(1 for s in statuses if s.status == '正常')}拠点")
    
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
                        st.rerun()
                    st.caption(f"📋 {site.scenario.split('. ', 1)[-1]}")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("ステータス", site.status)
                    m2.metric("アラーム", f"{site.alarm_count}件")
                    m3.metric("MTTR", site.mttr_estimate)

def render_triage_center():
    """image_8a2426.png のトリアージUXを完全復元"""
    st.subheader("🚨 トリアージ・コマンドセンター")
    statuses = build_site_statuses()
    
    # 停止または要対応の拠点のみを抽出
    alert_sites = [s for s in statuses if s.status in ["停止", "要対応"]]
    
    if not alert_sites:
        st.info("現在、トリアージが必要な緊急インシデントはありません。")
        return

    for site in alert_sites:
        # image_8a2426.png の赤いバナー表示を再現
        st.error(f"{site.display_name}: {site.status} (Alarm: {site.alarm_count})")
        
        # 「対応開始」ボタンをバナーの直下に配置
        if st.button(f"対応開始 ({site.display_name[0]})", key=f"triage_btn_{site.site_id}"):
            st.session_state.active_site = site.site_id
            st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
