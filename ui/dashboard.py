# ui/dashboard.py
import streamlit as st
from dataclasses import dataclass
from typing import List
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
    is_maintenance: bool

def build_site_statuses() -> List[SiteStatus]:
    sites = list_sites()
    statuses = []
    for site_id in sites:
        scenario = st.session_state.site_scenarios.get(site_id, "正常稼働")
        paths = get_paths(site_id)
        topo = load_topology(paths.topology_path)
        alarms = generate_alarms_for_scenario(topo, scenario)
        summary = get_alarm_summary(alarms)
        status = get_status_from_alarms(scenario, alarms)
        statuses.append(SiteStatus(
            site_id, get_display_name(site_id), scenario, status,
            summary['total'], summary['critical'],
            st.session_state.maint_flags.get(site_id, False)
        ))
    return sorted(statuses, key=lambda s: ({"停止":0, "要対応":1}.get(s.status, 9), -s.alarm_count))

def render_site_status_board():
    st.subheader("🏢 拠点状態ボード")
    statuses = build_site_statuses()
    
    # Summary Metrics
    cols = st.columns(4)
    cols[0].metric("🔴 障害", f"{sum(1 for s in statuses if s.status=='停止')}拠点")
    cols[1].metric("🟠 要対応", f"{sum(1 for s in statuses if s.status=='要対応')}拠点")
    
    st.divider()
    if not statuses: return
    
    for i in range(0, len(statuses), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i+j < len(statuses):
                _render_site_card(col, statuses[i+j])

def _render_site_card(col, site: SiteStatus):
    with col:
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"### {get_status_icon(site.status)} {site.display_name}")
            if c2.button("詳細", key=f"btn_{site.site_id}"):
                st.session_state.active_site = site.site_id
                st.rerun()
            st.caption(f"📋 {site.scenario}")
            st.metric("アラーム", f"{site.alarm_count}件", delta=f"{site.critical_count} Critical", delta_color="inverse")

def render_triage_center():
    st.subheader("🚨 トリアージ・コマンドセンター")
    statuses = build_site_statuses()
    # (Simplified triage view for brevity)
    for site in statuses:
        if site.status in ["停止", "要対応"]:
            st.error(f"**{site.display_name}**: {site.status} (Alarm: {site.alarm_count})")
            if st.button(f"対応開始 ({site.site_id})", key=f"triage_{site.site_id}"):
                st.session_state.active_site = site.site_id
                st.rerun()
