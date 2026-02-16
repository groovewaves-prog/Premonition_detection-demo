import streamlit as st
import os
from registry import list_sites, get_display_name, load_topology, get_paths
from utils.const import SCENARIO_MAP
from utils.llm_helper import get_rate_limiter, GENAI_AVAILABLE

def render_sidebar():
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
                
                if selected != current:
                    st.session_state.site_scenarios[site_id] = selected
                    # キャッシュクリア
                    keys_to_remove = [k for k in list(st.session_state.report_cache.keys()) if site_id in k]
                    for k in keys_to_remove:
                        del st.session_state.report_cache[k]
                    if st.session_state.active_site == site_id:
                        st.session_state.generated_report = None
                        st.session_state.remediation_plan = None
                        st.session_state.messages = []
                        st.session_state.chat_session = None
                        st.session_state.live_result = None
        
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
        
        # --- 予兆シミュレーション (完全版) ---
        _render_weak_signal_injection()
        
        return _render_api_key_input()


def _render_weak_signal_injection():
    """
    予兆シミュレーションUI
    AIエンジンが検知可能なリアルなログメッセージを生成する
    """
    with st.expander("🔮 予兆シミュレーション", expanded=True):
        st.caption("正常稼働中の機器に微細なシグナルを注入し、AIによる予兆検知をデモします。")
        
        # デバイスリスト生成
        active = st.session_state.get("active_site")
        site_for_list = active if active else (list_sites()[0] if list_sites() else None)
        
        device_options = []
        if site_for_list:
            try:
                paths = get_paths(site_for_list)
                topo = load_topology(paths.topology_path)
                if topo:
                    # 配下数カウント
                    child_count = {}
                    for dev_id, info in topo.items():
                        if isinstance(info, dict):
                            pid = info.get('parent_id')
                        else:
                            pid = getattr(info, 'parent_id', None)
                        if pid:
                            child_count[pid] = child_count.get(pid, 0) + 1
                    
                    # リスト作成（配下持ち優先）
                    for dev_id, info in topo.items():
                        if child_count.get(dev_id, 0) > 0:
                            if isinstance(info, dict):
                                dtype = info.get('type', '')
                                layer = info.get('layer', 0)
                                rg = info.get('redundancy_group')
                            else:
                                dtype = getattr(info, 'type', '')
                                layer = getattr(info, 'layer', 0)
                                rg = getattr(info, 'redundancy_group', None)
                            
                            n_children = child_count.get(dev_id, 0)
                            tag = "⚠SPOF" if not rg else "HA"
                            device_options.append((dev_id, f"L{layer} {dev_id} ({dtype}) [{tag}, 配下{n_children}台]"))
                    
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
        
        # --- リアルなログメッセージ生成 (ここが重要) ---
        log_messages = []
        if degradation_level > 0:
            if "Optical" in scenario_type:
                dbm = -23.0 - (degradation_level * 0.4)
                # ルール "optical" (rx power) にヒットさせる
                log_messages.append(
                    f"%TRANSCEIVER-4-THRESHOLD_VIOLATION: Rx Power {dbm:.1f} dBm (Threshold -25.0 dBm). Signal degrading."
                )
                if degradation_level >= 2:
                    crc = degradation_level * 150
                    log_messages.append(
                        f"%LINK-3-ERROR: CRC errors increasing on Gi0/0/0 (Count: {crc}/min). Input queue drops detected."
                    )
                if degradation_level >= 4:
                    log_messages.append(
                        "%OSPF-4-ADJCHANGE: Neighbor keepalive delayed (3 consecutive misses). Stability warning."
                    )
            
            elif "Microburst" in scenario_type:
                drops = degradation_level * 200
                # ルール "microburst" にヒットさせる
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
                # ルール "route_instability" にヒットさせる
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
        
        # Session State に保存
        if log_messages:
            st.session_state["injected_weak_signal"] = {
                "device_id": target_device,
                "messages": log_messages,
                "message": log_messages[0],
                "level": degradation_level,
                "scenario": scenario_type,
            }
            st.info(f"💉 **{len(log_messages)}件のシグナル注入中** (Level {degradation_level}/5)")
            for i, msg in enumerate(log_messages, 1):
                disp_msg = f"{msg[:60]}..." if len(msg) > 60 else msg
                st.caption(f"{i}. `{disp_msg}`")
        else:
            st.session_state["injected_weak_signal"] = None


def _render_api_key_input():
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
