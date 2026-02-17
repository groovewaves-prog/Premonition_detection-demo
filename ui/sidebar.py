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
        # ============================================================
        # Level = 注入シグナル件数（Level1→1件 … Level5→5件）
        # 各メッセージは EscalationRule.semantic_phrases に確実ヒットする文言を使用
        #
        # Optical Decay ルール "optical":
        #   phrases: ["rx power", "optical signal", "transceiver", "light level", "dbm"]
        # Microburst ルール "microburst":
        #   phrases: ["queue drops", "buffer overflow", "output drops", "asic_error", "qos-4-policer"]
        # Route Instability ルール "route_instability":
        #   phrases: ["route instability", "bgp neighbor", "neighbor down", "route updates", "retransmission"]
        # ============================================================
        log_messages = []
        if degradation_level > 0:
            if "Optical" in scenario_type:
                dbm = -23.0 - (degradation_level * 0.4)
                # Level1: optical/rx power/dbm ヒット
                _l1 = (f"%TRANSCEIVER-4-THRESHOLD_VIOLATION: Rx Power {dbm:.1f} dBm "
                       f"(optical signal degrading). transceiver rx power below threshold.")
                # Level2: optical signal / light level ヒット
                _l2 = (f"%OPTICAL-3-SIGNAL_WARN: optical signal level degrading. "
                       f"light level {dbm+1.5:.1f} dBm. transceiver rx power loss detected.")
                # Level3: output drops / queue drops ヒット (複合劣化)
                crc = degradation_level * 150
                _l3 = (f"%LINK-3-ERROR: output drops increasing on Gi0/0/0 "
                       f"(Count: {crc}/min). queue drops detected. signal integrity degraded.")
                # Level4: buffer overflow ヒット
                _l4 = (f"%HARDWARE-4-BUFFER: buffer overflow risk on optical interface. "
                       f"asic_error queue drops {degradation_level*80}. rx power unstable.")
                # Level5: retransmission / route updates ヒット (L1への波及)
                _l5 = (f"%OSPF-4-ADJCHANGE: retransmission increase detected. "
                       f"route updates {degradation_level*200}/min. "
                       f"optical signal instability causing neighbor keepalive delay.")
                _pool = [_l1, _l2, _l3, _l4, _l5]
                log_messages = _pool[:degradation_level]

            elif "Microburst" in scenario_type:
                drops = degradation_level * 200
                # Level1: queue drops / asic_error ヒット
                _l1 = (f"%HARDWARE-3-ASIC_ERROR: asic_error queue drops detected "
                       f"(Count: {drops}). output drops on burst traffic.")
                # Level2: buffer overflow ヒット
                _l2 = (f"%QOS-4-BUFFER: buffer overflow risk on ge-0/0/1. "
                       f"queue drops {drops+100}/sec. output drops increasing.")
                # Level3: qos-4-policer ヒット
                _l3 = (f"%QOS-4-POLICER: qos-4-policer traffic exceeding CIR. "
                       f"output drops {degradation_level*80}/min. buffer overflow imminent.")
                # Level4: output drops + asic_error 複合
                _l4 = (f"%HARDWARE-4-ASIC: asic_error escalation. output drops {drops*2}/min. "
                       f"queue drops buffer overflow threshold reached.")
                # Level5: 全症状集約
                retrans = degradation_level * 50
                _l5 = (f"%TCP-5-RETRANSMIT: retransmission {retrans}/sec. "
                       f"queue drops buffer overflow. asic_error output drops critical level.")
                _pool = [_l1, _l2, _l3, _l4, _l5]
                log_messages = _pool[:degradation_level]

            elif "Route" in scenario_type:
                updates = degradation_level * 500
                # Level1: route updates / bgp neighbor ヒット
                _l1 = (f"BGP-5-NEIGHBOR: bgp neighbor route updates {updates}/min. "
                       f"route instability warning detected.")
                # Level2: route instability / retransmission ヒット
                _l2 = (f"%BGP-4-INSTABILITY: route instability detected. "
                       f"retransmission rate increasing. neighbor down risk.")
                # Level3: neighbor down ヒット
                _l3 = (f"%BGP-4-ADJCHANGE: bgp neighbor down event. "
                       f"route updates {updates+500}/min. route instability escalating.")
                # Level4: 複合シグナル
                _l4 = (f"%ROUTING-3-CONVERGENCE: route instability causing convergence delay. "
                       f"retransmission {degradation_level*30}/sec. neighbor down on 2 peers.")
                # Level5: 全症状集約
                _l5 = (f"%BGP-5-WITHDRAW: route updates withdrawal detected. "
                       f"route instability critical. retransmission burst. "
                       f"neighbor down multiple peers. bgp neighbor flapping.")
                _pool = [_l1, _l2, _l3, _l4, _l5]
                log_messages = _pool[:degradation_level]
        
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
