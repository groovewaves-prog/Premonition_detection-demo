# ui/sidebar.py  ―  Streamlit サイドバー UI（予兆シミュレーション・シナリオ設定）
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
                    
                    # ★ 予兆シミュレーションとの競合を防ぐため自動クリア
                    injected = st.session_state.get("injected_weak_signal")
                    if injected and selected != "正常稼働":
                        # 障害シナリオ選択時は予兆シミュレーションをクリア
                        st.session_state["injected_weak_signal"] = None
                        
                        # ★ 関連するセッションステートキーも完全クリア
                        dt_prev_key = f"dt_prev_sim_device_{site_id}"
                        if dt_prev_key in st.session_state:
                            del st.session_state[dt_prev_key]
                        
                        st.info(
                            f"🔄 障害シナリオ「{selected}」を選択したため、"
                            "予兆シミュレーションを自動的にクリアしました。"
                        )
                    
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
                    
                    # ★ 重要：セッションステート変更後は必ず rerun
                    st.rerun()
        
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
        # ★ 改善: シード固定で同一レベルは同一メッセージを生成
        #   → predict キャッシュが有効になり、スライダー操作が高速化
        #   → デバイス＋シナリオが変わればシードも変わるので多様性は維持
        # ============================================================
        import random as _rng
        _seed = hash(f"{target_device}_{scenario_type}_{degradation_level}")
        _rng_local = _rng.Random(_seed)
        
        log_messages = []
        if degradation_level > 0:
            if "Optical" in scenario_type:
                # ★ 利用可能な光インターフェース
                optical_interfaces = [
                    "Gi0/0/1", "Gi0/0/2", "Gi0/0/3", "Gi0/0/4",
                    "Te1/0/1", "Te1/0/2", "Te1/0/3", "Te1/0/4"
                ]
                
                # レベルに応じて影響を受けるインターフェース数を決定
                num_affected = min(degradation_level + (1 if degradation_level >= 5 else 0), len(optical_interfaces))
                
                # ★ シード固定ランダム: 同一条件で同一結果
                selected_interfaces = _rng_local.sample(optical_interfaces, num_affected)
                
                dbm = -23.0 - (degradation_level * 0.4)
                
                for i, intf in enumerate(selected_interfaces):
                    # レベルに応じて異なるメッセージパターン
                    if i == 0 or degradation_level >= 3:
                        _msg = (f"%TRANSCEIVER-4-THRESHOLD_VIOLATION: Rx Power {dbm:.1f} dBm on {intf} "
                               f"(optical signal degrading). transceiver rx power below threshold.")
                        log_messages.append(_msg)
                    
                    if (i == 1 or degradation_level >= 4) and len(log_messages) < degradation_level:
                        _msg = (f"%OPTICAL-3-SIGNAL_WARN: optical signal level degrading on {intf}. "
                               f"light level {dbm+1.5:.1f} dBm. transceiver rx power loss detected.")
                        log_messages.append(_msg)

            elif "Microburst" in scenario_type:
                data_interfaces = [
                    "Gi0/1/0", "Gi0/1/1", "Gi0/1/2", "Gi0/1/3",
                    "Gi0/1/4", "Gi0/1/5", "Gi0/1/6", "Gi0/1/7"
                ]
                
                num_affected = min(degradation_level + (0 if degradation_level < 5 else 1), len(data_interfaces))
                selected_interfaces = _rng_local.sample(data_interfaces, num_affected)
                
                drops = degradation_level * 200
                
                for i, intf in enumerate(selected_interfaces):
                    if i == 0 or degradation_level >= 3:
                        _msg = (f"%HARDWARE-3-ASIC_ERROR: asic_error queue drops detected on {intf} "
                               f"(Count: {drops}). output drops on burst traffic.")
                        log_messages.append(_msg)
                    
                    if (i == 1 or degradation_level >= 4) and len(log_messages) < degradation_level:
                        _msg = (f"%QOS-4-BUFFER: buffer overflow risk on {intf}. "
                               f"queue drops {drops+100}/sec. output drops increasing.")
                        log_messages.append(_msg)

            elif "Route" in scenario_type:
                bgp_peers = [
                    ("10.1.1.1", "AS65001"),
                    ("10.1.1.2", "AS65002"),
                    ("10.1.1.3", "AS65003"),
                    ("10.1.1.4", "AS65004"),
                    ("10.2.1.1", "AS65010"),
                ]
                
                num_affected = min(degradation_level, len(bgp_peers))
                selected_peers = _rng_local.sample(bgp_peers, num_affected)
                
                updates = degradation_level * 500
                
                for i, (peer_ip, peer_as) in enumerate(selected_peers):
                    if i == 0 or degradation_level >= 3:
                        _msg = (f"BGP-5-NEIGHBOR: bgp neighbor {peer_ip} ({peer_as}) route updates {updates}/min. "
                               f"route instability warning detected.")
                        log_messages.append(_msg)
                    
                    if (i == 1 or degradation_level >= 4) and len(log_messages) < degradation_level:
                        _msg = (f"%BGP-4-INSTABILITY: route instability detected on peer {peer_ip} ({peer_as}). "
                               f"retransmission rate increasing. neighbor down risk.")
                        log_messages.append(_msg)
        
        
        # Session State に保存
        if log_messages:
            # ★ 障害シナリオとの競合チェック
            active_site = st.session_state.get("active_site")
            if active_site:
                current_scenario = st.session_state.site_scenarios.get(active_site, "正常稼働")
                if current_scenario != "正常稼働":
                    st.error(
                        f"⛔ **競合エラー**\n\n"
                        f"現在、拠点 `{active_site}` では障害シナリオ「**{current_scenario}**」が実行中です。\n"
                        "予兆シミュレーションは **正常稼働時** にのみ有効です。\n\n"
                        "💡 対処方法:\n"
                        "1. 拠点シナリオ設定を「**正常稼働**」に戻す\n"
                        "2. 予兆シミュレーションを実行\n"
                        "3. （オプション）その後、障害シナリオに切り替えて予兆の的中を確認"
                    )
                    st.session_state["injected_weak_signal"] = None
                    return  # 早期終了
            
            # ★ 改善: 古い予兆をクリア（重複防止）
            # 同じデバイスの古いシミュレーション予兆を削除
            _prev_injected = st.session_state.get("injected_weak_signal")
            if _prev_injected and _prev_injected.get("device_id") == target_device:
                # 同じデバイスで連続実行 → forecast_ledger をクリア
                dt_key = f"dt_engine_{active_site}"
                if dt_key in st.session_state:
                    dt_engine = st.session_state[dt_key]
                    # simulation sourceの予兆を削除
                    try:
                        if dt_engine and dt_engine.storage._conn:
                            with dt_engine.storage._db_lock:
                                dt_engine.storage._conn.execute("""
                                    DELETE FROM forecast_ledger
                                    WHERE device_id=? AND status='open' AND source='simulation'
                                """, (target_device,))
                                dt_engine.storage._conn.commit()
                    except Exception as e:
                        pass  # エラーは無視
            
            st.session_state["injected_weak_signal"] = {
                "device_id": target_device,
                "messages": log_messages,
                "message": log_messages[0],
                "level": degradation_level,
                "scenario": scenario_type,
            }
            st.info(f"💉 **{len(log_messages)}件のシグナル注入中** (Level {degradation_level}/5)")
            for i, msg in enumerate(log_messages, 1):
                disp_msg = f"{msg[:80]}..." if len(msg) > 80 else msg
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


# ─────────────────────────────────────────────────────────
# LLM 自動振り分け（UIなし — 設計ドキュメント）
# ─────────────────────────────────────────────────────────
# 1つの GOOGLE_API_KEY で2モデルを自動振り分け:
#   スコアリング・判定       → gemma-3-12b-it      (llm_client.py)
#   レポート・推奨アクション → gemini-2.0-flash-exp (engine.py / cockpit.py)
