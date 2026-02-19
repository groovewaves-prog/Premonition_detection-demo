# ui/cockpit.py  ―  AIOps インシデント・コックピット（Phase1 predict_api + RUL予測表示）
import streamlit as st
import pandas as pd
import json
import time
import hashlib
from typing import Optional, List, Dict, Any

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    try:
        import google.generativeai as genai  # 旧SDK互換フォールバック
        GENAI_AVAILABLE = True
    except ImportError:
        genai = None
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
    """
    チャット用CIコンテキストを構築。
    対象ノードのmetadata・config に加え、
    トポロジーJSONの親子関係・冗長グループ・隣接デバイス情報も含める。
    """
    node = topology.get(target_node_id)
    if node and hasattr(node, 'metadata'):
        md = node.metadata or {}
    elif isinstance(node, dict):
        md = node.get('metadata', {})
    else:
        md = {}

    def _get(obj, attr, default=None):
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)

    # ---- 基本CI情報 ----
    ci = {
        "device_id": target_node_id or "",
        "hostname":  _pick_first(md, ["hostname", "host", "name"],            default=(target_node_id or "")),
        "vendor":    _pick_first(md, ["vendor", "manufacturer", "maker", "brand"], default=""),
        "os":        _pick_first(md, ["os", "platform", "os_name"],           default=""),
        "model":     _pick_first(md, ["model", "hw_model", "product"],        default=""),
        "role":      _pick_first(md, ["role", "type", "device_role"],         default=""),
        "layer":     _pick_first(md, ["layer", "level", "network_layer"],     default=""),
        "site":      _pick_first(md, ["site", "dc", "location"],              default=""),
    }

    # ---- トポロジーJSONから親子・冗長構成を取得 ----
    if node and topology:
        parent_id       = _get(node, 'parent_id')
        redundancy_group = _get(node, 'redundancy_group')
        node_type       = _get(node, 'type', '')
        node_layer      = _get(node, 'layer', '')

        ci["node_type"]        = node_type
        ci["network_layer"]    = node_layer
        ci["redundancy_group"] = redundancy_group or "なし（SPOF）"

        # 親デバイス情報
        if parent_id and parent_id in topology:
            p_node = topology[parent_id]
            p_md = _get(p_node, 'metadata') or {}
            ci["parent_device"] = {
                "id":     parent_id,
                "type":   _get(p_node, 'type', ''),
                "vendor": _pick_first(p_md, ["vendor", "manufacturer"], default=""),
                "os":     _pick_first(p_md, ["os", "platform"], default=""),
            }
        else:
            ci["parent_device"] = None  # ルートデバイス

        # 子デバイス一覧（直接の配下）
        children = []
        for nid, n in topology.items():
            if _get(n, 'parent_id') == target_node_id:
                n_md = _get(n, 'metadata') or {}
                children.append({
                    "id":     nid,
                    "type":   _get(n, 'type', ''),
                    "vendor": _pick_first(n_md, ["vendor", "manufacturer"], default=""),
                    "os":     _pick_first(n_md, ["os", "platform"], default=""),
                })
        ci["children_devices"] = children
        ci["children_count"]   = len(children)

        # 冗長ペアデバイス（同じredundancy_groupに属する他のデバイス）
        if redundancy_group:
            peers = []
            for nid, n in topology.items():
                if nid == target_node_id:
                    continue
                if _get(n, 'redundancy_group') == redundancy_group:
                    n_md = _get(n, 'metadata') or {}
                    peers.append({
                        "id":     nid,
                        "type":   _get(n, 'type', ''),
                        "vendor": _pick_first(n_md, ["vendor", "manufacturer"], default=""),
                        "os":     _pick_first(n_md, ["os", "platform"], default=""),
                    })
            ci["redundancy_peers"] = peers
        else:
            ci["redundancy_peers"] = []  # SPOFであることを明示

        # 同一レイヤーのデバイス一覧（参考情報）
        same_layer = [nid for nid, n in topology.items()
                      if _get(n, 'layer') == node_layer and nid != target_node_id]
        ci["same_layer_devices"] = same_layer

    # ---- コンフィグファイル（configsフォルダ） ----
    try:
        conf = load_config_by_id(target_node_id) if target_node_id else ""
        if conf:
            ci["config_excerpt"] = conf[:1500]
    except Exception:
        pass

    return ci


def _sanitize_prediction_context(text: str, max_len: int = 800) -> str:
    """
    LLMプロンプト用サニタイズ:
    - 個人情報・パスワード・IP直書き・制御文字を除去
    - max_len で切り詰め（プロンプト肥大化防止 → 速度改善）
    """
    import re as _re
    # 制御文字除去（\x01含む全C0制御文字）
    text = _re.sub(r'[\x00-\x1f\x7f]', '', text or "")
    # パスワード・シークレット系を遮蔽
    text = _re.sub(r'(?i)(password|passwd|secret|token|api.?key)\s*[=:]\s*\S+', r'\1=[MASKED]', text)
    # プライベートIP は最後オクテットをマスク
    text = _re.sub(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.)\d{1,3}', r'\1***', text)
    return text[:max_len]


def _build_prediction_report_scenario(cand: dict, signal_count: int = 1) -> str:
    """
    予兆用レポートシナリオを構築（RUL予測ベース）
    運用者視点: 「今後N日後に障害発生」表現で統一
    """
    dev_id        = cand.get('id', '不明')
    pred_state    = cand.get('predicted_state') or cand.get('label', '').replace('🔮 [予兆] ', '') or '不明'
    pred_affected = int(cand.get('prediction_affected_count', 0))
    ttf_hours     = int(cand.get('prediction_time_to_failure_hours', 0))
    failure_dt    = cand.get('prediction_failure_datetime', '')
    ttc_min       = int(cand.get('prediction_time_to_critical_min', 60))
    confidence    = float(cand.get('confidence', cand.get('prob', 0.5)))
    rule_pattern  = cand.get('rule_pattern', '')
    reasons       = cand.get('reasons', [])

    # RUL表示: 今後N日後に障害発生
    if ttf_hours >= 24:
        ttf_display = f"今後{ttf_hours // 24}日後に障害発生の見込み"
        if failure_dt:
            ttf_display += f"（{failure_dt}頃）"
    elif ttf_hours > 0:
        ttf_display = f"今後{ttf_hours}時間後に障害発生の見込み"
        if failure_dt:
            ttf_display += f"（{failure_dt}頃）"
    else:
        ttf_display = "障害が切迫しています"

    reason_summary = "; ".join(
        _sanitize_prediction_context(r, 120) for r in reasons[:3]
    ) if reasons else rule_pattern

    lines = [
        f"[RUL予測] {dev_id}で障害の前兆を検出（信頼度{confidence*100:.0f}%）。{signal_count}件の微弱シグナルを確認。",
        f"・予測障害: {pred_state}",
        f"・障害発生予測: {ttf_display}",
        f"・急性期進行: 症状発症後{ttc_min}分でサービス断に至る恐れ",
        f"・影響範囲: 配下{pred_affected}台に通信断リスク",
        f"・検出シグナル: {reason_summary}",
        "以下を簡潔に提供してください（各項目3行以内）:",
        "1.予兆パターン解説 2.確認コマンド 3.判定基準 4.予防措置 5.エスカレーション",
    ]
    return "\n".join(lines)


def _build_prevention_plan_scenario(cand: dict) -> str:
    """予防措置プラン用シナリオ（RUL予測ベース）"""
    dev_id        = cand.get('id', '不明')
    pred_state    = cand.get('predicted_state') or cand.get('label', '').replace('🔮 [予兆] ', '') or '不明'
    pred_affected = int(cand.get('prediction_affected_count', 0))
    ttc_min       = int(cand.get('prediction_time_to_critical_min', 60))
    ttf_hours     = int(cand.get('prediction_time_to_failure_hours', 0))
    failure_dt    = cand.get('prediction_failure_datetime', '')
    rec_actions   = cand.get('recommended_actions', [])

    # RUL表示
    if ttf_hours >= 24:
        ttf_ctx = f"今後{ttf_hours // 24}日後に障害発生"
        if failure_dt:
            ttf_ctx += f"（{failure_dt}頃）"
    else:
        ttf_ctx = f"今後{ttf_hours}時間後に障害発生" if ttf_hours > 0 else "障害が切迫"

    actions_txt = ""
    if rec_actions:
        actions_txt = " 既知の推奨: " + ", ".join(
            _sanitize_prediction_context(a.get('title',''), 60) for a in rec_actions[:3])

    lines = [
        f"[予防措置] {dev_id}の障害予兆に対する予防措置プラン。",
        f"・予測障害: {pred_state}",
        f"・障害発生予測: {ttf_ctx}",
        f"・急性期進行: 症状発症後{ttc_min}分でサービス断",
        f"・影響範囲: 配下{pred_affected}台{actions_txt}",
        "「復旧」ではなく「予防措置・事前対応」として簡潔に提示（各手順2行以内）:",
        "1.即時点検 2.予防コマンド 3.メンテナンス計画 4.監視強化 5.エスカレーション判断基準",
    ]
    return "\n".join(lines)
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
            #st.rerun()  # Disabled to prevent white screen

    # トポロジー読み込み
    paths = get_paths(site_id)
    topology = load_topology(paths.topology_path)

    if not topology:
        st.error("トポロジーが読み込めませんでした。")
        return

    # アラーム生成
    alarms = generate_alarms_for_scenario(topology, scenario)
    status = get_status_from_alarms(scenario, alarms)
    
    # ★ 将来の拡張: 障害発生時に予兆を自動確認（コメントアウト）
    # if dt_engine and scenario != "正常稼働":
    #     # CRITICAL アラームが発生したデバイスの予兆を自動的に confirmed_incident に更新
    #     critical_devices = {a.device_id for a in alarms if a.severity == "CRITICAL"}
    #     for dev_id in critical_devices:
    #         confirmed_count = dt_engine.forecast_auto_confirm_on_incident(
    #             dev_id, scenario=scenario, note="障害シナリオ発生により自動確認"
    #         )
    #         if confirmed_count > 0:
    #             logger.info(f"Auto-confirmed {confirmed_count} predictions for {dev_id} on scenario: {scenario}")

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
    # ★ Phase1: DigitalTwinEngine.predict_api() 接続
    # シミュレーション注入 OR 正常シナリオで dt_engine を呼ぶ
    # =====================================================
    dt_key     = f"dt_engine_{site_id}"
    dt_err_key = f"dt_engine_error_{site_id}"
    dt_engine  = st.session_state.get(dt_key)

    # ★ 壊れたエンジン(ロック取得不可)の検出とリセット
    if dt_engine is not None:
        try:
            # ノンブロッキングで_db_lockの取得を試みる
            # 取得できない場合はエンジンが壊れている(デッドロック状態)
            _lock_acquired = dt_engine.storage._db_lock.acquire(blocking=False)
            if _lock_acquired:
                dt_engine.storage._db_lock.release()
            else:
                # ロック取得失敗 → エンジンがデッドロック状態 → リセット
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    f"[dt_engine] {site_id}: DB lock deadlock detected. Resetting engine.")
                st.session_state[dt_key] = None
                st.session_state[dt_err_key] = None
                # dt_auto_confirmed / dt_auto_mitigated もクリアして再確認できるようにする
                for _k in [k for k in list(st.session_state.keys())
                           if k.startswith(f"dt_auto_") and site_id in k]:
                    del st.session_state[_k]
                dt_engine = None
        except Exception:
            pass

    if dt_engine is None and not st.session_state.get(dt_err_key):
        try:
            from digital_twin_pkg import DigitalTwinEngine as _DTE
            _children_map: dict = {}
            for _nid, _n in topology.items():
                _pid = (_n.get('parent_id') if isinstance(_n, dict)
                        else getattr(_n, 'parent_id', None))
                if _pid:
                    _children_map.setdefault(_pid, []).append(_nid)
            dt_engine = _DTE(topology=topology, children_map=_children_map, tenant_id=site_id)
            st.session_state[dt_key]     = dt_engine
            st.session_state[dt_err_key] = None
        except Exception as _dte_err:
            import traceback as _tb
            st.session_state[dt_err_key] = f"{type(_dte_err).__name__}: {_dte_err}\n{_tb.format_exc()}"
            dt_engine = None

    # 期限切れ予兆を定期的に解消（rate limit: 5分に1回）
    _expire_key = f"dt_expire_ts_{site_id}"
    if dt_engine and (time.time() - st.session_state.get(_expire_key, 0)) > 300:
        dt_engine.forecast_expire_open()
        st.session_state[_expire_key] = time.time()

    # =====================================================
    # ★ 競合検出: 障害シナリオと予兆シミュレーションの排他制御
    # ─────────────────────────────────────────────────────
    # 「予兆シミュレーション」の本来の意味:
    #   正常稼働中に弱いシグナルを注入 → DTが予兆を検知
    # 障害シナリオ active 時は意味論的に時系列逆転するため排他制御
    # =====================================================
    _injected        = st.session_state.get("injected_weak_signal")
    _scenario_active = (scenario != "正常稼働")
    _sim_active      = bool(_injected and _injected.get("device_id") in topology)

    # 競合状態: 障害シナリオ中に予兆シミュレーションが注入されている
    _conflict = _scenario_active and _sim_active

    if _conflict:
        # 競合デバイスが実障害と重なるか確認
        _sim_device     = _injected.get("device_id", "")
        _critical_set   = {a.device_id for a in alarms if a.severity == "CRITICAL"}
        _warning_set    = {a.device_id for a in alarms if a.severity == "WARNING"}
        _conflict_level = ("CRITICAL" if _sim_device in _critical_set
                           else "WARNING" if _sim_device in _warning_set
                           else "OTHER")

        # 競合警告をUIに表示
        if _conflict_level in ("CRITICAL", "WARNING"):
            st.warning(
                "⚠️ **予兆シミュレーション競合検出**\n\n"
                f"現在の障害シナリオ「**{scenario}**」により `{_sim_device}` は既に "
                f"**{_conflict_level}** 状態です。\n"
                "予兆シミュレーションは **無効化** されています。\n\n"
                "💡 予兆→障害の流れをデモするには:\n"
                "1. シナリオを「正常稼働」に戻す\n"
                "2. 予兆シミュレーションを実行（アンバー色でハイライト）\n"
                "3. 障害シナリオに切り替えて「予兆が的中した」を確認"
            )
        else:
            # 異なるデバイスへの注入は許容するが注意喚起
            st.info(
                f"ℹ️ 障害シナリオ「**{scenario}**」実行中です。\n"
                f"`{_sim_device}` への予兆シミュレーションは継続しますが、"
                "forecast_ledger の自動 CONFIRMED 登録は **抑制** されます。"
            )

    # 注入シグナル OR 実アラームを dt_engine に通して予兆リストを生成
    dt_predictions: List[dict] = []
    if dt_engine:
        _msg_sources = []

        # A) 予兆シミュレーション注入シグナル
        #    競合かつ同デバイスが障害中の場合は無効化
        if _sim_active:
            _sim_dev  = _injected.get("device_id", "")
            # _critical_set/_warning_set は _conflict=True 時のみ定義済み
            _alarm_devices = {a.device_id for a in alarms
                              if a.severity in ("CRITICAL", "WARNING")}
            _disabled = (_conflict and _sim_dev in _alarm_devices)
            if not _disabled:
                _msgs = _injected.get("messages", [_injected.get("message", "")])
                for _m in _msgs:
                    if _m:
                        _msg_sources.append((_sim_dev, _m, "simulation"))

        # degradation_level を sidebar から取得 + デバイス変更時のレポートリセット
        _sim_level = int((_injected or {}).get("level", 1)) if _sim_active else 1
        _prev_sim_dev_key = f"dt_prev_sim_device_{site_id}"
        _cur_sim_dev = (_injected or {}).get("device_id", "")
        if _cur_sim_dev != st.session_state.get(_prev_sim_dev_key, ""):
            for _k in [k for k in list(st.session_state.report_cache.keys())
                       if "analyst" in k and site_id in k]:
                del st.session_state.report_cache[_k]
            st.session_state.generated_report   = None
            st.session_state.remediation_plan   = None
            st.session_state.verification_log   = None
            st.session_state[_prev_sim_dev_key] = _cur_sim_dev

        # B) 実アラームの WARNING/INFO（障害確定前の弱いシグナル）
        for _a in alarms:
            if _a.severity in ("WARNING", "INFO") and not _a.is_root_cause:
                _msg_sources.append((_a.device_id, _a.message, "real"))

        _signal_count = len(_msg_sources)

        # ★ デバイスIDごとにメッセージをグループ化してから predict_api を1回だけ呼ぶ
        # 従来: 1メッセージ × N回 → LLMに1件しか渡らず推奨アクションの精度が低かった
        # 修正: デバイス単位で全メッセージをまとめて1回 → LLMに全文脈を渡す
        #       forecast_ledger への重複登録も防ぐ（"optical 16件" 問題の解消）
        _dev_msg_map: dict = {}
        for _dev_id, _msg, _src in _msg_sources:
            if _dev_id not in _dev_msg_map:
                _dev_msg_map[_dev_id] = {"messages": [], "src": _src}
            _dev_msg_map[_dev_id]["messages"].append(_msg)
            if _src == "simulation":  # simulation が混じれば simulation 優先
                _dev_msg_map[_dev_id]["src"] = "simulation"

        for _dev_id, _dev_data in _dev_msg_map.items():
            _msgs = _dev_data["messages"]
            _src  = _dev_data["src"]
            _resp = dt_engine.predict_api({
                "tenant_id":       site_id,
                "device_id":       _dev_id,
                "msg":             _msgs[0],    # 後方互換: 先頭メッセージ
                "messages":        _msgs,        # ★ 全メッセージをLLMに渡す
                "timestamp":       time.time(),
                "record_forecast": True,
                "api_key":         api_key or "",
                "attrs":           {
                    "source":            _src,
                    "degradation_level": _sim_level if _src == "simulation" else 1,
                }
            })
            if _resp.get("ok"):
                for _p in _resp.get("predictions", []):
                    _p["id"]     = _dev_id
                    _p["source"] = _src
                    _p["prediction_signal_count"] = _signal_count

                    # デバイスごとに1エントリ（重複なし）
                    _existing_idx = next(
                        (i for i, d in enumerate(dt_predictions) if d.get("id") == _dev_id),
                        None
                    )
                    if _existing_idx is None:
                        dt_predictions.append(_p)
                    else:
                        # 既存あり: 推奨アクションが強化されている場合のみ上書き
                        _existing = dt_predictions[_existing_idx]
                        _new_acts = _p.get("recommended_actions", [])
                        _old_acts = _existing.get("recommended_actions", [])
                        _new_has_high = any(a.get("priority") == "high" for a in _new_acts)
                        _old_has_high = any(a.get("priority") == "high" for a in _old_acts)
                        if (_new_has_high and not _old_has_high) or (len(_new_acts) > len(_old_acts)):
                            dt_predictions[_existing_idx] = _p

        # ── 自動 outcome 登録 ──────────────────────────────
        # Execute 成功済みデバイス → MITIGATED（競合状態に関わらず有効）
        for _rid, _recovered in list(st.session_state.get("recovered_devices", {}).items()):
            if _recovered:
                _auto_key = f"dt_auto_mitigated_{site_id}_{_rid}"
                if not st.session_state.get(_auto_key):
                    dt_engine.forecast_auto_resolve(
                        _rid, "mitigated", note="Execute 成功による自動解消")
                    st.session_state[_auto_key] = True

        # CRITICAL アラーム確定 → CONFIRMED
        # ただし競合状態（障害シナリオ active）では抑制して誤自動登録を防ぐ
        if not _conflict:
            _critical_devices = {a.device_id for a in alarms if a.severity == "CRITICAL"}
            for _cd in _critical_devices:
                _auto_key = f"dt_auto_confirmed_{site_id}_{_cd}"
                if not st.session_state.get(_auto_key):
                    dt_engine.forecast_auto_resolve(
                        _cd, "confirmed_incident",
                        note="CRITICAL アラームによる自動確定")
                    st.session_state[_auto_key] = True

    # DT予兆を analysis_results にマージ（既存の is_prediction 結果と重複しない）
    existing_pred_ids = {r.get("id") for r in analysis_results if r.get("is_prediction")}
    for _dp in dt_predictions:
        if _dp.get("id") not in existing_pred_ids:
            analysis_results.append(_dp)

    # =====================================================
    # ★ デバイス変更検知: 予兆シミュ対象が変わったらレポートをリセット
    # =====================================================
    _sim_device_now = (_injected.get("device_id") if _injected else None)
    _sim_device_key = f"dt_last_sim_device_{site_id}"
    _sim_device_prev = st.session_state.get(_sim_device_key)
    if _sim_device_now != _sim_device_prev:
        st.session_state.generated_report   = None
        st.session_state.remediation_plan   = None
        st.session_state.verification_log   = None
        # レポートキャッシュも予兆系のエントリだけ削除
        _keys_to_del = [k for k in st.session_state.get("report_cache", {})
                        if "analyst" in k or "remediation" in k]
        for _k in _keys_to_del:
            st.session_state.report_cache.pop(_k, None)
        st.session_state[_sim_device_key] = _sim_device_now

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
                    prob_pct        = f"{pred_item.get('prob', 0)*100:.0f}%"
                    confidence      = pred_item.get('confidence', pred_item.get('prob', 0))
                    pred_timeline   = pred_item.get('prediction_timeline', '不明')
                    ttc_min         = pred_item.get('prediction_time_to_critical_min',
                                       pred_item.get('time_to_critical_min', 0))
                    pred_affected   = pred_item.get('prediction_affected_count', 0)
                    pred_label      = (pred_item.get('predicted_state')
                                       or pred_item.get('label', '').replace('🔮 [予兆] ', '')
                                       or '不明')
                    ttf_hours       = pred_item.get('prediction_time_to_failure_hours', 0)
                    failure_dt      = pred_item.get('prediction_failure_datetime', '')
                    rule_pattern    = pred_item.get('rule_pattern', '')
                    criticality     = pred_item.get('criticality', 'standard')
                    reasons         = pred_item.get('reasons', [])
                    rec_actions     = pred_item.get('recommended_actions', [])
                    source          = pred_item.get('source', 'real')

                    # ── ヘッダー: 機器名 + 予兆種別 ──────────────
                    _crit_badge = "🔴 CRITICAL" if criticality == "critical" else "🟠 STANDARD"
                    _src_badge  = "🔬 シミュ" if source == "simulation" else "📡 実測"
                    st.markdown(
                        f"<div style='background:#FFF8E1;border-left:4px solid #FFB300;"
                        f"padding:8px 12px;border-radius:4px;margin-bottom:8px;'>"
                        f"<b>📍 {pred_item['id']}</b>"
                        f"<span style='float:right;font-size:11px;color:#BF360C;'>"
                        f"{_crit_badge} {_src_badge}</span></div>",
                        unsafe_allow_html=True
                    )

                    # ── 確信度 + タイムライン ─────────────────────
                    st.markdown(
                        f"<div style='text-align:center;padding:8px 0;'>"
                        f"<span style='font-size:40px;font-weight:bold;color:#E65100;'>"
                        f"{prob_pct}</span>"
                        f"<br><span style='color:#666;font-size:13px;'>"
                        f"障害発生確信度</span></div>",
                        unsafe_allow_html=True
                    )

                    # ── RUL予測詳細カード ─────────────────────────────
                    # 障害発生予測時刻の表示
                    if ttf_hours >= 24:
                        ttf_display = f"今後 <b>{ttf_hours // 24}日後</b>"
                        if failure_dt:
                            ttf_display += f" ({failure_dt}頃)"
                    elif ttf_hours > 0:
                        ttf_display = f"今後 <b>{ttf_hours}時間後</b>"
                        if failure_dt:
                            ttf_display += f" ({failure_dt}頃)"
                    else:
                        ttf_display = "<span style='color:#d32f2f'>障害が切迫</span>"
                    
                    st.markdown(
                        f"<div style='background:#FFF3E0;border-radius:6px;"
                        f"padding:10px 12px;margin:6px 0;font-size:13px;'>"
                        f"<b>🔮 予測障害:</b> {pred_label}<br>"
                        f"<b>📅 障害発生予測:</b> {ttf_display}<br>"
                        f"<b>⏱️ 急性期進行:</b> "
                        + (f"症状発症後 <span style='color:#d32f2f;font-weight:bold;'>{ttc_min}分</span> でサービス断"
                           if ttc_min > 0 else "<span style='color:#d32f2f'>不明</span>")
                        + (f"<br><b>📡 影響範囲:</b> 配下 <b>{pred_affected}台</b> 通信断リスク"
                           if pred_affected > 0 else "")
                        + f"</div>",
                        unsafe_allow_html=True
                    )

                    # ── 検知シグナル ───────────────────────────────
                    if reasons:
                        with st.expander("🔍 検知シグナル詳細", expanded=False):
                            for _r in reasons:
                                st.caption(f"• {_r}")
                            if rule_pattern:
                                st.caption(f"適用ルール: `{rule_pattern}`")

                    # ── 推奨アクション ─────────────────────────────
                    if rec_actions:
                        with st.expander("🛠️ 推奨アクション", expanded=True):
                            for idx, _act in enumerate(rec_actions, 1):
                                _title = _act.get('title', '')
                                _effect = _act.get('effect', '')
                                _priority = _act.get('priority', 'medium')
                                _rationale = _act.get('rationale', '')
                                _steps = _act.get('steps', '')
                                
                                # 優先度に応じた色とアイコン
                                if _priority == 'high':
                                    _bg_color = '#FFEBEE'  # 薄い赤
                                    _border_color = '#D32F2F'  # 濃い赤
                                    _icon = '🔴'
                                    _priority_label = '最優先'
                                elif _priority == 'medium':
                                    _bg_color = '#FFF3E0'  # 薄いオレンジ
                                    _border_color = '#FF6F00'  # 濃いオレンジ
                                    _icon = '🟠'
                                    _priority_label = '推奨'
                                else:  # low
                                    _bg_color = '#E8F5E9'  # 薄い緑
                                    _border_color = '#2E7D32'  # 濃い緑
                                    _icon = '🟢'
                                    _priority_label = '補助'
                                
                                st.markdown(
                                    f"<div style='background:{_bg_color};padding:10px 12px;"
                                    f"border-left:4px solid {_border_color};border-radius:4px;"
                                    f"margin:8px 0;font-size:13px;'>"
                                    f"<div style='margin-bottom:4px;'>"
                                    f"<b>{_icon} {idx}. {_title}</b>"
                                    f"<span style='float:right;background:{_border_color};color:white;"
                                    f"padding:2px 8px;border-radius:3px;font-size:11px;'>{_priority_label}</span>"
                                    f"</div>"
                                    + (f"<div style='color:#424242;margin:4px 0;font-size:12px;'>"
                                       f"💡 効果: {_effect}</div>" if _effect else "")
                                    + (f"<div style='color:#616161;margin:4px 0;font-size:11px;'>"
                                       f"📌 根拠: {_rationale}</div>" if _rationale else "")
                                    + (f"<div style='background:white;padding:6px;border-radius:3px;"
                                       f"margin-top:6px;font-size:11px;color:#424242;white-space:pre-wrap;'>"
                                       f"<b>📋 手順:</b><br>{_steps}</div>" if _steps else "")
                                    + "</div>",
                                    unsafe_allow_html=True
                                )
                    else:
                        st.caption("推奨アクションなし")
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
                #st.rerun()  # Disabled to prevent white screen

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
    # old_app.py の構造を完全復元
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
                            "id":       getattr(t_node, "id",       None) if t_node else None,
                            "type":     getattr(t_node, "type",     None) if t_node else None,
                            "layer":    getattr(t_node, "layer",    None) if t_node else None,
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

                        # 予兆の場合: バッチ化・サニタイズ済みヘルパーで構築（速度改善）
                        report_scenario = scenario
                        if is_pred:
                            _sig_count      = cand.get('prediction_signal_count', 1)
                            report_scenario = _build_prediction_report_scenario(cand, _sig_count)

                        cache_key_analyst = "|".join([
                            "analyst", site_id, scenario,
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

                                # ★ 正しいシグネチャ: topology_context= を使用
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
                # レポート表示（height=400スクロールコンテナ）
                with st.container(height=400, border=True):
                    st.markdown(st.session_state.generated_report)
                if st.button("🔄 レポート再作成"):
                    st.session_state.generated_report = None
                    #st.rerun()  # Disabled to prevent white screen

        # ============================================
        # B. Remediation & Chat
        #    ★ Generate Fix / Execute / Cancel はすべてここに配置
        # ============================================
        st.markdown("---")
        st.subheader("🤖 Remediation & Chat")

        if selected_incident_candidate and selected_incident_candidate["prob"] > 0.6:
            is_pred_rem = selected_incident_candidate.get('is_prediction')

            # ステータスバナー
            if is_pred_rem:
                timeline    = selected_incident_candidate.get('prediction_timeline', '不明')
                affected    = selected_incident_candidate.get('prediction_affected_count', 0)
                ttf_hours   = selected_incident_candidate.get('prediction_time_to_failure_hours', 0)
                failure_dt  = selected_incident_candidate.get('prediction_failure_datetime', '')
                
                # RUL表示
                if ttf_hours >= 24:
                    ttf_display = f"今後 <b>{ttf_hours // 24}日後</b>"
                    if failure_dt:
                        ttf_display += f" ({failure_dt}頃)"
                elif ttf_hours > 0:
                    ttf_display = f"今後 <b>{ttf_hours}時間後</b>"
                    if failure_dt:
                        ttf_display += f" ({failure_dt}頃)"
                else:
                    ttf_display = "<b>障害が切迫</b>"
                
                st.markdown(f"""
                <div style="background-color:#fff3e0;padding:10px;border-radius:5px;border:1px solid #ff9800;color:#e65100;margin-bottom:10px;">
                    <strong>🔮 Digital Twin 未来予測 (Predictive Maintenance)</strong><br>
                    <b>{selected_incident_candidate['id']}</b> で障害の兆候を検出しました。<br>
                    ・障害発生予測: {ttf_display}<br>
                    ・急性期進行: 症状発症後 <b>{timeline}</b> でサービス断の恐れ<br>
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

            # ★ Generate Fix ボタン（remediation_plan 未生成時のみ表示）
            if st.session_state.remediation_plan is None:
                fix_label    = "🔮 予防措置プランを生成 (Preventive Measures)" if is_pred_rem else "✨ 修復プランを作成 (Generate Fix)"
                report_prereq = "「🔮 予兆の確認手順を生成」" if is_pred_rem else "「📝 詳細レポートを作成 (Generate Report)」"

                if st.button(fix_label):
                    if st.session_state.generated_report is None:
                        st.warning(f"先に{report_prereq}を実行してください。")
                    else:
                        remediation_container = st.empty()
                        t_node = topology.get(selected_incident_candidate["id"])

                        rem_scenario = scenario
                        if is_pred_rem:
                            rem_scenario = _build_prevention_plan_scenario(selected_incident_candidate)

                        cache_key_rem = "|".join([
                            "remediation", site_id, scenario,
                            str(selected_incident_candidate.get("id")),
                            _hash_text(st.session_state.generated_report or ""),
                        ])

                        if cache_key_rem in st.session_state.report_cache:
                            remediation_text = st.session_state.report_cache[cache_key_rem]
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

                                st.session_state.report_cache[cache_key_rem] = remediation_text
                            except Exception as e:
                                remediation_text = f"⚠️ 復旧プラン生成に失敗しました: {type(e).__name__}: {e}"
                                remediation_container.markdown(remediation_text)

                        st.session_state.remediation_plan = remediation_text
                        #st.rerun()  # Disabled to prevent white screen

            # ★ 復旧手順表示 + Execute / Cancel ボタン（remediation_plan 生成済み時）
            if st.session_state.remediation_plan is not None:
                with st.container(height=400, border=True):
                    st.info("AI Generated Recovery Procedure（復旧手順）")
                    st.markdown(st.session_state.remediation_plan)

                # Execute / Cancel ボタン（復旧手順コンテナの直下）
                col_exec1, col_exec2 = st.columns(2)
                exec_clicked   = col_exec1.button("🚀 修復実行 (Execute)", type="primary")
                cancel_clicked = col_exec2.button("キャンセル")

                if cancel_clicked:
                    st.session_state.remediation_plan  = None
                    st.session_state.verification_log  = None
                    #st.rerun()  # Disabled to prevent white screen

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

            # ★ Phase1: 予兆ステータス履歴（グループ化・人間可読化）
            if dt_engine and selected_incident_candidate:
                _oc_device = selected_incident_candidate.get("id", "")
                _open_preds = dt_engine.forecast_list_open(device_id=_oc_device)
                if _open_preds:
                    st.markdown("---")
                    st.markdown("##### 📜 予兆ステータス履歴")
                    st.caption(
                        f"対象機器 `{_oc_device}` の未対応予兆: **{len(_open_preds)}件**  \n"
                        f"💡 類似予兆はグループ化されています。一括操作も可能です。"
                    )
                    
                    # ★ コンポーネント名抽出関数
                    import re
                    def _extract_component(message: str) -> str:
                        """ログメッセージからコンポーネント名を抽出"""
                        if not message:
                            return ""
                        
                        # インターフェース名（Gi, Te, ge など）
                        interface_patterns = [
                            (r'(Gi\d+/\d+/\d+)', lambda m: m.group(1)),
                            (r'(Te\d+/\d+/\d+)', lambda m: m.group(1)),
                            (r'(ge-\d+/\d+/\d+)', lambda m: m.group(1)),
                            (r'(Ethernet\d+/\d+/\d+)', lambda m: m.group(1)),
                        ]
                        for pattern, formatter in interface_patterns:
                            match = re.search(pattern, message)
                            if match:
                                return formatter(match)
                        
                        # BGP peer (IP address)
                        ip_match = re.search(r'(?:peer|neighbor)\s+(\d+\.\d+\.\d+\.\d+)', message, re.IGNORECASE)
                        if ip_match:
                            return f"Peer {ip_match.group(1)}"
                        
                        # AS番号
                        as_match = re.search(r'\(AS(\d+)\)', message)
                        if as_match:
                            return f"AS{as_match.group(1)}"
                        
                        return ""
                    
                    # ★ 予兆をルールパターンごとにグループ化
                    from collections import defaultdict
                    from datetime import datetime
                    
                    grouped_preds = defaultdict(list)
                    for _fp in _open_preds:
                        _rule = _fp.get("rule_pattern", "不明")
                        grouped_preds[_rule].append(_fp)
                    
                    # グループごとに表示
                    for _rule_pattern, _pred_group in grouped_preds.items():
                        _group_size = len(_pred_group)
                        
                        # グループ統計情報
                        _confidences = [float(p.get("confidence", 0.0)) for p in _pred_group]
                        _avg_conf = sum(_confidences) / len(_confidences) if _confidences else 0.0

                        # ★ 全レコードを横断して影響インターフェースとシグナル数を集計
                        _all_ifaces: set = set()
                        _total_signals: int = 0
                        for _fp2 in _pred_group:
                            _ifaces = _fp2.get("affected_interfaces", [])
                            _all_ifaces.update(_ifaces)
                            _sc = int(_fp2.get("signal_count", 0))
                            if _sc:
                                _total_signals = max(_total_signals, _sc)
                        # インターフェース名が無い場合（BGP peer等）はシグナル数で代替
                        _iface_count  = len(_all_ifaces) if _all_ifaces else _total_signals

                        # 最古と最新の検出時刻（Unix timestamp → 人間可読）
                        _timestamps = []
                        for p in _pred_group:
                            _created_raw = p.get("created_at", 0)
                            try:
                                _ts = float(_created_raw)
                                _timestamps.append(_ts)
                            except (ValueError, TypeError):
                                pass
                        
                        if _timestamps:
                            _oldest_ts = min(_timestamps)
                            _newest_ts = max(_timestamps)
                            _oldest_dt = datetime.fromtimestamp(_oldest_ts).strftime("%m/%d %H:%M")
                            _newest_dt = datetime.fromtimestamp(_newest_ts).strftime("%m/%d %H:%M")
                            
                            # 相対時間計算（最新の検出からの経過時間）
                            _elapsed_sec = time.time() - _newest_ts
                            if _elapsed_sec < 3600:
                                _relative = f"{int(_elapsed_sec / 60)}分前"
                            elif _elapsed_sec < 86400:
                                _relative = f"{int(_elapsed_sec / 3600)}時間前"
                            else:
                                _relative = f"{int(_elapsed_sec / 86400)}日前"
                        else:
                            _oldest_dt = "不明"
                            _newest_dt = "不明"
                            _relative  = ""

                        # ★ ヘッダー文字列: 影響インターフェース数が取れる場合はそれを表示
                        if _iface_count > 0:
                            _header_suffix = f"影響 {_iface_count}インターフェース | 信頼度: {_avg_conf*100:.0f}% | 最新: {_relative}"
                        else:
                            _header_suffix = f"信頼度: {_avg_conf*100:.0f}% | 最新: {_relative}"

                        # グループヘッダー（折りたたみ可能）
                        with st.expander(
                            f"🔖 {_rule_pattern}  ({_header_suffix})",
                            expanded=True
                        ):
                            # ★ 影響インターフェース一覧（実運用で有用な情報）
                            if _all_ifaces:
                                _sorted_ifaces = sorted(_all_ifaces)
                                _iface_chips = " ".join(
                                    f"<code style='background:#E3F2FD;padding:2px 6px;"
                                    f"border-radius:3px;font-size:11px;margin:1px;'>{_if}</code>"
                                    for _if in _sorted_ifaces
                                )
                                st.markdown(
                                    f"<div style='background:#F5F5F5;padding:6px 10px;"
                                    f"border-radius:4px;margin-bottom:8px;'>"
                                    f"<small>📡 影響インターフェース ({len(_sorted_ifaces)}個): "
                                    f"{_iface_chips}</small><br>"
                                    f"<small>📊 検出シグナル数: {_total_signals}件 | "
                                    f"📅 検出期間: {_oldest_dt} 〜 {_newest_dt}</small>"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(
                                    f"<div style='background:#F5F5F5;padding:6px 10px;border-radius:4px;margin-bottom:8px;'>"
                                    f"<small>📅 検出期間: {_oldest_dt} 〜 {_newest_dt}</small>"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
                            
                            # 一括操作ボタン
                            _bulk_col1, _bulk_col2, _bulk_col3 = st.columns([1, 1, 2])
                            with _bulk_col1:
                                if st.button(
                                    f"✅ 対応済みにする",
                                    key=f"bulk_handled_{_rule_pattern[:20]}",
                                    help=f"この予兆を対応済みにします",
                                    use_container_width=True
                                ):
                                    _success_count = 0
                                    for _fp in _pred_group:
                                        _fid = _fp.get("forecast_id", "")
                                        r = dt_engine.forecast_register_outcome(
                                            _fid, "mitigated",
                                            note=f"一括対応済み登録: {_rule_pattern} (device={_oc_device})"
                                        )
                                        if r.get("ok"):
                                            _success_count += 1
                                    st.success(f"✅ 対応済みとして登録しました")
                                    #st.rerun()  # Disabled to prevent white screen
                            
                            with _bulk_col2:
                                if st.button(
                                    f"❌ 誤検知にする",
                                    key=f"bulk_false_{_rule_pattern[:20]}",
                                    help=f"この予兆を誤検知にします",
                                    use_container_width=True
                                ):
                                    _success_count = 0
                                    for _fp in _pred_group:
                                        _fid = _fp.get("forecast_id", "")
                                        r = dt_engine.forecast_register_outcome(
                                            _fid, "false_alarm",
                                            note=f"一括誤検知登録: {_rule_pattern} (device={_oc_device})"
                                        )
                                        if r.get("ok"):
                                            _success_count += 1
                                    st.info(f"❌ 誤検知として登録しました")
                                    #st.rerun()  # Disabled to prevent white screen
                            
                            st.markdown("---")
                            
                            # 個別の予兆詳細
                            for idx, _fp in enumerate(_pred_group, 1):
                                _fid        = _fp.get("forecast_id", "")
                                _conf       = float(_fp.get("confidence", 0.0))
                                _created_raw= _fp.get("created_at", 0)
                                _ttf_hours  = _fp.get("time_to_failure_hours", 0)
                                _failure_dt = _fp.get("predicted_failure_datetime", "")
                                _source_msg = _fp.get("message", "")
                                _fp_ifaces  = _fp.get("affected_interfaces", [])
                                _fp_signals = int(_fp.get("signal_count", 0))

                                # ★ 表示IDの決定：affected_interfaces > messageパース > fallback
                                if _fp_ifaces:
                                    if len(_fp_ifaces) == 1:
                                        _display_id = _fp_ifaces[0]
                                    else:
                                        _display_id = f"{_fp_ifaces[0]} 他{len(_fp_ifaces)-1}個"
                                else:
                                    _component = _extract_component(_source_msg)
                                    _display_id = _component if _component else f"#{idx}"

                                # 検出時刻を人間可読化
                                try:
                                    _created_ts = float(_created_raw)
                                    _created_readable = datetime.fromtimestamp(_created_ts).strftime("%Y-%m-%d %H:%M:%S")
                                except (ValueError, TypeError):
                                    _created_readable = str(_created_raw)

                                with st.container():
                                    _signal_label = f" | シグナル: {_fp_signals}件" if _fp_signals else ""
                                    st.markdown(
                                        f"<div style='background:#FAFAFA;border-left:2px solid #CCC;"
                                        f"padding:6px 10px;margin:4px 0;border-radius:3px;'>"
                                        f"<small><b>{_display_id}</b> | 検出: {_created_readable} | 信頼度: {_conf*100:.0f}%{_signal_label}</small>",
                                        unsafe_allow_html=True
                                    )
                                    if _ttf_hours > 0:
                                        if _ttf_hours >= 24:
                                            ttf_display = f"障害予測: {_ttf_hours // 24}日後"
                                            if _failure_dt:
                                                ttf_display += f" ({_failure_dt})"
                                        else:
                                            ttf_display = f"障害予測: {_ttf_hours}時間後"
                                        st.markdown(f"<small>⏰ {ttf_display}</small>", unsafe_allow_html=True)
                                    st.markdown("</div>", unsafe_allow_html=True)
                                    
                                    # 個別ボタン（必要に応じて）
                                    _ind_col1, _ind_col2, _ind_spacer = st.columns([1, 1, 2])
                                    with _ind_col1:
                                        if st.button(
                                            "✅",
                                            key=f"ind_handled_{_fid[:8]}",
                                            help="この予兆のみ対応済み",
                                            use_container_width=True
                                        ):
                                            r = dt_engine.forecast_register_outcome(
                                                _fid, "mitigated",
                                                note=f"個別対応済み: {_rule_pattern} (device={_oc_device})"
                                            )
                                            if r.get("ok"):
                                                st.success(f"✅ 登録完了")
                                                #st.rerun()  # Disabled to prevent white screen
                                    
                                    with _ind_col2:
                                        if st.button(
                                            "❌",
                                            key=f"ind_false_{_fid[:8]}",
                                            help="この予兆のみ誤検知",
                                            use_container_width=True
                                        ):
                                            r = dt_engine.forecast_register_outcome(
                                                _fid, "false_alarm",
                                                note=f"個別誤検知: {_rule_pattern} (device={_oc_device})"
                                            )
                                            if r.get("ok"):
                                                st.info(f"❌ 登録完了")
                                                #st.rerun()  # Disabled to prevent white screen


        else:
            # prob <= 0.6 or no candidate
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

        # ============================================
        # C. Chat with AI Agent（Expander形式・旧UI復元）
        # ============================================
        with st.expander("💬 Chat with AI Agent", expanded=False):
            _chat_target_id = ""
            if selected_incident_candidate:
                _chat_target_id = selected_incident_candidate.get("id", "") or ""
            if not _chat_target_id and target_device_id:
                _chat_target_id = target_device_id

            _chat_ci = _build_ci_context_for_chat(topology, _chat_target_id) if _chat_target_id else {}
            if _chat_ci:
                _vendor     = _chat_ci.get("vendor", "") or "Unknown"
                _os         = _chat_ci.get("os",     "") or "Unknown"
                _model_name = _chat_ci.get("model",  "") or "Unknown"
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
                try:
                    # 新SDK: Client.chats.create()
                    _client = genai.Client(api_key=api_key)
                    st.session_state.chat_session = _client.chats.create(model="gemma-3-12b-it")
                except AttributeError:
                    # 旧SDK互換フォールバック
                    genai.configure(api_key=api_key)
                    model_obj = genai.GenerativeModel("gemma-3-12b-it")
                    st.session_state.chat_session = model_obj.start_chat(history=[])

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
                        #st.rerun()  # Disabled to prevent white screen

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
                    #st.rerun()  # Disabled to prevent white screen

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
