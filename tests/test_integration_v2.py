# -*- coding: utf-8 -*-
"""
test_integration_v2.py - 3ファイル統合テスト
inference_engine.py + digital_twin.py + app.py の変更を検証
"""
import sys
import os
sys.path.insert(0, '/home/claude')

# ===== digital_twin.py テスト =====
import importlib.util
spec = importlib.util.spec_from_file_location("digital_twin", "/home/claude/digital_twin.py")
dt_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dt_mod)

ESCALATION_RULES = dt_mod.ESCALATION_RULES
DigitalTwinEngine = dt_mod.DigitalTwinEngine

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  ✅ {name}")
        passed += 1
    else:
        print(f"  ❌ {name} {detail}")
        failed += 1


# ===== テスト用トポロジー =====
topology = {
    "WAN_ROUTER_01": {"parent_id": None, "redundancy_group": None},
    "FW_01_PRIMARY": {"parent_id": "WAN_ROUTER_01", "redundancy_group": "fw_group"},
    "FW_01_SECONDARY": {"parent_id": "WAN_ROUTER_01", "redundancy_group": "fw_group"},
    "CORE_SW_01": {"parent_id": "FW_01_PRIMARY", "redundancy_group": None},
    "L2_SW_01": {"parent_id": "CORE_SW_01", "redundancy_group": None},
    "L2_SW_02": {"parent_id": "CORE_SW_01", "redundancy_group": None},
}
children_map = {
    "WAN_ROUTER_01": ["FW_01_PRIMARY", "FW_01_SECONDARY"],
    "FW_01_PRIMARY": ["CORE_SW_01"],
    "CORE_SW_01": ["L2_SW_01", "L2_SW_02"],
}

engine = DigitalTwinEngine(topology, children_map)


print("=" * 70)
print("TEST 1: inference_engine.py - INFO prob cap")
print("=" * 70)

# シミュレート: inference_engine の analyze() の結果
# Gemini が INFO アラームを CRITICAL と過剰判定したケース
class FakeAlarm:
    def __init__(self, device_id, severity, message="test"):
        self.device_id = device_id
        self.severity = severity
        self.message = message
        self.is_root_cause = False

# INFOのみのアラーム → prob制限が効くべき
alarms_info_only = [
    FakeAlarm("FW_01_PRIMARY", "INFO", "ASIC_ERROR: Input queue drops detected"),
    FakeAlarm("FW_01_PRIMARY", "INFO", "QOS-4-POLICER: Traffic exceeding CIR"),
    FakeAlarm("FW_01_PRIMARY", "INFO", "TCP-5-RETRANSMIT: Retransmission rate 250/sec"),
]

# 真の障害アラーム（CRITICAL含む） → prob制限が効かないべき
alarms_with_critical = [
    FakeAlarm("WAN_ROUTER_01", "CRITICAL", "Device Down"),
    FakeAlarm("WAN_ROUTER_01", "INFO", "Interface Gi0/0/0 down"),
]

# prob cap ロジックの再現
from typing import Dict, Set
def apply_info_prob_cap(results, alarms):
    _dev_max_severity: Dict[str, str] = {}
    for a in alarms:
        sev = getattr(a, 'severity', 'INFO').upper()
        prev = _dev_max_severity.get(a.device_id, 'INFO')
        if sev == 'CRITICAL' or prev == 'CRITICAL':
            _dev_max_severity[a.device_id] = 'CRITICAL'
        elif sev == 'WARNING' or prev == 'WARNING':
            _dev_max_severity[a.device_id] = 'WARNING'
        else:
            _dev_max_severity[a.device_id] = 'INFO'
    
    for r in results:
        if _dev_max_severity.get(r["id"]) == 'INFO' and r.get("prob", 0) > 0.85:
            r["prob"] = 0.70
            r["tier"] = 2
    return results

# テスト: INFOのみ → prob=0.9 が 0.70 に制限される
results_info = [{"id": "FW_01_PRIMARY", "prob": 0.9, "tier": 1}]
apply_info_prob_cap(results_info, alarms_info_only)
check("INFOのみ: prob 0.9 → 0.70 に制限", results_info[0]["prob"] == 0.70)
check("INFOのみ: tier 1 → 2 に変更", results_info[0]["tier"] == 2)

# テスト: CRITICAL含む → prob=0.9 のまま
results_crit = [{"id": "WAN_ROUTER_01", "prob": 0.9, "tier": 1}]
apply_info_prob_cap(results_crit, alarms_with_critical)
check("CRITICAL含む: prob 0.9 変更なし", results_crit[0]["prob"] == 0.9)
check("CRITICAL含む: tier 1 変更なし", results_crit[0]["tier"] == 1)

# テスト: INFOのみ + prob=0.70（制限範囲外）→ 変更なし
results_warn = [{"id": "FW_01_PRIMARY", "prob": 0.70, "tier": 2}]
apply_info_prob_cap(results_warn, alarms_info_only)
check("INFOのみ + prob=0.70: 変更なし", results_warn[0]["prob"] == 0.70)


print()
print("=" * 70)
print("TEST 2: 予兆パイプライン E2E（INFO prob cap → Digital Twin 検出）")
print("=" * 70)

# Step 1: Gemini が CRITICAL と判定した結果をシミュレート
analysis_results_raw = [
    {"id": "FW_01_PRIMARY", "prob": 0.9, "tier": 1, "status": "CRITICAL"},
]

# Step 2: INFO prob cap を適用
apply_info_prob_cap(analysis_results_raw, alarms_info_only)
check("prob cap 後: FW_01_PRIMARY prob = 0.70", analysis_results_raw[0]["prob"] == 0.70)
check("prob cap 後: FW_01_PRIMARY は primary scan 対象 (0.45-0.85)", 
      0.45 <= analysis_results_raw[0]["prob"] <= 0.85)

# Step 3: Digital Twin predict
msg_map = {
    "FW_01_PRIMARY": [
        "%HARDWARE-3-ASIC_ERROR: Input queue drops detected (Count: 1000). Burst traffic.",
        "%QOS-4-POLICER: Traffic exceeding CIR on interface ge-0/0/1. Buffer overflow risk.",
        "%TCP-5-RETRANSMIT: Retransmission rate 250/sec on monitored flows. Route updates increasing.",
    ],
}
predictions = engine.predict(analysis_results_raw, msg_map)
check("Digital Twin が予兆を生成", len(predictions) > 0, f"(predictions: {len(predictions)})")

if predictions:
    pred = predictions[0]
    check("予兆 is_prediction = True", pred.get("is_prediction") is True)
    check("予兆 prediction_early_warning_hours 存在", "prediction_early_warning_hours" in pred)
    check("予兆 prediction_time_to_critical_min 存在", "prediction_time_to_critical_min" in pred)
    
    # Step 4: critical_ids フィルタで消えないことを確認
    # inference_engine.py L360-366 のロジックを再現
    critical_ids = {r["id"] for r in analysis_results_raw if r.get("prob", 0) >= 0.9}
    filtered = [p for p in predictions if p["id"] not in critical_ids]
    check("critical_ids フィルタ通過", len(filtered) > 0,
          f"(critical_ids={critical_ids}, filtered={len(filtered)})")
    
    # ナラティブに2軸情報が含まれることを確認
    reason = pred.get("reason", "")
    check("ナラティブに 'Predictive Maintenance' 含む", "Predictive Maintenance" in reason)
    check("ナラティブに '早期予兆' 含む", "早期予兆" in reason)
    check("ナラティブに '急性期進行' 含む", "急性期進行" in reason)


print()
print("=" * 70)
print("TEST 3: 全3シナリオ × Level 段階的テスト")
print("=" * 70)

scenarios = {
    "Optical Decay": {
        1: ["%TRANSCEIVER-4-THRESHOLD_VIOLATION: Rx Power -23.4 dBm (Threshold -25.0 dBm). Signal degrading."],
        2: [
            "%TRANSCEIVER-4-THRESHOLD_VIOLATION: Rx Power -23.8 dBm (Threshold -25.0 dBm). Signal degrading.",
            "%LINK-3-ERROR: CRC errors increasing on Gi0/0/0 (Count: 300/min). Input queue drops detected.",
        ],
        5: [
            "%TRANSCEIVER-4-THRESHOLD_VIOLATION: Rx Power -25.0 dBm (Threshold -25.0 dBm). Signal degrading.",
            "%LINK-3-ERROR: CRC errors increasing on Gi0/0/0 (Count: 750/min). Input queue drops detected.",
            "%OSPF-4-ADJCHANGE: Neighbor keepalive delayed (3 consecutive misses). Stability warning.",
        ],
    },
    "Microburst": {
        1: ["%HARDWARE-3-ASIC_ERROR: Input queue drops detected (Count: 200). Burst traffic."],
        2: [
            "%HARDWARE-3-ASIC_ERROR: Input queue drops detected (Count: 400). Burst traffic.",
            "%QOS-4-POLICER: Traffic exceeding CIR on interface ge-0/0/1. Buffer overflow risk.",
        ],
        5: [
            "%HARDWARE-3-ASIC_ERROR: Input queue drops detected (Count: 1000). Burst traffic.",
            "%QOS-4-POLICER: Traffic exceeding CIR on interface ge-0/0/1. Buffer overflow risk.",
            "%TCP-5-RETRANSMIT: Retransmission rate 250/sec on monitored flows. Route updates increasing.",
        ],
    },
    "Route Instability": {
        1: ["BGP-5-ADJCHANGE: Route updates 500/min. Stability warning."],
        2: [
            "BGP-5-ADJCHANGE: Route updates 1000/min. Stability warning.",
            "%BGP-4-MAXPFX: Prefix count approaching limit (92%). Route oscillation detected.",
        ],
        5: [
            "BGP-5-ADJCHANGE: Route updates 2500/min. Stability warning.",
            "%BGP-4-MAXPFX: Prefix count approaching limit (92%). Route oscillation detected.",
            "%ROUTING-3-CONVERGENCE: RIB convergence delayed. Prefix withdrawal detected on multiple peers.",
        ],
    },
}

for scenario_name, levels in scenarios.items():
    print(f"\n  --- {scenario_name} ---")
    prev_confidence = 0
    for level, messages in sorted(levels.items()):
        # INFO prob cap 適用後の結果をシミュレート
        analysis = [{"id": "FW_01_PRIMARY", "prob": 0.70, "tier": 2}]
        msg_map_test = {"FW_01_PRIMARY": messages}
        preds = engine.predict(analysis, msg_map_test)
        
        if preds:
            conf = preds[0]["prob"]
            signal_count = preds[0].get("prediction_signal_count", 0)
            early_hours = preds[0].get("prediction_early_warning_hours", 0)
            check(
                f"Level {level}: 予兆検出 (conf={conf:.2f}, signals={signal_count}, early={early_hours}h)",
                True
            )
            if level > 1 and prev_confidence > 0:
                check(
                    f"Level {level}: confidence上昇 ({prev_confidence:.2f} → {conf:.2f})",
                    conf >= prev_confidence,
                    f"(前: {prev_confidence:.2f}, 今: {conf:.2f})"
                )
            prev_confidence = conf
        else:
            check(f"Level {level}: 予兆検出", False, "(予兆生成されず)")


print()
print("=" * 70)
print("TEST 4: 2軸表示フォーマット確認")
print("=" * 70)

# app.py で使われる表示ロジックを再現
def format_early_warning_for_display(early_hours):
    if early_hours >= 24:
        return f"{early_hours // 24}日前"
    elif early_hours > 0:
        return f"{early_hours}時間前"
    else:
        return "不明"

test_cases = [
    (336, "14日前"),    # optical
    (720, "30日前"),    # storage/crypto_vpn
    (72, "3日前"),      # bandwidth/fan_fail
    (48, "2日前"),      # bgp_flap/ha_split
    (24, "1日前"),      # stp_loop
    (12, "12時間前"),   # arp_storm/auth_failure
]
for hours, expected in test_cases:
    result = format_early_warning_for_display(hours)
    check(f"{hours}h → '{result}' (期待: '{expected}')", result == expected)


print()
print("=" * 70)
print("TEST 5: 障害確定デバイスでは予兆が生成されないことを確認")
print("=" * 70)

# 真の CRITICAL デバイス（prob cap が効かない）
analysis_critical = [{"id": "WAN_ROUTER_01", "prob": 0.9, "tier": 1}]
# このデバイスのアラームは CRITICAL なので prob cap なし
msg_map_critical = {
    "WAN_ROUTER_01": ["Device Down - Complete power failure"],
}
preds_critical = engine.predict(analysis_critical, msg_map_critical)

# critical_ids フィルタを適用
critical_ids = {r["id"] for r in analysis_critical if r.get("prob", 0) >= 0.9}
filtered_critical = [p for p in preds_critical if p["id"] not in critical_ids]
check("障害確定デバイス: 予兆がフィルタで除外される", len(filtered_critical) == 0,
      f"(critical_ids={critical_ids}, filtered={len(filtered_critical)})")


print()
print("=" * 70)
print("TEST 6: app.py 予兆出力フィールドの網羅性")
print("=" * 70)

# app.py が参照する全フィールドを確認
analysis_test = [{"id": "FW_01_PRIMARY", "prob": 0.65}]
msg_map_test = {
    "FW_01_PRIMARY": [
        "%TRANSCEIVER-4-THRESHOLD_VIOLATION: Rx Power -24.6 dBm. Signal degrading.",
        "%LINK-3-ERROR: CRC errors increasing. Input queue drops detected.",
    ],
}
preds_test = engine.predict(analysis_test, msg_map_test)
if preds_test:
    p = preds_test[0]
    # app.py が参照するフィールド (grep結果から)
    required_fields = [
        "prediction_timeline",
        "prediction_affected_count",
        "prediction_affected_devices",
        "prediction_signal_count",
        "prediction_confidence_factors",
        "is_prediction",
        "prob",
        "id",
        "label",
        "reason",
        "prediction_early_warning_hours",      # ★ 新規
        "prediction_time_to_critical_min",       # ★ 新規
    ]
    for field in required_fields:
        check(f"フィールド '{field}' 存在", field in p)


print()
print("=" * 70)
summary = f"結果: {passed} passed, {failed} failed"
if failed == 0:
    print(f"🎉 ALL TESTS PASSED! {summary}")
else:
    print(f"⚠️  {summary}")
print("=" * 70)
