#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D1 + D2 テスト: app.py の表示ロジック検証
==========================================
app.py の Streamlit 依存を分離し、表示判定ロジックのみをテストする。
"""
import sys
sys.path.insert(0, "/home/claude")

PASS = 0
FAIL = 0
def check(cond, msg):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ✅ {msg}")
    else: FAIL += 1; print(f"  ❌ {msg}")

# ====================================================
# app.py のステータス判定ロジックを関数として抽出して検証
# ====================================================
def determine_status(cand, alarm_info):
    """app.py L1025-1050 のステータス判定ロジックを再現"""
    cand_type = cand.get('type', 'UNKNOWN')
    prob = cand.get('prob', 0)

    if cand.get('is_prediction'):
        status_text = "🔮 予兆検知"
        timeline = cand.get('prediction_timeline', '')
        affected = cand.get('prediction_affected_count', 0)
        if timeline and affected:
            action = f"⚡ {timeline}以内に対処 ({affected}台影響)"
        else:
            action = "⚡ 予防的対処を推奨"
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

    return status_text, action

def determine_remediation_banner(cand):
    """app.py L1236-1255 のRemediation バナー判定"""
    if cand.get('is_prediction'):
        return "prediction", "#fff3e0"  # オレンジ (予兆)
    else:
        return "confirmed", "#e8f5e9"   # グリーン (確定)


# ====================================================
# Test Suite
# ====================================================
print("=" * 65)
print("  D1 + D2 Test Suite: 表示ロジック検証")
print("=" * 65)

# ------ D1-1: 予兆アラートのステータス表示 ------
print("\n--- D1-1: 予兆アラートのステータス表示 ---")
pred_cand = {
    "id": "FW_01_PRIMARY",
    "prob": 0.72,
    "type": "Predictive/Software/Resource",
    "label": "🔮 [予兆] メモリ枯渇によるプロセスクラッシュ",
    "is_prediction": True,
    "prediction_timeline": "30分後",
    "prediction_affected_count": 7,
}
alarm_info_warning = {'severity': 'WARNING', 'is_silent': False}

status, action = determine_status(pred_cand, alarm_info_warning)
check("🔮" in status, f"ステータスに🔮アイコン: '{status}'")
check("予兆" in status, f"ステータスに '予兆' 含む: '{status}'")
check("30分後" in action, f"アクションにタイムライン: '{action}'")
check("7台" in action, f"アクションに影響台数: '{action}'")

# ------ D1-2: 既存ステータスが影響を受けないこと ------
print("\n--- D1-2: 既存ステータスの温存確認 ---")

# CRITICAL
crit_cand = {"id": "WAN_ROUTER_01", "prob": 0.9, "type": "Hardware/Power"}
crit_info = {'severity': 'CRITICAL', 'is_silent': False}
status, action = determine_status(crit_cand, crit_info)
check("🔴" in status, f"CRITICAL = 🔴: '{status}'")
check("自動修復" in action, f"CRITICAL action: '{action}'")

# WARNING
warn_cand = {"id": "FW_01_PRIMARY", "prob": 0.7, "type": "Software/Resource"}
warn_info = {'severity': 'WARNING', 'is_silent': False}
status, action = determine_status(warn_cand, warn_info)
check("🟡" in status, f"WARNING = 🟡: '{status}'")

# Silent Failure
silent_cand = {"id": "CORE_SW_01", "prob": 0.8, "type": "Network/SilentFailure"}
silent_info = {'severity': 'INFO', 'is_silent': True}
status, action = determine_status(silent_cand, silent_info)
check("🟣" in status, f"Silent = 🟣: '{status}'")

# 監視中
low_cand = {"id": "AP_01", "prob": 0.3, "type": "test"}
low_info = {'severity': 'INFO', 'is_silent': False}
status, action = determine_status(low_cand, low_info)
check("⚪" in status, f"低確度 = ⚪: '{status}'")

# ------ D1-3: Remediation バナーの分岐 ------
print("\n--- D1-3: Remediation バナー (予兆 vs 確定) ---")
banner_type, color = determine_remediation_banner(pred_cand)
check(banner_type == "prediction", f"予兆 → prediction バナー")
check(color == "#fff3e0", f"予兆 → オレンジ色 ({color})")

banner_type, color = determine_remediation_banner(crit_cand)
check(banner_type == "confirmed", f"確定 → confirmed バナー")
check(color == "#e8f5e9", f"確定 → グリーン色 ({color})")

# ------ D2-1: KPI メトリクス計算 ------
print("\n--- D2-1: KPI 予兆検知カウント ---")
analysis_results = [
    {"id": "FW_01_PRIMARY", "prob": 0.72, "is_prediction": True},
    {"id": "CORE_SW_01", "prob": 0.8, "type": "Network/SilentFailure"},
    {"id": "L2_SW_01", "prob": 0.2, "type": "Network/Unreachable"},
]
prediction_results = [r for r in analysis_results if r.get('is_prediction')]
prediction_count = len(prediction_results)
suspect_count = len([r for r in analysis_results if r.get('prob', 0) > 0.5])

check(prediction_count == 1, f"予兆検知数 = 1 (actual={prediction_count})")
check(suspect_count == 2, f"被疑箇所数 = 2 (actual={suspect_count})")

# ------ D2-2: 予兆なし時のKPI ------
print("\n--- D2-2: 予兆なし時のKPI ---")
no_pred_results = [
    {"id": "WAN_ROUTER_01", "prob": 0.9},
    {"id": "CORE_SW_01", "prob": 0.2},
]
pred_count_zero = len([r for r in no_pred_results if r.get('is_prediction')])
check(pred_count_zero == 0, f"予兆なし = 0 (actual={pred_count_zero})")

# ------ D1-4: 候補テーブルのフロー検証 ------
print("\n--- D1-4: 候補テーブルの列データ検証 ---")
# app.py の df_data 構築を再現
df_data = []
for rank, cand in enumerate([pred_cand, crit_cand], 1):
    prob = cand.get('prob', 0)
    device_id = cand['id']
    if cand.get('is_prediction'):
        ai = alarm_info_warning
    else:
        ai = crit_info
    status_text, act = determine_status(cand, ai)
    df_data.append({
        "順位": rank,
        "ステータス": status_text,
        "デバイス": device_id,
        "原因": cand.get('label', ''),
        "確信度": f"{prob*100:.0f}%",
        "推奨アクション": act,
    })

check(df_data[0]["ステータス"] == "🔮 予兆検知", f"1位のステータス: '{df_data[0]['ステータス']}'")
check(df_data[0]["確信度"] == "72%", f"1位の確信度: '{df_data[0]['確信度']}'")
check(df_data[1]["ステータス"] == "🔴 危険 (根本原因)", f"2位のステータス: '{df_data[1]['ステータス']}'")

# ------ D1-5: 予兆バナーの表示条件 ------
print("\n--- D1-5: 予兆検知バナーの表示条件 ---")
root_candidates_with_pred = [pred_cand, crit_cand]
pred_candidates = [c for c in root_candidates_with_pred if c.get('is_prediction')]
check(len(pred_candidates) == 1, f"予兆候補 = 1 (バナー表示あり)")

root_candidates_no_pred = [crit_cand]
pred_candidates2 = [c for c in root_candidates_no_pred if c.get('is_prediction')]
check(len(pred_candidates2) == 0, f"予兆候補 = 0 (バナー非表示)")


# ====================================================
# Summary
# ====================================================
print(f"\n{'='*65}")
total = PASS + FAIL
if FAIL == 0:
    print(f"  ✅ D1+D2 ALL {total} ASSERTIONS PASSED")
else:
    print(f"  ❌ {FAIL} FAILED, {PASS} passed ({PASS}/{total})")
print(f"{'='*65}")
