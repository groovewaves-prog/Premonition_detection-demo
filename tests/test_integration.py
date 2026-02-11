#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C1 + C2 結合テスト: Digital Twin + inference_engine.py
======================================================
実トポロジー (topology_a.json, topology_b.json) を使用し、
7つのシナリオで動作を検証する。

テスト対象:
  - digital_twin.py (お手元の最終版 - 3 Fixes適用済み)
  - inference_engine.py (既存コード + 最小限の修正)
"""

import sys, os, json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# パス設定
sys.path.insert(0, "/home/claude")

# --- 既存 Alarm クラス互換 ---
@dataclass
class Alarm:
    device_id: str
    message: str
    severity: str
    is_root_cause: bool = False
    is_silent_suspect: bool = False

# --- digital_twin を直接使うテスト (inference_engine の google.generativeai 依存を回避) ---
from digital_twin import DigitalTwinEngine

def load_topo(path):
    with open(path, 'r') as f:
        return json.load(f)

class IntegrationTestRCA:
    """
    既存 LogicalRCA の analyze() ロジックを再現するテスト用クラス。
    LLM (Gemma) 呼び出し部分のみスキップし、ルールベース判定 + Digital Twin を検証する。
    """
    def __init__(self, topology):
        self.topology = topology
        self.children_map = {}
        for dev_id, info in topology.items():
            p = info.get("parent_id") if isinstance(info, dict) else getattr(info, "parent_id", None)
            if p:
                self.children_map.setdefault(p, []).append(dev_id)

        # ★ Digital Twin (inference_engine.py __init__ と同じ)
        self.digital_twin = DigitalTwinEngine(topology, self.children_map)

    def analyze(self, alarms):
        """既存 LogicalRCA.analyze() のルールベース部分を再現"""
        if not alarms:
            return [{"id":"SYSTEM","label":"No alerts","prob":0.0,"type":"Normal","tier":0,"reason":"No alarms"}]

        # msg_map 構築 (既存 L229-231 と同じ)
        msg_map = {}
        for a in alarms:
            msg_map.setdefault(a.device_id, []).append(a.message)

        results = []
        for device_id, messages in msg_map.items():
            joined = " ".join(messages)
            joined_lower = joined.lower()

            # 既存ルールベース判定を再現 (analyze_redundancy_depth 相当)
            if "dual loss" in joined_lower or "device down" in joined_lower or "thermal shutdown" in joined_lower:
                prob, tier = 0.9, 1
            elif "power supply" in joined_lower and "failed" in joined_lower:
                prob, tier = 0.7, 2  # PSU冗長あり前提
            elif "fan fail" in joined_lower or ("fan" in joined_lower and "fail" in joined_lower):
                prob, tier = 0.7, 2
            elif "memory high" in joined_lower or "memory leak" in joined_lower:
                prob, tier = 0.7, 2
            elif "bgp" in joined_lower and ("flap" in joined_lower or "down" in joined_lower):
                prob, tier = 0.7, 2
            elif "heartbeat" in joined_lower or "ha state" in joined_lower or "degraded" in joined_lower:
                prob, tier = 0.7, 2
            elif "unreachable" in joined_lower:
                prob, tier = 0.2, 3
            else:
                prob, tier = 0.5, 3

            results.append({
                "id": device_id,
                "label": " / ".join(messages),
                "prob": prob,
                "type": "test",
                "tier": tier,
                "reason": f"Rule-based: {joined_lower[:50]}"
            })

        results.sort(key=lambda x: x["prob"], reverse=True)

        # ★ Digital Twin 予兆検知 (inference_engine.py に追加したブロックと同じ)
        if self.digital_twin is not None:
            try:
                predictions = self.digital_twin.predict(
                    analysis_results=results,
                    msg_map=msg_map,
                    alarms=alarms,
                )
                if predictions:
                    critical_ids = {r["id"] for r in results if r.get("prob", 0) >= 0.9}
                    filtered = [p for p in predictions if p["id"] not in critical_ids]
                    pred_ids = {p["id"] for p in filtered}
                    results = [r for r in results if r["id"] not in pred_ids]
                    results.extend(filtered)
                    results.sort(key=lambda x: (
                        0 if x.get("is_prediction") else 1,
                        -x.get("prob", 0),
                    ))
            except Exception as e:
                print(f"[!] Digital Twin error: {e}")
                import traceback; traceback.print_exc()

        return results


# ==========================================================
# Test Runner
# ==========================================================
PASS_COUNT = 0
FAIL_COUNT = 0

def assert_test(condition, msg):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"    ✅ {msg}")
    else:
        FAIL_COUNT += 1
        print(f"    ❌ FAIL: {msg}")

def run_test(name, topo_path, alarms):
    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"{'='*65}")
    engine = IntegrationTestRCA(load_topo(topo_path))
    results = engine.analyze(alarms)

    for i, r in enumerate(results, 1):
        ip = r.get("is_prediction", False)
        m = "🔮" if ip else "  "
        label = r['label'][:65]
        print(f"  {m} #{i} {r['id']:20s} | prob={r['prob']:.2f} | {label}")
        if ip:
            fc = r.get("prediction_confidence_factors", {})
            print(f"       Timeline: {r.get('prediction_timeline','?')} | Affected: {r.get('prediction_affected_count',0)} | SPOF: {fc.get('is_spof','?')} | HA: {fc.get('has_redundancy','?')}")

    return results


# ==========================================================
# Main
# ==========================================================
def main():
    topo_a = "/home/claude/existing_tool/Multiple-locations-demo-main/topologies/topology_a.json"
    topo_b = "/home/claude/existing_tool/Multiple-locations-demo-main/topologies/topology_b.json"

    print("=" * 65)
    print("  C1+C2 Integration Test Suite")
    print("  digital_twin.py (Final Fix) + inference_engine.py")
    print("=" * 65)

    # ----------------------------------------------------------
    # Test 1: FW Memory High (A拠点) - 仕様書デモシナリオ
    # Expected: 予測生成あり、配下 CORE_SW + L2_SW + AP に影響
    # ----------------------------------------------------------
    r = run_test("T1: FWメモリリーク (A拠点) - 仕様書デモ", topo_a, [
        Alarm("FW_01_PRIMARY", "Memory High", "WARNING", True)
    ])
    preds = [x for x in r if x.get("is_prediction")]
    assert_test(len(preds) >= 1, f"予測が生成された (count={len(preds)})")
    if preds:
        assert_test(preds[0]["prob"] >= 0.50, f"信頼度が閾値以上 (prob={preds[0]['prob']:.2f})")
        assert_test(preds[0].get("prediction_affected_count", 0) >= 3, f"配下3台以上に影響 (count={preds[0].get('prediction_affected_count',0)})")
        assert_test("🔮" in preds[0]["label"], "ラベルに🔮アイコンあり")
        assert_test(isinstance(preds[0]["prob"], float), f"prob が float 型 (type={type(preds[0]['prob']).__name__})")

    # ----------------------------------------------------------
    # Test 2: WAN Router Fan Failure (A拠点)
    # Expected: 予測あり、SPOF (WAN_ROUTER_01 は冗長グループなし)
    # ----------------------------------------------------------
    r = run_test("T2: WANルーターFAN障害 (A拠点) - SPOF", topo_a, [
        Alarm("WAN_ROUTER_01", "Fan Fail", "WARNING", True)
    ])
    preds = [x for x in r if x.get("is_prediction")]
    assert_test(len(preds) >= 1, f"予測が生成された (count={len(preds)})")
    if preds:
        fc = preds[0].get("prediction_confidence_factors", {})
        assert_test(fc.get("is_spof") == True, f"SPOF として検出 (is_spof={fc.get('is_spof')})")

    # ----------------------------------------------------------
    # Test 3: BGP Flapping (B拠点)
    # Expected: 高信頼度の予測 (BGPは高速エスカレーション)
    # ----------------------------------------------------------
    r = run_test("T3: BGPフラッピング (B拠点)", topo_b, [
        Alarm("EDGE_ROUTER_B01", "BGP Flapping", "WARNING", True)
    ])
    preds = [x for x in r if x.get("is_prediction")]
    assert_test(len(preds) >= 1, f"予測が生成された (count={len(preds)})")
    if preds:
        assert_test(preds[0]["prob"] >= 0.70, f"高信頼度 (prob={preds[0]['prob']:.2f})")
        assert_test(preds[0].get("prediction_affected_count", 0) >= 5, f"大規模影響 (count={preds[0].get('prediction_affected_count',0)})")

    # ----------------------------------------------------------
    # Test 4: FW HA Degraded (A拠点) - パートナーあり
    # Expected: 予測あり、HA割引で信頼度が低め
    # ----------------------------------------------------------
    r = run_test("T4: FW HA低下 (A拠点) - パートナーあり", topo_a, [
        Alarm("FW_01_PRIMARY", "Heartbeat Loss", "WARNING", True),
    ])
    preds = [x for x in r if x.get("is_prediction")]
    if preds:
        fc = preds[0].get("prediction_confidence_factors", {})
        assert_test(fc.get("has_redundancy") == True, f"冗長構成を検出 (has_redundancy={fc.get('has_redundancy')})")
        # HA割引が効いているか: REDUNDANCY_DISCOUNT=0.15 なので base 0.75 * 0.85 = ~0.64
        assert_test(preds[0]["prob"] < 0.70, f"HA割引で信頼度低下 (prob={preds[0]['prob']:.2f} < 0.70)")
        print(f"    ℹ️  HA構成での信頼度: {preds[0]['prob']:.2f}")
    else:
        assert_test(False, "予測が生成されるべき (HA割引後も閾値以上)")

    # ----------------------------------------------------------
    # Test 5: Both PSU Lost = CRITICAL → 予測不要
    # Expected: 予測なし (既に CRITICAL なので)
    # ----------------------------------------------------------
    r = run_test("T5: 両系電源障害 (A拠点) - 予測不要", topo_a, [
        Alarm("WAN_ROUTER_01", "Power Supply: Dual Loss (Device Down)", "CRITICAL", True)
    ])
    preds = [x for x in r if x.get("is_prediction")]
    assert_test(len(preds) == 0, f"CRITICAL 機器に予測なし (count={len(preds)})")

    # ----------------------------------------------------------
    # Test 6: SPOF L2 Switch (B拠点) - psu_count=1
    # Expected: 予測あり、SPOF ブースト
    # ----------------------------------------------------------
    r = run_test("T6: L2SW SPOF (B拠点)", topo_b, [
        Alarm("L2_SW_B03", "Memory High", "WARNING", True)
    ])
    preds = [x for x in r if x.get("is_prediction")]
    assert_test(len(preds) >= 1, f"予測が生成された (count={len(preds)})")
    if preds:
        fc = preds[0].get("prediction_confidence_factors", {})
        assert_test(fc.get("is_spof") == True, f"SPOF として検出 (is_spof={fc.get('is_spof')})")
        assert_test(preds[0].get("prediction_affected_count", 0) >= 2, f"配下AP に影響 (count={preds[0].get('prediction_affected_count',0)})")

    # ----------------------------------------------------------
    # Test 7: Normal - アラームなし
    # Expected: 予測なし、SYSTEM のみ
    # ----------------------------------------------------------
    r = run_test("T7: 正常稼働 (A拠点)", topo_a, [])
    preds = [x for x in r if x.get("is_prediction")]
    assert_test(len(preds) == 0, f"正常時に予測なし (count={len(preds)})")
    assert_test(r[0]["id"] == "SYSTEM", f"SYSTEM レスポンスのみ")

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print(f"\n{'='*65}")
    total = PASS_COUNT + FAIL_COUNT
    if FAIL_COUNT == 0:
        print(f"  ✅ ALL {total} ASSERTIONS PASSED ({PASS_COUNT}/{total})")
    else:
        print(f"  ❌ {FAIL_COUNT} FAILED, {PASS_COUNT} passed ({PASS_COUNT}/{total})")
    print(f"{'='*65}")

    return FAIL_COUNT == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
