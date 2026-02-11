#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全体回帰テスト: C1+C2+D1+D2
============================
RCA温存 + Digital Twin予兆 + 表示ロジック の一気通貫検証
"""
import sys, json
sys.path.insert(0, "/home/claude")
from dataclasses import dataclass
from typing import List, Dict, Any
from digital_twin import DigitalTwinEngine

@dataclass
class Alarm:
    device_id: str
    message: str
    severity: str
    is_root_cause: bool = False
    is_silent_suspect: bool = False

def load_topo(path):
    with open(path, 'r') as f:
        return json.load(f)

class FullPipelineRCA:
    """既存 LogicalRCA + Digital Twin + 表示判定 の全パイプライン"""
    SILENT_MIN_CHILDREN = 2
    SILENT_RATIO = 0.5

    def __init__(self, topology):
        self.topology = topology
        self.children_map = {}
        for dev_id, info in topology.items():
            p = info.get("parent_id") if isinstance(info, dict) else None
            if p:
                self.children_map.setdefault(p, []).append(dev_id)
        self.digital_twin = DigitalTwinEngine(topology, self.children_map)

    def _get_parent_id(self, dev_id):
        info = self.topology.get(dev_id, {})
        return info.get("parent_id") if isinstance(info, dict) else None

    def _is_connection_loss(self, msg):
        return any(kw in msg.lower() for kw in ["connection lost", "unreachable", "link down"])

    def _detect_silent_failures(self, msg_map):
        suspects = {}
        for parent_id, children in self.children_map.items():
            if not children or parent_id in msg_map:
                continue
            affected = [c for c in children if any(self._is_connection_loss(m) for m in msg_map.get(c, []))]
            if not affected: continue
            total = len(children)
            ratio = len(affected) / max(total, 1)
            if len(affected) >= self.SILENT_MIN_CHILDREN and ratio >= self.SILENT_RATIO:
                suspects[parent_id] = {"children": affected, "evidence_count": len(affected), "total_children": total, "ratio": ratio}
        return suspects

    def analyze(self, alarms):
        if not alarms:
            return [{"id": "SYSTEM", "prob": 0.0, "type": "Normal", "label": "No alerts", "tier": 0, "reason": ""}]
        msg_map = {}
        for a in alarms:
            msg_map.setdefault(a.device_id, []).append(a.message)
        silent_suspects = self._detect_silent_failures(msg_map)
        for parent_id in silent_suspects:
            msg_map.setdefault(parent_id, []).append("Silent Failure Suspected")
        alarmed_ids = set(msg_map.keys())
        results = []
        for device_id, messages in msg_map.items():
            parent = self._get_parent_id(device_id)
            if parent in silent_suspects and any(self._is_connection_loss(m) for m in messages):
                results.append({"id": device_id, "prob": 0.4, "type": "Network/ConnectionLost", "label": " / ".join(messages), "tier": 3, "reason": f"Downstream of {parent}"})
                continue
            if any("unreachable" in m.lower() for m in messages) and parent and parent in alarmed_ids:
                results.append({"id": device_id, "prob": 0.2, "type": "Network/Unreachable", "label": " / ".join(messages), "tier": 3, "reason": f"Cascade from {parent}"})
                continue
            if device_id in silent_suspects:
                info = silent_suspects[device_id]
                results.append({"id": device_id, "prob": 0.8, "type": "Network/SilentFailure", "label": " / ".join(messages), "tier": 1, "reason": f"Silent: {info['evidence_count']}/{info['total_children']}"})
                continue
            joined = " ".join(messages).lower()
            if "dual loss" in joined or "device down" in joined:
                prob, tier = 0.9, 1
            elif any(kw in joined for kw in ["memory", "fan", "bgp", "heartbeat", "power supply"]):
                prob, tier = 0.7, 2
            else:
                prob, tier = 0.5, 3
            results.append({"id": device_id, "prob": prob, "type": "test", "label": " / ".join(messages), "tier": tier, "reason": "Rule-based"})
        results.sort(key=lambda x: x["prob"], reverse=True)
        # Digital Twin
        if self.digital_twin:
            try:
                preds = self.digital_twin.predict(results, msg_map, alarms)
                if preds:
                    crit_ids = {r["id"] for r in results if r.get("prob", 0) >= 0.9}
                    filtered = [p for p in preds if p["id"] not in crit_ids]
                    pred_ids = {p["id"] for p in filtered}
                    results = [r for r in results if r["id"] not in pred_ids]
                    results.extend(filtered)
                    results.sort(key=lambda x: (0 if x.get("is_prediction") else 1, -x.get("prob", 0)))
            except Exception as e:
                print(f"[!] DT error: {e}"); import traceback; traceback.print_exc()
        return results

    def build_display_data(self, alarms, analysis_results):
        """app.py の表示データ構築を再現"""
        # alarm_info_map
        aim = {}
        for a in alarms:
            if a.device_id not in aim:
                aim[a.device_id] = {'severity': 'INFO', 'is_silent': False}
            if a.severity == 'CRITICAL':
                aim[a.device_id]['severity'] = 'CRITICAL'
            elif a.severity == 'WARNING' and aim[a.device_id]['severity'] != 'CRITICAL':
                aim[a.device_id]['severity'] = 'WARNING'
            if a.is_silent_suspect:
                aim[a.device_id]['is_silent'] = True

        # 候補分離
        rc_ids = set(a.device_id for a in alarms if a.is_root_cause)
        ds_ids = set(a.device_id for a in alarms if not a.is_root_cause)
        root_cands, ds_devs = [], []
        for cand in analysis_results:
            did = cand.get('id', '')
            if did in rc_ids:
                root_cands.append(cand)
            elif did in ds_ids:
                ds_devs.append(cand)
            elif cand.get('prob', 0) > 0.5:
                root_cands.append(cand)

        # KPI
        prediction_count = len([r for r in analysis_results if r.get('is_prediction')])
        suspect_count = len([r for r in analysis_results if r.get('prob', 0) > 0.5])

        # ステータス判定
        df_data = []
        for rank, cand in enumerate(root_cands, 1):
            prob = cand.get('prob', 0)
            ct = cand.get('type', 'UNKNOWN')
            did = cand['id']
            ai = aim.get(did, {'severity': 'INFO', 'is_silent': False})
            if cand.get('is_prediction'):
                st = "🔮 予兆検知"
                tl = cand.get('prediction_timeline', '')
                af = cand.get('prediction_affected_count', 0)
                act = f"⚡ {tl}以内に対処 ({af}台影響)" if tl and af else "⚡ 予防的対処を推奨"
            elif ai['is_silent'] or "Silent" in ct:
                st, act = "🟣 サイレント疑い", "🔍 上位確認"
            elif ai['severity'] == 'CRITICAL':
                st, act = "🔴 危険 (根本原因)", "🚀 自動修復が可能"
            elif ai['severity'] == 'WARNING':
                st, act = "🟡 警告", "🔍 詳細調査"
            elif prob > 0.6:
                st, act = "🟡 被疑箇所", "🔍 詳細調査"
            else:
                st, act = "⚪ 監視中", "👁️ 静観"
            df_data.append({"rank": rank, "status": st, "device": did, "prob": prob, "action": act, "is_pred": cand.get("is_prediction", False)})

        # 予兆バナー
        pred_cands = [c for c in root_cands if c.get('is_prediction')]
        has_prediction_banner = len(pred_cands) > 0

        return {
            "table": df_data,
            "prediction_count": prediction_count,
            "suspect_count": suspect_count,
            "downstream_count": len(ds_devs),
            "has_prediction_banner": has_prediction_banner,
        }


# ====================================================
PASS = 0; FAIL = 0
def check(cond, msg):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ✅ {msg}")
    else: FAIL += 1; print(f"  ❌ {msg}")

topo_a = load_topo("/home/claude/existing_tool/Multiple-locations-demo-main/topologies/topology_a.json")
topo_b = load_topo("/home/claude/existing_tool/Multiple-locations-demo-main/topologies/topology_b.json")

# ============================================================
print("\n" + "="*65)
print("  S1: FW Memory High (WARNING) - 全パイプライン")
print("="*65)
eng = FullPipelineRCA(topo_a)
alarms = [Alarm("FW_01_PRIMARY", "Memory High", "WARNING", True)]
results = eng.analyze(alarms)
display = eng.build_display_data(alarms, results)

for row in display["table"]:
    print(f"  {row['status']:20s} {row['device']:20s} prob={row['prob']:.2f} | {row['action']}")

check(display["prediction_count"] == 1, f"KPI予兆=1 (actual={display['prediction_count']})")
check(display["suspect_count"] == 1, f"KPI被疑=1 (actual={display['suspect_count']})")
check(display["has_prediction_banner"], "予兆バナーあり")
check(display["table"][0]["status"] == "🔮 予兆検知", f"1位: 🔮 予兆検知")
check(display["table"][0]["is_pred"] == True, "1位は予測フラグON")

# ============================================================
print("\n" + "="*65)
print("  S2: CRITICAL + Cascade - 既存RCA温存")
print("="*65)
eng2 = FullPipelineRCA(topo_a)
alarms2 = [
    Alarm("FW_01_PRIMARY", "Power Supply: Dual Loss (Device Down)", "CRITICAL", True),
    Alarm("CORE_SW_01", "Unreachable - upstream failure", "WARNING", False),
]
results2 = eng2.analyze(alarms2)
display2 = eng2.build_display_data(alarms2, results2)

for row in display2["table"]:
    print(f"  {row['status']:20s} {row['device']:20s} prob={row['prob']:.2f} | {row['action']}")

check(display2["prediction_count"] == 0, f"KPI予兆=0 (actual={display2['prediction_count']})")
check(not display2["has_prediction_banner"], "予兆バナーなし")
check(display2["table"][0]["status"] == "🔴 危険 (根本原因)", f"1位: CRITICAL")
check(display2["downstream_count"] >= 1, f"下流デバイスあり (count={display2['downstream_count']})")

# ============================================================
print("\n" + "="*65)
print("  S3: サイレント障害 + WARNING予兆 混在")
print("="*65)
eng3 = FullPipelineRCA(topo_a)
alarms3 = [
    Alarm("L2_SW_01", "Connection Lost", "CRITICAL", False, False),
    Alarm("L2_SW_02", "Connection Lost", "CRITICAL", False, False),
    Alarm("FW_01_PRIMARY", "Memory High", "WARNING", True, False),
]
results3 = eng3.analyze(alarms3)
display3 = eng3.build_display_data(alarms3, results3)

for row in display3["table"]:
    print(f"  {row['status']:20s} {row['device']:20s} prob={row['prob']:.2f} | {row['action']}")

# サイレント疑い (CORE_SW_01) と予兆 (FW_01_PRIMARY) が共存するはず
has_silent = any(r["status"] == "🟣 サイレント疑い" for r in display3["table"])
has_pred = any(r["status"] == "🔮 予兆検知" for r in display3["table"])
check(has_silent, "サイレント疑いが表示されている")
check(has_pred, "予兆検知が表示されている")
check(display3["prediction_count"] >= 1, f"予兆カウント≥1 (actual={display3['prediction_count']})")

# ============================================================
print("\n" + "="*65)
print("  S4: B拠点 BGP Flap (SPOF) - 高信頼度予測")
print("="*65)
eng4 = FullPipelineRCA(topo_b)
alarms4 = [Alarm("EDGE_ROUTER_B01", "BGP Flapping", "WARNING", True)]
results4 = eng4.analyze(alarms4)
display4 = eng4.build_display_data(alarms4, results4)

for row in display4["table"]:
    print(f"  {row['status']:20s} {row['device']:20s} prob={row['prob']:.2f} | {row['action']}")

check(display4["table"][0]["status"] == "🔮 予兆検知", "BGP予測: 🔮 予兆検知")
check(display4["table"][0]["prob"] >= 0.85, f"高信頼度 (prob={display4['table'][0]['prob']:.2f})")
check("15分後" in display4["table"][0]["action"], f"アクションに15分表示: {display4['table'][0]['action']}")

# ============================================================
print("\n" + "="*65)
print("  S5: 正常稼働 - 予兆も通常アラームもなし")
print("="*65)
eng5 = FullPipelineRCA(topo_a)
results5 = eng5.analyze([])
display5 = eng5.build_display_data([], results5)
check(display5["prediction_count"] == 0, "予兆=0")
check(display5["suspect_count"] == 0, "被疑=0")
check(not display5["has_prediction_banner"], "バナーなし")

# ============================================================
print(f"\n{'='*65}")
total = PASS + FAIL
if FAIL == 0:
    print(f"  ✅ FULL REGRESSION: ALL {total} ASSERTIONS PASSED")
else:
    print(f"  ❌ {FAIL} FAILED, {PASS} passed ({PASS}/{total})")
print(f"{'='*65}")
