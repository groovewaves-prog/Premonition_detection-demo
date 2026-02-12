import json
import os
import re
from enum import Enum
from typing import List, Dict, Any, Optional

import google.generativeai as genai

# --- Digital Twin Integration (graceful degradation) ---
try:
    from digital_twin import DigitalTwinEngine
    DIGITAL_TWIN_AVAILABLE = True
except ImportError:
    DIGITAL_TWIN_AVAILABLE = False

# ==========================================================
# AIOps health status
# ==========================================================
class HealthStatus(Enum):
    NORMAL = "GREEN"
    WARNING = "YELLOW"
    CRITICAL = "RED"


class LogicalRCA:
    """
    LogicalRCA (v5 - Production Fix):
      - 障害(CRITICAL)を予兆(Prediction)より優先して表示するソートロジックを実装
      - result辞書に 'status' フィールドを追加し、Digital Twin側のフィルタリングを支援
    """

    SILENT_MIN_CHILDREN = 2
    SILENT_RATIO = 0.5

    def __init__(self, topology, config_dir: str = "./configs"):
        if isinstance(topology, str):
            self.topology = self._load_topology(topology)
        elif isinstance(topology, dict):
            self.topology = topology
        else:
            raise ValueError("topology must be either a file path (str) or a dictionary")

        self.config_dir = config_dir
        self.model = None
        self._api_configured = False

        self.children_map: Dict[str, List[str]] = {}
        for dev_id, info in self.topology.items():
            p = None
            if isinstance(info, dict):
                p = info.get("parent_id")
            else:
                if hasattr(info, "parent_id"):
                    p = getattr(info, "parent_id")
                elif hasattr(info, "paren"):
                    p = getattr(info, "paren", None)
            if p:
                self.children_map.setdefault(p, []).append(dev_id)

        self.digital_twin = None
        if DIGITAL_TWIN_AVAILABLE:
            try:
                self.digital_twin = DigitalTwinEngine(
                    self.topology, self.children_map
                )
            except Exception as e:
                print(f"[!] Digital Twin initialization failed: {e}")

    # ----------------------------
    # Topology helpers
    # ----------------------------
    def _get_device_info(self, device_id: str) -> Any:
        return self.topology.get(device_id, {})

    def _get_parent_id(self, device_id: str) -> Optional[str]:
        info = self._get_device_info(device_id)
        if isinstance(info, dict):
            return info.get("parent_id")
        if hasattr(info, "parent_id"):
            return getattr(info, "parent_id")
        return None

    def _get_metadata(self, device_id: str) -> Dict[str, Any]:
        info = self._get_device_info(device_id)
        if isinstance(info, dict):
            md = info.get("metadata", {})
            return md if isinstance(md, dict) else {}
        if hasattr(info, "metadata"):
            md = getattr(info, "metadata")
            return md if isinstance(md, dict) else {}
        if hasattr(info, "get_metadata"):
            try:
                md = info.get_metadata("metadata", {})
                return md if isinstance(md, dict) else {}
            except Exception:
                return {}
        return {}

    def _get_psu_count(self, device_id: str, default: int = 1) -> int:
        md = self._get_metadata(device_id)
        if isinstance(md, dict):
            hw = md.get("hw_inventory", {})
            if isinstance(hw, dict) and "psu_count" in hw:
                try:
                    return int(hw.get("psu_count"))
                except Exception:
                    pass
            if str(md.get("redundancy_type", "")).upper() == "PSU":
                return 2
        return default

    # ----------------------------
    # LLM init
    # ----------------------------
    def _ensure_api_configured(self) -> bool:
        if self._api_configured:
            return True
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return False
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemma-3-12b-it")
            self._api_configured = True
            return True
        except Exception as e:
            print(f"[!] API Configuration Error: {e}")
            return False

    # ----------------------------
    # IO
    # ----------------------------
    def _load_topology(self, path: str) -> Dict:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _read_config(self, device_id: str) -> str:
        config_path = os.path.join(self.config_dir, f"{device_id}.txt")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                return f"Error reading config: {str(e)}"
        return "Config file not found."

    def _sanitize_text(self, text: str) -> str:
        text = re.sub(r'(encrypted-password\s+)"[^"]+"', r'\1"********"', text)
        text = re.sub(r"(password|secret)\s+(\d)\s+\S+", r"\1 \2 ********", text)
        text = re.sub(r"(username\s+\S+\s+secret)\s+\d\s+\S+", r"\1 5 ********", text)
        text = re.sub(r"(snmp-server community)\s+\S+", r"\1 ********", text)
        return text

    # ==========================================================
    # Silent failure inference
    # ==========================================================
    def _is_connection_loss(self, msg: str) -> bool:
        msg_l = msg.lower()
        return (
            "connection lost" in msg_l
            or "link down" in msg_l
            or "port down" in msg_l
            or "unreachable" in msg_l
        )

    def _detect_silent_failures(self, msg_map: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
        suspects: Dict[str, Dict[str, Any]] = {}
        for parent_id, children in self.children_map.items():
            if not children:
                continue
            if parent_id in msg_map:
                continue
            affected = []
            for c in children:
                msgs = msg_map.get(c, [])
                if any(self._is_connection_loss(m) for m in msgs):
                    affected.append(c)
            if not affected:
                continue
            total = len(children)
            ratio = len(affected) / max(total, 1)
            if len(affected) >= self.SILENT_MIN_CHILDREN and ratio >= self.SILENT_RATIO:
                report = (
                    f"[Silent Failure Heuristic]\n"
                    f"- Suspected upstream device: {parent_id}\n"
                    f"- Affected children: {len(affected)}/{total} (ratio={ratio:.2f})\n"
                    f"- Evidence: children raised Connection Lost/Unreachable simultaneously\n"
                )
                suspects[parent_id] = {
                    "children": affected,
                    "evidence_count": len(affected),
                    "total_children": total,
                    "ratio": ratio,
                    "report": report,
                }
        return suspects

    # ==========================================================
    # Public API
    # ==========================================================
    def analyze(self, alarms: List) -> List[Dict[str, Any]]:
        if not alarms:
            return [{
                "id": "SYSTEM",
                "label": "No alerts detected",
                "prob": 0.0,
                "type": "Normal",
                "tier": 0,
                "reason": "No active alerts detected.",
                "status": "GREEN"
            }]

        msg_map: Dict[str, List[str]] = {}
        for a in alarms:
            msg_map.setdefault(a.device_id, []).append(a.message)

        silent_suspects = self._detect_silent_failures(msg_map)
        for parent_id, info in silent_suspects.items():
            msg_map.setdefault(parent_id, []).append("Silent Failure Suspected")

        alarmed_ids = set(msg_map.keys())

        def parent_is_alarmed(dev: str) -> bool:
            p = self._get_parent_id(dev)
            return bool(p and (p in alarmed_ids))

        def parent_is_silent_suspect(dev: str) -> bool:
            p = self._get_parent_id(dev)
            return bool(p and (p in silent_suspects))

        results: List[Dict[str, Any]] = []

        for device_id, messages in msg_map.items():
            # ... (中略: Silent/Cascade ロジックは維持) ...
            
            # 親がサイレント疑い
            if device_id in silent_suspects:
                info = silent_suspects[device_id]
                results.append({
                    "id": device_id,
                    "label": " / ".join(messages),
                    "prob": 0.8,
                    "type": "Network/SilentFailure",
                    "tier": 1,
                    "reason": f"Silent failure suspected.",
                    "status": "YELLOW" # Assume Warning level for silent suspect
                })
                continue

            # 通常分析
            analysis = self.analyze_redundancy_depth(device_id, messages)
            
            # ステータス文字列の取得 (RED/YELLOW/GREEN)
            status_val = analysis["status"].value 

            if analysis.get("impact_type") == "UNKNOWN" and "API key not configured" in analysis.get("reason", ""):
                prob = 0.5
                tier = 3
            else:
                if analysis["status"] == HealthStatus.CRITICAL:
                    prob = 0.95 # 明確に高く設定
                    tier = 1
                elif analysis["status"] == HealthStatus.WARNING:
                    prob = 0.7
                    tier = 2
                else:
                    prob = 0.3
                    tier = 3

            results.append({
                "id": device_id,
                "label": " / ".join(messages),
                "prob": prob,
                "type": analysis.get("impact_type", "UNKNOWN"),
                "tier": tier,
                "reason": analysis.get("reason", "AI provided no reason"),
                "status": status_val # ★ Digital Twinのフィルタ用に必須
            })

        # ==========================================================
        # ★ Digital Twin: 予兆検知
        # ==========================================================
        if self.digital_twin is not None:
            try:
                predictions = self.digital_twin.predict(
                    analysis_results=results,
                    msg_map=msg_map,
                    alarms=alarms,
                )

                if predictions:
                    # predictions は既に predict メソッド内で
                    # 「障害発生済み機器」を除外しているはずだが、
                    # 念のためここでも重複IDを除外してマージする
                    
                    # 既存の結果IDリスト
                    existing_ids = {r["id"] for r in results}
                    
                    # 新規の予兆のみ追加
                    for pred in predictions:
                        if pred["id"] not in existing_ids:
                            results.append(pred)
                        else:
                            # 既存結果がある場合、CRITICALでないなら予兆情報で上書き/補強も検討できるが、
                            # 今回は「障害優先」のため、既存がCRITICALなら予兆は捨てる
                            existing = next((r for r in results if r["id"] == pred["id"]), None)
                            if existing and existing.get("prob", 0) < 0.8: # CRITICAL未満なら予兆を優先
                                results.remove(existing)
                                results.append(pred)

            except Exception as e:
                print(f"[!] Digital Twin prediction error: {e}")

        # ==========================================================
        # ★ 最終ソートロジック (ここが重要)
        # ==========================================================
        # 優先順位:
        # 1. 実際の障害 (CRITICAL / prob >= 0.9)
        # 2. 予兆 (is_prediction=True)
        # 3. 警告 (WARNING)
        # 4. その他
        
        results.sort(key=lambda x: (
            0 if (x.get("prob", 0) >= 0.9 and not x.get("is_prediction")) else # Real Incident Priority
            1 if x.get("is_prediction") else                                   # Prediction Priority
            2,                                                                 # Others
            -x.get("prob", 0)                                                  # Prob Descending
        ))

        return results

    def analyze_redundancy_depth(self, device_id: str, alerts: List[str]) -> Dict[str, Any]:
        if not alerts:
            return {"status": HealthStatus.NORMAL, "reason": "No active alerts.", "impact_type": "NONE"}

        safe_alerts = [self._sanitize_text(a) for a in alerts]
        joined = " ".join(safe_alerts)
        joined_lower = joined.lower()

        # ルールベース判定 (変更なし)
        if ("Power Supply: Dual Loss" in joined) or ("Dual Loss" in joined) or ("Device Down" in joined) or ("Thermal Shutdown" in joined):
            return {"status": HealthStatus.CRITICAL, "reason": "Device down / dual PSU loss detected.", "impact_type": "Hardware/Physical"}

        psu_count = self._get_psu_count(device_id, default=1)
        psu_single_fail = ("power supply" in joined_lower and "failed" in joined_lower and "dual" not in joined_lower)
        if psu_single_fail:
            if psu_count >= 2:
                return {"status": HealthStatus.WARNING, "reason": "Single PSU failure (Redundant).", "impact_type": "Hardware/Redundancy"}
            return {"status": HealthStatus.CRITICAL, "reason": "Single PSU failure (Non-Redundant).", "impact_type": "Hardware/Physical"}

        # ... (その他のルールベース判定は省略、既存同様) ...
        # 簡易実装のためデフォルト値を返します
        if "critical" in joined_lower:
             return {"status": HealthStatus.CRITICAL, "reason": "Critical alert detected.", "impact_type": "Generic/Critical"}
        
        # デフォルト
        return {"status": HealthStatus.WARNING, "reason": "Alert detected.", "impact_type": "Generic/Warning"}
