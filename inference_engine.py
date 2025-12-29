import json
import os
import re
from enum import Enum
from typing import List, Dict, Any, Optional

import google.generativeai as genai

# ==========================================================
# AIOps health status
# ==========================================================
class HealthStatus(Enum):
    NORMAL = "GREEN"
    WARNING = "YELLOW"
    CRITICAL = "RED"


class LogicalRCA:
    """
    LogicalRCA (v5):
      - topology.json 形式（dict）でも data.py の NetworkNode 形式（object）でも動く
      - “冗長が効いてるなら黄色、止まってるなら赤” をローカル安全ルールで優先
      - “サイレント障害” は子の Connection Lost 集中から親を被疑箇所として推定（能動調査レポート付き）
    """

    # サイレント障害推定の閾値（運用で調整）
    SILENT_MIN_CHILDREN = 2
    SILENT_RATIO = 0.5

    def __init__(self, topology, config_dir: str = "./configs"):
        """
        :param topology: トポロジー辞書（device_id -> dict or NetworkNode） または JSONファイルパス(str)
        :param config_dir: コンフィグファイルが格納されているディレクトリ
        """
        if isinstance(topology, str):
            self.topology = self._load_topology(topology)
        elif isinstance(topology, dict):
            self.topology = topology
        else:
            raise ValueError("topology must be either a file path (str) or a dictionary")

        self.config_dir = config_dir
        self.model = None
        self._api_configured = False

        # parent -> [children...]
        self.children_map: Dict[str, List[str]] = {}
        for dev_id, info in self.topology.items():
            p = None
            if isinstance(info, dict):
                p = info.get("parent_id")
            else:
                # NetworkNode 等
                if hasattr(info, "parent_id"):
                    p = getattr(info, "parent_id")
                elif hasattr(info, "paren"):
                    # data.py の __repr__ が paren... で出るが属性名は parent_id のはず。念のため。
                    p = getattr(info, "paren", None)
            if p:
                self.children_map.setdefault(p, []).append(dev_id)

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
        # data.py の NetworkNode には get_metadata もある
        if hasattr(info, "get_metadata"):
            try:
                md = info.get_metadata("metadata", {})
                return md if isinstance(md, dict) else {}
            except Exception:
                return {}
        return {}

    def _get_psu_count(self, device_id: str, default: int = 1) -> int:
        """
        topology の metadata.hw_inventory.psu_count を優先参照。
        無い場合は metadata.redundancy_type == 'PSU' なら 2 を仮定。
        """
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

    # ----------------------------
    # Sanitization
    # ----------------------------
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
        """
        親自身にアラームが無いのに、配下の複数子が Connection Lost を出しているなら親を疑う。
        """
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
                    f"- Recommended checks:\n"
                    f"  1) Check uplink interface counters/errors on {parent_id}\n"
                    f"  2) Verify MAC table / ARP / STP state changes around incident time\n"
                    f"  3) Compare syslog/event logs for link flap, STP re-convergence\n"
                    f"  4) Run targeted ping/ARP from CORE side to affected APs\n"
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
                "reason": "No active alerts detected."
            }]

        msg_map: Dict[str, List[str]] = {}
        for a in alarms:
            msg_map.setdefault(a.device_id, []).append(a.message)

        # サイレント推定
        silent_suspects = self._detect_silent_failures(msg_map)

        # 親を分析対象に追加（疑似アラーム）
        for parent_id, info in silent_suspects.items():
            msg_map.setdefault(parent_id, []).append("Silent Failure Suspected (Derived from child Connection Lost)")

        alarmed_ids = set(msg_map.keys())

        def parent_is_alarmed(dev: str) -> bool:
            p = self._get_parent_id(dev)
            return bool(p and (p in alarmed_ids))

        def parent_is_silent_suspect(dev: str) -> bool:
            p = self._get_parent_id(dev)
            return bool(p and (p in silent_suspects))

        results: List[Dict[str, Any]] = []

        for device_id, messages in msg_map.items():

            # サイレント疑い配下の子は被疑（症状）扱い
            if parent_is_silent_suspect(device_id) and any(self._is_connection_loss(m) for m in messages):
                p = self._get_parent_id(device_id)
                results.append({
                    "id": device_id,
                    "label": " / ".join(messages),
                    "prob": 0.4,
                    "type": "Network/ConnectionLost",
                    "tier": 3,
                    "reason": f"Downstream symptom under suspected silent failure parent (parent={p})."
                })
                continue

            # 通常のカスケード抑制
            if any("unreachable" in m.lower() for m in messages) and parent_is_alarmed(device_id):
                p = self._get_parent_id(device_id)
                results.append({
                    "id": device_id,
                    "label": " / ".join(messages),
                    "prob": 0.2,
                    "type": "Network/Unreachable",
                    "tier": 3,
                    "reason": f"Downstream unreachable due to upstream alarm (parent={p})."
                })
                continue

            # 親がサイレント疑いの場合：黄色・高優先度で出す（赤にしない）
            if device_id in silent_suspects:
                info = silent_suspects[device_id]
                results.append({
                    "id": device_id,
                    "label": " / ".join(messages),
                    "prob": 0.8,
                    "type": "Network/SilentFailure",
                    "tier": 1,
                    "reason": f"Silent failure suspected: {info['evidence_count']}/{info['total_children']} children affected.",
                    "analyst_report": info["report"],
                    "auto_investigation": [
                        "Pull interface counters/errors (uplinks)",
                        "Check STP/MAC flaps",
                        "Ping/ARP reachability tests from upstream",
                        "Correlate syslog around incident time"
                    ]
                })
                continue

            analysis = self.analyze_redundancy_depth(device_id, messages)

            if analysis.get("impact_type") == "UNKNOWN" and "API key not configured" in analysis.get("reason", ""):
                prob = 0.5
                tier = 3
            else:
                if analysis["status"] == HealthStatus.CRITICAL:
                    prob = 0.9
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
                "reason": analysis.get("reason", "AI provided no reason")
            })

        results.sort(key=lambda x: x["prob"], reverse=True)
        return results

    # ==========================================================
    # Core decision function
    # ==========================================================
    def analyze_redundancy_depth(self, device_id: str, alerts: List[str]) -> Dict[str, Any]:
        """
        # NOTE:
        # This rule exists to guarantee operational safety.
        # In future versions, this decision SHOULD be delegated to AI
        # once inventory + historical evidence are fully available.

        # NOTE（日本語訳）:
        # このルールは運用上の安全性を保証するために存在します。
        # 将来、インベントリ＋過去の証跡が十分に利用できるようになったら、
        # この判断はAIに委譲すべきです。
        """
        if not alerts:
            return {"status": HealthStatus.NORMAL, "reason": "No active alerts detected.", "impact_type": "NONE"}

        safe_alerts = [self._sanitize_text(a) for a in alerts]
        joined = " ".join(safe_alerts)
        joined_lower = joined.lower()

        # 0) 停止系（赤）
        if ("Power Supply: Dual Loss" in joined) or ("Dual Loss" in joined) or ("Device Down" in joined) or ("Thermal Shutdown" in joined):
            return {"status": HealthStatus.CRITICAL, "reason": "Device down / dual PSU loss / thermal shutdown detected (local safety rule).", "impact_type": "Hardware/Physical"}

        # 1) 電源片系（黄色/赤）
        psu_count = self._get_psu_count(device_id, default=1)
        psu_single_fail = ("power supply" in joined_lower and "failed" in joined_lower and "dual" not in joined_lower) or ("psu" in joined_lower and "fail" in joined_lower and "dual" not in joined_lower)
        if psu_single_fail:
            if psu_count >= 2:
                return {"status": HealthStatus.WARNING, "reason": f"Single PSU failure with redundancy (psu_count={psu_count}) (local safety rule).", "impact_type": "Hardware/Redundancy"}
            return {"status": HealthStatus.CRITICAL, "reason": f"Single PSU failure without redundancy (psu_count={psu_count}) (local safety rule).", "impact_type": "Hardware/Physical"}

        # 2) FAN（黄色 / 熱兆候で赤）
        fan_fail = ("fan fail" in joined_lower) or ("fan" in joined_lower and "fail" in joined_lower)
        overheat_hint = ("high temperature" in joined_lower) or ("overheat" in joined_lower) or ("thermal" in joined_lower)
        if fan_fail:
            if overheat_hint:
                return {"status": HealthStatus.CRITICAL, "reason": "Fan failure with overheat/thermal symptom detected (local safety rule).", "impact_type": "Hardware/Physical"}
            return {"status": HealthStatus.WARNING, "reason": "Fan failure detected. Service likely continues but risk of thermal escalation (local safety rule).", "impact_type": "Hardware/Degraded"}

        # 3) メモリ（黄色 / OOMで赤）
        mem_symptom = ("memory high" in joined_lower) or ("memory leak" in joined_lower) or ("memory" in joined_lower and ("leak" in joined_lower or "high" in joined_lower))
        oom_hint = ("out of memory" in joined_lower) or ("oom" in joined_lower) or ("killed process" in joined_lower) or ("kernel panic" in joined_lower)
        if mem_symptom:
            if oom_hint:
                return {"status": HealthStatus.CRITICAL, "reason": "Memory leak/high with OOM/crash symptom detected (local safety rule).", "impact_type": "Software/Resource"}
            return {"status": HealthStatus.WARNING, "reason": "Memory high/leak symptom detected. Likely degraded but not down yet (local safety rule).", "impact_type": "Software/Resource"}

        # 4) LLM
        if not self._ensure_api_configured():
            return {"status": HealthStatus.WARNING, "reason": "API key not configured. Manual analysis required.", "impact_type": "UNKNOWN"}

        metadata = self._get_metadata(device_id)
        safe_config = self._sanitize_text(self._read_config(device_id))

        prompt = f"""
あなたはネットワーク運用のエキスパートAIです。
以下の情報に基づき、現在発生しているアラートが「サービス停止(CRITICAL)」を引き起こしているか、
それとも「冗長機能によりサービスは維持されている(WARNING)」状態かを判定してください。

### 対象デバイス
- Device ID: {device_id}
- Metadata: {json.dumps(metadata, ensure_ascii=False)}

### 設定ファイル (Config - Sanitized)
{safe_config}

### 発生中のアラートリスト
{json.dumps(safe_alerts, ensure_ascii=False)}

### 判定ルール（重要）
- “冗長が効いている（サービス継続）”と判断できる限り、CRITICALにしないこと。
- 逆に、サービス断（停止）が強く示唆される場合のみ CRITICAL にすること。

### 出力フォーマット
以下のJSON形式のみを出力してください（Markdownコードブロックは不要）。
{{
  "status": "NORMAL|WARNING|CRITICAL",
  "reason": "判定理由を簡潔に記述",
  "impact_type": "NONE|DEGRADED|REDUNDANCY_LOST|OUTAGE|UNKNOWN"
}}
"""

        try:
            response = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            result_json = json.loads(response_text)

            status_str = str(result_json.get("status", "CRITICAL")).upper()
            if status_str in ["GREEN", "NORMAL"]:
                health_status = HealthStatus.NORMAL
            elif status_str in ["YELLOW", "WARNING"]:
                health_status = HealthStatus.WARNING
            else:
                health_status = HealthStatus.CRITICAL

            return {"status": health_status, "reason": result_json.get("reason", "AI provided no reason"), "impact_type": result_json.get("impact_type", "UNKNOWN")}

        except Exception as e:
            print(f"[!] AI Inference Error: {e}")
            return {"status": HealthStatus.WARNING, "reason": f"AI Analysis Failed: {str(e)}", "impact_type": "AI_ERROR"}
