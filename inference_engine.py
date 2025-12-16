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
    LogicalRCA:
      - LLM によるコンフィグ解釈（ベンダ差分の吸収）
      - トポロジー文脈（親子関係）を用いたカスケード抑制
      - “冗長が効いてるなら黄色、止まってるなら赤” を優先
    """

    def __init__(self, topology, config_dir: str = "./configs"):
        """
        :param topology: トポロジー辞書オブジェクト または JSONファイルパス(str)
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

    # ----------------------------
    # Topology helpers
    # ----------------------------
    def _get_device_info(self, device_id: str) -> Any:
        return self.topology.get(device_id, {})

    def _get_parent_id(self, device_id: str) -> Optional[str]:
        info = self._get_device_info(device_id)
        if hasattr(info, "parent_id"):
            return getattr(info, "parent_id")
        if isinstance(info, dict):
            return info.get("parent_id")
        return None

    def _get_metadata(self, device_id: str) -> Dict[str, Any]:
        info = self._get_device_info(device_id)
        if hasattr(info, "metadata"):
            md = getattr(info, "metadata")
            return md if isinstance(md, dict) else {}
        if isinstance(info, dict):
            md = info.get("metadata", {})
            return md if isinstance(md, dict) else {}
        return {}

    def _get_psu_count(self, device_id: str, default: int = 1) -> int:
        """
        topology.json の metadata.hw_inventory.psu_count を参照。
        存在しない場合は default（1）を返す。
        """
        md = self._get_metadata(device_id)
        hw = md.get("hw_inventory", {}) if isinstance(md, dict) else {}
        try:
            psu = hw.get("psu_count", default) if isinstance(hw, dict) else default
            return int(psu)
        except Exception:
            return default

    # ----------------------------
    # LLM init
    # ----------------------------
    def _ensure_api_configured(self) -> bool:
        """APIキーの設定を確認・初期化（遅延評価）"""
        if self._api_configured:
            return True

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return False

        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
            self._api_configured = True
            return True
        except Exception as e:
            print(f"[!] API Configuration Error: {e}")
            return False

    # ----------------------------
    # IO
    # ----------------------------
    def _load_topology(self, path: str) -> Dict:
        """JSONファイルからトポロジー情報を読み込む"""
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _read_config(self, device_id: str) -> str:
        """デバイスIDに対応するコンフィグファイルを読み込む"""
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
        """機密情報のサニタイズ処理"""
        text = re.sub(r'(encrypted-password\s+)"[^"]+"', r'\1"********"', text)
        text = re.sub(r"(password|secret)\s+(\d)\s+\S+", r"\1 \2 ********", text)
        text = re.sub(r"(username\s+\S+\s+secret)\s+\d\s+\S+", r"\1 5 ********", text)
        text = re.sub(r"(snmp-server community)\s+\S+", r"\1 ********", text)
        return text

    # ==========================================================
    # Public API
    # ==========================================================
    def analyze(self, alarms: List) -> List[Dict[str, Any]]:
        """
        アラームリストを分析して根本原因候補を返す（辞書形式）
        """
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

        alarmed_ids = set(msg_map.keys())

        def parent_is_alarmed(dev: str) -> bool:
            p = self._get_parent_id(dev)
            return bool(p and (p in alarmed_ids))

        results: List[Dict[str, Any]] = []

        for device_id, messages in msg_map.items():
            if any("Unreachable" in m for m in messages) and parent_is_alarmed(device_id):
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
        冗長性深度とサービス影響を判定する。

        ローカル安全ルールで“確定できるもの”は確定し、
        それ以外は LLM に委譲する（APIキーがある場合）。

        # NOTE（日本語訳）:
        # このルールは運用上の安全性を保証するために存在します。
        # 将来、インベントリ＋過去の証跡が十分に利用できるようになったら、
        # この判断はAIに委譲すべきです。
        """
        if not alerts:
            return {
                "status": HealthStatus.NORMAL,
                "reason": "No active alerts detected.",
                "impact_type": "NONE"
            }

        safe_alerts = [self._sanitize_text(a) for a in alerts]
        joined = " ".join(safe_alerts)

        if ("Dual Loss" in joined) or ("Device Down" in joined) or ("Thermal Shutdown" in joined):
            return {
                "status": HealthStatus.CRITICAL,
                "reason": "Dual PSU loss / device down detected (local safety rule).",
                "impact_type": "Hardware/Physical"
            }

        psu_count = self._get_psu_count(device_id, default=1)
        psu_single_fail = (
            ("Power Supply" in joined and "Failed" in joined and "Dual" not in joined)
            or ("PSU" in joined and "Fail" in joined and "Dual" not in joined)
        )
        if psu_single_fail:
            if psu_count >= 2:
                return {
                    "status": HealthStatus.WARNING,
                    "reason": f"Single PSU failure with redundancy (psu_count={psu_count}) (local safety rule).",
                    "impact_type": "Hardware/Redundancy"
                }
            return {
                "status": HealthStatus.CRITICAL,
                "reason": f"Single PSU failure without redundancy (psu_count={psu_count}) (local safety rule).",
                "impact_type": "Hardware/Physical"
            }

        if not self._ensure_api_configured():
            return {
                "status": HealthStatus.WARNING,
                "reason": "API key not configured. Manual analysis required.",
                "impact_type": "UNKNOWN"
            }

        metadata = self._get_metadata(device_id)
        raw_config = self._read_config(device_id)
        safe_config = self._sanitize_text(raw_config)

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
            response = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
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

            return {
                "status": health_status,
                "reason": result_json.get("reason", "AI provided no reason"),
                "impact_type": result_json.get("impact_type", "UNKNOWN")
            }

        except Exception as e:
            print(f"[!] AI Inference Error: {e}")
            return {
                "status": HealthStatus.WARNING,
                "reason": f"AI Analysis Failed: {str(e)}",
                "impact_type": "AI_ERROR"
            }
