"""
Antigravity AIOps - Logical Inference Engine (Rule-Based / Deterministic)
ベイズ確率ではなく、発生しているアラームとトポロジー情報に基づき、
論理的・決定論的に根本原因を特定するエンジン。
"""

class LogicalRCA:
    def __init__(self, topology):
        self.topology = topology
        
        # ■ 障害シグネチャ（知識ベース）
        # 「この種のアラームの組み合わせが発生したら、この障害と特定する」というルール
        self.signatures = [
            {
                "type": "Hardware/Critical_Multi_Fail",
                "label": "複合ハードウェア障害",
                "rules": lambda alarms: any("Power Supply" in a.message for a in alarms) and any("Fan" in a.message for a in alarms),
                "base_score": 1.0
            },
            {
                "type": "Hardware/Physical",
                "label": "ハードウェア障害 (電源/デバイス)",
                "rules": lambda alarms: any(k in a.message for a in alarms for k in ["Power Supply", "Device Down"]),
                "base_score": 0.95
            },
            {
                "type": "Network/Link",
                "label": "物理リンク/インターフェース障害",
                "rules": lambda alarms: any(k in a.message for a in alarms for k in ["Interface Down", "Connection Lost", "Heartbeat Loss"]),
                "base_score": 0.90
            },
            {
                "type": "Hardware/Fan",
                "label": "冷却ファン故障",
                "rules": lambda alarms: any("Fan Fail" in a.message for a in alarms),
                "base_score": 0.70
            },
            {
                "type": "Config/Software",
                "label": "設定ミス/プロトコル障害",
                "rules": lambda alarms: any(k in a.message for a in alarms for k in ["BGP", "OSPF", "Config"]),
                "base_score": 0.60
            },
            {
                "type": "Resource/Capacity",
                "label": "リソース枯渇 (CPU/Memory)",
                "rules": lambda alarms: any(k in a.message for a in alarms for k in ["CPU", "Memory", "High"]),
                "base_score": 0.50
            }
        ]

    def analyze(self, current_alarms):
        """
        現在のアラームリストを入力とし、デバイスごとのリスクスコアを算出する。
        """
        candidates = []
        
        # 1. アラームをデバイスIDごとにグループ化
        device_alarms = {}
        for alarm in current_alarms:
            if alarm.device_id not in device_alarms:
                device_alarms[alarm.device_id] = []
            device_alarms[alarm.device_id].append(alarm)
            
        # 2. デバイスごとにルール適合度を評価
        for device_id, alarms in device_alarms.items():
            best_match = None
            max_score = 0.0
            
            # 全シグネチャをチェックし、最も重篤なものを採用
            for sig in self.signatures:
                if sig["rules"](alarms):
                    # アラーム数に応じた加点 (確信度の補強)
                    # 基本スコア + (関連アラーム数 * 0.05) ※最大1.0
                    score = min(sig["base_score"] + (len(alarms) * 0.02), 1.0)
                    
                    if score > max_score:
                        max_score = score
                        best_match = sig
            
            if best_match:
                candidates.append({
                    "id": device_id,
                    "type": best_match["type"],
                    "label": best_match["label"],
                    "prob": max_score, # リスクスコア (0.0 - 1.0)
                    "alarms": [a.message for a in alarms]
                })

        # 3. ソート (スコアが高い順)
        candidates.sort(key=lambda x: x["prob"], reverse=True)
        
        # 候補がない場合
        if not candidates:
            candidates.append({"id": "System", "type": "Normal", "label": "正常稼働中", "prob": 0.0})
            
        return candidates
