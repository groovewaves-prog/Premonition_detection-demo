# utils/const.py
class ImpactLevel:
    """影響度レベル定義"""
    COMPLETE_OUTAGE = 100
    CRITICAL = 90
    DEGRADED_HIGH = 80
    DEGRADED_MID = 70
    DOWNSTREAM = 50
    LOW_PRIORITY = 20

SCENARIO_IMPACT_MAP = {
    "正常稼働": 0,
    "WAN全回線断": ImpactLevel.COMPLETE_OUTAGE,
    "[WAN] 電源障害：両系": ImpactLevel.COMPLETE_OUTAGE,
    "[L2SW] 電源障害：両系": ImpactLevel.COMPLETE_OUTAGE,
    "[Core] 両系故障": ImpactLevel.CRITICAL,
    "[FW] 電源障害：両系": ImpactLevel.CRITICAL,
    "[FW] 電源障害：片系": ImpactLevel.DEGRADED_HIGH,
    "FW片系障害": ImpactLevel.DEGRADED_HIGH,
    "[WAN] 電源障害：片系": ImpactLevel.DEGRADED_MID,
    "[L2SW] 電源障害：片系": ImpactLevel.DEGRADED_MID,
    "L2SWサイレント障害": ImpactLevel.DEGRADED_HIGH,
    "[WAN] BGPルートフラッピング": ImpactLevel.DEGRADED_HIGH,
    "[WAN] FAN故障": ImpactLevel.DEGRADED_MID,
    "[FW] FAN故障": ImpactLevel.DEGRADED_MID,
    "[L2SW] FAN故障": ImpactLevel.DEGRADED_MID,
    "[WAN] メモリリーク": ImpactLevel.DEGRADED_MID,
    "[FW] メモリリーク": ImpactLevel.DEGRADED_MID,
    "[L2SW] メモリリーク": ImpactLevel.DEGRADED_MID,
    "[WAN] 複合障害：電源＆FAN": ImpactLevel.DEGRADED_HIGH,
    "[Complex] 同時多発：FW & AP": ImpactLevel.DEGRADED_HIGH,
}

SCENARIO_MAP = {
    "基本・広域障害": [
        "正常稼働",
        "1. WAN全回線断",
        "2. FW片系障害",
        "3. L2SWサイレント障害"
    ],
    "WAN Router": [
        "4. [WAN] 電源障害：片系",
        "5. [WAN] 電源障害：両系",
        "6. [WAN] BGPルートフラッピング",
        "7. [WAN] FAN故障",
        "8. [WAN] メモリリーク"
    ],
    "Firewall": [
        "9. [FW] 電源障害：片系",
        "10. [FW] 電源障害：両系",
        "11. [FW] FAN故障",
        "12. [FW] メモリリーク"
    ],
    "L2 Switch": [
        "13. [L2SW] 電源障害：片系",
        "14. [L2SW] 電源障害：両系",
        "15. [L2SW] FAN故障",
        "16. [L2SW] メモリリーク"
    ],
    "複合・その他": [
        "17. [WAN] 複合障害：電源＆FAN",
        "18. [Complex] 同時多発：FW & AP"
    ]
}
