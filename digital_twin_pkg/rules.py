import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class EscalationRule:
    pattern: str
    semantic_phrases: List[str]
    escalated_state: str
    time_to_critical_min: int
    early_warning_hours: int
    base_confidence: float
    category: str = "Generic"
    embedding_threshold: float = 0.40
    
    requires_trend: bool = False
    trend_metric_regex: str = "" 
    trend_min_slope: float = 0.0
    trend_window_hours: int = 24
    
    requires_volatility: bool = False
    max_volatility: float = 0.0 
    volatility_window_hours: int = 3
    
    metric_name: str = "value"
    risk_bias_offset: float = 0.0
    ignore_phrases: List[str] = field(default_factory=list)
    
    recommended_actions: List[Dict[str, str]] = field(default_factory=list)
    runbook_url: str = ""
    criticality: str = "standard"
    
    # Persisted thresholds
    paging_threshold: Optional[float] = None
    logging_threshold: Optional[float] = None
    
    _compiled_regex: Any = field(init=False, repr=False, default=None)
    _metric_regex: Any = field(init=False, repr=False, default=None)

    def __post_init__(self):
        escaped = re.escape(self.pattern)
        try:
            self._compiled_regex = re.compile(f"(?<![a-zA-Z0-9_]){escaped}(?![a-zA-Z0-9_])", re.IGNORECASE)
        except:
            self._compiled_regex = re.compile(re.escape(self.pattern), re.IGNORECASE)
            
        if self.trend_metric_regex:
            try:
                self._metric_regex = re.compile(self.trend_metric_regex, re.IGNORECASE)
            except: pass

# --- Enhanced Default Rules for Simulation Scenarios ---
DEFAULT_RULES = [
    # 1. Optical Decay (光減衰)
    EscalationRule("optical", ["rx power", "optical signal", "transceiver", "light level", "dbm"], 
                   "光信号劣化によるリンクダウン", 60, 336, 0.95, "Hardware/Optical", 0.45,
                   requires_trend=True, trend_metric_regex=r"([-\d\.]+)\s*dBm", trend_min_slope=-0.05, 
                   metric_name="rx_power_dbm", risk_bias_offset=-0.2, criticality="critical",
                   recommended_actions=[{"title": "SFPモジュールの予備交換", "effect": "トランシーバ故障による劣化を解消"}, {"title": "光ファイバー清掃", "effect": "汚れによる減衰を回復"}],
                   runbook_url="https://wiki.company.com/ops/optical_troubleshooting"),
    
    # 2. Microburst (パケット破棄) -> Queue Drops
    EscalationRule("microburst", ["queue drops", "buffer overflow", "output drops", "asic_error", "qos-4-policer"], 
                   "マイクロバーストによるバッファ枯渇", 15, 24, 0.85, "Network/QoS", 0.45,
                   requires_volatility=True, max_volatility=100.0,
                   recommended_actions=[{"title": "QoSポリシーの調整", "effect": "バッファ割り当ての最適化"}, {"title": "帯域増強", "effect": "物理的な輻輳解消"}],
                   runbook_url="https://wiki.company.com/ops/qos_tuning"),

    # 3. Route Instability (経路揺らぎ) -> BGP/OSPF
    EscalationRule("route_instability", ["route instability", "bgp neighbor", "neighbor down", "route updates", "retransmission"], 
                   "経路不安定による大規模通信断", 30, 48, 0.90, "Network/Routing", 0.45,
                   recommended_actions=[{"title": "BGPフラップダンピングの確認", "effect": "不安定な経路の抑制"}, {"title": "ルート広報の検証", "effect": "誤った経路情報の修正"}]),

    # 4. STP Loop
    EscalationRule("stp_loop", ["stp loop", "tcn received", "blocking port"], 
                   "L2ループによるブロードキャストストーム", 5, 24, 0.95, "Network/L2", 0.42,
                   risk_bias_offset=-0.5, criticality="critical",
                   recommended_actions=[{"title": "該当ポートのshutdown", "effect": "ループ源を物理的に遮断"}]),

    # 5. Resource / Generic
    EscalationRule("memory_leak", ["memory usage high", "malloc fail"], 
                   "メモリ枯渇によるシステムクラッシュ", 180, 336, 0.85, "Software/Resource", 0.38,
                   requires_trend=True, trend_metric_regex=r"usage (\d+)%", trend_min_slope=1.0, 
                   metric_name="memory_usage_pct",
                   recommended_actions=[{"title": "計画的再起動(Reload)", "effect": "メモリ領域を開放"}]),
    
    EscalationRule("generic_error", ["error", "fail", "critical", "warning"], 
                   "未分類のサービス劣化", 30, 24, 0.50, "Generic", 0.35,
                   requires_volatility=True, trend_metric_regex=r"time=(\d+)ms", max_volatility=50.0, 
                   metric_name="latency_ms", risk_bias_offset=0.5,
                   recommended_actions=[{"title": "ログ詳細調査", "effect": "原因特定"}]),
    
    EscalationRule("analysis_signal", ["analysis_anomaly"], 
                   "AI分析による異常検知", 30, 24, 0.60, "Generic", 0.5,
                   risk_bias_offset=0.2,
                   recommended_actions=[{"title": "詳細分析確認", "effect": "状況把握"}]),
]

MAINTENANCE_SIGNATURES = [
    (r"administratively down", 0.9), (r"ifAdminStatus.*down", 0.9), 
    (r"reload requested by", 0.9), (r"system reboot", 0.8), 
    (r"image upgrade", 0.9), (r"config.*change", 0.7), (r"commit.*success", 0.6)
]
