# -*- coding: utf-8 -*-
"""
digital_twin.py (Universal Edition - Final Fix v2.1)
====================================================
AIOps Digital Twin Engine

[修正履歴]
 - Fix: 候補選定ロジックを拡張。
        これまでは Warning (prob >= 0.45) の機器のみを対象としていたが、
        INFOレベルの予兆シグナル (Weak Signal) も検知できるよう、
        「ログメッセージが存在する全機器」もスキャン対象に追加。

設計方針:
 - UX変更なし: inference_engine から呼び出し、予兆データを注入する。
 - Hybrid Matching: キーワード一致(高速) + Embedding類似度(柔軟)
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass

# --- 依存ライブラリのインポート ---
try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_BERT = True
except ImportError:
    HAS_BERT = False

logger = logging.getLogger(__name__)

# ==========================================================
# Escalation Rules (ドメイン知識 - 科学的根拠版)
# ==========================================================
@dataclass
class EscalationRule:
    """
    WARNING がどう CRITICAL に進行するかのルール定義
    
    Attributes:
        time_to_critical_min: 急性期の進行速度（発症してから全断するまでの時間）
        early_warning_hours:  早期予兆検知可能時間（TCN増加やCRCエラーなど、前兆が出始める最早時間）
    """
    pattern: str                    # キーワード (lowercase)
    semantic_phrases: List[str]     # Embedding マッチ用フレーズ群
    escalated_state: str            # 進行後の状態
    time_to_critical_min: int       # 急性期 (分)
    early_warning_hours: int        # 早期予兆 (時間)
    base_confidence: float          # 基礎信頼度
    category: str = "Generic"       # 分類

# 科学的知見・統計データに基づくパラメータ設定
ESCALATION_RULES: List[EscalationRule] = [
    # --- Network / L2 (Critical & Fast) ---
    EscalationRule(
        "stp_loop",
        ["stp loop", "spanning tree topology change", "tcn received", "bpdu guard", 
         "blocking port", "loop guard", "root bridge change", "excessive broadcasts"],
        "L2ループによるブロードキャストストーム（全断）", 
        time_to_critical_min=5, early_warning_hours=24, base_confidence=0.95,
        category="Network/L2"
    ),
    EscalationRule(
        "mac_flap",
        ["mac flapping", "host moving", "mac address move", "mac table overflow", 
         "learning disable", "duplicate mac"],
        "MACテーブル不安定化によるフレーム消失", 
        time_to_critical_min=10, early_warning_hours=24, base_confidence=0.90,
        category="Network/L2"
    ),
    EscalationRule(
        "arp_storm",
        ["arp storm", "duplicate ip", "gratuitous arp", "arp table overflow", 
         "arp rate limit", "neighbor table full"],
        "ARPテーブル汚染による通信断", 
        time_to_critical_min=10, early_warning_hours=12, base_confidence=0.85,
        category="Network/L2"
    ),

    # --- Network / L3 & Routing ---
    EscalationRule(
        "bgp_flap",
        ["bgp flapping", "bgp neighbor down", "bgp session reset", "route oscillation", 
         "prefix withdrawal", "hold timer expired", "notification received"],
        "BGPセッション不安定化による大規模経路消失", 
        time_to_critical_min=15, early_warning_hours=48, base_confidence=0.90,
        category="Network/Routing"
    ),
    EscalationRule(
        "ospf_adj",
        ["ospf adjacency down", "neighbor down", "dead timer expired", "lsa age", 
         "database description", "retransmission limit", "spf calculation"],
        "OSPFネイバー喪失によるルーティングループ/ブラックホール", 
        time_to_critical_min=15, early_warning_hours=24, base_confidence=0.85,
        category="Network/Routing"
    ),
    
    # --- Network / HA ---
    EscalationRule(
        "ha_split",
        ["ha state degraded", "failover state changed", "standby not ready", 
         "heartbeat lost", "split brain", "cluster link down"],
        "HA同期不全によるスプリットブレイン発生", 
        time_to_critical_min=30, early_warning_hours=48, base_confidence=0.85,
        category="Network/HA"
    ),

    # --- Network / QoS & Performance ---
    EscalationRule(
        "bandwidth",
        ["bandwidth exceeded", "interface congestion", "shaping active", "policing drop", 
         "tail drop", "queue full", "output drops"],
        "帯域飽和によるサービス品質劣化（遅延・パケットロス）", 
        time_to_critical_min=20, early_warning_hours=72, base_confidence=0.80,
        category="Network/QoS"
    ),
    EscalationRule(
        "drop_error",
        ["input errors", "crc error", "symbol error", "runts", "giants", 
         "interface resets", "fcs error"],
        "物理回線品質劣化によるスループット低下", 
        time_to_critical_min=30, early_warning_hours=168, base_confidence=0.75,
        category="Network/Interface"
    ),

    # --- Network / Services ---
    EscalationRule(
        "ntp_drift",
        ["ntp unsynchronized", "stratum change", "peer unreachable", "time drift", 
         "clock offset", "leap second"],
        "時刻不整合によるログ不全・認証エラー・証明書無効化", 
        time_to_critical_min=120, early_warning_hours=168, base_confidence=0.70,
        category="Network/Service"
    ),
    EscalationRule(
        "dhcp_dns",
        ["dhcp pool exhausted", "no ip address available", "dns timeout", "nxdomain", 
         "server not responding", "discovery failed"],
        "新規クライアントのネットワーク接続不可", 
        time_to_critical_min=30, early_warning_hours=48, base_confidence=0.80,
        category="Network/Service"
    ),

    # --- Hardware / Environmental ---
    EscalationRule(
        "optical",
        ["rx power low", "optical signal degradation", "light level warning", 
         "transceiver threshold", "dbm low", "link fluctuation"],
        "光信号劣化による突然のリンクダウン", 
        time_to_critical_min=60, early_warning_hours=336, base_confidence=0.90,
        category="Hardware/Optical"
    ),
    EscalationRule(
        "temperature",
        ["temperature high", "overheat", "thermal threshold", "intake air temp", 
         "exhaust temp", "sensor alarm"],
        "熱暴走による緊急シャットダウン", 
        time_to_critical_min=30, early_warning_hours=48, base_confidence=0.85,
        category="Hardware/Thermal"
    ),
    EscalationRule(
        "fan_fail",
        ["fan failure", "fan malfunction", "fan speed low", "fan tray removed"],
        "冷却能力喪失による温度上昇", 
        time_to_critical_min=45, early_warning_hours=72, base_confidence=0.80,
        category="Hardware/Thermal"
    ),
    EscalationRule(
        "power_quality",
        ["ups on battery", "input voltage low", "pdu alarm", "redundancy lost", 
         "power supply failed", "psu failure"],
        "電源供給不安定による予期せぬ再起動", 
        time_to_critical_min=15, early_warning_hours=24, base_confidence=0.85,
        category="Hardware/Power"
    ),
    EscalationRule(
        "storage",
        ["flash error", "file system full", "nvram corruption", "disk fail", 
         "write protect", "read error", "smart error"],
        "ストレージ障害による設定喪失・起動不能", 
        time_to_critical_min=180, early_warning_hours=720, base_confidence=0.75,
        category="Hardware/Storage"
    ),

    # --- Software / Resources & Process ---
    EscalationRule(
        "memory_leak",
        ["memory usage high", "malloc fail", "memory pool depletion", "heap exhaustion", 
         "fragmentation", "leak detected"],
        "メモリ枯渇によるOOM Killer発動・システムクラッシュ", 
        time_to_critical_min=180, early_warning_hours=336, base_confidence=0.85,
        category="Software/Resource"
    ),
    EscalationRule(
        "cpu_load",
        ["cpu usage high", "cpu spike", "load average high", "control plane overload", 
         "process stuck", "interrupt storm"],
        "CPU枯渇による管理不能・プロトコルダウン", 
        time_to_critical_min=20, early_warning_hours=48, base_confidence=0.85,
        category="Software/Resource"
    ),
    EscalationRule(
        "process_crash",
        ["process terminated", "segmentation fault", "core dump", "watchdog timeout", 
         "daemon exit", "service restart"],
        "重要プロセス停止による制御機能喪失", 
        time_to_critical_min=10, early_warning_hours=24, base_confidence=0.90,
        category="Software/Process"
    ),

    # --- Security ---
    EscalationRule(
        "auth_failure",
        ["authentication failed", "radius timeout", "tacacs unreachable", "invalid user", 
         "login failed", "aaa server down"],
        "認証基盤障害による管理アクセス・ユーザー接続不能", 
        time_to_critical_min=15, early_warning_hours=12, base_confidence=0.80,
        category="Security/Auth"
    ),
    EscalationRule(
        "crypto_vpn",
        ["ike sa deleted", "ipsec phase1 failed", "certificate expired", "decryption error", 
         "vpn tunnel down", "proposal mismatch"],
        "VPN/暗号化トンネルの切断", 
        time_to_critical_min=60, early_warning_hours=720, base_confidence=0.80,
        category="Security/Crypto"
    ),

    # --- Fallback ---
    EscalationRule(
        "generic_error",
        ["error", "fail", "critical", "warning", "emergency", "alert"],
        "未分類のサービス劣化進行", 
        time_to_critical_min=30, early_warning_hours=24, base_confidence=0.50,
        category="Generic"
    ),
]

# ==========================================================
# Digital Twin Engine
# ==========================================================
class DigitalTwinEngine:
    """
    Digital Twin Engine for predictive fault analysis.
    """
    
    # --- Class-level cache (Singleton Pattern) ---
    _model: Optional[Any] = None
    _rule_embeddings: Optional[Dict[str, Any]] = None
    _model_loaded: bool = False

    # --- Configuration ---
    MIN_PREDICTION_CONFIDENCE = 0.40 
    MAX_PROPAGATION_HOPS = 3
    HOP_DECAY_RATE = 0.10
    REDUNDANCY_DISCOUNT = 0.15
    SPOF_BOOST = 1.10
    EMBEDDING_THRESHOLD = 0.40

    def __init__(self, topology: Dict[str, Any], children_map: Optional[Dict[str, List[str]]] = None):
        self.topology = topology
        self.children_map = children_map or {}

        # --- NetworkX グラフ構築 ---
        self.graph = None
        if HAS_NX:
            self.graph = nx.DiGraph()
            for node_id, attrs in topology.items():
                node_attrs = attrs if isinstance(attrs, dict) else vars(attrs)
                self.graph.add_node(node_id, **node_attrs)
                
                parent_id = node_attrs.get("parent_id")
                if parent_id and parent_id in topology:
                    self.graph.add_edge(parent_id, node_id, relation="downstream")
                    self.graph.add_edge(node_id, parent_id, relation="upstream")

        self._redundancy_groups = self._build_redundancy_map()
        self._ensure_model_loaded()

    def _build_redundancy_map(self) -> Dict[str, List[str]]:
        rg_map = {}
        for dev_id, info in self.topology.items():
            attrs = info if isinstance(info, dict) else vars(info)
            rg = attrs.get('redundancy_group')
            if rg:
                rg_map.setdefault(rg, []).append(dev_id)
        return rg_map

    @classmethod
    def _ensure_model_loaded(cls):
        if cls._model_loaded:
            return

        if not HAS_BERT:
            logger.warning("sentence-transformers not available. Semantic matching disabled.")
            cls._model_loaded = True
            return

        try:
            cls._model = SentenceTransformer('all-MiniLM-L6-v2')
            
            all_phrases = []
            phrase_to_rule_idx = []
            
            for idx, rule in enumerate(ESCALATION_RULES):
                for phrase in rule.semantic_phrases:
                    all_phrases.append(phrase)
                    phrase_to_rule_idx.append(idx)
            
            if all_phrases:
                embeddings = cls._model.encode(all_phrases, convert_to_numpy=True)
                cls._rule_embeddings = {
                    "vectors": embeddings,
                    "phrase_to_rule_idx": phrase_to_rule_idx,
                    "phrases": all_phrases
                }
            
            cls._model_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            cls._model = None
            cls._model_loaded = True

    # ----------------------------------------------------------
    # Hybrid Matching Logic
    # ----------------------------------------------------------
    def _match_rule(self, alarm_text: str) -> Tuple[Optional[EscalationRule], float]:
        text_lower = alarm_text.lower()
        for rule in ESCALATION_RULES:
            if rule.pattern in text_lower:
                return rule, 1.0
        
        if self._model and self._rule_embeddings:
            try:
                query_vec = self._model.encode([alarm_text], convert_to_numpy=True)
                rule_vecs = self._rule_embeddings["vectors"]
                
                similarities = np.dot(rule_vecs, query_vec.T).flatten()
                norms = np.linalg.norm(rule_vecs, axis=1) * np.linalg.norm(query_vec)
                norms = np.where(norms == 0, 1e-10, norms)
                cosine_sim = similarities / norms
                
                best_idx = np.argmax(cosine_sim)
                best_score = float(cosine_sim[best_idx])
                
                if best_score >= self.EMBEDDING_THRESHOLD:
                    rule_idx = self._rule_embeddings["phrase_to_rule_idx"][best_idx]
                    return ESCALATION_RULES[rule_idx], best_score
                    
            except Exception as e:
                logger.error(f"Embedding matching error: {e}")
        
        return None, 0.0

    # ----------------------------------------------------------
    # Graph & Reliability Logic
    # ----------------------------------------------------------
    def _get_downstream_impact(self, root_id: str) -> List[Tuple[str, int]]:
        impacts = []
        if not self.graph or root_id not in self.graph:
            return impacts
            
        try:
            def downstream_filter(u, v):
                return self.graph[u][v].get("relation") == "downstream"
            
            subgraph = nx.subgraph_view(self.graph, filter_edge=downstream_filter)
            tree = nx.bfs_tree(subgraph, root_id, depth_limit=self.MAX_PROPAGATION_HOPS)
            
            for node in tree:
                if node == root_id: continue
                dist = nx.shortest_path_length(subgraph, root_id, node)
                impacts.append((node, dist))
                
        except Exception as e:
            logger.error(f"Graph traversal error: {e}")
            
        return impacts

    def _calculate_confidence(self, rule: EscalationRule, device_id: str, match_quality: float) -> float:
        attrs = self.topology.get(device_id, {})
        if not isinstance(attrs, dict): attrs = vars(attrs)
        
        rg = attrs.get('redundancy_group')
        has_redundancy = False
        if rg and len(self._redundancy_groups.get(rg, [])) > 1:
            has_redundancy = True
            
        is_spof = False
        children = self.children_map.get(device_id, [])
        if children and not has_redundancy:
            is_spof = True
            
        confidence = rule.base_confidence
        confidence *= (0.8 + 0.2 * match_quality)
        
        if has_redundancy:
            confidence *= (1.0 - self.REDUNDANCY_DISCOUNT)
            
        if is_spof:
            confidence *= self.SPOF_BOOST
            
        return min(0.99, max(0.1, confidence))

    # ----------------------------------------------------------
    # Main Prediction Method
    # ----------------------------------------------------------
    def predict(self, 
                analysis_results: List[Dict[str, Any]], 
                msg_map: Dict[str, List[str]], 
                alarms: Optional[List] = None) -> List[Dict[str, Any]]:
        """
        予兆検知のメインプロセス
        """
        predictions = []
        
        # 【修正箇所】候補選定ロジックの拡張
        # 1. 既存の分析で Warning が出ている機器 (prob 0.45-0.85)
        warning_ids = {
            r["id"] for r in analysis_results
            if 0.45 <= float(r.get("prob", 0)) <= 0.85
            and r.get("id", "") != "SYSTEM"
        }
        
        # 2. メッセージ(ログ)を持っている全機器
        #    理由: "INFO"レベルの予兆シグナル(Weak Signal)は analysis_results では
        #          "Normal"(prob < 0.45) と判定されるため、warning_ids には含まれない。
        #          Digital Twin はこれらも含めてスキャンする必要がある。
        active_ids = set(msg_map.keys())
        
        # 候補の統合
        candidates = warning_ids.union(active_ids)
        
        processed_devices = set()
        
        for dev_id in candidates:
            if dev_id in processed_devices: continue
            
            messages = msg_map.get(dev_id, [])
            if not messages: continue
            msg = messages[0]
            
            rule, quality = self._match_rule(msg)
            if not rule: continue
            
            downstream = self._get_downstream_impact(dev_id)
            impact_count = len(downstream)
            
            confidence = self._calculate_confidence(rule, dev_id, quality)
            
            if confidence < self.MIN_PREDICTION_CONFIDENCE:
                continue
                
            affected_names = [d[0] for d in downstream[:3]]
            if impact_count > 3: affected_names.append(f"他{impact_count-3}台")
            affected_str = ", ".join(affected_names) if affected_names else "配下なし"
            
            # ナラティブの強化: 「早期予兆」と「急性期」の2軸で表現
            if rule.early_warning_hours >= 24:
                early_str = f"最大 {rule.early_warning_hours // 24}日前"
            else:
                early_str = f"最大 {rule.early_warning_hours}時間前"

            pred = {
                "id": dev_id,
                "label": f"🔮 [予兆] {rule.escalated_state}",
                "severity": "CRITICAL", 
                "status": "CRITICAL",
                "prob": round(confidence, 2),
                "type": f"Predictive/{rule.category}",
                "tier": 1,
                "reason": (
                    f"【Digital Twin未来予測 (Predictive Maintenance)】\n"
                    f"・観測: {msg} (Match: {quality:.2f})\n"
                    f"・早期予兆: {early_str} から検知可能なパターン\n"
                    f"・急性期進行: 発症後 {rule.time_to_critical_min}分 で深刻化する恐れ\n"
                    f"・影響: {affected_str} が連鎖的に通信断になります。\n"
                    f"・推奨: メンテナンスウィンドウでの予防交換/対応\n"
                    f"(信頼度スコア: {confidence:.2f})"
                ),
                "is_prediction": True,
                # UI表示用の生データも渡しておく
                "prediction_early_warning_hours": rule.early_warning_hours,
                "prediction_time_to_critical_min": rule.time_to_critical_min
            }
            
            predictions.append(pred)
            processed_devices.add(dev_id)
            
        return predictions

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    topology = {"FW01": {"parent_id": None, "redundancy_group": "FW_HA"}, "SW01": {"parent_id": "FW01"}}
    msgs = {"FW01": ["Rx Power -24.8 dBm (Low)"]} 
    dummy_results = [{"id": "FW01", "prob": 0.10, "status": "NORMAL"}] # INFO相当
    
    dt = DigitalTwinEngine(topology)
    preds = dt.predict(dummy_results, msgs, alarms=[])
    
    for p in preds:
        print(f"Label: {p['label']}")
        print(f"Reason: {p['reason']}")
