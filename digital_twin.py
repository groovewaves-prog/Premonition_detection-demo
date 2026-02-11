# -*- coding: utf-8 -*-
"""
digital_twin.py (Universal Edition - Final Fix)
===============================================
AIOps Digital Twin Engine

[修正履歴]
 - Fix 1: prob を str("72%") から float(0.72) に変更 (app.py互換性確保)
 - Fix 2: predict() に alarms 引数を追加 (inference_engine呼び出し互換性確保)
 - Fix 3: 候補選定を厳格化 (確率0.45-0.85のWarningのみ対象、Critical除外)
 - Fix 4: Enum比較バグ修正 (status文字列判定ではなくprob数値判定へ変更)

設計方針:
 - UX変更なし: inference_engine から呼び出し、予兆データを注入する。
 - Hybrid Matching: キーワード一致(高速) + Embedding類似度(柔軟)
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass

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
# Escalation Rules (ドメイン知識)
# ==========================================================
@dataclass
class EscalationRule:
    pattern: str
    semantic_phrases: List[str]
    escalated_state: str
    time_to_critical_min: int
    base_confidence: float
    category: str = "Generic"

ESCALATION_RULES: List[EscalationRule] = [
    EscalationRule(
        "memory",
        ["memory usage high", "high memory utilization", "memory threshold exceeded",
         "mbuf cluster limit reached", "memory pool depletion", "heap exhaustion",
         "malloc fail", "resource exhaustion"],
        "メモリ枯渇によるプロセスクラッシュ", 30, 0.85, "Software/Resource",
    ),
    EscalationRule(
        "cpu",
        ["cpu usage high", "high cpu utilization", "cpu threshold exceeded",
         "cpuhog", "cpu spike", "control plane overload", "process stuck"],
        "CPU枯渇によるコントロールプレーン停止", 20, 0.80, "Software/Resource",
    ),
    EscalationRule(
        "fan",
        ["fan failure", "fan malfunction", "cooling failure",
         "fan speed critical", "fan tray removed", "temperature high"],
        "冷却不能による熱暴走シャットダウン", 45, 0.70, "Hardware/Thermal",
    ),
    EscalationRule(
        "power",
        ["power supply failed", "psu failure", "power redundancy lost",
         "power feed interrupted", "input power absent"],
        "残存PSU障害時の完全電源喪失", 60, 0.60, "Hardware/Power",
    ),
    EscalationRule(
        "bgp",
        ["bgp flapping", "bgp neighbor down", "bgp session reset",
         "route oscillation", "prefix withdrawal"],
        "BGPセッション完全断による経路消失", 15, 0.80, "Network/Routing",
    ),
    EscalationRule(
        "ha",
        ["ha state degraded", "failover state changed", "standby not ready",
         "redundancy state change", "heartbeat lost"],
        "HA切替失敗（Active側完全ダウン時）", 30, 0.75, "Network/HA",
    ),
    EscalationRule(
        "heartbeat",
        ["heartbeat lost", "heartbeat failure", "keepalive timeout",
         "peer unreachable", "cluster communication lost"],
        "スプリットブレイン発生", 20, 0.70, "Network/HA",
    ),
    EscalationRule(
        "generic_error",
        ["error", "fail", "critical", "warning"],
        "サービス劣化の進行", 30, 0.50, "Generic"
    ),
]

# ==========================================================
# Digital Twin Engine
# ==========================================================
class DigitalTwinEngine:
    _model: Optional[Any] = None
    _rule_embeddings: Optional[Dict[str, Any]] = None
    _model_loaded: bool = False

    MIN_PREDICTION_CONFIDENCE = 0.50
    MAX_PROPAGATION_HOPS = 3
    HOP_DECAY_RATE = 0.10
    REDUNDANCY_DISCOUNT = 0.15
    SPOF_BOOST = 1.10
    EMBEDDING_THRESHOLD = 0.40

    def __init__(self, topology: Dict[str, Any], children_map: Optional[Dict[str, List[str]]] = None):
        self.topology = topology
        self.children_map = children_map or {}

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
            import os
            # モデルパス解決（エアギャップ対応）:
            #   1. 環境変数 DIGITAL_TWIN_MODEL_PATH (deploy_airgap.sh が設定)
            #   2. ローカルディレクトリ ./models/all-MiniLM-L6-v2
            #   3. HuggingFace名 (オンライン環境のみ)
            model_path = os.environ.get("DIGITAL_TWIN_MODEL_PATH")
            if not model_path or not os.path.isdir(model_path):
                local_candidate = os.path.join(os.path.dirname(__file__), "models", "all-MiniLM-L6-v2")
                if os.path.isdir(local_candidate):
                    model_path = local_candidate
                else:
                    model_path = "all-MiniLM-L6-v2"  # HuggingFace fallback
            logger.info(f"Loading embedding model from: {model_path}")
            cls._model = SentenceTransformer(model_path)
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

    def predict(self, analysis_results: List[Dict[str, Any]], msg_map: Dict[str, List[str]], alarms: Optional[List] = None) -> List[Dict[str, Any]]:
        predictions = []
        warning_seeds = [
            r for r in analysis_results
            if 0.45 <= float(r.get("prob", 0)) <= 0.85
            and r.get("id", "") != "SYSTEM"
        ]
        candidates = {r["id"] for r in warning_seeds}
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
            pred = {
                "id": dev_id,
                "label": f"🔮 [予兆] {rule.escalated_state}",
                "severity": "CRITICAL",
                "status": "CRITICAL",
                "prob": round(confidence, 2),
                "type": f"Predictive/{rule.category}",
                "tier": 1,
                "reason": (
                    f"【Digital Twin未来予測】\n"
                    f"現在: {msg} (Match: {quality:.2f})\n"
                    f"予測: {rule.time_to_critical_min}分後に深刻化の恐れ\n"
                    f"影響: {affected_str} が通信断になる可能性が高いです。\n"
                    f"(信頼度スコア: {confidence:.2f})"
                ),
                "is_prediction": True,
                "prediction_timeline": f"{rule.time_to_critical_min}分後",
                "prediction_affected_count": impact_count,
                "prediction_affected_devices": [d[0] for d in downstream],
                "prediction_confidence_factors": {
                    "base": rule.base_confidence,
                    "match_quality": quality,
                    "has_redundancy": bool(self.topology.get(dev_id, {}).get('redundancy_group')),
                    "is_spof": bool(self.children_map.get(dev_id) and not self.topology.get(dev_id, {}).get('redundancy_group')),
                    "downstream_count": impact_count,
                },
            }
            predictions.append(pred)
            processed_devices.add(dev_id)
        return predictions
