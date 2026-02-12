# -*- coding: utf-8 -*-
"""
digital_twin.py (v2.0 - Predictive Maintenance Edition)
========================================================
AIOps Digital Twin Engine

[修正履歴]
 - Fix 1: prob を str("72%") から float(0.72) に変更 (app.py互換性確保)
 - Fix 2: predict() に alarms 引数を追加 (inference_engine呼び出し互換性確保)
 - Fix 3: 候補選定を厳格化 (確率0.45-0.85のWarningのみ対象、Critical除外)
 - Fix 4: Enum比較バグ修正 (status文字列判定ではなくprob数値判定へ変更)
 - v2.0: EscalationRule に early_warning_hours 追加、11→21ルール拡充
         ナラティブ2軸化（早期予兆+急性期）、新規出力フィールド追加
         MIN_PREDICTION_CONFIDENCE 0.50→0.40

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
# Escalation Rules (ドメイン知識) - v2.0: L1-L7全レイヤーカバー
# ==========================================================
@dataclass
class EscalationRule:
    pattern: str
    semantic_phrases: List[str]
    escalated_state: str
    time_to_critical_min: int       # 急性期（分）- 既存
    early_warning_hours: int        # 早期予兆（時間）- ★新規
    base_confidence: float
    category: str = "Generic"

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
         "prefix withdrawal", "hold timer expired", "notification received",
         # ★ app.py注入互換: 旧route_flapフレーズ
         "route updates", "adjchange", "stability warning", "prefix flapping", "neighbor flap"],
        "BGPセッション不安定化による大規模経路消失",
        time_to_critical_min=15, early_warning_hours=48, base_confidence=0.90,
        category="Network/Routing"
    ),
    EscalationRule(
        "ospf_adj",
        ["ospf adjacency down", "neighbor down", "dead timer expired", "lsa age",
         "database description", "retransmission limit", "spf calculation",
         # ★ app.py注入互換: OSPF ADJCHANGE メッセージ対応
         "adjchange", "keepalive delayed", "keepalive timeout"],
        "OSPFネイバー喪失によるルーティングループ/ブラックホール",
        time_to_critical_min=15, early_warning_hours=24, base_confidence=0.85,
        category="Network/Routing"
    ),

    # --- Network / HA ---
    EscalationRule(
        "ha_split",
        ["ha state degraded", "failover state changed", "standby not ready",
         "heartbeat lost", "split brain", "cluster link down",
         # ★ 旧heartbeatルール統合
         "heartbeat failure", "keepalive timeout", "peer unreachable",
         "cluster communication lost", "redundancy state change"],
        "HA同期不全によるスプリットブレイン発生",
        time_to_critical_min=30, early_warning_hours=48, base_confidence=0.85,
        category="Network/HA"
    ),

    # --- Network / QoS & Performance ---
    EscalationRule(
        "bandwidth",
        ["bandwidth exceeded", "interface congestion", "shaping active", "policing drop",
         "tail drop", "queue full", "output drops",
         # ★ app.py注入互換: 旧dropフレーズ(QoS系)
         "microburst", "buffer overflow", "queue congestion", "burst traffic"],
        "帯域飽和によるサービス品質劣化（遅延・パケットロス）",
        time_to_critical_min=20, early_warning_hours=72, base_confidence=0.80,
        category="Network/QoS"
    ),
    EscalationRule(
        "drop_error",
        ["input errors", "crc error", "symbol error", "runts", "giants",
         "interface resets", "fcs error",
         # ★ app.py注入互換: 旧dropフレーズ(エラー系)
         "input queue drops", "packet drops detected", "asic error"],
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
         "transceiver threshold", "dbm low", "link fluctuation",
         # ★ app.py注入互換: 旧opticalフレーズ
         "rx power", "signal degrading", "signal degradation",
         "threshold violation", "sfp rx power"],
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
        ["fan failure", "fan malfunction", "fan speed low", "fan tray removed",
         # ★ 旧fanルール互換
         "cooling failure", "fan speed critical", "temperature high"],
        "冷却能力喪失による温度上昇",
        time_to_critical_min=45, early_warning_hours=72, base_confidence=0.80,
        category="Hardware/Thermal"
    ),
    EscalationRule(
        "power_quality",
        ["ups on battery", "input voltage low", "pdu alarm", "redundancy lost",
         "power supply failed", "psu failure",
         # ★ 旧powerルール互換
         "power redundancy lost", "power feed interrupted", "input power absent"],
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
         "fragmentation", "leak detected",
         # ★ 旧memoryルール互換
         "high memory utilization", "memory threshold exceeded",
         "mbuf cluster limit reached", "resource exhaustion"],
        "メモリ枯渇によるOOM Killer発動・システムクラッシュ",
        time_to_critical_min=180, early_warning_hours=336, base_confidence=0.85,
        category="Software/Resource"
    ),
    EscalationRule(
        "cpu_load",
        ["cpu usage high", "cpu spike", "load average high", "control plane overload",
         "process stuck", "interrupt storm",
         # ★ 旧cpuルール互換
         "high cpu utilization", "cpu threshold exceeded", "cpuhog"],
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
    _model: Optional[Any] = None
    _rule_embeddings: Optional[Dict[str, Any]] = None
    _model_loaded: bool = False

    MIN_PREDICTION_CONFIDENCE = 0.40   # ★ v2.0: 0.50→0.40 (早期予兆検知感度向上)
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

        # Phase 1: semantic_phrases キーワード照合（全ルールをスコアリング）
        # generic_error は catch-all なので Phase 1 では除外
        best_rule = None
        best_score = 0
        for rule in ESCALATION_RULES:
            if rule.pattern == "generic_error":
                continue
            hits = sum(1 for phrase in rule.semantic_phrases if phrase.lower() in text_lower)
            if hits > 0:
                score = hits + (hits / max(len(rule.semantic_phrases), 1)) * 0.1
                if score > best_score:
                    best_score = score
                    best_rule = rule
        if best_rule and best_score >= 1.0:
            quality = min(1.0, 0.7 + 0.15 * best_score)
            return best_rule, quality

        # Phase 2: 単語境界つき pattern マッチ（フォールバック）
        import re
        for rule in ESCALATION_RULES:
            if rule.pattern == "generic_error":
                continue
            if re.search(r'\b' + re.escape(rule.pattern) + r'\b', text_lower):
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

    @staticmethod
    def _format_early_warning(hours: int) -> str:
        """早期予兆の表示フォーマット"""
        if hours >= 24:
            return f"最大 {hours // 24}日前"
        return f"最大 {hours}時間前"

    def _build_narrative(self, primary_rule: EscalationRule, matched_signals: list,
                         affected_str: str, confidence: float, extra_signals: int,
                         multi_signal_boost: float) -> str:
        """★ v2.0: 2軸ナラティブ（早期予兆 + 急性期）を生成"""
        signal_lines = []
        for i, (r, q, m) in enumerate(matched_signals, 1):
            signal_lines.append(f"  シグナル{i}: {m[:80]} (Match: {q:.2f})")
        signals_text = "\n".join(signal_lines)

        correlation_note = ""
        if extra_signals > 0:
            correlation_note = (
                f"\n  ★ {len(matched_signals)}件の相関シグナルを検出 → "
                f"信頼度 +{extra_signals * multi_signal_boost:.0%} ブースト"
            )

        early_warning_str = self._format_early_warning(primary_rule.early_warning_hours)

        return (
            f"【Digital Twin未来予測 (Predictive Maintenance)】\n"
            f"{signals_text}{correlation_note}\n"
            f"・早期予兆: {early_warning_str} から検知可能なパターン\n"
            f"・急性期進行: 発症後 {primary_rule.time_to_critical_min}分 で深刻化する恐れ\n"
            f"・推奨: メンテナンスウィンドウでの予防交換/対応\n"
            f"影響: {affected_str} が連鎖的に通信断になります。\n"
            f"(信頼度スコア: {confidence:.2f})"
        )

    def _build_prediction(self, dev_id: str, primary_rule: EscalationRule,
                          primary_quality: float, matched_signals: list,
                          confidence: float, extra_signals: int,
                          multi_signal_boost: float) -> Dict[str, Any]:
        """予兆予測辞書を構築（Primary / Secondary 共通）"""
        downstream = self._get_downstream_impact(dev_id)
        impact_count = len(downstream)

        affected_names = [d[0] for d in downstream[:3]]
        if impact_count > 3:
            affected_names.append(f"他{impact_count - 3}台")
        affected_str = ", ".join(affected_names) if affected_names else "配下なし"

        reason = self._build_narrative(
            primary_rule, matched_signals, affected_str,
            confidence, extra_signals, multi_signal_boost
        )

        return {
            # --- 既存フィールド（維持必須） ---
            "id": dev_id,
            "label": f"🔮 [予兆] {primary_rule.escalated_state}",
            "severity": "CRITICAL",
            "status": "CRITICAL",
            "prob": round(confidence, 2),
            "type": f"Predictive/{primary_rule.category}",
            "tier": 1,
            "reason": reason,
            "is_prediction": True,
            "prediction_timeline": f"{primary_rule.time_to_critical_min}分後",       # ★維持必須
            "prediction_affected_count": impact_count,                                # ★維持必須
            "prediction_affected_devices": [d[0] for d in downstream],
            "prediction_signal_count": len(matched_signals),                          # ★維持必須
            "prediction_confidence_factors": {                                         # ★維持必須
                "base": primary_rule.base_confidence,
                "match_quality": primary_quality,
                "has_redundancy": bool(self.topology.get(dev_id, {}).get('redundancy_group')),
                "is_spof": bool(self.children_map.get(dev_id) and not self.topology.get(dev_id, {}).get('redundancy_group')),
                "downstream_count": impact_count,
                "correlated_signals": len(matched_signals),
                "correlation_boost": extra_signals * multi_signal_boost if extra_signals > 0 else 0,
            },
            # --- 新規フィールド（v2.0） ---
            "prediction_early_warning_hours": primary_rule.early_warning_hours,        # ★新規
            "prediction_time_to_critical_min": primary_rule.time_to_critical_min,       # ★新規
        }

    def predict(self, analysis_results: List[Dict[str, Any]], msg_map: Dict[str, List[str]], alarms: Optional[List] = None) -> List[Dict[str, Any]]:
        predictions = []
        MULTI_SIGNAL_BOOST = 0.08  # シグナル1件追加ごとのブースト量

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

            # ★ 全メッセージをスキャンし、マッチするルールを収集
            matched_signals = []
            for msg in messages:
                rule, quality = self._match_rule(msg)
                if rule and quality >= 0.30 and rule.pattern != "generic_error":
                    matched_signals.append((rule, quality, msg))

            if not matched_signals:
                # フォールバック: 最初のメッセージで generic を含めて試行
                rule, quality = self._match_rule(messages[0])
                if not rule: continue
                matched_signals = [(rule, quality, messages[0])]

            # 最も高品質なルールを主ルールとして採用
            matched_signals.sort(key=lambda x: x[1], reverse=True)
            primary_rule, primary_quality, primary_msg = matched_signals[0]

            downstream = self._get_downstream_impact(dev_id)
            confidence = self._calculate_confidence(primary_rule, dev_id, primary_quality)

            # ★ 複数シグナル相関ブースト
            extra_signals = len(matched_signals) - 1
            if extra_signals > 0:
                boost = min(extra_signals * MULTI_SIGNAL_BOOST, 0.20)
                confidence = min(0.99, confidence + boost)

            if confidence < self.MIN_PREDICTION_CONFIDENCE:
                continue

            pred = self._build_prediction(
                dev_id, primary_rule, primary_quality, matched_signals,
                confidence, extra_signals, MULTI_SIGNAL_BOOST
            )
            predictions.append(pred)
            processed_devices.add(dev_id)

        # ★ Secondary scan: Weak Signal 直接検出 + 複数シグナル相関ブースト
        # LogicalRCA が低スコア (< 0.45) をつけた INFO アラームでも、
        # Digital Twin のルールに合致すれば予兆として検出する。
        # 複数の微弱シグナルが相関する場合、信頼度を段階的にブーストする。

        for dev_id, messages in msg_map.items():
            if dev_id in processed_devices:
                continue
            if dev_id == "SYSTEM":
                continue
            if dev_id not in self.topology:
                continue

            # 全メッセージをスキャンし、マッチするルールを収集
            matched_signals = []  # [(rule, quality, msg), ...]
            seen_patterns = set()
            for msg in messages:
                rule, quality = self._match_rule(msg)
                if not rule or quality < 0.30:
                    continue
                if rule.pattern == "generic_error":
                    continue
                matched_signals.append((rule, quality, msg))
                seen_patterns.add(rule.pattern)

            if not matched_signals:
                continue

            # 最も高品質なルールを主ルールとして採用
            matched_signals.sort(key=lambda x: x[1], reverse=True)
            primary_rule, primary_quality, primary_msg = matched_signals[0]

            downstream = self._get_downstream_impact(dev_id)
            confidence = self._calculate_confidence(primary_rule, dev_id, primary_quality)

            # ★ 複数シグナル相関ブースト（論文: correlated weak signals）
            # 異なるルールにマッチするシグナルが多いほど確信度が上がる
            extra_signals = len(matched_signals) - 1
            if extra_signals > 0:
                boost = min(extra_signals * MULTI_SIGNAL_BOOST, 0.20)  # 最大+20%
                confidence = min(0.99, confidence + boost)

            if confidence < 0.40:
                continue

            pred = self._build_prediction(
                dev_id, primary_rule, primary_quality, matched_signals,
                confidence, extra_signals, MULTI_SIGNAL_BOOST
            )
            predictions.append(pred)
            processed_devices.add(dev_id)

        return predictions
