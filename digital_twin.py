# -*- coding: utf-8 -*-
"""
digital_twin.py (Universal Edition - Production Ready v3.0)
===========================================================
AIOps Digital Twin Engine

[修正履歴]
 - Fix 1: 影響範囲(0台)のバグ修正。Graph構築時のエッジ方向(Downstream)を正しく定義。
 - Fix 2: 複数シグナル対応。1デバイスに対し複数の予兆ログがある場合、それらを統合してスコアをブースト。
 - Fix 3: 障害済み除外。既にCRITICAL判定されているデバイスは予兆対象から外す。
 - Fix 4: 未定義メソッドエラーの修正 (_build_prediction を実装)。

設計方針:
 - このモジュールは「本番運用」に耐えうるロジックを持つ。
 - 入力元がデモ(app.py)でも本番(Syslog)でも動作する。
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
# Escalation Rules (科学的根拠に基づく設定)
# ==========================================================
@dataclass
class EscalationRule:
    pattern: str
    semantic_phrases: List[str]
    escalated_state: str
    time_to_critical_min: int
    early_warning_hours: int
    base_confidence: float
    category: str = "Generic"

ESCALATION_RULES: List[EscalationRule] = [
    EscalationRule("stp_loop", ["stp loop", "tcn received", "blocking port"], "L2ループによるブロードキャストストーム", 5, 24, 0.95, "Network/L2"),
    EscalationRule("mac_flap", ["mac flapping", "host moving"], "MACテーブル不安定化によるフレーム消失", 10, 24, 0.90, "Network/L2"),
    EscalationRule("arp_storm", ["arp storm", "duplicate ip"], "ARPテーブル汚染による通信断", 10, 12, 0.85, "Network/L2"),
    EscalationRule("bgp_flap", ["bgp flapping", "neighbor down", "route oscillation"], "BGPセッション不安定化による経路消失", 15, 48, 0.90, "Network/Routing"),
    EscalationRule("ospf_adj", ["ospf adjacency down", "dead timer"], "OSPFネイバー喪失によるブラックホール", 15, 24, 0.85, "Network/Routing"),
    EscalationRule("ha_split", ["ha state degraded", "heartbeat lost"], "HA同期不全によるスプリットブレイン", 30, 48, 0.85, "Network/HA"),
    EscalationRule("bandwidth", ["bandwidth exceeded", "output drops"], "帯域飽和によるサービス品質劣化", 20, 72, 0.80, "Network/QoS"),
    EscalationRule("drop_error", ["input errors", "crc error", "fcs error"], "物理回線品質劣化によるスループット低下", 30, 168, 0.75, "Network/Interface"),
    EscalationRule("ntp_drift", ["ntp unsynchronized", "time drift"], "時刻不整合による認証エラー", 120, 168, 0.70, "Network/Service"),
    EscalationRule("dhcp_dns", ["dhcp pool exhausted", "dns timeout"], "新規クライアント接続不可", 30, 48, 0.80, "Network/Service"),
    EscalationRule("optical", ["rx power", "optical signal", "transceiver", "light level", "dbm"], "光信号劣化による突然のリンクダウン", 60, 336, 0.95, "Hardware/Optical"),
    EscalationRule("temperature", ["temperature high", "overheat"], "熱暴走による緊急シャットダウン", 30, 48, 0.85, "Hardware/Thermal"),
    EscalationRule("fan_fail", ["fan failure", "fan malfunction"], "冷却能力喪失による温度上昇", 45, 72, 0.80, "Hardware/Thermal"),
    EscalationRule("power_quality", ["ups on battery", "power supply failed"], "電源供給不安定による再起動", 15, 24, 0.85, "Hardware/Power"),
    EscalationRule("storage", ["flash error", "nvram corruption"], "ストレージ障害による起動不能", 180, 720, 0.75, "Hardware/Storage"),
    EscalationRule("memory_leak", ["memory usage high", "malloc fail"], "メモリ枯渇によるシステムクラッシュ", 180, 336, 0.85, "Software/Resource"),
    EscalationRule("cpu_load", ["cpu usage high", "load average high"], "CPU枯渇によるプロトコルダウン", 20, 48, 0.85, "Software/Resource"),
    EscalationRule("process_crash", ["process terminated", "core dump"], "重要プロセス停止", 10, 24, 0.90, "Software/Process"),
    EscalationRule("auth_failure", ["authentication failed", "radius timeout"], "認証基盤障害", 15, 12, 0.80, "Security/Auth"),
    EscalationRule("crypto_vpn", ["ike sa deleted", "vpn tunnel down"], "VPNトンネル切断", 60, 720, 0.80, "Security/Crypto"),
    EscalationRule("generic_error", ["error", "fail", "critical", "warning"], "未分類のサービス劣化", 30, 24, 0.50, "Generic"),
]

# ==========================================================
# Digital Twin Engine
# ==========================================================
class DigitalTwinEngine:
    """
    本番運用に耐えうるAIOpsエンジン。
    シングルトンパターンによるモデル管理と、NetworkXによるグラフ解析を提供する。
    """
    _model: Optional[Any] = None
    _rule_embeddings: Optional[Dict[str, Any]] = None
    _model_loaded: bool = False

    # 設定パラメータ
    MIN_PREDICTION_CONFIDENCE = 0.40
    MAX_PROPAGATION_HOPS = 3
    REDUNDANCY_DISCOUNT = 0.15
    SPOF_BOOST = 1.10
    EMBEDDING_THRESHOLD = 0.40
    MULTI_SIGNAL_BOOST = 0.05 # 追加シグナルごとの加点

    def __init__(self, topology: Dict[str, Any], children_map: Optional[Dict[str, List[str]]] = None):
        self.topology = topology
        self.children_map = children_map or {}
        self.graph = None
        
        # --- NetworkX グラフ構築 (修正版: 方向性の厳密化) ---
        if HAS_NX:
            self.graph = nx.DiGraph()
            for node_id, attrs in topology.items():
                node_attrs = attrs if isinstance(attrs, dict) else vars(attrs)
                self.graph.add_node(node_id, **node_attrs)
                
                # トポロジーからエッジを構築
                # 親(Upstream) -> 子(Downstream) の方向を "downstream" relation として定義
                parent_id = node_attrs.get("parent_id")
                if parent_id and parent_id in topology:
                    self.graph.add_edge(parent_id, node_id, relation="downstream")
                    self.graph.add_edge(node_id, parent_id, relation="upstream")
                
                # children_map からの補完 (親IDを持たないルートノード用)
                if node_id in self.children_map:
                    for child in self.children_map[node_id]:
                        if child in topology:
                            # 既存エッジがなければ追加
                            if not self.graph.has_edge(node_id, child):
                                self.graph.add_edge(node_id, child, relation="downstream")
                            if not self.graph.has_edge(child, node_id):
                                self.graph.add_edge(child, node_id, relation="upstream")

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
        """Embeddingモデルのロード (初回のみ)"""
        if cls._model_loaded: return
        if not HAS_BERT:
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
                cls._rule_embeddings = {"vectors": embeddings, "phrase_to_rule_idx": phrase_to_rule_idx, "phrases": all_phrases}
            cls._model_loaded = True
        except:
            cls._model = None
            cls._model_loaded = True

    def _match_rule(self, alarm_text: str) -> Tuple[Optional[EscalationRule], float]:
        """ハイブリッドマッチング (キーワード + Embedding)"""
        text_lower = alarm_text.lower()
        # 1. Keyword Match
        for rule in ESCALATION_RULES:
            if rule.pattern in text_lower:
                return rule, 1.0
        # 2. Embedding Match
        if self._model and self._rule_embeddings:
            try:
                query_vec = self._model.encode([alarm_text], convert_to_numpy=True)
                rule_vecs = self._rule_embeddings["vectors"]
                similarities = np.dot(rule_vecs, query_vec.T).flatten()
                norms = np.linalg.norm(rule_vecs, axis=1) * np.linalg.norm(query_vec)
                cosine_sim = similarities / np.where(norms==0, 1e-10, norms)
                best_idx = np.argmax(cosine_sim)
                best_score = float(cosine_sim[best_idx])
                if best_score >= self.EMBEDDING_THRESHOLD:
                    return ESCALATION_RULES[self._rule_embeddings["phrase_to_rule_idx"][best_idx]], best_score
            except: pass
        return None, 0.0

    def _get_downstream_impact(self, root_id: str) -> List[Tuple[str, int]]:
        """配下デバイスの探索 (Graph Traversal)"""
        impacts = []
        if not self.graph or root_id not in self.graph: return impacts
        try:
            # relation="downstream" のエッジだけを辿るフィルタ
            def downstream_filter(u, v):
                edge_data = self.graph[u][v]
                return edge_data.get("relation") == "downstream"
            
            subgraph = nx.subgraph_view(self.graph, filter_edge=downstream_filter)
            # BFS実行
            tree = nx.bfs_tree(subgraph, root_id, depth_limit=self.MAX_PROPAGATION_HOPS)
            for node in tree:
                if node == root_id: continue
                dist = nx.shortest_path_length(subgraph, root_id, node)
                impacts.append((node, dist))
        except Exception as e:
            logger.error(f"Traversal error: {e}")
        return impacts

    def _calculate_confidence(self, rule: EscalationRule, device_id: str, match_quality: float) -> float:
        """信頼度スコア計算"""
        attrs = self.topology.get(device_id, {})
        if not isinstance(attrs, dict): attrs = vars(attrs)
        
        rg = attrs.get('redundancy_group')
        has_redundancy = bool(rg and len(self._redundancy_groups.get(rg, [])) > 1)
        # 配下がいるのに冗長化されていない場合はSPOFとみなす
        downstream_count = len(self._get_downstream_impact(device_id))
        is_spof = bool(downstream_count > 0 and not has_redundancy)
        
        confidence = rule.base_confidence
        confidence *= (0.8 + 0.2 * match_quality)
        if has_redundancy: confidence *= (1.0 - self.REDUNDANCY_DISCOUNT)
        if is_spof: confidence *= self.SPOF_BOOST
        return min(0.99, max(0.1, confidence))

    def _build_prediction(self, dev_id, rule, quality, matched_signals, confidence, extra_signal_count, boost_factor):
        """予測データの構築 (ナラティブ生成含む)"""
        # 影響範囲
        downstream = self._get_downstream_impact(dev_id)
        impact_count = len(downstream)
        
        # 配下デバイス名の生成
        if impact_count == 0:
            affected_str = "配下なし(End Node)"
        else:
            names = [d[0] for d in downstream[:3]]
            if impact_count > 3:
                names.append(f"他{impact_count-3}台")
            affected_str = ", ".join(names)

        # 早期予兆テキスト
        if rule.early_warning_hours >= 24:
            early_str = f"最大 {rule.early_warning_hours // 24}日前"
        else:
            early_str = f"最大 {rule.early_warning_hours}時間前"
        
        # 複数シグナルの注釈
        multi_signal_note = ""
        if extra_signal_count > 0:
            boost_val = min(extra_signals * boost_factor, 0.20)
            multi_signal_note = f"\n・相関分析: 他 {extra_signal_count} 件の関連シグナルを検知 (確信度 +{boost_val:.0%})"

        # メインの検知根拠
        primary_msg = matched_signals[0][2]

        return {
            "id": dev_id,
            "label": f"🔮 [予兆] {rule.escalated_state}",
            "severity": "CRITICAL",
            "status": "CRITICAL",
            "prob": round(confidence, 2),
            "type": f"Predictive/{rule.category}",
            "tier": 1,
            "reason": (
                f"【Digital Twin未来予測】\n"
                f"・早期予兆: {early_str} から検知可能なパターン\n"
                f"・急性期: 発症後 {rule.time_to_critical_min}分 で深刻化する恐れ\n"
                f"・影響範囲: {affected_str} ({impact_count}台) が通信断になるリスク\n"
                f"・推奨: 次回メンテナンスウィンドウでの予防交換/対応\n"
                f"--------------------------------\n"
                f"・検出根拠: {primary_msg} (Match: {quality:.2f}){multi_signal_note}"
            ),
            "is_prediction": True,
            # UI表示用のメタデータ
            "prediction_timeline": f"{rule.time_to_critical_min}分後",
            "prediction_early_warning": early_str,
            "prediction_affected_count": impact_count,
            "prediction_escalated_state": rule.escalated_state
        }

    def predict(self, analysis_results: List[Dict[str, Any]], msg_map: Dict[str, List[str]], alarms: Optional[List] = None) -> List[Dict[str, Any]]:
        """
        予兆検知の実行メインメソッド
        """
        predictions = []
        
        # 1. 既に障害(CRITICAL)判定されている機器は除外
        critical_ids = {
            r["id"] for r in analysis_results 
            if str(r.get("status")) == "HealthStatus.CRITICAL" or r.get("status") == "CRITICAL" or r.get("severity") == "CRITICAL"
        }

        # 2. 候補選定
        # Warning機器(prob 0.45~0.85) + ログがある全機器(INFO予兆含む)
        warning_ids = {
            r["id"] for r in analysis_results
            if 0.45 <= float(r.get("prob", 0)) <= 0.85
        }
        active_ids = set(msg_map.keys())
        # 障害済みを除外した候補リスト
        candidates = (warning_ids.union(active_ids)) - critical_ids
        
        processed_devices = set()

        for dev_id in candidates:
            if dev_id in processed_devices: continue
            
            # ログ取得
            messages = msg_map.get(dev_id, [])
            if not messages: continue

            # 3. 複数シグナルの全スキャン
            matched_signals = []  # [(rule, quality, msg), ...]
            
            for msg in messages:
                rule, quality = self._match_rule(msg)
                # マッチ度が低い、またはGenericなものはノイズとして除外
                if not rule or quality < 0.35: continue
                if rule.pattern == "generic_error": continue
                
                matched_signals.append((rule, quality, msg))
            
            if not matched_signals: continue

            # 最も確信度が高いルールをメインとして採用
            matched_signals.sort(key=lambda x: x[1], reverse=True)
            primary_rule, primary_quality, _ = matched_signals[0]

            # 信頼度計算
            confidence = self._calculate_confidence(primary_rule, dev_id, primary_quality)
            
            # 複数シグナルブースト
            extra_signals = len(matched_signals) - 1
            if extra_signals > 0:
                boost = min(extra_signals * self.MULTI_SIGNAL_BOOST, 0.20)
                confidence = min(0.99, confidence + boost)

            # 閾値判定
            if confidence < self.MIN_PREDICTION_CONFIDENCE: continue

            # 予測生成
            pred = self._build_prediction(
                dev_id, primary_rule, primary_quality, matched_signals, 
                confidence, extra_signals, self.MULTI_SIGNAL_BOOST
            )
            
            predictions.append(pred)
            processed_devices.add(dev_id)
            
        return predictions

if __name__ == "__main__":
    # 簡易動作テスト
    logging.basicConfig(level=logging.INFO)
    print("Initializing Digital Twin Engine...")
    
    # テスト用トポロジー
    topo = {
        "WAN_ROUTER": {"parent_id": None},
        "FW": {"parent_id": "WAN_ROUTER"},
        "SW": {"parent_id": "FW"}
    }
    # テスト用ログ (複数シグナル)
    msgs = {
        "WAN_ROUTER": [
            "Rx Power -25.5 dBm (Threshold -25.0)", # Optical
            "Interface CRC errors increasing",      # Drop Error
            "Link fluctuation detected"             # Optical (Semantic match)
        ]
    }
    dummy_results = [] # 何も障害が出ていない状態
    
    dt = DigitalTwinEngine(topo)
    preds = dt.predict(dummy_results, msgs)
    
    print(f"Predictions Generated: {len(preds)}")
    for p in preds:
        print(f"[{p['id']}] {p['label']} (Prob: {p['prob']})")
        print(p['reason'])
