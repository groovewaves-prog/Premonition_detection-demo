# digital_twin_pkg/engine.py  ―  DigitalTwinEngine コアロジック（Phase1 Predict API + RUL予測）
import logging
import time
import json
import uuid
import re
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict
import traceback

from .config import *
from .rules import EscalationRule, DEFAULT_RULES, MAINTENANCE_SIGNATURES
from .storage import StorageManager
from .audit import AuditBuilder
from .tuning import AutoTuner
from .bayesian import BayesianInferenceEngine
from .gnn import create_gnn_engine

try:
    from sentence_transformers import SentenceTransformer
    HAS_BERT = True
except ImportError:
    HAS_BERT = False

logger = logging.getLogger(__name__)



# ==============================================================
# Phase1: Predict API + Forecast Ledger  (digital_twin_pkg)
# ==============================================================

import traceback
from dataclasses import asdict as _asdict

# DTO ─────────────────────────────────────────────────────────
from dataclasses import dataclass as _dc, field as _field
from typing import Optional as _Opt

@_dc
class PredictRequest:
    tenant_id:  str
    device_id:  str
    msg:        str
    timestamp:  float
    attrs:      dict = _field(default_factory=dict)

    def to_dict(self):
        return {"tenant_id": self.tenant_id, "device_id": self.device_id,
                "msg": self.msg, "timestamp": self.timestamp, "attrs": self.attrs or {}}

@_dc
class PredictResult:
    predicted_state:      str
    confidence:           float
    rule_pattern:         str
    category:             str
    reasons:              list = _field(default_factory=list)
    recommended_actions:  list = _field(default_factory=list)
    runbook_url:          str  = ""
    criticality:          str  = "standard"
    time_to_critical_min: int  = 60
    early_warning_hours:  int  = 24
    time_to_failure_hours: int = 336  # ★ RUL: 今から完全故障まで（時間）
    predicted_failure_datetime: str = ""  # ★ 故障発生予測日時（ISO形式）

    def to_dict(self, affected_count: int = 0, source: str = "real"):
        return {
            "predicted_state":      self.predicted_state,
            "confidence":           float(self.confidence),
            "rule_pattern":         self.rule_pattern,
            "category":             self.category,
            "reasons":              self.reasons or [],
            "recommended_actions":  self.recommended_actions or [],
            "runbook_url":          self.runbook_url or "",
            "criticality":          self.criticality or "standard",
            "time_to_critical_min": int(self.time_to_critical_min),
            "early_warning_hours":  int(self.early_warning_hours),
            "time_to_failure_hours": int(self.time_to_failure_hours),
            "predicted_failure_datetime": self.predicted_failure_datetime,
            # ── cockpit.py 互換フィールド ──────────────────────
            "is_prediction":        True,
            "source":               source,
            "prob":                 float(self.confidence),
            "label":                f"🔮 [予兆] {self.predicted_state}",
            "type":                 f"Predictive/{self.category}",
            "tier":                 1,
            "prediction_timeline":  f"{self.time_to_critical_min}分後",
            "prediction_time_to_critical_min": int(self.time_to_critical_min),
            "prediction_early_warning_hours":  int(self.early_warning_hours),
            "prediction_affected_count":       int(affected_count),
            "prediction_time_to_failure_hours": int(self.time_to_failure_hours),
            "prediction_failure_datetime":      self.predicted_failure_datetime,
        }


class DigitalTwinEngine:
    def __init__(self, topology: Dict[str, Any], children_map: Optional[Dict[str, List[str]]] = None, tenant_id: str = "default"):
        if not tenant_id or len(tenant_id) > 64: raise ValueError("Invalid tenant_id")
        self.tenant_id = tenant_id.lower()
        self.topology = topology
        self.children_map = children_map or {}
        self.storage = StorageManager(self.tenant_id, BASE_DIR)
        self.tuner = AutoTuner(self)
        self.bayesian = BayesianInferenceEngine(self.storage)  # ★ ベイズ推論エンジン
        self.gnn = create_gnn_engine(topology, children_map)  # ★ GNN予測エンジン
        self.rules: List[EscalationRule] = []
        self._metric_rules: List[EscalationRule] = []
        self.history: List[Dict] = []
        self.outcomes: List[Dict] = []
        self.incident_register: List[Dict] = []
        self.maintenance_windows: List[Dict] = []
        self.evaluation_state: Dict = {}
        self.shadow_eval_state: Dict = {}
        self._model = None
        self._rule_embeddings = None
        self._model_loaded = False
        self._rules_sot = (os.environ.get(ENV_RULES_SOT, "json") or "json").strip().lower()
        self.reload_all()
        self._ensure_model_loaded()

    def reload_all(self):
        self._load_rules()
        self.history = self.storage.load_json("history", [])
        self.outcomes = self.storage.load_json("outcomes", [])
        self.incident_register = self.storage.load_json("incident_register", [])
        self.maintenance_windows = self.storage.load_json("maintenance_windows", [])
        self.evaluation_state = self.storage.load_json("evaluation_state", {})
        self.shadow_eval_state = self.storage.load_json("shadow_eval_state", {})
        self._init_forecast_ledger()

    def _sanitize_rule_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in data.items() if not k.startswith('_')}

    def _load_rules(self):
        loaded_from_db = False
        if self._rules_sot == "db":
            db_rules_json = self.storage.rule_config_get_all_json_strs()
            if db_rules_json:
                try:
                    self.rules = [EscalationRule(**self._sanitize_rule_data(json.loads(s))) for s in db_rules_json]
                    loaded_from_db = True
                except: pass
        if not loaded_from_db:
            path = self.storage.paths["rules"]
            if not os.path.exists(path):
                self.rules = [EscalationRule(**self._sanitize_rule_data(asdict(r))) for r in DEFAULT_RULES]
                self.storage.save_json_atomic("rules", [self._sanitize_rule_data(asdict(r)) for r in self.rules])
            else:
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self.rules = [EscalationRule(**self._sanitize_rule_data(item)) for item in data]
                except Exception as e:
                    self.rules = [EscalationRule(**self._sanitize_rule_data(asdict(r))) for r in DEFAULT_RULES]
            self.storage._seed_rule_config_from_rules_json([self._sanitize_rule_data(asdict(r)) for r in self.rules])
        self._metric_rules = [r for r in self.rules if (r.requires_trend or r.requires_volatility) and r.trend_metric_regex]

    def _ensure_model_loaded(self):
        if self._model_loaded: return
        if not HAS_BERT:
            self._model_loaded = True
            return
        try:
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
            phrases = []
            indices = []
            for idx, r in enumerate(self.rules):
                for p in r.semantic_phrases:
                    phrases.append(p)
                    indices.append(idx)
            if phrases:
                embeddings = self._model.encode(phrases, convert_to_numpy=True)
                self._rule_embeddings = {"vectors": embeddings, "indices": indices}
            self._model_loaded = True
        except: self._model_loaded = True

    def _match_rule(self, alarm_text: str) -> Tuple[Optional[EscalationRule], float]:
        text_lower = alarm_text.lower()
        for rule in self.rules:
            if rule._compiled_regex and rule._compiled_regex.search(alarm_text):
                return rule, 1.0
            if rule.pattern in text_lower:
                return rule, 1.0
        if self._model and self._rule_embeddings:
            try:
                query_vec = self._model.encode([alarm_text], convert_to_numpy=True)
                rule_vecs = self._rule_embeddings["vectors"]
                similarities = np.dot(rule_vecs, query_vec.T).flatten()
                norms = np.linalg.norm(rule_vecs, axis=1) * np.linalg.norm(query_vec)
                cosine_sim = similarities / np.where(norms==0, 1e-10, norms)
                best_idx = np.argmax(cosine_sim)
                best_score = float(cosine_sim[best_idx])
                rule_idx = self._rule_embeddings["indices"][best_idx]
                rule = self.rules[rule_idx]
                if best_score >= (rule.embedding_threshold or 0.40):
                    return rule, best_score
            except Exception: pass
        return None, 0.0

    def _calculate_confidence(self, rule: EscalationRule, device_id: str, match_quality: float) -> float:
        attrs = self.topology.get(device_id, {})
        if not isinstance(attrs, dict):
            try: attrs = vars(attrs)
            except: attrs = {}
        rg = attrs.get('redundancy_group')
        has_redundancy = bool(rg)
        children = self.children_map.get(device_id, [])
        is_spof = bool(children and not has_redundancy)
        confidence = rule.base_confidence
        confidence *= (0.8 + 0.2 * match_quality)
        if has_redundancy: confidence *= (1.0 - ROI_CONSERVATIVE_FACTOR * 0.2)
        if is_spof: confidence *= 1.1
        return min(0.99, max(0.1, confidence))

    def _sanitize_for_llm(self, text: str) -> str:
        """
        LLM送信前のデータサニタイズ
        
        - IPアドレスのマスキング
        - プライベート情報の除去
        - 機密情報の匿名化
        """
        import re
        
        sanitized = text
        
        # IPv4アドレスのマスキング
        sanitized = re.sub(
            r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'IP_MASKED',
            sanitized
        )
        
        # IPv6アドレスのマスキング
        sanitized = re.sub(
            r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',
            'IPV6_MASKED',
            sanitized
        )
        
        # MACアドレスのマスキング
        sanitized = re.sub(
            r'\b(?:[0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2}\b',
            'MAC_MASKED',
            sanitized
        )
        
        # ホスト名の一般化（prod-, dev-, test-などを除去）
        sanitized = re.sub(
            r'\b(prod|dev|test|stage|staging)-[\w-]+',
            'HOSTNAME_MASKED',
            sanitized,
            flags=re.IGNORECASE
        )
        
        # ASN (AS番号)のマスキング
        sanitized = re.sub(
            r'\bAS\d+\b',
            'AS_MASKED',
            sanitized
        )
        
        # VLAN IDのマスキング
        sanitized = re.sub(
            r'\bVLAN\s*\d+\b',
            'VLAN_MASKED',
            sanitized,
            flags=re.IGNORECASE
        )
        
        return sanitized


    def _generate_smart_recommendations(
        self,
        rule_pattern: str,
        affected_count: int,
        base_actions: list,
    ) -> list:
        """
        シグナル件数（affected_count）に基づいて静的ルールで推奨アクションを決定する。
        外部LLMは使用しない（即時・決定論的・優先順序を明示制御）。

        閾値:
          1〜2件 : 個別部品故障 → 単体SFP/インターフェース対応
          3〜4件 : ラインカード/モジュール単位の問題
          5件以上: 筐体レベル（電源・ファームウェア・基板）の問題
        """
        WIDE_RANGE_THRESHOLD = 5
        MID_RANGE_THRESHOLD  = 3

        if "optical" in rule_pattern:
            if affected_count >= WIDE_RANGE_THRESHOLD:
                return [
                    {
                        "title": "筐体電源系統の確認（PSU冗長・負荷状況）",
                        "effect": f"{affected_count}個の光モジュール同時劣化の主因を排除",
                        "priority": "high",
                        "rationale": f"{affected_count}個が同時劣化 → 単発SFP故障では説明困難。電源電圧不安定を最初に疑う。",
                        "steps": "1. show environment power\n2. show platform\n3. 各PSUの出力電圧/負荷率を確認"
                    },
                    {
                        "title": "筐体内温度・冷却ファンの確認",
                        "effect": "過熱による光モジュール特性劣化を解消",
                        "priority": "high",
                        "rationale": "広範囲の光信号劣化は筐体内過熱でも発生する。",
                        "steps": "1. show environment temperature\n2. show environment fan\n3. データセンター空調状況も確認"
                    },
                    {
                        "title": "IOS/ファームウェアバージョンの確認",
                        "effect": "ソフトウェア起因の誤検知・光制御異常を解消",
                        "priority": "medium",
                        "rationale": "既知のバグで光パワー読み値が異常になるケースあり。リリースノート確認。",
                        "steps": "1. show version\n2. ベンダーの既知障害情報を照合\n3. 該当バグがあればパッチ適用を検討"
                    },
                    {
                        "title": "SFPモジュールの個別確認（最終手段）",
                        "effect": "残留する個別モジュール故障を解消",
                        "priority": "low",
                        "rationale": f"上記で解消しない場合のみ。{affected_count}個全交換は費用対効果が低い。",
                        "steps": "1. show interfaces transceiver\n2. Rx/Tx Powerが閾値外のポートを特定\n3. 該当SFPのみ交換"
                    },
                ]
            elif affected_count >= MID_RANGE_THRESHOLD:
                return [
                    {
                        "title": "該当ラインカード／スロットの確認",
                        "effect": f"{affected_count}個が同一カードに集中している場合、カード交換で解決",
                        "priority": "high",
                        "rationale": "複数ポートが同じラインカードに属している場合はカード障害が主因。",
                        "steps": "1. show interfaces transceiver で影響ポートのスロットを確認\n2. show platform slot で該当スロットの状態を確認\n3. 同スロット集中なら予備カードと交換"
                    },
                    {
                        "title": "光ファイバーの接続状態・清掃",
                        "effect": "コネクタ汚れ・曲げによる光損失を回復",
                        "priority": "medium",
                        "rationale": "複数ポートで同時に光損失 → パッチパネル側の共通障害も疑う。",
                        "steps": "1. 光コネクタを顕微鏡検査\n2. アルコール綿棒で清掃\n3. Rx Powerを再測定"
                    },
                    {
                        "title": "SFPモジュールの個別確認",
                        "effect": "故障モジュールを特定・交換",
                        "priority": "low",
                        "rationale": "上記で改善しない場合に個別SFPを交換。",
                        "steps": "1. show interfaces transceiver detail\n2. Rx Power最低値のポートから順に交換"
                    },
                ]
            else:
                return base_actions

        elif "microburst" in rule_pattern:
            if affected_count >= WIDE_RANGE_THRESHOLD:
                return [
                    {
                        "title": "ASIC／チップセットの診断",
                        "effect": f"{affected_count}個のインターフェースでのバッファ問題を根本解消",
                        "priority": "high",
                        "rationale": "広範囲のqueue dropsはASICのバグ・故障の可能性が高い。",
                        "steps": "1. show platform resources\n2. show platform hardware\n3. ベンダーのASIC既知バグを照合"
                    },
                    {
                        "title": "IOS/ファームウェアのバグ確認",
                        "effect": "QoS処理の異常を解消",
                        "priority": "high",
                        "rationale": "複数ポート同時発生はソフトウェアバグの可能性。",
                        "steps": "1. show version\n2. リリースノートでQoS関連のバグを確認\n3. 修正済みバージョンへのアップグレード"
                    },
                    {
                        "title": "トラフィックパターンの分析",
                        "effect": "異常トラフィック発生源の特定・遮断",
                        "priority": "medium",
                        "rationale": "DDoS・異常フローによる全ポート同時輻輳の可能性。",
                        "steps": "1. show interfaces | include drops\n2. NetFlow/sFlowで異常フローを特定\n3. ACLで遮断"
                    },
                    {
                        "title": "QoSポリシーの最適化",
                        "effect": "バッファ割り当てを改善し一時的な輻輳を緩和",
                        "priority": "low",
                        "rationale": "根本解決後の最適化として実施。",
                        "steps": "1. show policy-map interface\n2. キュー深度・重み付けを調整"
                    },
                ]
            else:
                return base_actions

        elif "route_instability" in rule_pattern or "bgp" in rule_pattern:
            if affected_count >= MID_RANGE_THRESHOLD:
                return [
                    {
                        "title": "BGP設定の包括的レビュー",
                        "effect": f"{affected_count}個のピアの不安定さを解消",
                        "priority": "high",
                        "rationale": "複数ピア同時不安定 → 設定ミス or 上流ISP側の問題を最初に確認。",
                        "steps": "1. show bgp summary\n2. 各ピアのhold-timer/keepalive設定を確認\n3. 上流ISPにNOC問い合わせ"
                    },
                    {
                        "title": "IOS/ファームウェアのBGP実装確認",
                        "effect": "BGP処理バグによる経路不安定を回避",
                        "priority": "medium",
                        "rationale": "既知のBGP実装バグで複数ピア同時フラップが発生するケースあり。",
                        "steps": "1. show version\n2. ベンダーの既知BGPバグを照合\n3. 修正バージョンへのアップグレードを検討"
                    },
                    {
                        "title": "BGPフラップダンピングの設定",
                        "effect": "不安定なピアの経路広報を抑制",
                        "priority": "low",
                        "rationale": "根本解決が難しい場合の緩和策。",
                        "steps": "1. bgp dampening コマンドを設定\n2. show bgp dampened-paths で抑制状況を確認"
                    },
                ]
            else:
                return base_actions

        else:
            if affected_count >= WIDE_RANGE_THRESHOLD:
                return base_actions + [
                    {
                        "title": "システム全体の健全性確認",
                        "effect": f"{affected_count}件のシグナルの根本原因を特定",
                        "priority": "high",
                        "rationale": "広範囲のシグナル発生は電源・ファームウェア・環境問題を疑う。",
                        "steps": "1. show environment all\n2. show version\n3. ベンダーサポートへのエスカレーション検討"
                    }
                ]
            return base_actions


    def predict(self, analysis_results: List[Dict], msg_map: Dict[str, List[str]], alarms: Optional[List] = None) -> List[Dict]:
        self.reload_all()
        predictions = []
        critical_ids = {r["id"] for r in analysis_results if r.get("status") in ["RED", "CRITICAL"] or r.get("severity") == "CRITICAL" or float(r.get("prob", 0)) >= 0.85}
        warning_ids = {r["id"] for r in analysis_results if 0.45 <= float(r.get("prob", 0)) <= 0.85}
        active_ids = set(msg_map.keys())
        candidates = (warning_ids.union(active_ids)) - critical_ids
        processed_devices = set()
        multi_signal_boost = 0.05
        
        for dev_id in candidates:
            if dev_id in processed_devices: continue
            messages = msg_map.get(dev_id, [])
            if not messages: continue
            matched_signals = []
            for msg in messages:
                rule, quality = self._match_rule(msg)
                if rule and quality >= 0.30 and rule.pattern != "generic_error":
                    matched_signals.append((rule, quality, msg))
            if not matched_signals:
                rule, quality = self._match_rule(messages[0])
                if not rule: continue
                matched_signals = [(rule, quality, messages[0])]

            matched_signals.sort(key=lambda x: x[1], reverse=True)
            primary_rule, primary_quality, primary_msg = matched_signals[0]
            confidence = self._calculate_confidence(primary_rule, dev_id, primary_quality)
            extra_signals = len(matched_signals) - 1
            if extra_signals > 0:
                boost = min(extra_signals * multi_signal_boost, 0.20)
                confidence = min(0.99, confidence + boost)
            
            # ★ ベイズ推論による信頼度の更新
            confidence, bayesian_debug = self.bayesian.calculate_posterior_confidence(
                device_id=dev_id,
                rule_pattern=primary_rule.pattern,
                current_confidence=confidence,
                time_window_hours=168  # 過去7日間
            )
            
            # ★ GNN予測による信頼度の補正（オプション）
            if self.gnn and self._model:
                try:
                    # 現在のアラームメッセージをBERT埋め込みに変換
                    alarm_embeddings = {}
                    for msg_dev_id, msg_list in msg_map.items():
                        if msg_list:
                            # 複数メッセージの平均埋め込み
                            embeddings = self._model.encode(msg_list, convert_to_numpy=True)
                            alarm_embeddings[msg_dev_id] = embeddings.mean(axis=0)
                    
                    # GNNで予測
                    gnn_confidence, gnn_ttf = self.gnn.predict_with_gnn(
                        alarm_embeddings, dev_id
                    )
                    
                    # ベイズ推論とGNN予測の加重平均（GNNの重みは控えめ）
                    confidence = 0.7 * confidence + 0.3 * gnn_confidence
                    confidence = min(0.99, max(0.1, confidence))
                    
                except Exception as e:
                    logger.warning(f"GNN prediction failed: {e}")

            threshold = MIN_PREDICTION_CONFIDENCE
            if primary_rule.paging_threshold is not None:
                threshold = primary_rule.paging_threshold
            if confidence < threshold: continue

            impact_count = 0
            if dev_id in self.children_map:
                impact_count = len(self.children_map[dev_id])
            
            # ★ シグナル件数（matched_signals数）を affected_count として使用
            # インターフェース名の有無に関わらずシグナル数が実態を最も正確に表す
            import re as _re_comp
            unique_components = set()
            for _, _, _m in matched_signals:
                unique_components.update(
                    _re_comp.findall(r'\b(?:Gi|Te|Fa|Et)\d+/\d+/\d+|\b(?:Gi|Te|Fa|Et)\d+/\d+', _m))
            # インターフェース名が抽出できた場合はそれを、できない場合はシグナル件数を使用
            component_count = len(unique_components) if unique_components else len(matched_signals)

            smart_actions = self._generate_smart_recommendations(
                rule_pattern=primary_rule.pattern,
                affected_count=component_count,
                base_actions=primary_rule.recommended_actions,
            )
            
            pred = {
                "id": dev_id,
                "label": f"🔮 [予兆] {primary_rule.escalated_state}",
                "severity": "CRITICAL",
                "status": "CRITICAL",
                "prob": round(confidence, 2),
                "type": f"Predictive/{primary_rule.category}",
                "tier": 1,
                "reason": f"Digital Twin Prediction: {primary_rule.time_to_critical_min}min to critical. Root: {primary_msg}",
                "is_prediction": True,
                "prediction_timeline": f"{primary_rule.time_to_critical_min}分後",
                "prediction_time_to_critical_min": primary_rule.time_to_critical_min,
                "prediction_early_warning_hours": primary_rule.early_warning_hours,
                "prediction_affected_count": impact_count,
                "prediction_signal_count": len(matched_signals),
                "prediction_confidence_factors": {"base": primary_rule.base_confidence, "match_quality": primary_quality},
                "recommended_actions": smart_actions,  # LLMベースの動的アクション
                "base_recommended_actions": primary_rule.recommended_actions,  # 元の固定アクション（参考用）
                "runbook_url": primary_rule.runbook_url
            }
            pid = str(uuid.uuid4())
            self.history.append({"prediction_id": pid, "device_id": dev_id, "rule_pattern": primary_rule.pattern, "timestamp": time.time(), "prob": confidence, "anchor_event_time": time.time(), "raw_msg": primary_msg})
            self.storage.save_json_atomic("history", self.history)
            predictions.append(pred)
            processed_devices.add(dev_id)
        return predictions

    def generate_tuning_report(self, days: int = 30) -> Dict[str, Any]:
        return self.tuner.generate_report(days)

    def apply_tuning_proposals_if_auto(self, proposals: List[Dict]) -> Dict:
        applied = []
        skipped = []
        with self.storage.global_lock(timeout_sec=30.0):
            for p in proposals:
                rp = p.get("rule_pattern")
                rec = p.get("apply_recommendation", {})
                if rec.get("apply_mode") != "auto":
                    skipped.append({"rule": rp, "reason": "not_auto"})
                    continue
                prop = p.get("proposal", {})
                pt = float(prop.get("paging_threshold", 0.0))
                lt = float(prop.get("logging_threshold", 0.0))
                old_json_str = self.storage.rule_config_get_json_str(rp)
                rj_str = old_json_str
                if rj_str:
                    d = json.loads(rj_str)
                    d["paging_threshold"] = pt
                    d["logging_threshold"] = lt
                    rj_str = json.dumps(d, ensure_ascii=False)
                success = self.storage.rule_config_upsert(rp, pt, lt, rj_str)
                if success:
                    applied.append({"rule": rp, "paging": pt})
                else:
                    skipped.append({"rule": rp, "reason": "db_write_fail"})
        return {"applied": applied, "skipped": skipped}

    def repair_db_from_rules_json(self) -> bool:
        try:
            path = self.storage.paths["rules"]
            if not os.path.exists(path): return False
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            sanitized = [self._sanitize_rule_data(item) for item in data]
            self.storage._seed_rule_config_from_rules_json(sanitized)
            return True
        except Exception: return False
    # ==============================================================
    # Phase1: Predict API helpers
    # ==============================================================

    def _parse_timestamp(self, ts) -> float:
        if ts is None:
            return time.time()
        if isinstance(ts, (int, float)):
            return float(ts)
        s = str(ts).strip()
        try:
            return float(s)
        except Exception:
            pass
        try:
            from datetime import datetime as _dt
            return _dt.fromisoformat(s.replace("Z", "+00:00")).timestamp()
        except Exception:
            return time.time()

    def _should_ignore(self, msg: str) -> bool:
        m = (msg or "").lower()
        ignore = ["dry-run", "test message", "synthetic-monitor", "healthcheck"]
        return any(ph in m for ph in ignore)

    def _rule_match_simple(self, rule, msg: str):
        """regex + semantic phrase マッチ。(hit, reasons) を返す"""
        reasons = []
        hit = False
        try:
            if rule._compiled_regex and rule._compiled_regex.search(msg or ""):
                hit = True
                reasons.append(f"pattern matched: {rule.pattern}")
        except Exception:
            pass
        if not hit:
            low = (msg or "").lower()
            for sp in (rule.semantic_phrases or []):
                if sp and sp.lower() in low:
                    hit = True
                    reasons.append(f"semantic hit: {sp}")
                    break
        return hit, reasons

    def predict(self, device_id: str, msg: str, timestamp: float,
                attrs: Optional[Dict[str, Any]] = None,
                degradation_level: int = 1,
                source: str = "real") -> List[Dict[str, Any]]:
        """
        EscalationRule ベースの予兆予測。
        degradation_level (1-5): Level に応じて confidence をブースト、
                                  time_to_critical を短縮、early_warning を延長。
        source: "simulation" | "real"
        戻り値は PredictResult.to_dict() のリスト（confidence 降順）。
        """
        try:
            msg_n = self._normalize_msg(msg or "")
        except AttributeError:
            msg_n = (msg or "").strip()
        except Exception:
            msg_n = (msg or "").strip()

        if self._should_ignore(msg_n):
            return []

        _min_conf = float(MIN_PREDICTION_CONFIDENCE)

        # Level ブースト係数（Level1=0.0 → Level5=0.20）
        _level = max(1, min(5, int(degradation_level or 1)))
        _conf_boost    = (_level - 1) * 0.05          # +0.00〜+0.20
        _ttc_factor    = 1.0 - (_level - 1) * 0.12   # ×1.0〜×0.52（短縮）
        _early_factor  = 1.0 + (_level - 1) * 0.20   # ×1.0〜×1.80（延長）
        
        # ★ RUL (Remaining Useful Life) 予測 ──────────────────
        # Temporal GNN論文: time_to_failure = f(degradation_level)
        # Level↑ → 故障が近い → RUL↓
        _ttf_scale = (6 - _level) / 5  # L1=1.0(初期), L5=0.2(末期)
        # L1=100%, L2=80%, L3=60%, L4=40%, L5=20%

        # 影響範囲: children_map から再帰的に配下デバイス数を算出
        def _count_children(dev_id: str, visited=None) -> int:
            if visited is None: visited = set()
            if dev_id in visited: return 0
            visited.add(dev_id)
            children = (self.children_map or {}).get(dev_id, [])
            return len(children) + sum(_count_children(c, visited) for c in children)

        _affected_count = _count_children(device_id)

        results = []
        for rule in (self.rules or []):
            try:
                hit, reasons = self._rule_match_simple(rule, msg_n)
                if not hit:
                    continue
                base_conf = float(getattr(rule, "base_confidence", 0.5) or 0.5)
                conf = min(0.99, base_conf + _conf_boost)
                if conf < _min_conf:
                    continue
                _base_ttc   = int(getattr(rule, "time_to_critical_min", 60) or 60)
                _base_early = int(getattr(rule, "early_warning_hours", 24) or 24)
                _ttc   = max(5,  int(_base_ttc   * _ttc_factor))
                _early = max(1,  int(_base_early * _early_factor))
                
                # ★ RUL計算: early_warning_hours をベースに故障までの時間を算出
                _base_ttf_hours = int(getattr(rule, "early_warning_hours", 336) or 336)
                _ttf_hours = max(1, int(_base_ttf_hours * _ttf_scale))
                # L1=336h(14日), L2=269h(11日), L3=202h(8日), L4=134h(6日), L5=67h(3日)
                
                # 故障予測日時を算出
                from datetime import datetime, timedelta
                _failure_dt = datetime.now() + timedelta(hours=_ttf_hours)
                _failure_dt_str = _failure_dt.strftime("%Y-%m-%d %H:%M")
                
                pr = PredictResult(
                    predicted_state      = str(getattr(rule, "escalated_state", "unknown")),
                    confidence           = conf,
                    rule_pattern         = str(getattr(rule, "pattern", "unknown")),
                    category             = str(getattr(rule, "category", "Generic")),
                    reasons              = reasons,
                    recommended_actions  = list(getattr(rule, "recommended_actions", []) or []),
                    runbook_url          = str(getattr(rule, "runbook_url", "") or ""),
                    criticality          = str(getattr(rule, "criticality", "standard") or "standard"),
                    time_to_critical_min = _ttc,
                    early_warning_hours  = _early,
                    time_to_failure_hours = _ttf_hours,
                    predicted_failure_datetime = _failure_dt_str,
                )
                # ★ シグナル件数ベースの静的ルールで推奨アクションを決定（外部LLM不使用）
                import re as _re_comp
                _all_messages: List[str] = (attrs or {}).get("all_messages", [])
                if not _all_messages:
                    _all_messages = [msg]

                # affected_count: 全メッセージから抽出したインターフェース数 or シグナル件数
                # ※ deg_level * 2 による過大評価は行わない
                _all_components: set = set()
                for _am in _all_messages:
                    _all_components.update(
                        _re_comp.findall(
                            r'\b(?:Gi|Te|Fa|Et)\d+/\d+/\d+|\b(?:Gi|Te|Fa|Et)\d+/\d+', _am or ""))
                _affected_est = len(_all_components) if _all_components else len(_all_messages)

                _rule_pat  = str(getattr(rule, "pattern", "unknown"))
                _base_acts = list(getattr(rule, "recommended_actions", []) or [])

                _smart_acts = self._generate_smart_recommendations(
                    rule_pattern   = _rule_pat,
                    affected_count = _affected_est,
                    base_actions   = _base_acts,
                )
                if _smart_acts != _base_acts:
                    pr.recommended_actions = _smart_acts
                    logger.debug(f"[Static] smart actions applied for {device_id} "
                                 f"(pattern={_rule_pat}, affected={_affected_est})")

                results.append(pr)
            except Exception:
                continue
        results.sort(key=lambda x: x.confidence, reverse=True)
        return [r.to_dict(affected_count=_affected_count, source=source) for r in results]

    def predict_api(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cockpit / Simulator 共通エントリーポイント。
        record_forecast=True (デフォルト) のとき forecast_ledger に登録する。

        ★ バッチ対応: request に "messages" (List[str]) を含めると
           全メッセージを使ってLLM推奨アクションを生成する。
           "msg" は後方互換性のために残す。
        """
        try:
            tenant_id = (request.get("tenant_id") or self.tenant_id or "default").strip().lower()
            device_id = str(request.get("device_id") or "").strip()
            ts        = self._parse_timestamp(request.get("timestamp"))
            if not device_id:
                raise ValueError("device_id is required")

            # ★ "messages" (複数) を優先、なければ "msg" (単一) にフォールバック
            messages_list: List[str] = []
            _raw_messages = request.get("messages")
            if isinstance(_raw_messages, list) and _raw_messages:
                messages_list = [str(m) for m in _raw_messages if m]
            if not messages_list:
                _single = str(request.get("msg") or "").strip()
                if _single:
                    messages_list = [_single]
            if not messages_list:
                raise ValueError("msg or messages is required")

            # 後方互換: 先頭メッセージを "msg" として扱う
            msg = messages_list[0]

            attrs = request.get("attrs") or {}
            if not isinstance(attrs, dict):
                attrs = {"raw_attrs": str(attrs)}

            # ★ 全メッセージを attrs 経由で predict() に伝搬（シグナル件数集計用）
            attrs = dict(attrs)
            attrs["all_messages"] = messages_list

            req   = PredictRequest(tenant_id=tenant_id, device_id=device_id,
                                   msg=msg, timestamp=ts, attrs=attrs)
            _level  = int((attrs or {}).get("degradation_level", 1))
            _source = str((attrs or {}).get("source", "real"))
            preds = self.predict(device_id=device_id, msg=msg, timestamp=ts,
                                 attrs=attrs, degradation_level=_level, source=_source)

            # ★ forecast_ledger への登録はデバイスごとに1回だけ行う
            # (複数メッセージでも重複登録しない)
            record_forecast = bool(request.get("record_forecast", True))
            forecast_ids: List[str] = []
            if record_forecast and preds:
                fid = self._forecast_record(req=req.to_dict(), top_prediction=preds[0])
                if fid:
                    forecast_ids.append(fid)

            return {"ok": True, "input": req.to_dict(),
                    "predictions": preds, "forecast_ids": forecast_ids}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}",
                    "trace": traceback.format_exc()}

    # ==============================================================
    # Phase1: Forecast Ledger DDL（_init_sqlite から呼ばれる）
    # ==============================================================

    def _init_forecast_ledger(self):
        """forecast_ledger テーブルと migration を実施"""
        if not self.storage._conn:
            return
        try:
            with self.storage._db_lock:
                self.storage._conn.execute("""
                    CREATE TABLE IF NOT EXISTS forecast_ledger (
                        forecast_id      TEXT PRIMARY KEY,
                        created_at       REAL,
                        tenant_id        TEXT,
                        device_id        TEXT,
                        rule_pattern     TEXT,
                        predicted_state  TEXT,
                        confidence       REAL,
                        horizon_sec      INTEGER,
                        eval_deadline_ts REAL,
                        source           TEXT,
                        status           TEXT,
                        outcome_type     TEXT,
                        outcome_ts       REAL,
                        outcome_note     TEXT,
                        input_json       TEXT,
                        prediction_json  TEXT
                    )
                """)
                self.storage._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_fl_open "
                    "ON forecast_ledger (status, eval_deadline_ts)")
                self.storage._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_fl_device "
                    "ON forecast_ledger (device_id, created_at)")
                # migration: add source column if missing
                cur = self.storage._conn.cursor()
                cur.execute("PRAGMA table_info(forecast_ledger)")
                cols = [r[1] for r in cur.fetchall()]
                if "source" not in cols:
                    self.storage._conn.execute(
                        "ALTER TABLE forecast_ledger ADD COLUMN source TEXT")
                self.storage._conn.commit()
        except Exception as e:
            logger.warning(f"_init_forecast_ledger: {e}")

    def _forecast_horizon_sec(self, rule_pattern: str) -> int:
        for r in (self.rules or []):
            if (getattr(r, "pattern", "") or "").lower() == (rule_pattern or "").lower():
                ttc = getattr(r, "time_to_critical_min", None)
                if isinstance(ttc, int) and ttc > 0:
                    return max(1800, ttc * 60)
        return 3600

    def _forecast_record(self, req: Dict[str, Any], top_prediction: Dict[str, Any],
                         source: str = "real") -> Optional[str]:
        """
        forecast_ledger に UPSERT（同一 device_id + rule_pattern の open 行を更新）。
        新規の場合は INSERT、既存 open の場合は prediction_json / confidence のみ更新。
        forecast_id を返す。
        """
        if not self.storage._conn:
            return None
        try:
            tenant_id       = str(req.get("tenant_id") or self.tenant_id)
            device_id       = str(req.get("device_id") or "")
            rule_pattern    = str(top_prediction.get("rule_pattern") or "unknown")
            predicted_state = str(top_prediction.get("predicted_state") or "unknown")
            confidence      = float(top_prediction.get("confidence") or 0.0)
            horizon_sec     = self._forecast_horizon_sec(rule_pattern)
            event_ts        = float(req.get("timestamp") or time.time())
            eval_deadline_ts = event_ts + horizon_sec
            input_json      = json.dumps(req, ensure_ascii=False)
            prediction_json = json.dumps(top_prediction, ensure_ascii=False)

            with self.storage._db_lock:
                cur = self.storage._conn.cursor()

                # ★ 同一 device_id + rule_pattern の open レコードを検索
                cur.execute("""
                    SELECT forecast_id FROM forecast_ledger
                    WHERE device_id=? AND rule_pattern=? AND status='open'
                    ORDER BY created_at DESC LIMIT 1
                """, (device_id, rule_pattern))
                existing = cur.fetchone()

                if existing:
                    # ★ 既存 open レコードを更新（推奨アクション・信頼度を最新に）
                    fid = existing[0]
                    self.storage._conn.execute("""
                        UPDATE forecast_ledger
                        SET confidence=?, prediction_json=?, input_json=?,
                            eval_deadline_ts=?, predicted_state=?
                        WHERE forecast_id=?
                    """, (confidence, prediction_json, input_json,
                          eval_deadline_ts, predicted_state, fid))
                else:
                    # ★ 新規 INSERT
                    fid = "f_" + uuid.uuid4().hex[:12]
                    created_at = time.time()
                    self.storage._conn.execute("""
                        INSERT INTO forecast_ledger
                        (forecast_id, created_at, tenant_id, device_id, rule_pattern, predicted_state,
                         confidence, horizon_sec, eval_deadline_ts, source, status,
                         outcome_type, outcome_ts, outcome_note, input_json, prediction_json)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (fid, created_at, tenant_id, device_id, rule_pattern, predicted_state,
                          confidence, horizon_sec, eval_deadline_ts, source, "open",
                          None, None, None, input_json, prediction_json))

                self.storage._conn.commit()
            return fid
        except Exception as e:
            logger.warning(f"_forecast_record: {e}")
            return None

    def forecast_get(self, forecast_id: str) -> Optional[Dict[str, Any]]:
        if not self.storage._conn:
            return None
        try:
            with self.storage._db_lock:
                cur = self.storage._conn.cursor()
                cur.execute("""
                    SELECT forecast_id, created_at, tenant_id, device_id, rule_pattern,
                           predicted_state, confidence, horizon_sec, eval_deadline_ts,
                           source, status, outcome_type, outcome_ts, outcome_note
                    FROM forecast_ledger WHERE forecast_id=?""", (forecast_id,))
                row = cur.fetchone()
            if not row:
                return None
            keys = ["forecast_id","created_at","tenant_id","device_id","rule_pattern",
                    "predicted_state","confidence","horizon_sec","eval_deadline_ts",
                    "source","status","outcome_type","outcome_ts","outcome_note"]
            return dict(zip(keys, row))
        except Exception:
            return None

    def forecast_register_outcome(self, forecast_id: str, outcome_type: str,
                                  outcome_ts=None, note: str = "",
                                  auto: bool = False) -> Dict[str, Any]:
        """
        予見成功判定:
          deadline 以内に OUTCOME_CONFIRMED → status=confirmed, success=True
          deadline 超過後   → status=confirmed_late, success=False
          自動登録 (auto=True) は audit_log に actor="auto" で記録
        """
        if not self.storage._conn:
            return {"ok": False, "reason": "sqlite_disabled"}
        fid = str(forecast_id or "").strip()
        if not fid:
            return {"ok": False, "reason": "missing_forecast_id"}

        ts  = time.time() if outcome_ts is None else self._parse_timestamp(outcome_ts)
        rec = self.forecast_get(fid)
        if not rec:
            return {"ok": False, "reason": "not_found"}
        if rec.get("status") not in ["open"]:
            return {"ok": False, "reason": "not_open", "status": rec.get("status")}

        deadline = float(rec.get("eval_deadline_ts") or 0.0)
        success  = bool(ts <= deadline) if deadline > 0 else False

        if outcome_type == "confirmed_incident":
            new_status = "confirmed" if success else "confirmed_late"
        elif outcome_type == "mitigated":
            new_status = "mitigated"
            success = True
        elif outcome_type == "false_alarm":
            new_status = "false_alarm"
            success = False
        else:
            new_status = "closed"
            success = False

        actor     = "auto" if auto else "operator"
        note_s    = (note or "")[:512]
        rule_pat  = str(rec.get("rule_pattern") or "")

        try:
            with self.storage._db_lock:
                self.storage._conn.execute("""
                    UPDATE forecast_ledger
                    SET status=?, outcome_type=?, outcome_ts=?, outcome_note=?
                    WHERE forecast_id=?""",
                    (new_status, outcome_type, ts, note_s, fid))
                # audit_log に記録
                self.storage.audit_log_generic({
                    "event_id":    str(uuid.uuid4()),
                    "timestamp":   ts,
                    "event_type":  "forecast_outcome",
                    "actor":       actor,
                    "rule_pattern": rule_pat,
                    "details": {"forecast_id": fid, "outcome_type": outcome_type,
                                "success": success, "status": new_status, "auto": auto}
                })
                self.storage._conn.commit()
        except Exception as e:
            return {"ok": False, "reason": str(e)}

        return {"ok": True, "forecast_id": fid, "success": success, "status": new_status}

    def forecast_expire_open(self, now_ts: Optional[float] = None,
                             limit: int = 200) -> Dict[str, Any]:
        """期限切れの open 予兆を expired に更新"""
        if not self.storage._conn:
            return {"ok": False}
        now = float(now_ts or time.time())
        expired = 0
        try:
            with self.storage._db_lock:
                cur = self.storage._conn.cursor()
                cur.execute("""
                    SELECT forecast_id, rule_pattern FROM forecast_ledger
                    WHERE status='open' AND eval_deadline_ts < ?
                    ORDER BY eval_deadline_ts ASC LIMIT ?""", (now, limit))
                rows = cur.fetchall() or []
                for fid, rp in rows:
                    self.storage._conn.execute(
                        "UPDATE forecast_ledger SET status='expired', outcome_type='false_alarm', outcome_ts=? "
                        "WHERE forecast_id=?", (now, fid))
                    expired += 1
                if expired:
                    self.storage._conn.commit()
        except Exception as e:
            logger.warning(f"forecast_expire_open: {e}")
        return {"ok": True, "expired": expired}

    def forecast_auto_resolve(self, device_id: str, outcome_type: str,
                              note: str = "") -> int:
        """
        device_id の open 予兆を自動 outcome 登録。
        cockpit の Execute 成功時・アラーム確定時から呼ばれる。
        解決した件数を返す。
        """
        if not self.storage._conn:
            return 0
        resolved = 0
        try:
            with self.storage._db_lock:
                cur = self.storage._conn.cursor()
                cur.execute("""
                    SELECT forecast_id FROM forecast_ledger
                    WHERE device_id=? AND status='open'
                    ORDER BY created_at DESC""", (device_id,))
                rows = cur.fetchall() or []
            for (fid,) in rows:
                r = self.forecast_register_outcome(
                    fid, outcome_type, note=note, auto=True)
                if r.get("ok"):
                    resolved += 1
        except Exception as e:
            logger.warning(f"forecast_auto_resolve: {e}")
        return resolved

    def forecast_auto_confirm_on_incident(self, device_id: str, scenario: str = "",
                                          note: str = "") -> int:
        """
        障害発生時に該当デバイスの open 予兆を自動的に confirmed_incident に更新
        
        運用実態に即した設計:
        - 運用者が「障害確認済み」を手動登録するのは非現実的
        - 障害シナリオ発生時に自動判定する方が正確
        
        Args:
            device_id: 障害が発生したデバイスID
            scenario: 発生した障害シナリオ名（ログ用）
            note: 追加メモ
        
        Returns:
            confirmed に更新した予兆の件数
        """
        if not self.storage._conn:
            return 0
        confirmed = 0
        auto_note = f"Auto-confirmed on incident: {scenario}" if scenario else "Auto-confirmed on incident"
        if note:
            auto_note += f" | {note}"
        
        try:
            with self.storage._db_lock:
                cur = self.storage._conn.cursor()
                cur.execute("""
                    SELECT forecast_id FROM forecast_ledger
                    WHERE device_id=? AND status='open'
                    ORDER BY created_at DESC""", (device_id,))
                rows = cur.fetchall() or []
            
            for (fid,) in rows:
                r = self.forecast_register_outcome(
                    fid, "confirmed_incident", note=auto_note, auto=True)
                if r.get("ok"):
                    confirmed += 1
                    logger.info(f"Auto-confirmed forecast {fid[:12]} on incident: {scenario}")
        except Exception as e:
            logger.warning(f"forecast_auto_confirm_on_incident: {e}")
        
        return confirmed

    def forecast_list_open(self, device_id: Optional[str] = None,
                           limit: int = 50) -> List[Dict[str, Any]]:
        """open 中の予兆リストを返す（UI表示用）
        
        prediction_json から推奨アクションを取得して返す（最新のLLM強化済みアクション含む）。
        """
        if not self.storage._conn:
            return []
        try:
            with self.storage._db_lock:
                cur = self.storage._conn.cursor()
                if device_id:
                    cur.execute("""
                        SELECT forecast_id, created_at, device_id, rule_pattern,
                               predicted_state, confidence, eval_deadline_ts, source,
                               input_json, prediction_json
                        FROM forecast_ledger
                        WHERE status='open' AND device_id=?
                        ORDER BY created_at DESC LIMIT ?""", (device_id, limit))
                else:
                    cur.execute("""
                        SELECT forecast_id, created_at, device_id, rule_pattern,
                               predicted_state, confidence, eval_deadline_ts, source,
                               input_json, prediction_json
                        FROM forecast_ledger
                        WHERE status='open'
                        ORDER BY confidence DESC, created_at DESC LIMIT ?""", (limit,))
                rows = cur.fetchall() or []
            keys = ["forecast_id","created_at","device_id","rule_pattern",
                    "predicted_state","confidence","eval_deadline_ts","source",
                    "input_json","prediction_json"]
            result = []
            for r in rows:
                d = dict(zip(keys, r))
                # input_json からログメッセージを抽出
                try:
                    if d.get("input_json"):
                        input_data = json.loads(d["input_json"])
                        d["message"] = input_data.get("msg", "")
                except Exception:
                    d["message"] = ""
                # ★ prediction_json から推奨アクション・根拠を抽出（最新LLM強化済み）
                try:
                    if d.get("prediction_json"):
                        pred_data = json.loads(d["prediction_json"])
                        d["recommended_actions"] = pred_data.get("recommended_actions", [])
                        d["reasons"] = pred_data.get("reasons", [])
                        d["criticality"] = pred_data.get("criticality", "standard")
                        d["time_to_critical_min"] = pred_data.get(
                            "time_to_critical_min",
                            pred_data.get("prediction_time_to_critical_min", 0))
                        d["time_to_failure_hours"] = pred_data.get(
                            "time_to_failure_hours",
                            pred_data.get("prediction_time_to_failure_hours", 0))
                        d["predicted_failure_datetime"] = pred_data.get(
                            "predicted_failure_datetime",
                            pred_data.get("prediction_failure_datetime", ""))
                except Exception:
                    d["recommended_actions"] = []
                    d["reasons"] = []
                # prediction_json は返却不要（メモリ節約）
                d.pop("prediction_json", None)
                result.append(d)
            return result
        except Exception:
            return []
