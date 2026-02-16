# digital_twin_pkg/engine.py (Knowledge Transfer Update)
import logging
import time
import json
import uuid
import re
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict

from .config import *
from .rules import EscalationRule, DEFAULT_RULES, MAINTENANCE_SIGNATURES
from .storage import StorageManager
from .audit import AuditBuilder
from .tuning import AutoTuner

try:
    from sentence_transformers import SentenceTransformer
    HAS_BERT = True
except ImportError:
    HAS_BERT = False

logger = logging.getLogger(__name__)

class DigitalTwinEngine:
    def __init__(self, topology: Dict[str, Any], children_map: Optional[Dict[str, List[str]]] = None, tenant_id: str = "default"):
        if not tenant_id or len(tenant_id) > 64: raise ValueError("Invalid tenant_id")
        self.tenant_id = tenant_id.lower()
        self.topology = topology
        self.children_map = children_map or {}
        self.storage = StorageManager(self.tenant_id, BASE_DIR)
        self.tuner = AutoTuner(self)
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
                self.rules = [EscalationRule(**asdict(r)) for r in DEFAULT_RULES]
                self.storage.save_json_atomic("rules", [asdict(r) for r in self.rules])
            else:
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self.rules = [EscalationRule(**self._sanitize_rule_data(item)) for item in data]
                except Exception as e:
                    self.rules = [EscalationRule(**asdict(r)) for r in DEFAULT_RULES]
            self.storage._seed_rule_config_from_rules_json([asdict(r) for r in self.rules])
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

            threshold = MIN_PREDICTION_CONFIDENCE
            if primary_rule.paging_threshold is not None:
                threshold = primary_rule.paging_threshold
            if confidence < threshold: continue

            impact_count = 0
            if dev_id in self.children_map:
                impact_count = len(self.children_map[dev_id])
            
            # --- 予測結果に「運用者向けの具体的な知識」を注入 ---
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
                "recommended_actions": primary_rule.recommended_actions,
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
