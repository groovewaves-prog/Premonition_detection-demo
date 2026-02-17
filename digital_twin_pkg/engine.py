# digital_twin_pkg/engine.py (Knowledge Transfer Update)
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
        }


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
                )
                results.append(pr)
            except Exception:
                continue
        results.sort(key=lambda x: x.confidence, reverse=True)
        return [r.to_dict(affected_count=_affected_count, source=source) for r in results]

    def predict_api(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cockpit / Simulator 共通エントリーポイント。
        record_forecast=True (デフォルト) のとき forecast_ledger に登録する。
        """
        try:
            tenant_id = (request.get("tenant_id") or self.tenant_id or "default").strip().lower()
            device_id = str(request.get("device_id") or "").strip()
            msg       = str(request.get("msg") or "").strip()
            ts        = self._parse_timestamp(request.get("timestamp"))
            if not device_id:
                raise ValueError("device_id is required")
            if not msg:
                raise ValueError("msg is required")
            attrs = request.get("attrs") or {}
            if not isinstance(attrs, dict):
                attrs = {"raw_attrs": str(attrs)}

            req   = PredictRequest(tenant_id=tenant_id, device_id=device_id,
                                   msg=msg, timestamp=ts, attrs=attrs)
            _level  = int((attrs or {}).get("degradation_level", 1))
            _source = str((attrs or {}).get("source", "real"))
            preds = self.predict(device_id=device_id, msg=msg, timestamp=ts,
                                 attrs=attrs, degradation_level=_level, source=_source)

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
        """forecast_ledger に1行 INSERT（原子的）。forecast_id を返す。"""
        if not self.storage._conn:
            return None
        try:
            forecast_id     = "f_" + uuid.uuid4().hex[:12]
            created_at      = time.time()
            tenant_id       = str(req.get("tenant_id") or self.tenant_id)
            device_id       = str(req.get("device_id") or "")
            rule_pattern    = str(top_prediction.get("rule_pattern") or "unknown")
            predicted_state = str(top_prediction.get("predicted_state") or "unknown")
            confidence      = float(top_prediction.get("confidence") or 0.0)
            horizon_sec     = self._forecast_horizon_sec(rule_pattern)
            event_ts        = float(req.get("timestamp") or created_at)
            eval_deadline_ts = event_ts + horizon_sec
            input_json      = json.dumps(req, ensure_ascii=False)
            prediction_json = json.dumps(top_prediction, ensure_ascii=False)

            with self.storage._db_lock:
                self.storage._conn.execute("""
                    INSERT INTO forecast_ledger
                    (forecast_id, created_at, tenant_id, device_id, rule_pattern, predicted_state,
                     confidence, horizon_sec, eval_deadline_ts, source, status,
                     outcome_type, outcome_ts, outcome_note, input_json, prediction_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (forecast_id, created_at, tenant_id, device_id, rule_pattern, predicted_state,
                      confidence, horizon_sec, eval_deadline_ts, source, "open",
                      None, None, None, input_json, prediction_json))
                self.storage._conn.commit()
            return forecast_id
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

    def forecast_list_open(self, device_id: Optional[str] = None,
                           limit: int = 50) -> List[Dict[str, Any]]:
        """open 中の予兆リストを返す（UI表示用）"""
        if not self.storage._conn:
            return []
        try:
            with self.storage._db_lock:
                cur = self.storage._conn.cursor()
                if device_id:
                    cur.execute("""
                        SELECT forecast_id, created_at, device_id, rule_pattern,
                               predicted_state, confidence, eval_deadline_ts, source
                        FROM forecast_ledger
                        WHERE status='open' AND device_id=?
                        ORDER BY created_at DESC LIMIT ?""", (device_id, limit))
                else:
                    cur.execute("""
                        SELECT forecast_id, created_at, device_id, rule_pattern,
                               predicted_state, confidence, eval_deadline_ts, source
                        FROM forecast_ledger
                        WHERE status='open'
                        ORDER BY confidence DESC, created_at DESC LIMIT ?""", (limit,))
                rows = cur.fetchall() or []
            keys = ["forecast_id","created_at","device_id","rule_pattern",
                    "predicted_state","confidence","eval_deadline_ts","source"]
            return [dict(zip(keys, r)) for r in rows]
        except Exception:
            return []
