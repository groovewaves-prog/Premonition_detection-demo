import time
import numpy as np
from typing import List, Dict, Any
from .config import *

class AutoTuner:
    """
    Handles threshold simulation, shadow mode evaluation, and proposal generation.
    """
    def __init__(self, engine):
        self.engine = engine # Reference to main engine for data access

    def generate_report(self, days: int) -> Dict[str, Any]:
        current_time = time.time()
        scan_start = current_time - (days * 24 * 3600)
        
        # Load data via engine properties
        hist_buffer_start = scan_start - (7 * 24 * 3600)
        
        # Filter for valid tuning data (exclude suppressed/suspected)
        history_map = {}
        for h in self.engine.history:
            if h["timestamp"] < hist_buffer_start: continue
            if h.get("suppressed"): continue
            if float(h.get("maintenance_prob", 0.0) or 0.0) >= 0.3: continue
            history_map[h["prediction_id"]] = h
            
        outcomes = [o for o in self.engine.outcomes if o["timestamp"] >= scan_start]
        missed_incidents = [i for i in self.engine.incident_register if i["timestamp"] >= scan_start]
        
        # Filter outcomes
        filtered_outcomes = []
        for out in outcomes:
            if out["user_action"] in [OUTCOME_SUSPECTED_MAINTENANCE, OUTCOME_PLANNED_MAINTENANCE]: continue
            if out.get("prediction_id") in history_map:
                filtered_outcomes.append(out)
                
        # Filter missed
        filtered_missed = []
        for inc in missed_incidents:
            if inc.get("suppressed_by_maintenance"): continue
            filtered_missed.append(inc)
            
        # --- Aggregation ---
        rule_stats = {}
        unknown_missed_total = 0
        _shadow_dirty = False
        
        # 1. Outcomes
        for out in filtered_outcomes:
            pid = out.get("prediction_id")
            hist = history_map.get(pid)
            rule = hist.get("rule_pattern", "unknown")
            prob = float(hist.get("prob", 0.0))
            
            if rule not in rule_stats:
                rule_stats[rule] = {"scores_tp": [], "scores_fp": [], "missed_count": 0, "prediction_count": 0, "labeled_count": 0}
            
            rule_stats[rule]["labeled_count"] += 1
            if out["user_action"] in [OUTCOME_CONFIRMED, OUTCOME_MITIGATED]:
                rule_stats[rule]["scores_tp"].append(prob)
            elif out["user_action"] == OUTCOME_FALSE_ALARM:
                rule_stats[rule]["scores_fp"].append(prob)
                
        # 2. Missed
        for inc in filtered_missed:
            rule = inc.get("rule_pattern", "unknown")
            if not rule or rule == "unknown":
                unknown_missed_total += 1
                continue
            if rule not in rule_stats:
                rule_stats[rule] = {"scores_tp": [], "scores_fp": [], "missed_count": 0, "prediction_count": 0, "labeled_count": 0}
            rule_stats[rule]["missed_count"] += 1
            
        # 3. Simulation
        proposals = []
        # Total predictions for coverage (approx from history keys)
        for h in history_map.values():
            if h["timestamp"] >= scan_start:
                r = h.get("rule_pattern", "unknown")
                if r in rule_stats: rule_stats[r]["prediction_count"] += 1

        for rule_ptrn, stats in rule_stats.items():
            # Find rule definition for criticality
            rule_def = next((r for r in self.engine.rules if r.pattern == rule_ptrn), None)
            is_critical = (rule_def and rule_def.criticality == "critical")
            
            # Guards
            scores_tp, scores_fp = stats["scores_tp"], stats["scores_fp"]
            known_missed = stats["missed_count"]
            total_samples = len(scores_tp) + len(scores_fp) + known_missed
            if total_samples < 30: continue
            
            pred_count = stats.get("prediction_count", total_samples)
            coverage_ratio = stats["labeled_count"] / pred_count if pred_count > 0 else 0.0
            
            # Simulation Logic
            current_conf = float(self.engine.MIN_PREDICTION_CONFIDENCE)
            
            # Calc Current FN/Recall
            curr_tp_count = sum(1 for s in scores_tp if s >= current_conf)
            curr_fn_count = (len(scores_tp) - curr_tp_count) + known_missed
            curr_fp_count = sum(1 for s in scores_fp if s >= current_conf)
            curr_recall = curr_tp_count / (curr_tp_count + curr_fn_count) if (curr_tp_count + curr_fn_count) > 0 else 0.0
            
            # Search Best
            best_thresh, best_score = current_conf, -1.0
            recall_floor = 0.98 if is_critical else 0.90
            
            for thresh in np.arange(0.40, 0.96, 0.01):
                sim_tp = sum(1 for s in scores_tp if s >= thresh)
                sim_fn = (len(scores_tp) - sim_tp) + known_missed
                sim_fp = sum(1 for s in scores_fp if s >= thresh)
                
                prec = sim_tp / (sim_tp + sim_fp) if (sim_tp + sim_fp) else 0.0
                rec = sim_tp / (sim_tp + sim_fn) if (sim_tp + sim_fn) else 0.0
                f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) else 0.0
                
                if rec < recall_floor: continue
                
                if is_critical:
                    # Penalize FN increase heavily
                    if sim_fn > curr_fn_count: score = -1.0
                    else: score = f1
                else:
                    score = f1
                
                if score > best_score:
                    best_score = score
                    best_thresh = thresh
            
            # Stability & Auto-Apply Eligibility
            delta = abs(best_thresh - current_conf)
            if delta < 0.01: continue
            
            if is_critical:
                sim_fn_best = (len(scores_tp) - sum(1 for s in scores_tp if s >= best_thresh)) + known_missed
                if sim_fn_best > curr_fn_count: continue

            # Impact
            best_fp = sum(1 for s in scores_fp if s >= best_thresh)
            fp_reduction = (1.0 - (best_fp / max(1, curr_fp_count))) if curr_fp_count > 0 else 0.0
            
            # Check Reasons
            reasons = []
            if coverage_ratio < 0.7: reasons.append("coverage_low")
            if unknown_missed_total > 0: reasons.append("unknown_missed_risk")
            if delta > SHADOW_MAX_THRESHOLD_DELTA: reasons.append("large_delta")
            
            auto_eligible = (len(reasons) == 0)
            apply_mode = "manual"

            # --- Shadow Mode Update ---
            shadow_rec = self.engine.shadow_eval_state.get(rule_ptrn, {})
            proposed_key = f"{best_thresh:.2f}"
            prev_key = shadow_rec.get("proposed_paging_threshold")
            
            if prev_key != proposed_key:
                # Reset if proposal changed
                shadow_rec = {
                    "proposed_paging_threshold": proposed_key,
                    "proposed_at": current_time,
                    "last_seen_at": current_time,
                    "status": "pending",
                    "pass_history": [],
                    "consecutive_passes": 0,
                    "consecutive_pass_window_days": None
                }
                _shadow_dirty = True
            else:
                shadow_rec["last_seen_at"] = current_time
                _shadow_dirty = True
            
            self.engine.shadow_eval_state[rule_ptrn] = shadow_rec
            
            # Evaluate Pass (if time elapsed)
            shadow_note = "shadow_pending"
            if prev_key == proposed_key and shadow_rec.get("proposed_at"):
                # Check Logic
                passed = True
                # Time check
                if (current_time - shadow_rec["proposed_at"]) < (SHADOW_MIN_HOURS_SINCE_PROPOSED * 3600):
                    passed = False # Too early
                
                # Metric checks
                best_fn = (len(scores_tp) - sum(1 for s in scores_tp if s >= best_thresh)) + known_missed
                best_rec = sum(1 for s in scores_tp if s >= best_thresh) / (sum(1 for s in scores_tp if s >= best_thresh) + best_fn) if (sum(1 for s in scores_tp if s >= best_thresh) + best_fn) else 0.0
                
                if (best_fn - curr_fn_count) > SHADOW_MAX_FN_DELTA: passed = False
                if fp_reduction < SHADOW_MIN_FP_REDUCTION: passed = False
                if best_rec < (recall_floor + SHADOW_MIN_RECALL_MARGIN): passed = False
                
                # Push History
                self._push_pass_history(shadow_rec, passed, current_time, days)
                shadow_rec["status"] = "passed" if passed else "pending"
                
                # Promotion Check
                is_promotable = self._check_promotion(shadow_rec, days)
                
                if passed and auto_eligible and is_promotable:
                    apply_mode = "auto"
                    shadow_note = "shadow_promoted"
                else:
                    shadow_note = "shadow_waiting"

            # Construct Proposal
            proposals.append({
                "rule_pattern": rule_ptrn,
                "type": "Critical" if is_critical else "Standard",
                "current_stats": {
                    "TP": len(scores_tp), "FP": len(scores_fp), "FN": curr_fn_count,
                    "recall": round(curr_recall, 3), "coverage": round(coverage_ratio, 2)
                },
                "proposal": {
                    "paging_threshold": round(best_thresh, 2),
                    "logging_threshold": min(current_conf, 0.45)
                },
                "expected_impact": {
                    "fp_reduction": round(fp_reduction, 2),
                    "fn_delta": int(best_fn - curr_fn_count),
                    "recall_best": round(best_rec if 'best_rec' in locals() else 0.0, 3)
                },
                "apply_recommendation": {
                    "apply_mode": apply_mode,
                    "auto_eligible": auto_eligible,
                    "block_reasons": reasons,
                    "shadow_note": shadow_note
                }
            })
            
        if _shadow_dirty:
            self.engine.storage.save_json_atomic("shadow_eval_state", self.engine.shadow_eval_state)
            
        return {"tuning_proposals": proposals}

    def _push_pass_history(self, rec, passed, ts, window):
        hist = rec.get("pass_history", [])
        hist.append({"passed": passed, "checked_at": ts, "window_days": window})
        if len(hist) > SHADOW_PASS_HISTORY_MAX: hist = hist[-SHADOW_PASS_HISTORY_MAX:]
        rec["pass_history"] = hist
        
        # Consecutive logic
        last_win = rec.get("consecutive_pass_window_days")
        if passed:
            if last_win == window: rec["consecutive_passes"] = rec.get("consecutive_passes", 0) + 1
            else: 
                rec["consecutive_passes"] = 1
                rec["consecutive_pass_window_days"] = window
        else:
            rec["consecutive_passes"] = 0
            rec["consecutive_pass_window_days"] = window

    def _check_promotion(self, rec, window):
        return rec.get("consecutive_passes", 0) >= PROMOTE_REQUIRED_CONSECUTIVE_PASSES and rec.get("consecutive_pass_window_days") == window
