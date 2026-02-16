import hashlib
import copy
from typing import Dict, Any

class AuditBuilder:
    @staticmethod
    def hash_file_sha256(path: str) -> str:
        try:
            if not path: return ""
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for c in iter(lambda: f.read(4096), b""): h.update(c)
            return h.hexdigest()
        except: return ""

    @staticmethod
    def build_evidence(shadow_state: Dict, proposal: Dict) -> Dict[str, Any]:
        """Bundle shadow history and proposal assessment for audit trail."""
        evidence = {"rule_pattern": proposal.get("rule_pattern")}
        
        # Shadow state snapshot
        try:
            sh = shadow_state.get(proposal.get("rule_pattern")) or {}
            evidence["shadow"] = {
                "status": sh.get("status"),
                "consecutive_passes": sh.get("consecutive_passes"),
                "last_check_pass": sh.get("last_check_pass"),
                "proposed_at": sh.get("proposed_at"),
                # Deep copy history to freeze it in time
                "pass_history": copy.deepcopy(sh.get("pass_history", []))
            }
        except: evidence["shadow"] = {}
        
        # Proposal snapshot
        try: evidence["agent_assessment"] = copy.deepcopy(proposal.get("agent_assessment", {}))
        except: evidence["agent_assessment"] = {}
        
        try: evidence["expected_impact"] = copy.deepcopy(proposal.get("expected_impact", {}))
        except: evidence["expected_impact"] = {}
        
        return evidence
