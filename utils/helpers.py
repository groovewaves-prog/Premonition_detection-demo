# utils/helpers.py
import hashlib
import os
from typing import List
from alarm_generator import Alarm
from .const import SCENARIO_IMPACT_MAP, ImpactLevel

def get_scenario_impact_level(scenario: str) -> int:
    for key, value in SCENARIO_IMPACT_MAP.items():
        if key in scenario:
            return value
    return ImpactLevel.DEGRADED_MID

def get_status_from_alarms(scenario: str, alarms: List[Alarm]) -> str:
    if not alarms: return "正常"
    impact = get_scenario_impact_level(scenario)
    if impact >= ImpactLevel.COMPLETE_OUTAGE: return "停止"
    elif impact >= ImpactLevel.DEGRADED_HIGH: return "要対応"
    elif impact >= ImpactLevel.DEGRADED_MID:
        if any(a.severity == "CRITICAL" for a in alarms): return "要対応"
        return "注意"
    elif impact >= ImpactLevel.DOWNSTREAM: return "注意"
    else: return "正常"

def get_status_icon(status: str) -> str:
    return {"停止": "🔴", "要対応": "🟠", "注意": "🟡", "正常": "🟢"}.get(status, "⚪")

def hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]

def load_config_by_id(device_id: str) -> str:
    possible_paths = [f"configs/{device_id}.txt", f"{device_id}.txt"]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f: return f.read()
            except: pass
    return "Config file not found."
