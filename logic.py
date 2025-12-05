"""
Google Antigravity AIOps Agent - ロジックモジュール (Message Preservation)
HA構成時の判定において、具体的なアラームメッセージを保持するように修正。
"""

from typing import List, Dict, Set, Optional
from dataclasses import dataclass
from data import TOPOLOGY, NetworkNode

@dataclass
class Alarm:
    device_id: str
    message: str
    severity: str # CRITICAL, WARNING, INFO

@dataclass
class InferenceResult:
    root_cause_node: Optional[NetworkNode]
    root_cause_reason: str
    sop_key: str
    related_alarms: List[Alarm]
    severity: str = "CRITICAL"

class CausalInferenceEngine:
    def __init__(self, topology: Dict[str, NetworkNode]):
        self.topology = topology

    def analyze_alarms(self, alarms: List[Alarm]) -> InferenceResult:
        alarmed_device_ids = {a.device_id for a in alarms}
        # アラームIDからメッセージを引けるようにマップ化
        alarm_map = {a.device_id: a for a in alarms}

        sorted_alarms = sorted(
            alarms, 
            key=lambda a: self.topology[a.device_id].layer if a.device_id in self.topology else 999
        )
        
        if not sorted_alarms:
            return InferenceResult(None, "アラームなし", "DEFAULT", [], "INFO")

        top_alarm = sorted_alarms[0]
        top_node = self.topology.get(top_alarm.device_id)
        
        if not top_node:
             return InferenceResult(None, "不明なデバイス", "DEFAULT", alarms, "UNKNOWN")

        # A. 冗長性ルール (HA構成)
        if top_node.redundancy_group:
            # alarm_mapを渡して、具体的なメッセージを取得できるようにする
            return self._analyze_redundancy(top_node, alarmed_device_ids, alarms, alarm_map)

        # B. サイレント障害推論
        if top_node.parent_id:
            silent_res = self._check_silent_failure_for_parent(top_node.parent_id, alarmed_device_ids)
            if silent_res:
                return silent_res

        # C. 単一機器障害
        root_severity = top_alarm.severity
        return InferenceResult(
            root_cause_node=top_node,
            root_cause_reason=f"階層ルール: 最上位デバイス {top_node.id} でアラーム検知。詳細: [{top_alarm.message}]",
            sop_key="HIERARCHY_FAILURE",
            related_alarms=alarms,
            severity=root_severity
        )

    def _analyze_redundancy(self, node: NetworkNode, alarmed_ids: Set[str], alarms: List[Alarm], alarm_map: Dict[str, Alarm]) -> InferenceResult:
        group_members = [n for n in self.topology.values() if n.redundancy_group == node.redundancy_group]
        down_members = [n for n in group_members if n.id in alarmed_ids]
        
        # 障害の詳細メッセージを取得 (例: "Fan Fail", "Power Supply Failed")
        error_details = []
        for m in down_members:
            if m.id in alarm_map:
                error_details.append(f"{m.id}: {alarm_map[m.id].message}")
        details_str = ", ".join(error_details)

        if len(down_members) == len(group_members):
            # 両系ダウン
            return InferenceResult(
                root_cause_node=node,
                root_cause_reason=f"冗長性ルール: HAグループ {node.redundancy_group} 全停止。詳細: [{details_str}]",
                sop_key="HA_TOTAL_FAILURE",
                related_alarms=alarms,
                severity="CRITICAL"
            )
        else:
            # 片系ダウン
            return InferenceResult(
                root_cause_node=node,
                # 【修正】ここで具体的なアラーム内容(details_str)を含める
                root_cause_reason=f"冗長性ルール: HAグループ {node.redundancy_group} 片系障害 (稼働継続)。検知内容: [{details_str}]",
                sop_key="HA_PARTIAL_FAILURE",
                related_alarms=alarms,
                severity="WARNING"
            )

    def _check_silent_failure_for_parent(self, parent_id: str, alarmed_ids: Set[str]) -> Optional[InferenceResult]:
        parent_node = self.topology.get(parent_id)
        if not parent_node:
            return None
            
        children = [n for n in self.topology.values() if n.parent_id == parent_id]
        children_down = sum(1 for c in children if c.id in alarmed_ids)
        
        if len(children) > 0 and children_down == len(children):
             return InferenceResult(
                root_cause_node=parent_node,
                root_cause_reason=f"サイレント障害推論: 親デバイス {parent_id} 無応答、配下全滅。電源断またはハングアップの疑い。",
                sop_key="SILENT_FAILURE",
                related_alarms=[],
                severity="CRITICAL"
            )
        return None
