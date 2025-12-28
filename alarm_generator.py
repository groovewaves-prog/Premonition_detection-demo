# -*- coding: utf-8 -*-
"""
AIOps Agent - Alarm Generator Module
=====================================
シナリオに基づいてアラームを生成するモジュール
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Alarm:
    """アラームを表現するデータクラス"""
    device_id: str
    message: str
    severity: str  # CRITICAL, WARNING, INFO
    
    def __post_init__(self):
        valid_severities = {"CRITICAL", "WARNING", "INFO"}
        if self.severity not in valid_severities:
            self.severity = "WARNING"


def _find_node_by_type(topology: dict, node_type: str, layer: Optional[int] = None) -> Optional[str]:
    """トポロジーから指定タイプのノードを検索"""
    for node_id, node in topology.items():
        # NetworkNode オブジェクトまたは dict の両方に対応
        if hasattr(node, 'type'):
            n_type = node.type
            n_layer = node.layer
        else:
            n_type = node.get('type', '')
            n_layer = node.get('layer', 99)
        
        if n_type == node_type:
            if layer is None or n_layer == layer:
                return node_id
    return None


def _find_nodes_by_type(topology: dict, node_type: str) -> List[str]:
    """トポロジーから指定タイプの全ノードを検索"""
    results = []
    for node_id, node in topology.items():
        if hasattr(node, 'type'):
            n_type = node.type
        else:
            n_type = node.get('type', '')
        
        if n_type == node_type:
            results.append(node_id)
    return results


def generate_alarms_for_scenario(topology: dict, scenario: str) -> List[Alarm]:
    """
    シナリオに基づいてアラームを生成
    
    Args:
        topology: トポロジー辞書
        scenario: シナリオ名
    
    Returns:
        アラームのリスト
    """
    if not topology or not scenario:
        return []
    
    # 正常系・選択なし
    if "---" in scenario or "正常" in scenario:
        return []
    
    if "Live" in scenario or "[Live]" in scenario:
        return []
    
    alarms = []
    
    # =====================================================
    # WAN関連シナリオ
    # =====================================================
    if "WAN全回線断" in scenario:
        router_id = _find_node_by_type(topology, "ROUTER")
        if router_id:
            alarms.extend([
                Alarm(router_id, "BGP Peer Down", "CRITICAL"),
                Alarm(router_id, "All Uplinks Down", "CRITICAL"),
            ])
    
    elif "[WAN]" in scenario or "WAN" in scenario:
        router_id = _find_node_by_type(topology, "ROUTER")
        if router_id:
            if "電源障害：両系" in scenario:
                alarms.append(Alarm(router_id, "Power Supply: Dual Loss", "CRITICAL"))
            elif "電源障害：片系" in scenario:
                alarms.append(Alarm(router_id, "Power Supply 1 Failed", "WARNING"))
            elif "BGP" in scenario:
                alarms.append(Alarm(router_id, "BGP Flapping", "WARNING"))
            elif "FAN" in scenario:
                alarms.append(Alarm(router_id, "Fan Fail", "WARNING"))
            elif "メモリ" in scenario:
                alarms.append(Alarm(router_id, "Memory High", "WARNING"))
            elif "複合" in scenario:
                alarms.extend([
                    Alarm(router_id, "Power Supply 1 Failed", "WARNING"),
                    Alarm(router_id, "Fan Fail", "WARNING"),
                ])
    
    # =====================================================
    # FW関連シナリオ
    # =====================================================
    elif "FW片系障害" in scenario or ("[FW]" in scenario and "片系" in scenario):
        fw_id = _find_node_by_type(topology, "FIREWALL")
        if fw_id:
            alarms.extend([
                Alarm(fw_id, "Heartbeat Loss", "WARNING"),
                Alarm(fw_id, "HA State: Degraded", "WARNING"),
            ])
    
    elif "[FW]" in scenario or "FW" in scenario:
        fw_id = _find_node_by_type(topology, "FIREWALL")
        if fw_id:
            if "電源障害：両系" in scenario:
                alarms.append(Alarm(fw_id, "Power Supply: Dual Loss", "CRITICAL"))
            elif "電源障害：片系" in scenario:
                alarms.append(Alarm(fw_id, "Power Supply 1 Failed", "WARNING"))
            elif "FAN" in scenario:
                alarms.append(Alarm(fw_id, "Fan Fail", "WARNING"))
            elif "メモリ" in scenario:
                alarms.append(Alarm(fw_id, "Memory High", "WARNING"))
    
    # =====================================================
    # L2SW関連シナリオ
    # =====================================================
    elif "L2SWサイレント障害" in scenario:
        # サイレント障害: 親スイッチは沈黙、配下のAPがConnection Lost
        switch_id = _find_node_by_type(topology, "SWITCH", layer=4)
        ap_ids = _find_nodes_by_type(topology, "ACCESS_POINT")
        
        # APのConnection Lost（親スイッチの障害を示唆）
        for ap_id in ap_ids[:3]:  # 最大3つ
            alarms.append(Alarm(ap_id, "Connection Lost", "WARNING"))
    
    elif "[L2SW]" in scenario or "L2SW" in scenario:
        switch_id = _find_node_by_type(topology, "SWITCH", layer=4)
        if switch_id:
            if "電源障害：両系" in scenario:
                alarms.append(Alarm(switch_id, "Power Supply: Dual Loss", "CRITICAL"))
            elif "電源障害：片系" in scenario:
                alarms.append(Alarm(switch_id, "Power Supply 1 Failed", "WARNING"))
            elif "FAN" in scenario:
                alarms.append(Alarm(switch_id, "Fan Fail", "WARNING"))
            elif "メモリ" in scenario:
                alarms.append(Alarm(switch_id, "Memory High", "WARNING"))
    
    # =====================================================
    # 複合シナリオ
    # =====================================================
    elif "[Complex]" in scenario or "同時多発" in scenario:
        fw_id = _find_node_by_type(topology, "FIREWALL")
        ap_ids = _find_nodes_by_type(topology, "ACCESS_POINT")
        
        if fw_id:
            alarms.append(Alarm(fw_id, "HA State: Degraded", "WARNING"))
        
        for ap_id in ap_ids[:2]:
            alarms.append(Alarm(ap_id, "Connection Lost", "WARNING"))
    
    # =====================================================
    # Core関連シナリオ
    # =====================================================
    elif "[Core]" in scenario:
        core_id = _find_node_by_type(topology, "SWITCH", layer=3)
        if core_id:
            if "両系" in scenario:
                alarms.append(Alarm(core_id, "Stack Failure", "CRITICAL"))
            else:
                alarms.append(Alarm(core_id, "Stack Member Down", "WARNING"))
    
    return alarms


def get_alarm_summary(alarms: List[Alarm]) -> Dict[str, Any]:
    """アラームのサマリーを生成"""
    if not alarms:
        return {
            "total": 0,
            "critical": 0,
            "warning": 0,
            "info": 0,
            "devices": [],
            "status": "正常"
        }
    
    critical = sum(1 for a in alarms if a.severity == "CRITICAL")
    warning = sum(1 for a in alarms if a.severity == "WARNING")
    info = sum(1 for a in alarms if a.severity == "INFO")
    devices = list(set(a.device_id for a in alarms))
    
    if critical > 0:
        status = "停止"
    elif warning > 0:
        status = "要対応"
    else:
        status = "注意"
    
    return {
        "total": len(alarms),
        "critical": critical,
        "warning": warning,
        "info": info,
        "devices": devices,
        "status": status
    }
