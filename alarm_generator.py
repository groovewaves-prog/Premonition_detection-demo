# -*- coding: utf-8 -*-
"""
AIOps Agent - Alarm Generator Module
=====================================
シナリオに基づいてアラームを生成するモジュール

■ ノード色の定義（永続的ルール）
| 状態 | 色 | 条件 |
|------|-----|------|
| 根本原因（サービス停止） | 赤色 #ffcdd2 | 両系障害、Device Down等（CRITICAL） |
| 根本原因（冗長性低下） | 黄色 #fff9c4 | 片系障害、Warning等（WARNING） |
| サイレント障害疑い | 薄紫色 #e1bee7 | 自身はアラームなし、配下に影響 |
| 影響デバイス | グレー #cfd8dc | 上流障害の影響で到達不能 |
| 正常 | グリーン #e8f5e9 | 問題なし |
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


# =====================================================
# ノード色定義（永続的ルール）
# =====================================================
class NodeColor:
    """ノード色の定義"""
    ROOT_CAUSE_CRITICAL = "#ffcdd2"  # 赤色 - サービス停止レベル
    ROOT_CAUSE_WARNING = "#fff9c4"   # 黄色 - 冗長性低下レベル
    SILENT_FAILURE = "#e1bee7"       # 薄紫色 - サイレント障害疑い
    UNREACHABLE = "#cfd8dc"          # グレー - 影響デバイス
    NORMAL = "#e8f5e9"               # グリーン - 正常


@dataclass
class Alarm:
    """アラームを表現するデータクラス"""
    device_id: str
    message: str
    severity: str  # CRITICAL, WARNING, INFO
    is_root_cause: bool = False  # 根本原因フラグ
    is_silent_suspect: bool = False  # サイレント障害疑いフラグ
    
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


def _get_all_downstream_devices(topology: dict, root_id: str) -> List[str]:
    """指定デバイスの配下にある全デバイスを再帰的に取得"""
    downstream = []
    for node_id, node in topology.items():
        if node_id == root_id:
            continue
        # parent_idを取得
        if hasattr(node, 'parent_id'):
            parent_id = node.parent_id
        else:
            parent_id = node.get('parent_id')
        
        if parent_id == root_id:
            downstream.append(node_id)
            # 再帰的に配下を取得
            downstream.extend(_get_all_downstream_devices(topology, node_id))
    
    return downstream


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
    # 複合シナリオ（先にチェック - より具体的なパターン）
    # =====================================================
    if "[Complex]" in scenario or "同時多発" in scenario:
        # FW & AP 同時多発障害
        fw_ids = _find_nodes_by_type(topology, "FIREWALL")
        ap_ids = _find_nodes_by_type(topology, "ACCESS_POINT")
        
        # FW片系障害（根本原因1）
        if fw_ids:
            alarms.append(Alarm(fw_ids[0], "HA State: Degraded", "WARNING", is_root_cause=True))
        
        # AP障害（根本原因2 - サイレント障害の可能性）
        # L2SWは沈黙しているが配下のAPが影響を受けている
        switch_id = _find_node_by_type(topology, "SWITCH", layer=4)
        if switch_id:
            # サイレント障害疑いのスイッチ
            alarms.append(Alarm(switch_id, "Silent Failure Suspected", "WARNING", is_root_cause=True, is_silent_suspect=True))
        
        # 配下のAPがConnection Lost
        for ap_id in ap_ids[:2]:
            alarms.append(Alarm(ap_id, "Connection Lost", "WARNING", is_root_cause=False))
        
        return alarms
    
    # =====================================================
    # WAN複合障害
    # =====================================================
    if "[WAN]" in scenario and "複合" in scenario:
        router_id = _find_node_by_type(topology, "ROUTER")
        if router_id:
            alarms.extend([
                Alarm(router_id, "Power Supply 1 Failed", "WARNING", is_root_cause=True),
                Alarm(router_id, "Fan Fail", "WARNING", is_root_cause=True),
            ])
        return alarms
    
    # =====================================================
    # WAN全回線断
    # =====================================================
    if "WAN全回線断" in scenario:
        router_id = _find_node_by_type(topology, "ROUTER")
        if router_id:
            alarms.extend([
                Alarm(router_id, "BGP Peer Down", "CRITICAL", is_root_cause=True),
                Alarm(router_id, "All Uplinks Down", "CRITICAL", is_root_cause=True),
            ])
            # 配下デバイスにUnreachableアラームを追加
            downstream = _get_all_downstream_devices(topology, router_id)
            for dev_id in downstream:
                alarms.append(Alarm(dev_id, "Device Unreachable", "CRITICAL", is_root_cause=False))
        return alarms
    
    # =====================================================
    # WAN関連シナリオ
    # =====================================================
    if "[WAN]" in scenario:
        router_id = _find_node_by_type(topology, "ROUTER")
        if router_id:
            if "電源障害：両系" in scenario:
                alarms.append(Alarm(router_id, "Power Supply: Dual Loss", "CRITICAL", is_root_cause=True))
                # 配下デバイスにUnreachableアラームを追加
                downstream = _get_all_downstream_devices(topology, router_id)
                for dev_id in downstream:
                    alarms.append(Alarm(dev_id, "Device Unreachable", "CRITICAL", is_root_cause=False))
            elif "電源障害：片系" in scenario:
                alarms.append(Alarm(router_id, "Power Supply 1 Failed", "WARNING", is_root_cause=True))
            elif "BGP" in scenario:
                alarms.append(Alarm(router_id, "BGP Flapping", "WARNING", is_root_cause=True))
            elif "FAN" in scenario:
                alarms.append(Alarm(router_id, "Fan Fail", "WARNING", is_root_cause=True))
            elif "メモリ" in scenario:
                alarms.append(Alarm(router_id, "Memory High", "WARNING", is_root_cause=True))
        return alarms
    
    # =====================================================
    # FW片系障害
    # =====================================================
    if "FW片系障害" in scenario:
        fw_id = _find_node_by_type(topology, "FIREWALL")
        if fw_id:
            alarms.extend([
                Alarm(fw_id, "Heartbeat Loss", "WARNING", is_root_cause=True),
                Alarm(fw_id, "HA State: Degraded", "WARNING", is_root_cause=True),
            ])
        return alarms
    
    # =====================================================
    # FW関連シナリオ
    # =====================================================
    if "[FW]" in scenario:
        fw_id = _find_node_by_type(topology, "FIREWALL")
        if fw_id:
            if "電源障害：両系" in scenario:
                alarms.append(Alarm(fw_id, "Power Supply: Dual Loss", "CRITICAL", is_root_cause=True))
                # 配下デバイスにUnreachableアラームを追加
                downstream = _get_all_downstream_devices(topology, fw_id)
                for dev_id in downstream:
                    alarms.append(Alarm(dev_id, "Device Unreachable", "CRITICAL", is_root_cause=False))
            elif "電源障害：片系" in scenario:
                alarms.append(Alarm(fw_id, "Power Supply 1 Failed", "WARNING", is_root_cause=True))
            elif "FAN" in scenario:
                alarms.append(Alarm(fw_id, "Fan Fail", "WARNING", is_root_cause=True))
            elif "メモリ" in scenario:
                alarms.append(Alarm(fw_id, "Memory High", "WARNING", is_root_cause=True))
        return alarms
    
    # =====================================================
    # L2SWサイレント障害
    # =====================================================
    if "L2SWサイレント障害" in scenario:
        # サイレント障害: 親スイッチは沈黙、配下のAPがConnection Lost
        # L2_SW_01を対象とする（layer=4の最初のスイッチ）
        switch_id = _find_node_by_type(topology, "SWITCH", layer=4)
        
        # スイッチ自体はアラームを出さない（サイレント）が、推定で追加
        if switch_id:
            alarms.append(Alarm(switch_id, "Silent Failure Suspected", "WARNING", is_root_cause=True, is_silent_suspect=True))
            
            # このスイッチの直接配下のAPのみがConnection Lost
            downstream_aps = []
            for node_id, node in topology.items():
                if hasattr(node, 'parent_id'):
                    parent_id = node.parent_id
                    node_type = node.type
                else:
                    parent_id = node.get('parent_id')
                    node_type = node.get('type', '')
                
                if parent_id == switch_id and node_type == "ACCESS_POINT":
                    downstream_aps.append(node_id)
            
            # 配下APのConnection Lost
            for ap_id in downstream_aps:
                alarms.append(Alarm(ap_id, "Connection Lost", "WARNING", is_root_cause=False))
        
        return alarms
    
    # =====================================================
    # L2SW関連シナリオ
    # =====================================================
    if "[L2SW]" in scenario:
        switch_id = _find_node_by_type(topology, "SWITCH", layer=4)
        if switch_id:
            if "電源障害：両系" in scenario:
                alarms.append(Alarm(switch_id, "Power Supply: Dual Loss", "CRITICAL", is_root_cause=True))
                # 配下デバイスにUnreachableアラームを追加
                downstream = _get_all_downstream_devices(topology, switch_id)
                for dev_id in downstream:
                    alarms.append(Alarm(dev_id, "Device Unreachable", "CRITICAL", is_root_cause=False))
            elif "電源障害：片系" in scenario:
                alarms.append(Alarm(switch_id, "Power Supply 1 Failed", "WARNING", is_root_cause=True))
            elif "FAN" in scenario:
                alarms.append(Alarm(switch_id, "Fan Fail", "WARNING", is_root_cause=True))
            elif "メモリ" in scenario:
                alarms.append(Alarm(switch_id, "Memory High", "WARNING", is_root_cause=True))
        return alarms
    
    # =====================================================
    # Core関連シナリオ
    # =====================================================
    if "[Core]" in scenario:
        core_id = _find_node_by_type(topology, "SWITCH", layer=3)
        if core_id:
            if "両系" in scenario:
                alarms.append(Alarm(core_id, "Stack Failure", "CRITICAL", is_root_cause=True))
            else:
                alarms.append(Alarm(core_id, "Stack Member Down", "WARNING", is_root_cause=True))
        return alarms
    
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
