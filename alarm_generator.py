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

■ 冗長構成における影響範囲の定義（永続的ルール）
| 冗長タイプ | 片系障害 | 両系障害 |
|------------|----------|----------|
| HA冗長（FW等） | 障害機のみWARNING、配下影響なし | 全機CRITICAL、配下Unreachable |
| PSU冗長（単体機） | 障害機のみWARNING、配下影響なし | 障害機CRITICAL、配下Unreachable |
"""

from typing import List, Dict, Any, Optional, Set
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


# =====================================================
# トポロジー検索ヘルパー関数
# =====================================================
def _find_node_by_type(topology: dict, node_type: str, layer: Optional[int] = None) -> Optional[str]:
    """トポロジーから指定タイプのノードを検索（最初の1つ）"""
    for node_id, node in topology.items():
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


def _find_redundancy_group_members(topology: dict, group_name: str) -> List[str]:
    """冗長グループに属する全メンバーを検索"""
    members = []
    for node_id, node in topology.items():
        if hasattr(node, 'redundancy_group'):
            rg = node.redundancy_group
        else:
            rg = node.get('redundancy_group')
        
        if rg == group_name:
            members.append(node_id)
    return members


def _get_redundancy_group(topology: dict, node_id: str) -> Optional[str]:
    """ノードが属する冗長グループを取得"""
    node = topology.get(node_id)
    if not node:
        return None
    
    if hasattr(node, 'redundancy_group'):
        return node.redundancy_group
    else:
        return node.get('redundancy_group')


def _get_all_downstream_devices(topology: dict, root_ids: List[str]) -> List[str]:
    """
    指定デバイス群の配下にある全デバイスを再帰的に取得
    冗長グループを考慮：冗長グループの場合、グループ全体の配下を取得
    """
    downstream = set()
    visited = set(root_ids)
    
    def _get_children(parent_id: str) -> List[str]:
        children = []
        for node_id, node in topology.items():
            if node_id in visited:
                continue
            
            if hasattr(node, 'parent_id'):
                pid = node.parent_id
            else:
                pid = node.get('parent_id')
            
            # 親IDが直接一致、または親が冗長グループのメンバー
            if pid == parent_id:
                children.append(node_id)
            elif pid:
                # 親が冗長グループのメンバーかチェック
                parent_rg = _get_redundancy_group(topology, pid)
                root_rg = _get_redundancy_group(topology, parent_id)
                if parent_rg and root_rg and parent_rg == root_rg:
                    children.append(node_id)
        return children
    
    # 各root_idから配下を探索
    for root_id in root_ids:
        queue = [root_id]
        while queue:
            current = queue.pop(0)
            children = _get_children(current)
            for child in children:
                if child not in downstream and child not in root_ids:
                    downstream.add(child)
                    visited.add(child)
                    queue.append(child)
    
    return list(downstream)


def _get_downstream_of_single_device(topology: dict, root_id: str) -> List[str]:
    """単一デバイスの直接配下を取得（冗長グループ考慮なし）"""
    downstream = []
    
    def _recurse(parent_id: str):
        for node_id, node in topology.items():
            if hasattr(node, 'parent_id'):
                pid = node.parent_id
            else:
                pid = node.get('parent_id')
            
            if pid == parent_id:
                downstream.append(node_id)
                _recurse(node_id)
    
    _recurse(root_id)
    return downstream


# =====================================================
# アラーム生成メイン関数
# =====================================================
def generate_alarms_for_scenario(topology: dict, scenario: str) -> List[Alarm]:
    """
    シナリオに基づいてアラームを生成
    
    ■ 冗長構成における影響範囲ルール:
    - 片系障害: 障害機のみアラーム、配下への影響なし（冗長が機能）
    - 両系障害: 冗長グループ全体にアラーム、配下はUnreachable
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
    # 複合シナリオ（先にチェック）
    # =====================================================
    if "[Complex]" in scenario or "同時多発" in scenario:
        # FW & AP 同時多発障害
        fw_ids = _find_nodes_by_type(topology, "FIREWALL")
        
        # FW片系障害（根本原因1）- 片系なので配下影響なし
        if fw_ids:
            alarms.append(Alarm(fw_ids[0], "HA State: Degraded", "WARNING", is_root_cause=True))
        
        # L2SWサイレント障害（根本原因2）
        switch_id = _find_node_by_type(topology, "SWITCH", layer=4)
        if switch_id:
            alarms.append(Alarm(switch_id, "Silent Failure Suspected", "WARNING", is_root_cause=True, is_silent_suspect=True))
            
            # このスイッチの直接配下のAPのみがConnection Lost
            for node_id, node in topology.items():
                pid = node.parent_id if hasattr(node, 'parent_id') else node.get('parent_id')
                ntype = node.type if hasattr(node, 'type') else node.get('type', '')
                if pid == switch_id and ntype == "ACCESS_POINT":
                    alarms.append(Alarm(node_id, "Connection Lost", "WARNING", is_root_cause=False))
        
        return alarms
    
    # =====================================================
    # WAN複合障害（電源＆FAN）- 冗長性低下、配下影響なし
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
    # WAN全回線断 - サービス停止、配下すべて影響
    # =====================================================
    if "WAN全回線断" in scenario:
        router_id = _find_node_by_type(topology, "ROUTER")
        if router_id:
            alarms.extend([
                Alarm(router_id, "BGP Peer Down", "CRITICAL", is_root_cause=True),
                Alarm(router_id, "All Uplinks Down", "CRITICAL", is_root_cause=True),
            ])
            # 配下デバイスにUnreachableアラームを追加
            downstream = _get_downstream_of_single_device(topology, router_id)
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
                # 両系障害：デバイスダウン、配下すべて影響
                alarms.append(Alarm(router_id, "Power Supply: Dual Loss (Device Down)", "CRITICAL", is_root_cause=True))
                downstream = _get_downstream_of_single_device(topology, router_id)
                for dev_id in downstream:
                    alarms.append(Alarm(dev_id, "Device Unreachable", "CRITICAL", is_root_cause=False))
            elif "電源障害：片系" in scenario:
                # 片系障害：冗長性低下、配下影響なし（PSU冗長で継続）
                alarms.append(Alarm(router_id, "Power Supply 1 Failed", "WARNING", is_root_cause=True))
            elif "BGP" in scenario:
                alarms.append(Alarm(router_id, "BGP Flapping", "WARNING", is_root_cause=True))
            elif "FAN" in scenario:
                alarms.append(Alarm(router_id, "Fan Fail", "WARNING", is_root_cause=True))
            elif "メモリ" in scenario:
                alarms.append(Alarm(router_id, "Memory High", "WARNING", is_root_cause=True))
        return alarms
    
    # =====================================================
    # FW片系障害 - HA冗長性低下、配下影響なし
    # =====================================================
    if "FW片系障害" in scenario:
        fw_ids = _find_nodes_by_type(topology, "FIREWALL")
        if fw_ids:
            # 片系障害：PRIMARY のみ障害、SECONDARY が引き継ぎ
            primary_fw = fw_ids[0]  # 最初のFW（PRIMARY想定）
            alarms.extend([
                Alarm(primary_fw, "Heartbeat Loss", "WARNING", is_root_cause=True),
                Alarm(primary_fw, "HA State: Degraded", "WARNING", is_root_cause=True),
            ])
            # 配下への影響なし（SECONDARYが引き継ぐため）
        return alarms
    
    # =====================================================
    # FW関連シナリオ
    # 「両系」= 機器内の電源ユニット両系（PSU Redundancy）
    # HA冗長があるため、1台ダウンしても配下への影響なし
    # =====================================================
    if "[FW]" in scenario:
        fw_ids = _find_nodes_by_type(topology, "FIREWALL")
        if fw_ids:
            primary_fw = fw_ids[0]  # FW_01_PRIMARY
            
            if "電源障害：両系" in scenario:
                # PRIMARY の PSU 両系障害 → デバイスダウン
                # ただし SECONDARY が引き継ぐため配下への影響なし
                alarms.append(Alarm(primary_fw, "Power Supply: Dual Loss (Device Down)", "CRITICAL", is_root_cause=True))
                # 配下への影響なし（HA冗長でSECONDARYが引き継ぐ）
            
            elif "電源障害：片系" in scenario:
                # PRIMARY の PSU 片系障害 → 冗長性低下のみ
                alarms.append(Alarm(primary_fw, "Power Supply 1 Failed", "WARNING", is_root_cause=True))
            elif "FAN" in scenario:
                alarms.append(Alarm(primary_fw, "Fan Fail", "WARNING", is_root_cause=True))
            elif "メモリ" in scenario:
                alarms.append(Alarm(primary_fw, "Memory High", "WARNING", is_root_cause=True))
        return alarms
    
    # =====================================================
    # L2SWサイレント障害
    # =====================================================
    if "L2SWサイレント障害" in scenario:
        switch_id = _find_node_by_type(topology, "SWITCH", layer=4)
        
        if switch_id:
            alarms.append(Alarm(switch_id, "Silent Failure Suspected", "WARNING", is_root_cause=True, is_silent_suspect=True))
            
            # このスイッチの直接配下のAPのみがConnection Lost
            for node_id, node in topology.items():
                pid = node.parent_id if hasattr(node, 'parent_id') else node.get('parent_id')
                ntype = node.type if hasattr(node, 'type') else node.get('type', '')
                if pid == switch_id and ntype == "ACCESS_POINT":
                    alarms.append(Alarm(node_id, "Connection Lost", "WARNING", is_root_cause=False))
        
        return alarms
    
    # =====================================================
    # L2SW関連シナリオ
    # =====================================================
    if "[L2SW]" in scenario:
        switch_id = _find_node_by_type(topology, "SWITCH", layer=4)
        if switch_id:
            if "電源障害：両系" in scenario:
                alarms.append(Alarm(switch_id, "Power Supply: Dual Loss (Device Down)", "CRITICAL", is_root_cause=True))
                downstream = _get_downstream_of_single_device(topology, switch_id)
                for dev_id in downstream:
                    alarms.append(Alarm(dev_id, "Device Unreachable", "CRITICAL", is_root_cause=False))
            elif "電源障害：片系" in scenario:
                # 片系障害：冗長性低下、配下影響なし
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
                alarms.append(Alarm(core_id, "Stack Failure (Device Down)", "CRITICAL", is_root_cause=True))
                downstream = _get_downstream_of_single_device(topology, core_id)
                for dev_id in downstream:
                    alarms.append(Alarm(dev_id, "Device Unreachable", "CRITICAL", is_root_cause=False))
            else:
                # 片系：冗長性低下、配下影響なし
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
