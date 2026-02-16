import graphviz
from alarm_generator import NodeColor, Alarm
from typing import List

def render_topology_graph(topology: dict, alarms: List[Alarm], analysis_results: List[dict]):
    """
    トポロジーグラフを描画（以前の描画ロジックを完全再現）
    """
    graph = graphviz.Digraph()
    graph.attr(rankdir='TB')
    graph.attr('node', shape='box', style='rounded,filled', fontname='Helvetica')
    
    # アラーム情報をデバイスIDでマッピング
    alarm_map = {}
    for a in alarms:
        if a.device_id not in alarm_map:
            alarm_map[a.device_id] = {
                'is_root_cause': False,
                'is_silent_suspect': False,
                'max_severity': 'INFO'
            }
        info = alarm_map[a.device_id]
        if a.is_root_cause: info['is_root_cause'] = True
        if a.is_silent_suspect: info['is_silent_suspect'] = True
        
        # 深刻度の更新
        severity_order = {'CRITICAL': 3, 'WARNING': 2, 'INFO': 1}
        if severity_order.get(a.severity, 0) > severity_order.get(info['max_severity'], 0):
            info['max_severity'] = a.severity
    
    # 予兆検知IDのセット
    predicted_ids = {r['id'] for r in analysis_results if r.get('is_prediction')}
    
    for node_id, node in topology.items():
        # 型チェックを厳密に行いAttributeErrorを回避
        if isinstance(node, dict):
            node_type = node.get('type', 'UNKNOWN')
            metadata = node.get('metadata', {})
            redundancy_type = metadata.get('redundancy_type')
        else:
            node_type = getattr(node, 'type', 'UNKNOWN')
            metadata = getattr(node, 'metadata', {})
            redundancy_type = metadata.get('redundancy_type') if isinstance(metadata, dict) else getattr(metadata, 'redundancy_type', None)
        
        # 色決定の優先順位
        color = NodeColor.NORMAL
        penwidth = "1"
        fontcolor = "black"
        label = f"{node_id}\n({node_type})"
        
        if redundancy_type:
            label += f"\n[{redundancy_type}]"
            
        # 1. 予兆（薄紫色）
        if node_id in predicted_ids:
            color = "#E1BEE7"
            penwidth = "4"
            fontcolor = "#4A148C"
            label += "\n🔮 [PREDICTION]"

        # 2. アラームに基づく上書き（以前の仕様）
        if node_id in alarm_map:
            info = alarm_map[node_id]
            if info['is_root_cause']:
                if info['is_silent_suspect']:
                    color = NodeColor.SILENT_FAILURE
                    label += "\n[SILENT SUSPECT]"
                elif info['max_severity'] == 'CRITICAL':
                    color = NodeColor.ROOT_CAUSE_CRITICAL
                    penwidth = "3"
                    label += "\n[ROOT CAUSE]"
                else:
                    color = NodeColor.ROOT_CAUSE_WARNING
                    penwidth = "2"
                    label += "\n[WARNING]"
            else:
                # 影響デバイス（グレー）
                color = NodeColor.UNREACHABLE
                fontcolor = "#546e7a"
                label += "\n[Unreachable]"
        
        graph.node(node_id, label=label, fillcolor=color, color='black', penwidth=penwidth, fontcolor=fontcolor)
    
    # リンク描画と冗長グループ線
    for node_id, node in topology.items():
        parent_id = node.get('parent_id') if isinstance(node, dict) else getattr(node, 'parent_id', None)
        if parent_id:
            graph.edge(parent_id, node_id)
            # 冗長ペアの取得
            p_node = topology.get(parent_id)
            if p_node:
                rg = p_node.get('redundancy_group') if isinstance(p_node, dict) else getattr(p_node, 'redundancy_group', None)
                if rg:
                    for nid, n in topology.items():
                        n_rg = n.get('redundancy_group') if isinstance(n, dict) else getattr(n, 'redundancy_group', None)
                        if n_rg == rg and nid != parent_id:
                            graph.edge(nid, node_id)
    return graph
