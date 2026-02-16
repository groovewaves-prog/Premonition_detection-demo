import graphviz
from alarm_generator import NodeColor, Alarm
from typing import List

def render_topology_graph(topology: dict, alarms: List[Alarm], analysis_results: List[dict]):
    """
    トポロジーグラフを描画（オブジェクト/辞書両対応版）
    """
    graph = graphviz.Digraph()
    graph.attr(rankdir='TB')
    graph.attr('node', shape='box', style='rounded,filled', fontname='Helvetica')
    
    # アラーム情報をマッピング
    alarm_map = {}
    for a in alarms:
        if a.device_id not in alarm_map:
            alarm_map[a.device_id] = {
                'is_root_cause': False,
                'is_silent_suspect': False,
                'max_severity': 'INFO',
                'messages': []
            }
        info = alarm_map[a.device_id]
        info['messages'].append(a.message)
        if a.is_root_cause:
            info['is_root_cause'] = True
        if a.is_silent_suspect:
            info['is_silent_suspect'] = True
        
        severity_order = {'CRITICAL': 3, 'WARNING': 2, 'INFO': 1}
        if severity_order.get(a.severity, 0) > severity_order.get(info['max_severity'], 0):
            info['max_severity'] = a.severity
    
    for node_id, node in topology.items():
        # --- 安全な属性取得ロジック (Fix: AttributeError) ---
        if isinstance(node, dict):
            node_type = node.get('type', 'UNKNOWN')
            metadata = node.get('metadata', {})
            redundancy_type = metadata.get('redundancy_type')
        else:
            # NetworkNodeオブジェクトの場合
            node_type = getattr(node, 'type', 'UNKNOWN')
            metadata = getattr(node, 'metadata', {})
            # metadataが辞書かオブジェクトかで分岐
            if isinstance(metadata, dict):
                redundancy_type = metadata.get('redundancy_type')
            else:
                redundancy_type = getattr(metadata, 'redundancy_type', None)
        
        # 色決定ロジック
        color = NodeColor.NORMAL
        penwidth = "1"
        fontcolor = "black"
        label = f"{node_id}\n({node_type})"
        status_label = ""
        
        if redundancy_type:
            label += f"\n[{redundancy_type}]"
        
        if node_id in alarm_map:
            info = alarm_map[node_id]
            if info['is_root_cause']:
                if info['is_silent_suspect']:
                    color = NodeColor.SILENT_FAILURE
                    penwidth = "3"
                    status_label = "\n[SILENT SUSPECT]"
                elif info['max_severity'] == 'CRITICAL':
                    color = NodeColor.ROOT_CAUSE_CRITICAL
                    penwidth = "3"
                    status_label = "\n[ROOT CAUSE]"
                else:
                    color = NodeColor.ROOT_CAUSE_WARNING
                    penwidth = "2"
                    status_label = "\n[WARNING]"
            else:
                color = NodeColor.UNREACHABLE
                fontcolor = "#546e7a"
                status_label = "\n[Unreachable]"
        
        label += status_label
        graph.node(node_id, label=label, fillcolor=color, color='black', penwidth=penwidth, fontcolor=fontcolor)
    
    # リンク描画
    for node_id, node in topology.items():
        if isinstance(node, dict):
            parent_id = node.get('parent_id')
        else:
            parent_id = getattr(node, 'parent_id', None)
        
        if parent_id:
            graph.edge(parent_id, node_id)
            
            # 冗長リンクの描画
            p_node = topology.get(parent_id)
            if p_node:
                if isinstance(p_node, dict):
                    rg = p_node.get('redundancy_group')
                else:
                    rg = getattr(p_node, 'redundancy_group', None)
                
                if rg:
                    for nid, n in topology.items():
                        if isinstance(n, dict):
                            n_rg = n.get('redundancy_group')
                        else:
                            n_rg = getattr(n, 'redundancy_group', None)
                            
                        if n_rg == rg and nid != parent_id:
                            graph.edge(nid, node_id)
    return graph
