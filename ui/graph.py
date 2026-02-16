# ui/graph.py
import graphviz
from alarm_generator import NodeColor, Alarm
from typing import List

def render_topology_graph(topology: dict, alarms: List[Alarm], analysis_results: List[dict]):
    graph = graphviz.Digraph()
    graph.attr(rankdir='TB')
    graph.attr('node', shape='box', style='rounded,filled', fontname='Helvetica')
    
    alarm_map = {}
    for a in alarms:
        if a.device_id not in alarm_map:
            alarm_map[a.device_id] = {'is_root_cause': False, 'is_silent_suspect': False, 'max_severity': 'INFO'}
        info = alarm_map[a.device_id]
        if a.is_root_cause: info['is_root_cause'] = True
        if a.is_silent_suspect: info['is_silent_suspect'] = True
        
        sev_order = {'CRITICAL': 3, 'WARNING': 2, 'INFO': 1}
        if sev_order.get(a.severity, 0) > sev_order.get(info['max_severity'], 0):
            info['max_severity'] = a.severity
    
    for node_id, node in topology.items():
        node_type = getattr(node, 'type', node.get('type', 'UNKNOWN'))
        metadata = getattr(node, 'metadata', node.get('metadata', {}))
        
        color = NodeColor.NORMAL
        penwidth = "1"
        fontcolor = "black"
        label = f"{node_id}\n({node_type})"
        
        if metadata.get("redundancy_type"): label += f"\n[{metadata['redundancy_type']}]"
        
        if node_id in alarm_map:
            info = alarm_map[node_id]
            if info['is_root_cause']:
                if info['is_silent_suspect']:
                    color = NodeColor.SILENT_FAILURE
                    penwidth = "3"
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
                color = NodeColor.UNREACHABLE
                fontcolor = "#546e7a"
                label += "\n[Unreachable]"
        
        graph.node(node_id, label=label, fillcolor=color, color='black', penwidth=penwidth, fontcolor=fontcolor)
    
    for node_id, node in topology.items():
        parent_id = getattr(node, 'parent_id', node.get('parent_id'))
        if parent_id:
            graph.edge(parent_id, node_id)
            # Add redundancy links
            p_node = topology.get(parent_id)
            rg = getattr(p_node, 'redundancy_group', p_node.get('redundancy_group')) if p_node else None
            if rg:
                for nid, n in topology.items():
                    n_rg = getattr(n, 'redundancy_group', n.get('redundancy_group'))
                    if n_rg == rg and nid != parent_id:
                        graph.edge(nid, node_id)
    return graph
