# digital_twin_pkg/gnn.py - Graph Neural Network モジュール (Heterogeneous GNN)

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    # ★変更: HeteroData, HeteroConv, Linear などを追加
    from torch_geometric.nn import GATConv, SAGEConv, HeteroConv, Linear
    from torch_geometric.data import HeteroData
    HAS_PYTORCH_GEOMETRIC = True
except ImportError:
    HAS_PYTORCH_GEOMETRIC = False
    logger.warning("PyTorch Geometric not available. GNN features disabled.")
    nn = None
    F = None


if HAS_PYTORCH_GEOMETRIC:
    class HeteroNetworkGNN(nn.Module):
        """
        ネットワークトポロジーのための Heterogeneous Graph Neural Network
        
        エッジの種類に応じて異なるConvolutionを適用し、
        「親子関係（波及）」と「冗長構成（補完）」をAIに区別させる。
        """
        def __init__(
            self,
            input_dim: int = 768,  # BERT embedding dimension
            hidden_dim: int = 128,
            output_dim: int = 64,
            num_layers: int = 3,
            dropout: float = 0.2
        ):
            super().__init__()
            self.dropout = dropout
            
            # ノード特徴量の次元削減（プロジェクション）
            self.node_proj = Linear(input_dim, hidden_dim)
            
            # 異種グラフ畳み込み層 (HeteroConv)
            self.convs = nn.ModuleList()
            for i in range(num_layers):
                conv = HeteroConv({
                    # 親子関係: カスケード障害の波及をAttention(GAT)で学習
                    ('device', 'depends_on', 'device'): GATConv(-1, hidden_dim, add_self_loops=False),
                    # 冗長関係: 互いの状態を平均化(SAGE)して補完関係を学習
                    ('device', 'redundant_with', 'device'): SAGEConv(-1, hidden_dim)
                }, aggr='sum')
                self.convs.append(conv)
            
            # 最終予測層
            self.fc_confidence = nn.Linear(hidden_dim, 1)
            self.fc_time_to_failure = nn.Linear(hidden_dim, 1)
        
        def forward(self, x_dict, edge_index_dict):
            # 初期プロジェクション
            x = self.node_proj(x_dict['device'])
            x = F.relu(x)
            x_dict_current = {'device': x}
            
            # GNNレイヤーの伝播 (Residual connections)
            for conv in self.convs:
                x_in = x_dict_current['device']
                x_dict_current = conv(x_dict_current, edge_index_dict)
                x_out = x_dict_current['device']
                
                x_out = F.relu(x_out)
                x_out = F.dropout(x_out, p=self.dropout, training=self.training)
                
                # Residual connection
                if x_in.shape == x_out.shape:
                    x_out = x_out + x_in
                
                x_dict_current['device'] = x_out
            
            out = x_dict_current['device']
            confidence = torch.sigmoid(self.fc_confidence(out))
            time_to_failure = F.relu(self.fc_time_to_failure(out))
            
            return confidence, time_to_failure
else:
    class HeteroNetworkGNN:
        """Dummy class when PyTorch Geometric is not installed"""
        def __init__(self, *args, **kwargs): pass


class GNNPredictionEngine:
    def __init__(
        self,
        topology: Dict[str, Any],
        children_map: Dict[str, List[str]],
        model_path: Optional[str] = None
    ):
        self.topology = topology
        self.children_map = children_map
        self.model = None
        # トポロジキャッシュ用変数
        self._cached_topology_structure = None 
        
        if not HAS_PYTORCH_GEOMETRIC:
            logger.warning("PyTorch Geometric not available. GNN features disabled.")
            return
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # ★変更: HeteroNetworkGNN に切り替え
        self.model = HeteroNetworkGNN().to(self.device)
        
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                logger.info(f"Loaded pretrained GNN model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model from {model_path}: {e}")
    
    def topology_to_graph(
        self,
        alarm_embeddings: Dict[str, np.ndarray],
        device_states: Optional[Dict[str, Dict]] = None
    ) -> Optional[HeteroData]:
        if not HAS_PYTORCH_GEOMETRIC:
            return None
        
        # ========================================================
        # ★推論の効率化: グラフ構造のキャッシュ
        # ネットワークの形(エッジ)は変わらないため、初回のみ計算する
        # ========================================================
        if self._cached_topology_structure is None:
            device_ids = list(self.topology.keys())
            device_to_idx = {dev_id: idx for idx, dev_id in enumerate(device_ids)}
            
            # 親子関係エッジ (depends_on)
            depends_edges = []
            for parent_id, children in self.children_map.items():
                if parent_id not in device_to_idx: continue
                p_idx = device_to_idx[parent_id]
                for child_id in children:
                    if child_id not in device_to_idx: continue
                    c_idx = device_to_idx[child_id]
                    depends_edges.append([p_idx, c_idx])
                    depends_edges.append([c_idx, p_idx])
            
            # 冗長関係エッジ (redundant_with)
            redundant_groups = {}
            for dev_id, attrs in self.topology.items():
                rg = attrs.get('redundancy_group') if isinstance(attrs, dict) else getattr(attrs, 'redundancy_group', None)
                if rg:
                    redundant_groups.setdefault(rg, []).append(device_to_idx[dev_id])
            
            redundant_edges = []
            for members in redundant_groups.values():
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        redundant_edges.append([members[i], members[j]])
                        redundant_edges.append([members[j], members[i]])
            
            # エッジが0本の場合のエラー防止用ダミーエッジ
            if not depends_edges: depends_edges = [[0, 0]]
            if not redundant_edges: redundant_edges = [[0, 0]]
            
            self._cached_topology_structure = {
                'device_ids': device_ids,
                'device_to_idx': device_to_idx,
                'depends_on': torch.tensor(depends_edges, dtype=torch.long).t().contiguous(),
                'redundant_with': torch.tensor(redundant_edges, dtype=torch.long).t().contiguous()
            }
        
        cache = self._cached_topology_structure
        device_ids = cache['device_ids']
        
        # ノード特徴量（アラーム情報）の構築のみ毎回実行
        node_features = []
        for dev_id in device_ids:
            if dev_id in alarm_embeddings:
                feature = alarm_embeddings[dev_id]
            else:
                feature = np.zeros(768)
            node_features.append(feature)
        
        x = torch.tensor(np.array(node_features), dtype=torch.float)
        
        # ★変更: HeteroData オブジェクトの構築
        data = HeteroData()
        data['device'].x = x
        data['device', 'depends_on', 'device'].edge_index = cache['depends_on']
        data['device', 'redundant_with', 'device'].edge_index = cache['redundant_with']
        
        data.device_ids = device_ids
        data.device_to_idx = cache['device_to_idx']
        
        return data
    
    def predict_with_gnn(
        self,
        alarm_embeddings: Dict[str, np.ndarray],
        target_device_id: str
    ) -> Tuple[float, float]:
        if not HAS_PYTORCH_GEOMETRIC or self.model is None:
            return 0.5, 336.0
        
        data = self.topology_to_graph(alarm_embeddings)
        if data is None or target_device_id not in data.device_to_idx:
            return 0.5, 336.0
        
        target_idx = data.device_to_idx[target_device_id]
        data = data.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            # ★変更: 辞書型の特徴量とエッジを渡す
            confidence, time_to_failure = self.model(data.x_dict, data.edge_index_dict)
        
        target_confidence = float(confidence[target_idx].cpu().numpy())
        target_ttf = float(time_to_failure[target_idx].cpu().numpy())
        
        return target_confidence, target_ttf
    
    def train_on_historical_data(
        self,
        training_data: List[Dict],
        epochs: int = 100,
        lr: float = 0.001
    ):
        if not HAS_PYTORCH_GEOMETRIC or self.model is None:
            logger.warning("GNN training not available")
            return
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion_confidence = nn.BCELoss()
        criterion_ttf = nn.MSELoss()
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for sample in training_data:
                optimizer.zero_grad()
                
                data = self.topology_to_graph(sample['alarm_embeddings'])
                if data is None: continue
                
                data = data.to(self.device)
                
                # ★変更: 辞書型でForward
                pred_conf, pred_ttf = self.model(data.x_dict, data.edge_index_dict)
                
                target_idx = data.device_to_idx.get(sample['device_id'])
                if target_idx is None: continue
                
                target_conf = torch.tensor([1.0 if sample['actual_failure'] else 0.0],
                                          dtype=torch.float, device=self.device)
                target_ttf_val = torch.tensor([sample['time_to_failure']],
                                            dtype=torch.float, device=self.device)
                
                loss_conf = criterion_confidence(pred_conf[target_idx], target_conf)
                loss_ttf = criterion_ttf(pred_ttf[target_idx], target_ttf_val)
                loss = loss_conf + 0.1 * loss_ttf
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(training_data)
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


def create_gnn_engine(topology: dict, children_map: dict) -> Optional[GNNPredictionEngine]:
    if not HAS_PYTORCH_GEOMETRIC:
        return None
    try:
        return GNNPredictionEngine(topology, children_map)
    except Exception as e:
        logger.error(f"Failed to create Hetero GNN engine: {e}")
        return None
