# digital_twin_pkg/gnn.py - Graph Neural Network モジュール

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    HAS_PYTORCH_GEOMETRIC = True
except ImportError:
    HAS_PYTORCH_GEOMETRIC = False
    logger.warning("PyTorch Geometric not available. GNN features disabled.")
    # Define dummy classes to prevent NameError
    nn = None
    F = None


if HAS_PYTORCH_GEOMETRIC:
    class NetworkGNN(nn.Module):
        """
        ネットワークトポロジーのためのGraph Neural Network
        
        ノード: ネットワーク機器
        エッジ: 接続関係（親子、冗長性グループ）
        特徴量: アラーム埋め込み、デバイス属性
        """
        
        def __init__(
            self,
            input_dim: int = 768,  # BERT embedding dimension
            hidden_dim: int = 128,
            output_dim: int = 64,
            num_layers: int = 3,
            dropout: float = 0.2,
            use_attention: bool = True
        ):
            super().__init__()
            
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.num_layers = num_layers
            self.dropout = dropout
            self.use_attention = use_attention
            
            # Input projection
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            
            # GNN layers
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            
            for i in range(num_layers):
                in_channels = hidden_dim
                out_channels = hidden_dim if i < num_layers - 1 else output_dim
                
                if use_attention:
                    # Graph Attention Network
                    conv = GATConv(in_channels, out_channels, heads=4, concat=False)
                else:
                    # Graph Convolutional Network
                    conv = GCNConv(in_channels, out_channels)
                
                self.convs.append(conv)
                
                if i < num_layers - 1:
                    self.bns.append(nn.BatchNorm1d(out_channels))
            
            # Output layers
            self.fc_confidence = nn.Linear(output_dim, 1)
            self.fc_time_to_failure = nn.Linear(output_dim, 1)
        
        def forward(self, x, edge_index, batch=None):
            """
            Args:
                x: Node features [num_nodes, input_dim]
                edge_index: Edge indices [2, num_edges]
                batch: Batch assignment [num_nodes] (for batched graphs)
            
            Returns:
                confidence: Predicted confidence [num_nodes, 1]
                time_to_failure: Predicted time to failure [num_nodes, 1]
            """
            # Input projection
            x = self.input_proj(x)
            x = F.relu(x)
            
            # GNN layers with residual connections
            for i, conv in enumerate(self.convs):
                x_in = x
                x = conv(x, edge_index)
                
                if i < self.num_layers - 1:
                    x = self.bns[i](x)
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                    
                    # Residual connection
                    if x_in.shape == x.shape:
                        x = x + x_in
            
            # Output predictions
            confidence = torch.sigmoid(self.fc_confidence(x))
            time_to_failure = F.relu(self.fc_time_to_failure(x))
            
            return confidence, time_to_failure
else:
    # Dummy class when PyTorch Geometric is not available
    class NetworkGNN:
        """Dummy NetworkGNN class when PyTorch Geometric is not installed"""
        def __init__(self, *args, **kwargs):
            pass


class GNNPredictionEngine:
    """
    GNNベースの予測エンジン
    """
    
    def __init__(
        self,
        topology: Dict[str, Any],
        children_map: Dict[str, List[str]],
        model_path: Optional[str] = None
    ):
        """
        Args:
            topology: ネットワークトポロジー
            children_map: 親子関係マップ
            model_path: 学習済みモデルのパス（オプション）
        """
        self.topology = topology
        self.children_map = children_map
        self.model = None
        
        if not HAS_PYTORCH_GEOMETRIC:
            logger.warning("PyTorch Geometric not available. GNN features disabled.")
            return
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = NetworkGNN().to(self.device)
        
        # Load pretrained model if available
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
    ) -> Optional[Data]:
        """
        トポロジーをPyTorch Geometric Dataに変換
        
        Args:
            alarm_embeddings: デバイスIDごとのアラーム埋め込み
            device_states: デバイスの状態情報（オプション）
        
        Returns:
            PyTorch Geometric Data object
        """
        if not HAS_PYTORCH_GEOMETRIC:
            return None
        
        # Create node mapping
        device_ids = list(self.topology.keys())
        device_to_idx = {dev_id: idx for idx, dev_id in enumerate(device_ids)}
        
        # Create node features
        node_features = []
        for dev_id in device_ids:
            if dev_id in alarm_embeddings:
                feature = alarm_embeddings[dev_id]
            else:
                # No alarm: use zero embedding
                feature = np.zeros(768)  # BERT embedding dimension
            
            node_features.append(feature)
        
        x = torch.tensor(np.array(node_features), dtype=torch.float)
        
        # Create edges from parent-child relationships
        edge_list = []
        for parent_id, children in self.children_map.items():
            if parent_id not in device_to_idx:
                continue
            parent_idx = device_to_idx[parent_id]
            
            for child_id in children:
                if child_id not in device_to_idx:
                    continue
                child_idx = device_to_idx[child_id]
                
                # Bidirectional edges
                edge_list.append([parent_idx, child_idx])
                edge_list.append([child_idx, parent_idx])
        
        # Add redundancy group edges
        redundancy_groups = {}
        for dev_id, attrs in self.topology.items():
            if isinstance(attrs, dict):
                rg = attrs.get('redundancy_group')
                if rg:
                    if rg not in redundancy_groups:
                        redundancy_groups[rg] = []
                    if dev_id in device_to_idx:
                        redundancy_groups[rg].append(device_to_idx[dev_id])
        
        # Connect devices in same redundancy group
        for rg, members in redundancy_groups.items():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    edge_list.append([members[i], members[j]])
                    edge_list.append([members[j], members[i]])
        
        if not edge_list:
            # No edges: create self-loops
            edge_list = [[i, i] for i in range(len(device_ids))]
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Create PyTorch Geometric Data
        data = Data(x=x, edge_index=edge_index)
        
        # Add device IDs as metadata
        data.device_ids = device_ids
        data.device_to_idx = device_to_idx
        
        return data
    
    def predict_with_gnn(
        self,
        alarm_embeddings: Dict[str, np.ndarray],
        target_device_id: str
    ) -> Tuple[float, float]:
        """
        GNNを使った予測
        
        Args:
            alarm_embeddings: デバイスIDごとのアラーム埋め込み
            target_device_id: 予測対象のデバイスID
        
        Returns:
            (confidence, time_to_failure_hours)
        """
        if not HAS_PYTORCH_GEOMETRIC or self.model is None:
            return 0.5, 336.0  # Default values
        
        # Convert topology to graph
        data = self.topology_to_graph(alarm_embeddings)
        if data is None:
            return 0.5, 336.0
        
        # Get target device index
        if target_device_id not in data.device_to_idx:
            return 0.5, 336.0
        
        target_idx = data.device_to_idx[target_device_id]
        
        # Move to device
        data = data.to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            confidence, time_to_failure = self.model(data.x, data.edge_index)
        
        # Extract predictions for target device
        target_confidence = float(confidence[target_idx].cpu().numpy())
        target_ttf = float(time_to_failure[target_idx].cpu().numpy())
        
        return target_confidence, target_ttf
    
    def train_on_historical_data(
        self,
        training_data: List[Dict],
        epochs: int = 100,
        lr: float = 0.001
    ):
        """
        履歴データでGNNを学習
        
        Args:
            training_data: 学習データ
                [{
                    'alarm_embeddings': {...},
                    'device_id': str,
                    'actual_failure': bool,
                    'time_to_failure': float
                }]
            epochs: エポック数
            lr: 学習率
        """
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
                
                # Create graph
                data = self.topology_to_graph(sample['alarm_embeddings'])
                if data is None:
                    continue
                
                data = data.to(self.device)
                
                # Forward pass
                pred_conf, pred_ttf = self.model(data.x, data.edge_index)
                
                # Get target
                target_idx = data.device_to_idx.get(sample['device_id'])
                if target_idx is None:
                    continue
                
                # Calculate loss
                target_conf = torch.tensor([1.0 if sample['actual_failure'] else 0.0],
                                          dtype=torch.float, device=self.device)
                target_ttf_val = torch.tensor([sample['time_to_failure']],
                                            dtype=torch.float, device=self.device)
                
                loss_conf = criterion_confidence(pred_conf[target_idx], target_conf)
                loss_ttf = criterion_ttf(pred_ttf[target_idx], target_ttf_val)
                
                loss = loss_conf + 0.1 * loss_ttf  # Weight TTF loss less
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(training_data)
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


# Utility function for integration
def create_gnn_engine(topology: dict, children_map: dict) -> Optional[GNNPredictionEngine]:
    """
    GNN予測エンジンを作成
    
    PyTorch Geometricが利用できない場合はNoneを返す
    """
    if not HAS_PYTORCH_GEOMETRIC:
        return None
    
    try:
        return GNNPredictionEngine(topology, children_map)
    except Exception as e:
        logger.error(f"Failed to create GNN engine: {e}")
        return None
