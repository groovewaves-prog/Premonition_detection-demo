# -*- coding: utf-8 -*-
"""
AIOps Agent - Multi-Site Registry Module
=========================================
複数拠点（サイト）のトポロジーと設定を管理するモジュール

設計方針:
- 拡張性: 新しい拠点の追加が容易
- 柔軟性: 拠点ごとに異なるトポロジー構造をサポート
- 互換性: 既存のdata.py/logic.pyとの互換性を維持
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# =====================================================
# 設定
# =====================================================
BASE_DIR = Path(__file__).parent
TOPOLOGIES_DIR = BASE_DIR / "topologies"
CONFIGS_DIR = BASE_DIR / "configs"

# =====================================================
# データクラス
# =====================================================
@dataclass
class SitePaths:
    """拠点関連のファイルパスを保持"""
    site_id: str
    network_id: str
    topology_path: Path
    config_dir: Path
    
    def __post_init__(self):
        self.topology_path = Path(self.topology_path)
        self.config_dir = Path(self.config_dir)


@dataclass
class NetworkNode:
    """ネットワークノードを表現するデータクラス（data.pyと互換）"""
    id: str
    layer: int
    type: str
    parent_id: Optional[str] = None
    redundancy_group: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not isinstance(self.layer, int):
            try:
                self.layer = int(self.layer)
            except (ValueError, TypeError):
                self.layer = 99
        if not isinstance(self.metadata, dict):
            self.metadata = {}

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)


@dataclass
class SiteConfig:
    """拠点の設定情報"""
    site_id: str
    display_name: str
    topology_file: str
    networks: List[str] = field(default_factory=lambda: ["default"])
    metadata: Dict[str, Any] = field(default_factory=dict)


# =====================================================
# 拠点レジストリ
# =====================================================
class SiteRegistry:
    """
    拠点管理のシングルトンクラス
    
    使用例:
        registry = SiteRegistry()
        sites = registry.list_sites()
        topology = registry.load_topology("A")
    """
    _instance: Optional['SiteRegistry'] = None
    
    # デフォルトの拠点設定
    DEFAULT_SITES: Dict[str, SiteConfig] = {
        "A": SiteConfig(
            site_id="A",
            display_name="A拠点",
            topology_file="topology_a.json",
            networks=["default"],
            metadata={"region": "Tokyo", "tier": 1}
        ),
        "B": SiteConfig(
            site_id="B",
            display_name="B拠点",
            topology_file="topology_b.json",
            networks=["default"],
            metadata={"region": "Osaka", "tier": 2}
        ),
    }
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._sites: Dict[str, SiteConfig] = {}
        self._topology_cache: Dict[str, Dict[str, NetworkNode]] = {}
        self._load_sites()
        self._initialized = True
    
    def _load_sites(self):
        """拠点設定を読み込み"""
        # まずデフォルト設定を適用
        self._sites = self.DEFAULT_SITES.copy()
        
        # 設定ファイルがあれば上書き（将来の拡張用）
        config_path = BASE_DIR / "sites_config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for site_id, site_data in data.items():
                    self._sites[site_id] = SiteConfig(
                        site_id=site_id,
                        display_name=site_data.get("display_name", f"{site_id}拠点"),
                        topology_file=site_data.get("topology_file", f"topology_{site_id.lower()}.json"),
                        networks=site_data.get("networks", ["default"]),
                        metadata=site_data.get("metadata", {})
                    )
                logger.info(f"Loaded {len(self._sites)} sites from config")
            except Exception as e:
                logger.warning(f"Failed to load sites config: {e}, using defaults")
    
    def list_sites(self) -> List[str]:
        """登録されている拠点IDのリストを返す"""
        return list(self._sites.keys())
    
    def list_networks(self, site_id: str) -> List[str]:
        """指定拠点のネットワークリストを返す"""
        site = self._sites.get(site_id)
        if site:
            return site.networks
        return ["default"]
    
    def get_site_config(self, site_id: str) -> Optional[SiteConfig]:
        """拠点設定を取得"""
        return self._sites.get(site_id)
    
    def get_display_name(self, site_id: str) -> str:
        """拠点の表示名を取得"""
        site = self._sites.get(site_id)
        if site:
            return site.display_name
        return f"{site_id}拠点"
    
    def get_paths(self, site_id: str, network_id: str = "default") -> SitePaths:
        """拠点のファイルパスを取得"""
        site = self._sites.get(site_id)
        if site:
            topology_path = TOPOLOGIES_DIR / site.topology_file
        else:
            topology_path = TOPOLOGIES_DIR / f"topology_{site_id.lower()}.json"
        
        return SitePaths(
            site_id=site_id,
            network_id=network_id,
            topology_path=topology_path,
            config_dir=CONFIGS_DIR
        )
    
    def load_topology(self, site_id: str, network_id: str = "default", 
                      force_reload: bool = False) -> Dict[str, NetworkNode]:
        """
        トポロジーを読み込み
        
        Args:
            site_id: 拠点ID
            network_id: ネットワークID（将来の拡張用）
            force_reload: キャッシュを無視して再読み込み
        
        Returns:
            デバイスID -> NetworkNode のマッピング
        """
        cache_key = f"{site_id}/{network_id}"
        
        if not force_reload and cache_key in self._topology_cache:
            return self._topology_cache[cache_key]
        
        paths = self.get_paths(site_id, network_id)
        topology = {}
        
        if paths.topology_path.exists():
            try:
                with open(paths.topology_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                for node_id, node_data in raw_data.items():
                    topology[node_id] = NetworkNode(
                        id=node_id,
                        layer=node_data.get("layer", 99),
                        type=node_data.get("type", "UNKNOWN"),
                        parent_id=node_data.get("parent_id"),
                        redundancy_group=node_data.get("redundancy_group"),
                        metadata=node_data.get("metadata", {})
                    )
                
                logger.info(f"Loaded topology for {site_id}: {len(topology)} nodes")
            except Exception as e:
                logger.error(f"Failed to load topology for {site_id}: {e}")
        else:
            logger.warning(f"Topology file not found: {paths.topology_path}")
        
        self._topology_cache[cache_key] = topology
        return topology
    
    def get_topology_mtime(self, site_id: str, network_id: str = "default") -> float:
        """トポロジーファイルの更新時刻を取得"""
        paths = self.get_paths(site_id, network_id)
        if paths.topology_path.exists():
            return paths.topology_path.stat().st_mtime
        return 0.0
    
    def clear_cache(self, site_id: Optional[str] = None):
        """キャッシュをクリア"""
        if site_id:
            keys_to_remove = [k for k in self._topology_cache if k.startswith(f"{site_id}/")]
            for k in keys_to_remove:
                del self._topology_cache[k]
        else:
            self._topology_cache.clear()


# =====================================================
# モジュールレベルの便利関数（後方互換性のため）
# =====================================================
_registry: Optional[SiteRegistry] = None

def _get_registry() -> SiteRegistry:
    global _registry
    if _registry is None:
        _registry = SiteRegistry()
    return _registry

def list_tenants() -> List[str]:
    """登録されている拠点IDのリストを返す（後方互換）"""
    return _get_registry().list_sites()

def list_sites() -> List[str]:
    """登録されている拠点IDのリストを返す"""
    return _get_registry().list_sites()

def list_networks(site_id: str) -> List[str]:
    """指定拠点のネットワークリストを返す"""
    return _get_registry().list_networks(site_id)

def get_paths(site_id: str, network_id: str = "default") -> SitePaths:
    """拠点のファイルパスを取得"""
    return _get_registry().get_paths(site_id, network_id)

def load_topology(topology_path) -> Dict[str, NetworkNode]:
    """
    トポロジーを読み込み（パスまたは拠点IDを受け付け）
    
    後方互換性のため、文字列パスを直接受け取ることも可能
    """
    if isinstance(topology_path, (str, Path)):
        path = Path(topology_path)
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                topology = {}
                for node_id, node_data in raw_data.items():
                    topology[node_id] = NetworkNode(
                        id=node_id,
                        layer=node_data.get("layer", 99),
                        type=node_data.get("type", "UNKNOWN"),
                        parent_id=node_data.get("parent_id"),
                        redundancy_group=node_data.get("redundancy_group"),
                        metadata=node_data.get("metadata", {})
                    )
                return topology
            except Exception as e:
                logger.error(f"Failed to load topology from {path}: {e}")
                return {}
    return {}

def topology_mtime(site_id: str, network_id: str = "default") -> float:
    """トポロジーファイルの更新時刻を取得"""
    return _get_registry().get_topology_mtime(site_id, network_id)

def get_display_name(site_id: str) -> str:
    """拠点の表示名を取得"""
    return _get_registry().get_display_name(site_id)


# =====================================================
# 初期化時の自動検出
# =====================================================
def _auto_discover_sites():
    """topologiesディレクトリから拠点を自動検出"""
    if not TOPOLOGIES_DIR.exists():
        return
    
    for f in TOPOLOGIES_DIR.glob("topology_*.json"):
        site_id = f.stem.replace("topology_", "").upper()
        if site_id not in SiteRegistry.DEFAULT_SITES:
            SiteRegistry.DEFAULT_SITES[site_id] = SiteConfig(
                site_id=site_id,
                display_name=f"{site_id}拠点",
                topology_file=f.name,
                networks=["default"],
                metadata={}
            )
            logger.info(f"Auto-discovered site: {site_id}")

# モジュール読み込み時に自動検出
_auto_discover_sites()
