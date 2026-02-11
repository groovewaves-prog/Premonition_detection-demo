# -*- coding: utf-8 -*-
"""
streamlit_cache.py
==================
Streamlit 環境での Digital Twin Engine キャッシュラッパー。

DigitalTwinEngine 自身が Singleton パターンでクラスレベルのモデルキャッシュを
持っているが、Streamlit の @st.cache_resource で二重にキャッシュすることで
プロセス再起動時以外はモデルロードが発生しないことを保証する。

使い方 (app.py 内):
    from streamlit_cache import get_digital_twin_engine
    engine = get_digital_twin_engine(topology, children_map)
"""

import logging

logger = logging.getLogger(__name__)

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

try:
    from digital_twin import DigitalTwinEngine
    HAS_DT = True
except ImportError:
    HAS_DT = False


if HAS_STREAMLIT and HAS_DT:
    @st.cache_resource
    def _load_digital_twin_singleton(_topology_hash: str, topology: dict, children_map: dict):
        """
        Streamlit のプロセスキャッシュにエンジンを常駐させる。
        _topology_hash は topology の変更検知用（Streamlit の hash 不可オブジェクト対策）。
        """
        logger.info("Initializing Digital Twin Engine (cached)...")
        return DigitalTwinEngine(topology, children_map)

    def get_digital_twin_engine(topology: dict, children_map: dict):
        """
        app.py から呼ぶエントリポイント。
        topology が同じなら同じインスタンスを返す。
        """
        import hashlib, json
        topo_hash = hashlib.md5(
            json.dumps(sorted(topology.keys())).encode()
        ).hexdigest()
        return _load_digital_twin_singleton(topo_hash, topology, children_map)

else:
    def get_digital_twin_engine(topology: dict, children_map: dict):
        """Streamlit 外 or digital_twin 未導入の場合のフォールバック"""
        if HAS_DT:
            return DigitalTwinEngine(topology, children_map)
        return None
