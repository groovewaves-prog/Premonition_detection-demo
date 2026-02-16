# streamlit_cache.py
import logging

logger = logging.getLogger(__name__)

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

try:
    # V45 Import
    from digital_twin_pkg import DigitalTwinEngine
    HAS_DT = True
except ImportError:
    HAS_DT = False


if HAS_STREAMLIT and HAS_DT:
    @st.cache_resource
    def _load_digital_twin_singleton(_topology_hash: str, topology: dict, children_map: dict):
        """
        Streamlit process-level cache for the engine.
        """
        logger.info("Initializing Digital Twin Engine (V45 - cached)...")
        # Initialize with tenant="default"
        return DigitalTwinEngine(topology, children_map, tenant_id="default")

    def get_digital_twin_engine(topology: dict, children_map: dict):
        """
        Entry point from app.py.
        """
        import hashlib, json
        # Simple hash of topology keys to detect site changes
        topo_hash = hashlib.md5(
            json.dumps(sorted(topology.keys())).encode()
        ).hexdigest()
        return _load_digital_twin_singleton(topo_hash, topology, children_map)

else:
    # Mock for testing without Streamlit
    def get_digital_twin_engine(topology: dict, children_map: dict):
        if HAS_DT:
            return DigitalTwinEngine(topology, children_map, tenant_id="default")
        return None
