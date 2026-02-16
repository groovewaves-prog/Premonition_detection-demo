# utils/llm_helper.py
import time
import os
import streamlit as st
from google.api_core import exceptions as google_exceptions
from rate_limiter import GlobalRateLimiter, RateLimitConfig

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

@st.cache_resource
def get_rate_limiter():
    return GlobalRateLimiter(RateLimitConfig(rpm=30, rpd=14400, safety_margin=0.9))

def generate_content_with_retry(model, prompt, stream=True, retries=3):
    limiter = get_rate_limiter()
    for i in range(retries):
        try:
            if not limiter.wait_for_slot(timeout=60):
                raise RuntimeError("Rate limit timeout")
            limiter.record_request()
            return model.generate_content(prompt, stream=stream)
        except google_exceptions.ServiceUnavailable:
            if i == retries - 1: raise
            time.sleep(2 * (i + 1))
    return None

def build_ci_context_for_chat(topology: dict, target_node_id: str) -> dict:
    from .helpers import load_config_by_id
    
    node = topology.get(target_node_id) if target_node_id else None
    md = {}
    if node:
        md = node.metadata if hasattr(node, 'metadata') else node.get('metadata', {})
    
    def _pick(keys, default=""):
        for k in keys:
            v = md.get(k)
            if v: return str(v).strip()
        return default

    ci = {
        "device_id": target_node_id or "",
        "hostname": _pick(["hostname", "host", "name"], default=(target_node_id or "")),
        "vendor": _pick(["vendor", "manufacturer"], default=""),
        "os": _pick(["os", "platform"], default=""),
        "model": _pick(["model", "hw_model"], default=""),
        "role": _pick(["role", "type"], default=""),
        "site": _pick(["site", "location"], default=""),
    }
    
    conf = load_config_by_id(target_node_id)
    if conf: ci["config_excerpt"] = conf[:1500]
    return ci
