# utils/llm_helper.py  ―  LLM helper utilities

import streamlit as st

# GenAI availability check
try:
    import google.generativeai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class RateLimitConfig:
    def __init__(self, rpm=30, rpd=14400, safety_margin=0.9):
        self.rpm = rpm
        self.rpd = rpd
        self.safety_margin = safety_margin


class GlobalRateLimiter:
    def __init__(self, config):
        self.config = config
        self.requests_last_minute = 0
        self.requests_today = 0
        
    def wait_for_slot(self, timeout=60):
        return True
        
    def record_request(self):
        self.requests_last_minute += 1
        self.requests_today += 1
        
    def get_stats(self):
        return {
            'requests_last_minute': self.requests_last_minute,
            'requests_today': self.requests_today,
            'rpm_limit': self.config.rpm,
            'rpd_limit': self.config.rpd
        }


@st.cache_resource
def get_rate_limiter():
    """レートリミッターのシングルトン"""
    return GlobalRateLimiter(RateLimitConfig(rpm=30, rpd=14400, safety_margin=0.9))
