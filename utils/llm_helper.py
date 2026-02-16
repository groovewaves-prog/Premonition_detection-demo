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
