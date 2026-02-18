# utils/llm_helper.py  ―  LLM helper utilities (google-genai新SDK対応)

import streamlit as st
import time

# GenAI availability check (新SDK: google-genai)
try:
    from google import genai as _genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# 旧SDK(google-generativeai)との互換フォールバック
if not GENAI_AVAILABLE:
    try:
        import google.generativeai as _genai_legacy
        GENAI_AVAILABLE = True
        _genai = None  # 旧SDKはネームスペースが異なるため個別管理
    except ImportError:
        _genai = None

# google.api_core例外（新SDKでは不要だが互換性のため保持）
try:
    from google.api_core import exceptions as google_exceptions
except ImportError:
    class _DummyExceptions:
        ServiceUnavailable = Exception
        ResourceExhausted = Exception
    google_exceptions = _DummyExceptions()


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


def get_genai_client(api_key: str):
    """
    新SDK(google-genai)のClientを生成して返す。
    呼び出し元で `client.models.generate_content(...)` 等を利用する。
    """
    if not GENAI_AVAILABLE or _genai is None:
        raise RuntimeError("google-genai package is not installed.")
    return _genai.Client(api_key=api_key)


def generate_content_with_retry(client, model_name: str, prompt: str, stream: bool = True, retries: int = 3):
    """
    リトライ付きコンテンツ生成（新SDK対応版）

    Args:
        client: genai.Client インスタンス
        model_name: 使用するモデル名（例: "gemma-3-12b-it"）
        prompt: プロンプト文字列
        stream: True でストリーミング応答
        retries: リトライ回数

    Returns:
        新SDK: stream=True → イテレータ, stream=False → GenerateContentResponse
    """
    limiter = get_rate_limiter()
    for i in range(retries):
        try:
            if not limiter.wait_for_slot(timeout=60):
                raise RuntimeError("Rate limit timeout")
            limiter.record_request()
            if stream:
                return client.models.generate_content_stream(
                    model=model_name,
                    contents=prompt
                )
            else:
                return client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
        except Exception as e:
            err_str = str(e).lower()
            if "resource exhausted" in err_str or "quota" in err_str:
                if i == retries - 1:
                    raise
                time.sleep(5 * (i + 1))
            elif i == retries - 1:
                raise
            else:
                time.sleep(2 * (i + 1))
    return None
