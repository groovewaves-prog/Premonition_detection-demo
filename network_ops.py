# -*- coding: utf-8 -*-
"""
AIOps Agent - Network Operations Module (Improved v2)
=====================================================
改善点:
1. グローバルレートリミッター統合
2. キャッシュ活用強化
3. プロンプト最適化（トークン削減）
4. 429/503エラーの統一的なハンドリング
5. ストリーミング応答の安定化
"""

import re
import os
import time
import json
import hashlib
import logging
import concurrent.futures
from typing import Dict, List, Optional, Generator, Any
from enum import Enum

import google.generativeai as genai
from netmiko import ConnectHandler

from rate_limiter import (
    GlobalRateLimiter,
    rate_limited_with_retry,
    RateLimitConfig,
    estimate_tokens
)

logger = logging.getLogger(__name__)

# =====================================================
# 定数・設定
# =====================================================
MODEL_NAME = "gemma-3-12b-it"

SANDBOX_DEVICE = {
    'device_type': 'cisco_nxos',
    'host': 'sandbox-nxos-1.cisco.com',
    'username': 'admin',
    'password': 'Admin_1234!',
    'port': 22,
    'global_delay_factor': 2,
    'banner_timeout': 30,
    'conn_timeout': 30,
}


class RemediationEnvironment(Enum):
    DEMO = "demo"
    TEST = "test"
    PRODUCTION = "prod"


class RemediationResult:
    def __init__(self, step_name: str, status: str, data=None, error=None):
        self.step_name = step_name
        self.status = status
        self.data = data
        self.error = error
        self.timestamp = time.time()
    
    def __str__(self):
        if self.status == "success":
            return f"✅ {self.step_name}: {self.data}"
        elif self.status == "timeout":
            return f"⏱️ {self.step_name}: Timeout"
        return f"❌ {self.step_name}: {self.error}"
    
    def to_dict(self):
        return {
            "step": self.step_name,
            "status": self.status,
            "data": self.data,
            "error": self.error,
            "timestamp": self.timestamp
        }


# =====================================================
# グローバル初期化
# =====================================================
_rate_limiter: Optional[GlobalRateLimiter] = None
_model: Optional[genai.GenerativeModel] = None
_api_configured = False


def _get_rate_limiter() -> GlobalRateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = GlobalRateLimiter()
    return _rate_limiter


def _ensure_api_configured(api_key: str) -> Optional[genai.GenerativeModel]:
    global _model, _api_configured
    if _api_configured and _model:
        return _model
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        _model = genai.GenerativeModel(MODEL_NAME, generation_config={"temperature": 0.0})
        _api_configured = True
        logger.info(f"API configured with model: {MODEL_NAME}")
        return _model
    except Exception as e:
        logger.error(f"API Configuration Error: {e}")
        return None


# =====================================================
# ユーティリティ関数
# =====================================================
def sanitize_output(text: str) -> str:
    rules = [
        (r'(password|secret) \d+ \S+', r'\1 <HIDDEN>'),
        (r'(encrypted password) \S+', r'\1 <HIDDEN>'),
        (r'(snmp-server community) \S+', r'\1 <HIDDEN>'),
        (r'(username \S+ privilege \d+ secret \d+) \S+', r'\1 <HIDDEN>'),
        (r'\b(?!(?:10|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.)\d{1,3}\.(?:\d{1,3}\.){2}\d{1,3}\b', '<MASKED_IP>'),
        (r'([0-9A-Fa-f]{4}\.){2}[0-9A-Fa-f]{4}', '<MASKED_MAC>'),
    ]
    for pattern, replacement in rules:
        text = re.sub(pattern, replacement, text)
    return text


def compute_cache_hash(scenario: str, device_id: str, extra: str = "") -> str:
    content = f"{scenario}|{device_id}|{extra}"
    return hashlib.md5(content.encode()).hexdigest()


def filter_hallucination(text: str) -> str:
    """AI生成テキストから不要な免責事項等を除去"""
    patterns = [
        (r'【免責事項】.*?(?=##|$)', '', re.DOTALL),
        (r'免責事項.*?(?=##|$)', '', re.DOTALL),
        (r'【注記】.*?(?=##|$)', '', re.DOTALL),
        (r'【警告】.*?(?=##|$)', '', re.DOTALL),
        (r'※.*?\n', ''),
        (r'\n\n\n+', '\n\n'),
    ]
    result = text
    for pattern, replacement, *flags in patterns:
        flag = flags[0] if flags else 0
        result = re.sub(pattern, replacement, result, flags=flag)
    return result.strip()


def validate_response(response) -> bool:
    """APIレスポンスの有効性チェック"""
    try:
        if not response or not hasattr(response, 'candidates'):
            return False
        if not response.candidates:
            return False
        candidate = response.candidates[0]
        if not hasattr(candidate, 'content') or not candidate.content:
            return False
        if hasattr(candidate.content, 'parts') and len(candidate.content.parts) == 0:
            return False
        return True
    except Exception:
        return False


# =====================================================
# LLM呼び出し関数（レートリミッター統合）
# =====================================================
def _call_llm_with_rate_limit(
    model: genai.GenerativeModel,
    prompt: str,
    stream: bool = False,
    max_retries: int = 3
) -> Any:
    """
    レートリミッター付きLLM呼び出し
    """
    limiter = _get_rate_limiter()
    config = limiter.config
    
    for attempt in range(max_retries + 1):
        try:
            # レート制限待機
            if not limiter.wait_for_slot(timeout=120):
                raise RuntimeError("Rate limit timeout")
            
            limiter.record_request()
            
            if stream:
                return model.generate_content(prompt, stream=True)
            else:
                return model.generate_content(prompt)
        
        except Exception as e:
            error_msg = str(e).lower()
            
            # リトライ対象のエラー
            should_retry = any(x in error_msg for x in ['429', '503', 'overloaded', 'rate', 'quota'])
            
            if should_retry and attempt < max_retries:
                delay = min(config.retry_base_delay * (2 ** attempt), config.retry_max_delay)
                logger.warning(f"Retry {attempt + 1}/{max_retries}: waiting {delay:.1f}s after: {e}")
                time.sleep(delay)
            else:
                raise
    
    return None


# =====================================================
# 障害ログ生成（キャッシュ付き）
# =====================================================
def generate_fake_log_by_ai(scenario_name: str, target_node, api_key: str) -> str:
    """シナリオに基づく障害ログ生成"""
    if not api_key:
        return "Error: API Key Missing"
    
    model = _ensure_api_configured(api_key)
    if not model:
        return "Error: API Configuration Failed"
    
    limiter = _get_rate_limiter()
    
    # キャッシュチェック
    cache_key = compute_cache_hash(scenario_name, target_node.id, "fake_log")
    cached = limiter.get_cache(cache_key)
    if cached:
        logger.info(f"Cache hit for fake log: {scenario_name}")
        return cached
    
    vendor = target_node.metadata.get("vendor", "Generic")
    os_type = target_node.metadata.get("os", "Generic OS")
    hostname = target_node.id
    
    # ★最適化されたプロンプト（トークン削減）
    prompt = f"""CLIシミュレータ。障害ログを生成。

ホスト: {hostname}
ベンダー: {vendor}
OS: {os_type}
シナリオ: {scenario_name}

要件:
- 確認コマンド2-3個とその出力
- シナリオに応じた異常状態を表示
- 解説不要、CLIテキストのみ
"""

    try:
        response = _call_llm_with_rate_limit(model, prompt, stream=False)
        result = response.text if response else "Error: No response"
        limiter.set_cache(cache_key, result)
        return result
    except Exception as e:
        logger.error(f"generate_fake_log_by_ai error: {e}")
        return f"AI Generation Error: {e}"


# =====================================================
# 初期症状予測（キャッシュ付き）
# =====================================================
def predict_initial_symptoms(scenario_name: str, api_key: str) -> Dict:
    """障害シナリオから初期症状を予測"""
    if not api_key:
        return {}
    
    model = _ensure_api_configured(api_key)
    if not model:
        return {}
    
    limiter = _get_rate_limiter()
    
    # キャッシュチェック
    cache_key = compute_cache_hash(scenario_name, "", "symptoms")
    cached = limiter.get_cache(cache_key)
    if cached:
        return cached
    
    # ★最適化されたプロンプト
    prompt = f"""シナリオ「{scenario_name}」の初期症状をJSON出力。

キー: alarm, ping, log
値は以下から選択（該当なしは空文字）:
- alarm: BGP Flapping, Fan Fail, Heartbeat Loss, Connection Lost, Power Supply 1 Failed, Power Supply: Dual Loss
- log: Interface Down, Power Fail, Config Error, High Temperature
- ping: NG, OK

例: {{"alarm": "BGP Flapping", "ping": "OK", "log": ""}}
"""

    try:
        response = _call_llm_with_rate_limit(model, prompt, stream=False)
        text = response.text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        limiter.set_cache(cache_key, result)
        return result
    except Exception as e:
        logger.error(f"predict_initial_symptoms error: {e}")
        return {}


# =====================================================
# 原因分析レポート生成（ストリーミング）
# =====================================================
def generate_analyst_report(
    scenario: str, 
    target_node, 
    topology_context: str,
    target_conf: str, 
    verification_context: str, 
    api_key: str
) -> str:
    """原因分析レポート生成（非ストリーミング）"""
    if not api_key:
        return "Error: API Key Missing"
    
    model = _ensure_api_configured(api_key)
    if not model:
        return "Error: API Configuration Failed"
    
    limiter = _get_rate_limiter()
    
    # キャッシュチェック
    cache_key = compute_cache_hash(scenario, target_node.id, "analyst_report")
    cached = limiter.get_cache(cache_key)
    if cached:
        return cached
    
    vendor = target_node.metadata.get("vendor", "Unknown")
    os_type = target_node.metadata.get("os", "Unknown OS")
    
    prompt = f"""ネットワーク障害の原因分析

シナリオ: {scenario}
デバイス: {target_node.id} ({vendor} {os_type})

以下の構成で記載:
## 障害概要
## 発生原因
## 影響範囲
## 技術的根拠
## 切り分け判断

復旧手順は含めない。
"""

    try:
        response = _call_llm_with_rate_limit(model, prompt, stream=False)
        result = filter_hallucination(response.text) if response else "Error: No response"
        limiter.set_cache(cache_key, result)
        return result
    except Exception as e:
        logger.error(f"generate_analyst_report error: {e}")
        return f"Error: {e}"


def generate_analyst_report_streaming(
    scenario: str,
    target_node,
    topology_context: str,
    target_conf: str,
    verification_context: str,
    api_key: str,
    max_retries: int = 2,
    backoff: float = 5.0
) -> Generator[str, None, None]:
    """原因分析レポート生成（ストリーミング版）"""
    if not api_key:
        yield "Error: API Key Missing"
        return
    
    model = _ensure_api_configured(api_key)
    if not model:
        yield "Error: API Configuration Failed"
        return
    
    limiter = _get_rate_limiter()
    
    # キャッシュチェック
    cache_key = compute_cache_hash(scenario, target_node.id, "analyst_streaming")
    cached = limiter.get_cache(cache_key)
    if cached:
        yield cached
        return
    
    vendor = target_node.metadata.get("vendor", "Unknown")
    os_type = target_node.metadata.get("os", "Unknown OS")
    
    prompt = f"""ネットワーク障害の原因分析

シナリオ: {scenario}
デバイス: {target_node.id} ({vendor} {os_type})

以下の構成で記載:
## 障害概要
## 発生原因
## 影響範囲
## 技術的根拠
## 切り分け判断
"""

    for attempt in range(max_retries + 1):
        try:
            response = _call_llm_with_rate_limit(model, prompt, stream=True)
            
            if not validate_response(response):
                if attempt < max_retries:
                    wait_time = backoff * (attempt + 1)
                    yield f"\n\n⏳ **再試行中... {wait_time:.0f}秒後**\n\n"
                    time.sleep(wait_time)
                    continue
                yield "❌ 有効な応答が得られませんでした。"
                return
            
            full_text = ""
            for chunk in response:
                if chunk.text:
                    full_text += chunk.text
                    yield chunk.text
            
            # キャッシュ保存
            filtered = filter_hallucination(full_text)
            limiter.set_cache(cache_key, filtered)
            return
        
        except Exception as e:
            error_msg = str(e).lower()
            if any(x in error_msg for x in ['503', '429', 'overloaded']):
                if attempt < max_retries:
                    wait_time = backoff * (attempt + 1)
                    yield f"\n\n⏳ **API混雑中... {wait_time:.0f}秒後に再試行**\n\n"
                    time.sleep(wait_time)
                    continue
            yield f"❌ エラー: {e}"
            return


# =====================================================
# 復旧コマンド生成
# =====================================================
def generate_remediation_commands(
    scenario: str,
    analysis_result: str,
    target_node,
    api_key: str
) -> str:
    """復旧手順生成"""
    if not api_key:
        return "Error: API Key Missing"
    
    model = _ensure_api_configured(api_key)
    if not model:
        return "Error: API Configuration Failed"
    
    limiter = _get_rate_limiter()
    
    # キャッシュチェック
    cache_key = compute_cache_hash(scenario, target_node.id, "remediation")
    cached = limiter.get_cache(cache_key)
    if cached:
        return cached
    
    vendor = target_node.metadata.get("vendor", "Unknown")
    os_type = target_node.metadata.get("os", "Unknown OS")
    
    prompt = f"""復旧手順を作成

デバイス: {target_node.id} ({vendor} {os_type})
シナリオ: {scenario}

以下の構成で記載:
### 1. 物理・前提アクション
### 2. 復旧コマンド (コードブロックで)
### 3. 正常性確認コマンド (3つ以上)
"""

    try:
        response = _call_llm_with_rate_limit(model, prompt, stream=False)
        result = filter_hallucination(response.text) if response else "Error: No response"
        limiter.set_cache(cache_key, result)
        return result
    except Exception as e:
        logger.error(f"generate_remediation_commands error: {e}")
        return f"Error: {e}"


def generate_remediation_commands_streaming(
    scenario: str,
    analysis_result: str,
    target_node,
    api_key: str,
    max_retries: int = 2,
    backoff: float = 5.0
) -> Generator[str, None, None]:
    """復旧手順生成（ストリーミング版）"""
    if not api_key:
        yield "Error: API Key Missing"
        return
    
    model = _ensure_api_configured(api_key)
    if not model:
        yield "Error: API Configuration Failed"
        return
    
    limiter = _get_rate_limiter()
    
    cache_key = compute_cache_hash(scenario, target_node.id, "remediation_streaming")
    cached = limiter.get_cache(cache_key)
    if cached:
        yield cached
        return
    
    prompt = f"""復旧手順を作成

デバイス: {target_node.id}
シナリオ: {scenario}

以下の構成:
## 実施前提
## バックアップ手順
## 復旧手順
## ロールバック手順
## 正常性確認
"""

    for attempt in range(max_retries + 1):
        try:
            response = _call_llm_with_rate_limit(model, prompt, stream=True)
            
            if not validate_response(response):
                if attempt < max_retries:
                    wait_time = backoff * (attempt + 1)
                    yield f"\n\n⏳ **再試行中... {wait_time:.0f}秒後**\n\n"
                    time.sleep(wait_time)
                    continue
                yield "❌ 有効な応答が得られませんでした。"
                return
            
            full_text = ""
            for chunk in response:
                if chunk.text:
                    full_text += chunk.text
                    yield chunk.text
            
            filtered = filter_hallucination(full_text)
            limiter.set_cache(cache_key, filtered)
            return
        
        except Exception as e:
            error_msg = str(e).lower()
            if any(x in error_msg for x in ['503', '429', 'overloaded']):
                if attempt < max_retries:
                    wait_time = backoff * (attempt + 1)
                    yield f"\n\n⏳ **API混雑中... {wait_time:.0f}秒後に再試行**\n\n"
                    time.sleep(wait_time)
                    continue
            yield f"❌ エラー: {e}"
            return


# =====================================================
# 診断シミュレーション
# =====================================================
def run_diagnostic_simulation(
    scenario_type: str,
    target_node=None,
    api_key: str = None
) -> Dict:
    time.sleep(1.5)
    
    if "---" in scenario_type or "正常" in scenario_type:
        return {"status": "SKIPPED", "sanitized_log": "No action required.", "error": None}
    
    if "[Live]" in scenario_type:
        commands = ["terminal length 0", "show version", "show interface brief", "show ip route"]
        try:
            with ConnectHandler(**SANDBOX_DEVICE) as ssh:
                if not ssh.check_enable_mode():
                    ssh.enable()
                prompt = ssh.find_prompt()
                raw_output = f"Connected to: {prompt}\n"
                for cmd in commands:
                    output = ssh.send_command(cmd)
                    raw_output += f"\n{'='*30}\n[Command] {cmd}\n{output}\n"
        except Exception as e:
            return {"status": "ERROR", "sanitized_log": "", "error": str(e)}
        return {"status": "SUCCESS", "sanitized_log": sanitize_output(raw_output), "error": None}
    
    elif "全回線断" in scenario_type or "サイレント" in scenario_type or "両系" in scenario_type:
        return {"status": "ERROR", "sanitized_log": "", "error": "Connection timed out"}
    
    else:
        if api_key and target_node:
            raw_output = generate_fake_log_by_ai(scenario_type, target_node, api_key)
            return {"status": "SUCCESS", "sanitized_log": sanitize_output(raw_output), "error": None}
        return {"status": "ERROR", "sanitized_log": "", "error": "API Key or Target Node Missing"}


# =====================================================
# 並列修復処理
# =====================================================
def run_remediation_parallel_v2(
    device_id: str,
    device_info: dict,
    scenario: str,
    environment: RemediationEnvironment = RemediationEnvironment.DEMO,
    timeout_per_step: int = 30
) -> Dict[str, RemediationResult]:
    """複数の修復ステップを並列実行"""
    
    def backup_step():
        time.sleep(2)
        return RemediationResult(
            step_name="Backup",
            status="success",
            data=f"Backup created for {device_id}"
        )
    
    def apply_step():
        time.sleep(3)
        return RemediationResult(
            step_name="Apply",
            status="success",
            data="Applied remediation commands"
        )
    
    def verify_step():
        time.sleep(2)
        return RemediationResult(
            step_name="Verify",
            status="success",
            data={"overall": "HEALTHY"}
        )
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_step = {
            executor.submit(backup_step): "Backup",
            executor.submit(apply_step): "Apply",
            executor.submit(verify_step): "Verify",
        }
        
        results = {}
        for future in concurrent.futures.as_completed(future_to_step):
            step_name = future_to_step[future]
            try:
                result = future.result(timeout=timeout_per_step)
                results[step_name] = result
            except concurrent.futures.TimeoutError:
                results[step_name] = RemediationResult(
                    step_name=step_name,
                    status="timeout",
                    error=f"Timeout after {timeout_per_step} seconds"
                )
            except Exception as e:
                results[step_name] = RemediationResult(
                    step_name=step_name,
                    status="failed",
                    error=str(e)
                )
    
    return results


# =====================================================
# その他のユーティリティ関数
# =====================================================
def generate_config_from_intent(target_node, current_config: str, intent_text: str, api_key: str) -> str:
    """インテントから設定生成"""
    if not api_key:
        return "Error: API Key Missing"
    
    model = _ensure_api_configured(api_key)
    if not model:
        return "Error: API Configuration Failed"
    
    vendor = target_node.metadata.get("vendor", "Unknown")
    os_type = target_node.metadata.get("os", "Unknown OS")
    
    prompt = f"""設定生成
対象: {target_node.id} ({vendor} {os_type})
Intent: {intent_text}
出力: 投入用コマンドのみ (コードブロック)
"""

    try:
        response = _call_llm_with_rate_limit(model, prompt, stream=False)
        return response.text if response else "Error: No response"
    except Exception as e:
        return f"Config Gen Error: {e}"


def generate_health_check_commands(target_node, api_key: str) -> str:
    """正常性確認コマンド生成"""
    if not api_key:
        return "Error: API Key Missing"
    
    model = _ensure_api_configured(api_key)
    if not model:
        return "Error: API Configuration Failed"
    
    vendor = target_node.metadata.get("vendor", "Unknown")
    os_type = target_node.metadata.get("os", "Unknown OS")
    
    prompt = f"正常性確認コマンドを3つ生成。対象: {vendor} {os_type}。コマンドのみ箇条書き。"
    
    try:
        response = _call_llm_with_rate_limit(model, prompt, stream=False)
        return response.text if response else "Error: No response"
    except Exception as e:
        return f"Command Gen Error: {e}"
