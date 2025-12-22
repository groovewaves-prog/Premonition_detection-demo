# -*- coding: utf-8 -*-
"""
AIOps Agent - Network Operations Module (v3 - 根本修正版)
=========================================================
★根本修正:
1. validate_response 完全削除（ストリーミングでは使用不可）
2. ストリーミングチャンク即時yield
3. 不要な待機処理の排除
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

from rate_limiter import GlobalRateLimiter, RateLimitConfig

logger = logging.getLogger(__name__)

# =====================================================
# 定数
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
# グローバル状態
# =====================================================
_rate_limiter: Optional[GlobalRateLimiter] = None
_model: Optional[genai.GenerativeModel] = None
_api_configured = False


def _get_limiter() -> GlobalRateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = GlobalRateLimiter()
    return _rate_limiter


def _get_model(api_key: str) -> Optional[genai.GenerativeModel]:
    global _model, _api_configured
    if _api_configured and _model:
        return _model
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        _model = genai.GenerativeModel(MODEL_NAME)
        _api_configured = True
        return _model
    except Exception as e:
        logger.error(f"API config error: {e}")
        return None


# =====================================================
# ユーティリティ
# =====================================================
def sanitize_output(text: str) -> str:
    """機密情報をマスク"""
    rules = [
        (r'(password|secret) \d+ \S+', r'\1 <HIDDEN>'),
        (r'(snmp-server community) \S+', r'\1 <HIDDEN>'),
    ]
    for pattern, replacement in rules:
        text = re.sub(pattern, replacement, text)
    return text


def compute_cache_hash(scenario: str, device_id: str, extra: str = "") -> str:
    """キャッシュキー生成"""
    return hashlib.md5(f"{scenario}|{device_id}|{extra}".encode()).hexdigest()


def _extract_text(chunk) -> str:
    """ストリーミングチャンクからテキスト抽出（安全版）"""
    # 方法1: 直接textプロパティ
    try:
        if hasattr(chunk, 'text') and chunk.text:
            return chunk.text
    except Exception:
        pass
    
    # 方法2: candidates経由
    try:
        if hasattr(chunk, 'candidates') and chunk.candidates:
            parts = chunk.candidates[0].content.parts
            return ''.join(p.text for p in parts if hasattr(p, 'text'))
    except Exception:
        pass
    
    return ''


def _is_retryable_error(e: Exception) -> bool:
    """リトライ可能なエラーか判定"""
    msg = str(e).lower()
    return any(x in msg for x in ['429', '503', 'overloaded', 'resource_exhausted'])


# =====================================================
# ★ストリーミングLLM呼び出し（遅延解消・検証削除）
# =====================================================
def _stream_generate(
    model: genai.GenerativeModel,
    prompt: str,
    max_retries: int = 2
) -> Generator[str, None, None]:
    """
    ストリーミング生成（根本修正版）
    
    ★改善ポイント:
    - validate_response 完全削除
    - レスポンス取得後は即座にイテレート開始
    - チャンクは即座にyield
    """
    limiter = _get_limiter()
    
    for attempt in range(max_retries + 1):
        try:
            # レート制限チェック（即時判定）
            if not limiter.wait_for_slot(timeout=30):
                if attempt < max_retries:
                    yield "⏳ レート制限中...\n"
                    continue
                yield "❌ レート制限タイムアウト"
                return
            
            limiter.record_request()
            
            # ★ストリーミング開始 - 即座にイテレート
            response = model.generate_content(prompt, stream=True)
            
            # ★検証なし - 直接イテレート
            has_content = False
            for chunk in response:
                text = _extract_text(chunk)
                if text:
                    has_content = True
                    yield text
            
            if has_content:
                return
            
            # 空レスポンスの場合のみリトライ
            if attempt < max_retries:
                yield "\n⏳ 再試行中...\n"
                time.sleep(2)
                continue
            
            yield "❌ 応答が空でした"
            return
            
        except Exception as e:
            if _is_retryable_error(e) and attempt < max_retries:
                yield f"\n⏳ API混雑中...再試行します\n"
                time.sleep(3 * (attempt + 1))
                continue
            yield f"\n❌ エラー: {e}"
            return


# =====================================================
# 障害ログ生成
# =====================================================
def generate_fake_log_by_ai(scenario_name: str, target_node, api_key: str) -> str:
    """シナリオに基づく障害ログ生成"""
    model = _get_model(api_key)
    if not model:
        return "Error: API not configured"

    limiter = _get_limiter()
    cache_key = compute_cache_hash(scenario_name, target_node.id, "log")
    
    cached = limiter.get_cache(cache_key)
    if cached:
        return cached

    vendor = target_node.metadata.get("vendor", "Generic")
    prompt = f"CLIログ生成。ホスト:{target_node.id} ベンダー:{vendor} シナリオ:{scenario_name}。コマンド2個と出力のみ。"

    try:
        if not limiter.wait_for_slot(timeout=30):
            return "Error: Rate limit"
        limiter.record_request()
        
        response = model.generate_content(prompt)
        result = response.text if response else "Error: No response"
        limiter.set_cache(cache_key, result)
        return result
    except Exception as e:
        return f"Error: {e}"


# =====================================================
# 初期症状予測
# =====================================================
def predict_initial_symptoms(scenario_name: str, api_key: str) -> Dict:
    """障害シナリオから初期症状を予測"""
    model = _get_model(api_key)
    if not model:
        return {}

    limiter = _get_limiter()
    cache_key = compute_cache_hash(scenario_name, "", "symptoms")
    
    cached = limiter.get_cache(cache_key)
    if cached:
        return cached

    prompt = f'シナリオ「{scenario_name}」の症状をJSON出力。キー:alarm,ping,log'

    try:
        if not limiter.wait_for_slot(timeout=30):
            return {}
        limiter.record_request()
        
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        limiter.set_cache(cache_key, result)
        return result
    except Exception:
        return {}


# =====================================================
# 原因分析レポート（非ストリーミング）
# =====================================================
def generate_analyst_report(
    scenario: str,
    target_node,
    topology_context: str,
    target_conf: str,
    verification_context: str,
    api_key: str
) -> str:
    """原因分析レポート（非ストリーミング版）"""
    model = _get_model(api_key)
    if not model:
        return "Error: API not configured"

    limiter = _get_limiter()
    cache_key = compute_cache_hash(scenario, target_node.id if target_node else "", "report")
    
    cached = limiter.get_cache(cache_key)
    if cached:
        return cached

    device_id = target_node.id if target_node else "Unknown"
    vendor = target_node.metadata.get("vendor", "Unknown") if target_node else "Unknown"

    prompt = f"""障害分析レポート
シナリオ: {scenario}
デバイス: {device_id} ({vendor})

【重要】文体は「です、ます」調で記載してください。

## 障害概要
## 発生原因
## 影響範囲
## 技術的根拠
"""

    try:
        if not limiter.wait_for_slot(timeout=30):
            return "Error: Rate limit"
        limiter.record_request()
        
        response = model.generate_content(prompt)
        result = response.text if response else "Error: No response"
        limiter.set_cache(cache_key, result)
        return result
    except Exception as e:
        return f"Error: {e}"


# =====================================================
# ★原因分析レポート（ストリーミング版・遅延解消）
# =====================================================
def generate_analyst_report_streaming(
    scenario: str,
    target_node,
    topology_context,
    target_conf: str,
    verification_context: str,
    api_key: str,
    max_retries: int = 2,
    backoff: float = 3.0
) -> Generator[str, None, None]:
    """
    原因分析レポート（ストリーミング版）
    
    ★根本修正:
    - validate_response 削除
    - キャッシュヒット時は即座に全文返却
    - ストリーミング開始後は即座にyield
    """
    model = _get_model(api_key)
    if not model:
        yield "Error: API not configured"
        return

    limiter = _get_limiter()
    
    # ★キャッシュチェック - ヒット時は即座に返却
    device_id = target_node.id if target_node else "Unknown"
    cache_key = compute_cache_hash(scenario, device_id, "report_stream")
    
    cached = limiter.get_cache(cache_key)
    if cached:
        yield cached
        return

    vendor = target_node.metadata.get("vendor", "Unknown") if target_node else "Unknown"

    prompt = f"""障害分析レポート
シナリオ: {scenario}
デバイス: {device_id} ({vendor})

【重要】文体は「です、ます」調で記載してください。

## 障害概要
## 発生原因
## 影響範囲
## 技術的根拠
## 切り分け判断
"""

    # ★ストリーミング生成（遅延なし）
    full_text = ""
    for chunk in _stream_generate(model, prompt, max_retries):
        full_text += chunk
        yield chunk

    # 完了後にキャッシュ保存
    if full_text and not full_text.startswith("❌"):
        limiter.set_cache(cache_key, full_text)


# =====================================================
# 復旧コマンド生成（非ストリーミング）
# =====================================================
def generate_remediation_commands(
    scenario: str,
    analysis_result: str,
    target_node,
    api_key: str
) -> str:
    """復旧手順（非ストリーミング版）"""
    model = _get_model(api_key)
    if not model:
        return "Error: API not configured"

    limiter = _get_limiter()
    device_id = target_node.id if target_node else "Unknown"
    cache_key = compute_cache_hash(scenario, device_id, "remediation")
    
    cached = limiter.get_cache(cache_key)
    if cached:
        return cached

    vendor = target_node.metadata.get("vendor", "Unknown") if target_node else "Unknown"

    prompt = f"""復旧手順
デバイス: {device_id} ({vendor})
シナリオ: {scenario}

【重要】文体は「です、ます」調で記載してください。

## 前提作業
## 復旧コマンド
## 正常性確認
"""

    try:
        if not limiter.wait_for_slot(timeout=30):
            return "Error: Rate limit"
        limiter.record_request()
        
        response = model.generate_content(prompt)
        result = response.text if response else "Error: No response"
        limiter.set_cache(cache_key, result)
        return result
    except Exception as e:
        return f"Error: {e}"


# =====================================================
# ★復旧コマンド生成（ストリーミング版・遅延解消）
# =====================================================
def generate_remediation_commands_streaming(
    scenario: str,
    analysis_result: str,
    target_node,
    api_key: str,
    max_retries: int = 2,
    backoff: float = 3.0
) -> Generator[str, None, None]:
    """
    復旧手順（ストリーミング版）
    
    ★根本修正:
    - validate_response 削除
    - キャッシュヒット時は即座に全文返却
    - ストリーミング開始後は即座にyield
    """
    model = _get_model(api_key)
    if not model:
        yield "Error: API not configured"
        return

    limiter = _get_limiter()
    
    # ★キャッシュチェック - ヒット時は即座に返却
    device_id = target_node.id if target_node else "Unknown"
    cache_key = compute_cache_hash(scenario, device_id, "remediation_stream")
    
    cached = limiter.get_cache(cache_key)
    if cached:
        yield cached
        return

    vendor = target_node.metadata.get("vendor", "Unknown") if target_node else "Unknown"

    prompt = f"""復旧手順
デバイス: {device_id} ({vendor})
シナリオ: {scenario}

【重要】文体は「です、ます」調で記載してください。

## 実施前提
## バックアップ手順
## 復旧コマンド
## ロールバック手順
## 正常性確認
"""

    # ★ストリーミング生成（遅延なし）
    full_text = ""
    for chunk in _stream_generate(model, prompt, max_retries):
        full_text += chunk
        yield chunk

    # 完了後にキャッシュ保存
    if full_text and not full_text.startswith("❌"):
        limiter.set_cache(cache_key, full_text)


# =====================================================
# 診断シミュレーション
# =====================================================
def run_diagnostic_simulation(
    scenario_type: str,
    target_node=None,
    api_key: str = None
) -> Dict:
    """診断シミュレーション実行"""
    time.sleep(1)

    if "正常" in scenario_type:
        return {"status": "SKIPPED", "sanitized_log": "No action required.", "error": None}

    if "[Live]" in scenario_type:
        commands = ["terminal length 0", "show version", "show interface brief"]
        try:
            with ConnectHandler(**SANDBOX_DEVICE) as ssh:
                if not ssh.check_enable_mode():
                    ssh.enable()
                raw_output = f"Connected to: {ssh.find_prompt()}\n"
                for cmd in commands:
                    raw_output += f"\n[{cmd}]\n{ssh.send_command(cmd)}\n"
        except Exception as e:
            return {"status": "ERROR", "sanitized_log": "", "error": str(e)}
        return {"status": "SUCCESS", "sanitized_log": sanitize_output(raw_output), "error": None}

    elif "全回線断" in scenario_type or "サイレント" in scenario_type or "両系" in scenario_type:
        return {"status": "ERROR", "sanitized_log": "", "error": "Connection timed out"}

    else:
        if api_key and target_node:
            raw_output = generate_fake_log_by_ai(scenario_type, target_node, api_key)
            return {"status": "SUCCESS", "sanitized_log": sanitize_output(raw_output), "error": None}
        return {"status": "ERROR", "sanitized_log": "", "error": "Missing API key or target"}


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
    """修復ステップを並列実行"""

    def backup_step():
        time.sleep(1)
        return RemediationResult("Backup", "success", f"Backup created for {device_id}")

    def apply_step():
        time.sleep(2)
        return RemediationResult("Apply", "success", "Applied remediation")

    def verify_step():
        time.sleep(1)
        return RemediationResult("Verify", "success", {"overall": "HEALTHY"})

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(backup_step): "Backup",
            executor.submit(apply_step): "Apply",
            executor.submit(verify_step): "Verify",
        }

        results = {}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result(timeout=timeout_per_step)
            except Exception as e:
                results[name] = RemediationResult(name, "failed", error=str(e))

    return results
