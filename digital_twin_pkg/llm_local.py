# digital_twin_pkg/llm_local.py
# Phase 6b: Ollama ローカル LLM クライアント
#
# 依存: requests のみ（標準ライブラリ相当）
# 推奨モデル: qwen2.5:7b（日本語/JSON安定）、llama3.1:8b、mistral:7b

from __future__ import annotations
import json
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_TIMEOUT_SEC   = 60
_PING_TIMEOUT  = 5
_MAX_RETRIES   = 2

_SYSTEM_PROMPT = """\
あなたはネットワーク・サーバ運用の専門家AIです。
指示に従い JSON のみを出力してください。
Markdown のコードフェンス（```）や余分な説明文は絶対に出力しないでください。
JSON キーは英語（ASCII）、数値フィールドは数値型で返してください。
"""


class OllamaClient:
    """Ollama サーバへの HTTP クライアント。"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
        timeout: int = _TIMEOUT_SEC,
    ):
        self.base_url = base_url.rstrip("/")
        self.model    = model
        self.timeout  = timeout

    def ping(self) -> bool:
        try:
            import requests
            r = requests.get(f"{self.base_url}/api/version", timeout=_PING_TIMEOUT)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        try:
            import requests
            r = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if r.status_code == 200:
                return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            pass
        return []

    def chat(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        import requests
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        for attempt in range(_MAX_RETRIES + 1):
            try:
                r = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload, timeout=self.timeout,
                )
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                return r.json().get("message", {}).get("content", "").strip()
            except requests.exceptions.Timeout:
                if attempt >= _MAX_RETRIES:
                    raise RuntimeError(f"Ollama timeout (model={self.model})")
            except Exception as e:
                if attempt >= _MAX_RETRIES:
                    raise RuntimeError(f"OllamaClient: {e}") from e
        raise RuntimeError("OllamaClient: all retries exhausted")


def test_ollama_connection(
    base_url: str = "http://localhost:11434",
    model: str = "qwen2.5:7b",
) -> Dict[str, Any]:
    """UI から呼ぶ接続テスト。"""
    result: Dict[str, Any] = {
        "connected": False, "models": [], "model_installed": False,
        "inference_ok": False, "inference_ms": 0, "error": None,
    }
    client = OllamaClient(base_url=base_url, model=model, timeout=30)
    if not client.ping():
        result["error"] = (
            f"Ollama に接続できません: {base_url}\n"
            "`ollama serve` が起動していることを確認してください。"
        )
        return result
    result["connected"] = True
    models = client.list_models()
    result["models"] = models
    result["model_installed"] = any(m.startswith(model.split(":")[0]) for m in models)
    if not result["model_installed"]:
        result["error"] = (
            f"モデル '{model}' が未インストールです。\n"
            f"`ollama pull {model}` でインストールしてください。\n"
            f"検出済みモデル: {models}"
        )
        return result
    try:
        t0 = time.time()
        text = client.chat('{"semantic":0.5}', max_tokens=64, temperature=0.0)
        result["inference_ms"] = int((time.time() - t0) * 1000)
        json.loads(text.replace("```json","").replace("```","").strip())
        result["inference_ok"] = True
    except Exception as e:
        result["error"] = f"推論テストエラー: {e}"
    return result
