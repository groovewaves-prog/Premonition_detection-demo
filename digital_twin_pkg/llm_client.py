# digital_twin_pkg/llm_client.py
# Phase 6a/6b: AIOps 内部 LLM クライアント
#
# ════════════════════════════════════════════════════════
#  ハイブリッドルーティング設計（Phase 6b 最終版）
# ════════════════════════════════════════════════════════
#
#  使用LLM:
#    Ollama ローカル: gemma-3-12b-it（標準スコアリング）
#    Google API:      gemma-3-12b-it（エッジケース・品質要求タスク）
#
#  ルーティング:
#    backend="google"                → Google Gemma API 固定
#    backend="ollama", 標準タスク    → Ollama 優先（失敗時 Google にFB）
#    backend="ollama", エッジケース  → Google Gemma API 固定（品質保証）
#      エッジケース: maintenance_plan / pseudo_normal / degradation_trajectory
#
#  Phase 6a: topology.metadata から vendor/os/model/last_change を抽出し
#            LLMプロンプトに「機器コンテキスト」として注入する。

from __future__ import annotations

import json
import logging
import time
import hashlib
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── Google Gemma SDK ────────────────────────────────────
try:
    import google.genai as genai
    import google.genai.types as genai_types
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False
    logger.info("google-genai SDK not installed. `pip install google-genai`")

# ── Ollama クライアント ──────────────────────────────────
try:
    from .llm_local import OllamaClient
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
    OllamaClient = None  # type: ignore


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# データクラス
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class LLMScores:
    semantic:    float = 0.5
    trend:       float = 0.5
    volatility:  float = 0.5
    history:     float = 0.5
    interaction: float = 0.5
    narrative:   str   = ""

@dataclass
class LLMResult:
    scores:            LLMScores
    anomaly_type_hint: str = "point"
    error:             str = ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# スコアキャッシュ（in-memory, 30分 TTL, 512エントリ上限）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _ScoreCache:
    _TTL = 1800
    _MAX = 512

    def __init__(self):
        self._store: Dict[str, tuple] = {}

    def _key(self, text: str, ctx: str) -> str:
        return hashlib.md5(f"{text}|{ctx}".encode()).hexdigest()

    def get(self, text: str, ctx: str) -> Optional[LLMResult]:
        k = self._key(text, ctx)
        entry = self._store.get(k)
        if entry and (time.time() - entry[0]) < self._TTL:
            return entry[1]
        return None

    def set(self, text: str, ctx: str, result: LLMResult):
        if len(self._store) >= self._MAX:
            oldest = min(self._store, key=lambda k: self._store[k][0])
            del self._store[oldest]
        self._store[self._key(text, ctx)] = (time.time(), result)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# プロンプトテンプレート
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SCORING_PROMPT = """\
あなたはネットワーク・サーバ運用の専門家AIです。
以下のアラーム情報を分析し、JSONのみを返してください（説明文・コードフェンス禁止）。

【アラームテキスト】
{alarm_text}

【デバイス種別】
{device_type}

【機器コンテキスト（ベンダー・OS・モデル・構成変更履歴）】
{vendor_context}

【追加コンテキスト】
{context}

返すJSONの形式:
{{
  "semantic":    0.0〜1.0,
  "trend":       0.0〜1.0,
  "volatility":  0.0〜1.0,
  "history":     0.0〜1.0,
  "interaction": 0.0〜1.0,
  "anomaly_type": "point|contextual|collective|cascading",
  "narrative": "運用者向け説明文（3文以内・日本語）"
}}
"""

_MAINTENANCE_PROMPT = """\
以下の機器状態からメンテナンス計画文を生成してください（日本語・簡潔に）。

故障モード: {failure_mode}
推定残存寿命: {rul_hours}時間（±{rul_margin}時間）
推奨保守窓: {maintenance_window}
影響範囲: {impact_desc}

返すJSONのみ（コードフェンス禁止）:
{{
  "summary": "状況の1文要約",
  "actions": ["優先アクション1", "優先アクション2", "優先アクション3"],
  "risk_if_ignored": "対処しない場合のリスク（1文）"
}}
"""

_DEGRADATION_PROMPT = """\
デバイス種別: {device_type}
故障モード: {failure_mode}

この故障モードに至る典型的な劣化軌跡を時系列JSONで生成してください。
正常期（0-70%）→ 劣化開始（70-90%）→ 故障直前（90-100%）の業界統計パターンで。
点数: {n_points}点。RULラベル（各時点での残余時間[時間]）も含めること。

返すJSONのみ（コードフェンス禁止）:
{{
  "trajectory": [数値のリスト（{n_points}個）],
  "rul_labels": [数値のリスト（{n_points}個）],
  "unit": "dBm等の単位",
  "failure_threshold": 閾値数値
}}
"""

_PSEUDO_NORMAL_PROMPT = """\
デバイス種別: {device_type}
メトリクス名: {metric_name}

このデバイスの典型的な正常動作時のメトリクス値を疑似生成してください。
時間窓: {window_hours}時間分、サンプリング間隔: {interval_min}分。
業界統計・季節性・日内変動を考慮した現実的な値にすること。

返すJSONのみ（コードフェンス禁止）:
{{
  "values": [数値のリスト],
  "timestamps_offset_min": [開始からの経過分のリスト],
  "mean": 平均値,
  "std": 標準偏差,
  "unit": "単位"
}}
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# メインクライアント
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class InternalLLMClient:
    """
    AIOps 内部 LLM クライアント（Phase 6b 最終版）

    使用モデル:
      - Ollama ローカル:  gemma-3-12b-it (標準スコアリング用)
      - Google Gemma API: gemma-3-12b-it (エッジケース・品質要求タスク用)

    ルーティングルール:
      backend="google"             → Google Gemma API 固定
      backend="ollama", 標準タスク  → Ollama 優先（失敗時 Google にフォールバック）
      backend="ollama", エッジケース → Google Gemma API 固定（品質保証）
    """

    # Google API を優先するタスク種別
    _GOOGLE_PREFERRED_TASKS = frozenset([
        "maintenance_plan",       # メンテナンス計画文（日本語品質・構成力が必要）
        "pseudo_normal",          # 疑似正常データ生成（統計的整合性・長文JSON）
        "degradation_trajectory", # 疑似劣化軌跡生成（1000点時系列・相関考慮）
    ])

    def __init__(
        self,
        sanitize_fn,
        api_key:         Optional[str] = None,      # Google AI Studio API Key
        llm_backend:     str = "ollama",             # "ollama" | "google"
        ollama_base_url: str = "http://localhost:11434",
        ollama_model:    str = "gemma3:12b",         # Ollama モデル名
        google_model:    str = "gemma-3-12b-it",     # Google API モデル名
    ):
        self._sanitize      = sanitize_fn
        self._cache         = _ScoreCache()
        self._google_client = None
        self._ollama_client = None
        self._llm_backend   = llm_backend
        self._google_model  = google_model
        self._ollama_model  = ollama_model

        # Google Gemma API は常に初期化を試みる（エッジケースのFB用）
        if HAS_GOOGLE:
            try:
                self._google_client = (
                    genai.Client(api_key=api_key) if api_key
                    else genai.Client()
                )
                logger.info(
                    f"InternalLLMClient: Google Gemma API OK (model={google_model})"
                )
            except Exception as e:
                logger.warning(
                    f"InternalLLMClient: Google Gemma API init failed: {e}"
                )

        # Ollama はバックエンドが "ollama" のときのみ初期化
        if llm_backend == "ollama" and HAS_OLLAMA:
            try:
                oc = OllamaClient(base_url=ollama_base_url, model=ollama_model)
                if oc.ping():
                    self._ollama_client = oc
                    logger.info(
                        f"InternalLLMClient: Ollama OK "
                        f"({ollama_base_url}, model={ollama_model})"
                    )
                else:
                    logger.warning(
                        f"InternalLLMClient: Ollama ping failed ({ollama_base_url})"
                    )
            except Exception as e:
                logger.warning(f"InternalLLMClient: Ollama init failed: {e}")

    @property
    def available(self) -> bool:
        return (self._ollama_client is not None) or (self._google_client is not None)

    @property
    def backend_name(self) -> str:
        ollama_ok = self._ollama_client is not None
        google_ok = self._google_client is not None
        if self._llm_backend == "ollama":
            if ollama_ok and google_ok:
                return (
                    f"Hybrid: Ollama({self._ollama_model}) + "
                    f"Google Gemma API({self._google_model})"
                )
            elif ollama_ok:
                return f"Ollama ({self._ollama_model})"
            elif google_ok:
                return f"Google Gemma API ({self._google_model}) [Ollama不可・FB]"
        elif google_ok:
            return f"Google Gemma API ({self._google_model})"
        return "Rule-based (LLM unavailable)"

    # ── 統一呼び出しルーター ─────────────────────────────

    def _call_llm(
        self,
        prompt:    str,
        max_tokens: int = 512,
        task_type:  str = "standard",
    ) -> str:
        """
        タスク種別でバックエンドを自動選択。

          backend="google"              → Google Gemma API 固定
          backend="ollama", 標準タスク   → Ollama 優先 → 失敗時 Google
          backend="ollama", エッジケース → Google Gemma API 固定
        """
        use_google = (
            self._llm_backend == "google"
            or task_type in self._GOOGLE_PREFERRED_TASKS
        )

        # Ollama（標準タスク）
        if not use_google and self._ollama_client is not None:
            try:
                return self._ollama_client.chat(prompt, max_tokens=max_tokens)
            except Exception as e:
                logger.warning(
                    f"Ollama failed (task={task_type}), "
                    f"falling back to Google Gemma API: {e}"
                )

        # Google Gemma API
        if self._google_client is not None:
            try:
                resp = self._google_client.models.generate_content(
                    model=self._google_model,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=0.1,
                    ),
                )
                return resp.text.strip()
            except Exception as e:
                logger.warning(f"Google Gemma API failed: {e}")
                raise RuntimeError(f"Google Gemma API error: {e}") from e

        raise RuntimeError(
            "No LLM backend available. "
            "Ollama を起動するか Google API キーを設定してください。"
        )

    # ── 標準タスク: 6次元スコアリング ──────────────────────

    def score_alarm(
        self,
        alarm_text:     str,
        device_id:      str,
        device_type:    str = "network",
        signal_count:   int = 1,
        affected_count: int = 0,
        rule_pattern:   str = "",
        vendor_context: Optional[str] = None,
    ) -> LLMResult:
        """
        アラームテキストから 6 次元スコアを生成（Ollama 優先）。

        vendor_context (Phase 6a):
          topology.metadata から抽出したベンダー情報。
          例: "vendor=Cisco | os=IOS-XE | model=ISR4451-X |
               last_change=(2025-01-15 OSPF変更)"
        """
        sanitized = self._sanitize(alarm_text)
        ctx       = (
            f"signal_count={signal_count} "
            f"affected={affected_count} "
            f"pattern={rule_pattern}"
        )
        cache_ctx = f"{ctx}|{vendor_context or ''}"

        cached = self._cache.get(sanitized, cache_ctx)
        if cached:
            return cached

        if not self.available:
            return self._fallback(rule_pattern, signal_count, affected_count)

        try:
            prompt = _SCORING_PROMPT.format(
                alarm_text     = sanitized,
                device_type    = device_type,
                vendor_context = vendor_context or "情報なし",
                context        = ctx,
            )
            raw    = self._call_llm(prompt, max_tokens=512, task_type="standard")
            result = self._parse_score(raw)
            self._cache.set(sanitized, cache_ctx, result)
            return result
        except Exception as e:
            logger.warning(f"score_alarm failed: {e}")
            return self._fallback(rule_pattern, signal_count, affected_count)

    # ── エッジケース: Google Gemma API 固定タスク ──────────

    def generate_maintenance_plan(
        self,
        failure_mode:       str,
        rul_hours:          int,
        rul_margin:         int,
        maintenance_window: str,
        impact_desc:        str,
    ) -> Dict:
        """メンテナンス計画文を生成（Google Gemma API 固定）。"""
        if not self.available:
            return {
                "summary": (
                    f"{failure_mode}による劣化を検知。"
                    f"推定{rul_hours}時間後に限界値へ到達。"
                ),
                "actions": [
                    "詳細状態確認・ログ取得",
                    "予備品手配",
                    f"{maintenance_window}での部品交換",
                ],
                "risk_if_ignored": "放置すると予期しない障害に連鎖するリスクがあります。",
            }
        try:
            prompt = _MAINTENANCE_PROMPT.format(
                failure_mode=failure_mode, rul_hours=rul_hours,
                rul_margin=rul_margin, maintenance_window=maintenance_window,
                impact_desc=impact_desc,
            )
            # task_type="maintenance_plan" → Google Gemma API 固定
            raw = self._call_llm(prompt, max_tokens=512, task_type="maintenance_plan")
            return json.loads(raw.replace("```json","").replace("```","").strip())
        except Exception as e:
            logger.warning(f"generate_maintenance_plan failed: {e}")
            return {"summary": "", "actions": [], "risk_if_ignored": ""}

    def generate_degradation_trajectory(
        self,
        device_type:  str,
        failure_mode: str,
        n_points:     int = 200,
    ) -> Dict:
        """PHM コールドスタート用の疑似劣化軌跡を生成（Google Gemma API 固定）。"""
        if not self.available:
            return {"trajectory":[], "rul_labels":[], "unit":"", "failure_threshold":0}
        try:
            prompt = _DEGRADATION_PROMPT.format(
                device_type=device_type, failure_mode=failure_mode, n_points=n_points,
            )
            raw = self._call_llm(prompt, max_tokens=2048, task_type="degradation_trajectory")
            return json.loads(raw.replace("```json","").replace("```","").strip())
        except Exception as e:
            logger.warning(f"generate_degradation_trajectory failed: {e}")
            return {"trajectory":[], "rul_labels":[], "unit":"", "failure_threshold":0}

    def generate_pseudo_normal(
        self,
        device_type:  str,
        metric_name:  str,
        window_hours: int = 48,
        interval_min: int = 5,
    ) -> Dict:
        """GDN コールドスタート用の疑似正常データを生成（Google Gemma API 固定）。"""
        if not self.available:
            return {"values":[], "timestamps_offset_min":[], "mean":0, "std":0, "unit":""}
        try:
            prompt = _PSEUDO_NORMAL_PROMPT.format(
                device_type=device_type, metric_name=metric_name,
                window_hours=window_hours, interval_min=interval_min,
            )
            raw = self._call_llm(prompt, max_tokens=2048, task_type="pseudo_normal")
            return json.loads(raw.replace("```json","").replace("```","").strip())
        except Exception as e:
            logger.warning(f"generate_pseudo_normal failed: {e}")
            return {"values":[], "timestamps_offset_min":[], "mean":0, "std":0, "unit":""}

    # ── 内部ユーティリティ ───────────────────────────────

    def _parse_score(self, raw: str) -> LLMResult:
        clean = raw.replace("```json","").replace("```","").strip()
        data  = json.loads(clean)
        scores = LLMScores(
            semantic    = float(data.get("semantic",    0.5)),
            trend       = float(data.get("trend",       0.5)),
            volatility  = float(data.get("volatility",  0.5)),
            history     = float(data.get("history",     0.5)),
            interaction = float(data.get("interaction", 0.5)),
            narrative   = str(data.get("narrative",     "")),
        )
        return LLMResult(
            scores            = scores,
            anomaly_type_hint = str(data.get("anomaly_type", "point")),
        )

    def _fallback(
        self, rule_pattern: str, signal_count: int, affected_count: int
    ) -> LLMResult:
        """LLM 不使用時のルールベースフォールバック。"""
        semantic_map = {
            "optical":0.80, "stp_loop":0.90, "route_instability":0.75,
            "microburst":0.65, "memory_leak":0.60,
            "generic_error":0.40, "analysis_signal":0.55,
        }
        semantic    = semantic_map.get(rule_pattern, 0.50)
        interaction = (
            0.90 if signal_count >= 6 else
            0.70 if signal_count >= 3 else
            min(0.99, 0.40 + signal_count * 0.05)
        )
        if affected_count >= 5:
            interaction = min(0.99, interaction + 0.10)
        scores = LLMScores(
            semantic=semantic, trend=0.50, volatility=0.50,
            history=0.50, interaction=interaction,
            narrative=f"{rule_pattern}に関連するアラームを検知しました。" if rule_pattern else "アラームを検知しました。",
        )
        return LLMResult(scores=scores, anomaly_type_hint="point", error="llm_unavailable")
