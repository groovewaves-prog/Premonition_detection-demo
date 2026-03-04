# digital_twin_pkg/engine.py  ―  DigitalTwinEngine コアロジック（Phase1 Predict API + RUL予測）
import logging
import time
import json
import uuid
import re
import os
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict
import traceback

from .config import *
from .rules import EscalationRule, DEFAULT_RULES, MAINTENANCE_SIGNATURES
from .storage import StorageManager
from .audit import AuditBuilder
from .tuning import AutoTuner
from .bayesian import BayesianInferenceEngine
from .gnn import create_gnn_engine
from .llm_client import InternalLLMClient, LLMScores  # Phase 6a/6b
from .vector_store import VectorStore  # Phase 6c*

try:
    from sentence_transformers import SentenceTransformer
    HAS_BERT = True
except ImportError:
    HAS_BERT = False

logger = logging.getLogger(__name__)



# ==============================================================
# Phase1: Predict API + Forecast Ledger  (digital_twin_pkg)
# ==============================================================

import traceback
from dataclasses import asdict as _asdict

# DTO ─────────────────────────────────────────────────────────
from dataclasses import dataclass as _dc, field as _field
from typing import Optional as _Opt

@_dc
class PredictRequest:
    tenant_id:  str
    device_id:  str
    msg:        str
    timestamp:  float
    attrs:      dict = _field(default_factory=dict)

    def to_dict(self):
        return {"tenant_id": self.tenant_id, "device_id": self.device_id,
                "msg": self.msg, "timestamp": self.timestamp, "attrs": self.attrs or {}}

@_dc
class PredictResult:
    predicted_state:      str
    confidence:           float
    rule_pattern:         str
    category:             str
    reasons:              list = _field(default_factory=list)
    recommended_actions:  list = _field(default_factory=list)
    runbook_url:          str  = ""
    criticality:          str  = "standard"
    time_to_critical_min: int  = 60
    early_warning_hours:  int  = 24
    time_to_failure_hours: int = 336  # ★ RUL: 今から完全故障まで（時間）
    predicted_failure_datetime: str = ""  # ★ 故障発生予測日時（ISO形式）

    def to_dict(
        self,
        affected_count: int = 0,
        source: str = "real",
        llm_narrative: str = "",
        llm_anomaly_type: str = "point",
        llm_error: str = "",
        score_breakdown: dict = None,
        vendor_context: str = None,
        alarm_text: str = "",
    ):
        _expl = {
            "narrative":       llm_narrative,
            "anomaly_type":    llm_anomaly_type,
            "llm_error":       llm_error or None,
            "score_breakdown": score_breakdown or {},
            "vendor_context":  vendor_context,
        }
        return {
            "predicted_state":      self.predicted_state,
            "confidence":           float(self.confidence),
            "rule_pattern":         self.rule_pattern,
            "category":             self.category,
            "reasons":              self.reasons or [],
            "recommended_actions":  self.recommended_actions or [],
            "runbook_url":          self.runbook_url or "",
            "criticality":          self.criticality or "standard",
            "time_to_critical_min": int(self.time_to_critical_min),
            "early_warning_hours":  int(self.early_warning_hours),
            "time_to_failure_hours": int(self.time_to_failure_hours),
            "predicted_failure_datetime": self.predicted_failure_datetime,
            "is_prediction":        True,
            "source":               source,
            "prob":                 float(self.confidence),
            "label":                f"🔮 [予兆] {self.predicted_state}",
            "type":                 f"Predictive/{self.category}",
            "tier":                 1,
            "prediction_timeline":  f"{self.time_to_critical_min}分後",
            "prediction_time_to_critical_min": int(self.time_to_critical_min),
            "prediction_early_warning_hours":  int(self.early_warning_hours),
            "prediction_affected_count":       int(affected_count),
            "prediction_time_to_failure_hours": int(self.time_to_failure_hours),
            "prediction_failure_datetime":      self.predicted_failure_datetime,
            # Phase 6a
            "alarm_text":    alarm_text,
            "vendor_context": vendor_context,
            "llm_narrative": llm_narrative,
            "explanation":   _expl,
        }


class DigitalTwinEngine:
    def __init__(
        self,
        topology: Dict[str, Any],
        children_map: Optional[Dict[str, List[str]]] = None,
        tenant_id: str = "default",
        llm_config: Optional[Dict[str, Any]] = None,
    ):
        if not tenant_id or len(tenant_id) > 64: raise ValueError("Invalid tenant_id")
        self.tenant_id = tenant_id.lower()
        self.topology = topology
        self.children_map = children_map or {}
        self.storage = StorageManager(self.tenant_id, BASE_DIR)
        self.tuner = AutoTuner(self)
        self.bayesian = BayesianInferenceEngine(self.storage)
        self.gnn = create_gnn_engine(topology, children_map)

        # ★ LLM クライアント初期化
        #   スコアリング（6次元）: gemma-3-12b-it（軽量・高速）
        #   推奨アクション生成: gemini-2.0-flash-exp（高推論能力）← engine.py 内で直接呼出
        _api_key = (
            (llm_config or {}).get("google_key")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        self.llm = InternalLLMClient(
            sanitize_fn     = self._sanitize_for_llm,
            api_key         = _api_key,
            llm_backend     = "google",                  # 常に Google API
            google_model    = "gemma-3-12b-it",          # スコアリング用（軽量）
        )

        # Phase 6c*: ChromaDB ベクトルストア（類似インシデント検索）
        _vs_dir = os.path.join(BASE_DIR, ".dt_storage", self.tenant_id, "chromadb")
        self.vector_store = VectorStore(
            persist_directory=_vs_dir,
            tenant_id=self.tenant_id,
        )
        self.rules: List[EscalationRule] = []
        self._metric_rules: List[EscalationRule] = []
        self.history: List[Dict] = []
        self.outcomes: List[Dict] = []
        self.incident_register: List[Dict] = []
        self.maintenance_windows: List[Dict] = []
        self.evaluation_state: Dict = {}
        self.shadow_eval_state: Dict = {}
        self._model = None
        self._rule_embeddings = None
        self._model_loaded = False
        # ★ 予測結果キャッシュ（再描画時の冗長API呼び出し防止）
        self._predict_cache: Dict[str, Any] = {}   # key: "dev_id|pattern|level" → predictions
        self._predict_cache_ttl = 30.0              # 30秒間キャッシュ有効
        self._rules_sot = (os.environ.get(ENV_RULES_SOT, "json") or "json").strip().lower()
        self.reload_all()
        # ★ BUG FIX: _ensure_model_loaded() を __init__ から除去
        #   SentenceTransformer('all-MiniLM-L6-v2') が未キャッシュ時にモデルDLを
        #   試み、ネットワーク制限下で無期限ハング → Streamlit 白い画面の原因。
        #   predict_api → _rule_match_simple はBERTモデル不要のため遅延ロードで十分。
        #   _match_rule() 初回呼出時にのみロードする。

    def reload_all(self):
        self._load_rules()
        self.history = self.storage.load_json("history", [])
        self.outcomes = self.storage.load_json("outcomes", [])
        self.incident_register = self.storage.load_json("incident_register", [])
        self.maintenance_windows = self.storage.load_json("maintenance_windows", [])
        self.evaluation_state = self.storage.load_json("evaluation_state", {})
        self.shadow_eval_state = self.storage.load_json("shadow_eval_state", {})
        self._init_forecast_ledger()

    def _sanitize_rule_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in data.items() if not k.startswith('_')}

    def _load_rules(self):
        loaded_from_db = False
        if self._rules_sot == "db":
            db_rules_json = self.storage.rule_config_get_all_json_strs()
            if db_rules_json:
                try:
                    self.rules = [EscalationRule(**self._sanitize_rule_data(json.loads(s))) for s in db_rules_json]
                    loaded_from_db = True
                except: pass
        if not loaded_from_db:
            path = self.storage.paths["rules"]
            if not os.path.exists(path):
                self.rules = [EscalationRule(**self._sanitize_rule_data(asdict(r))) for r in DEFAULT_RULES]
                self.storage.save_json_atomic("rules", [self._sanitize_rule_data(asdict(r)) for r in self.rules])
            else:
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self.rules = [EscalationRule(**self._sanitize_rule_data(item)) for item in data]
                except Exception as e:
                    self.rules = [EscalationRule(**self._sanitize_rule_data(asdict(r))) for r in DEFAULT_RULES]
            self.storage._seed_rule_config_from_rules_json([self._sanitize_rule_data(asdict(r)) for r in self.rules])
        self._metric_rules = [r for r in self.rules if (r.requires_trend or r.requires_volatility) and r.trend_metric_regex]

    def _ensure_model_loaded(self):
        """
        SentenceTransformer モデルを遅延ロードする。
        タイムアウト付き: ネットワーク制限下でDLがハングするのを防止。
        """
        if self._model_loaded: return
        if not HAS_BERT:
            self._model_loaded = True
            return
        try:
            import threading
            _load_ok = [False]
            _model_ref = [None]
            _embeddings_ref = [None]

            def _load_worker():
                try:
                    m = SentenceTransformer('all-MiniLM-L6-v2')
                    phrases = []
                    indices = []
                    for idx, r in enumerate(self.rules):
                        for p in r.semantic_phrases:
                            phrases.append(p)
                            indices.append(idx)
                    emb = None
                    if phrases:
                        emb = {"vectors": m.encode(phrases, convert_to_numpy=True),
                               "indices": indices}
                    _model_ref[0] = m
                    _embeddings_ref[0] = emb
                    _load_ok[0] = True
                except Exception:
                    pass

            t = threading.Thread(target=_load_worker, daemon=True)
            t.start()
            t.join(timeout=10)  # ★ 最大10秒で打ち切り

            if _load_ok[0]:
                self._model = _model_ref[0]
                self._rule_embeddings = _embeddings_ref[0]
                logger.info("SentenceTransformer loaded successfully")
            else:
                logger.info("SentenceTransformer skipped (timeout or error)")
            self._model_loaded = True
        except Exception:
            self._model_loaded = True

    def _match_rule(self, alarm_text: str) -> Tuple[Optional[EscalationRule], float]:
        # ★ 遅延ロード: 初回呼出時にのみ BERT モデルをロード
        if not self._model_loaded:
            self._ensure_model_loaded()

        text_lower = alarm_text.lower()
        for rule in self.rules:
            if rule._compiled_regex and rule._compiled_regex.search(alarm_text):
                return rule, 1.0
            if rule.pattern in text_lower:
                return rule, 1.0
        if self._model and self._rule_embeddings:
            try:
                query_vec = self._model.encode([alarm_text], convert_to_numpy=True)
                rule_vecs = self._rule_embeddings["vectors"]
                similarities = np.dot(rule_vecs, query_vec.T).flatten()
                norms = np.linalg.norm(rule_vecs, axis=1) * np.linalg.norm(query_vec)
                cosine_sim = similarities / np.where(norms==0, 1e-10, norms)
                best_idx = np.argmax(cosine_sim)
                best_score = float(cosine_sim[best_idx])
                rule_idx = self._rule_embeddings["indices"][best_idx]
                rule = self.rules[rule_idx]
                if best_score >= (rule.embedding_threshold or 0.40):
                    return rule, best_score
            except Exception: pass
        return None, 0.0

    def _calculate_confidence(
        self,
        rule: EscalationRule,
        device_id: str,
        match_quality: float,
        llm_scores: Optional[LLMScores] = None,  # Phase 6a
    ) -> float:
        attrs = self.topology.get(device_id, {})
        if not isinstance(attrs, dict):
            try: attrs = vars(attrs)
            except: attrs = {}
        rg = attrs.get('redundancy_group')
        has_redundancy = bool(rg)
        children = self.children_map.get(device_id, [])
        is_spof = bool(children and not has_redundancy)
        confidence = rule.base_confidence
        confidence *= (0.8 + 0.2 * match_quality)
        if has_redundancy: confidence *= (1.0 - ROI_CONSERVATIVE_FACTOR * 0.2)
        if is_spof: confidence *= 1.1

        # Phase 6a: LLM 6次元スコアを信頼度に反映
        if llm_scores is not None:
            # semantic・interaction・trend の加重平均で信頼度を補正
            llm_factor = (
                llm_scores.semantic    * 0.35 +
                llm_scores.interaction * 0.30 +
                llm_scores.trend       * 0.20 +
                llm_scores.volatility  * 0.15
            )
            # LLM スコアと既存スコアを 60:40 でブレンド
            confidence = confidence * 0.60 + (confidence * llm_factor) * 0.40

        return min(0.99, max(0.1, confidence))

    def _sanitize_for_llm(self, text: str) -> str:
        """
        LLM送信前のデータサニタイズ
        
        - IPアドレスのマスキング
        - プライベート情報の除去
        - 機密情報の匿名化
        """
        import re
        
        sanitized = text
        
        # IPv4アドレスのマスキング
        sanitized = re.sub(
            r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'IP_MASKED',
            sanitized
        )
        
        # IPv6アドレスのマスキング
        sanitized = re.sub(
            r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',
            'IPV6_MASKED',
            sanitized
        )
        
        # MACアドレスのマスキング
        sanitized = re.sub(
            r'\b(?:[0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2}\b',
            'MAC_MASKED',
            sanitized
        )
        
        # ホスト名の一般化（prod-, dev-, test-などを除去）
        sanitized = re.sub(
            r'\b(prod|dev|test|stage|staging)-[\w-]+',
            'HOSTNAME_MASKED',
            sanitized,
            flags=re.IGNORECASE
        )
        
        # ASN (AS番号)のマスキング
        sanitized = re.sub(
            r'\bAS\d+\b',
            'AS_MASKED',
            sanitized
        )
        
        # VLAN IDのマスキング
        sanitized = re.sub(
            r'\bVLAN\s*\d+\b',
            'VLAN_MASKED',
            sanitized,
            flags=re.IGNORECASE
        )
        
        return sanitized

    def _batch_generate_llm_recommendations(
        self,
        candidates: set,
        msg_map: Dict[str, List[str]]
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        複数デバイスの推奨アクションをバッチ生成（コスト削減・性能向上）
        
        同じルールパターンの予兆をグループ化し、1回のLLM呼び出しで処理
        
        Args:
            candidates: 予兆候補デバイスIDのセット
            msg_map: デバイスID → アラームメッセージのマップ
        
        Returns:
            ルールパターン → 推奨アクションのマップ
        """
        from collections import defaultdict
        
        # ルールパターンでグループ化
        pattern_groups = defaultdict(list)
        
        for dev_id in candidates:
            messages = msg_map.get(dev_id, [])
            if not messages:
                continue
            
            # 主要ルールを特定
            matched_signals = []
            for msg in messages:
                rule, quality = self._match_rule(msg)
                if rule and quality >= 0.30 and rule.pattern != "generic_error":
                    matched_signals.append((rule, quality, msg))
            
            if not matched_signals:
                rule, quality = self._match_rule(messages[0])
                if not rule:
                    continue
                matched_signals = [(rule, quality, messages[0])]
            
            matched_signals.sort(key=lambda x: x[1], reverse=True)
            primary_rule, primary_quality, primary_msg = matched_signals[0]
            
            pattern_groups[primary_rule.pattern].append({
                'device_id': dev_id,
                'messages': [s[2] for s in matched_signals[:3]],
                'affected_count': len(matched_signals),
                'confidence': self._calculate_confidence(primary_rule, dev_id, primary_quality)
            })
        
        # バッチLLM呼び出し
        llm_cache = {}
        WIDE_RANGE_THRESHOLD = 3
        
        for pattern, devices in pattern_groups.items():
            total_affected = sum(d['affected_count'] for d in devices)
            
            # 広範囲障害の場合のみLLM呼び出し
            if total_affected >= WIDE_RANGE_THRESHOLD:
                # 代表的なデバイスのデータを使用
                representative = max(devices, key=lambda d: d['affected_count'])
                
                llm_actions = self._generate_actions_with_gemini(
                    rule_pattern=pattern,
                    affected_count=total_affected,
                    confidence=sum(d['confidence'] for d in devices) / len(devices),
                    messages=representative['messages'],
                    device_id=representative['device_id']
                )
                
                if llm_actions:
                    llm_cache[pattern] = llm_actions
                    logger.info(f"Batch LLM: Generated actions for {pattern} "
                              f"({len(devices)} devices, {total_affected} total affected)")
        
        return llm_cache

    def _generate_actions_with_gemini(
        self,
        rule_pattern: str,
        affected_count: int,
        confidence: float,
        messages: List[str],
        device_id: str
    ) -> Optional[List[Dict[str, str]]]:
        """
        Gemini API を使って状況に応じた推奨アクションを動的生成
        
        ⚠️ セキュリティ: データをサニタイズしてから送信
        
        Args:
            rule_pattern: 検出されたルールパターン
            affected_count: 影響を受けたコンポーネント数
            confidence: 予測信頼度
            messages: アラームメッセージのリスト
            device_id: デバイスID
        
        Returns:
            推奨アクションのリスト、または None（生成失敗時）
        """
        try:
            import google.generativeai as genai
            import os
            import json
            
            # ★ LLM送信の設定確認（オプトアウト可能）
            enable_llm = os.environ.get("ENABLE_LLM_RECOMMENDATIONS", "true").lower()
            if enable_llm not in ["true", "1", "yes"]:
                logger.info("LLM recommendations disabled by configuration")
                return None
            
            # API キーの取得
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                logger.warning("GEMINI_API_KEY not found. Using static recommendations.")
                return None
            
            # ★ データのサニタイズ（機密情報の除去）
            sanitized_device_id = self._sanitize_for_llm(device_id)
            # ★ 全メッセージをサニタイズ（最大50件まで）
            sanitized_messages = [self._sanitize_for_llm(msg) for msg in messages[:50]]
            
            # Gemini API の設定
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            # デバイスタイプの推定（一般化）
            device_type = "Network Device"
            if "ROUTER" in device_id.upper():
                device_type = "Router"
            elif "SWITCH" in device_id.upper():
                device_type = "Switch"
            elif "FIREWALL" in device_id.upper():
                device_type = "Firewall"
            
            # ★ プロンプトの生成（全アラームメッセージを分析させる）
            prompt = f"""あなたは20年以上の経験を持つネットワーク機器の障害対応エキスパートです。

【🚨 緊急: 広範囲障害が検出されました】

【検出された予兆の詳細】
- デバイスタイプ: {device_type}
- 検出パターン: {rule_pattern}
- **影響範囲: {affected_count}個のコンポーネント/インターフェース**
- 予測信頼度: {confidence * 100:.0f}%
- **アラーム件数: {len(sanitized_messages)}件**

【📋 実際に検出されたアラームメッセージ（全{len(sanitized_messages)}件、匿名化済み）】
```
{chr(10).join(f"{i+1:3d}. {msg[:200]}" for i, msg in enumerate(sanitized_messages))}
```

【🔍 あなたのタスク - ステップ1: アラームパターンの分析】
上記の{len(sanitized_messages)}件のアラームメッセージを詳しく分析してください：

**質問:**
1. どのコンポーネント/インターフェースが影響を受けていますか？
   - 同じインターフェース番号が繰り返し？
   - 複数の異なるインターフェース？
   - 範囲は？（例: Gi0/0/1からGi0/0/28まで）

2. エラーの種類は？
   - 光信号レベルの低下（Rx Power, Tx Power）？
   - パケットドロップ？
   - リンクフラッピング？
   
3. パターンの共通点は？
   - 全て同じタイプのエラー？
   - 特定の範囲のポート番号に集中？
   - 時系列的なパターン？

【🔴 ステップ2: 真因の推論】
**{affected_count}個のコンポーネントで同時に{rule_pattern}パターンが検出されています。**

上記のアラームパターン分析に基づいて、**最も可能性が高い真因**を特定してください：

**A. 筐体レベルの問題（全ポートに影響する場合）**
   - 電源ユニット（PSU）の故障または電圧不安定
   - マザーボード/制御基板の問題
   - 筐体内の過熱（冷却ファン故障）

**B. ソフトウェアレベルの問題**
   - ファームウェア/IOS/NOS のバグ
   - 設定ミスによる全ポート影響

**C. 環境レベルの問題**
   - データセンター空調の問題
   - 電源供給の問題（UPS、配電盤）

**判断基準:**
- 全{affected_count}個が同時に影響 → 電源またはファームウェアの可能性が高い
- 特定範囲のポートのみ → ラインカード、モジュールレベルの問題
- 光信号レベル低下 → 電源、温度、または個別SFP故障

【📋 ステップ3: 推奨アクション生成】
上記の分析結果に基づいて、優先順位順に**4-5個**の具体的な推奨アクションを生成してください。

**優先度の決定:**
- **high（最優先）**: アラームパターンから推測される真因に対する対応
  - 例: 全ポート影響 → 電源調査をhigh
  - 例: 光信号レベル低下 → 温度・電源調査をhigh
- **medium（推奨）**: 補助的な確認
- **low（最後の手段）**: 個別部品交換（{affected_count}個全交換は非現実的）

【出力形式 - JSON配列のみ】
**以下の形式で4-5個のアクションを出力してください（JSON以外の文字は一切含めない）:**

[
  {{
    "title": "具体的なアクション名",
    "effect": "期待される効果（{affected_count}個への影響を明記）",
    "priority": "high/medium/low",
    "rationale": "アラームパターンから推測した根拠（具体的に）",
    "steps": "1. 実行手順\\n2. CLIコマンド例\\n3. 次のステップ"
  }}
]

**🚨 出力ルール:**
1. JSON配列のみ出力（説明文・マークダウン・コメント不要）
2. 必ず4-5個のアクション
3. high優先度を2個以上
4. rationaleには「アラームメッセージから○○が確認できるため」など具体的根拠
5. 個別部品交換はlow優先度
6. stepsには\\nで改行"""

            # Gemini API 呼び出し
            logger.info(f"Calling Gemini API for {affected_count} affected components")
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # JSON パース
            # Markdown コードブロックを除去
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            actions = json.loads(response_text)
            
            # バリデーション
            if not isinstance(actions, list):
                logger.warning("Gemini API returned invalid format (not a list)")
                return None
            
            # 必須フィールドの確認
            validated_actions = []
            for action in actions:
                if all(k in action for k in ["title", "effect", "priority", "rationale"]):
                    validated_actions.append(action)
            
            if validated_actions:
                logger.info(f"Generated {len(validated_actions)} actions using Gemini API")
                return validated_actions
            else:
                logger.warning("No valid actions in Gemini API response")
                return None
        
        except Exception as e:
            logger.warning(f"Failed to generate actions with Gemini API: {e}")
            return None

    def _generate_smart_recommendations(
        self,
        rule_pattern: str,
        affected_count: int,
        confidence: float,
        messages: List[str],
        device_id: str,
        base_actions: List[Dict[str, str]],
        llm_cache: Optional[Dict[str, List[Dict[str, str]]]] = None
    ) -> List[Dict[str, str]]:
        """
        ★ 推奨アクション生成（LLM真因推論 + ルールベースフォールバック）

        戦略:
        - affected_count >= 3 → 常に gemini-2.0-flash-exp で真因分析
        - LLM 失敗時 → ルールベースの知的フォールバック
        - affected_count < 3  → ルール定義の base_actions を返す
        """
        WIDE_RANGE_THRESHOLD = 3   # 3個以上で「広範囲」と判定

        # ★ キャッシュから取得（バッチ生成済み）
        if llm_cache and rule_pattern in llm_cache:
            return llm_cache[rule_pattern]

        # ★ 広範囲障害（affected >= 3）: LLM に真因推論を依頼
        if affected_count >= WIDE_RANGE_THRESHOLD:
            llm_actions = self._generate_actions_with_gemini(
                rule_pattern=rule_pattern,
                affected_count=affected_count,
                confidence=confidence,
                messages=messages,
                device_id=device_id
            )
            if llm_actions:
                return llm_actions

            # ★ LLM失敗時のフォールバック（パターン別知的推論）
            return self._fallback_wide_range_actions(rule_pattern, affected_count)

        # 単一/少数コンポーネント → ルール定義のbase_actionsを返す
        return base_actions

    # ================================================================
    # ★ シミュレーション用レベル対応スコアリング
    #   LLM API を呼ばずに、劣化レベルに応じた動的な6次元スコア、
    #   narrative、anomaly_type を生成する
    # ================================================================
    def _simulation_level_scoring(
        self,
        rule_pattern: str,
        level: int,
        signal_count: int,
        affected_count: int,
        messages: List[str] = None,
    ) -> Dict[str, Any]:
        """
        シミュレーション時のレベル対応スコアリング。
        レベル 1-5 に応じてレーダーチャート全6軸が動的に変化し、
        narrative（AI解説文）と anomaly_type も段階的に進化する。
        """
        _level = max(1, min(5, level))
        _msgs = messages or []
        _msg_count = len(_msgs)

        # ── ルールパターン別の基本特性 ──
        _RULE_PROFILES = {
            "optical": {
                "name": "光減衰",
                "semantic_base": 0.65, "semantic_growth": 0.08,
                "trend_base": 0.30,    "trend_growth": 0.15,
                "volatility_base": 0.20, "volatility_growth": 0.12,
                "history_base": 0.25,  "history_growth": 0.15,
                "interaction_base": 0.20, "interaction_growth": 0.18,
                "change_base": 0.05,   "change_growth": 0.04,
            },
            "microburst": {
                "name": "マイクロバースト",
                "semantic_base": 0.55, "semantic_growth": 0.09,
                "trend_base": 0.40,    "trend_growth": 0.12,
                "volatility_base": 0.50, "volatility_growth": 0.10,
                "history_base": 0.30,  "history_growth": 0.12,
                "interaction_base": 0.25, "interaction_growth": 0.16,
                "change_base": 0.10,   "change_growth": 0.06,
            },
            "route_instability": {
                "name": "経路不安定",
                "semantic_base": 0.60, "semantic_growth": 0.08,
                "trend_base": 0.35,    "trend_growth": 0.14,
                "volatility_base": 0.40, "volatility_growth": 0.12,
                "history_base": 0.20,  "history_growth": 0.14,
                "interaction_base": 0.30, "interaction_growth": 0.15,
                "change_base": 0.15,   "change_growth": 0.08,
            },
            "memory_leak": {
                "name": "メモリリーク",
                "semantic_base": 0.60, "semantic_growth": 0.09,
                "trend_base": 0.50,    "trend_growth": 0.12,
                "volatility_base": 0.15, "volatility_growth": 0.08,
                "history_base": 0.35,  "history_growth": 0.14,
                "interaction_base": 0.20, "interaction_growth": 0.16,
                "change_base": 0.10,   "change_growth": 0.06,
            },
        }
        # デフォルトプロファイル
        _default_profile = {
            "name": rule_pattern or "不明",
            "semantic_base": 0.50, "semantic_growth": 0.10,
            "trend_base": 0.30,    "trend_growth": 0.12,
            "volatility_base": 0.30, "volatility_growth": 0.10,
            "history_base": 0.25,  "history_growth": 0.12,
            "interaction_base": 0.20, "interaction_growth": 0.15,
            "change_base": 0.05,   "change_growth": 0.05,
        }
        _profile = _RULE_PROFILES.get(rule_pattern, _default_profile)
        _name = _profile["name"]

        # ── 6次元スコア計算: base + growth * (level-1) ──
        def _calc(base_key: str, growth_key: str) -> float:
            base = _profile.get(base_key, 0.30)
            growth = _profile.get(growth_key, 0.10)
            return min(0.99, base + growth * (_level - 1))

        _semantic    = _calc("semantic_base", "semantic_growth")
        _trend       = _calc("trend_base", "trend_growth")
        _volatility  = _calc("volatility_base", "volatility_growth")
        _history     = _calc("history_base", "history_growth")
        _interaction = _calc("interaction_base", "interaction_growth")
        _change      = _calc("change_base", "change_growth")

        # シグナル数による interaction ブースト
        if _msg_count >= 5:
            _interaction = min(0.99, _interaction + 0.10)
        elif _msg_count >= 3:
            _interaction = min(0.99, _interaction + 0.05)

        # affected_count による interaction ブースト
        if affected_count >= 5:
            _interaction = min(0.99, _interaction + 0.05)

        scores = {
            "semantic":      round(_semantic, 2),
            "trend":         round(_trend, 2),
            "volatility":    round(_volatility, 2),
            "history":       round(_history, 2),
            "interaction":   round(_interaction, 2),
            "change_impact": round(_change, 2),
        }

        # ── 異常タイプ: レベルに応じて進化 ──
        if _level <= 1:
            anomaly_type = "point"
        elif _level <= 2:
            anomaly_type = "contextual"
        elif _level <= 3:
            anomaly_type = "collective" if _msg_count >= 3 else "contextual"
        else:  # 4-5
            anomaly_type = "cascading" if _msg_count >= 4 else "collective"

        # ── ナラティブ（AI解説文）: レベル×ルール別 ──
        _NARRATIVES = {
            "optical": {
                1: f"光信号レベルの微小な変動を検知。現時点では許容範囲内ですが、SFPモジュールの経年劣化の初期兆候の可能性があります。",
                2: f"複数の光インターフェースで受信パワーの低下傾向を確認。{_msg_count}件のシグナルが閾値に接近しています。計画的な点検を推奨します。",
                3: f"光信号劣化が加速。{_msg_count}件のインターフェースで受信パワーが警告閾値を下回りました。トランシーバモジュールの劣化が進行中です。48-72時間以内の対応を推奨します。",
                4: f"重大な光減衰を検知。{_msg_count}件のインターフェースが危険水準に達し、リンクダウンのリスクが切迫しています。共通の筐体電源系統またはファイバー経路の障害が疑われます。緊急対応が必要です。",
                5: f"光信号が壊滅的レベルまで劣化。{_msg_count}件のインターフェースで同時にリンクダウン寸前の状態です。筐体レベルの障害（電源・制御基板・光バックプレーン）の可能性が極めて高く、即座の対応が不可欠です。",
            },
            "microburst": {
                1: f"軽微なバッファ使用率の上昇を検知。トラフィックパターンの一時的な変動の可能性があります。",
                2: f"複数ポートでキュードロップが増加傾向。{_msg_count}件のシグナルが短時間に集中しています。QoSポリシーの確認を推奨します。",
                3: f"マイクロバーストの頻度が増大。{_msg_count}件のインターフェースでバッファオーバーフローが発生しています。帯域飽和が進行中です。",
                4: f"深刻なバッファ枯渇を検知。{_msg_count}件のポートで連続的なパケットドロップが発生。アプリケーションへの影響が出始めています。",
                5: f"バッファ枯渇が全ポートに波及。サービス品質が著しく低下しており、回線増強またはトラフィック制御の緊急実施が不可欠です。",
            },
            "route_instability": {
                1: f"BGPセッションで軽微な経路更新の増加を検知。通常の経路収束の範囲内ですが、監視を継続します。",
                2: f"複数のBGPピアで経路フラップを検知。{_msg_count}件のシグナルが短時間に発生しています。ルーティングテーブルの安定性を確認してください。",
                3: f"経路不安定が拡大。{_msg_count}件のピアで再送率が上昇し、経路収束に時間がかかっています。ネットワーク全体の経路品質が劣化中です。",
                4: f"重大な経路障害の兆候。複数のBGPピアでネイバーダウンのリスクが切迫しており、大規模な通信断につながる恐れがあります。",
                5: f"経路制御が崩壊寸前。BGPセッションの大部分で再送が発生し、経路テーブルの整合性が失われています。即座の経路制御介入が不可欠です。",
            },
            "memory_leak": {
                1: f"メモリ使用量の微小な上昇傾向を検知。現時点では許容範囲内ですが、プロセッサプールの空き容量を継続監視してください。",
                2: f"メモリ使用率の上昇が継続。{_msg_count}件のメモリ関連警告が発生しています。特定プロセスによるメモリリークの初期兆候の可能性があります。",
                3: f"メモリ枯渇が進行中。{_msg_count}件のシグナルが警告閾値を超えています。メモリアロケーション失敗（MALLOCFAIL）が発生し始めており、プロセスの異常終了リスクが高まっています。",
                4: f"重大なメモリ枯渇を検知。プロセッサプールの空き容量が危険水準に達し、メモリアロケーション失敗が頻発しています。システムクラッシュのリスクが切迫しており、計画的な再起動を強く推奨します。",
                5: f"メモリが壊滅的に枯渇。プロセッサプールの空き容量がほぼゼロに達し、連続的なMALLOCFAILが発生しています。自然復旧は不可能であり、即座の再起動が不可欠です。",
            },
        }
        _default_narratives = {
            1: f"{_name}に関連する微弱なシグナルを検知しました。",
            2: f"{_name}の兆候が複数確認されています（{_msg_count}件）。",
            3: f"{_name}が進行中です。{_msg_count}件のシグナルが警告レベルに達しています。",
            4: f"{_name}が重大な水準に達しました。早急な対応が必要です。",
            5: f"{_name}が壊滅的レベルです。即座の緊急対応が不可欠です。",
        }
        _narratives_for_rule = _NARRATIVES.get(rule_pattern, _default_narratives)
        narrative = _narratives_for_rule.get(_level, _default_narratives.get(_level, ""))

        return {
            "scores": scores,
            "anomaly_type": anomaly_type,
            "narrative": narrative,
        }

    def _fallback_wide_range_actions(
        self, rule_pattern: str, affected_count: int
    ) -> List[Dict[str, str]]:
        """広範囲障害時のルールベースフォールバック推奨アクション"""
        if "optical" in rule_pattern:
            return [
                {
                    "title": "⚠️ 筐体電源系統の調査（最優先）",
                    "effect": f"電源ユニット故障による{affected_count}個の光モジュール同時劣化を解消",
                    "priority": "high",
                    "rationale": f"{affected_count}個の光モジュール同時劣化は単発SFP故障では説明困難。共通電源系統の問題を疑う。",
                    "steps": "1. show environment power で電源状態を確認\n2. PSU 出力電圧の計測（仕様値との乖離確認）\n3. 電源冗長構成の確認\n4. UPS給電状態の確認",
                },
                {
                    "title": "⚠️ IOS/ファームウェアのバグ調査",
                    "effect": "ソフトウェア起因の光モジュール制御異常を解消",
                    "priority": "high",
                    "rationale": f"広範囲({affected_count}個)の同時劣化はファームウェアバグによる光モジュール制御異常の可能性あり",
                    "steps": "1. show version でファームウェアバージョン確認\n2. ベンダーの既知バグ(Field Notice)を照合\n3. 推奨バージョンへのアップデート計画策定",
                },
                {
                    "title": "制御基板の温度/環境調査",
                    "effect": "筐体内過熱による全モジュール劣化を解消",
                    "priority": "medium",
                    "rationale": "環境温度上昇による光モジュール特性劣化を確認",
                    "steps": "1. show environment temperature で内部温度確認\n2. ファン動作状態の確認（show environment fan）\n3. 設置環境の空調確認",
                },
                {
                    "title": "SFPモジュールの個別交換（最後の手段）",
                    "effect": "個別モジュール故障の解消",
                    "priority": "low",
                    "rationale": f"{affected_count}個全交換は非現実的。上記の共通原因調査を優先",
                    "steps": "1. show interfaces transceiver で個別Rx Power確認\n2. 最も劣化の激しいSFPから交換\n3. 交換後にRx Powerの回復を確認",
                }
            ]
        elif "microburst" in rule_pattern:
            return [
                {
                    "title": "⚠️ ASIC/ハードウェアの調査",
                    "effect": f"{affected_count}個のインターフェースでのバッファ問題を解消",
                    "priority": "high",
                    "rationale": "広範囲のqueue dropsはASIC/チップセット問題の可能性",
                    "steps": "1. show platform hardware capacity\n2. ASIC エラーカウンタの確認\n3. ラインカード診断の実行",
                },
                {
                    "title": "⚠️ IOS/ファームウェアのバグ確認",
                    "effect": "QoS処理の異常を解消",
                    "priority": "high",
                    "rationale": "複数ポートでの同時発生はソフトウェアバグの可能性",
                    "steps": "1. show version でバージョン確認\n2. ベンダーバグDBを照合\n3. QoSプロセスのリスタート検討",
                },
                {
                    "title": "トラフィックパターンの分析",
                    "effect": "異常トラフィックの検出・対処",
                    "priority": "medium",
                    "rationale": "DDoS攻撃や異常トラフィックの可能性を確認",
                    "steps": "1. NetFlow/sFlow でトラフィック発生源を特定\n2. 異常パターンの分析\n3. ACL/Rate-limit による緩和",
                },
            ]
        elif "route" in rule_pattern:
            return [
                {
                    "title": "⚠️ ルーティングプロセスの調査",
                    "effect": "経路不安定の根本原因を解消",
                    "priority": "high",
                    "rationale": "広範囲の経路揺らぎはルーティングプロセスの異常を示唆",
                    "steps": "1. show ip bgp summary で全ネイバー状態確認\n2. show processes cpu でCPU負荷確認\n3. メモリ使用率の確認",
                },
                {
                    "title": "MTU/MSS ミスマッチの確認",
                    "effect": "パケット分割によるBGP不安定を解消",
                    "priority": "medium",
                    "rationale": "MTU不一致はBGPセッションの間欠的断絶の一般的原因",
                    "steps": "1. 全インターフェースのMTU値を確認\n2. ping -s <size> でMTU検証\n3. 必要に応じてMTU調整",
                },
            ]
        # その他のパターン
        return [
            {
                "title": "⚠️ 共通原因の調査（筐体/電源/ソフトウェア）",
                "effect": f"{affected_count}個のコンポーネントへの広範囲影響を解消",
                "priority": "high",
                "rationale": f"{affected_count}個同時の異常は単一コンポーネント故障では説明困難",
                "steps": "1. show environment で筐体全体の状態確認\n2. show version でファームウェア確認\n3. show logging で共通エラーパターンを調査",
            },
            {
                "title": "ベンダーへのエスカレーション",
                "effect": "専門家による根本原因分析",
                "priority": "medium",
                "rationale": "広範囲障害はベンダーサポートによる詳細調査が有効",
                "steps": "1. show tech-support の取得\n2. サポートケースのオープン\n3. ログ・診断情報の送付",
            },
        ]

    def predict(self, analysis_results: List[Dict], msg_map: Dict[str, List[str]], alarms: Optional[List] = None) -> List[Dict]:
        self.reload_all()
        predictions = []
        critical_ids = {r["id"] for r in analysis_results if r.get("status") in ["RED", "CRITICAL"] or r.get("severity") == "CRITICAL" or float(r.get("prob", 0)) >= 0.85}
        warning_ids = {r["id"] for r in analysis_results if 0.45 <= float(r.get("prob", 0)) <= 0.85}
        active_ids = set(msg_map.keys())
        candidates = (warning_ids.union(active_ids)) - critical_ids
        processed_devices = set()
        multi_signal_boost = 0.05
        
        # ★ バッチLLM推奨アクション生成（コスト削減・性能向上）
        llm_recommendations_cache = self._batch_generate_llm_recommendations(
            candidates=candidates,
            msg_map=msg_map
        )

        for dev_id in candidates:
            if dev_id in processed_devices: continue
            messages = msg_map.get(dev_id, [])
            if not messages: continue
            matched_signals = []
            for msg in messages:
                rule, quality = self._match_rule(msg)
                if rule and quality >= 0.30 and rule.pattern != "generic_error":
                    matched_signals.append((rule, quality, msg))
            if not matched_signals:
                rule, quality = self._match_rule(messages[0])
                if not rule: continue
                matched_signals = [(rule, quality, messages[0])]

            matched_signals.sort(key=lambda x: x[1], reverse=True)
            primary_rule, primary_quality, primary_msg = matched_signals[0]

            # Phase 6a: topology.metadata から vendor コンテキストを抽出
            _node = self.topology.get(dev_id, {})
            _meta = (_node.get("metadata", {}) or {}) if isinstance(_node, dict) else {}
            _vc_parts = []
            if _meta.get("vendor"): _vc_parts.append(f"vendor={_meta['vendor']}")
            if _meta.get("os"):     _vc_parts.append(f"os={_meta['os']}")
            if _meta.get("model"):  _vc_parts.append(f"model={_meta['model']}")
            _lc = _meta.get("last_change")
            if _lc:
                _lc_str = (
                    f"{_lc.get('timestamp','')} {_lc.get('description','')}"
                    if isinstance(_lc, dict) else str(_lc)
                ).strip()
                _vc_parts.append(f"last_change=({_lc_str})")
            _vendor_ctx = " | ".join(_vc_parts) if _vc_parts else None

            # Phase 6a: LLM スコアリング（vendor コンテキスト付き）
            _llm_result = self.llm.score_alarm(
                alarm_text     = primary_msg,
                device_id      = dev_id,
                device_type    = _meta.get("type") or (
                    _node.get("type", "network") if isinstance(_node, dict) else "network"
                ),
                signal_count   = len(matched_signals),
                affected_count = len(self.children_map.get(dev_id, [])),
                rule_pattern   = primary_rule.pattern,
                vendor_context = _vendor_ctx,
            )
            _llm_scores = _llm_result.scores

            confidence = self._calculate_confidence(
                primary_rule, dev_id, primary_quality, llm_scores=_llm_scores
            )
            extra_signals = len(matched_signals) - 1
            if extra_signals > 0:
                boost = min(extra_signals * multi_signal_boost, 0.20)
                confidence = min(0.99, confidence + boost)
            
            # ★ ベイズ推論による信頼度の更新
            confidence, bayesian_debug = self.bayesian.calculate_posterior_confidence(
                device_id=dev_id,
                rule_pattern=primary_rule.pattern,
                current_confidence=confidence,
                time_window_hours=168  # 過去7日間
            )
            
            # ★ GNN予測による信頼度の補正（オプション）
            if self.gnn and self._model:
                try:
                    # 現在のアラームメッセージをBERT埋め込みに変換
                    alarm_embeddings = {}
                    for msg_dev_id, msg_list in msg_map.items():
                        if msg_list:
                            # 複数メッセージの平均埋め込み
                            embeddings = self._model.encode(msg_list, convert_to_numpy=True)
                            alarm_embeddings[msg_dev_id] = embeddings.mean(axis=0)
                    
                    # GNNで予測
                    gnn_confidence, gnn_ttf = self.gnn.predict_with_gnn(
                        alarm_embeddings, dev_id
                    )
                    
                    # ベイズ推論とGNN予測の加重平均（GNNの重みは控えめ）
                    confidence = 0.7 * confidence + 0.3 * gnn_confidence
                    confidence = min(0.99, max(0.1, confidence))
                    
                except Exception as e:
                    logger.warning(f"GNN prediction failed: {e}")

            threshold = MIN_PREDICTION_CONFIDENCE
            if primary_rule.paging_threshold is not None:
                threshold = primary_rule.paging_threshold
            if confidence < threshold: continue

            impact_count = 0
            if dev_id in self.children_map:
                impact_count = len(self.children_map[dev_id])
            
            # --- 予測結果に「運用者向けの具体的な知識」を注入 ---
            
            # ★ LLMベースの動的推奨アクション生成（広範囲障害に対応）
            # affected_count: メッセージから抽出されるコンポーネント数を計算
            unique_components = set()
            for _, _, msg in matched_signals:
                # メッセージからコンポーネント名を抽出（例: Gi0/0/1, Te1/0/1）
                import re
                components = re.findall(r'\b(?:Gi|Te|Fa|Et)\d+/\d+/\d+|\b(?:Gi|Te|Fa|Et)\d+/\d+', msg)
                unique_components.update(components)
            
            # コンポーネント数がカウントできない場合はシグナル数を使用
            component_count = len(unique_components) if unique_components else len(matched_signals)
            
            smart_actions = self._generate_smart_recommendations(
                rule_pattern=primary_rule.pattern,
                affected_count=component_count,  # ★ 修正: コンポーネント数を使用
                confidence=confidence,
                messages=[s[2] for s in matched_signals],  # ★ 全メッセージを送信（[:3]を削除）
                device_id=dev_id,
                base_actions=primary_rule.recommended_actions,
                llm_cache=llm_recommendations_cache  # ★ バッチ生成されたキャッシュを使用
            )
            
            pred = {
                "id": dev_id,
                "label": f"🔮 [予兆] {primary_rule.escalated_state}",
                "severity": "CRITICAL",
                "status": "CRITICAL",
                "prob": round(confidence, 2),
                "confidence": round(confidence, 2),
                "type": f"Predictive/{primary_rule.category}",
                "tier": 1,
                "rule_pattern": primary_rule.pattern,
                "alarm_text": primary_msg,
                "reason": f"Digital Twin Prediction: {primary_rule.time_to_critical_min}min to critical. Root: {primary_msg}",
                "is_prediction": True,
                "prediction_timeline": f"{primary_rule.time_to_critical_min}分後",
                "prediction_time_to_critical_min": primary_rule.time_to_critical_min,
                "prediction_early_warning_hours": primary_rule.early_warning_hours,
                "prediction_affected_count": impact_count,
                "prediction_signal_count": len(matched_signals),
                "prediction_confidence_factors": {"base": primary_rule.base_confidence, "match_quality": primary_quality},
                "recommended_actions": smart_actions,
                "base_recommended_actions": primary_rule.recommended_actions,
                "runbook_url": primary_rule.runbook_url,
                # Phase 6a: LLM 生成情報
                "llm_narrative": _llm_scores.narrative,
                "llm_anomaly_type": _llm_result.anomaly_type_hint,
                "vendor_context": _vendor_ctx,
                "explanation": {
                    "narrative":       _llm_scores.narrative,
                    "anomaly_type":    _llm_result.anomaly_type_hint,
                    "llm_error":       _llm_result.error or None,
                    "score_breakdown": {
                        "semantic":      _llm_scores.semantic,
                        "trend":         _llm_scores.trend,
                        "volatility":    _llm_scores.volatility,
                        "history":       _llm_scores.history,
                        "interaction":   _llm_scores.interaction,
                        "change_impact": _llm_scores.change_impact,
                    },
                    "vendor_context": _vendor_ctx,
                },
            }
            pid = str(uuid.uuid4())
            self.history.append({"prediction_id": pid, "device_id": dev_id, "rule_pattern": primary_rule.pattern, "timestamp": time.time(), "prob": confidence, "anchor_event_time": time.time(), "raw_msg": primary_msg})
            self.storage.save_json_atomic("history", self.history)

            # Phase 6c*: ベクトルストアに予兆予測を登録
            if self.vector_store and self.vector_store.is_ready:
                try:
                    self.vector_store.add_incident(
                        alarm_text      = primary_msg,
                        device_id       = dev_id,
                        rule_pattern    = primary_rule.pattern,
                        confidence      = confidence,
                        vendor_context  = _vendor_ctx or "",
                        anomaly_type    = _llm_result.anomaly_type_hint,
                        score_breakdown = pred.get("explanation", {}).get("score_breakdown"),
                        outcome         = "pending",
                        incident_id     = pid,
                    )
                except Exception as _vs_err:
                    logger.debug(f"vector_store.add_incident skipped: {_vs_err}")

            predictions.append(pred)
            processed_devices.add(dev_id)
        return predictions

    def generate_tuning_report(self, days: int = 30) -> Dict[str, Any]:
        return self.tuner.generate_report(days)

    def apply_tuning_proposals_if_auto(self, proposals: List[Dict]) -> Dict:
        applied = []
        skipped = []
        with self.storage.global_lock(timeout_sec=30.0):
            for p in proposals:
                rp = p.get("rule_pattern")
                rec = p.get("apply_recommendation", {})
                if rec.get("apply_mode") != "auto":
                    skipped.append({"rule": rp, "reason": "not_auto"})
                    continue
                prop = p.get("proposal", {})
                pt = float(prop.get("paging_threshold", 0.0))
                lt = float(prop.get("logging_threshold", 0.0))
                old_json_str = self.storage.rule_config_get_json_str(rp)
                rj_str = old_json_str
                if rj_str:
                    d = json.loads(rj_str)
                    d["paging_threshold"] = pt
                    d["logging_threshold"] = lt
                    rj_str = json.dumps(d, ensure_ascii=False)
                success = self.storage.rule_config_upsert(rp, pt, lt, rj_str)
                if success:
                    applied.append({"rule": rp, "paging": pt})
                else:
                    skipped.append({"rule": rp, "reason": "db_write_fail"})
        return {"applied": applied, "skipped": skipped}

    def repair_db_from_rules_json(self) -> bool:
        try:
            path = self.storage.paths["rules"]
            if not os.path.exists(path): return False
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            sanitized = [self._sanitize_rule_data(item) for item in data]
            self.storage._seed_rule_config_from_rules_json(sanitized)
            return True
        except Exception: return False
    # ==============================================================
    # Phase1: Predict API helpers
    # ==============================================================

    def _parse_timestamp(self, ts) -> float:
        if ts is None:
            return time.time()
        if isinstance(ts, (int, float)):
            return float(ts)
        s = str(ts).strip()
        try:
            return float(s)
        except Exception:
            pass
        try:
            from datetime import datetime as _dt
            return _dt.fromisoformat(s.replace("Z", "+00:00")).timestamp()
        except Exception:
            return time.time()

    def _should_ignore(self, msg: str) -> bool:
        m = (msg or "").lower()
        ignore = ["dry-run", "test message", "synthetic-monitor", "healthcheck"]
        return any(ph in m for ph in ignore)

    def _rule_match_simple(self, rule, msg: str):
        """regex + semantic phrase マッチ。(hit, reasons) を返す"""
        reasons = []
        hit = False
        try:
            if rule._compiled_regex and rule._compiled_regex.search(msg or ""):
                hit = True
                reasons.append(f"pattern matched: {rule.pattern}")
        except Exception:
            pass
        if not hit:
            low = (msg or "").lower()
            for sp in (rule.semantic_phrases or []):
                if sp and sp.lower() in low:
                    hit = True
                    reasons.append(f"semantic hit: {sp}")
                    break
        return hit, reasons

    # ★ Phase 6c* 統合: NumPy 時系列トレンドによる動的 RUL 予測
    def _predict_rul_with_trend(self, device_id: str, current_level: int,
                                base_ttf_hours: int, source: str = "real") -> int:
        """
        残存寿命 (RUL) を予測する。

        ■ simulation モード: 決定論的な指数減衰モデル
          - Level が上がるにつれ RUL が滑らかに短縮
          - スライダー操作で履歴が汚染されない
        ■ real モード: 過去の *real* 履歴からトレンドを線形回帰
          - データ不足時は simulation と同じ決定論的モデルにフォールバック
        """
        # ─── 決定論的 RUL（simulation 用、real のフォールバックも兼用） ───
        #   指数減衰: base_hours * decay_factor
        #   Level 1: 100% → Level 5: ~1.5%
        _DETERMINISTIC_DECAY = {
            1: 1.00,   # 14日  (optical 336h)
            2: 0.50,   # 7日
            3: 0.21,   # 2.9日 (≈70h)
            4: 0.07,   # 1日   (≈24h)
            5: 0.015,  # 5時間
        }

        _level = max(1, min(5, current_level))

        if source == "simulation":
            _decay = _DETERMINISTIC_DECAY.get(_level, 0.50)
            return max(1, int(base_ttf_hours * _decay))

        # ─── real モード: トレンド分析 ─────────────────────
        now = time.time()
        window_start = now - (24 * 3600)  # 過去24時間

        # ★ simulation 由来の履歴を除外して real のみ使用
        history_data = [
            h for h in self.history
            if (h.get("device_id") == device_id
                and h.get("timestamp", 0) >= window_start
                and h.get("source", "real") != "simulation")
        ]

        # データ不足時は決定論的モデルにフォールバック
        if len(history_data) < 3:
            _decay = _DETERMINISTIC_DECAY.get(_level, 0.50)
            return max(1, int(base_ttf_hours * _decay))

        try:
            timestamps = np.array([h.get("timestamp") for h in history_data])
            x = (timestamps - window_start) / 3600.0  # 時間単位
            y = np.array([h.get("prob", 0.1) for h in history_data])

            # 現在の状態を追加
            x = np.append(x, 24.0)
            current_prob = min(0.99, _level * 0.2)
            y = np.append(y, current_prob)

            coeffs = np.polyfit(x, y, 1)
            slope = coeffs[0]
            intercept = coeffs[1]

            if slope > 0.001:
                target_x = (0.95 - intercept) / slope
                predicted_rul_hours = target_x - 24.0
                # sanity check: RUL は決定論的モデルの 0.3倍〜3倍の範囲に制約
                _det_rul = max(1, int(base_ttf_hours * _DETERMINISTIC_DECAY.get(_level, 0.50)))
                _min_rul = max(1, int(_det_rul * 0.3))
                _max_rul = min(base_ttf_hours, int(_det_rul * 3.0))
                return max(_min_rul, min(int(predicted_rul_hours), _max_rul))
            else:
                _decay = _DETERMINISTIC_DECAY.get(_level, 0.50)
                return max(1, int(base_ttf_hours * _decay))

        except Exception as e:
            logger.warning(f"RUL trend prediction failed for {device_id}: {e}")
            _decay = _DETERMINISTIC_DECAY.get(_level, 0.50)
            return max(1, int(base_ttf_hours * _decay))

    def predict(self, device_id: str, msg: str, timestamp: float,
                attrs: Optional[Dict[str, Any]] = None,
                degradation_level: int = 1,
                source: str = "real") -> List[Dict[str, Any]]:
        """
        EscalationRule ベースの予兆予測。
        degradation_level (1-5): Level に応じて confidence をブースト、
                                  time_to_critical を短縮、early_warning を延長。
        source: "simulation" | "real"
        戻り値は PredictResult.to_dict() のリスト（confidence 降順）。
        """
        # ★ 高速化: 同一条件の予測結果をキャッシュ（Streamlit再描画対策）
        _cache_key = f"{device_id}|{degradation_level}|{hash(msg[:200])}"
        _now = time.time()
        _cached = self._predict_cache.get(_cache_key)
        if _cached and (_now - _cached["ts"]) < self._predict_cache_ttl:
            logger.debug(f"predict cache HIT: {device_id} level={degradation_level}")
            return _cached["results"]

        try:
            msg_n = self._normalize_msg(msg or "")
        except AttributeError:
            msg_n = (msg or "").strip()
        except Exception:
            msg_n = (msg or "").strip()

        if self._should_ignore(msg_n):
            return []

        _min_conf = float(MIN_PREDICTION_CONFIDENCE)

        # ★ 再設計: Level-based Confidence Model
        #   Level が上がるにつれ、確信度が明確に上昇する
        #   base_confidence (ルール固有) はレンジ内の位置を決定
        _level = max(1, min(5, int(degradation_level or 1)))
        _LEVEL_CONF_RANGE = {
            # (lower_bound, upper_bound) — base_confidence が位置を決定
            1: (0.52, 0.65),  # 52-65%: 微弱な兆候
            2: (0.66, 0.76),  # 66-76%: 注意段階
            3: (0.77, 0.86),  # 77-86%: 警戒段階
            4: (0.87, 0.93),  # 87-93%: 危険段階
            5: (0.94, 0.98),  # 94-98%: 緊急段階
        }

        # 急性期進行パラメータ（既存互換）
        _ttc_factor    = 1.0 - (_level - 1) * 0.12   # ×1.0〜×0.52（短縮）
        _early_factor  = 1.0 + (_level - 1) * 0.20   # ×1.0〜×1.80（延長）

        # 影響範囲: children_map から再帰的に配下デバイス数を算出
        def _count_children(dev_id: str, visited=None) -> int:
            if visited is None: visited = set()
            if dev_id in visited: return 0
            visited.add(dev_id)
            children = (self.children_map or {}).get(dev_id, [])
            return len(children) + sum(_count_children(c, visited) for c in children)

        _affected_count = _count_children(device_id)

        results = []
        for rule in (self.rules or []):
            try:
                hit, reasons = self._rule_match_simple(rule, msg_n)
                if not hit:
                    continue

                # ★ 新 Confidence モデル: レベルに応じたレンジ内で base_confidence が位置を決定
                _base_conf = float(getattr(rule, "base_confidence", 0.5) or 0.5)
                _low, _high = _LEVEL_CONF_RANGE.get(_level, (0.50, 0.60))
                conf = _low + (_high - _low) * min(1.0, _base_conf)
                if conf < _min_conf:
                    continue

                _base_ttc   = int(getattr(rule, "time_to_critical_min", 60) or 60)
                _base_early = int(getattr(rule, "early_warning_hours", 24) or 24)
                _ttc   = max(5,  int(_base_ttc   * _ttc_factor))
                _early = max(1,  int(_base_early * _early_factor))
                
                # ★ RUL計算: source を渡して simulation/real を区別
                _base_ttf_hours = int(getattr(rule, "early_warning_hours", 336) or 336)
                _ttf_hours = self._predict_rul_with_trend(
                    device_id=device_id,
                    current_level=_level,
                    base_ttf_hours=_base_ttf_hours,
                    source=source,
                )
                
                # 故障予測日時を算出
                _failure_dt = datetime.now() + timedelta(hours=_ttf_hours)
                _failure_dt_str = _failure_dt.strftime("%Y-%m-%d %H:%M")
                
                pr = PredictResult(
                    predicted_state      = str(getattr(rule, "escalated_state", "unknown")),
                    confidence           = conf,
                    rule_pattern         = str(getattr(rule, "pattern", "unknown")),
                    category             = str(getattr(rule, "category", "Generic")),
                    reasons              = reasons,
                    recommended_actions  = list(getattr(rule, "recommended_actions", []) or []),
                    runbook_url          = str(getattr(rule, "runbook_url", "") or ""),
                    criticality          = str(getattr(rule, "criticality", "standard") or "standard"),
                    time_to_critical_min = _ttc,
                    early_warning_hours  = _early,
                    time_to_failure_hours = _ttf_hours,
                    predicted_failure_datetime = _failure_dt_str,
                )
                # ★ シミュレーション時: リッチな検知シグナル詳細を生成
                if source == "simulation" and pr.reasons:
                    import re as _re_sig
                    _interfaces = _re_sig.findall(r'\b(?:Gi|Te|Fa|Et)\d+/\d+(?:/\d+)?', msg_n)
                    _dbm_vals = _re_sig.findall(r'([-\d.]+)\s*dBm', msg_n)
                    _drops = _re_sig.findall(r'drops[:\s]+(\d+)', msg_n)
                    _msgs_lines = [m.strip() for m in msg_n.split("\n") if m.strip()]
                    _sig_reasons = [f"pattern matched: {pr.rule_pattern}"]
                    if _interfaces:
                        _sig_reasons.append(f"影響インターフェース: {', '.join(set(_interfaces[:6]))} ({len(set(_interfaces))}個)")
                    if _dbm_vals:
                        _sig_reasons.append(f"光受信パワー: {', '.join(_dbm_vals[:3])} dBm")
                    if _drops:
                        _sig_reasons.append(f"キュードロップ数: {', '.join(_drops[:3])}")
                    for sp in (getattr(rule, "semantic_phrases", []) or []):
                        if sp and sp.lower() in msg_n.lower():
                            _sig_reasons.append(f"semantic hit: {sp}")
                    _sig_reasons.append(f"劣化レベル: {_level}/5 ({['微弱','注意','警戒','危険','緊急'][_level-1]})")
                    _sig_reasons.append(f"検知シグナル数: {len(_msgs_lines)}件")
                    pr.reasons = _sig_reasons
                results.append(pr)
            except Exception:
                continue
        results.sort(key=lambda x: x.confidence, reverse=True)

        # Phase 6a: LLM スコアリング + vendor コンテキスト抽出（最上位ルールに対して実行）
        _llm_narrative    = ""
        _llm_anomaly_type = "point"
        _llm_error        = ""
        _score_breakdown  = {}
        _vendor_ctx       = None

        _node = self.topology.get(device_id, {})
        _meta = (_node.get("metadata", {}) or {}) if isinstance(_node, dict) else {}
        _vc_parts = []
        if _meta.get("vendor"): _vc_parts.append(f"vendor={_meta['vendor']}")
        if _meta.get("os"):     _vc_parts.append(f"os={_meta['os']}")
        if _meta.get("model"):  _vc_parts.append(f"model={_meta['model']}")
        _lc = _meta.get("last_change")
        if _lc:
            _lc_str = (
                f"{_lc.get('timestamp','')} {_lc.get('description','')}"
                if isinstance(_lc, dict) else str(_lc)
            ).strip()
            _vc_parts.append(f"last_change=({_lc_str})")
        _vendor_ctx = " | ".join(_vc_parts) if _vc_parts else None

        # ★ 高速化: simulation 時は LLM API をスキップ → レベル対応スコアリング
        #   LLM の真価は「レポート生成」「Generate Fix」で発揮
        _use_fast_scoring = (source == "simulation")

        if results:
            _top = results[0]
            try:
                if _use_fast_scoring:
                    # ★ シミュレーション高速パス: レベルに応じた動的スコア生成
                    _sim_scores = self._simulation_level_scoring(
                        rule_pattern   = _top.rule_pattern,
                        level          = _level,
                        signal_count   = len(results),
                        affected_count = _affected_count,
                        messages       = [m.strip() for m in msg_n.split("\n") if m.strip()],
                    )
                    _llm_narrative    = _sim_scores["narrative"]
                    _llm_anomaly_type = _sim_scores["anomaly_type"]
                    _llm_error        = ""  # シミュレーションでは「LLM不可」を表示しない
                    _score_breakdown  = _sim_scores["scores"]
                    logger.debug(f"Simulation scoring (level={_level}): {device_id}")
                else:
                    # ★ 通常パス: LLM API でスコアリング
                    _llm_result = self.llm.score_alarm(
                        alarm_text     = msg_n,
                        device_id      = device_id,
                        device_type    = _meta.get("type") or (
                            _node.get("type", "network") if isinstance(_node, dict) else "network"
                        ),
                        signal_count   = len(results),
                        affected_count = _affected_count,
                        rule_pattern   = _top.rule_pattern,
                        vendor_context = _vendor_ctx,
                    )
                    _llm_narrative    = _llm_result.scores.narrative
                    _llm_anomaly_type = _llm_result.anomaly_type_hint
                    _llm_error        = _llm_result.error
                    _score_breakdown  = {
                        "semantic":      _llm_result.scores.semantic,
                        "trend":         _llm_result.scores.trend,
                        "volatility":    _llm_result.scores.volatility,
                        "history":       _llm_result.scores.history,
                        "interaction":   _llm_result.scores.interaction,
                        "change_impact": _llm_result.scores.change_impact,
                    }
                    # ★ LLM スコアで confidence を ±微調整
                    _lf = (
                        _llm_result.scores.semantic      * 0.25 +
                        _llm_result.scores.interaction   * 0.25 +
                        _llm_result.scores.trend         * 0.15 +
                        _llm_result.scores.volatility    * 0.10 +
                        _llm_result.scores.change_impact * 0.25
                    )
                    _lf_centered = _lf - 0.50
                    _adjustment = _lf_centered * 0.10
                    _top.confidence = min(0.99, max(0.30, _top.confidence + _adjustment))
            except Exception as _le:
                logger.debug(f"LLM score skipped: {_le}")

        # Phase 6c*: ベクトルストアに予兆予測を登録（最上位ルールのみ）
        # ★ 高速化: simulation 時はベクトルストア書き込みをスキップ
        if results and self.vector_store and self.vector_store.is_ready and not _use_fast_scoring:
            try:
                _top = results[0]
                self.vector_store.add_incident(
                    alarm_text      = msg_n,
                    device_id       = device_id,
                    rule_pattern    = _top.rule_pattern,
                    confidence      = _top.confidence,
                    vendor_context  = _vendor_ctx or "",
                    anomaly_type    = _llm_anomaly_type,
                    score_breakdown = _score_breakdown,
                    outcome         = "pending",
                )
            except Exception as _vs_err:
                logger.debug(f"vector_store.add_incident skipped: {_vs_err}")

        # ★ LLM ベースの推奨アクション生成
        #   simulation 時: ルールベースフォールバック（高速）
        #   real 時: gemini-2.0-flash-exp で真因推論
        if results:
            _top = results[0]
            try:
                import re as _re
                _components = set()
                for _part in msg_n.split("\n"):
                    _components.update(
                        _re.findall(r'\b(?:Gi|Te|Fa|Et)\d+/\d+(?:/\d+)?', _part)
                    )
                _comp_count = max(len(_components), 1)
                
                if _use_fast_scoring and _comp_count >= 3:
                    # ★ 高速パス: LLM API をスキップし、ルールベースの推奨を直接返す
                    _smart = self._fallback_wide_range_actions(
                        _top.rule_pattern, _comp_count
                    )
                else:
                    _smart = self._generate_smart_recommendations(
                        rule_pattern   = _top.rule_pattern,
                        affected_count = _comp_count,
                        confidence     = _top.confidence,
                        messages       = [m.strip() for m in msg_n.split("\n") if m.strip()],
                        device_id      = device_id,
                        base_actions   = _top.recommended_actions,
                    )
                if _smart:
                    _top.recommended_actions = _smart
            except Exception as _ra_err:
                logger.debug(f"Smart recommendations skipped: {_ra_err}")

        # ★ 予兆履歴を JSON に記録（AutoTuner が参照するため）
        # simulation 時: source タグ付きでメモリのみ（trend 分析で除外可能に）
        if results:
            _top = results[0]
            import uuid
            _pid = str(uuid.uuid4())[:12]
            self.history.append({
                "prediction_id": _pid,
                "device_id":     device_id,
                "rule_pattern":  _top.rule_pattern,
                "timestamp":     time.time(),
                "prob":          _top.confidence,
                "anchor_event_time": time.time(),
                "raw_msg":       msg_n[:200],
                "source":        source,  # ★ simulation/real を記録
            })
            # 履歴が 500 件超えたら古いものを削除
            if len(self.history) > 500:
                self.history = self.history[-300:]
            # ★ simulation 時はディスク書き込みをスキップ
            if not _use_fast_scoring:
                self.storage.save_json_atomic("history", self.history)

        _final_results = [
            r.to_dict(
                affected_count    = _affected_count,
                source            = source,
                llm_narrative     = _llm_narrative     if i == 0 else "",
                llm_anomaly_type  = _llm_anomaly_type  if i == 0 else "point",
                llm_error         = _llm_error          if i == 0 else "",
                score_breakdown   = _score_breakdown    if i == 0 else {},
                vendor_context    = _vendor_ctx,
                alarm_text        = msg_n,
            )
            for i, r in enumerate(results)
        ]

        # ★ 高速化: 結果をキャッシュに保存
        self._predict_cache[_cache_key] = {"ts": time.time(), "results": _final_results}
        # キャッシュサイズ制限（最大50件）
        if len(self._predict_cache) > 50:
            _oldest = sorted(self._predict_cache.items(), key=lambda x: x[1]["ts"])[:20]
            for k, _ in _oldest:
                self._predict_cache.pop(k, None)

        return _final_results

    def predict_api(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cockpit / Simulator 共通エントリーポイント。
        record_forecast=True (デフォルト) のとき forecast_ledger に登録する。
        """
        try:
            tenant_id = (request.get("tenant_id") or self.tenant_id or "default").strip().lower()
            device_id = str(request.get("device_id") or "").strip()
            msg       = str(request.get("msg") or "").strip()
            ts        = self._parse_timestamp(request.get("timestamp"))
            if not device_id:
                raise ValueError("device_id is required")
            if not msg:
                raise ValueError("msg is required")
            attrs = request.get("attrs") or {}
            if not isinstance(attrs, dict):
                attrs = {"raw_attrs": str(attrs)}

            req   = PredictRequest(tenant_id=tenant_id, device_id=device_id,
                                   msg=msg, timestamp=ts, attrs=attrs)
            _level  = int((attrs or {}).get("degradation_level", 1))
            _source = str((attrs or {}).get("source", "real"))
            preds = self.predict(device_id=device_id, msg=msg, timestamp=ts,
                                 attrs=attrs, degradation_level=_level, source=_source)

            record_forecast = bool(request.get("record_forecast", True))
            forecast_ids: List[str] = []
            if record_forecast and preds:
                fid = self._forecast_record(
                    req=req.to_dict(),
                    top_prediction=preds[0],
                    source=_source,  # ★ simulation/real を正しく記録
                )
                if fid:
                    forecast_ids.append(fid)

            return {"ok": True, "input": req.to_dict(),
                    "predictions": preds, "forecast_ids": forecast_ids}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}",
                    "trace": traceback.format_exc()}

    # ==============================================================
    # Phase1: Forecast Ledger DDL（_init_sqlite から呼ばれる）
    # ==============================================================

    def _init_forecast_ledger(self):
        """forecast_ledger テーブルと migration を実施"""
        if not self.storage._conn:
            return
        try:
            with self.storage._db_lock:
                self.storage._conn.execute("""
                    CREATE TABLE IF NOT EXISTS forecast_ledger (
                        forecast_id      TEXT PRIMARY KEY,
                        created_at       REAL,
                        tenant_id        TEXT,
                        device_id        TEXT,
                        rule_pattern     TEXT,
                        predicted_state  TEXT,
                        confidence       REAL,
                        horizon_sec      INTEGER,
                        eval_deadline_ts REAL,
                        source           TEXT,
                        status           TEXT,
                        outcome_type     TEXT,
                        outcome_ts       REAL,
                        outcome_note     TEXT,
                        input_json       TEXT,
                        prediction_json  TEXT
                    )
                """)
                self.storage._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_fl_open "
                    "ON forecast_ledger (status, eval_deadline_ts)")
                self.storage._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_fl_device "
                    "ON forecast_ledger (device_id, created_at)")
                # migration: add source column if missing
                cur = self.storage._conn.cursor()
                cur.execute("PRAGMA table_info(forecast_ledger)")
                cols = [r[1] for r in cur.fetchall()]
                if "source" not in cols:
                    self.storage._conn.execute(
                        "ALTER TABLE forecast_ledger ADD COLUMN source TEXT")
                self.storage._conn.commit()
        except Exception as e:
            logger.warning(f"_init_forecast_ledger: {e}")

    def _forecast_horizon_sec(self, rule_pattern: str) -> int:
        for r in (self.rules or []):
            if (getattr(r, "pattern", "") or "").lower() == (rule_pattern or "").lower():
                ttc = getattr(r, "time_to_critical_min", None)
                if isinstance(ttc, int) and ttc > 0:
                    return max(1800, ttc * 60)
        return 3600

    def _forecast_record(self, req: Dict[str, Any], top_prediction: Dict[str, Any],
                         source: str = "real") -> Optional[str]:
        """forecast_ledger に1行 INSERT（原子的）。forecast_id を返す。"""
        if not self.storage._conn:
            return None
        try:
            forecast_id     = "f_" + uuid.uuid4().hex[:12]
            created_at      = time.time()
            tenant_id       = str(req.get("tenant_id") or self.tenant_id)
            device_id       = str(req.get("device_id") or "")
            rule_pattern    = str(top_prediction.get("rule_pattern") or "unknown")
            predicted_state = str(top_prediction.get("predicted_state") or "unknown")
            confidence      = float(top_prediction.get("confidence") or 0.0)
            horizon_sec     = self._forecast_horizon_sec(rule_pattern)
            event_ts        = float(req.get("timestamp") or created_at)
            eval_deadline_ts = event_ts + horizon_sec
            input_json      = json.dumps(req, ensure_ascii=False)
            prediction_json = json.dumps(top_prediction, ensure_ascii=False)

            with self.storage._db_lock:
                # ★ simulation の場合: 同一デバイス＋ルールの古い open エントリを削除
                #   → スライダー操作で蓄積されるのを防止
                #   → 常に最新のレベルの予測のみが残る
                if source == "simulation":
                    self.storage._conn.execute("""
                        DELETE FROM forecast_ledger
                        WHERE device_id=? AND rule_pattern=? AND source='simulation' AND status='open'
                    """, (device_id, rule_pattern))

                self.storage._conn.execute("""
                    INSERT INTO forecast_ledger
                    (forecast_id, created_at, tenant_id, device_id, rule_pattern, predicted_state,
                     confidence, horizon_sec, eval_deadline_ts, source, status,
                     outcome_type, outcome_ts, outcome_note, input_json, prediction_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (forecast_id, created_at, tenant_id, device_id, rule_pattern, predicted_state,
                      confidence, horizon_sec, eval_deadline_ts, source, "open",
                      None, None, None, input_json, prediction_json))
                self.storage._conn.commit()
            return forecast_id
        except Exception as e:
            logger.warning(f"_forecast_record: {e}")
            return None

    def forecast_get(self, forecast_id: str) -> Optional[Dict[str, Any]]:
        if not self.storage._conn:
            return None
        try:
            with self.storage._db_lock:
                cur = self.storage._conn.cursor()
                cur.execute("""
                    SELECT forecast_id, created_at, tenant_id, device_id, rule_pattern,
                           predicted_state, confidence, horizon_sec, eval_deadline_ts,
                           source, status, outcome_type, outcome_ts, outcome_note
                    FROM forecast_ledger WHERE forecast_id=?""", (forecast_id,))
                row = cur.fetchone()
            if not row:
                return None
            keys = ["forecast_id","created_at","tenant_id","device_id","rule_pattern",
                    "predicted_state","confidence","horizon_sec","eval_deadline_ts",
                    "source","status","outcome_type","outcome_ts","outcome_note"]
            return dict(zip(keys, row))
        except Exception:
            return None

    def forecast_register_outcome(self, forecast_id: str, outcome_type: str,
                                  outcome_ts=None, note: str = "",
                                  auto: bool = False) -> Dict[str, Any]:
        """
        予見成功判定:
          deadline 以内に OUTCOME_CONFIRMED → status=confirmed, success=True
          deadline 超過後   → status=confirmed_late, success=False
          自動登録 (auto=True) は audit_log に actor="auto" で記録
        """
        if not self.storage._conn:
            return {"ok": False, "reason": "sqlite_disabled"}
        fid = str(forecast_id or "").strip()
        if not fid:
            return {"ok": False, "reason": "missing_forecast_id"}

        ts  = time.time() if outcome_ts is None else self._parse_timestamp(outcome_ts)
        rec = self.forecast_get(fid)
        if not rec:
            return {"ok": False, "reason": "not_found"}
        if rec.get("status") not in ["open"]:
            return {"ok": False, "reason": "not_open", "status": rec.get("status")}

        deadline = float(rec.get("eval_deadline_ts") or 0.0)
        success  = bool(ts <= deadline) if deadline > 0 else False

        if outcome_type == "confirmed_incident":
            new_status = "confirmed" if success else "confirmed_late"
        elif outcome_type == "mitigated":
            new_status = "mitigated"
            success = True
        elif outcome_type == "false_alarm":
            new_status = "false_alarm"
            success = False
        else:
            new_status = "closed"
            success = False

        actor     = "auto" if auto else "operator"
        note_s    = (note or "")[:512]
        rule_pat  = str(rec.get("rule_pattern") or "")

        try:
            with self.storage._db_lock:
                self.storage._conn.execute("""
                    UPDATE forecast_ledger
                    SET status=?, outcome_type=?, outcome_ts=?, outcome_note=?
                    WHERE forecast_id=?""",
                    (new_status, outcome_type, ts, note_s, fid))
                # audit_log に記録
                self.storage.audit_log_generic({
                    "event_id":    str(uuid.uuid4()),
                    "timestamp":   ts,
                    "event_type":  "forecast_outcome",
                    "actor":       actor,
                    "rule_pattern": rule_pat,
                    "details": {"forecast_id": fid, "outcome_type": outcome_type,
                                "success": success, "status": new_status, "auto": auto}
                })
                self.storage._conn.commit()

                # Phase 6c* 統合: ChromaDB にラベル付きインシデントを登録
                if hasattr(self, 'vector_store') and self.vector_store and self.vector_store.is_ready:
                    input_json_str = rec.get("input_json")
                    if input_json_str:
                        try:
                            input_data = json.loads(input_json_str)
                            msg_text = input_data.get("msg", "")
                            pred_json_str = rec.get("prediction_json")
                            v_ctx = ""
                            if pred_json_str:
                                pred_data = json.loads(pred_json_str)
                                v_ctx = pred_data.get("vendor_context", "")
                            if msg_text:
                                self.vector_store.add_incident(
                                    alarm_text     = msg_text,
                                    device_id      = str(rec.get("device_id", "")),
                                    rule_pattern   = rule_pat,
                                    confidence     = float(rec.get("confidence", 0)),
                                    vendor_context = v_ctx,
                                    outcome        = outcome_type,
                                    created_at     = float(rec.get("created_at", time.time())),
                                )
                        except Exception as ve:
                            logger.debug(f"vector_store write in forecast_register_outcome: {ve}")

        except Exception as e:
            return {"ok": False, "reason": str(e)}

        return {"ok": True, "forecast_id": fid, "success": success, "status": new_status}

    def forecast_expire_open(self, now_ts: Optional[float] = None,
                             limit: int = 200) -> Dict[str, Any]:
        """期限切れの open 予兆を expired に更新"""
        if not self.storage._conn:
            return {"ok": False}
        now = float(now_ts or time.time())
        expired = 0
        try:
            with self.storage._db_lock:
                cur = self.storage._conn.cursor()
                cur.execute("""
                    SELECT forecast_id, rule_pattern FROM forecast_ledger
                    WHERE status='open' AND eval_deadline_ts < ?
                    ORDER BY eval_deadline_ts ASC LIMIT ?""", (now, limit))
                rows = cur.fetchall() or []
                for fid, rp in rows:
                    self.storage._conn.execute(
                        "UPDATE forecast_ledger SET status='expired', outcome_type='false_alarm', outcome_ts=? "
                        "WHERE forecast_id=?", (now, fid))
                    expired += 1
                if expired:
                    self.storage._conn.commit()
        except Exception as e:
            logger.warning(f"forecast_expire_open: {e}")
        return {"ok": True, "expired": expired}

    def forecast_auto_resolve(self, device_id: str, outcome_type: str,
                              note: str = "") -> int:
        """
        device_id の open 予兆を自動 outcome 登録。
        cockpit の Execute 成功時・アラーム確定時から呼ばれる。
        解決した件数を返す。
        """
        if not self.storage._conn:
            return 0
        resolved = 0
        try:
            with self.storage._db_lock:
                cur = self.storage._conn.cursor()
                cur.execute("""
                    SELECT forecast_id FROM forecast_ledger
                    WHERE device_id=? AND status='open'
                    ORDER BY created_at DESC""", (device_id,))
                rows = cur.fetchall() or []
            for (fid,) in rows:
                r = self.forecast_register_outcome(
                    fid, outcome_type, note=note, auto=True)
                if r.get("ok"):
                    resolved += 1
        except Exception as e:
            logger.warning(f"forecast_auto_resolve: {e}")
        return resolved

    def forecast_auto_confirm_on_incident(self, device_id: str, scenario: str = "",
                                          note: str = "") -> int:
        """
        障害発生時に該当デバイスの open 予兆を自動的に confirmed_incident に更新
        
        運用実態に即した設計:
        - 運用者が「障害確認済み」を手動登録するのは非現実的
        - 障害シナリオ発生時に自動判定する方が正確
        
        Args:
            device_id: 障害が発生したデバイスID
            scenario: 発生した障害シナリオ名（ログ用）
            note: 追加メモ
        
        Returns:
            confirmed に更新した予兆の件数
        """
        if not self.storage._conn:
            return 0
        confirmed = 0
        auto_note = f"Auto-confirmed on incident: {scenario}" if scenario else "Auto-confirmed on incident"
        if note:
            auto_note += f" | {note}"
        
        try:
            with self.storage._db_lock:
                cur = self.storage._conn.cursor()
                cur.execute("""
                    SELECT forecast_id FROM forecast_ledger
                    WHERE device_id=? AND status='open'
                    ORDER BY created_at DESC""", (device_id,))
                rows = cur.fetchall() or []
            
            for (fid,) in rows:
                r = self.forecast_register_outcome(
                    fid, "confirmed_incident", note=auto_note, auto=True)
                if r.get("ok"):
                    confirmed += 1
                    logger.info(f"Auto-confirmed forecast {fid[:12]} on incident: {scenario}")
        except Exception as e:
            logger.warning(f"forecast_auto_confirm_on_incident: {e}")
        
        return confirmed

    def forecast_list_open(self, device_id: Optional[str] = None,
                           limit: int = 50) -> List[Dict[str, Any]]:
        """open 中の予兆リストを返す（UI表示用）"""
        if not self.storage._conn:
            return []
        try:
            with self.storage._db_lock:
                cur = self.storage._conn.cursor()
                if device_id:
                    cur.execute("""
                        SELECT forecast_id, created_at, device_id, rule_pattern,
                               predicted_state, confidence, eval_deadline_ts, source, input_json
                        FROM forecast_ledger
                        WHERE status='open' AND device_id=?
                        ORDER BY created_at DESC LIMIT ?""", (device_id, limit))
                else:
                    cur.execute("""
                        SELECT forecast_id, created_at, device_id, rule_pattern,
                               predicted_state, confidence, eval_deadline_ts, source, input_json
                        FROM forecast_ledger
                        WHERE status='open'
                        ORDER BY confidence DESC, created_at DESC LIMIT ?""", (limit,))
                rows = cur.fetchall() or []
            keys = ["forecast_id","created_at","device_id","rule_pattern",
                    "predicted_state","confidence","eval_deadline_ts","source","input_json"]
            result = []
            for r in rows:
                d = dict(zip(keys, r))
                # input_jsonからログメッセージを抽出
                try:
                    if d.get("input_json"):
                        input_data = json.loads(d["input_json"])
                        d["message"] = input_data.get("msg", "")
                except Exception:
                    d["message"] = ""
                result.append(d)
            return result
        except Exception:
            return []
