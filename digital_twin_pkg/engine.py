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
        llm_config: Optional[Dict[str, Any]] = None,  # Phase 6b
    ):
        if not tenant_id or len(tenant_id) > 64: raise ValueError("Invalid tenant_id")
        self.tenant_id = tenant_id.lower()
        self.topology = topology
        self.children_map = children_map or {}
        self.storage = StorageManager(self.tenant_id, BASE_DIR)
        self.tuner = AutoTuner(self)
        self.bayesian = BayesianInferenceEngine(self.storage)  # ★ ベイズ推論エンジン
        self.gnn = create_gnn_engine(topology, children_map)  # ★ GNN予測エンジン

        # Phase 6b: LLM クライアント初期化（Google Gemma-3-12b-it 対応）
        _cfg = llm_config or {}
        self.llm = InternalLLMClient(
            sanitize_fn     = self._sanitize_for_llm,
            api_key         = _cfg.get("google_key"),         # Google AI Studio API Key
            llm_backend     = _cfg.get("backend", "ollama"),  # デフォルト: ハイブリッド
            ollama_base_url = _cfg.get("ollama_url", "http://localhost:11434"),
            ollama_model    = _cfg.get("ollama_model", "gemma3:12b"),
            google_model    = _cfg.get("google_model", "gemma-3-12b-it"),
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
        WIDE_RANGE_THRESHOLD = 5
        
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
        LLMを使って状況に応じた推奨アクションを動的生成
        
        広範囲障害の場合は真因（電源/ファームウェア/環境）を推論
        
        Args:
            llm_cache: バッチ生成された推奨アクションのキャッシュ
        """
        # 広範囲障害の閾値
        WIDE_RANGE_THRESHOLD = 5
        
        # 広範囲障害でない場合は固定アクションを返す
        if affected_count < WIDE_RANGE_THRESHOLD:
            return base_actions
        
        # ★ キャッシュから取得（バッチ生成済み）
        if llm_cache and rule_pattern in llm_cache:
            logger.debug(f"Using cached LLM recommendations for {rule_pattern}")
            return llm_cache[rule_pattern]
        
        # ★ キャッシュになければ個別にGemini API呼び出し（フォールバック）
        llm_actions = self._generate_actions_with_gemini(
            rule_pattern=rule_pattern,
            affected_count=affected_count,
            confidence=confidence,
            messages=messages,
            device_id=device_id
        )
        
        if llm_actions:
            # LLM生成が成功した場合はそれを返す
            return llm_actions
        
        # LLM生成が失敗した場合はフォールバック（静的な高度なアクション）
        enhanced_actions = []
        
        if "optical" in rule_pattern:
            enhanced_actions = [
                {
                    "title": "⚠️ 筐体電源系統の調査（最優先）",
                    "effect": f"電源ユニット故障による{affected_count}個の光モジュール同時劣化を解消",
                    "priority": "high",
                    "rationale": f"{affected_count}個の光モジュール同時劣化は単発故障では説明困難。電源系統の問題を疑う。"
                },
                {
                    "title": "⚠️ IOS/ファームウェアのバグ調査",
                    "effect": "ソフトウェア起因の誤検知/制御異常を解消",
                    "priority": "high",
                    "rationale": "広範囲障害はファームウェアバグの可能性あり"
                },
                {
                    "title": "制御基板の温度/環境調査",
                    "effect": "筐体内過熱による劣化を解消",
                    "priority": "medium",
                    "rationale": "環境要因による全モジュール影響を確認"
                },
                {
                    "title": "SFPモジュールの個別交換（最後の手段）",
                    "effect": "個別モジュール故障の解消",
                    "priority": "low",
                    "rationale": f"{affected_count}個全交換は非現実的、上記を優先"
                }
            ]
        
        elif "microburst" in rule_pattern:
            enhanced_actions = [
                {
                    "title": "⚠️ ASIC/ハードウェアの調査",
                    "effect": f"{affected_count}個のインターフェースでのバッファ問題を解消",
                    "priority": "high",
                    "rationale": "広範囲のqueue dropsはASIC/チップセット問題の可能性"
                },
                {
                    "title": "IOS/ファームウェアのバグ確認",
                    "effect": "QoS処理の異常を解消",
                    "priority": "high",
                    "rationale": "複数ポートでの同時発生はソフトウェアバグの可能性"
                },
                {
                    "title": "トラフィックパターンの分析",
                    "effect": "異常トラフィックの検出・対処",
                    "priority": "medium",
                    "rationale": "DDoS攻撃や異常トラフィックの可能性を確認"
                },
                {
                    "title": "QoSポリシーの調整",
                    "effect": "バッファ割り当ての最適化",
                    "priority": "low",
                    "rationale": "根本原因解決後の最適化"
                }
            ]
        
        elif "route_instability" in rule_pattern or "bgp" in rule_pattern:
            enhanced_actions = [
                {
                    "title": "⚠️ BGP設定の包括的レビュー",
                    "effect": f"{affected_count}個のピアでの不安定さを解消",
                    "priority": "high",
                    "rationale": "複数ピアでの同時発生は設定ミスの可能性"
                },
                {
                    "title": "上流ISPとの連携",
                    "effect": "ISP側の問題を特定・対処",
                    "priority": "high",
                    "rationale": "広範囲ルート不安定はISP側問題の可能性"
                },
                {
                    "title": "IOS/ファームウェアの確認",
                    "effect": "BGP実装のバグを回避",
                    "priority": "medium",
                    "rationale": "BGP処理の異常による不安定さを確認"
                },
                {
                    "title": "BGPフラップダンピングの調整",
                    "effect": "不安定な経路の抑制",
                    "priority": "low",
                    "rationale": "症状の緩和（根本解決ではない）"
                }
            ]
        
        else:
            # デフォルト: 基本アクション + 広範囲調査を追加
            enhanced_actions = base_actions + [
                {
                    "title": "⚠️ システム全体の健全性確認",
                    "effect": f"{affected_count}個のコンポーネント障害の根本原因を特定",
                    "priority": "high",
                    "rationale": "広範囲障害は電源/ファームウェア/環境の問題を疑う"
                }
            ]
        
        return enhanced_actions if enhanced_actions else base_actions

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
    def _predict_rul_with_trend(self, device_id: str, current_level: int, base_ttf_hours: int) -> int:
        """
        過去の履歴データからトレンド（傾き）を NumPy 線形回帰で計算し、
        閾値到達までの RUL（残存寿命）を動的に予測する。
        データ不足時は静的スケール係数にフォールバック。
        """
        now = time.time()
        window_start = now - (24 * 3600)  # 過去24時間

        history_data = [
            h for h in self.history
            if h.get("device_id") == device_id and h.get("timestamp", 0) >= window_start
        ]

        # データ不足時はフォールバック
        if len(history_data) < 3:
            _ttf_scale = (6 - current_level) / 5.0
            return max(1, int(base_ttf_hours * _ttf_scale))

        timestamps = np.array([h.get("timestamp") for h in history_data])
        x = (timestamps - window_start) / 3600.0  # 時間単位
        y = np.array([h.get("prob", 0.1) for h in history_data])

        # 現在の状態を追加
        x = np.append(x, 24.0)
        current_prob = min(0.99, current_level * 0.2)
        y = np.append(y, current_prob)

        try:
            coeffs = np.polyfit(x, y, 1)
            slope = coeffs[0]
            intercept = coeffs[1]

            if slope > 0.001:
                # 閾値 0.95 到達までの時間を予測
                target_x = (0.95 - intercept) / slope
                predicted_rul_hours = target_x - 24.0
                return max(1, min(int(predicted_rul_hours), base_ttf_hours))
            else:
                # 傾きが無視できるほど小さい → 静的スケール
                _ttf_scale = (6 - current_level) / 5.0
                return max(1, int(base_ttf_hours * _ttf_scale))

        except Exception as e:
            logger.warning(f"RUL trend prediction failed for {device_id}: {e}")
            _ttf_scale = (6 - current_level) / 5.0
            return max(1, int(base_ttf_hours * _ttf_scale))

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
        try:
            msg_n = self._normalize_msg(msg or "")
        except AttributeError:
            msg_n = (msg or "").strip()
        except Exception:
            msg_n = (msg or "").strip()

        if self._should_ignore(msg_n):
            return []

        _min_conf = float(MIN_PREDICTION_CONFIDENCE)

        # Level ブースト係数（Level1=0.0 → Level5=0.20）
        _level = max(1, min(5, int(degradation_level or 1)))
        _conf_boost    = (_level - 1) * 0.05          # +0.00〜+0.20
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
                base_conf = float(getattr(rule, "base_confidence", 0.5) or 0.5)
                conf = min(0.99, base_conf + _conf_boost)
                if conf < _min_conf:
                    continue
                _base_ttc   = int(getattr(rule, "time_to_critical_min", 60) or 60)
                _base_early = int(getattr(rule, "early_warning_hours", 24) or 24)
                _ttc   = max(5,  int(_base_ttc   * _ttc_factor))
                _early = max(1,  int(_base_early * _early_factor))
                
                # ★ RUL計算: 過去の時系列トレンドを考慮した動的予測
                _base_ttf_hours = int(getattr(rule, "early_warning_hours", 336) or 336)
                _ttf_hours = self._predict_rul_with_trend(
                    device_id=device_id,
                    current_level=_level,
                    base_ttf_hours=_base_ttf_hours
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

        if results:
            _top = results[0]
            try:
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
                # ★ LLM スコアで confidence を補正（change_impact を加味した重み配分）
                _lf = (
                    _llm_result.scores.semantic      * 0.25 +
                    _llm_result.scores.interaction   * 0.25 +
                    _llm_result.scores.trend         * 0.15 +
                    _llm_result.scores.volatility    * 0.10 +
                    _llm_result.scores.change_impact * 0.25
                )
                _top.confidence = min(0.99, _top.confidence * 0.60 + _top.confidence * _lf * 0.40)
            except Exception as _le:
                logger.debug(f"LLM score skipped: {_le}")

        # Phase 6c*: ベクトルストアに予兆予測を登録（最上位ルールのみ）
        if results and self.vector_store and self.vector_store.is_ready:
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

        return [
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
                fid = self._forecast_record(req=req.to_dict(), top_prediction=preds[0])
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
