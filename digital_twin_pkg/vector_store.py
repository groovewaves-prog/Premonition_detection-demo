# digital_twin_pkg/vector_store.py
# Phase 6c*: ChromaDB ベクトルストア — 類似インシデント検索
#
# explanation_panel.py の _render_similar_incidents() から呼び出される。
# 期待インターフェース:
#   vs.search_similar_alarms(alarm_text=str, n_results=int) -> List[Dict]
#   vs.add_incident(...)
#   vs.update_outcome(incident_id, outcome)

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ChromaDB は optional — 未インストール時はダミーモードで動作


# ──────────────────────────────────────────────
# 埋め込み関数: 優先度順にフォールバック
#   1. SentenceTransformer (all-MiniLM-L6-v2)
#   2. ローカル N-gram ハッシュ埋め込み（依存なし）
# ──────────────────────────────────────────────

def _create_embedding_function():
    """利用可能な最良の埋め込み関数を返す。"""
    # 1) SentenceTransformer（高精度）
    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        ef(["test"])  # 動作確認
        logger.info("VectorStore: using SentenceTransformer embedding")
        return ef
    except Exception:
        pass

    # 2) ローカル N-gram ハッシュ埋め込み（フォールバック）
    logger.info("VectorStore: using local n-gram hash embedding (fallback)")
    return _LocalHashEmbeddingFunction()


class _LocalHashEmbeddingFunction:
    """
    SentenceTransformer が利用できない場合のフォールバック埋め込み関数。
    文字 N-gram のハッシュトリックで固定次元ベクトルを生成する。
    意味的類似度は限定的だが、構造的に類似したアラーム文の検索には十分。
    """

    DIM = 384  # all-MiniLM-L6-v2 と同次元
    NGRAM_RANGE = (2, 4)  # 文字 bigram ～ 4-gram

    def __call__(self, input: List[str]) -> List[List[float]]:
        import hashlib
        import math

        results = []
        for text in input:
            vec = [0.0] * self.DIM
            text_lower = text.lower().strip()
            token_count = 0
            for n in range(self.NGRAM_RANGE[0], self.NGRAM_RANGE[1] + 1):
                for i in range(len(text_lower) - n + 1):
                    gram = text_lower[i : i + n]
                    h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
                    idx = h % self.DIM
                    sign = 1.0 if (h // self.DIM) % 2 == 0 else -1.0
                    vec[idx] += sign
                    token_count += 1
            # L2 正規化
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vec = [v / norm for v in vec]
            results.append(vec)
        return results

    # ChromaDB EmbeddingFunction プロトコル準拠
    def name(self) -> str:
        return "local_ngram_hash"

    def embed_query(self, input: List[str]) -> List[List[float]]:
        """クエリ用埋め込み（ドキュメント埋め込みと同一）。"""
        return self.__call__(input)

    @staticmethod
    def build_from_config(config: Dict) -> "_LocalHashEmbeddingFunction":
        return _LocalHashEmbeddingFunction()

    def get_config(self) -> Dict:
        return {"dim": self.DIM, "ngram_range": list(self.NGRAM_RANGE)}


class VectorStore:
    """
    ChromaDB を用いたインシデントベクトルストア。

    - テナントごとに独立した永続ディレクトリを持つ
    - SentenceTransformer (all-MiniLM-L6-v2) で自動埋め込み
    - ChromaDB 未インストール時は全メソッドが安全に no-op を返す
    """

    # ChromaDB コレクション名
    COLLECTION_NAME = "incidents"

    def __init__(
        self,
        persist_directory: str,
        tenant_id: str = "default",
    ):
        self.tenant_id = tenant_id
        self.persist_directory = persist_directory
        self._client = None
        self._collection = None
        self._ready = False

        if not HAS_CHROMADB:
            logger.warning("VectorStore: chromadb unavailable — no-op mode")
            return

        try:
            os.makedirs(persist_directory, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                ),
            )
            # 埋め込み関数: SentenceTransformer → ローカルハッシュ のフォールバック
            self._embedding_fn = _create_embedding_function()
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=self._embedding_fn,
                metadata={
                    "hnsw:space": "cosine",       # コサイン類似度
                    "tenant_id": self.tenant_id,
                },
            )
            self._ready = True
            count = self._collection.count()
            logger.info(
                f"VectorStore ready: tenant={tenant_id}, "
                f"incidents={count}, path={persist_directory}"
            )
        except Exception as e:
            logger.error(f"VectorStore init failed: {e}")
            self._ready = False

    # ──────────────────────────────────────────────
    # 書き込み
    # ──────────────────────────────────────────────

    def add_incident(
        self,
        alarm_text: str,
        device_id: str = "",
        rule_pattern: str = "",
        confidence: float = 0.0,
        vendor_context: str = "",
        anomaly_type: str = "point",
        score_breakdown: Optional[Dict[str, float]] = None,
        outcome: str = "pending",
        incident_id: Optional[str] = None,
        created_at: Optional[float] = None,
    ) -> Optional[str]:
        """
        予兆予測結果を 1 件ベクトルストアに登録する。

        Returns:
            incident_id (str) or None if failed
        """
        if not self._ready:
            return None

        iid = incident_id or str(uuid.uuid4())
        ts = created_at or time.time()
        sb = score_breakdown or {}

        metadata = {
            "device_id":      device_id,
            "rule_pattern":   rule_pattern,
            "confidence":     float(confidence),
            "vendor_context": vendor_context or "",
            "anomaly_type":   anomaly_type,
            "outcome":        outcome,
            "created_at":     float(ts),
            # スコア内訳（ChromaDB metadata はフラット key-value のみ）
            "score_semantic":    float(sb.get("semantic", 0.0)),
            "score_trend":       float(sb.get("trend", 0.0)),
            "score_volatility":  float(sb.get("volatility", 0.0)),
            "score_history":     float(sb.get("history", 0.0)),
            "score_interaction": float(sb.get("interaction", 0.0)),
        }

        try:
            self._collection.upsert(
                ids=[iid],
                documents=[alarm_text],
                metadatas=[metadata],
            )
            logger.debug(f"VectorStore.add_incident: id={iid}, device={device_id}")
            return iid
        except Exception as e:
            logger.error(f"VectorStore.add_incident failed: {e}")
            return None

    # ──────────────────────────────────────────────
    # 検索
    # ──────────────────────────────────────────────

    def search_similar_alarms(
        self,
        alarm_text: str,
        n_results: int = 5,
        min_similarity: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        alarm_text に意味的に類似した過去インシデントを検索する。

        Returns:
            list of dict:
                - text: str
                - similarity: float (0-1, 高いほど類似)
                - outcome: str
                - vendor_context: str
                - created_at: float (epoch)
                - device_id: str
                - rule_pattern: str
                - incident_id: str
        """
        if not self._ready:
            return []

        try:
            count = self._collection.count()
            if count == 0:
                return []
            # n_results がコレクション件数を超えないように
            actual_n = min(n_results, count)
            results = self._collection.query(
                query_texts=[alarm_text],
                n_results=actual_n,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"VectorStore.search_similar_alarms failed: {e}")
            return []

        # ChromaDB cosine distance → similarity (1 - distance)
        out: List[Dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for i, (doc_id, doc, meta, dist) in enumerate(
            zip(ids, docs, metas, dists)
        ):
            # cosine distance: 0=identical, 2=opposite → similarity = 1 - dist/2
            # ただし ChromaDB の cosine space では distance ∈ [0, 2]
            similarity = max(0.0, 1.0 - dist / 2.0)
            if similarity < min_similarity:
                continue
            out.append(
                {
                    "incident_id":    doc_id,
                    "text":           doc or "",
                    "similarity":     round(similarity, 4),
                    "outcome":        meta.get("outcome", "pending"),
                    "vendor_context": meta.get("vendor_context", ""),
                    "created_at":     meta.get("created_at", 0),
                    "device_id":      meta.get("device_id", ""),
                    "rule_pattern":   meta.get("rule_pattern", ""),
                    "confidence":     meta.get("confidence", 0.0),
                    "anomaly_type":   meta.get("anomaly_type", "point"),
                }
            )

        # 類似度降順
        out.sort(key=lambda x: x["similarity"], reverse=True)
        return out

    # ──────────────────────────────────────────────
    # 更新
    # ──────────────────────────────────────────────

    def update_outcome(self, incident_id: str, outcome: str) -> bool:
        """
        登録済みインシデントの outcome を更新する。

        Args:
            incident_id: add_incident で返された ID
            outcome: "confirmed" | "mitigated" | "false_alarm" | "pending"
        """
        if not self._ready:
            return False

        try:
            existing = self._collection.get(ids=[incident_id], include=["metadatas"])
            if not existing or not existing["ids"]:
                logger.warning(f"VectorStore.update_outcome: id={incident_id} not found")
                return False
            meta = existing["metadatas"][0]
            meta["outcome"] = outcome
            self._collection.update(ids=[incident_id], metadatas=[meta])
            logger.debug(f"VectorStore.update_outcome: id={incident_id} → {outcome}")
            return True
        except Exception as e:
            logger.error(f"VectorStore.update_outcome failed: {e}")
            return False

    # ──────────────────────────────────────────────
    # ユーティリティ
    # ──────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """コレクション統計を返す（デバッグ用）。"""
        if not self._ready:
            return {"ready": False, "count": 0}
        try:
            count = self._collection.count()
            return {
                "ready": True,
                "count": count,
                "tenant_id": self.tenant_id,
                "persist_directory": self.persist_directory,
            }
        except Exception as e:
            return {"ready": False, "error": str(e)}

    def delete_all(self) -> bool:
        """全インシデントを削除（テスト用）。"""
        if not self._ready:
            return False
        try:
            # ChromaDB: コレクションを再作成
            self._client.delete_collection(self.COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={
                    "hnsw:space": "cosine",
                    "tenant_id": self.tenant_id,
                },
            )
            logger.info("VectorStore.delete_all: collection reset")
            return True
        except Exception as e:
            logger.error(f"VectorStore.delete_all failed: {e}")
            return False

    @property
    def is_ready(self) -> bool:
        return self._ready
