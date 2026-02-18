# digital_twin_pkg/bayesian.py - ベイズ推論エンジン

import logging
import time
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


class BayesianInferenceEngine:
    """
    ベイズ推論エンジン
    
    過去の予兆履歴から事後確率を計算し、予測精度を向上
    """
    
    def __init__(self, storage_manager):
        """
        Args:
            storage_manager: StorageManager インスタンス
        """
        self.storage = storage_manager
        self._prior_cache = {}  # device_id + rule_pattern → 事前確率
        self._last_cache_update = 0
        self.CACHE_TTL = 3600  # 1時間
    
    def calculate_posterior_confidence(
        self,
        device_id: str,
        rule_pattern: str,
        current_confidence: float,
        time_window_hours: int = 168  # 過去7日間
    ) -> Tuple[float, Dict[str, any]]:
        """
        ベイズ推論による事後確率の計算
        
        P(障害|シグナル) = P(シグナル|障害) × P(障害) / P(シグナル)
        
        Args:
            device_id: デバイスID
            rule_pattern: ルールパターン
            current_confidence: 現在の信頼度（事前確率）
            time_window_hours: 過去データの参照期間（時間）
        
        Returns:
            (事後確率, デバッグ情報)
        """
        # 過去の履歴を取得
        history = self._get_historical_data(device_id, rule_pattern, time_window_hours)
        
        if not history:
            # 履歴がない場合は現在の信頼度をそのまま返す
            return current_confidence, {
                "method": "no_history",
                "prior": current_confidence,
                "posterior": current_confidence
            }
        
        # 統計情報の計算
        stats = self._calculate_statistics(history)
        
        # ベイズ更新
        posterior = self._bayesian_update(
            prior=current_confidence,
            likelihood=stats['success_rate'],
            evidence=stats['total_count']
        )
        
        # 時系列パターンによるブースト
        temporal_boost = self._calculate_temporal_boost(history)
        posterior = min(0.99, posterior + temporal_boost)
        
        debug_info = {
            "method": "bayesian_inference",
            "prior": current_confidence,
            "likelihood": stats['success_rate'],
            "evidence_count": stats['total_count'],
            "temporal_boost": temporal_boost,
            "posterior": posterior,
            "history_summary": {
                "total": stats['total_count'],
                "confirmed": stats['confirmed_count'],
                "mitigated": stats['mitigated_count'],
                "false_alarm": stats['false_alarm_count']
            }
        }
        
        return posterior, debug_info
    
    def _get_historical_data(
        self,
        device_id: str,
        rule_pattern: str,
        time_window_hours: int
    ) -> List[Dict]:
        """過去の予兆履歴を取得"""
        if not self.storage._conn:
            return []
        
        cutoff_time = time.time() - (time_window_hours * 3600)
        
        try:
            with self.storage._db_lock:
                cur = self.storage._conn.cursor()
                cur.execute("""
                    SELECT forecast_id, created_at, confidence, status, 
                           outcome_type, outcome_ts, eval_deadline_ts
                    FROM forecast_ledger
                    WHERE device_id = ? 
                      AND rule_pattern = ?
                      AND created_at >= ?
                      AND status != 'open'
                    ORDER BY created_at DESC
                """, (device_id, rule_pattern, cutoff_time))
                
                rows = cur.fetchall() or []
                
                keys = ["forecast_id", "created_at", "confidence", "status",
                       "outcome_type", "outcome_ts", "eval_deadline_ts"]
                return [dict(zip(keys, row)) for row in rows]
        
        except Exception as e:
            logger.warning(f"Failed to get historical data: {e}")
            return []
    
    def _calculate_statistics(self, history: List[Dict]) -> Dict:
        """履歴から統計情報を計算"""
        total = len(history)
        confirmed = sum(1 for h in history if h.get('outcome_type') == 'confirmed_incident')
        mitigated = sum(1 for h in history if h.get('outcome_type') == 'mitigated')
        false_alarm = sum(1 for h in history if h.get('outcome_type') == 'false_alarm')
        
        # 成功率の計算
        # confirmed: 予兆が的中（成功）
        # mitigated: 予兆に基づいて対応し障害回避（成功）
        # false_alarm: 予兆が外れた（失敗）
        success_count = confirmed + mitigated
        success_rate = success_count / total if total > 0 else 0.5
        
        return {
            'total_count': total,
            'confirmed_count': confirmed,
            'mitigated_count': mitigated,
            'false_alarm_count': false_alarm,
            'success_rate': success_rate
        }
    
    def _bayesian_update(
        self,
        prior: float,
        likelihood: float,
        evidence: int
    ) -> float:
        """
        ベイズ更新
        
        事後確率 = (尤度 × 事前確率) / 証拠
        
        簡易版: 履歴の成功率を尤度として使用
        """
        # データ数が少ない場合は信頼性を下げる
        confidence_factor = min(1.0, evidence / 10)  # 10件以上で完全信頼
        
        # ベイズ更新（簡易版）
        # 完全版: P(H|E) = P(E|H) × P(H) / P(E)
        # 簡易版: 尤度と事前確率の加重平均
        posterior = (likelihood * confidence_factor) + (prior * (1 - confidence_factor))
        
        return min(0.99, max(0.1, posterior))
    
    def _calculate_temporal_boost(self, history: List[Dict]) -> float:
        """
        時系列パターンによるブースト計算
        
        断続的に繰り返し発生している場合、信頼度を上げる
        """
        if len(history) < 2:
            return 0.0
        
        # 最近1週間での発生回数
        recent_cutoff = time.time() - (7 * 24 * 3600)
        recent_count = sum(1 for h in history if h['created_at'] >= recent_cutoff)
        
        # 繰り返しパターンの検出
        if recent_count >= 3:
            # 3回以上繰り返し → 強いパターン
            return 0.10
        elif recent_count >= 2:
            # 2回繰り返し → 中程度のパターン
            return 0.05
        
        return 0.0
    
    def get_device_reliability_score(self, device_id: str) -> Dict[str, float]:
        """
        デバイスごとの信頼性スコアを計算
        
        過去の予兆がどれだけ的中したかを評価
        """
        if not self.storage._conn:
            return {"score": 0.5, "count": 0}
        
        try:
            with self.storage._db_lock:
                cur = self.storage._conn.cursor()
                cur.execute("""
                    SELECT outcome_type, COUNT(*) as count
                    FROM forecast_ledger
                    WHERE device_id = ? AND status != 'open'
                    GROUP BY outcome_type
                """, (device_id,))
                
                rows = cur.fetchall() or []
                
                stats = {outcome: count for outcome, count in rows}
                total = sum(stats.values())
                
                if total == 0:
                    return {"score": 0.5, "count": 0}
                
                confirmed = stats.get('confirmed_incident', 0)
                mitigated = stats.get('mitigated', 0)
                false_alarm = stats.get('false_alarm', 0)
                
                # 信頼性スコア = (的中 + 対応成功) / 総数
                score = (confirmed + mitigated) / total
                
                return {
                    "score": score,
                    "count": total,
                    "confirmed": confirmed,
                    "mitigated": mitigated,
                    "false_alarm": false_alarm
                }
        
        except Exception as e:
            logger.warning(f"Failed to calculate reliability score: {e}")
            return {"score": 0.5, "count": 0}
