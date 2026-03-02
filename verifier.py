# -*- coding: utf-8 -*-
"""
Google Antigravity AIOps Agent - Verification Module (Validated & Improved)
ハルシネーション防止のための高速・高精度な検証モジュール

改善点:
1. UTF-8エンコーディング宣言追加
2. Cisco形式Ping判定の修正
3. ロギング機能の追加
4. 型ヒントの完全化
5. エラーハンドリングの強化
6. ドキュメントの充実
"""
import re
import logging
from typing import Dict, Any, Optional

# =====================================================
# ロギング設定
# =====================================================

logger = logging.getLogger(__name__)

# =====================================================
# パターンキャッシュ（遅延初期化クラス）
# =====================================================

class _PatternCache:
    """
    正規表現パターンを保持するシングルトン
    
    パターンのコンパイルは高コストなので、1度だけ実行して再利用する。
    """
    _instance: Optional['_PatternCache'] = None
    _initialized: bool = False
    
    def __new__(cls) -> '_PatternCache':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logger.debug("PatternCache instance created")
        return cls._instance
    
    def __init__(self) -> None:
        if not self._initialized:
            self._compile_patterns()
            self._initialized = True
    
    def _compile_patterns(self) -> None:
        """正規表現パターンのコンパイル"""
        try:
            # Pingパターン
            self.ping_stats = re.compile(
                r'(?:(\d+)\s+packets?\s+transmitted.*?(\d+)\s+received)|'
                r'(?:success\s+rate\s+is\s+(\d+)\s*percent)|'
                r'(?:(\d+)%\s+packet\s+loss)',
                re.I
            )
            # 【修正】timeout誤検出を防ぐため、より厳密なパターンに変更
            self.ping_fail_fast = re.compile(
                r'(100%\s+packet\s+loss|unreachable|'
                r'(?:request|connection)\s+timed?\s*out|'
                r'(?:0|zero)\s+(?:packets?\s+)?received)', 
                re.I
            )
            
            # Cisco形式の成功パターン（!!!!!）
            self.cisco_ping_success = re.compile(r'!{3,}')
            
            # インターフェース
            self.admin_down = re.compile(r'administratively\s+down', re.I)
            self.if_status = re.compile(
                r'(?:line\s+protocol\s+is\s+(up|down))|'
                r'(?:interface\s+is\s+(up|down))|'
                r'(?:(err-disabled|notconnect))',
                re.I
            )
            # 【追加】show ip interface brief 用の簡略表記パターン
            self.if_brief_up = re.compile(r'\bup\s+up\b', re.I)
            self.if_brief_down = re.compile(r'\bdown\s+down\b', re.I)
            
            # ハードウェア
            self.hw_check = re.compile(
                r'(fan|power|psu|temp|environment|sensor).*?'
                r'(fail(ed|ure)?|fault(y)?|critical|ok|good|normal|warn(ing)?)',
                re.I | re.DOTALL
            )
            
            logger.debug("Regex patterns compiled successfully")
        
        except Exception as e:
            logger.error(f"Failed to compile regex patterns: {e}")
            raise

_cache: Optional[_PatternCache] = None

def _get_cache() -> _PatternCache:
    """キャッシュインスタンスの取得"""
    global _cache
    if _cache is None:
        _cache = _PatternCache()
    return _cache

# =====================================================
# 検証ロジック
# =====================================================

def verify_log_content(log_text: str) -> Dict[str, Any]:
    """
    ログテキストから客観的事実を抽出する
    
    Args:
        log_text: 検証対象のログテキスト
    
    Returns:
        検証結果を含む辞書:
        - ping_status: "OK" | "WARNING" | "CRITICAL" | "Unknown"
        - ping_confidence: 0.0 ~ 1.0
        - ping_evidence: 判定根拠
        - interface_status: "OK" | "WARNING" | "CRITICAL" | "INFO" | "Unknown"
        - interface_confidence: 0.0 ~ 1.0
        - interface_evidence: 判定根拠
        - hardware_status: "OK" | "WARNING" | "CRITICAL" | "Unknown"
        - hardware_confidence: 0.0 ~ 1.0
        - hardware_evidence: 判定根拠
        - error_keywords: 検出されたエラーキーワード
        - error_severity: 0.0 ~ 1.0
        - conflicts_detected: 矛盾のリスト
        - overall_confidence: 総合信頼度 0.0 ~ 1.0
    """
    logger.debug(f"Verifying log content ({len(log_text) if log_text else 0} chars)")
    
    # 空の結果定義
    result: Dict[str, Any] = {
        "ping_status": "Unknown", "ping_confidence": 0.0, "ping_evidence": "",
        "interface_status": "Unknown", "interface_confidence": 0.0, "interface_evidence": "",
        "hardware_status": "Unknown", "hardware_confidence": 0.0, "hardware_evidence": "",
        "error_keywords": "None", "error_severity": 0.0,
        "conflicts_detected": [], "overall_confidence": 0.0
    }

    if not log_text:
        logger.debug("Empty log text, returning default result")
        return result
    
    try:
        cache = _get_cache()
        text_lower = log_text.lower()
        
        # 1. Ping検証
        _fast_verify_ping(log_text, text_lower, cache, result)
        
        # 2. Interface検証
        _fast_verify_interface(text_lower, cache, result)
        
        # 3. Hardware検証
        if any(kw in text_lower for kw in ['fan', 'power', 'psu', 'temp', 'environment']):
            _fast_verify_hardware(text_lower, cache, result)
        
        # 4. エラーキーワード
        _fast_verify_errors(text_lower, result)
        
        # 5. 矛盾検知
        _detect_simple_conflicts(result)
        
        # 全体信頼度計算
        confidences = [
            result.get("ping_confidence", 0),
            result.get("interface_confidence", 0),
            result.get("hardware_confidence", 0)
        ]
        result["overall_confidence"] = max(confidences) if any(confidences) else 0.0
        
        logger.debug(f"Verification complete: {result['overall_confidence']:.0%} confidence")
    
    except Exception as e:
        logger.exception(f"Error during log verification: {e}")
        # エラー時も結果を返す（デフォルト値）
    
    return result

def _fast_verify_ping(
    text: str, 
    text_lower: str, 
    cache: _PatternCache, 
    result: Dict[str, Any]
) -> None:
    """
    Ping疎通性の検証
    
    検証パターン:
    1. 失敗パターン（100% packet loss, unreachable等）
    2. Cisco形式（!!!!! + success rate）
    3. 標準形式（packets transmitted/received）
    """
    if 'ping' not in text_lower and 'icmp' not in text_lower:
        return
    
    logger.debug("Verifying ping status")
    
    # 失敗パターンの高速チェック
    fail_match = cache.ping_fail_fast.search(text_lower)
    if fail_match:
        result.update({
            "ping_status": "CRITICAL",
            "ping_confidence": 0.9,
            "ping_evidence": f"Failure: {fail_match.group(1)}"
        })
        logger.debug(f"Ping CRITICAL detected: {fail_match.group(1)}")
        return
    
    # 【改善】Cisco形式の判定を強化
    cisco_match = cache.cisco_ping_success.search(text)
    if cisco_match:
        logger.debug("Cisco ping format detected (!!!!)")
        # "success rate is XX percent" を探す
        success_match = re.search(r'success\s+rate\s+is\s+(\d+)\s*percent', text_lower)
        if success_match:
            try:
                success_rate = int(success_match.group(1))
                logger.debug(f"Cisco ping success rate: {success_rate}%")
                
                if success_rate >= 80:
                    status, conf = "OK", 0.9
                elif success_rate >= 50:
                    status, conf = "WARNING", 0.7
                else:
                    status, conf = "CRITICAL", 0.8
                
                result.update({
                    "ping_status": status,
                    "ping_confidence": conf,
                    "ping_evidence": f"Success rate: {success_rate}%"
                })
                return
            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse Cisco ping success rate: {e}")
    
    # 標準形式の統計パターン
    stats_match = cache.ping_stats.search(text_lower)
    if stats_match:
        groups = stats_match.groups()
        success_rate = None
        try:
            if groups[0] and groups[1]:
                sent, received = int(groups[0]), int(groups[1])
                success_rate = (received / sent * 100) if sent > 0 else 0
                logger.debug(f"Ping stats: {sent} sent, {received} received")
            elif groups[2]:
                success_rate = int(groups[2])
                logger.debug(f"Ping success rate from group 2: {success_rate}%")
            elif groups[3]:
                success_rate = 100 - int(groups[3])
                logger.debug(f"Ping success rate from packet loss: {success_rate}%")
            
            if success_rate is not None:
                if success_rate >= 80:
                    status, conf = "OK", 0.9
                elif success_rate >= 50:
                    status, conf = "WARNING", 0.7
                else:
                    status, conf = "CRITICAL", 0.8
                
                result.update({
                    "ping_status": status,
                    "ping_confidence": conf,
                    "ping_evidence": f"Success rate: {success_rate:.0f}%"
                })
                logger.debug(f"Ping {status}: {success_rate:.0f}%")
        except (ValueError, ZeroDivisionError) as e:
            logger.warning(f"Failed to calculate ping success rate: {e}")

def _fast_verify_interface(
    text: str, 
    cache: _PatternCache, 
    result: Dict[str, Any]
) -> None:
    """
    インターフェース状態の検証
    
    検証パターン:
    1. Admin down（意図的なダウン）
    2. show ip interface brief 形式 (up up / down down)
    3. Line protocol up/down
    4. Interface up/down
    """
    logger.debug("Verifying interface status")
    
    # Admin downの特殊処理
    if cache.admin_down.search(text):
        result.update({
            "interface_status": "INFO",
            "interface_confidence": 0.9,
            "interface_evidence": "Admin down (intentional)"
        })
        logger.debug("Interface INFO: Admin down detected")
        return

    # =========================================================
    # 【追加】show ip interface brief 形式 (up up / down down) の高速検知
    # =========================================================
    if cache.if_brief_down.search(text):
        result.update({
            "interface_status": "CRITICAL",
            "interface_confidence": 0.9,
            "interface_evidence": "Link DOWN (show ip interface brief)"
        })
        logger.debug("Interface CRITICAL: 'down down' detected")
        return
        
    if cache.if_brief_up.search(text):
        result.update({
            "interface_status": "OK",
            "interface_confidence": 0.9,
            "interface_evidence": "Link UP (show ip interface brief)"
        })
        logger.debug("Interface OK: 'up up' detected")
        return
    # =========================================================
    
    # 従来のインターフェース状態の検出 (show interfaces等)
    status_match = cache.if_status.findall(text)
    if not status_match:
        return
    
    # インターフェース状態の検出
    status_match = cache.if_status.findall(text)
    if not status_match:
        return
    
    down_count = sum(
        1 for m in status_match 
        if 'down' in str(m).lower() or 'disabled' in str(m).lower()
    )
    up_count = sum(1 for m in status_match if 'up' in str(m).lower())
    
    logger.debug(f"Interface status: {up_count} UP, {down_count} DOWN")
    
    if down_count > up_count:
        result.update({
            "interface_status": "CRITICAL",
            "interface_confidence": 0.9,
            "interface_evidence": f"Link DOWN detected ({down_count} interfaces)"
        })
        logger.debug(f"Interface CRITICAL: {down_count} interfaces down")
    elif up_count > down_count:
        result.update({
            "interface_status": "OK",
            "interface_confidence": 0.8,
            "interface_evidence": f"Link UP ({up_count} interfaces)"
        })
        logger.debug(f"Interface OK: {up_count} interfaces up")
    else:
        result.update({
            "interface_status": "WARNING",
            "interface_confidence": 0.5,
            "interface_evidence": "Mixed states"
        })
        logger.debug("Interface WARNING: Mixed states")

def _fast_verify_hardware(
    text: str, 
    cache: _PatternCache, 
    result: Dict[str, Any]
) -> None:
    """
    ハードウェア状態の検証
    
    検証対象:
    - Fan（ファン）
    - Power Supply（電源）
    - Temperature（温度）
    """
    logger.debug("Verifying hardware status")
    
    hw_matches = cache.hw_check.findall(text)
    if not hw_matches:
        return
    
    critical_count = sum(
        1 for m in hw_matches 
        if any(k in str(m).lower() for k in ['fail', 'fault', 'critical'])
    )
    ok_count = sum(
        1 for m in hw_matches 
        if any(k in str(m).lower() for k in ['ok', 'good', 'normal'])
    )
    warning_count = sum(1 for m in hw_matches if 'warn' in str(m).lower())
    
    logger.debug(
        f"Hardware: {critical_count} critical, {warning_count} warning, {ok_count} ok"
    )
    
    if critical_count > 0:
        result.update({
            "hardware_status": "CRITICAL",
            "hardware_confidence": 0.9,
            "hardware_evidence": f"HW failure detected ({critical_count} issues)"
        })
        logger.debug(f"Hardware CRITICAL: {critical_count} issues")
    elif warning_count > 0:
        result.update({
            "hardware_status": "WARNING",
            "hardware_confidence": 0.8,
            "hardware_evidence": f"HW warning ({warning_count} issues)"
        })
        logger.debug(f"Hardware WARNING: {warning_count} issues")
    elif ok_count > 0:
        result.update({
            "hardware_status": "OK",
            "hardware_confidence": 0.8,
            "hardware_evidence": f"HW OK ({ok_count} components)"
        })
        logger.debug(f"Hardware OK: {ok_count} components")

def _fast_verify_errors(text: str, result: Dict[str, Any]) -> None:
    """
    エラーキーワードの検出
    
    検出レベル:
    - Critical: crash, panic, fatal, severe
    - Error: error, fail, exception, denied
    """
    logger.debug("Verifying error keywords")
    
    critical_keywords = ['crash', 'panic', 'fatal', 'severe']
    error_keywords = ['error', 'fail', 'exception', 'denied']
    
    found_critical = [k for k in critical_keywords if k in text]
    found_errors = [k for k in error_keywords if k in text and k not in found_critical]
    
    if found_critical:
        result.update({
            "error_keywords": f"Critical: {', '.join(found_critical[:3])}",
            "error_severity": 0.9
        })
        logger.debug(f"Critical keywords found: {found_critical}")
    elif found_errors:
        result.update({
            "error_keywords": f"Errors: {', '.join(found_errors[:3])}",
            "error_severity": 0.7
        })
        logger.debug(f"Error keywords found: {found_errors}")

def _detect_simple_conflicts(result: Dict[str, Any]) -> None:
    """
    検証結果の矛盾を検知
    
    例: Pingは成功しているがインターフェースがダウン
    """
    logger.debug("Detecting conflicts")
    
    conflicts = []
    ping_ok = result.get("ping_status") == "OK"
    if_down = result.get("interface_status") == "CRITICAL"
    
    if ping_ok and if_down:
        conflict = "矛盾検知: Ping疎通は成功していますが、I/Fダウンが検出されています"
        conflicts.append(conflict)
        logger.warning(conflict)
    
    result["conflicts_detected"] = conflicts

# =====================================================
# レポートフォーマット関数
# =====================================================

def format_verification_report(facts: Dict[str, Any]) -> str:
    """
    検証結果を整形して返す
    
    Args:
        facts: verify_log_content()の戻り値
    
    Returns:
        整形されたレポート文字列
    """
    overall_conf = facts.get('overall_confidence', 0)
    confidence_level = "高" if overall_conf >= 0.8 else "中" if overall_conf >= 0.5 else "低"
    
    report = f"""
【システム自動検証結果 (Ground Truth)】
※AIの推論はこの客観的事実と矛盾してはならない

◆ 総合信頼度: {confidence_level} ({overall_conf:.0%})

◆ 疎通: {facts.get('ping_status', 'N/A')} (信頼度: {facts.get('ping_confidence', 0):.0%})
  → {facts.get('ping_evidence', 'N/A')}

◆ インターフェース: {facts.get('interface_status', 'N/A')} (信頼度: {facts.get('interface_confidence', 0):.0%})
  → {facts.get('interface_evidence', 'N/A')}

◆ ハードウェア: {facts.get('hardware_status', 'N/A')} (信頼度: {facts.get('hardware_confidence', 0):.0%})
  → {facts.get('hardware_evidence', 'N/A')}

◆ エラー: {facts.get('error_keywords', 'N/A')}
"""
    
    if facts.get('conflicts_detected'):
        report += f"\n⚠️ **矛盾検知**: {'; '.join(facts['conflicts_detected'])}\n"
    
    return report
