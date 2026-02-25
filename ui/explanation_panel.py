# ui/explanation_panel.py
# Phase 6a: explanation_payload UI コンポーネント
#
# render_explanation_panel(pred_item, dt_engine=None)
#   → AI証拠パネル（異常種別バッジ・LLM narrative・レーダーチャート・ベイズ証拠）

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import streamlit as st


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 定数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ANOMALY_TYPE_LABELS = {
    "point":      {"label": "ポイント異常",   "color": "#1565C0", "desc": "単一機器・単一時点の局所的異常"},
    "contextual": {"label": "コンテキスト異常", "color": "#6A1B9A", "desc": "正常範囲内だが文脈上異常"},
    "collective": {"label": "集合的異常",     "color": "#BF360C", "desc": "複数機器で同種シグナルが同時発生"},
    "cascading":  {"label": "カスケード障害",   "color": "#880E4F", "desc": "上流から下流へ連鎖する障害"},
}

_SCORE_LABELS = {
    "semantic":      "意味的深刻度",
    "trend":         "劣化速度",
    "volatility":    "不安定さ",
    "history":       "発生頻度",
    "interaction":   "複合危険度",
    "change_impact": "構成変更影響",  # ★Phase 6c* 統合
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SVG レーダーチャート
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_radar_svg(score_breakdown: Dict[str, float], size: int = 440) -> str:
    # ★ 6次元レーダーチャート（拡大版: 440px — ラベル余白確保）
    keys = ["semantic", "trend", "volatility", "history", "interaction", "change_impact"]
    values = [float(score_breakdown.get(k, 0.5)) for k in keys]
    n = len(keys)
    cx, cy = size / 2, size / 2 - 8   # ★ 上部余白確保のため少し上にオフセット
    r = size * 0.28                     # ★ 半径を少し小さくしてラベル余白確保

    def polar(i, val, scale=1.0):
        angle = math.pi / 2 - (2 * math.pi * i / n)
        rv = r * val * scale
        return cx + rv * math.cos(angle), cy - rv * math.sin(angle)

    # 背景グリッド
    grid_lines = ""
    for lv in [0.25, 0.50, 0.75, 1.0]:
        pts = " ".join(f"{polar(i, lv)[0]:.1f},{polar(i, lv)[1]:.1f}" for i in range(n))
        color = "#FFCDD2" if lv == 1.0 else "#ECEFF1"
        grid_lines += f'<polygon points="{pts}" fill="none" stroke="{color}" stroke-width="1"/>'

    # 軸
    axes = "".join(
        f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{polar(i,1)[0]:.1f}" y2="{polar(i,1)[1]:.1f}" stroke="#CFD8DC" stroke-width="1"/>'
        for i in range(n)
    )

    # データ多角形
    pts_data = " ".join(f"{polar(i, values[i])[0]:.1f},{polar(i, values[i])[1]:.1f}" for i in range(n))

    # ラベル（★ 位置調整: ラベル距離拡大 + 下部ラベルの y オフセット強化）
    labels = ""
    for i, k in enumerate(keys):
        lx, ly = polar(i, 1.50)         # ★ 1.42 → 1.50 に拡大
        lbl = _SCORE_LABELS.get(k, k)
        val = values[i]
        color = "#B71C1C" if val >= 0.8 else "#1565C0"
        # ★ 下部のラベル（history=i3, interaction=i4）は y を大きく下げてクリッピング防止
        if i == 3:      # 発生頻度 (下部中央)
            _extra_y = 14
        elif i == 4:    # 複合危険度 (左下)
            _extra_y = 6
        elif i == 0:    # 意味的深刻度 (上部) — 上に少し寄せる
            _extra_y = -4
        else:
            _extra_y = 0
        labels += (
            f'<text x="{lx:.1f}" y="{ly + _extra_y:.1f}" font-size="12" text-anchor="middle" '
            f'fill="{color}" font-family="sans-serif">{lbl}</text>'
            f'<text x="{lx:.1f}" y="{ly + _extra_y + 15:.1f}" font-size="12" text-anchor="middle" '
            f'fill="{color}" font-family="sans-serif" font-weight="bold">{val:.2f}</text>'
        )

    return (
        f'<svg width="{size}" height="{size + 20}" xmlns="http://www.w3.org/2000/svg">'
        f'{grid_lines}{axes}'
        f'<polygon points="{pts_data}" fill="rgba(21,101,192,0.25)" stroke="#1565C0" stroke-width="1.5"/>'
        f'{labels}'
        f'</svg>'
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# メイン証拠パネル
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_explanation_panel(
    pred_item: Dict[str, Any],
    expanded: bool = False,
    dt_engine: Any = None,
) -> None:
    """
    予兆予測1件の explanation を証拠パネルとして描画する。

    Args:
        pred_item:  predict() が返す dict（explanation キーを含む）
        expanded:   expander の初期展開状態
        dt_engine:  DigitalTwinEngine（類似インシデント検索用・省略可）
    """
    explanation = pred_item.get("explanation") or {}
    if not explanation:
        return

    anomaly_type    = explanation.get("anomaly_type", "point")
    narrative       = explanation.get("narrative") or pred_item.get("llm_narrative", "")
    llm_error       = explanation.get("llm_error")
    score_breakdown = explanation.get("score_breakdown") or {}
    vendor_context  = explanation.get("vendor_context") or pred_item.get("vendor_context")

    atype_info  = ANOMALY_TYPE_LABELS.get(anomaly_type, ANOMALY_TYPE_LABELS["point"])
    badge_color = atype_info["color"]
    badge_label = atype_info["label"]

    with st.expander(f"🔬 AI証拠パネル  [{badge_label}]", expanded=expanded):

        # ── 異常種別バッジ ──
        st.markdown(
            f"<div style='margin-bottom:8px;'>"
            f"<span style='background:{badge_color};color:white;padding:3px 10px;"
            f"border-radius:12px;font-size:11px;font-weight:bold;'>"
            f"{badge_label}</span>"
            f"<span style='color:#757575;font-size:11px;margin-left:8px;'>"
            f"{atype_info['desc']}</span>"
            + (f"<br><span style='color:#9E9E9E;font-size:10px;'>"
               f"⚠️ LLM利用不可（フォールバックスコア）</span>" if llm_error else "")
            + "</div>",
            unsafe_allow_html=True,
        )

        # ── LLM narrative ──
        if narrative:
            st.info(f"💬 {narrative}")

        # ── ベンダーコンテキスト ──
        if vendor_context:
            st.caption(f"🔧 機器情報: {vendor_context}")

        # ── レーダーチャート（★拡大版: 440px）──
        if score_breakdown:
            svg = build_radar_svg(score_breakdown, size=440)
            st.markdown(svg, unsafe_allow_html=True)

        # ── 類似インシデント（ChromaDB があれば） ──
        alarm_text = pred_item.get("alarm_text", "")
        if dt_engine is not None and alarm_text:
            _render_similar_incidents(dt_engine, alarm_text)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 類似インシデント検索パネル
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_OUTCOME_BADGES = {
    "confirmed":   ("#B71C1C", "✅ 実障害(TP)"),
    "mitigated":   ("#1B5E20", "🛡️ 予防済(TP)"),
    "false_alarm": ("#E65100", "⚠️ 誤検知(FP)"),
    "pending":     ("#616161", "⏳ 評価中"),
}


def _render_similar_incidents(dt_engine: Any, alarm_text: str) -> None:
    """ChromaDB で類似インシデントを検索して表示する。"""
    try:
        vs = getattr(dt_engine, "vector_store", None)
        if vs is None:
            return
        results = vs.search_similar_alarms(alarm_text=alarm_text, n_results=5)
        if not results:
            return

        with st.expander(
            f"🔍 類似インシデント（{len(results)} 件 · ChromaDB）", expanded=False
        ):
            st.caption("意味的に近い過去の予兆履歴。類似度が高いほど構造が一致しています。")
            for r in results:
                color, badge = _OUTCOME_BADGES.get(
                    r.get("outcome", "pending"), _OUTCOME_BADGES["pending"]
                )
                sim_pct = int(r.get("similarity", 0) * 100)
                ts      = r.get("created_at", 0)
                ts_str  = (
                    time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "不明"
                )
                vc = r.get("vendor_context", "")
                text = r.get("text", "")
                st.markdown(
                    f"<div style='border-left:3px solid {color};padding:6px 10px;"
                    f"margin-bottom:6px;background:#FAFAFA;border-radius:0 4px 4px 0;font-size:12px;'>"
                    f"<span style='background:{color};color:white;padding:1px 6px;"
                    f"border-radius:8px;font-size:11px;'>{badge}</span>"
                    f"<span style='color:#616161;margin-left:8px;'>類似度: <b>{sim_pct}%</b></span>"
                    f"<span style='color:#9E9E9E;margin-left:8px;font-size:11px;'>{ts_str}</span>"
                    + (f"<br><span style='color:#78909C;font-size:11px;'>🔧 {vc}</span>" if vc else "")
                    + f"<br><span style='color:#424242;margin-top:3px;display:block;'>"
                    f"{text[:120]}{'...' if len(text) > 120 else ''}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug(f"_render_similar_incidents: {e}")
