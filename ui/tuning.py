"""
ui/tuning.py  ―  Streamlit UI 層（Digital Twin Tuning ダッシュボード）
"""
import streamlit as st
import pandas as pd
import sqlite3
import os
import time
import hashlib

# ==========================================
# ★ 追加: Streamlitグローバルキャッシュによるエンジンの保持
# ==========================================

def _compute_topo_hash(topology: dict) -> str:
    """トポロジーの構成変更を検知するための軽量ハッシュを計算"""
    try:
        keys = sorted(list(topology.keys()))
        state = []
        for k in keys:
            node = topology[k]
            pid = node.get('parent_id') if isinstance(node, dict) else getattr(node, 'parent_id', None)
            rg = node.get('redundancy_group') if isinstance(node, dict) else getattr(node, 'redundancy_group', None)
            state.append(f"{k}|{pid}|{rg}")
        return hashlib.md5(",".join(state).encode()).hexdigest()
    except Exception:
        return str(time.time())

@st.cache_resource(show_spinner="AI推論エンジンを初期化中...（初回のみ数秒かかります）")
def _get_cached_engine(site_id: str, topo_hash: str):  # ★修正: topo_hash を追加
    from digital_twin_pkg import DigitalTwinEngine
    from registry import get_paths, load_topology

    paths = get_paths(site_id)
    topology = load_topology(paths.topology_path)
    if not topology:
        return None

    children_map: dict = {}
    for node_id, node in topology.items():
        parent_id = (node.get('parent_id') if isinstance(node, dict)
                     else getattr(node, 'parent_id', None))
        if parent_id:
            children_map.setdefault(parent_id, []).append(node_id)
            
    # ★ 修正: 古い paths.data_dir を削除し、最新の引数仕様（キーワード引数）に合わせる
    engine = DigitalTwinEngine(
        topology=topology,
        children_map=children_map,
        tenant_id=site_id
    )
    return engine


def _get_or_init_dt_engine(site_id: str):
    """
    キャッシュされたエンジンを取得し、session_state にポインタを渡す（既存UIとの互換性維持）。
    """
    dt_key    = f"dt_engine_{site_id}"
    err_key   = f"dt_engine_error_{site_id}"

    # すでにセッションに存在すればそれを返す（高速）
    if dt_key in st.session_state and st.session_state[dt_key] is not None:
        return st.session_state[dt_key]

    try:
        # ★修正: ロードしたトポロジからハッシュを計算して渡す
        from registry import get_paths, load_topology
        paths = get_paths(site_id)
        current_topo = load_topology(paths.topology_path)
        if current_topo:
            current_topo_hash = _compute_topo_hash(current_topo)
        else:
            current_topo_hash = "empty"

        engine = _get_cached_engine(site_id, current_topo_hash)
        
        if not engine:
            st.session_state[dt_key]  = None
            st.session_state[err_key] = "topology が読み込めませんでした。"
            return None
            
        # 取得したエンジンをセッションにセット
        st.session_state[dt_key] = engine
        st.session_state[err_key] = None
        return engine
        
    except Exception as e:
        import traceback
        st.session_state[dt_key]  = None
        st.session_state[err_key] = f"{e}\n{traceback.format_exc()}"
        return None

def render_tuning_dashboard(site_id: str):
    st.subheader("🔧 Digital Twin Tuning & Audit")

    dt_engine = _get_or_init_dt_engine(site_id)

    if not dt_engine:
        err_detail = st.session_state.get(f"dt_engine_error_{site_id}", "不明なエラー")
        st.error("Digital Twin Engine unavailable. (エンジンモジュールがロードされていません)")
        with st.expander("🔍 エラー詳細（デバッグ用）", expanded=True):
            st.code(err_detail or "詳細情報なし", language="text")

        col_retry, _ = st.columns([1, 3])
        if col_retry.button("🔄 再試行"):
            for k in [f"dt_engine_{site_id}", f"dt_engine_error_{site_id}"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()
        return

    try:
        from registry import get_display_name
        display_name = get_display_name(site_id)
    except Exception:
        display_name = site_id
    st.caption(f"対象拠点: **{display_name}** | テナントID: `{site_id}`")

    tab1, tab2, tab3 = st.tabs(["⚡ Auto-Tuning", "📜 Audit Log", "🛑 Maintenance"])

    # ── Tab1: Auto-Tuning ──────────────────────────────────
    with tab1:
        st.caption("AIによる閾値自動調整の提案を確認し、適用します。")

        col1, _ = st.columns([1, 3])
        if col1.button("🔄 提案を生成 (Generate)"):
            with st.spinner("Analyzing prediction history..."):
                try:
                    report = dt_engine.generate_tuning_report(days=30)
                    st.session_state["tuning_report"] = report
                except Exception as e:
                    st.error(f"レポート生成エラー: {e}")

        report = st.session_state.get("tuning_report")
        if report and report.get("tuning_proposals"):
            for p in report["tuning_proposals"]:
                rule_pattern = p.get('rule_pattern', '不明')
                rec          = p.get('apply_recommendation', {})
                stats        = p.get('current_stats', {})
                proposal     = p.get('proposal', {})
                impact       = p.get('expected_impact', {})

                with st.expander(f"📦 {rule_pattern} ({rec.get('apply_mode', '-')})", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Recall (再現率)", f"{stats.get('recall', 0):.2f}")
                    c2.metric("New Threshold",   f"{proposal.get('paging_threshold', 0):.2f}")
                    c3.metric("FP Reduction",    f"-{impact.get('fp_reduction', 0)*100:.0f}%",
                              delta_color="normal")
                    st.markdown(f"**理由:** {rec.get('shadow_note', '-')}")
                    if rec.get('apply_mode') == 'auto':
                        st.success("✅ Auto-Eligible (推奨)")
                    if st.button("承認して適用 (Apply)", key=f"ap_{rule_pattern}"):
                        try:
                            res = dt_engine.apply_tuning_proposals_if_auto([p])
                            if res.get('applied'):
                                st.success(f"適用完了: {res['applied']}")
                            else:
                                st.error(f"適用失敗/スキップ: {res.get('skipped', [])}")
                        except Exception as e:
                            st.error(f"適用エラー: {e}")
        else:
            # ★ 統計情報を表示して、なぜ提案がないかを説明
            if report and report.get("stats"):
                stats = report["stats"]
                st.info(
                    f"📊 **分析結果**: 過去 **{stats['scan_days']}日間** の予兆履歴を分析しました。\n\n"
                    f"- 予兆予測総数: **{stats['total_predictions']}件**\n"
                    f"- 評価済み（outcome付き）: **{stats['total_with_outcome']}件**\n"
                    f"- ルールパターン数: **{stats['rules_analyzed']}種類**\n"
                    f"- 提案生成に必要な最低サンプル数: **{stats['min_samples_required']}件/ルール**\n\n"
                    "💡 提案が生成されるには、同一ルールパターンで最低"
                    f"**{stats['min_samples_required']}件以上**のラベル付きアウトカム"
                    "（TP: 実障害確定 / FP: 誤検知 / FN: 検知漏れ）が必要です。\n\n"
                    "**次のステップ:**\n"
                    "1. 予兆シミュレーションを複数回実行してください\n"
                    "2. 障害シナリオで予兆が的中したか確認してください\n"
                    "3. 十分なデータが蓄積されると、自動で提案が生成されます"
                )
            else:
                st.info("現在、適用すべき新しい提案はありません。\n\n"
                       "「🔄 提案を生成 (Generate)」ボタンを押して分析を開始してください。")

    # ── Tab2: Audit Log ────────────────────────────────────
    with tab2:
        st.caption("システムに加えられた変更の監査ログ（SQLite）を表示します。")
        db_path = dt_engine.storage.paths.get("sqlite_db", "")

        if db_path and os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                df = pd.read_sql(
                    "SELECT timestamp, event_type, actor, rule_pattern, status "
                    "FROM audit_log ORDER BY timestamp DESC LIMIT 50",
                    conn,
                )
                conn.close()
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("監査ログはまだありません。")
            except Exception as e:
                st.error(f"ログ読み込みエラー: {e}")
        else:
            st.warning(f"監査データベースが見つかりません。\n\nパス: `{db_path}`")

    # ── Tab3: Maintenance ──────────────────────────────────
    with tab3:
        st.markdown("#### System Maintenance")
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            if st.button("🚑 DB Repair (Self-Healing)"):
                try:
                    if dt_engine.repair_db_from_rules_json():
                        st.success("DBを rules.json から復元しました。")
                    else:
                        st.error("復元に失敗しました。rules.json が存在しない可能性があります。")
                except Exception as e:
                    st.error(f"DB修復エラー: {e}")

        with col_m2:
            if st.button("🧹 Cache Clear"):
                # グローバルキャッシュ（@st.cache_resource）をクリア
                st.cache_resource.clear()
                st.cache_data.clear()
                
                # セッションステートのポインタもクリア
                for k in [f"dt_engine_{site_id}", f"dt_engine_error_{site_id}"]:
                    if k in st.session_state:
                        del st.session_state[k]
                        
                st.success("キャッシュを完全にクリアしました。次回アクセス時にエンジンが再起動・再初期化されます。")
                time.sleep(1.5)
                st.rerun()

        st.divider()
        st.markdown("#### 📊 Engine Status")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("ルール数",   len(getattr(dt_engine, 'rules',   [])))
        col_s2.metric("履歴件数",   len(getattr(dt_engine, 'history', [])))
        col_s3.metric("アウトカム", len(getattr(dt_engine, 'outcomes', [])))
        _llm_name = getattr(getattr(dt_engine, 'llm', None), 'backend_name', '未初期化')
        col_s4.metric("LLMバックエンド", _llm_name.split("(")[0].strip())
        st.caption(f"🤖 {_llm_name}")
