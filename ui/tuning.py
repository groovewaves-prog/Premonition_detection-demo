import streamlit as st
import pandas as pd
import sqlite3
import os
import json

from registry import get_paths, load_topology, get_display_name


def _get_or_init_dt_engine(site_id: str):
    """
    Digital Twin Engine を取得または初期化する。

    【根本原因と対策】
    リファクタリング後、tuning.py は `engine.digital_twin` を参照しているが、
    LogicalRCA には digital_twin 属性が存在しない。

    対策:
    1. st.session_state に直接 dt_engine を保存するキーを用意する
    2. 未初期化なら DigitalTwinEngine をここで生成してキャッシュする
    3. cockpit.py 側でも同じキーで保存することで共有できる
    """
    dt_key = f"dt_engine_{site_id}"

    # すでに初期化済みならそのまま返す
    if st.session_state.get(dt_key) is not None:
        return st.session_state[dt_key]

    # digital_twin.py から DigitalTwinEngine をインポートして初期化
    try:
        from digital_twin import DigitalTwinEngine

        paths = get_paths(site_id)
        topology = load_topology(paths.topology_path)
        if not topology:
            return None

        # 子ノードマップを構築
        children_map = {}
        for node_id, node in topology.items():
            parent_id = (node.get('parent_id') if isinstance(node, dict)
                         else getattr(node, 'parent_id', None))
            if parent_id:
                children_map.setdefault(parent_id, []).append(node_id)

        dt_engine = DigitalTwinEngine(
            topology=topology,
            children_map=children_map,
            tenant_id=site_id
        )
        st.session_state[dt_key] = dt_engine
        return dt_engine

    except ImportError:
        return None
    except Exception as e:
        st.session_state[dt_key] = None  # 初期化失敗を記録（無限リトライ防止）
        return None


def render_tuning_dashboard(site_id: str):
    st.subheader("🔧 Digital Twin Tuning & Audit")

    # ── Digital Twin Engine 取得 ──────────────────────────
    dt_engine = _get_or_init_dt_engine(site_id)

    if not dt_engine:
        st.warning(
            "⚠️ Digital Twin Engine が初期化できませんでした。\n\n"
            "**考えられる原因:**\n"
            "- `digital_twin.py` がプロジェクトルートに存在しない\n"
            "- `DigitalTwinEngine.__init__()` で例外が発生した\n\n"
            "コックピット画面でシナリオを選択してからもう一度お試しください。"
        )
        return

    display_name = get_display_name(site_id)
    st.caption(f"対象拠点: **{display_name}** | テナントID: `{site_id}`")

    tab1, tab2, tab3 = st.tabs(["⚡ Auto-Tuning", "📜 Audit Log", "🛑 Maintenance"])

    # ── Tab1: Auto-Tuning ────────────────────────────────
    with tab1:
        st.caption("AIによる閾値自動調整の提案を確認し、適用します。")

        if st.button("🔄 提案を生成 (Generate)", key="tuning_gen"):
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
                rec = p.get('apply_recommendation', {})
                with st.expander(f"📦 {rule_pattern} ({rec.get('apply_mode', '-')})", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    stats = p.get('current_stats', {})
                    proposal = p.get('proposal', {})
                    impact = p.get('expected_impact', {})
                    c1.metric("Recall (再現率)", f"{stats.get('recall', 0):.2f}")
                    c2.metric("New Threshold",   f"{proposal.get('paging_threshold', 0):.2f}")
                    c3.metric("FP Reduction",    f"-{impact.get('fp_reduction', 0)*100:.0f}%",
                              delta_color="normal")
                    st.markdown(f"**理由:** {rec.get('shadow_note', '-')}")
                    if rec.get('apply_mode') == 'auto':
                        st.success("✅ Auto-Eligible (推奨)")
                    if st.button(f"承認して適用 (Apply)", key=f"ap_{rule_pattern}"):
                        try:
                            res = dt_engine.apply_tuning_proposals_if_auto([p])
                            if res.get('applied'):
                                st.success(f"適用完了: {res['applied']}")
                            else:
                                st.error(f"適用失敗/スキップ: {res.get('skipped', [])}")
                        except Exception as e:
                            st.error(f"適用エラー: {e}")
        else:
            st.info("現在、適用すべき新しい提案はありません。")

    # ── Tab2: Audit Log ──────────────────────────────────
    with tab2:
        st.caption("システムに加えられた変更の監査ログ（SQLite）を表示します。")

        # ★ 修正: dt_engine.storage.paths → dt_engine.paths（正しい属性名）
        db_path = dt_engine.paths.get("sqlite_db", "")

        if db_path and os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                df = pd.read_sql(
                    "SELECT timestamp, event_type, actor, rule_pattern, status "
                    "FROM audit_log ORDER BY timestamp DESC LIMIT 50",
                    conn
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

    # ── Tab3: Maintenance ────────────────────────────────
    with tab3:
        st.markdown("#### System Maintenance")

        col_m1, col_m2 = st.columns(2)

        with col_m1:
            if st.button("🚑 DB Repair (Self-Healing)", key="dt_repair"):
                try:
                    # repair_rule_config が存在する場合
                    if hasattr(dt_engine, 'repair_rule_config'):
                        result = dt_engine.repair_rule_config()
                    elif hasattr(dt_engine, 'repair_db_from_rules_json'):
                        result = dt_engine.repair_db_from_rules_json()
                    else:
                        result = False
                    if result:
                        st.success("DBを rules.json から復元しました。")
                    else:
                        st.error("復元に失敗しました（メソッドが利用できないか、対象データがありません）。")
                except Exception as e:
                    st.error(f"DB修復エラー: {e}")

        with col_m2:
            if st.button("🧹 Cache Clear", key="dt_cache_clear"):
                st.cache_data.clear()
                st.cache_resource.clear()
                # dt_engine キャッシュもリセット
                dt_key = f"dt_engine_{site_id}"
                if dt_key in st.session_state:
                    del st.session_state[dt_key]
                st.success("キャッシュをクリアしました。次回アクセス時に再初期化されます。")

        st.divider()
        st.markdown("#### 📊 Engine Status")
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("ルール数",   len(getattr(dt_engine, 'rules', [])))
        col_s2.metric("履歴件数",   len(getattr(dt_engine, 'history', [])))
        col_s3.metric("アウトカム", len(getattr(dt_engine, 'outcomes', [])))
