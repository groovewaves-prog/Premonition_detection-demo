import streamlit as st
import graphviz
import os
import google.generativeai as genai

from data import TOPOLOGY
from logic import CausalInferenceEngine, Alarm
# 修正したモジュールをインポート
from network_ops import run_diagnostic_simulation

st.set_page_config(page_title="Antigravity Live", page_icon="⚡", layout="wide")

# --- トポロジー描画 (変更なし) ---
def render_topology(alarms, root_cause_node):
    graph = graphviz.Digraph()
    graph.attr(rankdir='TB')
    graph.attr('node', shape='box', style='rounded,filled', fontname='Helvetica')
    alarmed_ids = {a.device_id for a in alarms}
    for node_id, node in TOPOLOGY.items():
        color = "#e8f5e9"
        penwidth = "1"
        if root_cause_node and node_id == root_cause_node.id:
            color = "#ffcdd2"
            penwidth = "3"
        elif node_id in alarmed_ids:
            color = "#fff9c4"
        graph.node(node_id, label=f"{node_id}\n({node.type})", fillcolor=color, color='black', penwidth=penwidth)
    for node_id, node in TOPOLOGY.items():
        if node.parent_id:
            graph.edge(node.parent_id, node_id)
            parent = TOPOLOGY.get(node.parent_id)
            if parent and parent.redundancy_group:
                partners = [n.id for n in TOPOLOGY.values() if n.redundancy_group == parent.redundancy_group and n.id != parent.id]
                for p in partners: graph.edge(p, node_id)
    return graph

# --- Config読み込み (変更なし) ---
def load_config_by_id(device_id):
    path = f"configs/{device_id}.txt"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f: return f.read()
    return None

# --- UI構築 ---
st.title("⚡ Antigravity AI Agent (Live Demo)")

api_key = None
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    api_key = os.environ.get("GOOGLE_API_KEY")

with st.sidebar:
    st.header("⚡ 運用モード選択")
    # 選択肢
    selected_scenario = st.radio(
        "シナリオ:", 
        ("正常稼働", "1. WAN全回線断", "2. FW片系障害", "3. L2SWサイレント障害", "4. [Live] Cisco実機診断")
    )
    if not api_key:
        st.warning("API Key Missing")
        user_key = st.text_input("Google API Key", type="password")
        if user_key: api_key = user_key

if "current_scenario" not in st.session_state:
    st.session_state.current_scenario = "正常稼働"
    st.session_state.messages = []
    st.session_state.chat_session = None 
    st.session_state.live_result = None

if st.session_state.current_scenario != selected_scenario:
    st.session_state.current_scenario = selected_scenario
    st.session_state.messages = []
    st.session_state.chat_session = None
    st.session_state.live_result = None
    st.rerun()

# --- Liveモード判定 ---
# 4番だけでなく、全シナリオで「自律調査」ボタンを使えるようにレイアウトを統合します
is_live_mode = selected_scenario == "4. [Live] Cisco実機診断"

# --- アラーム生成 (シミュレーション用) ---
alarms = []
if selected_scenario == "1. WAN全回線断":
    alarms = [Alarm("WAN_ROUTER_01", "Down", "CRITICAL"), Alarm("AP_01", "Unreach", "CRITICAL")]
elif selected_scenario == "2. FW片系障害":
    alarms = [Alarm("FW_01_PRIMARY", "HB Loss", "WARNING")]
elif selected_scenario == "3. L2SWサイレント障害":
    alarms = [Alarm("AP_01", "Lost", "CRITICAL"), Alarm("AP_02", "Lost", "CRITICAL")]

root_cause = None
reason = ""
if alarms:
    engine = CausalInferenceEngine(TOPOLOGY)
    res = engine.analyze_alarms(alarms)
    root_cause = res.root_cause_node
    reason = res.root_cause_reason

# --- メイン画面レイアウト ---
col1, col2 = st.columns([1, 1])

# 左カラム：トポロジー ＆ 実機調査ボタン
with col1:
    st.subheader("Network Status")
    st.graphviz_chart(render_topology(alarms, root_cause), use_container_width=True)
    
    if root_cause or is_live_mode:
        if root_cause:
            st.markdown(f'<div style="color:#d32f2f;background:#fdecea;padding:10px;border-radius:5px;">🚨 緊急アラート：{root_cause.id} ダウン</div>', unsafe_allow_html=True)
            st.caption(f"理由: {reason}")
        
        st.markdown("---")
        st.info("🛠 **自律調査エージェント**")
        st.markdown("SSH接続による詳細診断を実行します。")
        
        # 自律調査ボタン
        if st.session_state.live_result:
            res = st.session_state.live_result
            if res["status"] == "SUCCESS":
                # ★追加：セキュリティバナー
                st.success("🛡️ **Data Sanitized**: パスワード・IPアドレスをマスク処理しました。")
                
                with st.expander("取得ログ確認", expanded=True):
                    st.code(res["sanitized_log"], language="text")
            else:
                st.error(f"診断結果: {res['error']}")

        # 診断結果（ログまたはエラー）の表示
        if st.session_state.live_result:
            res = st.session_state.live_result
            if res["status"] == "SUCCESS":
                
                # ★ここを追加：セキュリティアピール用のバナー
                st.success("🛡️ **Security Filter Active**: 機密情報（IP, Password）はマスク処理後にAIへ送信されます。")
                
                with st.expander("📄 取得ログ (Sanitized View)", expanded=True):
                    # ログの中身を表示（<HIDDEN>が含まれていることをチラ見せする）
                    st.code(res["sanitized_log"], language="text")
            else:
                st.error(f"診断結果: {res['error']}")

# 右カラム：AIチャット
with col2:
    st.subheader("AI Analyst Report")
    
    # Live結果がある場合、それをプロンプトに含める
    if st.session_state.live_result:
        live_data = st.session_state.live_result
        
        # チャット初期化
        if st.session_state.chat_session is None:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash", generation_config={"temperature": 0.0})
            
            # プロンプトの【出力要件】を書き換え
            system_prompt = f"""
            あなたはネットワークエンジニアです。以下の診断結果に基づき、トラブルシューティングの経緯を報告してください。

            【診断入力データ】
            ステータス: {live_data['status']}
            詳細/ログ: {live_data.get('sanitized_log') or live_data.get('error')}
            推論された原因: {reason if reason else "実機調査モード"}

            【出力要件】
            以下のフォーマットで出力すること。
            
            ### 🛠 ネクストアクション実行レポート
            
            **1. データ保全と接続確認:**
            接続試行およびログ取得を実施。
            → **結果: {live_data['status']}** (🛡️ 機密情報はフィルタリング済み)
            
            **2. 詳細分析:**
            [接続できた場合はログ内容（Config/Interface）の分析、エラーの場合は要因推測]
            → [分析結果]
            
            **3. 物理/インターフェース確認:**
            [状況に応じた推論]
            → [分析結果]
            
            ---
            **最終判定:** [結論]
            """
            
            history = [{"role": "user", "parts": [system_prompt]}]
            chat = model.start_chat(history=history)
            
            with st.spinner("Gemini is analyzing diagnostic data..."):
                try:
                    response = chat.send_message("レポートを作成してください。")
                    st.session_state.chat_session = chat
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(str(e))

    # チャットUI
    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
    
    if prompt := st.chat_input("質問..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"): st.markdown(prompt)
        if st.session_state.chat_session:
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        res = st.session_state.chat_session.send_message(prompt)
                        st.markdown(res.text)
                        st.session_state.messages.append({"role": "assistant", "content": res.text})
