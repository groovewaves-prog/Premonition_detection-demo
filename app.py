import streamlit as st
import graphviz
import os
import google.generativeai as genai

from data import TOPOLOGY
from logic import CausalInferenceEngine, Alarm, simulate_cascade_failure
from network_ops import run_diagnostic_simulation

st.set_page_config(page_title="Antigravity Live", page_icon="⚡", layout="wide")

# --- トポロジー描画 ---
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

# --- Config読み込み (修正済み) ---
def load_config_by_id(device_id):
    path = f"configs/{device_id}.txt"
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return None
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
    st.session_state.trigger_analysis = False

if st.session_state.current_scenario != selected_scenario:
    st.session_state.current_scenario = selected_scenario
    st.session_state.messages = []
    st.session_state.chat_session = None
    st.session_state.live_result = None
    st.session_state.trigger_analysis = False
    st.rerun()

# --- アラーム生成 ---
alarms = []
if selected_scenario == "1. WAN全回線断":
    alarms = simulate_cascade_failure("WAN_ROUTER_01", TOPOLOGY)
elif selected_scenario == "2. FW片系障害":
    alarms = [Alarm("FW_01_PRIMARY", "Heartbeat Loss", "WARNING")]
elif selected_scenario == "3. L2SWサイレント障害":
    alarms = [Alarm("AP_01", "Connection Lost", "CRITICAL"), Alarm("AP_02", "Connection Lost", "CRITICAL")]

root_cause = None
reason = ""
if alarms:
    engine = CausalInferenceEngine(TOPOLOGY)
    res = engine.analyze_alarms(alarms)
    root_cause = res.root_cause_node
    reason = res.root_cause_reason

# --- 画面レイアウト ---
col1, col2 = st.columns([1, 1])

# 左カラム：トポロジー & 診断実行
with col1:
    st.subheader("Network Status")
    st.graphviz_chart(render_topology(alarms, root_cause), use_container_width=True)
    
    if root_cause:
        st.markdown(f'<div style="color:#d32f2f;background:#fdecea;padding:10px;border-radius:5px;">🚨 緊急アラート：{root_cause.id} ダウン</div>', unsafe_allow_html=True)
        st.caption(f"理由: {reason}")
    
    is_live_mode = (selected_scenario == "4. [Live] Cisco実機診断")
    
    if is_live_mode or root_cause:
        st.markdown("---")
        st.info("🛠 **自律調査エージェント**")
        
        if st.button("🚀 診断実行 (Auto-Diagnostic)", type="primary"):
            if not api_key:
                st.error("API Key Required")
            else:
                with st.status("Agent Operating...", expanded=True) as status:
                    st.write("🔌 Establishing Connection...")
                    res = run_diagnostic_simulation(selected_scenario)
                    st.session_state.live_result = res
                    
                    if res["status"] == "SUCCESS":
                        st.write("✅ Data Acquired.")
                        st.write("🧹 Sanitizing Sensitive Information...")
                        status.update(label="Complete!", state="complete", expanded=False)
                    else:
                        st.write("❌ Connection Failed.")
                        status.update(label="Target Unreachable", state="error", expanded=False)
                    
                    st.session_state.trigger_analysis = True
                    st.rerun()

        # 診断結果表示（タブ形式）
        if st.session_state.live_result:
            res = st.session_state.live_result
            if res["status"] == "SUCCESS":
                st.success("🛡️ **Data Sanitized**: 機密情報はマスク処理済み")
                
                # タブで生ログ（サニタイズ済）を見やすく表示
                tab_log, tab_raw = st.tabs(["🔒 Sanitized Log", "🔍 Raw (Debug)"])
                with tab_log:
                    st.code(res["sanitized_log"], language="text")
                with tab_raw:
                    st.warning("管理権限が必要です")
            else:
                st.error(f"診断結果: {res['error']}")

# 右カラム：AIチャット
with col2:
    st.subheader("AI Analyst Report")
    if not api_key: st.stop()

    # 初期化
    should_start_chat = (st.session_state.chat_session is None) and (selected_scenario != "正常稼働")
    if should_start_chat:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash", generation_config={"temperature": 0.0})
        
        system_prompt = ""
        if st.session_state.live_result:
            # Liveモードの初期化
            live_data = st.session_state.live_result
            log_content = live_data.get('sanitized_log') or f"Error: {live_data.get('error')}"
            system_prompt = f"診断結果に基づきレポートを作成せよ。\nステータス: {live_data['status']}\nログ: {log_content}"
        elif root_cause:
            conf = load_config_by_id(root_cause.id)
            system_prompt = f"障害報告: {root_cause.id}。理由: {reason}。"
            if conf: system_prompt += f"\nConfig:\n{conf}"
        
        if system_prompt:
            chat = model.start_chat(history=[{"role": "user", "parts": [system_prompt]}])
            try:
                with st.spinner("Analyzing..."):
                    res = chat.send_message("状況報告をお願いします。")
                    st.session_state.chat_session = chat
                    st.session_state.messages.append({"role": "assistant", "content": res.text})
            except Exception as e: st.error(str(e))

    # 診断後の追加分析 (トリガー)
    if st.session_state.trigger_analysis and st.session_state.chat_session:
        live_data = st.session_state.live_result
        log_content = live_data.get('sanitized_log') or f"Error: {live_data.get('error')}"
        
        prompt = f"""
        診断コマンドを実行しました。以下の結果に基づき『ネクストアクション実行レポート』を作成してください。
        
        【診断データ】
        ステータス: {live_data['status']}
        ログ: {log_content}
        
        【出力要件】
        1. 接続結果 (成功/失敗)
        2. ログ分析 (インターフェース状態、ルート情報など)
        3. 推奨アクション
        """
        st.session_state.messages.append({"role": "user", "content": "診断結果を分析してください。"})
        
        with st.spinner("Analyzing Diagnostic Data..."):
            try:
                res = st.session_state.chat_session.send_message(prompt)
                st.session_state.messages.append({"role": "assistant", "content": res.text})
            except Exception as e: st.error(str(e))
        
        st.session_state.trigger_analysis = False
        st.rerun()

    # チャットUI
    chat_container = st.container(height=600)
    with chat_container:
        for msg in st.session_state.messages:
            if "診断結果に基づき" in msg["content"]: continue
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
