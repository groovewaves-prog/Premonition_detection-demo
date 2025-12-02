import streamlit as st
import graphviz
import os
import google.generativeai as genai

from data import TOPOLOGY
from logic import CausalInferenceEngine, Alarm, simulate_cascade_failure
from network_ops import run_diagnostic_simulation

# --- ページ設定 ---
st.set_page_config(page_title="Antigravity Live", page_icon="⚡", layout="wide")

# --- 関数: トポロジー図の生成 ---
def render_topology(alarms, root_cause_node, root_severity="CRITICAL"):
    graph = graphviz.Digraph()
    graph.attr(rankdir='TB')
    graph.attr('node', shape='box', style='rounded,filled', fontname='Helvetica')
    
    alarmed_ids = {a.device_id for a in alarms}
    
    for node_id, node in TOPOLOGY.items():
        color = "#e8f5e9" # Default Green
        penwidth = "1"
        fontcolor = "black"
        label = f"{node_id}\n({node.type})"
        
        # 根本原因の強調 (エンジンの判定したSeverityに基づく)
        if root_cause_node and node_id == root_cause_node.id:
            if root_severity == "CRITICAL":
                color = "#ffcdd2" # Red (Down)
            elif root_severity == "WARNING":
                color = "#fff9c4" # Yellow (Warning)
            else:
                color = "#e8f5e9" # Green (Info)
            
            penwidth = "3"
            label += "\n[ROOT CAUSE]"
            
        elif node_id in alarmed_ids:
            color = "#fff9c4" # 連鎖アラームは黄色
        
        graph.node(node_id, label=label, fillcolor=color, color='black', penwidth=penwidth, fontcolor=fontcolor)
    
    for node_id, node in TOPOLOGY.items():
        if node.parent_id:
            graph.edge(node.parent_id, node_id)
            parent_node = TOPOLOGY.get(node.parent_id)
            if parent_node and parent_node.redundancy_group:
                partners = [n.id for n in TOPOLOGY.values() 
                           if n.redundancy_group == parent_node.redundancy_group and n.id != parent_node.id]
                for partner_id in partners:
                    graph.edge(partner_id, node_id)
    return graph

# --- 関数: Config自動読み込み ---
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
    
    scenario_options = (
        "正常稼働",
        "--- [広域障害] ---",
        "1. WAN全回線断",
        "2. FW片系障害",
        "3. L2SWサイレント障害",
        "--- [WAN Router] ---",
        "4. [WAN] BGPルートフラッピング",
        "5. [WAN] FAN故障",
        "6. [WAN] 電源故障",
        "7. [WAN] メモリリーク",
        "--- [Firewall] ---",
        "8. [FW] FAN故障",
        "9. [FW] 電源故障",
        "10. [FW] メモリリーク",
        "--- [L2 Switch] ---",
        "11. [L2SW] FAN故障",
        "12. [L2SW] 電源故障",
        "13. [L2SW] メモリリーク",
        "--- [Live Mode] ---",
        "99. [Live] Cisco実機診断"
    )
    selected_scenario = st.radio("シナリオを選択:", scenario_options)
    
    st.markdown("---")
    if api_key:
        st.success("API Connected")
    else:
        st.warning("API Key Missing")
        user_key = st.text_input("Google API Key", type="password")
        if user_key: api_key = user_key

if "---" in selected_scenario:
    st.warning("カテゴリ見出しです。具体的なシナリオを選択してください。")
    st.stop()

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

# --- アラーム生成 (実践的ロジックへの入力データ) ---
alarms = []

if "WAN全回線断" in selected_scenario:
    alarms = simulate_cascade_failure("WAN_ROUTER_01", TOPOLOGY)
elif "FW片系障害" in selected_scenario:
    alarms = [Alarm("FW_01_PRIMARY", "Heartbeat Loss", "WARNING")]
elif "L2SWサイレント障害" in selected_scenario:
    alarms = [Alarm("AP_01", "Connection Lost", "CRITICAL"), Alarm("AP_02", "Connection Lost", "CRITICAL")]

# ★改善点: UI側でSeverityを決め打ちせず、純粋なアラーム情報だけを作る
# ロジックエンジンがこのSeverityを見て、トポロジーの色を決定する
elif "[WAN]" in selected_scenario:
    msg = "Device Warning"
    sev = "WARNING" # FAN故障等はWarning
    if "BGP" in selected_scenario: msg = "BGP Flapping"
    elif "FAN" in selected_scenario: msg = "Fan Status: Fail"
    elif "電源" in selected_scenario: msg = "Power Supply: Redundant Loss"
    elif "メモリ" in selected_scenario: msg = "Memory High (90%)"
    alarms = [Alarm("WAN_ROUTER_01", msg, sev)]

elif "[FW]" in selected_scenario:
    # 同様に、FW単体の異常もWarningとして扱う
    alarms = [Alarm("FW_01_PRIMARY", "Hardware Alert", "WARNING")]

elif "[L2SW]" in selected_scenario:
    alarms = [Alarm("L2_SW_01", "Hardware Alert", "WARNING")]

root_cause = None
inference_result = None
reason = ""
root_severity = "CRITICAL" # デフォルト

if alarms:
    engine = CausalInferenceEngine(TOPOLOGY)
    inference_result = engine.analyze_alarms(alarms)
    root_cause = inference_result.root_cause_node
    reason = inference_result.root_cause_reason
    # ★重要: エンジンが判定したSeverityを取得する
    root_severity = inference_result.severity

# --- メイン画面 ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Network Status")
    # ロジック判定結果のSeverityを渡す
    st.graphviz_chart(render_topology(alarms, root_cause, root_severity), use_container_width=True)
    
    if root_cause:
        if root_severity == "CRITICAL":
            st.markdown(f'<div style="color:#d32f2f;background:#fdecea;padding:10px;border-radius:5px;">🚨 緊急アラート：{root_cause.id} ダウン</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="color:#856404;background:#fff3cd;padding:10px;border-radius:5px;">⚠️ 警告：{root_cause.id} 異常検知 (稼働中)</div>', unsafe_allow_html=True)
        
        st.caption(f"理由: {reason}")
    
    is_live_mode = ("[Live]" in selected_scenario)
    
    if is_live_mode or root_cause:
        st.markdown("---")
        st.info("🛠 **自律調査エージェント**")
        
        if st.button("🚀 診断実行 (Auto-Diagnostic)", type="primary"):
            if not api_key:
                st.error("API Key Required")
            else:
                with st.status("Agent Operating...", expanded=True) as status:
                    st.write("🔌 Executing Diagnostics...")
                    res = run_diagnostic_simulation(selected_scenario, api_key)
                    st.session_state.live_result = res
                    
                    if res["status"] == "SUCCESS":
                        st.write("✅ Data Acquired.")
                        st.write("🧹 Sanitizing...")
                        status.update(label="Complete!", state="complete", expanded=False)
                    elif res["status"] == "SKIPPED":
                        st.warning("No action needed.")
                        status.update(label="Skipped", state="complete")
                    else:
                        st.write("❌ Check Failed.")
                        status.update(label="Target Unreachable", state="error", expanded=False)
                    
                    st.session_state.trigger_analysis = True
                    st.rerun()

        if st.session_state.live_result:
            res = st.session_state.live_result
            if res["status"] == "SUCCESS":
                st.success("🛡️ **Data Sanitized**: 機密情報はマスク処理済み")
                with st.expander("📄 取得ログ (Sanitized)", expanded=True):
                    st.code(res["sanitized_log"], language="text")
            elif res["status"] == "ERROR":
                st.error(f"診断結果: {res['error']}")

with col2:
    st.subheader("AI Analyst Report")
    if not api_key: st.stop()

    should_start_chat = (st.session_state.chat_session is None) and (selected_scenario != "正常稼働")
    
    if should_start_chat:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash", generation_config={"temperature": 0.0})
        
        system_prompt = ""
        if st.session_state.live_result:
            live_data = st.session_state.live_result
            log_content = live_data.get('sanitized_log') or f"Error: {live_data.get('error')}"
            system_prompt = f"診断結果に基づきレポートを作成せよ。\nステータス: {live_data['status']}\nログ: {log_content}"
        elif root_cause:
            conf = load_config_by_id(root_cause.id)
            system_prompt = f"障害報告: {root_cause.id} ({root_cause.type})\n理由: {reason}\n重要度: {root_severity}"
            if conf: system_prompt += f"\nConfig:\n{conf}"
        
        if system_prompt:
            chat = model.start_chat(history=[{"role": "user", "parts": [system_prompt]}])
            try:
                with st.spinner("Analyzing..."):
                    res = chat.send_message("状況報告をお願いします。")
                    st.session_state.chat_session = chat
                    st.session_state.messages.append({"role": "assistant", "content": res.text})
            except Exception as e: st.error(str(e))

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
        2. ログ分析 (エラーメッセージやステータスの確認)
        3. 推奨アクション (交換、再起動、静観など)
        """
        st.session_state.messages.append({"role": "user", "content": "診断結果を分析してください。"})
        
        with st.spinner("Analyzing Diagnostic Data..."):
            try:
                res = st.session_state.chat_session.send_message(prompt)
                st.session_state.messages.append({"role": "assistant", "content": res.text})
            except Exception as e: st.error(str(e))
        
        st.session_state.trigger_analysis = False
        st.rerun()

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
