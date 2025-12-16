# -*- coding: utf-8 -*-
"""
Google Antigravity AIOps Agent - Streamlit Main Application
完全版: アラーム選別、真因特定、カスケード障害分析
"""

import streamlit as st
import os
import json
import time
from typing import List, Dict, Any
import google.generativeai as genai

# 既存モジュールのインポート
from data import TOPOLOGY, NetworkNode
from logic import CausalInferenceEngine, Alarm, simulate_cascade_failure
from inference_engine import LogicalRCA
from verifier import verify_log_content, format_verification_report
from network_ops import (
    generate_fake_log_by_ai,
    run_diagnostic_simulation,
    generate_remediation_commands,
    generate_health_check_commands
)

# =====================================================
# ページ設定
# =====================================================
st.set_page_config(
    page_title="AIOps - 障害分析システム",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 定数定義
# =====================================================
SCENARIO_CATEGORIES = {
    "正常稼働": {
        "正常稼働": "正常稼働"
    },
    "WAN機器": {
        "[WAN] 電源障害：片系": "[WAN] 電源障害：片系",
        "[WAN] 電源障害：両系": "[WAN] 電源障害：両系",
        "[WAN] BGPフラッピング": "[WAN] BGPフラッピング",
        "[WAN] メモリリーク": "[WAN] メモリリーク"
    },
    "ファイアウォール": {
        "[FW] 電源障害：片系": "[FW] 電源障害：片系",
        "[FW] 電源障害：両系": "[FW] 電源障害：両系",
        "[FW] FAN故障": "[FW] FAN故障",
        "[FW] メモリリーク": "[FW] メモリリーク"
    },
    "スイッチ": {
        "[L2SW] 電源障害：片系": "[L2SW] 電源障害：片系",
        "[L2SW] 電源障害：両系": "[L2SW] 電源障害：両系",
        "[L2SW] FAN故障": "[L2SW] FAN故障",
        "[L2SW] メモリリーク": "[L2SW] メモリリーク",
        "[L2SW] サイレント障害": "[L2SW] サイレント障害"
    },
    "アクセスポイント": {
        "[AP] AP_01ダウン": "[AP] AP_01ダウン",
        "[AP] AP_01ケーブル障害": "[AP] AP_01ケーブル障害"
    },
    "複合障害": {
        "[複合] FW_01_PRIMARYとAP_03の多重障害": "[複合] FW_01_PRIMARYとAP_03の多重障害",
        "[複合] WAN電源片系+FAN多重障害": "[複合] WAN電源片系+FAN多重障害"
    }
}

# =====================================================
# セッション状態の初期化
# =====================================================
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'current_scenario' not in st.session_state:
    st.session_state.current_scenario = None
if 'root_cause_result' not in st.session_state:
    st.session_state.root_cause_result = None
if 'generated_log' not in st.session_state:
    st.session_state.generated_log = ""
if 'remediation_executed' not in st.session_state:
    st.session_state.remediation_executed = False
if 'health_check_done' not in st.session_state:
    st.session_state.health_check_done = False

# =====================================================
# ヘルパー関数
# =====================================================

def get_target_node_from_scenario(scenario: str) -> str:
    """シナリオから対象ノードIDを推定"""
    if "[WAN]" in scenario:
        return "WAN_ROUTER_01"
    elif "[FW]" in scenario:
        return "FW_01_PRIMARY"
    elif "[L2SW]" in scenario:
        return "L2_SW_01"
    elif "[AP]" in scenario:
        return "AP_01"
    elif "FW_01_PRIMARYとAP_03" in scenario:
        return "FW_01_PRIMARY"
    elif "WAN電源" in scenario:
        return "WAN_ROUTER_01"
    return "WAN_ROUTER_01"

def generate_massive_alarms(scenario: str, root_device_id: str) -> List[Alarm]:
    """
    大量の冗長アラームを生成（50-200件）
    実際の運用では、配下の全機器から様々なアラームが上がってくる
    """
    import random
    
    alarms = []
    root_node = TOPOLOGY.get(root_device_id)
    
    if not root_node:
        return alarms
    
    # 根本原因のアラーム
    if "電源" in scenario:
        if "両系" in scenario:
            alarms.append(Alarm(root_device_id, "Power Supply 1 Failed", "CRITICAL"))
            alarms.append(Alarm(root_device_id, "Power Supply 2 Failed", "CRITICAL"))
            alarms.append(Alarm(root_device_id, "Device Unreachable", "CRITICAL"))
        else:
            alarms.append(Alarm(root_device_id, "Power Supply 1 Failed", "WARNING"))
            alarms.append(Alarm(root_device_id, "Redundancy Lost", "WARNING"))
    elif "BGP" in scenario:
        alarms.append(Alarm(root_device_id, "BGP Peer Flapping", "CRITICAL"))
        alarms.append(Alarm(root_device_id, "Route Instability Detected", "WARNING"))
    elif "FAN" in scenario:
        alarms.append(Alarm(root_device_id, "Fan Module Failed", "CRITICAL"))
        alarms.append(Alarm(root_device_id, "Temperature Warning", "WARNING"))
    elif "メモリリーク" in scenario:
        alarms.append(Alarm(root_device_id, "Memory Usage 95%", "CRITICAL"))
        alarms.append(Alarm(root_device_id, "System Performance Degraded", "WARNING"))
    elif "ケーブル" in scenario:
        alarms.append(Alarm(root_device_id, "Interface GigabitEthernet0/1 Down", "CRITICAL"))
        alarms.append(Alarm(root_device_id, "Link Status Changed", "WARNING"))
    elif "ダウン" in scenario:
        alarms.append(Alarm(root_device_id, "Device Down", "CRITICAL"))
        alarms.append(Alarm(root_device_id, "SNMP Timeout", "CRITICAL"))
    
    # カスケード障害のアラーム生成
    cascade_alarms = simulate_cascade_failure(root_device_id, TOPOLOGY, "Connection Lost")
    alarms.extend(cascade_alarms[1:])  # 重複を避けるため根本原因以外を追加
    
    # ノイズアラームを大量追加（50-200件に）
    noise_messages = [
        "SNMP Trap Received",
        "Interface Utilization 50%",
        "Minor Configuration Change",
        "Backup Job Started",
        "User Login Detected",
        "Temperature Normal",
        "Fan Speed Adjusted",
        "ARP Cache Updated",
        "Routing Table Updated",
        "VLAN Database Modified",
        "ACL Hit Count Threshold",
        "Port Security Violation (Info)",
        "NTP Sync OK",
        "DNS Query Timeout (Retry OK)",
        "DHCP Lease Expired (Auto Renewed)",
    ]
    
    target_count = random.randint(50, 200)
    while len(alarms) < target_count:
        random_device = random.choice(list(TOPOLOGY.keys()))
        random_message = random.choice(noise_messages)
        random_severity = random.choice(["INFO", "WARNING", "INFO", "INFO"])  # INFO多め
        alarms.append(Alarm(random_device, random_message, random_severity))
    
    return alarms

def filter_critical_alarms(all_alarms: List[Alarm], api_key: str) -> List[Alarm]:
    """
    AIを使って本当に重要なアラームだけを3-5件に絞る
    """
    if not api_key:
        # APIキーがない場合はCRITICALのみ返す
        return [a for a in all_alarms if a.severity == "CRITICAL"][:5]
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # アラーム情報を整形
    alarm_list = "\n".join([
        f"{i+1}. Device: {a.device_id}, Message: {a.message}, Severity: {a.severity}"
        for i, a in enumerate(all_alarms[:100])  # 最初の100件のみ送信
    ])
    
    prompt = f"""
あなたはネットワーク監視システムのアラームフィルタリングAIです。
以下の大量のアラームから、**根本原因に関連する重要なアラームだけを3〜5件選択**してください。

【アラームリスト】
{alarm_list}

【選択ルール】
1. CRITICAL / WARNING の重要なアラームを優先
2. INFO（情報通知）は基本的に無視
3. 同じデバイスからの重複アラームは1つにまとめる
4. カスケード障害（配下の機器のConnection Lost）は根本原因ではないため除外
5. 電源障害、Interface Down、BGP Flapping、Fan Failなど「直接的な障害」を選ぶ

【出力形式】
選択したアラームの番号をカンマ区切りで出力してください。
例: 1,3,5,12,18

番号のみを出力し、説明は不要です。
"""
    
    try:
        response = model.generate_content(prompt)
        selected_indices = [int(x.strip()) - 1 for x in response.text.strip().split(',')]
        return [all_alarms[i] for i in selected_indices if i < len(all_alarms)]
    except Exception as e:
        st.warning(f"AIフィルタリングエラー: {e}")
        return [a for a in all_alarms if a.severity in ["CRITICAL", "WARNING"]][:5]

def get_cascade_impact(root_device_id: str) -> Dict[str, Any]:
    """
    カスケード障害の影響範囲を分析
    """
    affected_nodes = []
    root_node = TOPOLOGY.get(root_device_id)
    
    if not root_node:
        return {"count": 0, "nodes": [], "reason": ""}
    
    # BFSで配下のノードを列挙
    queue = [root_device_id]
    processed = {root_device_id}
    
    while queue:
        current_id = queue.pop(0)
        children = [n for n in TOPOLOGY.values() if n.parent_id == current_id]
        
        for child in children:
            if child.id not in processed:
                affected_nodes.append(child)
                queue.append(child.id)
                processed.add(child.id)
    
    # 理由文を生成
    reason = f"""
**カスケード障害の詳細分析**

【直接原因】
{root_device_id} が完全にダウンしています。

【なぜ配下の機器が監視不能なのか】
{root_device_id} はネットワークトポロジーのLayer {root_node.layer}に位置し、
すべての通信の中継点となっています。このデバイスがダウンすると、
配下の全機器への通信経路が遮断されるため、監視システムから到達不能となります。

【影響を受けている機器（{len(affected_nodes)}台）】
"""
    
    for node in sorted(affected_nodes, key=lambda n: n.layer):
        reason += f"\n├ {node.id} (Layer {node.layer}, {node.type})"
    
    reason += """

⚠️ **重要な注意事項**
これらの配下の機器自体には障害は発生していません。
ネットワーク経路が遮断されているため「監視不能」状態になっているだけです。
{root_device_id} を復旧すれば、これらの機器は自動的に正常状態に戻ります。
"""
    
    return {
        "count": len(affected_nodes),
        "nodes": affected_nodes,
        "reason": reason
    }

def generate_topology_graph(root_cause_id: str = None, cascade_nodes: List[str] = None) -> str:
    """
    Graphvizフォーマットのトポロジー図を生成
    色分け: 赤=真因、オレンジ=カスケード影響、緑=正常
    """
    cascade_set = set(cascade_nodes) if cascade_nodes else set()
    
    dot = """
digraph Topology {
    rankdir=TB;
    node [shape=box, style=filled];
    
"""
    
    for node_id, node in TOPOLOGY.items():
        if node_id == root_cause_id:
            color = "red"
            label = f"{node_id}\\n❌ 真因"
        elif node_id in cascade_set:
            color = "orange"
            label = f"{node_id}\\n⚠️ 監視不能"
        else:
            color = "lightgreen"
            label = node_id
        
        dot += f'    "{node_id}" [label="{label}", fillcolor={color}];\n'
    
    # エッジの追加
    for node_id, node in TOPOLOGY.items():
        if node.parent_id:
            dot += f'    "{node.parent_id}" -> "{node_id}";\n'
    
    dot += "}\n"
    return dot

# =====================================================
# メイン画面
# =====================================================

def main():
    st.title("🛡️ AIOps 障害分析システム")
    st.markdown("---")
    
    # サイドバー
    with st.sidebar:
        st.header("⚙️ 設定")
        
        # APIキー設定
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            api_key = st.text_input("Google API Key", type="password")
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key
        else:
            st.success("✅ APIキー設定済み")
        
        st.markdown("---")
        
        # 2段階シナリオ選択
        st.subheader("📋 障害シナリオ選択")
        
        # 第1段階: カテゴリ選択
        category = st.selectbox(
            "カテゴリを選択",
            list(SCENARIO_CATEGORIES.keys()),
            index=0
        )
        
        # 第2段階: 詳細シナリオ選択
        scenarios_in_category = SCENARIO_CATEGORIES[category]
        selected_scenario = st.selectbox(
            "詳細シナリオを選択",
            list(scenarios_in_category.keys()),
            index=0
        )
        
        st.markdown("---")
        
        # 分析実行ボタン
        if st.button("🚀 障害分析を実行", type="primary", use_container_width=True):
            if not api_key:
                st.error("❌ APIキーを設定してください")
            else:
                st.session_state.current_scenario = selected_scenario
                st.session_state.analysis_done = False
                st.session_state.remediation_executed = False
                st.session_state.health_check_done = False
                st.rerun()
        
        # リセットボタン
        if st.button("🔄 リセット", use_container_width=True):
            st.session_state.analysis_done = False
            st.session_state.current_scenario = None
            st.session_state.root_cause_result = None
            st.session_state.generated_log = ""
            st.session_state.remediation_executed = False
            st.session_state.health_check_done = False
            st.rerun()
    
    # メインコンテンツ
    if st.session_state.current_scenario and not st.session_state.analysis_done:
        st.info(f"シナリオ「{st.session_state.current_scenario}」の分析を開始します...")
        perform_analysis(st.session_state.current_scenario, api_key)
    
    elif st.session_state.analysis_done and st.session_state.root_cause_result:
        display_results(st.session_state.root_cause_result, api_key)
    
    else:
        # 初期画面
        st.markdown("""
## 👋 AIOps 障害分析システムへようこそ

### 🎯 システムの特徴
- **大量アラームから真因を自動特定**: 50-200件のアラームから重要な3-5件に絞り込み
- **カスケード障害の自動分析**: 配下の機器が監視不能になる理由を詳細に説明
- **AI駆動の復旧手順生成**: 物理対応からコマンド実行まで完全な手順書を自動生成

### 📋 使い方
1. 左サイドバーから**カテゴリ**を選択
2. **詳細シナリオ**を選択
3. **障害分析を実行**ボタンをクリック

### 🚀 準備完了
APIキーが設定されています。シナリオを選択して分析を開始してください。
""")
        
        # デバッグ情報（開発時のみ）
        with st.expander("🔧 デバッグ情報", expanded=False):
            st.write("**セッション状態:**")
            st.json({
                "analysis_done": st.session_state.analysis_done,
                "current_scenario": st.session_state.current_scenario,
                "has_result": st.session_state.root_cause_result is not None
            })

def perform_analysis(scenario: str, api_key: str):
    """障害分析を実行"""
    
    try:
        progress_container = st.container()
        
        with progress_container:
            st.info("🔍 障害分析を開始します...")
            
            # 1. 対象ノード特定
            st.write("📍 ステップ1: 対象ノードを特定中...")
            target_device_id = get_target_node_from_scenario(scenario)
            target_node = TOPOLOGY.get(target_device_id)
            
            if not target_node:
                st.error(f"❌ デバイス {target_device_id} が見つかりません")
                return
            
            st.success(f"✅ 対象デバイス: {target_device_id}")
            
            # 2. 障害ログ生成
            st.write("📝 ステップ2: 障害ログを生成中...")
            try:
                log_result = run_diagnostic_simulation(scenario, target_node, api_key)
                generated_log = log_result.get("sanitized_log", "")
                st.session_state.generated_log = generated_log
                st.success(f"✅ ログ生成完了（{len(generated_log)}文字）")
            except Exception as e:
                st.error(f"❌ ログ生成エラー: {e}")
                generated_log = f"Error: {e}"
                st.session_state.generated_log = generated_log
            
            # 3. 大量アラーム生成
            st.write("🚨 ステップ3: アラームを生成中（50-200件）...")
            try:
                all_alarms = generate_massive_alarms(scenario, target_device_id)
                st.success(f"✅ {len(all_alarms)}件のアラームを生成しました")
            except Exception as e:
                st.error(f"❌ アラーム生成エラー: {e}")
                all_alarms = [Alarm(target_device_id, "Error generating alarms", "CRITICAL")]
            
            # 4. AIアラーム選別
            st.write("🎯 ステップ4: AIが重要なアラームを選別中...")
            try:
                critical_alarms = filter_critical_alarms(all_alarms, api_key)
                st.success(f"✅ {len(critical_alarms)}件の重要アラームを抽出しました")
            except Exception as e:
                st.error(f"❌ アラーム選別エラー: {e}")
                # フォールバック: CRITICALアラームのみ
                critical_alarms = [a for a in all_alarms if a.severity == "CRITICAL"][:5]
                st.warning(f"⚠️ フォールバック: {len(critical_alarms)}件のCRITICALアラームを使用")
            
            # 5. ログ検証
            st.write("🔬 ステップ5: ログを検証中...")
            try:
                verification = verify_log_content(generated_log)
                st.success("✅ ログ検証完了")
            except Exception as e:
                st.error(f"❌ ログ検証エラー: {e}")
                verification = {}
            
            # 6. 因果推論
            st.write("🧠 ステップ6: 因果推論エンジンで真因を特定中...")
            try:
                engine = CausalInferenceEngine(TOPOLOGY)
                inference_result = engine.analyze_alarms(critical_alarms)
                st.success("✅ 因果推論完了")
            except Exception as e:
                st.error(f"❌ 因果推論エラー: {e}")
                # デフォルト結果を作成
                from logic import InferenceResult
                inference_result = InferenceResult(
                    root_cause_node=target_node,
                    root_cause_reason=f"エラー: {e}",
                    sop_key="ERROR",
                    related_alarms=critical_alarms,
                    severity="CRITICAL"
                )
            
            # 7. LLM冗長性分析
            st.write("🤖 ステップ7: LLMで冗長性を分析中...")
            try:
                rca = LogicalRCA(TOPOLOGY)
                llm_analysis = rca.analyze(critical_alarms)
                st.success("✅ LLM分析完了")
            except Exception as e:
                st.error(f"❌ LLM分析エラー: {e}")
                llm_analysis = [{
                    "id": target_device_id,
                    "label": "Analysis failed",
                    "prob": 0.5,
                    "type": "ERROR",
                    "tier": 1,
                    "reason": str(e)
                }]
            
            # 8. カスケード影響分析
            st.write("📊 ステップ8: カスケード影響を分析中...")
            try:
                cascade_impact = get_cascade_impact(target_device_id)
                st.success(f"✅ 影響範囲: {cascade_impact['count']}台")
            except Exception as e:
                st.error(f"❌ カスケード分析エラー: {e}")
                cascade_impact = {"count": 0, "nodes": [], "reason": str(e)}
            
            # 9. 復旧手順生成
            st.write("📋 ステップ9: 復旧手順を生成中...")
            try:
                remediation = generate_remediation_commands(
                    scenario,
                    llm_analysis[0] if llm_analysis else {},
                    target_node,
                    api_key
                )
                st.success("✅ 復旧手順生成完了")
            except Exception as e:
                st.error(f"❌ 復旧手順生成エラー: {e}")
                remediation = f"""
### エラーが発生しました
復旧手順の生成中にエラーが発生しました: {e}

### 推奨アクション
1. APIキーが正しく設定されているか確認してください
2. ネットワーク接続を確認してください
3. 手動での対応を検討してください
"""
            
            # 結果を保存
            st.session_state.root_cause_result = {
                "scenario": scenario,
                "target_device": target_device_id,
                "target_node": target_node,
                "all_alarms_count": len(all_alarms),
                "critical_alarms": critical_alarms,
                "inference_result": inference_result,
                "llm_analysis": llm_analysis,
                "verification": verification,
                "cascade_impact": cascade_impact,
                "remediation": remediation,
                "generated_log": generated_log
            }
            
            st.session_state.analysis_done = True
            st.success("✅ すべての分析が完了しました！")
            time.sleep(1)
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ 致命的エラーが発生しました: {e}")
        st.exception(e)
        st.warning("デバッグ情報を確認してください。")


def display_results(result: Dict[str, Any], api_key: str):
    """分析結果を表示"""
    
    st.markdown("# 📊 分析結果")
    st.markdown("---")
    
    # 1. KPIメトリクス
    st.markdown("## 🎯 真因特定結果")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        noise_reduction = ((result['all_alarms_count'] - len(result['critical_alarms'])) / result['all_alarms_count'] * 100)
        st.metric(
            "📉 ノイズ削減率",
            f"{noise_reduction:.1f}%",
            delta="AI選別済み"
        )
    
    with col2:
        st.metric(
            "📨 総アラーム数",
            f"{result['all_alarms_count']}件",
            delta=f"-{result['all_alarms_count'] - len(result['critical_alarms'])}件"
        )
    
    with col3:
        st.metric(
            "✅ 重要アラーム",
            f"{len(result['critical_alarms'])}件",
            delta="選別済み"
        )
    
    with col4:
        st.metric(
            "🎯 真因",
            "1件特定",
            delta="分析完了"
        )
    
    st.markdown("---")
    
    # 2. 真因の大きな表示
    inference = result['inference_result']
    root_node = inference.root_cause_node
    
    if root_node:
        # 確信度の計算
        confidence = result['llm_analysis'][0]['prob'] * 100 if result['llm_analysis'] else 50
        
        st.markdown(f"""
<div style="background-color: #ff4444; padding: 30px; border-radius: 15px; color: white; margin: 20px 0;">
    <h2 style="color: white; margin-top: 0;">🚨 真因特定完了</h2>
    <hr style="border-color: white; opacity: 0.3;">
    <h3 style="color: white;">デバイス: {root_node.id}</h3>
    <p style="font-size: 20px; margin: 10px 0;"><strong>障害種別:</strong> {result['scenario']}</p>
    <p style="font-size: 20px; margin: 10px 0;"><strong>影響度:</strong> {inference.severity}</p>
    <p style="font-size: 20px; margin: 10px 0;"><strong>AI確信度:</strong> {confidence:.0f}%</p>
    <hr style="border-color: white; opacity: 0.3;">
    <p style="font-size: 16px; margin-top: 15px;"><strong>分析理由:</strong><br>{inference.root_cause_reason}</p>
</div>
""", unsafe_allow_html=True)
    else:
        st.warning("⚠️ 真因を特定できませんでした")
    
    st.markdown("---")
    
    # 3. チョイスされたアラーム表示
    with st.expander("🚨 チョイスされた重要アラーム", expanded=True):
        if result['critical_alarms']:
            for i, alarm in enumerate(result['critical_alarms'], 1):
                severity_color = "🔴" if alarm.severity == "CRITICAL" else "🟡" if alarm.severity == "WARNING" else "⚪"
                st.markdown(f"{severity_color} **{i}.** `{alarm.device_id}` → {alarm.message} `[{alarm.severity}]`")
        else:
            st.info("アラームはありません（正常稼働）")
    
    st.markdown("---")
    
    # 4. カスケード影響の説明
    cascade = result['cascade_impact']
    if cascade['count'] > 0:
        with st.expander("📊 カスケード障害の影響分析", expanded=True):
            st.markdown(cascade['reason'])
            
            # 影響を受けているノードのリスト
            st.markdown("### 影響を受けている機器の詳細")
            for node in cascade['nodes']:
                st.markdown(f"- **{node.id}** (Layer {node.layer}, {node.type})")
    else:
        st.info("✅ カスケード障害は発生していません")
    
    st.markdown("---")
    
    # 5. トポロジー図
    st.markdown("## 🗺️ ネットワークトポロジー（影響範囲の可視化）")
    
    try:
        cascade_node_ids = [n.id for n in cascade['nodes']]
        topology_graph = generate_topology_graph(
            root_cause_id=result['target_device'],
            cascade_nodes=cascade_node_ids
        )
        
        st.graphviz_chart(topology_graph)
        
        st.markdown("""
**凡例:**
- 🔴 **赤**: 真因（根本原因のデバイス）
- 🟠 **オレンジ**: 監視不能（カスケード影響を受けている）
- 🟢 **緑**: 正常稼働中
""")
    except Exception as e:
        st.error(f"トポロジー図の生成エラー: {e}")
    
    st.markdown("---")
    
    # 6. 生成された障害ログ
    with st.expander("📝 生成された障害ログ", expanded=False):
        st.code(result['generated_log'], language='text')
    
    # 7. 根本原因分析の詳細
    with st.expander("🔍 根本原因分析の詳細", expanded=False):
        st.markdown("### 因果推論エンジンの分析")
        st.markdown(f"""
- **SOP Key**: `{inference.sop_key}`
- **関連アラーム数**: {len(inference.related_alarms)}件
- **重大度**: {inference.severity}
""")
        
        st.markdown("### LLM分析結果")
        for i, analysis in enumerate(result['llm_analysis'], 1):
            st.markdown(f"**分析 {i}:**")
            st.json(analysis)
        
        st.markdown("### ログ検証結果（Ground Truth）")
        if result['verification']:
            st.text(format_verification_report(result['verification']))
        else:
            st.info("検証データがありません")
    
    st.markdown("---")
    
    # 8. 復旧手順
    st.markdown("## 📋 自動生成された復旧手順")
    
    st.markdown(result['remediation'])
    
    st.markdown("---")
    
    # 9. 復旧措置ボタン
    st.markdown("## 🔧 復旧アクション")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔧 復旧措置を実行", type="primary", use_container_width=True, key="remediation_btn"):
            with st.spinner("復旧措置を実行中..."):
                time.sleep(2)
                st.session_state.remediation_executed = True
                st.rerun()
    
    with col2:
        if st.button("✅ 正常性確認", use_container_width=True, key="health_check_btn"):
            with st.spinner("正常性確認中..."):
                time.sleep(2)
                st.session_state.health_check_done = True
                st.rerun()
    
    # 復旧措置の結果
    if st.session_state.remediation_executed:
        st.success("✅ 復旧措置が完了しました")
        st.markdown("""
**実行した内容:**
- ✅ 故障した電源ユニットを交換しました
- ✅ デバイスを再起動しました  
- ✅ インターフェースの状態を確認しました
- ✅ すべてのサービスが正常に稼働しています

**所要時間:** 約5分
""")
    
    # 正常性確認の結果
    if st.session_state.health_check_done:
        if result['scenario'] == "正常稼働":
            st.success("✅ すべてのデバイスが正常に稼働しています")
        else:
            try:
                target_node = result['target_node']
                health_commands = generate_health_check_commands(target_node, api_key)
                
                st.success("✅ 正常性確認が完了しました")
                st.markdown(f"""
**確認結果:**
- ✅ デバイス {result['target_device']} は正常に復旧しました
- ✅ すべてのインターフェースが UP 状態です
- ✅ 配下の機器も正常に通信可能です
- ✅ ネットワークトラフィックは正常です

**実行したコマンド:**
```
{health_commands}
```
""")
            except Exception as e:
                st.warning(f"正常性確認の一部でエラーが発生しました: {e}")
    
    st.markdown("---")
    
    # 10. AIチャット欄
    st.markdown("## 💬 AIアシスタント（詳細確認）")
    
    st.markdown("""
この障害分析について、さらに詳しく知りたいことがあれば質問してください。
例:
- この障害の影響範囲を教えて
- 復旧にかかる時間の見積もりは？
- 今後の予防策は？
""")
    
    user_question = st.text_input(
        "質問を入力してください",
        placeholder="例: この障害の影響範囲を詳しく教えて",
        key="chat_input"
    )
    
    if user_question:
        with st.spinner("AIが回答を生成中..."):
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                context = f"""
あなたはネットワーク障害分析のエキスパートAIアシスタントです。
以下の障害分析結果に基づいて、ユーザーの質問に丁寧かつ正確に答えてください。

【障害シナリオ】
{result['scenario']}

【真因デバイス】
{result['target_device']}

【分析結果】
{inference.root_cause_reason}

【影響範囲】
{cascade['count']}台の機器が影響を受けています

【重大度】
{inference.severity}

【確信度】
{confidence:.0f}%

【ユーザーの質問】
{user_question}

【回答の注意点】
- 技術的に正確な情報を提供してください
- 分かりやすく、実務的な回答を心がけてください
- 必要に応じて具体的な手順や数値を示してください
"""
                
                response = model.generate_content(context)
                st.markdown("### 🤖 AI回答")
                st.markdown(response.text)
                
            except Exception as e:
                st.error(f"AI回答の生成に失敗しました: {e}")
                st.info("APIキーの確認、またはネットワーク接続を確認してください。")

# =====================================================
# 実行
# =====================================================
if __name__ == "__main__":
    main()
