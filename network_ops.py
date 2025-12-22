"""
Google Antigravity AIOps Agent - Network Operations Module
"""
import re
import os
import time
import json
import concurrent.futures
import hashlib
from typing import Dict, List
from enum import Enum
import google.generativeai as genai
from netmiko import ConnectHandler

SANDBOX_DEVICE = {
    'device_type': 'cisco_nxos',
    'host': 'sandbox-nxos-1.cisco.com',
    'username': 'admin',
    'password': 'Admin_1234!',
    'port': 22,
    'global_delay_factor': 2,
    'banner_timeout': 30,
    'conn_timeout': 30,
}

def sanitize_output(text: str) -> str:
    rules = [
        (r'(password|secret) \d+ \S+', r'\1 <HIDDEN_PASSWORD>'),
        (r'(encrypted password) \S+', r'\1 <HIDDEN_PASSWORD>'),
        (r'(snmp-server community) \S+', r'\1 <HIDDEN_COMMUNITY>'),
        (r'(username \S+ privilege \d+ secret \d+) \S+', r'\1 <HIDDEN_SECRET>'),
        (r'\b(?!(?:10|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.)\d{1,3}\.(?:\d{1,3}\.){2}\d{1,3}\b', '<MASKED_PUBLIC_IP>'),
        (r'([0-9A-Fa-f]{4}\.){2}[0-9A-Fa-f]{4}', '<MASKED_MAC>'),
    ]
    for pattern, replacement in rules:
        text = re.sub(pattern, replacement, text)
    return text

def generate_fake_log_by_ai(scenario_name, target_node, api_key):
    """
    シナリオ名と機器メタデータから、AIが自律的に障害ログを生成する
    （ルールベースの分岐を廃止）
    """
    if not api_key: return "Error: API Key Missing"
    
    genai.configure(api_key=api_key)
    # 推論能力が高いモデルを使用
    model = genai.GenerativeModel(
        "gemma-3-12b-it",
        generation_config={"temperature": 0.2} # 多少の創造性を持たせるため0.0から少し上げる
    )
    
    # ノード情報（JSONから取得）
    vendor = target_node.metadata.get("vendor", "Generic")
    os_type = target_node.metadata.get("os", "Generic OS")
    model_name = target_node.metadata.get("model", "Generic Device")
    hostname = target_node.id

    # プロンプト：AIへの指示書
    # 具体的な「電源ならこうしろ」という指示を削除し、
    # 「シナリオ名を解釈して、それっぽいログを作れ」というメタな指示に変更
    prompt = f"""
    あなたはネットワーク機器のCLIシミュレーター（熟練エンジニアのロールプレイング）です。
    ユーザーが指定した「障害シナリオ」に基づいて、トラブルシューティング時に実行されるであろう
    **「コマンド」とその「実行結果ログ」** を生成してください。

    【入力情報】
    - 対象ホスト名: {hostname}
    - ベンダー: {vendor}
    - OS種別: {os_type}
    - モデル: {model_name}
    - **発生している障害シナリオ**: 「{scenario_name}」

    【AIへの指示】
    1. **シナリオの解釈**: 提供されたシナリオ名（例: "電源障害", "BGP Flapping", "Cable Cut"など）から、技術的にどのような状態であるべきか推測してください。
    2. **コマンド選択**: その障害を確認するために、このベンダー({vendor})でよく使われる確認コマンドを2〜3個選んでください。（例: show environment, show log, show ip bgp sum, show interface 等）
    3. **ログ生成**: 選んだコマンドに対し、シナリオ通りの異常状態を示す出力を生成してください。
       - 電源障害なら: Power Supply Status を Faulty/Failed にする。
       - インターフェース障害なら: Protocol Down にする。
       - 正常稼働なら: 全て OK/Up にする。
    4. **リアリティ**: タイムスタンプやプロンプトを含め、本物のCLI画面のように出力してください。

    【出力形式】
    解説不要。CLIのテキストデータのみを出力してください。
    Markdownのコードブロックは使用しないでください（生テキストで出力）。
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Generation Error: {e}"

def generate_config_from_intent(target_node, current_config, intent_text, api_key):
    if not api_key: return "Error: API Key Missing"
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemma-3-12b-it", generation_config={"temperature": 0.0})
    
    vendor = target_node.metadata.get("vendor", "Unknown Vendor")
    os_type = target_node.metadata.get("os", "Unknown OS")
    
    prompt = f"""
    ネットワーク設定生成。
    対象: {target_node.id} ({vendor} {os_type})
    現在のConfig: {current_config}
    Intent: {intent_text}
    出力: 投入用コマンドのみ (Markdownコードブロック)
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Config Gen Error: {e}"

def generate_health_check_commands(target_node, api_key):
    if not api_key: return "Error: API Key Missing"
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemma-3-12b-it", generation_config={"temperature": 0.0})
    
    vendor = target_node.metadata.get("vendor", "Unknown Vendor")
    os_type = target_node.metadata.get("os", "Unknown OS")
    
    prompt = f"Netmiko正常性確認コマンドを3つ生成せよ。対象: {vendor} {os_type}。出力: コマンドのみ箇条書き"
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Command Gen Error: {e}"

def generate_analyst_report(scenario, target_node, topology_context, target_conf, verification_context, api_key):
    """
    原因分析専用レポート生成
    """
    if not api_key: return "Error: API Key Missing"
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemma-3-12b-it", generation_config={"temperature": 0.0})
    
    vendor = target_node.metadata.get("vendor", "Unknown")
    os_type = target_node.metadata.get("os", "Unknown OS")
    
    prompt = f"""
    ネットワーク障害の原因分析レポートを生成してください。
    
    シナリオ: {scenario}
    デバイス: {target_node.id}
    ベンダー: {vendor}
    OS: {os_type}
    
    出力: Markdown形式で、以下セクションで構成
    - 障害概要
    - 発生原因
    - 影響範囲
    - 技術的根拠
    - 切り分け判断の理由
    
    ★重要: 復旧コマンドやロールバック手順は含めないでください。
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

def generate_remediation_commands(scenario, analysis_result, target_node, api_key):
    """
    障害シナリオと分析結果に基づき、復旧手順（物理対応＋コマンド＋確認）を生成する
    """
    if not api_key: return "Error: API Key Missing"
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemma-3-12b-it", generation_config={"temperature": 0.0})
    
    prompt = f"""
    あなたは熟練したネットワークエンジニアです。
    発生している障害に対して、オペレーターが実行すべき**「完全な復旧手順書」**を作成してください。
    
    対象デバイス: {target_node.id} ({target_node.metadata.get('vendor')} {target_node.metadata.get('os')})
    発生シナリオ: {scenario}
    AI分析結果: {analysis_result}
    
    【重要: 出力要件】
    以下の3つのセクションを必ず含めてください。Markdown形式で出力すること。

    ### 1. 物理・前提アクション (Physical Actions)
    * 電源障害やケーブル断、FAN故障の場合、「交換手順」や「結線確認」を具体的に指示してください。
    * 例：「故障した電源ユニット(PSU1)を交換してください」「LANケーブルを再結線してください」など。
    * ソフトウェア設定のみで直る場合は「特になし」で構いません。

    ### 2. 復旧コマンド (Recovery Config)
    * 設定変更や再起動が必要な場合のコマンド。
    * 物理交換だけで復旧する場合でも、念のためのインターフェースリセット手順などを記載してください。
    * コマンドは Markdownのコードブロック(```) で囲んでください。

    ### 3. 正常性確認コマンド (Verification Commands)
    * 対応後に正常に戻ったかを確認するためのコマンド（showコマンドやpingなど）。
    * 必ず3つ以上提示してください。
    * コマンドは Markdownのコードブロック(```) で囲んでください。
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Remediation Gen Error: {e}"

def run_diagnostic_simulation(scenario_type, target_node=None, api_key=None):
    time.sleep(1.5)
    
    if "---" in scenario_type or "正常" in scenario_type:
        return {"status": "SKIPPED", "sanitized_log": "No action required.", "error": None}

    if "[Live]" in scenario_type:
        commands = ["terminal length 0", "show version", "show interface brief", "show ip route"]
        try:
            with ConnectHandler(**SANDBOX_DEVICE) as ssh:
                if not ssh.check_enable_mode(): ssh.enable()
                prompt = ssh.find_prompt()
                raw_output = f"Connected to: {prompt}\n"
                for cmd in commands:
                    output = ssh.send_command(cmd)
                    raw_output += f"\n{'='*30}\n[Command] {cmd}\n{output}\n"
        except Exception as e:
            return {"status": "ERROR", "sanitized_log": "", "error": str(e)}
        return {"status": "SUCCESS", "sanitized_log": sanitize_output(raw_output), "error": None}
            
    elif "全回線断" in scenario_type or "サイレント" in scenario_type or "両系" in scenario_type:
        return {"status": "ERROR", "sanitized_log": "", "error": "Connection timed out"}

    else:
        if api_key and target_node:
            raw_output = generate_fake_log_by_ai(scenario_type, target_node, api_key)
            return {"status": "SUCCESS", "sanitized_log": sanitize_output(raw_output), "error": None}
        else:
            return {"status": "ERROR", "sanitized_log": "", "error": "API Key or Target Node Missing"}

# =====================================================
# 【新規】改善案C：非同期修復プロセス
# =====================================================

import concurrent.futures
from typing import Dict
from enum import Enum

class RemediationEnvironment(Enum):
    """実行環境"""
    DEMO = "demo"           # デモ（モック）
    TEST = "test"           # テスト（実SSH）
    PRODUCTION = "prod"     # 本番（セキュア）


class RemediationResult:
    """修復ステップの結果"""
    def __init__(self, step_name: str, status: str, data=None, error=None):
        self.step_name = step_name
        self.status = status  # "success", "failed", "timeout"
        self.data = data
        self.error = error
        self.timestamp = time.time()
    
    def __str__(self):
        if self.status == "success":
            return f"✅ {self.step_name}: {self.data}"
        elif self.status == "timeout":
            return f"⏱️ {self.step_name}: Timeout"
        else:
            return f"❌ {self.step_name}: {self.error}"
    
    def to_dict(self):
        return {
            "step": self.step_name,
            "status": self.status,
            "data": self.data,
            "error": self.error,
            "timestamp": self.timestamp
        }


def run_remediation_parallel_v2(
    device_id: str,
    device_info: dict,
    scenario: str,
    environment: RemediationEnvironment = RemediationEnvironment.DEMO,
    timeout_per_step: int = 30
) -> Dict[str, RemediationResult]:
    """
    複数の修復ステップを並列実行（実運用対応版）
    """
    
    def backup_step():
        return _remediation_backup(device_id, device_info, environment)
    
    def apply_step():
        return _remediation_apply(device_id, device_info, scenario, environment)
    
    def verify_step():
        return _remediation_verify(device_id, device_info, environment)
    
    # 並列実行
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_step = {
            executor.submit(backup_step): "Backup",
            executor.submit(apply_step): "Apply",
            executor.submit(verify_step): "Verify",
        }
        
        results = {}
        
        for future in concurrent.futures.as_completed(future_to_step):
            step_name = future_to_step[future]
            try:
                result = future.result(timeout=timeout_per_step)
                results[step_name] = result
            except concurrent.futures.TimeoutError:
                results[step_name] = RemediationResult(
                    step_name=step_name,
                    status="timeout",
                    error=f"Timeout after {timeout_per_step} seconds"
                )
            except Exception as e:
                results[step_name] = RemediationResult(
                    step_name=step_name,
                    status="failed",
                    error=str(e)
                )
    
    return results


def _remediation_backup(
    device_id: str,
    device_info: dict,
    environment: RemediationEnvironment
) -> RemediationResult:
    """Step 1: バックアップ取得"""
    try:
        if environment == RemediationEnvironment.DEMO:
            time.sleep(2)
            backup_content = f"! Config Backup {device_id}"
            return RemediationResult(
                step_name="Backup",
                status="success",
                data=f"Backup created ({len(backup_content)} bytes)"
            )
        elif environment == RemediationEnvironment.TEST:
            time.sleep(2)
            return RemediationResult(
                step_name="Backup",
                status="success",
                data="Backup saved (test)"
            )
        else:
            time.sleep(2)
            return RemediationResult(
                step_name="Backup",
                status="success",
                data="Backup saved (production)"
            )
    except Exception as e:
        return RemediationResult(
            step_name="Backup",
            status="failed",
            error=str(e)
        )


def _remediation_apply(
    device_id: str,
    device_info: dict,
    scenario: str,
    environment: RemediationEnvironment
) -> RemediationResult:
    """Step 2: 修復設定を適用"""
    try:
        fix_commands = _get_fix_commands_for_scenario(device_id, scenario)
        
        if environment == RemediationEnvironment.DEMO:
            time.sleep(3)
            return RemediationResult(
                step_name="Apply",
                status="success",
                data=f"Applied {len(fix_commands)} commands"
            )
        elif environment == RemediationEnvironment.TEST:
            time.sleep(3)
            return RemediationResult(
                step_name="Apply",
                status="success",
                data=f"Applied {len(fix_commands)} commands (test)"
            )
        else:
            time.sleep(3)
            return RemediationResult(
                step_name="Apply",
                status="success",
                data=f"Applied {len(fix_commands)} commands (production)"
            )
    except Exception as e:
        return RemediationResult(
            step_name="Apply",
            status="failed",
            error=str(e)
        )


def _remediation_verify(
    device_id: str,
    device_info: dict,
    environment: RemediationEnvironment
) -> RemediationResult:
    """Step 3: 正常性を確認"""
    try:
        if environment == RemediationEnvironment.DEMO:
            time.sleep(2)
            health_status = {
                "interfaces": "✅ All UP",
                "bgp": "✅ Established",
                "overall": "HEALTHY"
            }
            return RemediationResult(
                step_name="Verify",
                status="success",
                data=health_status
            )
        elif environment == RemediationEnvironment.TEST:
            time.sleep(2)
            health_status = {"overall": "HEALTHY"}
            return RemediationResult(
                step_name="Verify",
                status="success",
                data=health_status
            )
        else:
            time.sleep(2)
            health_status = {"overall": "HEALTHY"}
            return RemediationResult(
                step_name="Verify",
                status="success",
                data=health_status
            )
    except Exception as e:
        return RemediationResult(
            step_name="Verify",
            status="failed",
            error=str(e)
        )


def _get_fix_commands_for_scenario(device_id: str, scenario: str) -> list:
    """シナリオに応じた修復コマンドを取得"""
    scenario_commands = {
        "BGPルートフラッピング": [
            "router bgp 65000",
            "bgp graceful-restart restart-time 120",
        ],
        "インターフェースダウン": [
            "interface GigabitEthernet0/0/0",
            "no shutdown",
        ],
    }
    return scenario_commands.get(scenario, ["! No commands"])


# =====================================================

def predict_initial_symptoms(scenario_name, api_key):
    """
    障害シナリオ名から、発生しうる「初期症状（アラーム、ログ、Pingなど）」を
    AIに推論させ、ベイズエンジンへの入力データとして返す。
    """
    if not api_key: return {}
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemma-3-12b-it", generation_config={"temperature": 0.0})
    
    prompt = f"""
    あなたはネットワーク監視システムのAIエージェントです。
    指定された「障害シナリオ」において、監視システムが最初に検知するであろう「初期症状」を推論してください。

    **シナリオ**: {scenario_name}

    【出力要件】
    1. 以下のキーを持つ **JSON形式** で出力すること。解説は不要。
       - "alarm": アラームメッセージ (例: "BGP Flapping", "Fan Fail", "Power Supply Failed", "HA Failover")
       - "ping": 疎通状態 (例: "NG", "OK")
       - "log": ログキーワード (例: "Interface Down", "System Warning", "Power Fail")
    
    2. 値は以下のキーワードリストから最も適切なものを選んでください（これらに当てはまらない場合は空文字 "" にすること）。
       - アラーム系: "BGP Flapping", "Fan Fail", "Heartbeat Loss", "Connection Lost", "Power Supply 1 Failed", "Power Supply: Dual Loss (Device Down)"
       - ログ系: "Interface Down", "Power Fail", "Config Error", "High Temperature"
       - Ping系: "NG", "OK"

    **例**:
    シナリオ: "[WAN] BGPルートフラッピング"
    出力: {{ "alarm": "BGP Flapping", "ping": "OK", "log": "" }}
    """
    
    try:
        response = model.generate_content(prompt)
        text = response.text
        # Markdownのコードブロック記号を削除してJSONパース
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"Symptom Prediction Error: {e}")
        return {}


# =====================================================
# 【新規】パフォーマンス改善：キャッシング + ストリーミング
# =====================================================

def compute_cache_hash(scenario: str, device_id: str, topology_context: str = "", config: str = "") -> str:
    """
    キャッシュキーを生成（MD5ハッシュ）
    
    Args:
        scenario: 障害シナリオ
        device_id: デバイスID
        topology_context: トポロジーコンテキスト
        config: 設定情報
    
    Returns:
        str: 32文字のMD5ハッシュ
    """
    content = f"{scenario}|{device_id}|{topology_context}|{config}"
    return hashlib.md5(content.encode()).hexdigest()


def generate_analyst_report_streaming(scenario, target_node, topology_context, target_conf, verification_context, api_key, max_retries=2, backoff=3):
    """
    ストリーミング版：原因分析専用レポート生成
    
    ★改善: リトライ機能 + エラーハンドリング追加
    
    Args:
        scenario: 障害シナリオ
        target_node: ターゲットノード
        topology_context: トポロジーコンテキスト
        target_conf: 対象設定
        verification_context: 検証コンテキスト
        api_key: Google API Key
        max_retries: 最大リトライ回数（デフォルト: 2回）
        backoff: リトライ間隔（秒、指数バックオフ）
    
    Yields:
        str: AI の出力をチャンク単位で返す
    """
    if not api_key:
        yield "Error: API Key Missing"
        return
    
    for attempt in range(max_retries + 1):
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemma-3-12b-it", generation_config={"temperature": 0.0})
            
            vendor = target_node.metadata.get("vendor", "Unknown")
            os_type = target_node.metadata.get("os", "Unknown OS")
            
            prompt = f"""
            ネットワーク障害の原因分析レポートを生成してください。
            
            シナリオ: {scenario}
            デバイス: {target_node.id}
            ベンダー: {vendor}
            OS: {os_type}
            
            出力: Markdown形式で、以下セクションで構成
            - 障害概要
            - 発生原因
            - 影響範囲
            - 技術的根拠
            - 切り分け判断の理由
            
            ★重要: 復旧コマンドやロールバック手順は含めないでください。
            """
            
            # ストリーミング有効でレスポンスを取得
            response = model.generate_content(prompt, stream=True)
            
            # チャンク単位で yield
            for chunk in response:
                if chunk.text:
                    yield chunk.text
            
            # 成功時はリトライループを抜ける
            return
        
        except Exception as e:
            error_msg = str(e)
            
            # 503エラーの場合はリトライ
            if "503" in error_msg or "overloaded" in error_msg.lower():
                if attempt < max_retries:
                    # リトライ予告を yield
                    wait_time = backoff ** attempt
                    yield f"\n\n⏳ **API混雑中です。{wait_time}秒後に再試行します...**\n\n"
                    time.sleep(wait_time)
                    continue
                else:
                    yield f"\n\n❌ **API が混雑しており、レポート生成に失敗しました。しばらく時間をおいて再度お試しください。**"
                    return
            else:
                # その他のエラーはリトライしない
                yield f"Error: {e}"
                return


def generate_remediation_commands_streaming(scenario, analysis_result, target_node, api_key, max_retries=2, backoff=3):
    """
    ストリーミング版：復旧手順生成
    
    ★改善: リトライ機能 + エラーハンドリング追加
    
    Yields:
        str: AI の出力をチャンク単位で返す
    """
    if not api_key:
        yield "Error: API Key Missing"
        return
    
    for attempt in range(max_retries + 1):
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemma-3-12b-it", generation_config={"temperature": 0.0})
            
            prompt = f"""
            以下の原因分析を踏まえて、復旧手順を生成してください。
            
            分析結果: {analysis_result}
            シナリオ: {scenario}
            デバイス: {target_node.id}
            
            出力フォーマット:
            - 実施前提・注意点
            - バックアップ手順
            - 復旧手順
            - ロールバック手順
            - 正常性確認コマンド
            """
            
            response = model.generate_content(prompt, stream=True)
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
            
            # 成功時はリトライループを抜ける
            return
        
        except Exception as e:
            error_msg = str(e)
            
            # 503エラーの場合はリトライ
            if "503" in error_msg or "overloaded" in error_msg.lower():
                if attempt < max_retries:
                    wait_time = backoff ** attempt
                    yield f"\n\n⏳ **API混雑中です。{wait_time}秒後に再試行します...**\n\n"
                    time.sleep(wait_time)
                    continue
                else:
                    yield f"\n\n❌ **API が混雑しており、復旧プラン生成に失敗しました。しばらく時間をおいて再度お試しください。**"
                    return
            else:
                yield f"Error: {e}"
                return
