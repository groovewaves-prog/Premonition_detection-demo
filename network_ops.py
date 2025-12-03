"""
Google Antigravity AIOps Agent - Network Operations Module
"""
import re
import os
import time
import google.generativeai as genai
from netmiko import ConnectHandler

# Cisco DevNet Sandbox
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

def generate_fake_log_by_ai(scenario_name, api_key):
    """
    機器タイプと障害タイプを動的に組み合わせて、矛盾のないログを一括生成する
    """
    if not api_key: return "Error: API Key Missing"
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    # --- 1. Device Context (機器の特定) ---
    # シナリオ名に含まれるタグから、適切なハードウェアモデルとOSを決定する
    if "[FW]" in scenario_name:
        device_model = "Cisco ASA 5555-X (Adaptive Security Appliance)"
        os_type = "Cisco ASA Software"
        target_name = "FW_01_PRIMARY"
    elif "[L2SW]" in scenario_name:
        device_model = "Cisco Catalyst 9300 Series Switch (Dual Power Supply)"
        os_type = "Cisco IOS-XE"
        target_name = "L2_SW_01"
    else:
        # Default / [WAN]
        device_model = "Cisco ISR 4451-X Router (Dual Power Supply)"
        os_type = "Cisco IOS-XE"
        target_name = "WAN_ROUTER_01"

    # --- 2. Failure Context (障害状態の定義) ---
    # 機器の種類に関わらず共通する「あるべき状態」を定義
    
    status_instructions = ""
    
    # A. 電源障害 (片系)
    if "電源" in scenario_name and "片系" in scenario_name:
        status_instructions = """
        【状態定義: 電源冗長稼働中 (片系ダウン)】
        1. ハードウェア状態:
           - Power Supply 1 (A): **Faulty / Failed / No Input** (故障)
           - Power Supply 2 (B): **OK / Good / Active** (正常)
        2. サービス影響:
           - インターフェースは全て **UP/UP** (電源冗長によりダウンしていない)。
           - Ping疎通は **成功**。
           - システム稼働時間はリセットされていない (Uptime継続)。
        3. ログ出力:
           - 環境モニタリングコマンド (`show environment`, `show env all` 等) で上記状態を示すこと。
           - Syslogに `%PEM-3-PEMFAIL` や `%PLATFORM_ENV-1-PS_FAIL` 等のエラーを含めること。
        """

    # B. 電源障害 (両系) -> これは全断と同じ扱い
    elif "電源" in scenario_name and "両系" in scenario_name:
        status_instructions = """
        【状態定義: 電源喪失 (システムダウン)】
        1. ログ内容:
           - 本来はログ取得不可だが、診断ツールとして「Connection Refused」または「Console not responding」のエラーを出力するか、
           - あるいは「再起動直後のBootログ(System returned to ROM by power-on)」を出力すること。
        """

    # C. FAN故障
    elif "FAN" in scenario_name:
        status_instructions = """
        【状態定義: ファン故障 (稼働中)】
        1. ハードウェア状態:
           - Fan Tray 1: **Faulty / Failure** (回転数異常)
           - System Temperature: **Warning / Minor Alert** (上昇傾向だがCriticalではない)
        2. サービス影響:
           - インターフェースは **UP/UP**。
           - Ping **成功**。
        3. ログ出力:
           - `show environment` 等でFanステータス異常を表示。
           - Syslogに `%ENVMON-3-FAN_FAILED` 等を含める。
        """

    # D. メモリリーク
    elif "メモリ" in scenario_name:
        status_instructions = """
        【状態定義: メモリ枯渇 (稼働中)】
        1. システム状態:
           - Total Memory Used: **96% - 99%**
           - Warning: Processor memory is low.
        2. サービス影響:
           - インターフェースは **UP** だが、反応が遅い可能性あり。
           - Ping **成功**。
        3. ログ出力:
           - `show processes memory` (またはASAなら `show memory`) で枯渇を表示。
           - Syslogに `%SYS-2-MALLOCFAIL` 等を含める。
        """

    # E. BGPフラッピング
    elif "BGP" in scenario_name:
        status_instructions = """
        【状態定義: BGP不安定】
        1. プロトコル状態:
           - BGP Neighbor State: **Active / Idle / Established** を頻繁に遷移。
           - Uptimeが数秒～数分と短い。
        2. サービス影響:
           - 物理インターフェースは **UP/UP**。
           - Ping **成功**。
        3. ログ出力:
           - `show ip bgp summary` でフラッピングを確認できること。
           - Syslogに `%BGP-5-ADJCHANGE` (Up/Down) が多数記録されていること。
        """
    
    # F. 全回線断
    elif "全回線断" in scenario_name:
        status_instructions = """
        【状態定義: 完全通信断】
        1. インターフェース: **DOWN/DOWN**
        2. Ping: **100% Loss**
        """

    # --- プロンプト構築 ---
    prompt = f"""
    あなたはネットワーク機器のCLIシミュレーターです。
    指定された機種と障害状態に基づいて、エンジニアがトラブルシューティングを行った際の「コマンド実行結果ログ」を生成してください。

    **対象機器モデル**: {device_model}
    **OSタイプ**: {os_type}
    **ホスト名**: {target_name}
    **発生シナリオ**: {scenario_name}

    {status_instructions}

    **出力要件**:
    1. 対象OS ({os_type}) に適したコマンドを使用すること。
       (例: IOSなら `show ip int br`, ASAなら `show interface ip brief`, NX-OSなら `show interface brief`)
    2. 解説やMarkdown装飾は不要。**CLIの生テキストのみ**を出力すること。
    3. **絶対に矛盾させないこと** (例: 片系障害なのにインターフェースをダウンさせない)。
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Generation Error: {e}"

def run_diagnostic_simulation(scenario_type, api_key=None):
    """診断実行関数"""
    time.sleep(1.5)
    
    status = "SUCCESS"
    raw_output = ""
    error_msg = None

    if "---" in scenario_type or "正常" in scenario_type:
        return {"status": "SKIPPED", "sanitized_log": "No action required.", "error": None}

    # Live実機診断
    if "[Live]" in scenario_type:
        commands = ["terminal length 0", "show version", "show interface brief", "show ip route"]
        try:
            with ConnectHandler(**SANDBOX_DEVICE) as ssh:
                if not ssh.check_enable_mode(): ssh.enable()
                prompt = ssh.find_prompt()
                raw_output += f"Connected to: {prompt}\n"
                for cmd in commands:
                    output = ssh.send_command(cmd)
                    raw_output += f"\n{'='*30}\n[Command] {cmd}\n{output}\n"
        except Exception as e:
            status = "ERROR"
            error_msg = str(e)
            raw_output = f"Real Device Connection Failed: {error_msg}"
            
    # 全断・サイレント・両系電源障害（接続不可系）
    # AIにエラーログを作らせるより、ここで明示的にTimeoutを返したほうが確実
    elif "全回線断" in scenario_type or "サイレント" in scenario_type or "両系" in scenario_type:
        status = "ERROR"
        error_msg = "Connection timed out"
        raw_output = "SSH Connection Failed. Host Unreachable. (No Response from Console)"

    # その他のログ取得可能系（片系障害、FAN、メモリ、BGP）
    else:
        if api_key:
            raw_output = generate_fake_log_by_ai(scenario_type, api_key)
        else:
            status = "ERROR"
            error_msg = "API Key Required"
            raw_output = "Cannot generate logs without API Key."

    return {
        "status": status,
        "sanitized_log": sanitize_output(raw_output),
        "error": error_msg
    }
