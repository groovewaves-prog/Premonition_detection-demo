"""
Google Antigravity AIOps Agent - Network Operations Module
Cisco DevNet Sandbox (Nexus 9000) への接続とサニタイズを担当
"""
import re
import time
from netmiko import ConnectHandler

# Cisco DevNet Always-On Sandbox (Nexus 9000)
SANDBOX_DEVICE = {
    'device_type': 'cisco_nxos',    # Nexus OS
    'host': 'sandbox-nxos-1.cisco.com',
    'username': 'admin',
    'password': 'Admin_1234!',
    'port': 22,
    # 接続安定化オプション
    'global_delay_factor': 2,
    'banner_timeout': 30,
    'conn_timeout': 30,
}

def sanitize_output(text: str) -> str:
    """
    機密情報をマスク処理します (強化版)
    """
    rules = [
        # 1. Passwords / Secrets / Community Strings
        (r'(password|secret) \d+ \S+', r'\1 <HIDDEN_PASSWORD>'),
        (r'(encrypted password) \S+', r'\1 <HIDDEN_PASSWORD>'),
        (r'(snmp-server community) \S+', r'\1 <HIDDEN_COMMUNITY>'),
        (r'(username \S+ privilege \d+ secret \d+) \S+', r'\1 <HIDDEN_SECRET>'),
        
        # 2. Public IP Masking (プライベートIPは残し、グローバルIPのみ隠す)
        (r'\b(?!(?:10|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.)\d{1,3}\.(?:\d{1,3}\.){2}\d{1,3}\b', '<MASKED_PUBLIC_IP>'),
        
        # 3. MAC Address
        (r'([0-9A-Fa-f]{4}\.){2}[0-9A-Fa-f]{4}', '<MASKED_MAC>'),
    ]
    
    sanitized_text = text
    for pattern, replacement in rules:
        sanitized_text = re.sub(pattern, replacement, sanitized_text)
    return sanitized_text

def run_diagnostic_simulation(scenario_type):
    """
    実機接続を試行し、失敗した場合はシミュレーション結果を返すハイブリッド関数
    """
    # シミュレーション用シナリオの場合は即座にエラー等を返す
    if scenario_type == "1. WAN全回線断":
        return {
            "status": "ERROR",
            "sanitized_log": "Connection timed out (Host unreachable)",
            "error": "Timeout"
        }
    
    # Liveモードの場合は実機接続に挑戦
    if scenario_type == "4. [Live] Cisco実機診断":
        commands = [
            "terminal length 0",
            "show version",
            "show interface brief",
            "show ip route",
        ]
        
        raw_output = ""
        status = "SUCCESS"
        error_msg = None

        try:
            with ConnectHandler(**SANDBOX_DEVICE) as ssh:
                # Nexusはenableモード不要な場合が多いが念のため
                if not ssh.check_enable_mode():
                    ssh.enable()
                
                prompt = ssh.find_prompt()
                raw_output += f"Connected to: {prompt}\n"

                for cmd in commands:
                    output = ssh.send_command(cmd)
                    raw_output += f"\n{'='*30}\n[Command] {cmd}\n{output}\n"
                    time.sleep(0.5)
                    
        except Exception as e:
            # 実機接続失敗時のフォールバック (デモが止まらないように)
            status = "ERROR"
            error_msg = str(e)
            raw_output = f"SSH Connection Failed: {error_msg}\n(Sandbox may be busy or offline.)"

        return {
            "status": status,
            "sanitized_log": sanitize_output(raw_output),
            "error": error_msg
        }
        
    return {"status": "UNKNOWN", "sanitized_log": "", "error": "Unknown Scenario"}
