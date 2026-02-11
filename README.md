# AIOps Incident Cockpit — Multi-Site Edition

**ネットワーク障害の根本原因分析 + Digital Twin による予兆検知**

> 「一見、手品のような」ネットワーク障害予兆検知ツール — 現在のアラームから未来のインシデントを予測し、
> 既存ダッシュボードの中に 🔮 アイコン付きで表示します。UIの変更は一切不要です。

---

## 概要

本ツールは、複数拠点のネットワーク機器を監視する AIOps インシデント・コックピットです。
アラームのノイズを削減し、根本原因を特定し、AI による分析レポートと修復プランを自動生成します。

### Digital Twin Engine（予兆検知）

v2.0 で追加された Digital Twin Engine は、WARNING レベルのアラームから将来発生しうる
CRITICAL インシデントを予測します。

**動作原理:**

```
WARNING アラーム
    ↓
エスカレーションルール照合（キーワード + Embedding）
    ↓
BFS 影響伝搬シミュレーション（NetworkX）
    ↓
信頼度スコア算出（SPOF/HA/冗長性を考慮）
    ↓
🔮 予兆としてダッシュボードに表示
```

**表示例:**

| 順位 | ステータス | デバイス | 原因 | 確信度 | 推奨アクション |
|---|---|---|---|---|---|
| 1 | 🔮 予兆検知 | FW_01_PRIMARY | メモリ枯渇によるプロセスクラッシュ | 72% | ⚡ 30分後以内に対処 (7台影響) |
| 2 | 🟣 サイレント疑い | CORE_SW_01 | Silent Failure Suspected | 80% | 🔍 上位確認 |

---

## 機能一覧

### 根本原因分析（LogicalRCA）

- **カスケード抑制**: 上流機器の障害から派生したアラームを自動的に低スコア化
- **サイレント障害検出**: アラームを出さない上流機器の障害を、配下機器のパターンから推定
- **冗長性評価**: HA 構成・冗長電源の有無を考慮したスコアリング
- **ノイズ削減**: 大量のアラームから真の根本原因を数件に絞り込み

### Digital Twin 予兆検知

- **ルールベース + Embedding のハイブリッドマッチング**
  - キーワード一致（高速・確実）
  - all-MiniLM-L6-v2 による意味的類似度（未知パターン対応）
- **BFS 影響伝搬**: トポロジーグラフ上で障害の波及範囲を算出
- **SPOF 検出**: 単一障害点（冗長構成なし）の自動識別と信頼度ブースト
- **タイムライン予測**: ルールごとの推定エスカレーション時間を表示

### AI Analyst Report

- Google Gemma-3-12b-it による障害分析レポートの自動生成
- 機器設定ファイル・トポロジー情報を基にした文脈付き分析
- ストリーミング表示による即座のフィードバック

### 自動修復（Remediation）

- AI が復旧手順を自動生成
- ワンクリックでのシミュレーション実行
- 実行結果のリアルタイム表示

---

## 対応シナリオ（18種類）

| カテゴリ | シナリオ |
|---|---|
| **基本・広域障害** | 正常稼働 / WAN全回線断 / FW片系障害 / L2SWサイレント障害 |
| **WAN Router** | 電源障害（片系/両系） / BGPルートフラッピング / FAN故障 / メモリリーク |
| **Firewall** | 電源障害（片系/両系） / FAN故障 / メモリリーク |
| **L2 Switch** | 電源障害（片系/両系） / FAN故障 / メモリリーク |
| **複合・その他** | WAN複合障害（電源＆FAN） / 同時多発（FW & AP） |

---

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │ KPI      │  │ 候補     │  │ Topology │  │ AI Report  │  │
│  │ メトリクス │  │ テーブル  │  │ Graph    │  │ & Chat     │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│              inference_engine.py (LogicalRCA)                │
│  ┌──────────────────┐  ┌─────────────────────────────────┐  │
│  │ 既存RCA分析       │  │ Digital Twin 統合ブロック        │  │
│  │ - カスケード抑制   │  │ - predict() 呼び出し            │  │
│  │ - サイレント検出   │  │ - CRITICAL フィルタ             │  │
│  │ - 冗長性評価      │  │ - WARNING 置換マージ            │  │
│  └──────────────────┘  └───────────────┬─────────────────┘  │
└────────────────────────────────────────┼────────────────────┘
                                         │
┌────────────────────────────────────────┴────────────────────┐
│                digital_twin.py (予兆検知)                     │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────────────┐  │
│  │ Escalation │  │ Hybrid      │  │ BFS Propagation      │  │
│  │ Rules (9)  │  │ Matching    │  │ (NetworkX DiGraph)   │  │
│  │            │  │ Keyword +   │  │ + Confidence Scoring │  │
│  │            │  │ Embedding   │  │ + SPOF/HA Detection  │  │
│  └────────────┘  └─────────────┘  └──────────────────────┘  │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ all-MiniLM-L6-v2 (Singleton Cache)                      │ │
│  │ - クラス変数によるプロセス内キャッシュ                      │ │
│  │ - 初回のみロード、以降は 0.1ms 以下で応答                  │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## ファイル構成

```
Multiple-locations-demo-main/
│
│  # === メインアプリケーション ===
├── app.py                     # Streamlit UI (1,531行)
├── inference_engine.py        # LogicalRCA + Digital Twin統合 (497行)
├── digital_twin.py            # 予兆検知エンジン (295行)
├── streamlit_cache.py         # @st.cache_resource ラッパー (59行)
│
│  # === 既存モジュール（変更なし） ===
├── alarm_generator.py         # シナリオ→アラーム生成 (405行)
├── registry.py                # 拠点管理・トポロジー読込 (332行)
├── network_ops.py             # LLM連携・診断・修復 (611行)
├── verifier.py                # ログ検証 (470行)
├── rate_limiter.py            # APIレート制限 (236行)
├── dashboard.py               # ダッシュボード補助 (169行)
├── mock_data_gen.py           # モックデータ生成 (117行)
│
│  # === 設定・依存 ===
├── requirements.txt           # Python依存パッケージ
├── packages.txt               # apt-get依存 (graphviz)
├── .streamlit/
│   └── config.toml            # Streamlit テーマ設定
├── .gitignore
│
│  # === ネットワーク定義 ===
├── topologies/
│   ├── topology_a.json        # A拠点 (10ノード: Cisco/Juniper)
│   └── topology_b.json        # B拠点 (14ノード: Juniper/PaloAlto/Arista/Aruba)
├── configs/
│   ├── FW_01_PRIMARY.txt      # Junos SRX300 設定
│   └── WAN_ROUTER_01.txt      # Cisco IOS-XE BGP 設定
│
│  # === テスト ===
├── tests/
│   ├── test_integration.py    # 結合テスト: 7シナリオ, 18アサーション
│   ├── test_d1_d2.py          # 表示ロジックテスト: 21アサーション
│   └── test_full_regression.py # 全パイプライン回帰テスト: 18アサーション
│
│  # === デプロイ（エアギャップ環境用） ===
└── deploy_airgap.sh           # オフラインデプロイスクリプト
```

---

## セットアップ

### 前提条件

- Python 3.9 以上
- Google Gemini API キー（AI レポート機能を使用する場合）

### Streamlit Community Cloud（推奨）

1. **リポジトリを GitHub にプッシュ**

2. **Streamlit Community Cloud でデプロイ**
   - https://share.streamlit.io にアクセス
   - 「New app」→ リポジトリ・ブランチ・`app.py` を選択
   - 「Deploy!」をクリック

3. **API キーを設定**
   - アプリ管理画面 → Settings → Secrets に以下を入力:
   ```toml
   [general]
   GOOGLE_API_KEY = "AIza..."
   ```

4. **初回起動**
   - PyTorch (CPU) + all-MiniLM-L6-v2 のインストール・ダウンロードで 5〜10分

### ローカル開発

```bash
# クローン
git clone https://github.com/<your-repo>/Multiple-locations-demo-main.git
cd Multiple-locations-demo-main

# 仮想環境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 依存インストール
pip install -r requirements.txt

# 起動
streamlit run app.py
```

### エアギャップ環境

インターネット接続のない閉域ネットワークにデプロイする場合:

```bash
# === インターネット環境で実行 ===
./deploy_airgap.sh download    # モデル + wheel をダウンロード (~280MB)

# === USB等で転送後、エアギャップ環境で実行 ===
./deploy_airgap.sh install     # オフラインインストール + チェックサム検証
./deploy_airgap.sh verify      # 動作確認テスト
```

---

## requirements.txt について

```txt
# 既存依存
streamlit>=1.28.0
google-generativeai>=0.3.0
graphviz>=0.20
netmiko>=4.0.0
rich>=13.0.0
pandas>=2.0.0

# Digital Twin Engine
networkx>=3.0
numpy>=1.24.0
sentence-transformers>=2.2.2

# PyTorch CPU版（この行がないとGPU版2GB超がインストールされる）
--extra-index-url https://download.pytorch.org/whl/cpu
torch
```

> **重要**: `--extra-index-url` の行を削除すると GPU 版 PyTorch がインストールされ、
> Streamlit Community Cloud の 1GiB メモリ制限を超過してクラッシュします。

---

## トポロジー構成

### A拠点（10ノード）

```
INET_ROUTER_01 (Internet)
  └── WAN_ROUTER_01 (Cisco IOS-XE) ← SPOF
        └── FW_01_PRIMARY (Junos SRX300) ─── FW_01_SECONDARY [HA Pair]
              └── CORE_SW_01 (Core Switch)
                    ├── L2_SW_01 ── AP_01, AP_02
                    └── L2_SW_02 ── SRV_01
```

### B拠点（14ノード）

```
EDGE_ROUTER_B01 (Juniper) ← SPOF
  └── FW_B01_PRIMARY (PaloAlto) ─── FW_B01_SECONDARY [HA Pair]
        └── DIST_SW_B01 (Arista)
              ├── L2_SW_B01 ── AP_B01, AP_B02
              ├── L2_SW_B02 ── SRV_B01, SRV_B02
              └── L2_SW_B03 ← SPOF
                    └── AP_B03, AP_B04
```

---

## Digital Twin 信頼度スコアの計算

```
confidence = base_confidence
           × (0.8 + 0.2 × match_quality)
           × (1.0 - REDUNDANCY_DISCOUNT)   ... HA構成の場合
           × SPOF_BOOST                     ... 単一障害点の場合
```

| パラメータ | 値 | 説明 |
|---|---|---|
| `base_confidence` | ルールごと (0.50〜0.65) | エスカレーションルールが持つ基本信頼度 |
| `match_quality` | 0.0〜1.0 | キーワード一致=1.0, Embedding類似度=実測値 |
| `REDUNDANCY_DISCOUNT` | 0.15 | HA 構成の場合に信頼度を15%割引 |
| `SPOF_BOOST` | 1.10 | 単一障害点の場合に信頼度を10%加算 |
| `MIN_PREDICTION_CONFIDENCE` | 0.50 | これ未満の予測は破棄 |

### 信頼度のレンジ（テスト実績値）

| シナリオ | 信頼度 | 条件 |
|---|---|---|
| L2SW SPOF + Memory | 94% | SPOF + キーワード完全一致 |
| BGP Flap + SPOF | 88% | SPOF + BGP ルール |
| WAN Router FAN + SPOF | 77% | SPOF + FAN ルール |
| FW Memory + HA | 72% | HA 構成（冗長性あり） |
| FW Heartbeat + HA | 59% | HA 構成 + heartbeat ルール |

---

## キャッシュ設計（Streamlit 対応）

Streamlit はボタン操作のたびにスクリプト全体を再実行します。
モデルの再ロード（数秒〜数十秒）を防ぐため、2層のキャッシュを実装しています。

```
[ボタンクリック] → app.py 全行再実行
    │
    ├─ レイヤ1: st.session_state (app.py L857-860)
    │   if engine_key not in st.session_state:  ← 2回目以降: False
    │       LogicalRCA(topology)                ← スキップ（既存インスタンスを再利用）
    │   → DigitalTwinEngine.__init__() は呼ばれない
    │
    └─ レイヤ2: クラス変数 Singleton (digital_twin.py L104-106)
        _model_loaded = True  ← プロセス存続中は True のまま
        → 万が一 __init__() が呼ばれても、if _model_loaded: return で即座に終了
```

| タイミング | モデルロード | 応答時間 |
|---|---|---|
| 初回アクセス（コールドスタート） | 発生する | 5〜10秒 |
| ボタンクリック（同一セッション） | 発生しない | 0.1ms 以下 |
| 別拠点切替 | 発生しない | 0.1ms 以下 |
| スリープ復帰（プロセス再起動） | 発生する | 5〜10秒 |

---

## テスト

```bash
# 結合テスト（Digital Twin + LogicalRCA）
python tests/test_integration.py

# 表示ロジックテスト（D1: ステータス表示 + D2: KPI）
python tests/test_d1_d2.py

# 全パイプライン回帰テスト（RCA温存 + 予兆 + 表示）
python tests/test_full_regression.py
```

### テスト結果サマリ

| テストスイート | シナリオ数 | アサーション数 | 結果 |
|---|---|---|---|
| `test_integration.py` | 7 | 18 | ✅ ALL PASSED |
| `test_d1_d2.py` | 5 | 21 | ✅ ALL PASSED |
| `test_full_regression.py` | 5 | 18 | ✅ ALL PASSED |

---

## ステータスアイコン一覧

| アイコン | ステータス | 意味 |
|---|---|---|
| 🔮 | 予兆検知 | Digital Twin が将来の障害を予測 |
| 🔴 | 危険（根本原因） | CRITICAL — サービス停止中 |
| 🟡 | 警告 | WARNING — 冗長性低下 |
| 🟣 | サイレント疑い | アラームなしだが配下の状況から障害を推定 |
| ⚪ | 監視中 | 低確度、経過観察 |
| 🟢 | 正常 | 問題なし |

---

## セキュリティ

### all-MiniLM-L6-v2

- **エンコーダ専用モデル**: テキスト生成能力なし。プロンプトインジェクションのリスクなし
- **完全ローカル実行**: 外部通信なし。アラームデータはベクトルに変換されるのみ
- **safetensors 形式**: pickle 任意コード実行攻撃を構造的に防止

### エアギャップデプロイ

- `deploy_airgap.sh` が SHA-256 チェックサムを生成・検証
- ファイル改竄が検出された場合、インストールを中止

---

## ライセンス

（プロジェクトに合わせて記載してください）

---

## 参考文献

- TSP_CMC_68733.pdf — Agentic AI / Foundation Models によるゼロショット検知
- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — Embedding モデル
- [Streamlit Documentation](https://docs.streamlit.io/) — デプロイ・キャッシュ設計
