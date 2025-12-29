# AIOps インシデント・コックピット（複数拠点対応版）

## 概要

複数拠点のネットワーク障害を統合管理・分析するためのAIOpsダッシュボードです。

## 主な機能

### 🏢 拠点状態ボード
- 全拠点の状態を一覧表示（カード形式）
- ステータス別のKPIメトリクス
- 詳細画面への直接遷移

### 🚨 トリアージ・コマンドセンター
- 優先度順の対応管理
- ステータス別フィルタリング
- ワンクリックで対応開始

### 🛡️ インシデント・コックピット
- 根本原因候補の分析
- ネットワークトポロジー可視化
- AI支援による修復プラン生成

## 改善点（temp_app.pyからの変更）

| 項目 | 旧版 | 新版 |
|------|------|------|
| 名称 | 全社状態ボード / A社・B社 | 拠点状態ボード / A拠点・B拠点 |
| 拠点選択 | 機能しないヒートマップ | サイドバーでシナリオ選択 |
| 状態表示 | 複雑なPlotlyチャート | シンプルなカード形式 |
| トレンド分析 | モックデータのみ | **削除**（実データがないため） |
| Plotly依存 | 必須 | **削除**（不要） |
| registry.py | 存在しない | **新規作成** |

## ファイル構成

```
aiops_project/
├── app.py                  # メインアプリケーション
├── registry.py             # 拠点管理モジュール（新規）
├── alarm_generator.py      # アラーム生成モジュール（新規）
├── inference_engine.py     # 根本原因分析エンジン
├── network_ops.py          # ネットワーク操作
├── verifier.py             # ログ検証
├── rate_limiter.py         # APIレート制限
├── data.py                 # データ定義
├── logic.py                # 因果推論ロジック
├── requirements.txt        # 依存パッケージ
├── topologies/
│   ├── topology_a.json     # A拠点トポロジー
│   └── topology_b.json     # B拠点トポロジー
└── configs/
    ├── FW_01_PRIMARY.txt   # 設定ファイル
    └── WAN_ROUTER_01.txt   # 設定ファイル
```

## 起動方法

```bash
# 依存パッケージのインストール
pip install -r requirements.txt

# アプリケーション起動
streamlit run app.py
```

## 環境変数（オプション）

```bash
# Google Generative AI API Key（AI機能を使用する場合）
export GOOGLE_API_KEY="your-api-key"
```

## 拠点の追加方法

1. `topologies/` ディレクトリに新しいトポロジーファイルを追加
   - ファイル名: `topology_<拠点ID小文字>.json`（例: `topology_c.json`）

2. 自動検出されるため、`registry.py` の変更は不要

3. カスタム設定が必要な場合は `sites_config.json` を作成:

```json
{
  "C": {
    "display_name": "C拠点（名古屋）",
    "topology_file": "topology_c.json",
    "networks": ["default"],
    "metadata": {"region": "Nagoya", "tier": 2}
  }
}
```

## 使い方

### 拠点別シナリオの設定

1. サイドバーで各拠点のシナリオを選択
2. 異なる拠点で異なるシナリオを設定可能
3. 「拠点状態ボード」タブで全体像を確認

### インシデント対応

1. 「詳細」ボタンをクリックしてコックピット画面へ
2. 根本原因候補を確認
3. 「修復プランを生成」でAI支援を受ける

## 注意事項

- API Keyが未設定の場合、AI機能（修復プラン生成など）は使用できません
- トポロジーファイルが見つからない場合、その拠点は表示されません
