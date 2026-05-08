# Kindle Capture 調査結果

## 環境

- macOS Darwin 25.3.0
- Python 3.9.6（macOS 標準）
- Kindle app: `com.amazon.Lassen`

## 確認済みの技術要素

### 1. Kindle アプリの検知

```python
# System Events でプロセス確認
osascript -e 'tell application "System Events" to get name of every process whose background only is false'
# → "Kindle" が含まれるか確認
```

### 2. ウィンドウ情報の取得

```python
# Quartz/CoreGraphics で Window ID・サイズ・位置を取得
import Quartz.CoreGraphics as CG
windows = CG.CGWindowListCopyWindowInfo(CG.kCGWindowListOptionOnScreenOnly, CG.kCGNullWindowID)
# kCGWindowOwnerName に "Kindle" を含むものをフィルタ
```

取得済み情報:
- Window ID: 動的（例: 616071）
- Size: 1024x768
- Position: 567, 293

### 3. ウィンドウタイトル（本のタイトル取得）

- `kCGWindowName` → 本が開いていないときは `"Kindle"`
- 本を開くとウィンドウタイトルが変わる**可能性がある**が、現状の調査では確認できていない
- System Events の `title of window 1` でも `"Kindle"` のみ
- **代替手段**: ユーザーにタイトルを入力させるか、accessibility の static text を探索する

### 4. スクリーンショット取得

```bash
# screencapture -l <window_id> でウィンドウ単位のキャプチャが可能
screencapture -l 616071 -x /tmp/kindle_test.png
# -x: サウンドなし
# 動作確認済み（335KB の PNG が生成された）
```

### 5. ページめくり

```applescript
tell application "System Events"
    tell process "Kindle"
        set frontmost to true
        delay 0.3
        key code 124 -- right arrow
    end tell
end tell
```
- 動作確認済み
- `key code 124` = 右矢印キー（次のページ）
- `key code 123` = 左矢印キー（前のページ）

### 6. 終了検知（最終ページの判定）

- 最終ページで右矢印を押してもページが変わらない
- **画像のハッシュ比較**で同一ページを検知する方式が有効
  - `hashlib.md5(file_bytes).hexdigest()` で比較
  - 2-3 回連続で同一ハッシュなら最終ページと判定

## 未確認事項

1. **本を開いた状態でのウィンドウタイトル**: ライブラリ画面では `"Kindle"` だが、本を開くと変わるか
2. **Kindle のリーダー画面でのツールバー**: ツールバーが表示されている場合、スクショに含まれる。自動で非表示にする方法
3. **固定レイアウト vs リフロー**: 固定レイアウト本とリフロー本でページ送りの挙動が異なる可能性

## 設計方針

### アーキテクチャ

- Python 単一ファイル（`kindle_capture.py`）
- 外部ライブラリ不要（標準ライブラリ + macOS の Quartz フレームワーク）
- CLI ツールとして実装

### 処理フロー

1. Kindle プロセスの検知
2. Kindle ウィンドウ ID の取得（CGWindowList）
3. 本のタイトル取得（ウィンドウタイトル or ユーザー入力）
4. 出力ディレクトリ作成（`captures/<sanitized_title>/`）
5. Kindle をフォアグラウンドに
6. ループ:
   a. スクリーンショット取得（`screencapture -l <wid> -x`）
   b. 前のページとハッシュ比較 → 同一なら終了判定
   c. ファイル保存（`p001.png`, `p002.png`, ...）
   d. 右矢印キーでページめくり
   e. 適切な wait（ページレンダリング待ち）
7. 完了メッセージ出力

### エラーハンドリング

- Kindle 未起動 → エラーメッセージで終了
- 本が開かれていない → ウィンドウタイトルで検知、ユーザーに本を開くよう促す
- スクリーンショット失敗 → リトライ（最大 3 回）
- ページめくり失敗 → リトライ
- ディスク容量不足 → 事前チェック + 書き込みエラーキャッチ
- ユーザー中断（Ctrl+C）→ グレースフルシャットダウン
- Kindle が途中で閉じられた → プロセス存在チェック

### 設定可能パラメータ

- `--delay`: ページ間の待機時間（デフォルト: 1.5 秒）
- `--output`: 出力先ディレクトリ（デフォルト: `./captures/`）
- `--start-page`: 開始ページ番号（デフォルト: 1）
- `--max-pages`: 最大ページ数制限（安全弁）
- `--title`: 手動タイトル指定（自動検知できない場合）
