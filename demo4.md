# demo4.py — カメラピッキングのデモを mp4 に録画する

`demo3.py` とまったく同じピッキングデモ（俯瞰 RGB-D カメラで緑のキューブを見つけ、
Franka アームで掴んで持ち上げる）を実行しつつ、**その様子を mp4 に保存する**スクリプトです。

ビジョンまわりの中身は `demo3.py` と同一なので、**アルゴリズムの説明は `demo3.md` を参照**してください。
こちらの md は**録画まわりの差分**に絞って書いています。

録画が終わると自動的に終了するので、**ヘッドレス実行で動画だけ生成する**使い方ができます。

---

## 1. 起動方法

```powershell
# 仮想環境をアクティベート
PS C:\Users\takuy\dev> env_isaaclab\Scripts\activate

# プロジェクトへ移動して実行（ビューアあり）
PS C:\Users\takuy\dev\IsaacLab> .\isaaclab.bat -p demo4.py --viz kit
```

| やりたいこと | コマンド |
|---|---|
| ビューアで眺めながら録画 | `isaaclab.bat -p demo4.py --viz kit` |
| 画面を出さずに動画だけ作る | `isaaclab.bat -p demo4.py --headless` |
| 長め・高解像度＋カメラ映像も残す | `isaaclab.bat -p demo4.py --headless --video_length 900 --video_width 1280 --video_height 720 --record_camera` |

> **重要**: `demo3.py` と同じく、`source/isaaclab/isaaclab/utils/math.py` の
> `unproject_depth()` に修正を当てないと視覚推定が機能しません（`demo3.md` §6 ① 参照）。

### コマンドラインオプション

`demo3.py` の引数に加えて、録画用の引数が増えています。

| オプション | 既定値 | 内容 |
|---|---|---|
| `--position_threshold` | `0.01` | ステートマシンの到達判定の位置誤差 [m] |
| `--camera_width` | `640` | 俯瞰カメラの画像幅 [px] |
| `--camera_height` | `480` | 俯瞰カメラの画像高さ [px] |
| `--episode_length_s` | `8.0` | エピソード長 [s] |
| `--video_folder` | `videos/demo4` | 動画の出力先ディレクトリ |
| `--video_name` | `demo4` | シーン映像のファイル名のベース |
| `--video_length` | `480` | 録画するステップ数（1 ステップ 0.02 s なので既定で約 9.6 秒） |
| `--video_width` | `960` | シーン映像の幅 [px] |
| `--video_height` | `540` | シーン映像の高さ [px] |
| `--record_camera` | off | 俯瞰カメラの映像も別ファイルに録画する |

---

## 2. 出力される動画

| ファイル | 内容 | 条件 |
|---|---|---|
| `<video_folder>/<video_name>-step-13.mp4` | シーン全体を斜め前方から見た映像 | 常に出力 |
| `<video_folder>/<video_name>-overhead.mp4` | 位置推定に使っている**俯瞰カメラそのもの**の映像 | `--record_camera` 指定時のみ |

ファイル名末尾の `step-13` は**録画を開始したステップ番号**です
（`RENDER_WARMUP_STEPS = 12` の次のステップ、つまり 13 から録り始めます）。

俯瞰カメラ映像は「推定が外れたときに**カメラに何が映っていたか**」を後から確認するためのものです。
色マスクが痩せていないか、キューブがフレームから外れていないか、といった切り分けに使えます。

### シーン映像の視点

```python
VIDEO_EYE     = (1.5, -1.05, 0.95)   # 作業台の前方右寄り・斜め上 [m]
VIDEO_LOOK_AT = (0.45, 0.0, 0.25)    # 作業台の上あたり [m]
```

キューブ・グリッパ・俯瞰カメラの 3 つが**ピッキング中ずっと画面に収まる**ように決めた画角です。

---

## 3. 録画の仕組み

2 本の動画は**まったく違う方法**で作られています。

### (1) シーン映像 — `gymnasium.wrappers.RecordVideo`

```python
env_cfg.viewer.eye = VIDEO_EYE
env_cfg.viewer.lookat = VIDEO_LOOK_AT
env_cfg.video_recorder.backend_source = "renderer"
env_cfg.video_recorder.window_width = args_cli.video_width
env_cfg.video_recorder.window_height = args_cli.video_height

env = gym.make("IsaacContrib-Lift-Cube-Franka-IK-Abs", cfg=env_cfg, render_mode="rgb_array")
env = gym.wrappers.RecordVideo(
    env,
    video_folder=video_folder,
    step_trigger=lambda step: step == RENDER_WARMUP_STEPS + 1,
    video_length=args_cli.video_length,
    name_prefix=args_cli.video_name,
    disable_logger=True,
)
```

- `render_mode="rgb_array"` が、`RecordVideo` がフレームを引っぱってくる先です。
- `backend_source = "renderer"` にすることで、**対話ウィンドウではなくレンダラから**直接キャプチャします。
  これにより `--headless` でもビューアを開いていても、**同じ画角の動画**になります。
- `step_trigger` は 1 回だけ真になるラムダです。`RecordVideo` は**トリガ評価の前に**
  ステップカウンタを進めるので最初のステップが 1 になります。ウォームアップ 12 ステップの次、
  つまり `13` で録画を開始します。

**`RecordVideo` は全フレームをメモリに溜めてから書き出します。**
`--video_length` や解像度を上げるとメモリ使用量が増えます（1 フレームあたり おおよそ 幅 × 高さ × 3 バイト）。
既定（480 ステップ、960×540）なら約 750 MB 相当です。

### (2) 俯瞰カメラ映像 — `OverheadCameraVideo`（自前クラス）

こちらは**1 フレームずつ逐次エンコード**するので、長さによらずメモリ使用量は一定です。

```python
class OverheadCameraVideo:
    def __init__(self, path: str, fps: float):
        import imageio.v2 as imageio          # --record_camera のときだけ必要
        self._writer = imageio.get_writer(path, fps=fps, quality=8)

    def append(self, camera: Camera) -> None:
        output = camera.data.output
        if output is None:
            return                            # まだ描画されていないフレームは黙って捨てる
        rgb = output["rgb"].torch[0, ..., :3]
        if rgb.is_floating_point():
            rgb = (rgb.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        self._writer.append_data(rgb.cpu().numpy())
```

- `imageio` は**遅延 import** しています。`--record_camera` を使わない限り依存しません。
- fps は `1.0 / env.unwrapped.step_dt` から求めるので、**実時間と同じ速さで再生**されます。
- バックエンドが RGB を uint8 [0, 255] で返すか float [0, 1] で返すかは環境依存なので、両方に対応しています。

### (3) 終了条件

```python
recorded_steps = 0
while simulation_app.is_running() and recorded_steps < args_cli.video_length:
    ...
    recorded_steps += 1
```

`demo3.py` は無限ループでしたが、こちらは**指定ステップ数を録り終えると自動的に抜けます**。
そのため `--headless` でバッチ実行できます。

---

## 4. 実行したときのログ

起動時（`demo3.py` の出力に録画情報が加わります）:

```
[INFO] シミュレーション開始。俯瞰カメラの映像からキューブ位置を推定してピッキングします...
[INFO] カメラ解像度: 640x480 px
[INFO] 録画中: C:\Users\takuy\dev\IsaacLab\videos\demo4 (480 ステップ, 960x540 px)
[INFO] 俯瞰カメラ映像も録画中: C:\Users\takuy\dev\IsaacLab\videos\demo4\demo4-overhead.mp4
```

推定をラッチしてピッキングを開始した時点で 1 回:

```
[VISION] 推定位置(env-local): [0.503 0.118 0.034] / 真値との誤差: 4.2 mm (12 フレームの中央値)
```

録り終えたとき:

```
[INFO] 録画完了。動画は C:\Users\takuy\dev\IsaacLab\videos\demo4 に保存されました。
```

`--video_folder` は `os.path.abspath()` で絶対パス化されるので、
**Isaac Lab を起動したフォルダからの相対パス**として解釈されます。

---

## 5. `demo3.py` との差分

ビジョンとステートマシンのコードは**完全に同一**です。差分は次だけです。

| | `demo3.py` | `demo4.py` |
|---|---|---|
| 録画 | なし | `RecordVideo` ＋ 任意で俯瞰カメラ映像 |
| 追加引数 | — | `--video_folder` / `--video_name` / `--video_length` / `--video_width` / `--video_height` / `--record_camera` |
| `--headless` | ビューア既定のまま | `visualizer = ["none"]` に切り替える（§6 ①） |
| `render_mode` | 指定なし | `"rgb_array"` |
| ビューア姿勢 | 既定 | `VIDEO_EYE` / `VIDEO_LOOK_AT` で作業台に寄せる |
| ウォームアップ | `for _ in range(12)` のベタ書き | 定数 `RENDER_WARMUP_STEPS = 12`（トリガ計算と共有） |
| ループ終了 | 終了条件なし | `--video_length` ステップで終了 |
| 追加クラス | — | `OverheadCameraVideo` |
| 追加 import | — | `os`、（遅延）`imageio` |

---

## 6. ハマりどころ（録画まわり）

`demo3.md` §6 のハマりどころはすべてそのまま当てはまります。以下は録画固有のものです。

### ① `--headless` だけでは Kit ビューアの既定が残る

このスクリプトは `parser.set_defaults(visualizer=["kit"])` でビューアを既定 ON にしています。
ところが**ビジュアライザの選択は `headless` フラグより後で解決される**ため、
`--headless` を付けても Kit ウィンドウを要求し続けます
（`isaaclab_visualizers[kit]` が入っていない環境では、そこで落ちます）。

```python
if args_cli.headless:
    args_cli.visualizer = ["none"]
```

`AppLauncher` を作る**前に**明示的に上書きしています。

### ② `backend_source = "renderer"` にしないと画角が安定しない

対話ウィンドウからキャプチャする設定のままだと、ビューアを開いているかどうかで
**動画の見え方が変わります**。レンダラから直接取ることで、ヘッドレスでも同じ画になります。

### ③ `step_trigger` のステップ番号は 1 始まり

`RecordVideo` は**トリガを評価する前に**内部のステップカウンタを進めます。つまり最初のステップは 0 ではなく 1 です。
ウォームアップの 12 ステップを除外したいので `step == RENDER_WARMUP_STEPS + 1`（= 13）にしています。
0 始まりのつもりで書くと、**レンダラが追いついていない古いフレームが冒頭に混ざります**。

### ④ ウォームアップしないと最初のフレームが「前の姿勢」で描かれる

`set_world_poses_from_view()` でカメラを向けた直後の数フレームは、レンダラがまだ**前の姿勢**で描いています。
そのまま逆投影すると実在しないジオメトリの点群になります。
`RENDER_WARMUP_STEPS = 12` ぶん `env.step()` を空回しして追いつかせ、その後で録画を開始します。

### ⑤ `RecordVideo` はメモリを食う

全フレームを溜めてから書き出すため、`--video_length` と解像度の積がそのままメモリになります。
長時間録りたい場合は解像度を下げるか、`OverheadCameraVideo` と同じ**逐次エンコード**に置き換えてください。

### ⑥ 俯瞰カメラの最初のフレームは `None` のことがある

`camera.data.output` がまだ `None` の間は `append()` が黙って何もしません。
そのため俯瞰カメラ映像は**シーン映像より数フレーム短くなる**ことがあります。

### ⑦ `imageio` が必要（`--record_camera` 時のみ）

遅延 import なので通常実行では不要ですが、`--record_camera` を使うときは
`imageio` と mp4 エンコーダ（ffmpeg）が必要です。入っていなければ `pip install imageio[ffmpeg]` で入ります。

### ⑧ 終了時のクローズ順

```python
if camera_video is not None:
    camera_video.close()
env.close()
```

`env.close()` は、まだ録画中のシーン映像をディスクへフラッシュします。
`camera_video.close()` を忘れると mp4 のヘッダが書かれず、**再生できないファイル**が残ります。

---

## 7. 触ってみると面白いパラメータ

| 変えるもの | どうなるか |
|---|---|
| `--video_length` | 録画するステップ数。`480` ≒ 9.6 秒。長くするとメモリ使用量が増える |
| `--video_width` / `--video_height` | シーン映像の解像度。上げると綺麗だがメモリと時間を食う |
| `VIDEO_EYE` / `VIDEO_LOOK_AT`（コード内） | 撮影アングル。真横から撮るとグリッパの開閉が見やすい |
| `--record_camera` | 推定が外れたときの原因切り分けに有効 |
| `RENDER_WARMUP_STEPS`（コード内） | 冒頭に古いフレームが混ざる場合は増やす |
| `quality=8`（`OverheadCameraVideo`） | 俯瞰カメラ映像の画質（imageio のパラメータ） |

---

## 8. 既知の制限

`demo3.md` §8 の制限はすべてそのまま当てはまります。加えて:

- **シーン映像は全フレームをメモリに保持**します（`RecordVideo` の仕様）。
- `--video_length` が短いとピッキングの途中で録画が終わります。
  既定 480 ステップ（約 9.6 秒）は `--episode_length_s 8.0` のエピソードが 1 回収まる長さです。
- 俯瞰カメラ映像は最初の数フレームが欠けることがあります（§6 ⑥）。
- `--record_camera` は `imageio` に依存します（遅延 import なので未使用なら不要）。
- 録画開始のトリガは 1 回だけです。2 本目以降のエピソードは録画されません。

---

## 9. 参考

- ベースタスク: `IsaacContrib-Lift-Cube-Franka-IK-Abs`
- 元になった Isaac Lab のサンプル:
  - `scripts/environments/state_machine/lift_cube_sm.py`（Warp ステートマシン部分）
- 関連デモ:
  - `demo3.md` — **アルゴリズムの説明はこちら**（録画なしの同じデモ）
  - `demo5.md` — LiDAR で障害物を避けながら搬送する AGV のデモ
