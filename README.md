## 概要
フィジカルAIにおけるIsaacSim関連の技術

## ソースコード

| コードファイル名 | 概要 |
| ------ | ------- |
| demo.py | マップ障害物を理解させてからLLMにルートを決めさせ4足歩行を歩かせる |
| demo2.py | 4足歩行の巡回 |
| demo3.py | カメラでRGB検出してアームロボットでピッキング |


## Windowsでの起動方法
```
isaaclab.bat -p demo.py --viz kit
```

## demo3.pyを動かすにあたっての修正
下記を修正
source/isaaclab/isaaclab/utils/math.py の unproject_depth() (1246行目付近)
```
-    pixels = torch.nn.functional.pad(img_indices, (0, 0, 1, 0), mode="constant", value=1.0)
+    # append the homogeneous coordinate so each column is (u, v, 1); the normalization below divides
+    # by the last row, which must therefore be the homogeneous one and not the v index
+    pixels = torch.nn.functional.pad(img_indices, (0, 0, 0, 1), mode="constant", value=1.0)
```
内容: 同次座標の 1 を行方向の先頭ではなく末尾にパディングするよう変更。

なぜ必要だったか: 元のコードでは各列が (1, u, v) になっていました。しかし直後の
points = points / points[:, -1, :].unsqueeze(1)  # normalize by last coordinate
は最終行が同次座標の 1 であることを前提に割り算しているため、v(縦画素インデックス)で割ってしまい 、逆投影された点群が完全に壊れる
(u, v, 1) にすることで、K⁻¹ @ (u,v,1)ᵀ という正しいピンホール逆投影にする。

demo3.py は create_pointcloud_from_depth() → unproject_depth() 経由で深度画像を世界座標に逆投影してキューブ位置を推定しているので、この修正なしでは視覚推定がまったく機能しません。

補足: ライブラリではなく demo3.py 内で吸収した回避策

以下は共有アセット設定を触らず、スクリプト内のローカルな env_cfg 書き換えで対処しています（math.py 以外に変更なし）:

- patch_franka_usd_path() — Isaac 6.0 でコンテンツサーバ上の Robots/FrankaEmika/panda_instanceable.usd が Legacy/ 配下へ移動したため、FRANKA_PANDA_CFG の既定パスが 404 になる。spawn の usd_path をローカルで差し替え。
- CameraCfg(update_latest_camera_pose=True) — スクリプトから set_world_poses_from_view() でカメラを向けるため、これがないと camera.data.pos_w が spawn 時の姿勢を返し続け、外部パラメータと画像が食い違う。

なお demo3.py などはまだ未追跡（??）で、math.py の修正もコミットされていない作業ツリー上の変更のままです。