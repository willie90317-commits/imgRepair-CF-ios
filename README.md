# CodeFormer iOS

在 iPhone 上離線執行 CodeFormer 人臉修復:選照片 → 偵測人臉(iOS Vision)→ 對齊 → CodeFormer(Core ML)→ 貼回 → 存回相簿。

## 結構

| 路徑 | 內容 |
|---|---|
| `convert/cf_loader.py` | 只用 PyTorch 載入官方 CodeFormer 網路 |
| `convert/export_model.py` | 適合 Core ML 的等價改寫(argmax 查表、手寫 attention、fidelity 當輸入) |
| `convert/verify_wrapper.py` | 驗證改寫版與原版輸出一致(Windows/Linux/macOS 皆可) |
| `convert/convert_coreml.py` | 轉成 `CodeFormer.mlpackage` 並用 Core ML 實際推論驗證(僅 macOS) |
| `convert/reference_pipeline.py` | iPhone 端演算法的 Python 參考實作,與官方流程比對 |
| `ios/` | SwiftUI App 原始碼與 XcodeGen 專案設定 |
| `.github/workflows/build-ios.yml` | 雲端 Mac:轉換模型 → 編譯 → 產出未簽章 `.ipa` |

## 雲端編譯(不需要 Mac)

1. 推送到 GitHub 後,到 repo 的 **Actions → Build iOS app** 等它跑完(第一次約 20–30 分鐘,之後模型有快取會快很多)。
2. 下載 artifact **CodeFormer-ipa** 裡的 `CodeFormer-unsigned.ipa`。
3. 在 Windows 安裝 [Sideloadly](https://sideloadly.io/),iPhone 用傳輸線接上電腦,拖入 `.ipa`,登入 Apple ID 後安裝。
4. iPhone:「設定 → 一般 → VPN 與裝置管理」信任你的 Apple ID;iOS 16 以上還要開啟「設定 → 隱私權與安全性 → 開發者模式」。

免費 Apple ID 簽的 App 7 天後失效,重新用 Sideloadly 安裝即可。

## 在 Mac 上編譯

```bash
pip install torch==2.5.1 coremltools==8.3.0 numpy==1.26.4 pillow
git clone --depth 1 https://github.com/sczhou/CodeFormer.git /tmp/CodeFormer
cd convert && python convert_coreml.py /tmp/CodeFormer ../ios/Models /tmp/CodeFormer/inputs/cropped_faces/0143.png
cd ../ios && brew install xcodegen && xcodegen generate && open CodeFormerRepair.xcodeproj
```

## 限制

- 第一版只做人臉修復,不含背景放大、上色、補全、影片。
- 貼回使用官方的方形柔邊遮罩,沒有使用人臉解析模型,髮際線等邊緣可能比桌面版略生硬。
- CodeFormer 採用 S-Lab License 1.0,**僅限非商業用途**。
