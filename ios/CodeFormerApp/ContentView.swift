import Photos
import PhotosUI
import SwiftUI

@MainActor
final class RestoreViewModel: ObservableObject {
    @Published var original: UIImage?
    @Published var restored: UIImage?
    @Published var fidelity = 0.7
    @Published var isWorking = false
    @Published var status = ""
    @Published var isError = false

    private var restorer: FaceRestorer?

    func load(_ item: PhotosPickerItem?) async {
        guard let item else { return }
        restored = nil
        setStatus("")
        do {
            guard let data = try await item.loadTransferable(type: Data.self),
                  let image = UIImage(data: data) else {
                setStatus("無法讀取這張照片", error: true)
                return
            }
            original = image.normalizedUp()
        } catch {
            setStatus(error.localizedDescription, error: true)
        }
    }

    func restore() async {
        guard let image = original else { return }
        isWorking = true
        setStatus(restorer == nil ? "第一次使用,正在載入模型…" : "修復中…")
        let fidelity = fidelity
        let cached = restorer
        do {
            let (engine, result) = try await Task.detached(priority: .userInitiated) {
                let engine = try cached ?? FaceRestorer()
                return (engine, try engine.restore(image, fidelity: fidelity))
            }.value
            restorer = engine
            if result.faceCount == 0 {
                restored = nil
                setStatus("沒有偵測到人臉", error: true)
            } else {
                restored = result.image
                setStatus("完成:修復了 \(result.faceCount) 張人臉")
            }
        } catch {
            setStatus(error.localizedDescription, error: true)
        }
        isWorking = false
    }

    func save() async {
        guard let image = restored else { return }
        do {
            try await PHPhotoLibrary.shared().performChanges {
                PHAssetChangeRequest.creationRequestForAsset(from: image)
            }
            setStatus("已儲存到「照片」App 的最近項目")
        } catch {
            setStatus("儲存失敗:請在「設定」允許加入照片", error: true)
        }
    }

    private func setStatus(_ text: String, error: Bool = false) {
        status = text
        isError = error
    }
}

struct ContentView: View {
    @StateObject private var vm = RestoreViewModel()
    @State private var pickerItem: PhotosPickerItem?

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                preview
                if vm.original != nil {
                    controls
                }
            }
            .padding()
            .navigationTitle("CodeFormer")
            .toolbar {
                if vm.original != nil {
                    ToolbarItem(placement: .navigationBarTrailing) {
                        PhotosPicker(selection: $pickerItem, matching: .images) {
                            Image(systemName: "photo.on.rectangle")
                        }
                        .disabled(vm.isWorking)
                    }
                }
            }
            .onChange(of: pickerItem) { item in
                Task { await vm.load(item) }
            }
        }
    }

    @ViewBuilder
    private var preview: some View {
        if let original = vm.original {
            CompareView(before: original, after: vm.restored)
                .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 16))
                .clipShape(RoundedRectangle(cornerRadius: 16))
                .overlay {
                    if vm.isWorking {
                        ProgressView().controlSize(.large)
                            .padding(24)
                            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
                    }
                }
        } else {
            PhotosPicker(selection: $pickerItem, matching: .images) {
                VStack(spacing: 12) {
                    Image(systemName: "face.smiling")
                        .font(.system(size: 52))
                    Text("選擇照片").font(.title3.bold())
                    Text("會自動偵測並修復照片中的人臉")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 16))
            }
            .buttonStyle(.plain)
        }
    }

    private var controls: some View {
        VStack(spacing: 14) {
            VStack(spacing: 6) {
                HStack {
                    Text("保真度 w").font(.subheadline).foregroundStyle(.secondary)
                    Spacer()
                    Text(String(format: "%.2f", vm.fidelity)).font(.subheadline.bold()).monospacedDigit()
                }
                Slider(value: $vm.fidelity, in: 0...1, step: 0.05)
                HStack {
                    Text("← 畫質優先")
                    Spacer()
                    Text("更像本人 →")
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            .disabled(vm.isWorking)

            if !vm.status.isEmpty {
                Text(vm.status)
                    .font(.footnote)
                    .foregroundStyle(vm.isError ? .red : .secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            Button {
                Task { await vm.restore() }
            } label: {
                Text(vm.isWorking ? "處理中…" : "開始修復")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 6)
            }
            .buttonStyle(.borderedProminent)
            .disabled(vm.isWorking)

            if let restored = vm.restored {
                HStack(spacing: 12) {
                    Button {
                        Task { await vm.save() }
                    } label: {
                        Label("儲存到相簿", systemImage: "square.and.arrow.down")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    ShareLink(item: Image(uiImage: restored),
                              preview: SharePreview("修復結果", image: Image(uiImage: restored))) {
                        Label("分享", systemImage: "square.and.arrow.up")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                }
                .disabled(vm.isWorking)
            }
        }
    }
}

extension UIImage {
    /// Bakes EXIF orientation into the pixels so Vision and Core Image see an upright image.
    func normalizedUp() -> UIImage {
        guard imageOrientation != .up else { return self }
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = scale
        return UIGraphicsImageRenderer(size: size, format: format).image { _ in
            draw(in: CGRect(origin: .zero, size: size))
        }
    }
}
