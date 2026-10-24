import SwiftUI

/// Before/after view: drag horizontally to move the divider.
struct CompareView: View {
    let before: UIImage
    let after: UIImage?
    @State private var split: CGFloat = 0.5

    var body: some View {
        GeometryReader { geo in
            let rect = fit(before.size, in: geo.size)
            ZStack {
                Image(uiImage: after ?? before)
                    .resizable()
                    .frame(width: rect.width, height: rect.height)
                    .position(x: rect.midX, y: rect.midY)

                if after != nil {
                    Image(uiImage: before)
                        .resizable()
                        .frame(width: rect.width, height: rect.height)
                        .mask(alignment: .leading) {
                            Rectangle().frame(width: rect.width * split)
                        }
                        .position(x: rect.midX, y: rect.midY)

                    let x = rect.minX + rect.width * split
                    Rectangle()
                        .fill(.white)
                        .frame(width: 2, height: rect.height)
                        .position(x: x, y: rect.midY)
                    Circle()
                        .fill(.white)
                        .frame(width: 30, height: 30)
                        .overlay(Image(systemName: "chevron.left.chevron.right")
                            .font(.system(size: 11, weight: .bold))
                            .foregroundStyle(.black.opacity(0.7)))
                        .shadow(radius: 2)
                        .position(x: x, y: rect.midY)
                    chip("原圖").position(x: rect.minX + 30, y: rect.minY + 18)
                    chip("結果").position(x: rect.maxX - 30, y: rect.minY + 18)
                }
            }
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 0).onChanged { value in
                guard rect.width > 0 else { return }
                split = min(1, max(0, (value.location.x - rect.minX) / rect.width))
            })
        }
    }

    private func chip(_ text: String) -> some View {
        Text(text)
            .font(.caption.bold())
            .foregroundStyle(.white)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(.black.opacity(0.55), in: Capsule())
    }

    private func fit(_ image: CGSize, in box: CGSize) -> CGRect {
        guard image.width > 0, image.height > 0 else { return .zero }
        let s = min(box.width / image.width, box.height / image.height)
        let w = image.width * s, h = image.height * s
        return CGRect(x: (box.width - w) / 2, y: (box.height - h) / 2, width: w, height: h)
    }
}
