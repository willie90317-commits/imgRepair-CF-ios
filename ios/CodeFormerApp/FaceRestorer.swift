import CoreImage
import CoreImage.CIFilterBuiltins
import CoreML
import UIKit
import Vision

enum RestoreError: LocalizedError {
    case modelMissing
    case invalidImage
    case modelOutput

    var errorDescription: String? {
        switch self {
        case .modelMissing: return "找不到 CodeFormer 模型"
        case .invalidImage: return "無法讀取這張圖片"
        case .modelOutput: return "模型輸出格式不正確"
        }
    }
}

struct RestoreResult {
    let image: UIImage
    let faceCount: Int
}

/// 2D affine map in top-left-origin image coordinates:
/// x' = a*x + b*y + tx,  y' = c*x + d*y + ty
struct Affine {
    var a, b, c, d, tx, ty: Double

    func inverted() -> Affine {
        let det = a * d - b * c
        let ia = d / det, ib = -b / det, ic = -c / det, id = a / det
        return Affine(a: ia, b: ib, c: ic, d: id, tx: -(ia * tx + ib * ty), ty: -(ic * tx + id * ty))
    }

    /// Converts a map defined on OpenCV pixel indices (pixel centres at integers) to
    /// continuous coordinates (pixel centres at +0.5), which Core Image uses.
    func continuous() -> Affine {
        Affine(a: a, b: b, c: c, d: d,
               tx: tx + 0.5 - 0.5 * (a + b), ty: ty + 0.5 - 0.5 * (c + d))
    }

    /// Same map for Core Image's bottom-left origin, from a source of height srcH to a destination of height dstH.
    func coreImageTransform(srcHeight: Double, dstHeight: Double) -> CGAffineTransform {
        CGAffineTransform(a: a, b: -c, c: -b, d: d,
                          tx: b * srcHeight + tx, ty: dstHeight - d * srcHeight - ty)
    }

    /// Least-squares similarity transform (rotation, uniform scale, translation) mapping src onto dst.
    static func similarity(from src: [CGPoint], to dst: [CGPoint]) -> Affine {
        let n = Double(src.count)
        let sx = src.map(\.x).reduce(0, +) / n, sy = src.map(\.y).reduce(0, +) / n
        let dx = dst.map(\.x).reduce(0, +) / n, dy = dst.map(\.y).reduce(0, +) / n
        var num1 = 0.0, num2 = 0.0, den = 0.0
        for (p, q) in zip(src, dst) {
            let px = p.x - sx, py = p.y - sy, qx = q.x - dx, qy = q.y - dy
            num1 += px * qx + py * qy
            num2 += px * qy - py * qx
            den += px * px + py * py
        }
        let a = num1 / den, b = num2 / den
        return Affine(a: a, b: -b, c: b, d: a,
                      tx: dx - (a * sx - b * sy), ty: dy - (b * sx + a * sy))
    }
}

/// Mirrors convert/reference_pipeline.py: Vision landmarks -> FFHQ 5-point alignment ->
/// CodeFormer (Core ML) -> square soft-mask paste-back (facelib, without the parsing model).
final class FaceRestorer: @unchecked Sendable {
    static let faceSize = 512.0
    /// FFHQ 5-point template (left eye, right eye, nose tip, left mouth corner, right mouth corner).
    static let template: [CGPoint] = [
        CGPoint(x: 192.98138, y: 239.94708), CGPoint(x: 318.90277, y: 240.1936),
        CGPoint(x: 256.63416, y: 314.01935), CGPoint(x: 201.26117, y: 371.41043),
        CGPoint(x: 313.08905, y: 371.15118),
    ]
    static let borderColor = CIColor(red: 132 / 255, green: 133 / 255, blue: 135 / 255)

    private let model: MLModel
    private let imageConstraint: MLImageConstraint
    private let context: CIContext
    private let sRGB = CGColorSpace(name: CGColorSpace.sRGB)!

    init() throws {
        guard let url = Bundle.main.url(forResource: "CodeFormer", withExtension: "mlmodelc") else {
            throw RestoreError.modelMissing
        }
        let config = MLModelConfiguration()
        config.computeUnits = .all
        model = try MLModel(contentsOf: url, configuration: config)
        guard let constraint = model.modelDescription.inputDescriptionsByName["image"]?.imageConstraint else {
            throw RestoreError.modelOutput
        }
        imageConstraint = constraint
        context = CIContext(options: [.workingColorSpace: sRGB, .outputColorSpace: sRGB])
    }

    func restore(_ image: UIImage, fidelity: Double) throws -> RestoreResult {
        guard let cgImage = image.cgImage else { throw RestoreError.invalidImage }
        let size = CGSize(width: cgImage.width, height: cgImage.height)
        let original = CIImage(cgImage: cgImage)
        let extent = original.extent
        let landmarkSets = try detectLandmarks(cgImage, size: size)

        var result = original
        for points in landmarkSets {
            // Vision points are continuous; the template uses OpenCV pixel indices.
            let indexPoints = points.map { CGPoint(x: $0.x - 0.5, y: $0.y - 0.5) }
            let align = Affine.similarity(from: indexPoints, to: Self.template).continuous()
            let crop = try alignedFace(original, align: align, imageHeight: size.height)
            let restored = try runModel(crop, fidelity: fidelity)
            result = paste(restored, onto: result, align: align, imageHeight: size.height, extent: extent)
        }

        let outSpace = (cgImage.colorSpace?.model == .rgb ? cgImage.colorSpace : nil) ?? sRGB
        guard let out = context.createCGImage(result, from: extent, format: .RGBA8, colorSpace: outSpace) else {
            throw RestoreError.invalidImage
        }
        return RestoreResult(image: UIImage(cgImage: out), faceCount: landmarkSets.count)
    }

    // MARK: - Landmarks

    /// Returns 5 points per face in top-left image coordinates, ordered like the template.
    private func detectLandmarks(_ cgImage: CGImage, size: CGSize) throws -> [[CGPoint]] {
        let request = VNDetectFaceLandmarksRequest()
        try VNImageRequestHandler(cgImage: cgImage, orientation: .up).perform([request])
        let faces = request.results ?? []

        func topLeft(_ region: VNFaceLandmarkRegion2D?) -> [CGPoint] {
            (region?.pointsInImage(imageSize: size) ?? []).map { CGPoint(x: $0.x, y: size.height - $0.y) }
        }
        func centroid(_ pts: [CGPoint]) -> CGPoint? {
            guard !pts.isEmpty else { return nil }
            return CGPoint(x: pts.map(\.x).reduce(0, +) / Double(pts.count),
                           y: pts.map(\.y).reduce(0, +) / Double(pts.count))
        }

        return faces.compactMap { face -> [CGPoint]? in
            guard let lm = face.landmarks,
                  let eyeA = centroid(topLeft(lm.leftPupil)) ?? centroid(topLeft(lm.leftEye)),
                  let eyeB = centroid(topLeft(lm.rightPupil)) ?? centroid(topLeft(lm.rightEye))
            else { return nil }
            let nose = topLeft(lm.noseCrest) + topLeft(lm.nose)
            let lips = topLeft(lm.outerLips)
            guard !nose.isEmpty, lips.count >= 2 else { return nil }

            let (leftEye, rightEye) = eyeA.x <= eyeB.x ? (eyeA, eyeB) : (eyeB, eyeA)
            let mid = CGPoint(x: (leftEye.x + rightEye.x) / 2, y: (leftEye.y + rightEye.y) / 2)
            // Nose tip: the nose point farthest from the eyes, robust to head roll.
            let tip = nose.max { hypot($0.x - mid.x, $0.y - mid.y) < hypot($1.x - mid.x, $1.y - mid.y) }!
            // Mouth corners: lip extremes along the eye axis.
            let axis = CGPoint(x: rightEye.x - leftEye.x, y: rightEye.y - leftEye.y)
            let along = { (p: CGPoint) in p.x * axis.x + p.y * axis.y }
            let mouthLeft = lips.min { along($0) < along($1) }!
            let mouthRight = lips.max { along($0) < along($1) }!
            return [leftEye, rightEye, tip, mouthLeft, mouthRight]
        }
    }

    // MARK: - Crop / model / paste

    private func alignedFace(_ image: CIImage, align: Affine, imageHeight: Double) throws -> CGImage {
        let rect = CGRect(x: 0, y: 0, width: Self.faceSize, height: Self.faceSize)
        let warped = image.transformed(by: align.coreImageTransform(srcHeight: imageHeight, dstHeight: Self.faceSize))
        let crop = warped.composited(over: CIImage(color: Self.borderColor).cropped(to: rect)).cropped(to: rect)
        guard let cg = context.createCGImage(crop, from: rect, format: .RGBA8, colorSpace: sRGB) else {
            throw RestoreError.invalidImage
        }
        return cg
    }

    private func runModel(_ face: CGImage, fidelity: Double) throws -> CIImage {
        let w = try MLMultiArray(shape: [1], dataType: .float32)
        w[0] = NSNumber(value: Float(fidelity))
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "image": try MLFeatureValue(cgImage: face, constraint: imageConstraint, options: nil),
            "fidelity": MLFeatureValue(multiArray: w),
        ])
        let output = try model.prediction(from: input)
        guard let buffer = output.featureValue(for: "restored")?.imageBufferValue else {
            throw RestoreError.modelOutput
        }
        return CIImage(cvPixelBuffer: buffer)
    }

    private func paste(_ face: CIImage, onto base: CIImage, align: Affine, imageHeight: Double, extent: CGRect) -> CIImage {
        let back = align.inverted().coreImageTransform(srcHeight: Self.faceSize, dstHeight: imageHeight)
        let faceRect = CGRect(x: 0, y: 0, width: Self.faceSize, height: Self.faceSize)
        let warpedFace = face.cropped(to: faceRect).transformed(by: back)

        // Square mask, eroded and feathered in proportion to the face size (facelib's formula).
        let black = CIImage(color: .black).cropped(to: extent)
        let mask = CIImage(color: .white).cropped(to: faceRect).transformed(by: back)
            .composited(over: black).cropped(to: extent)
        let scale = sqrt(abs(align.a * align.d - align.b * align.c))   // aligned px per image px
        let edge = Double(Int(Self.faceSize / scale) / 20)
        let erode = CIFilter.morphologyMinimum()
        erode.inputImage = mask.clampedToExtent()
        erode.radius = Float(edge + 1)
        let blur = CIFilter.gaussianBlur()
        blur.inputImage = erode.outputImage?.cropped(to: extent).clampedToExtent()
        blur.radius = Float(max(0.3 * (edge - 1) + 0.8, 0.5))
        let soft = (blur.outputImage ?? mask).cropped(to: extent)

        let blend = CIFilter.blendWithMask()
        blend.inputImage = warpedFace
        blend.backgroundImage = base
        blend.maskImage = soft
        return (blend.outputImage ?? base).cropped(to: extent)
    }
}
