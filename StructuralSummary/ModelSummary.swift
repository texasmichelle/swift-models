import TensorFlow
import StructuralCore

public protocol ModelSummarizable {
  var summary: String { get }
}

// Base cases.

extension Dense: ModelSummarizable {
  public var summary: String {
//    return "Dense"
    let t1 = Tensor<Float>(repeating: 0, shape: [1, 2, 3], on: Device.defaultXLA)
    let annotated = t1.annotate("type=Tensor<Float>")
    return annotated.annotations
  }
}

extension StructuralEmpty: ModelSummarizable {
  public var summary: String {
    return ""
  }
}

// Inductive cases.

extension StructuralProperty: ModelSummarizable 
where Value: ModelSummarizable {
  public var summary: String {
    " - \(name): \(value.summary)"
  }
}

extension StructuralCons: ModelSummarizable
where Value: ModelSummarizable, Next: ModelSummarizable {
  public var summary: String {
    "\(value.summary)\n\(next.summary)"
  }
}

extension StructuralStruct: ModelSummarizable
where Properties: ModelSummarizable {
  public var summary: String {
    "\(String(describing: type))\n\(properties.summary)"
  }
}

extension ModelSummarizable
where Self: Structural, Self.StructuralRepresentation: ModelSummarizable {
  public var summary: String {
    self.structuralRepresentation.summary
  }
} 
