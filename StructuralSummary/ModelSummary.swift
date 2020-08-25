import TensorFlow
import StructuralCore

public protocol ModelSummarizable {
  func summary(inputShape: TensorShape) -> String
}

// Base cases.

extension Dense: ModelSummarizable {
  public func summary(inputShape: TensorShape) -> String {
    let t1 = Tensor<Scalar>(repeating: 0, shape: inputShape, on: Device.defaultXLA)

    let dense = Self.self.init(copying: self, to: Device.defaultXLA)
    let output = dense(t1)
    let annotation = "type=\(Self.self)"
    let annotated = output.annotate(annotation)
    return annotated.annotations
  }
}

extension StructuralEmpty: ModelSummarizable {
  public func summary(inputShape: TensorShape) -> String {
    return ""
  }
}

// Inductive cases.

extension StructuralProperty: ModelSummarizable 
where Value: ModelSummarizable {
  public func summary(inputShape: TensorShape) -> String {
    " - \(name): \(value.summary(inputShape: inputShape))"
  }
}

extension StructuralCons: ModelSummarizable
where Value: ModelSummarizable, Next: ModelSummarizable {
  public func summary(inputShape: TensorShape) -> String {
    "\(value.summary(inputShape: inputShape))\n\(next.summary(inputShape: inputShape))"
  }
}

extension StructuralStruct: ModelSummarizable
where Properties: ModelSummarizable {
  public func summary(inputShape: TensorShape) -> String {
    "\(String(describing: type))\n\(properties.summary(inputShape: inputShape))"
  }
}

extension ModelSummarizable
where Self: Structural, Self.StructuralRepresentation: ModelSummarizable {
  public func summary(inputShape: TensorShape) -> String {
    self.structuralRepresentation.summary(inputShape: inputShape)
  }
} 
