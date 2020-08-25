import TensorFlow
import StructuralSummary
import StructuralCore

struct Model: ModelSummarizable, Layer {
  var dense1 = Dense<Float>(inputSize: 1, outputSize: 1)
  var dense2 = Dense<Float>(inputSize: 1, outputSize: 1)

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    let layer1 = dense1(input)
    let layer2 = dense2(layer1)
    return layer2
  }
}

let model = Model()
print(model.summary(inputShape: [1, 1, 1, 1]))

/// In the future, this extension will be compiler-generated.
extension Model: Structural {
  public typealias StructuralRepresentation =
    StructuralStruct<
      StructuralCons<StructuralProperty<Dense<Float>>,
      StructuralCons<StructuralProperty<Dense<Float>>,
      StructuralEmpty>>>

  public var structuralRepresentation: StructuralRepresentation {
    get {
      StructuralStruct(Model.self,
        StructuralCons(StructuralProperty("dense1", dense1, isMutable: true),
        StructuralCons(StructuralProperty("dense2", dense2, isMutable: true),
        StructuralEmpty())))
    }
    set { fatalError() }
  }

  public init(structuralRepresentation: StructuralRepresentation) { fatalError() }
}
