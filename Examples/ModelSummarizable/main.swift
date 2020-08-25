import TensorFlow
import StructuralSummary
import StructuralCore

struct Model: ModelSummarizable {
  var dense1 = Dense<Float>(inputSize: 1, outputSize: 1)
  var dense2 = Dense<Float>(inputSize: 1, outputSize: 1)
  var dense3 = Dense<Float>(inputSize: 1, outputSize: 1)
  var flatten = Flatten<Float>()

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    let layer1 = dense1(input)
    let layer2 = layer1.reshaped(to: [1, 4])
    let layer3 = dense2(layer2)
    let layer4 = dense3(layer3)
    return flatten(layer4)
  }
}

let model = Model()
print(model.summary(inputShape: [1, 1]))

/// In the future, this extension will be compiler-generated.
extension Model: Structural {
  public typealias StructuralRepresentation =
    StructuralStruct<
      StructuralCons<StructuralProperty<Dense<Float>>,
      StructuralCons<StructuralProperty<Dense<Float>>,
      StructuralCons<StructuralProperty<Dense<Float>>,
      StructuralCons<StructuralProperty<Flatten<Float>>,
      StructuralEmpty>>>>>

  public var structuralRepresentation: StructuralRepresentation {
    get {
      StructuralStruct(Model.self,
        StructuralCons(StructuralProperty("dense1", dense1, isMutable: true),
        StructuralCons(StructuralProperty("dense2", dense2, isMutable: true),
        StructuralCons(StructuralProperty("dense3", dense3, isMutable: true),
        StructuralCons(StructuralProperty("flatten", flatten, isMutable: true),
        StructuralEmpty())))))
    }
    set { fatalError() }
  }

  public init(structuralRepresentation: StructuralRepresentation) { fatalError() }
}
