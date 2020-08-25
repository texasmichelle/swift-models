import TensorFlow
import StructuralSummary
import StructuralCore

struct Model: ModelSummarizable {
  var dense1 = Dense<Float>(inputSize: 1, outputSize: 1)
  var dense2 = Dense<Float>(inputSize: 1, outputSize: 1)
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
