// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

public protocol TextEncoder {
    var dictionary: BijectiveDictionary<String, Int32> { get set }
    func encode(token: String) -> [String]
    static func decode(token: String) -> String

}

public enum EncoderVariant {
    /// - Source: [Learning to Discover, Ground, and Use Words with Segmental
    ///             Neural Language Models](
    ///             https://www.aclweb.org/anthology/P19-1645.pdf).
    case wordSeg
    /// - Source: [Language Models are Unsupervised Multitask Learners](
    ///             https://cdn.openai.com/better-language-models/
    ///             language_models_are_unsupervised_multitask_learners.pdf).
    case gpt2
    /// Default variant.
    /// - Source: [RoBERTa: A Robustly Optimized BERT Pretraining Approach](
    ///             https://arxiv.org/pdf/1907.11692.pdf).
    case roberta
}

