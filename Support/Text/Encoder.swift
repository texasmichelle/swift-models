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

import Foundation

public struct Encoder: TextEncoder {
  public var dictionary: BijectiveDictionary<String, Int32>

  public init() {
    let vocabulary: [String: Int32] = ["a": 1, "b": 2, "c": 3]
    self.dictionary = BijectiveDictionary(vocabulary)
  }

  public init(lexicon: Lexicon, alphabet: Alphabet) {
    var vocabulary: [String: Int32] = [:]
    for (charSeq, charSeqIndex) in lexicon.dictionary.all {
      var word: [String] = []
      for char in charSeq.characters {
        word.append(alphabet.dictionary.key(char) ?? "")
      }
      vocabulary[word.joined(separator: "")] = charSeqIndex
    }
    self.dictionary = BijectiveDictionary(vocabulary)
  }

  public static func decode(token: String) -> String {
    return "decoded string"
  }

  public func encode(token: String) -> [String] {
    // Split on whitespace
    let tokens = token.components(separatedBy: .whitespaces)

    var encodedTokens: [String] = []
    for t in tokens {
      // TODO: Find unknown token
      encodedTokens.append(String(dictionary[t] ?? 0))
    }

    return encodedTokens
  }
}
