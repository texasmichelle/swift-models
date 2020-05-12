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

import Datasets
import ModelSupport
import TensorFlow
import TextModels
import XCTest

final class TextUnsupervisedTests: XCTestCase {
    func testCreateWikiText103WithBpe() {
        do {
            let gpt2 = try GPT2()
            let dataset = TextUnsupervised(encoder: gpt2.bpe)

            var totalCount = 0
            for example in dataset.trainingDataset {
                XCTAssertEqual(example.first.shape[0], 1024)
                XCTAssertEqual(example.second.shape[0], 1024)
                totalCount += 1
            }
            for example in dataset.validationDataset {
                XCTAssertEqual(example.first.shape[0], 1024)
                XCTAssertEqual(example.second.shape[0], 1024)
                totalCount += 1
            }
            XCTAssertEqual(totalCount, 24)
        } catch {
            XCTFail(error.localizedDescription)
        }
    }

    func testCreateWikiText103WithoutBpe() {
        let dataset = TextUnsupervised(variant: .wikiText103)

        var totalCount = 0
        for example in dataset.trainingDataset {
            XCTAssertEqual(example.first.shape[0], 1024)
            XCTAssertEqual(example.second.shape[0], 1024)
            totalCount += 1
        }
        for example in dataset.validationDataset {
            XCTAssertEqual(example.first.shape[0], 1024)
            XCTAssertEqual(example.second.shape[0], 1024)
            totalCount += 1
        }
        XCTAssertEqual(totalCount, 24)
    }

    func testCreateWikiText2WithBpe() {
        do {
            let gpt2 = try GPT2()
            let dataset = TextUnsupervised(encoder: gpt2.bpe)

            var totalCount = 0
            for example in dataset.trainingDataset {
                XCTAssertEqual(example.first.shape[0], 1024)
                XCTAssertEqual(example.second.shape[0], 1024)
                totalCount += 1
            }
            for example in dataset.validationDataset {
                XCTAssertEqual(example.first.shape[0], 1024)
                XCTAssertEqual(example.second.shape[0], 1024)
                totalCount += 1
            }
            XCTAssertEqual(totalCount, 12)
        } catch {
            XCTFail(error.localizedDescription)
        }
    }

    func testCreateWikiText2WithoutBpe() {
        let dataset = TextUnsupervised()

        var totalCount = 0
        for example in dataset.trainingDataset {
            XCTAssertEqual(example.first.shape[0], 1024)
            XCTAssertEqual(example.second.shape[0], 1024)
            totalCount += 1
        }
        for example in dataset.validationDataset {
            XCTAssertEqual(example.first.shape[0], 1024)
            XCTAssertEqual(example.second.shape[0], 1024)
            totalCount += 1
        }
        XCTAssertEqual(totalCount, 12)
    }

    func testCreateWordSeg() {
        let variant = TextUnsupervisedVariant.wordSeg
        // TODO: Remove dir
        let dir: URL = FileManager.default.temporaryDirectory.appendingPathComponent(
            variant.rawValue, isDirectory: true)
        var encoder = Encoder()
        let dataset = TextUnsupervised(
            encoder: encoder, variant: variant,
            trainingBatchSize: 1, validationBatchSize: 1,
            sequenceLength: 1)
        var alphabet: Alphabet
        do {
            alphabet = try TextUnsupervised.loadAlphabet(in: dir, variant: .wordSeg)
        } catch {
            alphabet = dataset.alphabet
            XCTFail("Unable to load alphabet")
        }
        // let encoder = Encoder()

        let trainingRaw = dataset.trainingRaw
        let trainingCharSeq: [CharacterSequence]
        do {
            trainingCharSeq = try DataSet.convertDataset(
                trainingRaw, alphabet: alphabet)
        } catch {
            print("Unexpected error")
            trainingCharSeq = []
        }
        // let lexicon = Lexicon(from: trainingCharSeq, alphabet: alphabet)
        var lexicon: Lexicon
        do {
            lexicon = try TextUnsupervised.loadLexicon(in: dir, variant: .wordSeg)
        } catch {
            lexicon = dataset.lexicon
            XCTFail("Unable to load alphabet")
        }

        encoder = Encoder(lexicon: lexicon, alphabet: alphabet)
        let dataset2 = TextUnsupervised(
            encoder: encoder, variant: variant,
            trainingBatchSize: 1, validationBatchSize: 1,
            sequenceLength: 1)

        let modelConfig = SNLM.Parameters(
            chrVocab: alphabet, strVocab: lexicon)
        let wordSeg = SNLM(parameters: modelConfig)

        var totalCount = 0
        print("dataset2.trainingDataset: \(dataset2.trainingDataset.count)")
        for example in dataset2.trainingDataset {
//             print("example: \(example)")
            // XCTAssertEqual(example.first.shape[0], 1024)
            // XCTAssertEqual(example.second.shape[0], 1024)
            totalCount += 1
        }
        for example in dataset2.validationDataset {
            // XCTAssertEqual(example.first.shape[0], 1024)
            // XCTAssertEqual(example.second.shape[0], 1024)
            totalCount += 1
        }
        XCTAssertEqual(totalCount, 2)
    }
}

extension TextUnsupervisedTests {
    static var allTests = [
        // The WikiText103 dataset is large and should not run in kokoro.
        // Uncomment the following 2 lines to run individually.
        // ("testCreateWikiText103WithBpe", testCreateWikiText103WithBpe),
        // ("testCreateWikiText103WithoutBpe", testCreateWikiText103WithoutBpe),

//         ("testCreateWikiText2WithBpe", testCreateWikiText2WithBpe),
//        ("testCreateWikiText2WithoutBpe", testCreateWikiText2WithoutBpe),
        ("testCreateWordSeg", testCreateWordSeg),
    ]
}
