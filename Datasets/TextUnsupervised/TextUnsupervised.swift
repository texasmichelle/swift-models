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

import Batcher
import Foundation
import ModelSupport
import TensorFlow

public enum TextUnsupervisedVariant: String {
    /// - Source: [Einstein AI WikiText-103](
    ///             https://blog.einstein.ai/
    ///             the-wikitext-long-term-dependency-language-modeling-dataset/).
    case wikiText103 = "WikiText103"
    /// Default variant.
    /// - Source: [Einstein AI WikiText-2](
    ///             https://blog.einstein.ai/
    ///             the-wikitext-long-term-dependency-language-modeling-dataset/).
    case wikiText2 = "WikiText2"
    /// - Source: [Learning to Discover, Ground, and Use Words with Segmental Neural Language Models](
    ///             https://www.aclweb.org/anthology/P19-1645.pdf).
    case wordSeg = "WordSeg"
}

private protocol TextUnsupervisedVariantDetails {
    var variant: TextUnsupervisedVariant { get set }
    var location: URL { get set }
    var archiveFileName: String { get set }
    var archiveExtension: String { get set }
    var trainingFilePath: String { get set }
    var validationFilePath: String { get set }
    var encodedFileName: String? {get set}
}

public struct TextUnsupervised {
    private struct WikiText103Details: TextUnsupervisedVariantDetails {
        var variant = TextUnsupervisedVariant.wikiText103
        var location = URL(string: "https://s3.amazonaws.com/fast-ai-nlp/")!
        var archiveFileName = "wikitext-103"
        var archiveExtension = "tgz"
        var trainingFilePath = "train.csv"
        var validationFilePath = "test.csv"
        var encodedFileName: String? = nil
    }

    private struct WikiText2Details: TextUnsupervisedVariantDetails {
        var variant = TextUnsupervisedVariant.wikiText2
        var location = URL(string: "https://storage.googleapis.com/s4tf-hosted-binaries/datasets/WikiText2/")!
        var archiveFileName = "wikitext-2"
        var archiveExtension = "tgz"
        var trainingFilePath = "train.csv"
        var validationFilePath = "test.csv"
        var encodedFileName: String? = "wikitext-2-encoded"
    }

    private struct WordSegDetails: TextUnsupervisedVariantDetails {
        var variant = TextUnsupervisedVariant.wordSeg
        var location = URL(string: "https://s3.eu-west-2.amazonaws.com/k-kawakami")!
        var archiveFileName = "seg"
        var archiveExtension = "zip"
        var trainingFilePath = "br/br-text/tr.txt"
        var validationFilePath = "br/br-text/va.txt"
        var encodedFileName: String? = nil
    }

    public let trainingRaw: [String]
    public let trainingDataset: LanguageModelDataset<[[Int]]>
    public let validationDataset: LanguageModelDataset<[[Int]]>
    public let encoder: TextEncoder?
    public let variant: TextUnsupervisedVariant
    private let variantDetails: TextUnsupervisedVariantDetails
    public let alphabet: Alphabet
    public let lexicon: Lexicon

    public init(
        encoder: TextEncoder? = nil,
        variant: TextUnsupervisedVariant = TextUnsupervisedVariant.wikiText2,
        trainingBatchSize: Int = 8, validationBatchSize: Int = 4, sequenceLength: Int = 1024,
        trainingDocumentCount: Int = 4, validationDocumentCount: Int = 4
    ) {
        do {
            self.encoder = encoder

            self.variant = variant
/*
            let variantDetails: TextUnsupervisedVariantDetails
            switch variant {
            case .wikiText103:
                variantDetails = WikiText103Details()
            case .wikiText2:
                variantDetails = WikiText2Details()
            case .wordSeg:
                variantDetails = WordSegDetails()
            }
            self.variantDetails = variantDetails
*/
            self.variantDetails = Self.getVariantDetails(variant)

            let localStorageDirectory: URL = FileManager.default.temporaryDirectory
                .appendingPathComponent(
                    variant.rawValue, isDirectory: true)
            self.trainingRaw = try TextUnsupervised.loadTrainingRaw(
                localStorageDirectory: localStorageDirectory, 
                variantDetails: variantDetails, documentCount: trainingDocumentCount)
            self.alphabet = try TextUnsupervised.loadAlphabet(
                localStorageDirectory: localStorageDirectory, 
                variant: variant, documentCount: trainingDocumentCount)
            self.lexicon = try TextUnsupervised.loadLexicon(
                localStorageDirectory: localStorageDirectory, 
                variant: variant, documentCount: trainingDocumentCount)
            self.trainingDataset = try TextUnsupervised.loadTraining(
                localStorageDirectory: localStorageDirectory, encoder: encoder,
                variantDetails: variantDetails, batchSize: trainingBatchSize,
                sequenceLength: sequenceLength, documentCount: trainingDocumentCount)
            self.validationDataset = try TextUnsupervised.loadValidation(
                localStorageDirectory: localStorageDirectory, encoder: encoder,
                variantDetails: variantDetails, batchSize: validationBatchSize,
                sequenceLength: sequenceLength, documentCount: validationDocumentCount)
        } catch {
            fatalError("Could not load dataset for \(variant): \(error)")
        }
    }

    private static func downloadIfNotPresent(
        to directory: URL, variantDetails: TextUnsupervisedVariantDetails, downloadEncodedFile: Bool
    ) {
        let downloadPath = directory.appendingPathComponent(variantDetails.variant.rawValue).path
        let directoryExists = FileManager.default.fileExists(atPath: downloadPath)
        let contentsOfDir = try? FileManager.default.contentsOfDirectory(atPath: downloadPath)
        let directoryEmpty = (contentsOfDir == nil) || (contentsOfDir!.isEmpty)

        guard !directoryExists || directoryEmpty else { return }

        // Downloads and extracts dataset files.
        let _ = DatasetUtilities.downloadResource(
            filename: (downloadEncodedFile ? variantDetails.encodedFileName! : variantDetails.archiveFileName),
            fileExtension: variantDetails.archiveExtension,
            remoteRoot: variantDetails.location, localStorageDirectory: directory, extract: true)
    }

    private static func readCSV(in file: URL) throws -> [String] {
        let rawText = try! String(contentsOf: file, encoding: .utf8)
        var rows = rawText.components(separatedBy: "\"\n\"")
        // Removing the initial '"'.
        rows[0] = String(rows[0].dropFirst())
        // Removing the last '"\n'
        rows[rows.indices.last!] = String(rows.last!.dropLast(2))
        return rows
    }

    private static func readTXT(in file: URL) throws -> [String] {
        let rawText = try! String(contentsOf: file, encoding: .utf8)
        let rows = rawText.components(separatedBy: "\n")
        return rows
    }

    private static func readEncoded(in file: URL) throws -> [Int] {
        let rawText = try! String(contentsOf: file, encoding: .utf8)
        let rows = rawText.components(separatedBy: "\n")
        var tokens: Array<Int> = Array()
        for row in rows {
            guard let encoded = Int(row) else { continue }
            tokens.append(encoded)
        }
        return tokens
    }

    private static func embedding(for string: String, encoder: TextEncoder) -> [Int] {
        let tokens = encoder.encode(token: string)
        // TODO(michellecasbon): Decide how to prevent OOV or choose a better ID (probably not 0).
        let ids = tokens.map { Int(encoder.dictionary[$0] ?? 0) }
        return ids
    }

    private static func loadRaw(
        named name: String, in directory: URL,
        variantDetails: TextUnsupervisedVariantDetails,
        documentCount: Int = 4
    ) throws -> [String] {
        downloadIfNotPresent(to: directory, variantDetails: variantDetails, downloadEncodedFile: false)

        let path = directory.appendingPathComponent("\(variantDetails.archiveFileName)/\(name)")
        let documentsFull: [String]
        switch variantDetails.variant {
        case .wordSeg: 
            documentsFull = try readTXT(in: path)
        case .wikiText103, .wikiText2:
            documentsFull = try readCSV(in: path)
        }
        let documents = Array(documentsFull[0..<min(documentCount, documentsFull.count)])
        return documents
    }

    private static func getVariantDetails(
        _ variant: TextUnsupervisedVariant) -> TextUnsupervisedVariantDetails {
        switch variant {
        case .wikiText103:
            return WikiText103Details()
        case .wikiText2:
            return WikiText2Details()
        case .wordSeg:
            return WordSegDetails()
        }
    }

    public static func loadAlphabet(
        in directory: URL,
        variant: TextUnsupervisedVariant,
        documentCount: Int = 4,
        eos: String = "</s>",
        eow: String = "</w>",
        pad: String = "</pad>"
        ) throws -> Alphabet {
        let variantDetails = Self.getVariantDetails(variant)
        downloadIfNotPresent(
            to: directory, variantDetails: variantDetails,
            downloadEncodedFile: false)

        var path = directory.appendingPathComponent(variantDetails.archiveFileName)
        path = path.appendingPathComponent(variantDetails.trainingFilePath)
        let documentsFull: [String]
        switch variantDetails.variant {
        case .wordSeg: 
            documentsFull = try readTXT(in: path)
        case .wikiText103, .wikiText2:
            documentsFull = try readCSV(in: path)
        }
        let documents = Array(documentsFull[0..<min(documentCount, documentsFull.count)])

        var letters: Set<Character> = []
        for document in [documents] {
            for sentence in document {
                for character in sentence {
                    letters.insert(character)
                }
            }
        }

        // Sort the letters to make it easier to interpret ints vs letters.
        var sorted = Array(letters)
        sorted.sort()

        return Alphabet(sorted, eos: eos, eow: eow, pad: pad)
    }

    public static func loadLexicon(
        in directory: URL,
        variant: TextUnsupervisedVariant,
        documentCount: Int = 4
        ) throws -> Lexicon {
        let alphabet = try Self.loadAlphabet(in: directory, variant: variant, documentCount: documentCount)

        let variantDetails = Self.getVariantDetails(variant)
        downloadIfNotPresent(
            to: directory, variantDetails: variantDetails,
            downloadEncodedFile: false)

        var path = directory.appendingPathComponent(variantDetails.archiveFileName)
        path = path.appendingPathComponent(variantDetails.trainingFilePath)
        let documentsFull: [String]
        switch variantDetails.variant {
        case .wordSeg: 
            documentsFull = try readTXT(in: path)
        case .wikiText103, .wikiText2:
            documentsFull = try readCSV(in: path)
        }
        let documents = Array(documentsFull[0..<min(documentCount, documentsFull.count)])

        let trainingCharSeq: [CharacterSequence]
        do {
             trainingCharSeq = try DataSet.convertDataset(
                 documents, alphabet: alphabet)
        } catch {
            print("Unexpected error")
            trainingCharSeq = []
        }

        return Lexicon(from: trainingCharSeq, alphabet: alphabet,
            minFreq: 3)
    }


    /// Returns a LanguageModelDataset by processing files specified by 'variantDetails' which
    /// resides in 'directory'.
    ///
    /// Download the files if not present. If encoder is nil, skip encoding
    /// and download the encoded file instead.
    ///
    /// - Parameter name: name of the dataset. Ususally 'train' or 'test'.
    /// - Parameter directory: directory that files are read from.
    /// - Parameter encoder: encoder used for encoding text.
    /// - Parameter variantDetails: an object containing information of filename, location, etc.
    /// - Parameter batchSize: number of sequences in a batch.
    /// - Parameter sequenceLength: number of characters in a sequence.
    /// - Parameter documentCount: number of documents to proceed. (Refer func readCSV() to see how
    ///   a text file is chunked into documents.)
    private static func loadDirectory(
        named name: String, in directory: URL, encoder: TextEncoder?,
        variantDetails: TextUnsupervisedVariantDetails, batchSize: Int, sequenceLength: Int,
        documentCount: Int = 4
    ) throws -> LanguageModelDataset<[[Int]]> {
        // Determine whether to download encoded files.
        let downloadEncodedFile: Bool
        if encoder == nil && variantDetails.encodedFileName != nil {
            downloadEncodedFile = true
        } else {
            downloadEncodedFile = false
        }
        downloadIfNotPresent(to: directory, variantDetails: variantDetails, downloadEncodedFile: downloadEncodedFile)

        var encodedDocs: [[Int]] = []
        if let encoder = encoder {
            let path = directory.appendingPathComponent("\(variantDetails.archiveFileName)/\(name)")
            let documentsFull: [String]
            switch variantDetails.variant {
            case .wordSeg: 
                documentsFull = try readTXT(in: path)
            case .wikiText103, .wikiText2:
                documentsFull = try readCSV(in: path)
            }
            let documents = Array(documentsFull[0..<min(documentCount, documentsFull.count)])
            encodedDocs = documents.map { embedding(for: $0, encoder: encoder) }
        } else {
            precondition(variantDetails.encodedFileName != nil,
                "encoder must be provided when encodedFileName is nil.")
            let pathPrefix = directory.appendingPathComponent("\(variantDetails.encodedFileName!)/\(name)").path
            for i in 0..<documentCount {
                encodedDocs += [
                  try readEncoded(in: URL(fileURLWithPath: "\(pathPrefix)/doc_\(i).txt"))
                ]
            }
        }

        return LanguageModelDataset(
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            numericalizedTexts: encodedDocs,
            lengths: encodedDocs.map { $0.count },
            dropLast: false
        )
    }

    private static func loadTrainingRaw(
        localStorageDirectory: URL,
        variantDetails: TextUnsupervisedVariantDetails,
        documentCount: Int
    )
        throws
        -> [String]
    {
        return try loadRaw(
            named: variantDetails.trainingFilePath, in: localStorageDirectory,
            variantDetails: variantDetails, documentCount: documentCount)
    }

    private static func loadAlphabet(
        localStorageDirectory: URL,
        variant: TextUnsupervisedVariant,
        documentCount: Int
    )
        throws
        -> Alphabet
    {
        return try loadAlphabet(
            in: localStorageDirectory,
            variant: variant, documentCount: documentCount)
    }

    private static func loadLexicon(
        localStorageDirectory: URL,
        variant: TextUnsupervisedVariant,
        documentCount: Int
    )
        throws
        -> Lexicon
    {
        return try loadLexicon(
            in: localStorageDirectory,
            variant: variant, documentCount: documentCount)
    }

    private static func loadTraining(
        localStorageDirectory: URL, encoder: TextEncoder?,
        variantDetails: TextUnsupervisedVariantDetails, batchSize: Int, sequenceLength: Int,
        documentCount: Int
    )
        throws
        -> LanguageModelDataset<[[Int]]>
    {
        return try loadDirectory(
            named: variantDetails.trainingFilePath, in: localStorageDirectory, encoder: encoder,
            variantDetails: variantDetails, batchSize: batchSize, sequenceLength: sequenceLength,
            documentCount: documentCount)
    }

    private static func loadValidation(
        localStorageDirectory: URL, encoder: TextEncoder?,
        variantDetails: TextUnsupervisedVariantDetails, batchSize: Int, sequenceLength: Int,
        documentCount: Int
    )
        throws
        -> LanguageModelDataset<[[Int]]>
    {
        return try loadDirectory(
            named: variantDetails.validationFilePath, in: localStorageDirectory, encoder: encoder,
            variantDetails: variantDetails, batchSize: batchSize, sequenceLength: sequenceLength,
            documentCount: documentCount)
    }

  private static func makeAlphabet(
    datasets training: [String],
    _ otherSequences: [String]?...,
    eos: String = "</s>",
    eow: String = "</w>",
    pad: String = "</pad>"
  ) -> Alphabet {
    var letters: Set<Character> = []

    for dataset in otherSequences + [training] {
      guard let dataset = dataset else { continue }
      for sentence in dataset {
        for character in sentence {
          letters.insert(character)
        }
      }
    }

    // Sort the letters to make it easier to interpret ints vs letters.
    var sorted = Array(letters)
    sorted.sort()

    return Alphabet(sorted, eos: eos, eow: eow, pad: pad)
  }
}
