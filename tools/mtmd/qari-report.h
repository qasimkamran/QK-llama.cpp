// SPDX-License-Identifier: MIT OR LicenseRef-QARI-OCR-COMMERCIAL
// Qari OCR addition: see NOTICE-QARI-OCR.md and LICENSE-QARI-OCR-COMMERCIAL.md

#pragma once

#include <cstddef>
#include <string>

namespace qari {

struct GenerationReport {
    int generatedTokens = 0;
    int decodeCallCount = 0;
    double generationTotalMs = 0.0;
    double decodeTotalMs = 0.0;
    double averageDecodeMs = 0.0;
    double minimumDecodeMs = 0.0;
    double maximumDecodeMs = 0.0;
    double samplingTotalMs = 0.0;
    double consoleOutputTotalMs = 0.0;
    double continuationTotalMs = 0.0;
    double measuredNonDecodeMs = 0.0;
    double tokensPerSecond = 0.0;
};

struct ImageSummary {
    double imageLoadMs = 0.0;
    double tokenizationMs = 0.0;
    double multimodalEvalMs = 0.0;
    double generationMs = 0.0;
    double outputSaveMs = 0.0;
    double totalImageMs = 0.0;
    int outputTokens = 0;
    int contextTokens = 0;
};

void PrintUsage(const char* prog);
void PrintPhase(const char* phase);
void PrintImageHeader(size_t imageIndex, size_t imageCount, const std::string& imagePath);
void PrintTiming(const char* label, double elapsedMs);
void PrintContextTokenCount(int contextTokens);
void PrintContinuationPhase(int continuationIndex);
void PrintContinuationTiming(int continuationIndex, double elapsedMs);
void PrintGenerationReport(const GenerationReport& report);
void PrintImageSummary(const ImageSummary& summary);

} // namespace qari
