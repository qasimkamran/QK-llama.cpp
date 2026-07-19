// SPDX-License-Identifier: MIT OR LicenseRef-QARI-OCR-COMMERCIAL
// Qari OCR addition: see NOTICE-QARI-OCR.md and LICENSE-QARI-OCR-COMMERCIAL.md

#pragma once

#include <llama.h>
#include <mtmd.h>
#include "qari-loader.h"
#include "qari-report.h"
#include "qari-types.h"

#include <string>

namespace qari {

struct MultimodalInput {
    mtmd_bitmap* bitmap = nullptr;
    mtmd_input_chunks* chunks = nullptr;
};

struct GenerationStepResult {
    bool ok = true;
    bool hitEog = false;
    bool decodeAttempted = false;
    std::string piece;
    double samplingMs = 0.0;
    double decodeMs = 0.0;
    double consoleOutputMs = 0.0;
};

struct GenerationResult {
    bool ok = true;
    std::string outputText;
    GenerationReport report;
};

MultimodalInput GetMultimodalInputFromMultimodalBitmap(
    mtmd_context* mtmdCtx,
    mtmd_bitmap* bitmap,
    const std::string& prompt
);
mtmd_bitmap* PrepareBitmap(mtmd_context* mtmdCtx, const std::string& imagePath);
MultimodalInput PrepareMultimodalInput(
    mtmd_context* mtmdCtx,
    const std::string& imagePath,
    const std::string& prompt
);
bool EvaluateMultimodalInput(
    mtmd_context* mtmdCtx,
    llama_context* llamaCtx,
    mtmd_input_chunks* chunks,
    int nBatch,
    llama_pos& nPast
);
std::string BuildContinuationPrompt();
bool TokenToPiece(const llama_vocab* vocab, llama_token token, std::string& piece);
bool DecodeGeneratedToken(llama_context* llamaCtx, llama_token token, double& decodeMs);
bool EvaluateContinuationPrompt(
    mtmd_context* mtmdCtx,
    llama_context* llamaCtx,
    int nBatch,
    int continuationIndex,
    llama_pos& nPast,
    double& elapsedMs
);
GenerationStepResult GenerateNextToken(
    llama_context* llamaCtx,
    llama_sampler* sampler,
    const llama_vocab* vocab,
    double targetGpuDuty
);
GenerationResult GenerateText(
    mtmd_context* mtmdCtx,
    llama_context* llamaCtx,
    llama_sampler* sampler,
    const llama_vocab* vocab,
    const Options& options,
    const LanguageModelOptions& languageModelOptions,
    llama_pos& nPast
);
void FreeMultimodalInput(MultimodalInput& input);

} // namespace qari
