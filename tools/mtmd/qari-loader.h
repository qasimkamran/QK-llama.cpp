// SPDX-License-Identifier: MIT OR LicenseRef-QARI-OCR-COMMERCIAL
// Qari OCR addition: see NOTICE-QARI-OCR.md and LICENSE-QARI-OCR-COMMERCIAL.md

#pragma once

#include "llama.h"
#include "mtmd.h"
#include <string>

namespace qari {

struct LanguageModelOptions {
    std::string modelPath;
    int nGPULayers = 99;
    int nCtx = 8192;
    int nBatch = 1024;
    int nUBatch = 256;
    bool noPerf = false;
};

struct VisionModelOptions {
    std::string modelPath;
    bool useGPU = true;
    bool printTimings = true;
    int nThreads = 0;
    int imageMinTokens = 256;
    int imageMaxTokens = 1024;
};

bool LoadGGMLBackend();
llama_model* LoadLanguageModel(const std::string modelPath, const LanguageModelOptions& opts);
llama_context* CreateLanguageModelContext(llama_model* llamaModel, const LanguageModelOptions& opts);
mtmd_context* LoadMultimodalContext(const std::string modelPath, const llama_model* llamaModel, const VisionModelOptions& opts);

} // namespace qari
