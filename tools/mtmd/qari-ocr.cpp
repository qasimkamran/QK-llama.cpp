// SPDX-License-Identifier: MIT OR LicenseRef-QARI-OCR-COMMERCIAL
// Qari OCR addition: see NOTICE-QARI-OCR.md and LICENSE-QARI-OCR-COMMERCIAL.md

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "qari-gen.h"
#include "qari-loader.h"
#include "qari-output.h"
#include "qari-report.h"
#include "qari-types.h"

#include <chrono>
#include <clocale>
#include <cstdio>
#include <string>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#endif

using SteadyClock = std::chrono::steady_clock;

static double ElapsedMs(
    const SteadyClock::time_point & start,
    const SteadyClock::time_point & end
) {
    return std::chrono::duration<double, std::milli>(
        end - start
    ).count();
}

static int QariOcrMain(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    qari::Options options = qari::ParseOptions(argc, argv);
    if (!qari::OptionsValid(options)) {
        qari::PrintUsage(argv[0]);
        return 1;
    }

    if (options.mmprojPath.empty()) {
        options.mmprojPath = options.modelPath;
    }

    const std::vector<std::string> imagePaths = qari::CollectImagePaths(options);
    if (imagePaths.empty()) {
        return 1;
    }

    if (!qari::LoadGGMLBackend()) {
        fprintf(stderr, "error: failed to load ggml backend\n");
        return 1;
    }

    qari::LanguageModelOptions languageModelOptions;
    languageModelOptions.modelPath = options.modelPath;
    languageModelOptions.nGPULayers = options.nGpuLayers;

    llama_model * llamaModel = qari::LoadLanguageModel(options.modelPath, languageModelOptions);
    if (!llamaModel) {
        return 1;
    }

    llama_context * llamaCtx = qari::CreateLanguageModelContext(llamaModel, languageModelOptions);
    if (!llamaCtx) {
        llama_model_free(llamaModel);
        return 1;
    }

    qari::VisionModelOptions visionModelOptions;
    visionModelOptions.modelPath = options.mmprojPath;
    visionModelOptions.imageMinTokens = options.imageMinTokens;
    visionModelOptions.imageMaxTokens = options.imageMaxTokens;

    mtmd_context * mtmdCtx = qari::LoadMultimodalContext(options.mmprojPath, llamaModel, visionModelOptions);
    if (!mtmdCtx) {
        llama_free(llamaCtx);
        llama_model_free(llamaModel);
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(llamaModel);

    auto samplerParams = llama_sampler_chain_default_params();
    llama_sampler* sampler = llama_sampler_chain_init(samplerParams);
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    for (size_t imageIdx = 0; imageIdx < imagePaths.size(); ++imageIdx) {
        const std::string & currentImagePath = imagePaths[imageIdx];

        qari::PrintImageHeader(imageIdx, imagePaths.size(), currentImagePath);

        const auto imageTotalStart = SteadyClock::now();

        llama_memory_clear(llama_get_memory(llamaCtx), true);
        llama_sampler_reset(sampler);
        llama_perf_context_reset(llamaCtx);

        // ------------------------------------------------------------
        // 1. Load and decode the image file
        // ------------------------------------------------------------

        qari::PrintPhase("Loading image");
        const auto bitmapStart = SteadyClock::now();

        qari::MultimodalInput multimodalInput;
        multimodalInput.bitmap = qari::PrepareBitmap(mtmdCtx, currentImagePath);

        const auto bitmapEnd = SteadyClock::now();

        if (!multimodalInput.bitmap) {
            fprintf(
                stderr,
                "error: failed to load image: %s\n",
                currentImagePath.c_str()
            );

            llama_sampler_free(sampler);
            mtmd_free(mtmdCtx);
            llama_free(llamaCtx);
            llama_model_free(llamaModel);
            return 1;
        }

        qari::PrintTiming("Image loading", ElapsedMs(bitmapStart, bitmapEnd));

        // ------------------------------------------------------------
        // 2. Multimodal tokenisation and image preprocessing
        // ------------------------------------------------------------

        qari::PrintPhase("Tokenising and preprocessing multimodal input");

        const auto tokenizeStart = SteadyClock::now();

        multimodalInput = qari::GetMultimodalInputFromMultimodalBitmap(
            mtmdCtx,
            multimodalInput.bitmap,
            options.prompt
        );

        const auto tokenizeEnd = SteadyClock::now();

        qari::PrintTiming(
            "Multimodal tokenisation/preprocessing",
            ElapsedMs(tokenizeStart, tokenizeEnd)
        );

        if (!multimodalInput.chunks) {
            qari::FreeMultimodalInput(multimodalInput);
            llama_sampler_free(sampler);
            mtmd_free(mtmdCtx);
            llama_free(llamaCtx);
            llama_model_free(llamaModel);
            return 1;
        }

        // ------------------------------------------------------------
        // 3. Vision encoding and language-model image prefill
        //
        // mtmd's own print_timings output should further divide this
        // into "image slice encoded" and "image decoded".
        // ------------------------------------------------------------

        qari::PrintPhase("Evaluating vision input and image-token prefill");

        llama_pos nPast = 0;

        const auto multimodalEvalStart = SteadyClock::now();

        const bool evalOk = qari::EvaluateMultimodalInput(
            mtmdCtx,
            llamaCtx,
            multimodalInput.chunks,
            languageModelOptions.nBatch,
            nPast
        );

        const auto multimodalEvalEnd = SteadyClock::now();

        qari::PrintTiming(
            "Vision encoding + image prefill",
            ElapsedMs(multimodalEvalStart, multimodalEvalEnd)
        );
        qari::PrintContextTokenCount(static_cast<int>(nPast));

        if (!evalOk) {
            qari::FreeMultimodalInput(multimodalInput);
            llama_sampler_free(sampler);
            mtmd_free(mtmdCtx);
            llama_free(llamaCtx);
            llama_model_free(llamaModel);
            return 1;
        }

        // ------------------------------------------------------------
        // 4. Autoregressive text generation
        // ------------------------------------------------------------

        printf("\n");

        if (imagePaths.size() > 1) {
            fprintf(
                stderr,
                "[%zu/%zu] OCR: %s\n",
                imageIdx + 1,
                imagePaths.size(),
                currentImagePath.c_str()
            );
        }

        qari::GenerationResult generationResult = qari::GenerateText(
            mtmdCtx,
            llamaCtx,
            sampler,
            vocab,
            options,
            languageModelOptions,
            nPast
        );

        printf("\n");

        // ------------------------------------------------------------
        // 5. Generation timing report
        // ------------------------------------------------------------

        qari::PrintGenerationReport(generationResult.report);

        // Print llama.cpp's own internal performance counters.
        fprintf(
            stderr,
            "\n[OCR PERF] llama.cpp context performance:\n"
        );

        llama_perf_context_print(llamaCtx);

        // ------------------------------------------------------------
        // 6. Save output
        // ------------------------------------------------------------

        const auto outputSaveStart = SteadyClock::now();

        if (!qari::SaveOutputText(options, currentImagePath, imageIdx, generationResult.outputText)) {
            qari::FreeMultimodalInput(multimodalInput);
            llama_sampler_free(sampler);
            mtmd_free(mtmdCtx);
            llama_free(llamaCtx);
            llama_model_free(llamaModel);
            return 1;
        }

        const auto outputSaveEnd = SteadyClock::now();
        const auto imageTotalEnd = SteadyClock::now();

        qari::ImageSummary imageSummary;
        imageSummary.imageLoadMs = ElapsedMs(bitmapStart, bitmapEnd);
        imageSummary.tokenizationMs = ElapsedMs(tokenizeStart, tokenizeEnd);
        imageSummary.multimodalEvalMs = ElapsedMs(multimodalEvalStart, multimodalEvalEnd);
        imageSummary.generationMs = generationResult.report.generationTotalMs;
        imageSummary.outputSaveMs = ElapsedMs(outputSaveStart, outputSaveEnd);
        imageSummary.totalImageMs = ElapsedMs(imageTotalStart, imageTotalEnd);
        imageSummary.outputTokens = generationResult.report.generatedTokens;
        imageSummary.contextTokens = static_cast<int>(nPast);
        qari::PrintImageSummary(imageSummary);

        qari::FreeMultimodalInput(multimodalInput);
    }

    llama_sampler_free(sampler);
    mtmd_free(mtmdCtx);
    llama_free(llamaCtx);
    llama_model_free(llamaModel);

    return 0;
}

#if defined(_WIN32)
static std::string WideToUtf8(const wchar_t * wideString) {
    if (!wideString) {
        return {};
    }
    int size = WideCharToMultiByte(CP_UTF8, 0, wideString, -1, nullptr, 0, nullptr, nullptr);
    if (size <= 1) {
        return {};
    }
    std::string utf8((size_t) size - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, wideString, -1, utf8.data(), size, nullptr, nullptr);
    return utf8;
}

int main() {
    int argcW = 0;
    wchar_t ** argvW = CommandLineToArgvW(GetCommandLineW(), &argcW);
    if (!argvW || argcW <= 0) {
        fprintf(stderr, "error: failed to parse command line\n");
        return 1;
    }

    std::vector<std::string> argvStorage;
    argvStorage.reserve((size_t) argcW);
    std::vector<char *> argvUtf8;
    argvUtf8.reserve((size_t) argcW);

    for (int i = 0; i < argcW; ++i) {
        argvStorage.emplace_back(WideToUtf8(argvW[i]));
    }
    for (auto & arg : argvStorage) {
        argvUtf8.push_back(arg.data());
    }

    LocalFree(argvW);
    return QariOcrMain((int) argvUtf8.size(), argvUtf8.data());
}
#else
int main(int argc, char ** argv) {
    return QariOcrMain(argc, argv);
}
#endif
