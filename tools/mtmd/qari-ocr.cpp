// SPDX-License-Identifier: MIT OR LicenseRef-QARI-OCR-COMMERCIAL
// Qari OCR addition: see NOTICE-QARI-OCR.md and LICENSE-QARI-OCR-COMMERCIAL.md

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "qari-loader.h"
#include "qari-output.h"
#include "qari-report.h"
#include "qari-types.h"

#include <chrono>
#include <clocale>
#include <cstdio>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#endif

namespace {

static bool TokenToPiece(const llama_vocab * vocab, llama_token tok, std::string & outputPiece) {
    char tokenBuffer[256];
    const int pieceLength = llama_token_to_piece(vocab, tok, tokenBuffer, sizeof(tokenBuffer), 0, true);
    if (pieceLength < 0) {
        return false;
    }
    outputPiece.assign(tokenBuffer, pieceLength);
    return true;
}

} // namespace

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

    const llama_vocab * vocab = llama_model_get_vocab(llamaModel);
    auto samplerParams = llama_sampler_chain_default_params();
    llama_sampler * sampler = llama_sampler_chain_init(samplerParams);
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

        mtmd_bitmap * bitmap =
            mtmd_helper_bitmap_init_from_file(mtmdCtx, currentImagePath.c_str());

        const auto bitmapEnd = SteadyClock::now();

        if (!bitmap) {
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

        const mtmd_bitmap * bitmaps[] = { bitmap };

        const std::string userContent =
            std::string(mtmd_default_marker()) + "\n" + options.prompt;

        const std::string fullPrompt =
            "<|im_start|>user\n" +
            userContent +
            "<|im_end|>\n"
            "<|im_start|>assistant\n";

        mtmd_input_text inputText = {
            fullPrompt.c_str(),
            true,
            true,
        };

        mtmd_input_chunks * chunks = mtmd_input_chunks_init();

        if (!chunks) {
            fprintf(
                stderr,
                "error: failed to allocate mtmd_input_chunks\n"
            );

            mtmd_bitmap_free(bitmap);
            llama_sampler_free(sampler);
            mtmd_free(mtmdCtx);
            llama_free(llamaCtx);
            llama_model_free(llamaModel);
            return 1;
        }

        // ------------------------------------------------------------
        // 2. Multimodal tokenisation and image preprocessing
        // ------------------------------------------------------------

        qari::PrintPhase("Tokenising and preprocessing multimodal input");

        const auto tokenizeStart = SteadyClock::now();

        const int tokenizeResult =
            mtmd_tokenize(mtmdCtx, chunks, &inputText, bitmaps, 1);

        const auto tokenizeEnd = SteadyClock::now();

        qari::PrintTiming(
            "Multimodal tokenisation/preprocessing",
            ElapsedMs(tokenizeStart, tokenizeEnd)
        );

        if (tokenizeResult != 0) {
            fprintf(stderr, "error: mtmd_tokenize() failed\n");

            mtmd_input_chunks_free(chunks);
            mtmd_bitmap_free(bitmap);
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

        const int evalResult = mtmd_helper_eval_chunks(
            mtmdCtx,
            llamaCtx,
            chunks,
            nPast,
            0,
            languageModelOptions.nBatch,
            true,
            &nPast
        );

        const auto multimodalEvalEnd = SteadyClock::now();

        qari::PrintTiming(
            "Vision encoding + image prefill",
            ElapsedMs(multimodalEvalStart, multimodalEvalEnd)
        );
        qari::PrintContextTokenCount(static_cast<int>(nPast));

        if (evalResult != 0) {
            fprintf(
                stderr,
                "error: mtmd_helper_eval_chunks() failed\n"
            );

            mtmd_input_chunks_free(chunks);
            mtmd_bitmap_free(bitmap);
            llama_sampler_free(sampler);
            mtmd_free(mtmdCtx);
            llama_free(llamaCtx);
            llama_model_free(llamaModel);
            return 1;
        }

        // ------------------------------------------------------------
        // 4. Autoregressive text generation
        // ------------------------------------------------------------

        std::string outputText;

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

        qari::PrintPhase("Generating output tokens");

        const auto generationStart = SteadyClock::now();

        double samplingTotalMs = 0.0;
        double decodeTotalMs = 0.0;
        double consoleOutputTotalMs = 0.0;
        double continuationTotalMs = 0.0;

        double minimumDecodeMs =
            std::numeric_limits<double>::max();

        double maximumDecodeMs = 0.0;

        int decodeCallCount = 0;
        int generatedTotal = 0;
        int continueRound = 0;

        while (generatedTotal < options.nPredict) {
            bool hitEog = false;
            int generatedThisRound = 0;

            for (; generatedTotal < options.nPredict; ++generatedTotal) {
                // Measure sampling independently from inference.
                const double targetGpuDuty = 0.60;

                const auto sampleStart = SteadyClock::now();

                llama_token tok =
                    llama_sampler_sample(sampler, llamaCtx, -1);

                const auto sampleEnd = SteadyClock::now();

                const double sampleMs =
                    ElapsedMs(sampleStart, sampleEnd);

                samplingTotalMs += sampleMs;

                if (targetGpuDuty > 0.0 && targetGpuDuty < 1.0) {
                    const double requiredIdleMs =
                        sampleMs *
                        ((1.0 / targetGpuDuty) - 1.0);

                    std::this_thread::sleep_for(
                        std::chrono::duration<double, std::milli>(
                            requiredIdleMs
                        )
                    );
                }

                if (llama_vocab_is_eog(vocab, tok)) {
                    hitEog = true;
                    break;
                }

                std::string piece;

                if (!TokenToPiece(vocab, tok, piece)) {
                    fprintf(
                        stderr,
                        "\nerror: TokenToPiece failed\n"
                    );

                    hitEog = false;
                    generatedTotal = options.nPredict;
                    break;
                }

                // Console flushing can be surprisingly expensive on Windows,
                // so measure it separately.
                const auto consoleStart = SteadyClock::now();

                printf("%s", piece.c_str());
                fflush(stdout);

                const auto consoleEnd = SteadyClock::now();

                consoleOutputTotalMs +=
                    ElapsedMs(consoleStart, consoleEnd);

                outputText += piece;
                ++generatedThisRound;

                llama_sampler_accept(sampler, tok);

                llama_batch batch =
                    llama_batch_get_one(&tok, 1);

                // This is the GPU-heavy operation during token generation.
                const auto decodeStart = SteadyClock::now();

                const int decodeResult =
                    llama_decode(llamaCtx, batch);

                const auto decodeEnd = SteadyClock::now();

                const double currentDecodeMs =
                    ElapsedMs(decodeStart, decodeEnd);

                decodeTotalMs += currentDecodeMs;

                minimumDecodeMs =
                    std::min(minimumDecodeMs, currentDecodeMs);

                maximumDecodeMs =
                    std::max(maximumDecodeMs, currentDecodeMs);

                ++decodeCallCount;

                if (decodeResult != 0) {
                    fprintf(
                        stderr,
                        "\nerror: llama_decode failed while generating\n"
                    );

                    hitEog = false;
                    generatedTotal = options.nPredict;
                    break;
                }

                ++nPast;
            }

            if (!hitEog || generatedTotal >= options.nPredict) {
                break;
            }

            if (
                generatedThisRound == 0 ||
                continueRound >= options.maxContinueRounds
            ) {
                break;
            }

            const std::string continuePrompt =
                "<|im_end|>\n"
                "<|im_start|>user\n"
                "Continue exactly where you stopped. "
                "Do not repeat any previous text. "
                "Keep transcribing the remaining page.\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n";

            mtmd_input_text continueText = {
                continuePrompt.c_str(),
                false,
                true,
            };

            mtmd_input_chunks * continueChunks =
                mtmd_input_chunks_init();

            if (!continueChunks) {
                fprintf(
                    stderr,
                    "\nerror: failed to allocate continuation chunks\n"
                );
                break;
            }

            if (
                mtmd_tokenize(
                    mtmdCtx,
                    continueChunks,
                    &continueText,
                    nullptr,
                    0
                ) != 0
            ) {
                fprintf(
                    stderr,
                    "\nerror: failed to tokenize continuation prompt\n"
                );

                mtmd_input_chunks_free(continueChunks);
                break;
            }

            qari::PrintContinuationPhase(continueRound + 1);

            const auto continuationStart = SteadyClock::now();

            const int continuationResult =
                mtmd_helper_eval_chunks(
                    mtmdCtx,
                    llamaCtx,
                    continueChunks,
                    nPast,
                    0,
                    languageModelOptions.nBatch,
                    true,
                    &nPast
                );

            const auto continuationEnd = SteadyClock::now();

            const double currentContinuationMs =
                ElapsedMs(continuationStart, continuationEnd);

            continuationTotalMs += currentContinuationMs;

            qari::PrintContinuationTiming(continueRound + 1, currentContinuationMs);

            if (continuationResult != 0) {
                fprintf(
                    stderr,
                    "\nerror: failed to evaluate continuation prompt\n"
                );

                mtmd_input_chunks_free(continueChunks);
                break;
            }

            mtmd_input_chunks_free(continueChunks);
            ++continueRound;
        }

        const auto generationEnd = SteadyClock::now();

        printf("\n");

        // ------------------------------------------------------------
        // 5. Generation timing report
        // ------------------------------------------------------------

        const double generationTotalMs =
            ElapsedMs(generationStart, generationEnd);

        const double averageDecodeMs =
            decodeCallCount > 0
                ? decodeTotalMs /
                    static_cast<double>(decodeCallCount)
                : 0.0;

        if (decodeCallCount == 0) {
            minimumDecodeMs = 0.0;
        }

        const double tokensPerSecond =
            generationTotalMs > 0.0
                ? static_cast<double>(generatedTotal) /
                    (generationTotalMs / 1000.0)
                : 0.0;

        const double measuredNonDecodeMs =
            std::max(
                0.0,
                generationTotalMs -
                    decodeTotalMs -
                    continuationTotalMs
            );

        qari::GenerationReport generationReport;
        generationReport.generatedTokens = generatedTotal;
        generationReport.decodeCallCount = decodeCallCount;
        generationReport.generationTotalMs = generationTotalMs;
        generationReport.decodeTotalMs = decodeTotalMs;
        generationReport.averageDecodeMs = averageDecodeMs;
        generationReport.minimumDecodeMs = minimumDecodeMs;
        generationReport.maximumDecodeMs = maximumDecodeMs;
        generationReport.samplingTotalMs = samplingTotalMs;
        generationReport.consoleOutputTotalMs = consoleOutputTotalMs;
        generationReport.continuationTotalMs = continuationTotalMs;
        generationReport.measuredNonDecodeMs = measuredNonDecodeMs;
        generationReport.tokensPerSecond = tokensPerSecond;
        qari::PrintGenerationReport(generationReport);

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

        if (!qari::SaveOutputText(options, currentImagePath, imageIdx, outputText)) {
            mtmd_input_chunks_free(chunks);
            mtmd_bitmap_free(bitmap);
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
        imageSummary.generationMs = generationTotalMs;
        imageSummary.outputSaveMs = ElapsedMs(outputSaveStart, outputSaveEnd);
        imageSummary.totalImageMs = ElapsedMs(imageTotalStart, imageTotalEnd);
        imageSummary.outputTokens = generatedTotal;
        imageSummary.contextTokens = static_cast<int>(nPast);
        qari::PrintImageSummary(imageSummary);

        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bitmap);
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
