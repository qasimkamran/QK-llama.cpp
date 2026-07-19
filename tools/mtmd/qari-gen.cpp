// SPDX-License-Identifier: MIT OR LicenseRef-QARI-OCR-COMMERCIAL
// Qari OCR addition: see NOTICE-QARI-OCR.md and LICENSE-QARI-OCR-COMMERCIAL.md

#include "qari-gen.h"

#include "mtmd-helper.h"
#include "qari-report.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <thread>

namespace qari {

namespace {

using SteadyClock = std::chrono::steady_clock;

constexpr double TargetGpuDuty = 0.60;

struct GenerationMetrics {
    double minimumDecodeMs = std::numeric_limits<double>::max();
    double maximumDecodeMs = 0.0;
};

struct GenerationRoundResult {
    bool ok = true;
    bool hitEog = false;
    int generatedTokens = 0;
};

double ElapsedMs(
    const SteadyClock::time_point& start,
    const SteadyClock::time_point& end
)
{
    return std::chrono::duration<double, std::milli>(
        end - start
    ).count();
}

void AccumulateGenerationStep(
    GenerationResult& result,
    const GenerationStepResult& step,
    GenerationMetrics& metrics
)
{
    result.report.samplingTotalMs += step.samplingMs;
    result.report.consoleOutputTotalMs += step.consoleOutputMs;

    if (!step.decodeAttempted)
        return;

    result.report.decodeTotalMs += step.decodeMs;
    metrics.minimumDecodeMs = std::min(metrics.minimumDecodeMs, step.decodeMs);
    metrics.maximumDecodeMs = std::max(metrics.maximumDecodeMs, step.decodeMs);
    ++result.report.decodeCallCount;
}

void AcceptGeneratedStep(
    GenerationResult& result,
    const GenerationStepResult& step,
    llama_pos& nPast
)
{
    result.outputText += step.piece;
    ++nPast;
}

bool ShouldEvaluateContinuation(
    const GenerationRoundResult& round,
    int generatedTotal,
    int continueRound,
    const Options& options
)
{
    if (!round.ok || !round.hitEog)
        return false;

    if (generatedTotal >= options.nPredict)
        return false;

    if (round.generatedTokens == 0)
        return false;

    return continueRound < options.maxContinueRounds;
}

void FinalizeGenerationReport(
    GenerationResult& result,
    int generatedTotal,
    const GenerationMetrics& metrics,
    const SteadyClock::time_point& generationStart,
    const SteadyClock::time_point& generationEnd
)
{
    result.report.generatedTokens = generatedTotal;
    result.report.generationTotalMs = ElapsedMs(generationStart, generationEnd);
    result.report.averageDecodeMs = 0.0;
    if (result.report.decodeCallCount > 0) {
        result.report.averageDecodeMs =
            result.report.decodeTotalMs /
            static_cast<double>(result.report.decodeCallCount);
    }

    result.report.minimumDecodeMs = 0.0;
    if (result.report.decodeCallCount > 0) {
        result.report.minimumDecodeMs = metrics.minimumDecodeMs;
    }

    result.report.maximumDecodeMs = metrics.maximumDecodeMs;
    result.report.tokensPerSecond = 0.0;
    if (result.report.generationTotalMs > 0.0) {
        result.report.tokensPerSecond =
            static_cast<double>(generatedTotal) /
            (result.report.generationTotalMs / 1000.0);
    }

    result.report.measuredNonDecodeMs =
        std::max(
            0.0,
            result.report.generationTotalMs -
                result.report.decodeTotalMs -
                result.report.continuationTotalMs
        );
}

} // namespace

mtmd_bitmap* PrepareBitmap(mtmd_context* mtmdCtx, const std::string& imagePath)
{
    if (!mtmdCtx || imagePath.empty())
        return nullptr;

    return mtmd_helper_bitmap_init_from_file(mtmdCtx, imagePath.c_str());
}

MultimodalInput GetMultimodalInputFromMultimodalBitmap(
    mtmd_context* mtmdCtx,
    mtmd_bitmap* bitmap,
    const std::string& prompt
)
{
    MultimodalInput input;
    input.bitmap = bitmap;

    if (!mtmdCtx || !bitmap || prompt.empty())
        return input;

    const mtmd_bitmap* bitmaps[] = { bitmap };

    const std::string userContent =
        std::string(mtmd_default_marker()) + "\n" + prompt;

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

    input.chunks = mtmd_input_chunks_init();
    if (!input.chunks) {
        fprintf(stderr, "error: failed to allocate mtmd_input_chunks\n");
        return input;
    }

    if (mtmd_tokenize(mtmdCtx, input.chunks, &inputText, bitmaps, 1) != 0) {
        fprintf(stderr, "error: mtmd_tokenize() failed\n");
        mtmd_input_chunks_free(input.chunks);
        input.chunks = nullptr;
        return input;
    }

    return input;
}

MultimodalInput PrepareMultimodalInput(
    mtmd_context* mtmdCtx,
    const std::string& imagePath,
    const std::string& prompt
)
{
    MultimodalInput input;
    input.bitmap = PrepareBitmap(mtmdCtx, imagePath);
    if (!input.bitmap)
        return {};

    input = GetMultimodalInputFromMultimodalBitmap(
        mtmdCtx,
        input.bitmap,
        prompt
    );
    if (!input.chunks) {
        mtmd_bitmap_free(input.bitmap);
        input.bitmap = nullptr;
        return {};
    }

    return input;
}

bool EvaluateMultimodalInput(
    mtmd_context* mtmdCtx,
    llama_context* llamaCtx,
    mtmd_input_chunks* chunks,
    int nBatch,
    llama_pos& nPast
)
{
    if (!mtmdCtx || !llamaCtx || !chunks)
        return false;

    const int evalResult = mtmd_helper_eval_chunks(
        mtmdCtx,
        llamaCtx,
        chunks,
        nPast,
        0,
        nBatch,
        true,
        &nPast
    );

    if (evalResult != 0) {
        fprintf(stderr, "error: mtmd_helper_eval_chunks() failed\n");
        return false;
    }

    return true;
}

std::string BuildContinuationPrompt()
{
    return
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Continue exactly where you stopped. "
        "Do not repeat any previous text. "
        "Keep transcribing the remaining page.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n";
}

bool TokenToPiece(const llama_vocab* vocab, llama_token token, std::string& piece)
{
    char tokenBuffer[256];
    const int pieceLength = llama_token_to_piece(vocab, token, tokenBuffer, sizeof(tokenBuffer), 0, true);
    if (pieceLength < 0)
        return false;

    piece.assign(tokenBuffer, pieceLength);
    return true;
}

bool DecodeGeneratedToken(llama_context* llamaCtx, llama_token token, double& decodeMs)
{
    if (!llamaCtx)
        return false;

    llama_batch batch =
        llama_batch_get_one(&token, 1);

    const auto decodeStart = SteadyClock::now();
    const int decodeResult = llama_decode(llamaCtx, batch);
    const auto decodeEnd = SteadyClock::now();

    decodeMs = ElapsedMs(decodeStart, decodeEnd);
    return decodeResult == 0;
}

bool EvaluateContinuationPrompt(
    mtmd_context* mtmdCtx,
    llama_context* llamaCtx,
    int nBatch,
    int continuationIndex,
    llama_pos& nPast,
    double& elapsedMs
)
{
    if (!mtmdCtx || !llamaCtx)
        return false;

    const std::string continuePrompt = BuildContinuationPrompt();
    mtmd_input_text continueText = {
        continuePrompt.c_str(),
        false,
        true,
    };

    mtmd_input_chunks* continueChunks =
        mtmd_input_chunks_init();

    if (!continueChunks) {
        fprintf(stderr, "\nerror: failed to allocate continuation chunks\n");
        return false;
    }

    if (mtmd_tokenize(mtmdCtx, continueChunks, &continueText, nullptr, 0) != 0) {
        fprintf(stderr, "\nerror: failed to tokenize continuation prompt\n");
        mtmd_input_chunks_free(continueChunks);
        return false;
    }

    PrintContinuationPhase(continuationIndex);

    const auto continuationStart = SteadyClock::now();
    const int continuationResult =
        mtmd_helper_eval_chunks(
            mtmdCtx,
            llamaCtx,
            continueChunks,
            nPast,
            0,
            nBatch,
            true,
            &nPast
        );
    const auto continuationEnd = SteadyClock::now();

    elapsedMs = ElapsedMs(continuationStart, continuationEnd);
    PrintContinuationTiming(continuationIndex, elapsedMs);

    mtmd_input_chunks_free(continueChunks);

    if (continuationResult != 0) {
        fprintf(stderr, "\nerror: failed to evaluate continuation prompt\n");
        return false;
    }

    return true;
}

GenerationStepResult GenerateNextToken(
    llama_context* llamaCtx,
    llama_sampler* sampler,
    const llama_vocab* vocab,
    double targetGpuDuty
)
{
    GenerationStepResult result;
    if (!llamaCtx || !sampler || !vocab) {
        result.ok = false;
        return result;
    }

    const auto sampleStart = SteadyClock::now();
    llama_token token = llama_sampler_sample(sampler, llamaCtx, -1);
    const auto sampleEnd = SteadyClock::now();

    result.samplingMs = ElapsedMs(sampleStart, sampleEnd);

    if (targetGpuDuty > 0.0 && targetGpuDuty < 1.0) {
        const double requiredIdleMs =
            result.samplingMs *
            ((1.0 / targetGpuDuty) - 1.0);

        std::this_thread::sleep_for(
            std::chrono::duration<double, std::milli>(
                requiredIdleMs
            )
        );
    }

    if (llama_vocab_is_eog(vocab, token)) {
        result.hitEog = true;
        return result;
    }

    if (!TokenToPiece(vocab, token, result.piece)) {
        fprintf(stderr, "\nerror: TokenToPiece failed\n");
        result.ok = false;
        return result;
    }

    const auto consoleStart = SteadyClock::now();
    printf("%s", result.piece.c_str());
    fflush(stdout);
    const auto consoleEnd = SteadyClock::now();

    result.consoleOutputMs = ElapsedMs(consoleStart, consoleEnd);

    llama_sampler_accept(sampler, token);

    result.decodeAttempted = true;
    if (!DecodeGeneratedToken(llamaCtx, token, result.decodeMs)) {
        fprintf(stderr, "\nerror: llama_decode failed while generating\n");
        result.ok = false;
        return result;
    }

    return result;
}

namespace {

GenerationRoundResult GenerateTokenRound(
    llama_context* llamaCtx,
    llama_sampler* sampler,
    const llama_vocab* vocab,
    GenerationResult& result,
    GenerationMetrics& metrics,
    int& generatedTotal,
    int nPredict,
    llama_pos& nPast
)
{
    GenerationRoundResult round;

    while (generatedTotal < nPredict) {
        GenerationStepResult step = GenerateNextToken(
            llamaCtx,
            sampler,
            vocab,
            TargetGpuDuty
        );

        AccumulateGenerationStep(result, step, metrics);

        if (step.hitEog) {
            round.hitEog = true;
            break;
        }

        if (!step.ok) {
            result.ok = false;
            round.ok = false;
            break;
        }

        AcceptGeneratedStep(result, step, nPast);
        ++round.generatedTokens;
        ++generatedTotal;
    }

    return round;
}

bool EvaluateNextContinuation(
    mtmd_context* mtmdCtx,
    llama_context* llamaCtx,
    int nBatch,
    int continuationIndex,
    llama_pos& nPast,
    GenerationResult& result
)
{
    double currentContinuationMs = 0.0;
    if (!EvaluateContinuationPrompt(
            mtmdCtx,
            llamaCtx,
            nBatch,
            continuationIndex,
            nPast,
            currentContinuationMs
        )) {
        return false;
    }

    result.report.continuationTotalMs += currentContinuationMs;
    return true;
}

} // namespace

GenerationResult GenerateText(
    mtmd_context* mtmdCtx,
    llama_context* llamaCtx,
    llama_sampler* sampler,
    const llama_vocab* vocab,
    const Options& options,
    const LanguageModelOptions& languageModelOptions,
    llama_pos& nPast
)
{
    GenerationResult result;
    if (!mtmdCtx || !llamaCtx || !sampler || !vocab) {
        result.ok = false;
        return result;
    }

    PrintPhase("Generating output tokens");

    const auto generationStart = SteadyClock::now();
    GenerationMetrics metrics;
    int generatedTotal = 0;
    int continueRound = 0;

    while (generatedTotal < options.nPredict) {
        GenerationRoundResult round = GenerateTokenRound(
            llamaCtx,
            sampler,
            vocab,
            result,
            metrics,
            generatedTotal,
            options.nPredict,
            nPast
        );

        if (!ShouldEvaluateContinuation(
                round,
                generatedTotal,
                continueRound,
                options
            )) {
            break;
        }

        if (!EvaluateNextContinuation(
                mtmdCtx,
                llamaCtx,
                languageModelOptions.nBatch,
                continueRound + 1,
                nPast,
                result
            )) {
            break;
        }

        ++continueRound;
    }

    const auto generationEnd = SteadyClock::now();

    FinalizeGenerationReport(
        result,
        generatedTotal,
        metrics,
        generationStart,
        generationEnd
    );

    return result;
}

void FreeMultimodalInput(MultimodalInput& input)
{
    if (input.chunks) {
        mtmd_input_chunks_free(input.chunks);
        input.chunks = nullptr;
    }

    if (input.bitmap) {
        mtmd_bitmap_free(input.bitmap);
        input.bitmap = nullptr;
    }
}

} // namespace qari
