// SPDX-License-Identifier: MIT OR LicenseRef-QARI-OCR-COMMERCIAL
// Qari OCR addition: see NOTICE-QARI-OCR.md and LICENSE-QARI-OCR-COMMERCIAL.md

#include "qari-report.h"

#include <cstdio>

namespace qari {

void PrintUsage(const char* prog)
{
    fprintf(stderr,
        "Usage:\n"
        "  %s -m <text-model.gguf> (-i <image> | --image-dir <dir>) [--mmproj <mmproj.gguf>] [--prompt <text>] [-n <predict>] [-ngl <layers>] [--image-min-tokens <n>] [--image-max-tokens <n>] [--max-continue-rounds <n>] [--gpu-duty <0..1>] [-o <output.txt> | --output-dir <dir>]\n\n"
        "Example:\n"
        "  %s -m ../qari-ocr-q8_0.gguf --mmproj ../qari-mmproj-f16.gguf -i document.jpg --prompt \"Extract all text exactly.\" -n 512 -ngl 99\n"
        "  %s -m ../qari-ocr-q8_0.gguf --mmproj ../qari-mmproj-f16.gguf --image-dir ./docs --prompt \"Extract all text exactly.\" -n 512 -ngl 99\n"
        "\n"
        "Notes:\n"
        "  - For single-file multimodal models, set --mmproj to the same file as -m.\n"
        "  - Use either -i/--image or --image-dir (not both).\n"
        "  - Use either -o/--output or --output-dir (not both).\n"
        "  - Use -ngl based on your VRAM (99 tries to offload as much as possible).\n",
        prog, prog, prog);
}

void PrintPhase(const char* phase)
{
    fprintf(stderr, "[OCR PHASE] %s\n", phase);
}

void PrintImageHeader(size_t imageIndex, size_t imageCount, const std::string& imagePath)
{
    fprintf(
        stderr,
        "\n"
        "============================================================\n"
        "[OCR TIMING] image %zu/%zu: %s\n"
        "============================================================\n",
        imageIndex + 1,
        imageCount,
        imagePath.c_str()
    );
}

void PrintTiming(const char* label, double elapsedMs)
{
    fprintf(stderr, "[OCR TIMING] %s: %.2f ms\n", label, elapsedMs);
}

void PrintContextTokenCount(int contextTokens)
{
    fprintf(stderr, "[OCR INFO] Context tokens after image prefill: %d\n", contextTokens);
}

void PrintContinuationPhase(int continuationIndex)
{
    fprintf(stderr, "\n[OCR PHASE] Evaluating continuation prompt %d\n", continuationIndex);
}

void PrintContinuationTiming(int continuationIndex, double elapsedMs)
{
    fprintf(stderr, "[OCR TIMING] Continuation prompt %d: %.2f ms\n", continuationIndex, elapsedMs);
}

void PrintGenerationReport(const GenerationReport& report)
{
    fprintf(
        stderr,
        "\n"
        "---------------- GENERATION REPORT ----------------\n"
        "[OCR TIMING] Generated tokens:              %d\n"
        "[OCR TIMING] Generation wall time:          %.2f ms\n"
        "[OCR TIMING] Generation speed:              %.2f tokens/s\n"
        "[OCR TIMING] llama_decode calls:            %d\n"
        "[OCR TIMING] Total llama_decode time:       %.2f ms\n"
        "[OCR TIMING] Average llama_decode time:     %.2f ms\n"
        "[OCR TIMING] Minimum llama_decode time:     %.2f ms\n"
        "[OCR TIMING] Maximum llama_decode time:     %.2f ms\n"
        "[OCR TIMING] Sampling time:                 %.2f ms\n"
        "[OCR TIMING] Console output/flush time:     %.2f ms\n"
        "[OCR TIMING] Continuation evaluation time:  %.2f ms\n"
        "[OCR TIMING] Other generation time:        %.2f ms\n"
        "-----------------------------------------------------\n",
        report.generatedTokens,
        report.generationTotalMs,
        report.tokensPerSecond,
        report.decodeCallCount,
        report.decodeTotalMs,
        report.averageDecodeMs,
        report.minimumDecodeMs,
        report.maximumDecodeMs,
        report.samplingTotalMs,
        report.consoleOutputTotalMs,
        report.continuationTotalMs,
        report.measuredNonDecodeMs
    );
}

void PrintImageSummary(const ImageSummary& summary)
{
    fprintf(
        stderr,
        "\n"
        "================== IMAGE SUMMARY ==================\n"
        "[OCR SUMMARY] Image loading:               %.2f ms\n"
        "[OCR SUMMARY] Tokenisation/preprocessing:  %.2f ms\n"
        "[OCR SUMMARY] Vision + image prefill:      %.2f ms\n"
        "[OCR SUMMARY] Text generation:             %.2f ms\n"
        "[OCR SUMMARY] Output saving:               %.2f ms\n"
        "[OCR SUMMARY] Total image processing:      %.2f ms\n"
        "[OCR SUMMARY] Output tokens:               %d\n"
        "[OCR SUMMARY] Context tokens at end:       %d\n"
        "=====================================================\n",
        summary.imageLoadMs,
        summary.tokenizationMs,
        summary.multimodalEvalMs,
        summary.generationMs,
        summary.outputSaveMs,
        summary.totalImageMs,
        summary.outputTokens,
        summary.contextTokens
    );
}

} // namespace qari
