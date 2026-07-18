// SPDX-License-Identifier: MIT OR LicenseRef-QARI-OCR-COMMERCIAL
// Qari OCR addition: see NOTICE-QARI-OCR.md and LICENSE-QARI-OCR-COMMERCIAL.md

#include "qari-output.h"

#include "ggml.h"

#include <cstdio>
#include <filesystem>

namespace qari {

bool SaveOutputText(
    const Options& options,
    const std::string& imagePath,
    size_t imageIndex,
    const std::string& outputText
)
{
    if (HasOutputDir(options))
    {
        const std::string stem = std::filesystem::path(imagePath).stem().string();

        const std::string outputFilePath =
            (
                std::filesystem::path(options.outputDirPath) /
                (stem + ".txt")
            ).string();

        FILE* outputFile = ggml_fopen(outputFilePath.c_str(), "wb");
        if (!outputFile) {
            fprintf(stderr, "error: failed to open output file: %s\n", outputFilePath.c_str());
            return false;
        }

        if (!outputText.empty()) {
            fwrite(outputText.data(), 1, outputText.size(), outputFile);
        }

        fclose(outputFile);
        fprintf(stderr, "saved output to: %s\n", outputFilePath.c_str());
        return true;
    }

    if (HasOutputFile(options)) {
        FILE* outputFile = ggml_fopen(options.outputPath.c_str(), "ab");
        if (!outputFile) {
            fprintf(stderr, "error: failed to open output file: %s\n", options.outputPath.c_str());
            return false;
        }

        if (imageIndex > 0) {
            fwrite("\n", 1, 1, outputFile);
        }

        if (!outputText.empty()) {
            fwrite(outputText.data(), 1, outputText.size(), outputFile);
        }

        fclose(outputFile);
    }

    return true;
}

} // namespace qari
