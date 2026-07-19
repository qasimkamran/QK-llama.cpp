// SPDX-License-Identifier: MIT OR LicenseRef-QARI-OCR-COMMERCIAL
// Qari OCR addition: see NOTICE-QARI-OCR.md and LICENSE-QARI-OCR-COMMERCIAL.md

#pragma once

#include <string>
#include <vector>

namespace qari {

struct Options {
    std::string modelPath;
    std::string mmprojPath;
    std::string imagePath;
    std::string imageDirPath;
    std::string outputPath;
    std::string outputDirPath;
    std::string prompt = "Extract all text from this image. Return plain text only.";

    int nPredict = 512;
    int nGpuLayers = 99;
    int imageMinTokens = 256;
    int imageMaxTokens = 1024;
    int maxContinueRounds = 4;
    double gpuDuty = 0.6;
};

Options ParseOptions(int argc, char** argv);
bool OptionsValid(const Options& opts);
bool HasOutputFile(const Options& opts);
bool HasOutputDir(const Options& opts);
std::vector<std::string> CollectImagePaths(const Options& opts);

} // namespace qari
