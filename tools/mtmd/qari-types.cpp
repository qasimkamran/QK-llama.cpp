// SPDX-License-Identifier: MIT OR LicenseRef-QARI-OCR-COMMERCIAL
// Qari OCR addition: see NOTICE-QARI-OCR.md and LICENSE-QARI-OCR-COMMERCIAL.md

#include "qari-types.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace qari {

namespace {

bool HasSupportedImageExt(const std::string& path)
{
    std::filesystem::path p(path);
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    return ext == ".png" ||
           ext == ".jpg" ||
           ext == ".jpeg" ||
           ext == ".bmp" ||
           ext == ".webp" ||
           ext == ".tif" ||
           ext == ".tiff";
}

bool ParseIntValue(const char* optionName, const char* value, int& out)
{
    try {
        size_t parsedChars = 0;
        const int parsedValue = std::stoi(value, &parsedChars);
        if (parsedChars != std::strlen(value)) {
            fprintf(stderr, "error: invalid integer for %s: %s\n", optionName, value);
            return false;
        }

        out = parsedValue;
        return true;
    } catch (const std::exception&) {
        fprintf(stderr, "error: invalid integer for %s: %s\n", optionName, value);
        return false;
    }
}

bool ParseDoubleValue(const char* optionName, const char* value, double& out)
{
    try {
        size_t parsedChars = 0;
        const double parsedValue = std::stod(value, &parsedChars);
        if (parsedChars != std::strlen(value)) {
            fprintf(stderr, "error: invalid number for %s: %s\n", optionName, value);
            return false;
        }

        out = parsedValue;
        return true;
    } catch (const std::exception&) {
        fprintf(stderr, "error: invalid number for %s: %s\n", optionName, value);
        return false;
    }
}

} // namespace

Options ParseOptions(int argc, char** argv)
{
    Options opts;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            opts.modelPath = argv[++i];
        } else if (strcmp(argv[i], "--mmproj") == 0 && i + 1 < argc) {
            opts.mmprojPath = argv[++i];
        } else if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--image") == 0) && i + 1 < argc) {
            opts.imagePath = argv[++i];
        } else if (strcmp(argv[i], "--image-dir") == 0 && i + 1 < argc) {
            opts.imageDirPath = argv[++i];
        } else if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) && i + 1 < argc) {
            opts.prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            if (!ParseIntValue(argv[i], argv[i + 1], opts.nPredict)) {
                return {};
            }
            ++i;
        } else if (strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            if (!ParseIntValue(argv[i], argv[i + 1], opts.nGpuLayers)) {
                return {};
            }
            ++i;
        } else if (strcmp(argv[i], "--image-min-tokens") == 0 && i + 1 < argc) {
            if (!ParseIntValue(argv[i], argv[i + 1], opts.imageMinTokens)) {
                return {};
            }
            ++i;
        } else if (strcmp(argv[i], "--image-max-tokens") == 0 && i + 1 < argc) {
            if (!ParseIntValue(argv[i], argv[i + 1], opts.imageMaxTokens)) {
                return {};
            }
            ++i;
        } else if (strcmp(argv[i], "--max-continue-rounds") == 0 && i + 1 < argc) {
            if (!ParseIntValue(argv[i], argv[i + 1], opts.maxContinueRounds)) {
                return {};
            }
            ++i;
        } else if (strcmp(argv[i], "--gpu-duty") == 0 && i + 1 < argc) {
            if (!ParseDoubleValue(argv[i], argv[i + 1], opts.gpuDuty)) {
                return {};
            }
            ++i;
        } else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) {
            opts.outputPath = argv[++i];
        } else if (strcmp(argv[i], "--output-dir") == 0 && i + 1 < argc) {
            opts.outputDirPath = argv[++i];
        } else {
            return {};
        }
    }
    return opts;
}

bool OptionsValid(const Options& opts)
{
    if (opts.modelPath.empty()) {
        return false;
    }

    const bool hasSingleImage = !opts.imagePath.empty();
    const bool hasImageDir = !opts.imageDirPath.empty();
    if (hasSingleImage == hasImageDir) {
        fprintf(stderr, "error: provide exactly one of -i/--image or --image-dir\n");
        return false;
    }

    if (hasImageDir) {
        std::error_code ec;
        std::filesystem::path imageDir(opts.imageDirPath);
        if (!std::filesystem::is_directory(imageDir, ec)) {
            fprintf(stderr, "error: not a readable directory: %s\n", opts.imageDirPath.c_str());
            return false;
        }
    }

    if (HasOutputFile(opts) && HasOutputDir(opts)) {
        fprintf(stderr, "error: provide only one of -o/--output or --output-dir\n");
        return false;
    }

    if (HasOutputDir(opts)) {
        std::error_code ec;
        std::filesystem::path outDir(opts.outputDirPath);
        if (!std::filesystem::is_directory(outDir, ec)) {
            fprintf(stderr, "error: --output-dir is not a readable directory: %s\n", opts.outputDirPath.c_str());
            return false;
        }
    }

    if (!std::isfinite(opts.gpuDuty) || opts.gpuDuty < 0.0 || opts.gpuDuty > 1.0) {
        fprintf(stderr, "error: --gpu-duty must be between 0 and 1\n");
        return false;
    }

    return true;
}

bool HasOutputFile(const Options& opts)
{
    return !opts.outputPath.empty();
}

bool HasOutputDir(const Options& opts)
{
    return !opts.outputDirPath.empty();
}

std::vector<std::string> CollectImagePaths(const Options& opts)
{
    if (!opts.imagePath.empty()) {
        return { opts.imagePath };
    }

    std::vector<std::string> imagePaths;
    std::error_code ec;
    std::filesystem::path dirPath(opts.imageDirPath);
    for (const auto& entry : std::filesystem::directory_iterator(dirPath, ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string currentPath = entry.path().string();
        if (HasSupportedImageExt(currentPath)) {
            imagePaths.push_back(currentPath);
        }
    }

    if (ec) {
        fprintf(stderr, "error: failed to iterate directory: %s\n", opts.imageDirPath.c_str());
        return {};
    }

    std::sort(imagePaths.begin(), imagePaths.end());
    if (imagePaths.empty()) {
        fprintf(stderr, "error: no supported image files found in directory: %s\n", opts.imageDirPath.c_str());
        return {};
    }

    return imagePaths;
}

} // namespace qari
