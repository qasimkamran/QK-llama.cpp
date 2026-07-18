// SPDX-License-Identifier: MIT OR LicenseRef-QARI-OCR-COMMERCIAL
// Qari OCR addition: see NOTICE-QARI-OCR.md and LICENSE-QARI-OCR-COMMERCIAL.md

#pragma once

#include "qari-types.h"

#include <cstddef>
#include <string>

namespace qari {

bool SaveOutputText(
    const Options& options,
    const std::string& imagePath,
    size_t imageIndex,
    const std::string& outputText
);

} // namespace qari
