# QK-llama.cpp

This fork is pruned around one executable:

- `llama-qari-ocr`

The retained build graph is `tools/mtmd/qari-ocr.cpp`, `libmtmd`, `libllama`,
and `ggml` with CPU plus optional Vulkan backend support.

## Build With Vulkan

Using the preset:

```bash
cmake --preset qari-vulkan-release
cmake --build --preset qari-vulkan-release
```

Equivalent explicit CMake commands:

```bash
cmake -S . -B build-qari-vulkan-release \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON \
  -DGGML_OPENMP=OFF \
  -DGGML_LLAMAFILE=OFF \
  -DLLAMA_BUILD_QARI_OCR=ON

cmake --build build-qari-vulkan-release --target llama-qari-ocr -j
```

The binary is written to:

```text
build-qari-vulkan-release/bin/llama-qari-ocr
```

On Windows with a multi-config generator, use the same CMake options and build
the `llama-qari-ocr` target for `Release`.

## Run

```bash
./build-qari-vulkan-release/bin/llama-qari-ocr \
  -m ../qari-ocr-q8_0.gguf \
  --mmproj ../qari-mmproj-f16.gguf \
  -i document.jpg \
  --prompt "Extract all text exactly." \
  -n 2048 \
  -ngl 99 \
  --image-max-tokens 1024 \
  --max-continue-rounds 8 \
  -o ocr-output.txt
```

For single-file multimodal models, omit `--mmproj` or set it to the same file as
`-m`.

## Runtime Inputs

You still need:

- a text GGUF model
- a matching multimodal projector GGUF, unless using a single-file model
- image input files
- a Vulkan-capable driver/runtime when building with `GGML_VULKAN=ON`

## Licensing

Upstream llama.cpp and ggml code remain under their original MIT license.
Qari OCR fork-specific additions are described in `NOTICE-QARI-OCR.md`, with
commercial terms in `LICENSE-QARI-OCR-COMMERCIAL.md`.
