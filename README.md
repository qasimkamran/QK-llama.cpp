# QK-llama.cpp

This repository is a fork of `llama.cpp` focused on a practical pipeline for:

- Converting Hugging Face multimodal models to GGUF
- Generating matching `mmproj` files
- Running OCR and multimodal inference locally on GPUs
- Supporting AMD via Vulkan and NVIDIA via CUDA on Windows

## Purpose

Primary use case in this fork:

- HF model -> GGUF text model
- HF model -> GGUF mmproj
- local inference with `llama-qari-ocr` (added in this fork)

This is intended for Windows local inference workflows where AMD GPUs can be used through Vulkan.

## Model Conversion (HF -> GGUF)

Example for Qari OCR (Qwen2VL-based):

```bash
# text model GGUF
python convert_hf_to_gguf.py NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct \
  --outfile qari-ocr-q8_0.gguf \
  --outtype q8_0 \
  --remote

# multimodal projector GGUF (mmproj)
python convert_hf_to_gguf.py NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct \
  --mmproj \
  --outfile . \
  --outtype f16 \
  --remote
```

## Build

### Windows 11 / AMD (Vulkan)

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DGGML_VULKAN=ON `
  -DBUILD_SHARED_LIBS=OFF

cmake --build build --target llama-qari-ocr --config Release -j2
```

### Windows / NVIDIA (CUDA)

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DGGML_CUDA=ON `
  -DBUILD_SHARED_LIBS=OFF

cmake --build build --target llama-qari-ocr --config Release -j2
```

### Linux / WSL (Vulkan)

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON \
  -DBUILD_SHARED_LIBS=OFF

cmake --build build --target llama-qari-ocr -j2
```

## Inference (OCR)

The OCR tool in this fork is:

- `llama-qari-ocr`

### Windows run example

```powershell
.\build\bin\Release\llama-qari-ocr.exe \
  -m ..\qari-ocr-q8_0.gguf \
  --mmproj ..\mmproj-Qwen2-VL-2b-Instruct-F16.gguf \
  -i document.jpg \
  --prompt "Extract all text exactly." \
  -n 2048 -ngl 99 \
  --image-max-tokens 1024 \
  --max-continue-rounds 8 \
  -o ocr-output.txt
```

### Linux run example

```bash
./build/bin/llama-qari-ocr \
  -m ../qari-ocr-q8_0.gguf \
  --mmproj ../mmproj-Qwen2-VL-2b-Instruct-F16.gguf \
  -i document.jpg \
  --prompt "Extract all text exactly." \
  -n 2048 -ngl 99 \
  --image-max-tokens 1024 \
  --max-continue-rounds 8 \
  -o ocr-output.txt
```

## `llama-qari-ocr` flags

- `-m, --model` text GGUF
- `--mmproj` mmproj GGUF
- `-i, --image` input image
- `-p, --prompt` OCR instruction
- `-n` max generation tokens
- `-ngl` GPU layers for text model
- `--image-min-tokens` min image tokens for mtmd
- `--image-max-tokens` max image tokens for mtmd
- `--max-continue-rounds` auto-continue rounds after early EOG
- `-o, --output` save OCR output to file

## Notes

- AMD acceleration works through Vulkan.
- NVIDIA acceleration can use Vulkan or CUDA (CUDA is usually faster).
- For multimodal models, text GGUF and mmproj GGUF must match the same model family/version.

## Licensing

- Upstream llama.cpp code in this fork remains under the original MIT license.
- Qari OCR fork-specific additions are scoped in `NOTICE-QARI-OCR.md`.
- Commercial (paid) licensing terms for those additions are in
  `LICENSE-QARI-OCR-COMMERCIAL.md`.
