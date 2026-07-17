# MTMD For Qari OCR

This directory is retained because `llama-qari-ocr` depends on `libmtmd` for
multimodal projector loading, image preprocessing, tokenization, and image
embedding evaluation.

The only executable built from this directory in this fork is:

```text
llama-qari-ocr
```

Build it from the repository root with:

```bash
cmake --preset qari-vulkan-release
cmake --build --preset qari-vulkan-release
```
