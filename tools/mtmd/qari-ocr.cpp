// SPDX-License-Identifier: MIT OR LicenseRef-QARI-OCR-COMMERCIAL
// Qari OCR addition: see NOTICE-QARI-OCR.md and LICENSE-QARI-OCR-COMMERCIAL.md

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <clocale>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>
#include <limits>
#include <chrono>
#include <thread>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#endif

static void print_usage(const char * prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s -m <text-model.gguf> (-i <image> | --image-dir <dir>) [--mmproj <mmproj.gguf>] [--prompt <text>] [-n <predict>] [-ngl <layers>] [--image-min-tokens <n>] [--image-max-tokens <n>] [--max-continue-rounds <n>] [-o <output.txt> | --output-dir <dir>]\\n\\n"
        "Example:\n"
        "  %s -m ../qari-ocr-q8_0.gguf --mmproj ../qari-mmproj-f16.gguf -i document.jpg --prompt \"Extract all text exactly.\" -n 512 -ngl 99\\n"
        "  %s -m ../qari-ocr-q8_0.gguf --mmproj ../qari-mmproj-f16.gguf --image-dir ./docs --prompt \"Extract all text exactly.\" -n 512 -ngl 99\\n"
        "\n"
        "Notes:\n"
        "  - For single-file multimodal models, set --mmproj to the same file as -m.\n"
        "  - Use either -i/--image or --image-dir (not both).\n"
        "  - Use either -o/--output or --output-dir (not both).\n"
        "  - Use -ngl based on your VRAM (99 tries to offload as much as possible).\n",
        prog, prog, prog);
}

static bool token_to_piece(const llama_vocab * vocab, llama_token tok, std::string & out) {
    char buf[256];
    const int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
    if (n < 0) {
        return false;
    }
    out.assign(buf, n);
    return true;
}

static bool has_supported_image_ext(const std::string & path) {
    std::filesystem::path p(path);
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });

    return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".webp" || ext == ".tif" || ext == ".tiff";
}

using SteadyClock = std::chrono::steady_clock;

static double elapsed_ms(
    const SteadyClock::time_point & start,
    const SteadyClock::time_point & end
) {
    return std::chrono::duration<double, std::milli>(
        end - start
    ).count();
}

static int qari_ocr_main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string mmproj_path;
    std::string image_path;
    std::string image_dir_path;
    std::string output_path;
    std::string output_dir_path;
    std::string prompt = "Extract all text from this image. Return plain text only.";
    int n_predict = 512;
    int n_gpu_layers = 99;
    int image_min_tokens = 256;
    int image_max_tokens = 1024;
    int max_continue_rounds = 4;

    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--mmproj") == 0 && i + 1 < argc) {
            mmproj_path = argv[++i];
        } else if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--image") == 0) && i + 1 < argc) {
            image_path = argv[++i];
        } else if (strcmp(argv[i], "--image-dir") == 0 && i + 1 < argc) {
            image_dir_path = argv[++i];
        } else if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_predict = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            n_gpu_layers = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--image-min-tokens") == 0 && i + 1 < argc) {
            image_min_tokens = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--image-max-tokens") == 0 && i + 1 < argc) {
            image_max_tokens = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-continue-rounds") == 0 && i + 1 < argc) {
            max_continue_rounds = std::stoi(argv[++i]);
        } else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--output-dir") == 0 && i + 1 < argc) {
            output_dir_path = argv[++i];
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    const bool has_single_image = !image_path.empty();
    const bool has_image_dir = !image_dir_path.empty();
    if (has_single_image == has_image_dir) {
        fprintf(stderr, "error: provide exactly one of -i/--image or --image-dir\n");
        print_usage(argv[0]);
        return 1;
    }

    const bool has_output_file = !output_path.empty();
    const bool has_output_dir = !output_dir_path.empty();
    if (has_output_file && has_output_dir) {
        fprintf(stderr, "error: provide only one of -o/--output or --output-dir\n");
        print_usage(argv[0]);
        return 1;
    }

    if (mmproj_path.empty()) {
        mmproj_path = model_path;
    }

    std::vector<std::string> image_paths;
    if (has_single_image) {
        image_paths.push_back(image_path);
    } else {
        std::error_code ec;
        std::filesystem::path dir_path(image_dir_path);
        if (!std::filesystem::is_directory(dir_path, ec)) {
            fprintf(stderr, "error: not a readable directory: %s\n", image_dir_path.c_str());
            return 1;
        }

        for (const auto & entry : std::filesystem::directory_iterator(dir_path, ec)) {
            if (ec) {
                break;
            }
            if (!entry.is_regular_file()) {
                continue;
            }
            const std::string cur = entry.path().string();
            if (has_supported_image_ext(cur)) {
                image_paths.push_back(cur);
            }
        }
        if (ec) {
            fprintf(stderr, "error: failed to iterate directory: %s\n", image_dir_path.c_str());
            return 1;
        }

        std::sort(image_paths.begin(), image_paths.end());
        if (image_paths.empty()) {
            fprintf(stderr, "error: no supported image files found in directory: %s\n", image_dir_path.c_str());
            return 1;
        }
    }

    if (has_output_dir) {
        std::error_code ec;
        std::filesystem::path out_dir(output_dir_path);
        if (!std::filesystem::is_directory(out_dir, ec)) {
            fprintf(stderr, "error: --output-dir is not a readable directory: %s\n", output_dir_path.c_str());
            return 1;
        }
    }

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "error: failed to load text model: %s\n", model_path.c_str());
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 8192;
    ctx_params.n_batch = 1024;
    ctx_params.n_ubatch = 256;
    ctx_params.no_perf = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "error: failed to create llama context\n");
        llama_model_free(model);
        return 1;
    }

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = true;
    mparams.print_timings = true;
    mparams.n_threads = (int) std::max(1u, std::thread::hardware_concurrency());
    mparams.image_min_tokens = image_min_tokens;
    mparams.image_max_tokens = image_max_tokens;

    mtmd_context * mctx = mtmd_init_from_file(mmproj_path.c_str(), model, mparams);
    if (!mctx) {
        fprintf(stderr, "error: failed to load multimodal projector/model: %s\n", mmproj_path.c_str());
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (!mtmd_support_vision(mctx)) {
        fprintf(stderr, "error: loaded multimodal context does not support vision input\n");
        mtmd_free(mctx);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    for (size_t image_idx = 0; image_idx < image_paths.size(); ++image_idx) {
        const std::string & current_image_path = image_paths[image_idx];

        fprintf(
            stderr,
            "\n"
            "============================================================\n"
            "[OCR TIMING] image %zu/%zu: %s\n"
            "============================================================\n",
            image_idx + 1,
            image_paths.size(),
            current_image_path.c_str()
        );

        const auto image_total_start = SteadyClock::now();

        llama_memory_clear(llama_get_memory(ctx), true);
        llama_sampler_reset(sampler);
        llama_perf_context_reset(ctx);

        // ------------------------------------------------------------
        // 1. Load and decode the image file
        // ------------------------------------------------------------

        fprintf(stderr, "[OCR PHASE] Loading image\n");
        const auto bitmap_start = SteadyClock::now();

        mtmd_bitmap * bmp =
            mtmd_helper_bitmap_init_from_file(mctx, current_image_path.c_str());

        const auto bitmap_end = SteadyClock::now();

        if (!bmp) {
            fprintf(
                stderr,
                "error: failed to load image: %s\n",
                current_image_path.c_str()
            );

            llama_sampler_free(sampler);
            mtmd_free(mctx);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        fprintf(
            stderr,
            "[OCR TIMING] Image loading: %.2f ms\n",
            elapsed_ms(bitmap_start, bitmap_end)
        );

        const mtmd_bitmap * bitmaps[] = { bmp };

        const std::string user_content =
            std::string(mtmd_default_marker()) + "\n" + prompt;

        const std::string full_prompt =
            "<|im_start|>user\n" +
            user_content +
            "<|im_end|>\n"
            "<|im_start|>assistant\n";

        mtmd_input_text in_txt = {
            full_prompt.c_str(),
            true,
            true,
        };

        mtmd_input_chunks * chunks = mtmd_input_chunks_init();

        if (!chunks) {
            fprintf(
                stderr,
                "error: failed to allocate mtmd_input_chunks\n"
            );

            mtmd_bitmap_free(bmp);
            llama_sampler_free(sampler);
            mtmd_free(mctx);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        // ------------------------------------------------------------
        // 2. Multimodal tokenisation and image preprocessing
        // ------------------------------------------------------------

        fprintf(
            stderr,
            "[OCR PHASE] Tokenising and preprocessing multimodal input\n"
        );

        const auto tokenize_start = SteadyClock::now();

        const int tokenize_result =
            mtmd_tokenize(mctx, chunks, &in_txt, bitmaps, 1);

        const auto tokenize_end = SteadyClock::now();

        fprintf(
            stderr,
            "[OCR TIMING] Multimodal tokenisation/preprocessing: %.2f ms\n",
            elapsed_ms(tokenize_start, tokenize_end)
        );

        if (tokenize_result != 0) {
            fprintf(stderr, "error: mtmd_tokenize() failed\n");

            mtmd_input_chunks_free(chunks);
            mtmd_bitmap_free(bmp);
            llama_sampler_free(sampler);
            mtmd_free(mctx);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        // ------------------------------------------------------------
        // 3. Vision encoding and language-model image prefill
        //
        // mtmd's own print_timings output should further divide this
        // into "image slice encoded" and "image decoded".
        // ------------------------------------------------------------

        fprintf(
            stderr,
            "[OCR PHASE] Evaluating vision input and image-token prefill\n"
        );

        llama_pos n_past = 0;

        const auto multimodal_eval_start = SteadyClock::now();

        const int eval_result = mtmd_helper_eval_chunks(
            mctx,
            ctx,
            chunks,
            n_past,
            0,
            ctx_params.n_batch,
            true,
            &n_past
        );

        const auto multimodal_eval_end = SteadyClock::now();

        fprintf(
            stderr,
            "[OCR TIMING] Vision encoding + image prefill: %.2f ms\n",
            elapsed_ms(multimodal_eval_start, multimodal_eval_end)
        );

        fprintf(
            stderr,
            "[OCR INFO] Context tokens after image prefill: %d\n",
            static_cast<int>(n_past)
        );

        if (eval_result != 0) {
            fprintf(
                stderr,
                "error: mtmd_helper_eval_chunks() failed\n"
            );

            mtmd_input_chunks_free(chunks);
            mtmd_bitmap_free(bmp);
            llama_sampler_free(sampler);
            mtmd_free(mctx);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        // ------------------------------------------------------------
        // 4. Autoregressive text generation
        // ------------------------------------------------------------

        std::string output_text;

        printf("\n");

        if (image_paths.size() > 1) {
            fprintf(
                stderr,
                "[%zu/%zu] OCR: %s\n",
                image_idx + 1,
                image_paths.size(),
                current_image_path.c_str()
            );
        }

        fprintf(stderr, "[OCR PHASE] Generating output tokens\n");

        const auto generation_start = SteadyClock::now();

        double sampling_total_ms = 0.0;
        double decode_total_ms = 0.0;
        double console_output_total_ms = 0.0;
        double continuation_total_ms = 0.0;

        double minimum_decode_ms =
            std::numeric_limits<double>::max();

        double maximum_decode_ms = 0.0;

        int decode_call_count = 0;
        int generated_total = 0;
        int continue_round = 0;

        while (generated_total < n_predict) {
            bool hit_eog = false;
            int generated_this_round = 0;

            for (; generated_total < n_predict; ++generated_total) {
                // Measure sampling independently from inference.
                const double target_gpu_duty = 0.60;

                const auto sample_start = SteadyClock::now();

                llama_token tok =
                    llama_sampler_sample(sampler, ctx, -1);

                const auto sample_end = SteadyClock::now();

                const double sample_ms =
                    elapsed_ms(sample_start, sample_end);

                sampling_total_ms += sample_ms;

                if (target_gpu_duty > 0.0 && target_gpu_duty < 1.0) {
                    const double required_idle_ms =
                        sample_ms *
                        ((1.0 / target_gpu_duty) - 1.0);

                    std::this_thread::sleep_for(
                        std::chrono::duration<double, std::milli>(
                            required_idle_ms
                        )
                    );
                }

                if (llama_vocab_is_eog(vocab, tok)) {
                    hit_eog = true;
                    break;
                }

                std::string piece;

                if (!token_to_piece(vocab, tok, piece)) {
                    fprintf(
                        stderr,
                        "\nerror: token_to_piece failed\n"
                    );

                    hit_eog = false;
                    generated_total = n_predict;
                    break;
                }

                // Console flushing can be surprisingly expensive on Windows,
                // so measure it separately.
                const auto console_start = SteadyClock::now();

                printf("%s", piece.c_str());
                fflush(stdout);

                const auto console_end = SteadyClock::now();

                console_output_total_ms +=
                    elapsed_ms(console_start, console_end);

                output_text += piece;
                ++generated_this_round;

                llama_sampler_accept(sampler, tok);

                llama_batch batch =
                    llama_batch_get_one(&tok, 1);

                // This is the GPU-heavy operation during token generation.
                const auto decode_start = SteadyClock::now();

                const int decode_result =
                    llama_decode(ctx, batch);

                const auto decode_end = SteadyClock::now();

                const double current_decode_ms =
                    elapsed_ms(decode_start, decode_end);

                decode_total_ms += current_decode_ms;

                minimum_decode_ms =
                    std::min(minimum_decode_ms, current_decode_ms);

                maximum_decode_ms =
                    std::max(maximum_decode_ms, current_decode_ms);

                ++decode_call_count;

                if (decode_result != 0) {
                    fprintf(
                        stderr,
                        "\nerror: llama_decode failed while generating\n"
                    );

                    hit_eog = false;
                    generated_total = n_predict;
                    break;
                }

                ++n_past;
            }

            if (!hit_eog || generated_total >= n_predict) {
                break;
            }

            if (
                generated_this_round == 0 ||
                continue_round >= max_continue_rounds
            ) {
                break;
            }

            const std::string continue_prompt =
                "<|im_end|>\n"
                "<|im_start|>user\n"
                "Continue exactly where you stopped. "
                "Do not repeat any previous text. "
                "Keep transcribing the remaining page.\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n";

            mtmd_input_text continue_text = {
                continue_prompt.c_str(),
                false,
                true,
            };

            mtmd_input_chunks * continue_chunks =
                mtmd_input_chunks_init();

            if (!continue_chunks) {
                fprintf(
                    stderr,
                    "\nerror: failed to allocate continuation chunks\n"
                );
                break;
            }

            if (
                mtmd_tokenize(
                    mctx,
                    continue_chunks,
                    &continue_text,
                    nullptr,
                    0
                ) != 0
            ) {
                fprintf(
                    stderr,
                    "\nerror: failed to tokenize continuation prompt\n"
                );

                mtmd_input_chunks_free(continue_chunks);
                break;
            }

            fprintf(
                stderr,
                "\n[OCR PHASE] Evaluating continuation prompt %d\n",
                continue_round + 1
            );

            const auto continuation_start = SteadyClock::now();

            const int continuation_result =
                mtmd_helper_eval_chunks(
                    mctx,
                    ctx,
                    continue_chunks,
                    n_past,
                    0,
                    ctx_params.n_batch,
                    true,
                    &n_past
                );

            const auto continuation_end = SteadyClock::now();

            const double current_continuation_ms =
                elapsed_ms(continuation_start, continuation_end);

            continuation_total_ms += current_continuation_ms;

            fprintf(
                stderr,
                "[OCR TIMING] Continuation prompt %d: %.2f ms\n",
                continue_round + 1,
                current_continuation_ms
            );

            if (continuation_result != 0) {
                fprintf(
                    stderr,
                    "\nerror: failed to evaluate continuation prompt\n"
                );

                mtmd_input_chunks_free(continue_chunks);
                break;
            }

            mtmd_input_chunks_free(continue_chunks);
            ++continue_round;
        }

        const auto generation_end = SteadyClock::now();

        printf("\n");

        // ------------------------------------------------------------
        // 5. Generation timing report
        // ------------------------------------------------------------

        const double generation_total_ms =
            elapsed_ms(generation_start, generation_end);

        const double average_decode_ms =
            decode_call_count > 0
                ? decode_total_ms /
                    static_cast<double>(decode_call_count)
                : 0.0;

        if (decode_call_count == 0) {
            minimum_decode_ms = 0.0;
        }

        const double tokens_per_second =
            generation_total_ms > 0.0
                ? static_cast<double>(generated_total) /
                    (generation_total_ms / 1000.0)
                : 0.0;

        const double measured_non_decode_ms =
            std::max(
                0.0,
                generation_total_ms -
                    decode_total_ms -
                    continuation_total_ms
            );

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
            generated_total,
            generation_total_ms,
            tokens_per_second,
            decode_call_count,
            decode_total_ms,
            average_decode_ms,
            minimum_decode_ms,
            maximum_decode_ms,
            sampling_total_ms,
            console_output_total_ms,
            continuation_total_ms,
            measured_non_decode_ms
        );

        // Print llama.cpp's own internal performance counters.
        fprintf(
            stderr,
            "\n[OCR PERF] llama.cpp context performance:\n"
        );

        llama_perf_context_print(ctx);

        // ------------------------------------------------------------
        // 6. Save output
        // ------------------------------------------------------------

        const auto output_save_start = SteadyClock::now();

        if (has_output_dir) {
            const std::string stem =
                std::filesystem::path(
                    current_image_path
                ).stem().string();

            const std::string per_image_output_path =
                (
                    std::filesystem::path(output_dir_path) /
                    (stem + ".txt")
                ).string();

            FILE * fout =
                ggml_fopen(per_image_output_path.c_str(), "wb");

            if (!fout) {
                fprintf(
                    stderr,
                    "error: failed to open output file: %s\n",
                    per_image_output_path.c_str()
                );

                mtmd_input_chunks_free(chunks);
                mtmd_bitmap_free(bmp);
                llama_sampler_free(sampler);
                mtmd_free(mctx);
                llama_free(ctx);
                llama_model_free(model);
                return 1;
            }

            if (!output_text.empty()) {
                fwrite(
                    output_text.data(),
                    1,
                    output_text.size(),
                    fout
                );
            }

            fclose(fout);

            fprintf(
                stderr,
                "saved output to: %s\n",
                per_image_output_path.c_str()
            );
        } else if (has_output_file) {
            FILE * fout =
                ggml_fopen(output_path.c_str(), "ab");

            if (!fout) {
                fprintf(
                    stderr,
                    "error: failed to open output file: %s\n",
                    output_path.c_str()
                );

                mtmd_input_chunks_free(chunks);
                mtmd_bitmap_free(bmp);
                llama_sampler_free(sampler);
                mtmd_free(mctx);
                llama_free(ctx);
                llama_model_free(model);
                return 1;
            }

            if (image_idx > 0) {
                fwrite("\n", 1, 1, fout);
            }

            if (!output_text.empty()) {
                fwrite(
                    output_text.data(),
                    1,
                    output_text.size(),
                    fout
                );
            }

            fclose(fout);
        }

        const auto output_save_end = SteadyClock::now();
        const auto image_total_end = SteadyClock::now();

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
            elapsed_ms(bitmap_start, bitmap_end),
            elapsed_ms(tokenize_start, tokenize_end),
            elapsed_ms(multimodal_eval_start, multimodal_eval_end),
            generation_total_ms,
            elapsed_ms(output_save_start, output_save_end),
            elapsed_ms(image_total_start, image_total_end),
            generated_total,
            static_cast<int>(n_past)
        );

        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp);
    }

    llama_sampler_free(sampler);
    mtmd_free(mctx);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}

#if defined(_WIN32)
static std::string wide_to_utf8(const wchar_t * wstr) {
    if (!wstr) {
        return {};
    }
    int size = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, nullptr, 0, nullptr, nullptr);
    if (size <= 1) {
        return {};
    }
    std::string out((size_t) size - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, wstr, -1, out.data(), size, nullptr, nullptr);
    return out;
}

int main() {
    int argc_w = 0;
    wchar_t ** argv_w = CommandLineToArgvW(GetCommandLineW(), &argc_w);
    if (!argv_w || argc_w <= 0) {
        fprintf(stderr, "error: failed to parse command line\n");
        return 1;
    }

    std::vector<std::string> argv_storage;
    argv_storage.reserve((size_t) argc_w);
    std::vector<char *> argv_utf8;
    argv_utf8.reserve((size_t) argc_w);

    for (int i = 0; i < argc_w; ++i) {
        argv_storage.emplace_back(wide_to_utf8(argv_w[i]));
    }
    for (auto & arg : argv_storage) {
        argv_utf8.push_back(arg.data());
    }

    LocalFree(argv_w);
    return qari_ocr_main((int) argv_utf8.size(), argv_utf8.data());
}
#else
int main(int argc, char ** argv) {
    return qari_ocr_main(argc, argv);
}
#endif
