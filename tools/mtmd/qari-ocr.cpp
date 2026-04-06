#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static void print_usage(const char * prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s -m <text-model.gguf> -i <image> [--mmproj <mmproj.gguf>] [--prompt <text>] [-n <predict>] [-ngl <layers>] [--image-min-tokens <n>] [--image-max-tokens <n>] [-o <output.txt>]\\n\\n"
        "Example:\n"
        "  %s -m ../qari-ocr-q8_0.gguf --mmproj ../qari-mmproj-f16.gguf -i document.jpg --prompt \"Extract all text exactly.\" -n 512 -ngl 99\\n"
        "\n"
        "Notes:\n"
        "  - For single-file multimodal models, set --mmproj to the same file as -m.\n"
        "  - Use -ngl based on your VRAM (99 tries to offload as much as possible).\n",
        prog, prog);
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

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string mmproj_path;
    std::string image_path;
    std::string output_path;
    std::string prompt = "Extract all text from this image. Return plain text only.";
    int n_predict = 512;
    int n_gpu_layers = 99;
    int image_min_tokens = 256;
    int image_max_tokens = 1024;

    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--mmproj") == 0 && i + 1 < argc) {
            mmproj_path = argv[++i];
        } else if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--image") == 0) && i + 1 < argc) {
            image_path = argv[++i];
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
        } else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) {
            output_path = argv[++i];
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty() || image_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    if (mmproj_path.empty()) {
        mmproj_path = model_path;
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

    mtmd_bitmap * bmp = mtmd_helper_bitmap_init_from_file(mctx, image_path.c_str());
    if (!bmp) {
        fprintf(stderr, "error: failed to load image: %s\n", image_path.c_str());
        mtmd_free(mctx);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const mtmd_bitmap * bitmaps[] = { bmp };

    // Qwen2-VL style chat prompt: force an assistant turn so generation does not end immediately.
    const std::string user_content = std::string(mtmd_default_marker()) + "\n" + prompt;
    const std::string full_prompt =
        "<|im_start|>user\n" + user_content + "<|im_end|>\n"
        "<|im_start|>assistant\n";
    mtmd_input_text in_txt = {
        full_prompt.c_str(),
        true,
        true,
    };

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    if (!chunks) {
        fprintf(stderr, "error: failed to allocate mtmd_input_chunks\n");
        mtmd_bitmap_free(bmp);
        mtmd_free(mctx);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (mtmd_tokenize(mctx, chunks, &in_txt, bitmaps, 1) != 0) {
        fprintf(stderr, "error: mtmd_tokenize() failed\n");
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp);
        mtmd_free(mctx);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_pos n_past = 0;
    if (mtmd_helper_eval_chunks(mctx, ctx, chunks, n_past, 0, ctx_params.n_batch, true, &n_past) != 0) {
        fprintf(stderr, "error: mtmd_helper_eval_chunks() failed\n");
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp);
        mtmd_free(mctx);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    std::string output_text;
    printf("\n");
    for (int i = 0; i < n_predict; ++i) {
        llama_token tok = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) {
            break;
        }

        std::string piece;
        if (!token_to_piece(vocab, tok, piece)) {
            fprintf(stderr, "\nerror: token_to_piece failed\n");
            break;
        }

        printf("%s", piece.c_str());
        fflush(stdout);
        output_text += piece;

        llama_sampler_accept(sampler, tok);

        llama_batch batch = llama_batch_get_one(&tok, 1);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "\nerror: llama_decode failed while generating\n");
            break;
        }

        ++n_past;
    }
    printf("\n");

    if (!output_path.empty()) {
        FILE * fout = fopen(output_path.c_str(), "wb");
        if (!fout) {
            fprintf(stderr, "error: failed to open output file: %s\n", output_path.c_str());
            llama_sampler_free(sampler);
            mtmd_input_chunks_free(chunks);
            mtmd_bitmap_free(bmp);
            mtmd_free(mctx);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        if (!output_text.empty()) {
            fwrite(output_text.data(), 1, output_text.size(), fout);
        }
        fclose(fout);
        fprintf(stderr, "saved output to: %s\n", output_path.c_str());
    }

    llama_sampler_free(sampler);
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bmp);
    mtmd_free(mctx);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
