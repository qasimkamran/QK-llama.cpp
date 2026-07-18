#include "qari-loader.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <thread>

namespace qari {

bool LoadGGMLBackend()
{
    ggml_backend_load_all();
    return ggml_backend_reg_count() > 0;
}

llama_model* LoadLanguageModel(const std::string modelPath, const LanguageModelOptions& opts)
{
    if (modelPath.empty())
        return nullptr;

    llama_model_params modelParams = llama_model_default_params();
    modelParams.n_gpu_layers = opts.nGPULayers;

    llama_model* llamaModel = llama_model_load_from_file(modelPath.c_str(), modelParams);

    if (!llamaModel) {
        fprintf(stderr, "error: failed to load text model: %s\n", modelPath.c_str());
        return nullptr;
    }

    return llamaModel;
}

llama_context* CreateLanguageModelContext(llama_model* llamaModel, const LanguageModelOptions& opts)
{
    llama_context_params ctxParams = llama_context_default_params();
    ctxParams.n_ctx = opts.nCtx;
    ctxParams.n_batch = opts.nBatch;
    ctxParams.n_ubatch = opts.nUBatch;
    ctxParams.no_perf = opts.noPerf;

    llama_context* llamaCtx = llama_init_from_model(llamaModel, ctxParams);
    if (!llamaCtx) {
        fprintf(stderr, "error: failed to create llama context\n");
        return nullptr;
    }

    return llamaCtx;
}

mtmd_context* LoadMultimodalContext(const std::string modelPath, const llama_model* llamaModel, const VisionModelOptions& opts)
{
    if (modelPath.empty() || !llamaModel)
        return nullptr;

    mtmd_context_params params = mtmd_context_params_default();
    params.use_gpu = opts.useGPU;
    params.print_timings = opts.printTimings;
    params.n_threads = opts.nThreads > 0
        ? opts.nThreads
        : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    params.image_min_tokens = opts.imageMinTokens;
    params.image_max_tokens = opts.imageMaxTokens;

    mtmd_context* mtmdCtx = mtmd_init_from_file(modelPath.c_str(), llamaModel, params);
    if (!mtmdCtx) {
        fprintf(stderr, "error: failed to load multimodal projector/model: %s\n", modelPath.c_str());
        return nullptr;
    }

    if (!mtmd_support_vision(mtmdCtx)) {
        fprintf(stderr, "error: loaded multimodal context does not support vision input\n");
        mtmd_free(mtmdCtx);
        return nullptr;
    }

    return mtmdCtx;
}

} // namespace qari
