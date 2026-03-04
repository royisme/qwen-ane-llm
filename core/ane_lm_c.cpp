#include "include/ane_lm/ane_lm_c.h"
#include "utils.h"
#include "generate.h"
#include "include/ane_lm/common.h"
#include <nlohmann/json.hpp>
#include <memory>
#include <vector>
#include <string>

using namespace ane_lm;

// Opaque wrappers
struct ane_model {
    std::unique_ptr<LLMModel> model;
    std::vector<int> session_tokens;
};

struct ane_tokenizer {
    Tokenizer tokenizer;
};

ane_model_t* ane_load_model(const char* model_dir, bool use_ane_cache) {
    try {
        auto result = load(model_dir, use_ane_cache);
        auto* wrapper = new ane_model();
        wrapper->model = std::move(result.first);
        return (ane_model_t*)wrapper;
    } catch (...) {
        return nullptr;
    }
}

ane_tokenizer_t* ane_load_tokenizer(const char* model_dir) {
    try {
        auto result = load(model_dir); // load() returns pair of model + tokenizer
        auto* wrapper = new ane_tokenizer();
        wrapper->tokenizer = std::move(result.second);
        return (ane_tokenizer_t*)wrapper;
    } catch (...) {
        return nullptr;
    }
}

void ane_generate(
    ane_model_t* model_ptr,
    ane_tokenizer_t* tokenizer_ptr,
    const char* json_messages,
    int max_tokens,
    float temperature,
    float repetition_penalty,
    bool enable_thinking,
    bool reset_context,
    ane_callback_t callback,
    void* user_data
) {
    if (!model_ptr || !tokenizer_ptr || !json_messages) return;

    ane_model* m = (ane_model*)model_ptr;
    ane_tokenizer* t = (ane_tokenizer*)tokenizer_ptr;

    std::vector<std::pair<std::string, std::string>> messages;
    try {
        auto j = nlohmann::json::parse(json_messages);
        for (auto& item : j) {
            messages.push_back({item["role"].get<std::string>(), item["content"].get<std::string>()});
        }
    } catch (...) {
        return;
    }

    SamplingParams sampling;
    sampling.temperature = temperature;
    sampling.repetition_penalty = repetition_penalty;

    if (reset_context) {
        m->model->reset();
        m->session_tokens.clear();
    }

    stream_generate(*(m->model), t->tokenizer, messages,
        max_tokens, enable_thinking, sampling,
        [&](const GenerationResponse& r) {
            if (callback) {
                ane_response_t resp;
                resp.text = r.text.c_str();
                resp.token = r.token;
                resp.prompt_tokens = r.prompt_tokens;
                resp.prompt_tps = (float)r.prompt_tps;
                resp.generation_tokens = r.generation_tokens;
                resp.generation_tps = (float)r.generation_tps;
                callback(&resp, user_data);
            }
        },
        &m->session_tokens);
}

void ane_free_model(ane_model_t* model) {
    if (model) delete (ane_model*)model;
}

void ane_free_tokenizer(ane_tokenizer_t* tokenizer) {
    if (tokenizer) delete (ane_tokenizer*)tokenizer;
}
