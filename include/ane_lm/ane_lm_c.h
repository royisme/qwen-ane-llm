#ifndef ANE_LM_C_H
#define ANE_LM_C_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ane_model ane_model_t;
typedef struct ane_tokenizer ane_tokenizer_t;

// Generation response structure for callback
typedef struct {
    const char* text;
    int token;
    int prompt_tokens;
    float prompt_tps;
    int generation_tokens;
    float generation_tps;
} ane_response_t;

typedef void (*ane_callback_t)(ane_response_t* resp, void* user_data);

// API Functions
ane_model_t* ane_load_model(const char* model_dir, bool use_ane_cache);
ane_tokenizer_t* ane_load_tokenizer(const char* model_dir);

void ane_generate(
    ane_model_t* model,
    ane_tokenizer_t* tokenizer,
    const char* json_messages,
    int max_tokens,
    float temperature,
    float repetition_penalty,
    bool enable_thinking,
    bool reset_context,
    ane_callback_t callback,
    void* user_data
);

void ane_free_model(ane_model_t* model);
void ane_free_tokenizer(ane_tokenizer_t* tokenizer);

#ifdef __cplusplus
}
#endif

#endif // ANE_LM_C_H
