#ifndef TRANSFORMER_CORE_H
#define TRANSFORMER_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

typedef struct {
    int d_model;
    int n_heads;
    int d_k;            // d_model / n_heads
    int max_seq_len;
    int curr_len;
    
    // KV Cache: Layer-level continuous buffers
    // Keys: [max_seq_len, d_model]
    // Values: [max_seq_len, d_model]
    float* k_cache;
    float* v_cache;
    
    // Projections
    float* W_q;
    float* W_k;
    float* W_v;
    float* W_out;
} TransformerAttention;

TransformerAttention* transformer_create(int d_model, int n_heads, int max_seq_len);
void transformer_free(TransformerAttention* model);

// Forward single token step with KV-cache update
void transformer_step(TransformerAttention* model, const float* x_in, float* out_token);

// Process stream of sequence length L
void transformer_process_stream(TransformerAttention* model, const float* in_stream, int length, float* out_stream);

size_t transformer_get_kv_cache_bytes(TransformerAttention* model);

#ifdef __cplusplus
}
#endif

#endif // TRANSFORMER_CORE_H

