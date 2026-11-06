#include "transformer_core.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

TransformerAttention* transformer_create(int d_model, int n_heads, int max_seq_len) {
    TransformerAttention* m = (TransformerAttention*)malloc(sizeof(TransformerAttention));
    m->d_model = d_model;
    m->n_heads = n_heads;
    m->d_k = d_model / n_heads;
    m->max_seq_len = max_seq_len;
    m->curr_len = 0;

    size_t cache_size = (size_t)max_seq_len * d_model * sizeof(float);
    m->k_cache = (float*)malloc(cache_size);
    m->v_cache = (float*)malloc(cache_size);

    size_t proj_size = (size_t)d_model * d_model * sizeof(float);
    m->W_q = (float*)malloc(proj_size);
    m->W_k = (float*)malloc(proj_size);
    m->W_v = (float*)malloc(proj_size);
    m->W_out = (float*)malloc(proj_size);

    // Initialize deterministic weights
    srand(42);
    for (size_t i = 0; i < (size_t)d_model * d_model; ++i) {
        m->W_q[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.05f;
        m->W_k[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.05f;
        m->W_v[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.05f;
        m->W_out[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.05f;
    }

    return m;
}

void transformer_free(TransformerAttention* m) {
    if (!m) return;
    free(m->k_cache);
    free(m->v_cache);
    free(m->W_q);
    free(m->W_k);
    free(m->W_v);
    free(m->W_out);
    free(m);
}

void transformer_step(TransformerAttention* m, const float* x_in, float* out_token) {
    int d = m->d_model;
    int nh = m->n_heads;
    int dk = m->d_k;
    int t = m->curr_len;
    float scale = 1.0f / sqrtf((float)dk);

    float* q = (float*)alloca(d * sizeof(float));
    float* k_curr = &m->k_cache[t * d];
    float* v_curr = &m->v_cache[t * d];

    // 1. Project Q, K, V
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < d; ++i) {
        float sum_q = 0.0f;
        float sum_k = 0.0f;
        float sum_v = 0.0f;
        for (int j = 0; j < d; ++j) {
            sum_q += x_in[j] * m->W_q[i * d + j];
            sum_k += x_in[j] * m->W_k[i * d + j];
            sum_v += x_in[j] * m->W_v[i * d + j];
        }
        q[i] = sum_q;
        k_curr[i] = sum_k;
        v_curr[i] = sum_v;
    }

    int seq_len = t + 1;
    m->curr_len = seq_len;

    float* head_outputs = (float*)alloca(d * sizeof(float));

    // 2. Multi-Head Attention over all historical keys in cache (0 to t)
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < nh; ++h) {
        const float* q_h = &q[h * dk];
        float* attn_scores = (float*)alloca(seq_len * sizeof(float));

        float max_score = -1e9f;
        // Dot product with all cached keys
        for (int s = 0; s < seq_len; ++s) {
            const float* k_hs = &m->k_cache[s * d + h * dk];
            float dot = 0.0f;
            for (int k = 0; k < dk; ++k) {
                dot += q_h[k] * k_hs[k];
            }
            float score = dot * scale;
            attn_scores[s] = score;
            if (score > max_score) max_score = score;
        }

        // Softmax
        float sum_exp = 0.0f;
        for (int s = 0; s < seq_len; ++s) {
            attn_scores[s] = expf(attn_scores[s] - max_score);
            sum_exp += attn_scores[s];
        }
        float inv_sum = 1.0f / sum_exp;
        for (int s = 0; s < seq_len; ++s) {
            attn_scores[s] *= inv_sum;
        }

        // Weighted sum of values
        float* out_h = &head_outputs[h * dk];
        for (int k = 0; k < dk; ++k) {
            out_h[k] = 0.0f;
        }
        for (int s = 0; s < seq_len; ++s) {
            float weight = attn_scores[s];
            const float* v_hs = &m->v_cache[s * d + h * dk];
            for (int k = 0; k < dk; ++k) {
                out_h[k] += weight * v_hs[k];
            }
        }
    }

    // 3. Output Projection W_out
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < d; ++i) {
        float sum_out = 0.0f;
        for (int j = 0; j < d; ++j) {
            sum_out += head_outputs[j] * m->W_out[i * d + j];
        }
        out_token[i] = sum_out;
    }
}

void transformer_process_stream(TransformerAttention* m, const float* in_stream, int length, float* out_stream) {
    m->curr_len = 0;
    int d = m->d_model;
    for (int i = 0; i < length; ++i) {
        transformer_step(m, &in_stream[i * d], &out_stream[i * d]);
    }
}

size_t transformer_get_kv_cache_bytes(TransformerAttention* m) {
    return (size_t)m->curr_len * m->d_model * 2 * sizeof(float);
}

