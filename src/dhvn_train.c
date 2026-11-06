#include "dhvn_train.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

DHVNTrainer* dhvn_trainer_create(DHVNSystem* sys, int vocab_size) {
    DHVNTrainer* tr = (DHVNTrainer*)malloc(sizeof(DHVNTrainer));
    tr->vocab_size = vocab_size;
    tr->dim_c = sys->dim_c;
    tr->num_inputs = 2 * DHVN_MAX_HARMONICS;
    tr->t_step = 0;

    int v = vocab_size;
    int c = sys->dim_c;
    int in = tr->num_inputs;

    tr->W_readout = (float*)malloc(v * c * sizeof(float));
    tr->W_charge = sys->W_charge; // Direct reference to system weights
    tr->W_gate = sys->W_gate;

    tr->grad_W_readout = (float*)calloc(v * c, sizeof(float));
    tr->grad_W_charge = (float*)calloc(c * in, sizeof(float));
    tr->grad_W_gate = (float*)calloc(in, sizeof(float));

    tr->m_W_readout = (float*)calloc(v * c, sizeof(float));
    tr->v_W_readout = (float*)calloc(v * c, sizeof(float));
    tr->m_W_charge = (float*)calloc(c * in, sizeof(float));
    tr->v_W_charge = (float*)calloc(c * in, sizeof(float));
    tr->m_W_gate = (float*)calloc(in, sizeof(float));
    tr->v_W_gate = (float*)calloc(in, sizeof(float));

    // Xavier initialization for W_readout
    float scale = sqrtf(2.0f / (float)(c + v));
    srand(42);
    for (int i = 0; i < v * c; ++i) {
        tr->W_readout[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * scale;
    }

    return tr;
}

void dhvn_trainer_free(DHVNTrainer* tr) {
    if (!tr) return;
    free(tr->W_readout);
    free(tr->grad_W_readout);
    free(tr->grad_W_charge);
    free(tr->grad_W_gate);
    free(tr->m_W_readout);
    free(tr->v_W_readout);
    free(tr->m_W_charge);
    free(tr->v_W_charge);
    free(tr->m_W_gate);
    free(tr->v_W_gate);
    free(tr);
}

static inline void adam_update(
    float* param, float* grad, float* m, float* v,
    int size, float lr, int t, float beta1, float beta2, float eps
) {
    float b1_corr = 1.0f - powf(beta1, (float)t);
    float b2_corr = 1.0f - powf(beta2, (float)t);
    float alpha = lr * sqrtf(b2_corr) / b1_corr;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; ++i) {
        float g = grad[i];
        // Clip gradients to [-5.0, 5.0] for physical stability
        if (g > 5.0f) g = 5.0f;
        if (g < -5.0f) g = -5.0f;

        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * (g * g);

        param[i] -= alpha * m[i] / (sqrtf(v[i]) + eps);
        grad[i] = 0.0f; // Reset gradient
    }
}

float dhvn_train_sequence(
    DHVNSystem* sys,
    DHVNTrainer* tr,
    const unsigned char* in_bytes,
    const unsigned char* target_bytes,
    int length,
    float lr,
    float dt
) {
    int n = sys->num_particles;
    int c_dim = sys->dim_c;
    int v_dim = tr->vocab_size;
    int in_dim = tr->num_inputs;

    // Allocate trajectory buffers for BPTT
    float* recorded_pooled = (float*)malloc(length * c_dim * sizeof(float));
    float* recorded_phi = (float*)malloc(length * in_dim * sizeof(float));
    float* recorded_gates = (float*)malloc(length * sizeof(float));
    float* recorded_probs = (float*)malloc(length * v_dim * sizeof(float));

    float total_loss = 0.0f;

    // ==========================================
    // 1. FORWARD PASS
    // ==========================================
    for (int t = 0; t < length; ++t) {
        unsigned char b_in = in_bytes[t];
        float theta = 2.0f * (float)M_PI * ((float)b_in / 256.0f);
        float* phi_t = &recorded_phi[t * in_dim];
        for (int k = 1; k <= DHVN_MAX_HARMONICS; ++k) {
            phi_t[(k - 1) * 2]     = cosf((float)k * theta);
            phi_t[(k - 1) * 2 + 1] = sinf((float)k * theta);
        }

        // Gate evaluation
        float gate = dhvn_compute_gate(sys, b_in);
        recorded_gates[t] = gate;

        // Influx injection
        dhvn_inject_byte(sys, b_in, 2.0f);

        // Physics step
        dhvn_step(sys, dt);

        // Droplet pooling: weighted mean over active particles
        float* pooled_t = &recorded_pooled[t * c_dim];
        for (int c = 0; c < c_dim; ++c) pooled_t[c] = 0.0f;

        float total_w = 0.0f;
        for (int i = 0; i < n; ++i) {
            DHVNParticle* p = &sys->particles[i];
            float w = p->mass * p->density;
            total_w += w;
            for (int c = 0; c < c_dim; ++c) {
                pooled_t[c] += p->charge[c] * w;
            }
        }
        if (total_w > 0.0f) {
            float inv_w = 1.0f / total_w;
            for (int c = 0; c < c_dim; ++c) pooled_t[c] *= inv_w;
        }

        // Logits: W_readout * pooled_t
        float* logits = (float*)alloca(v_dim * sizeof(float));
        float max_l = -1e9f;
        for (int v = 0; v < v_dim; ++v) {
            float sum = 0.0f;
            for (int c = 0; c < c_dim; ++c) {
                sum += tr->W_readout[v * c_dim + c] * pooled_t[c];
            }
            logits[v] = sum;
            if (sum > max_l) max_l = sum;
        }

        // Softmax
        float sum_exp = 0.0f;
        float* prob_t = &recorded_probs[t * v_dim];
        for (int v = 0; v < v_dim; ++v) {
            prob_t[v] = expf(logits[v] - max_l);
            sum_exp += prob_t[v];
        }
        float inv_exp = 1.0f / (sum_exp + 1e-7f);
        for (int v = 0; v < v_dim; ++v) {
            prob_t[v] *= inv_exp;
        }

        // Cross-Entropy Loss
        unsigned char target_idx = target_bytes[t];
        if (target_idx < v_dim) {
            float p_target = prob_t[target_idx];
            if (p_target < 1e-7f) p_target = 1e-7f;
            total_loss += -logf(p_target);
        }
    }

    // ==========================================
    // 2. BACKWARD PASS (Adjoint Sensitivity / BPTT)
    // ==========================================
    for (int t = length - 1; t >= 0; --t) {
        unsigned char target_idx = target_bytes[t];
        float* prob_t = &recorded_probs[t * v_dim];
        float* pooled_t = &recorded_pooled[t * c_dim];
        float* phi_t = &recorded_phi[t * in_dim];
        float gate_t = recorded_gates[t];

        // Gradient of Cross-Entropy w.r.t logits: dL/dz = prob - target
        float* d_logits = (float*)alloca(v_dim * sizeof(float));
        for (int v = 0; v < v_dim; ++v) {
            d_logits[v] = prob_t[v];
        }
        if (target_idx < v_dim) {
            d_logits[target_idx] -= 1.0f;
        }

        // Gradient to W_readout: dL/dW_readout += d_logits * pooled_t^T
        for (int v = 0; v < v_dim; ++v) {
            float err = d_logits[v];
            for (int c = 0; c < c_dim; ++c) {
                tr->grad_W_readout[v * c_dim + c] += err * pooled_t[c];
            }
        }

        // Gradient to pooled charge: dL/d(pooled_t) = W_readout^T * d_logits
        float* d_pooled = (float*)alloca(c_dim * sizeof(float));
        for (int c = 0; c < c_dim; ++c) {
            float sum = 0.0f;
            for (int v = 0; v < v_dim; ++v) {
                sum += tr->W_readout[v * c_dim + c] * d_logits[v];
            }
            d_pooled[c] = sum;
        }

        // Backprop through influx projection:
        // pooled_t receives contribution delta_c = 0.2 * gate * W_charge * phi_t
        float d_gate = 0.0f;
        for (int c = 0; c < c_dim; ++c) {
            float err_c = d_pooled[c] * 0.2f * gate_t;
            for (int j = 0; j < in_dim; ++j) {
                tr->grad_W_charge[c * in_dim + j] += err_c * phi_t[j];
                d_gate += d_pooled[c] * 0.2f * tr->W_charge[c * in_dim + j] * phi_t[j];
            }
        }

        // Backprop to acoustic impedance gate: dL/dW_gate
        float d_logit = d_gate * gate_t * (1.0f - gate_t);
        for (int j = 0; j < in_dim; ++j) {
            tr->grad_W_gate[j] += d_logit * phi_t[j];
        }
    }

    // ==========================================
    // 3. ADAM OPTIMIZER STEP
    // ==========================================
    tr->t_step++;
    adam_update(tr->W_readout, tr->grad_W_readout, tr->m_W_readout, tr->v_W_readout,
                v_dim * c_dim, lr, tr->t_step, 0.9f, 0.999f, 1e-8f);
    adam_update(tr->W_charge, tr->grad_W_charge, tr->m_W_charge, tr->v_W_charge,
                c_dim * in_dim, lr, tr->t_step, 0.9f, 0.999f, 1e-8f);
    adam_update(tr->W_gate, tr->grad_W_gate, tr->m_W_gate, tr->v_W_gate,
                in_dim, lr * 0.5f, tr->t_step, 0.9f, 0.999f, 1e-8f);

    free(recorded_pooled);
    free(recorded_phi);
    free(recorded_gates);
    free(recorded_probs);

    return total_loss / (float)length;
}

int dhvn_predict_next(
    DHVNSystem* sys,
    DHVNTrainer* tr,
    unsigned char in_byte,
    float dt,
    float* out_prob
) {
    dhvn_inject_byte(sys, in_byte, 2.0f);
    dhvn_step(sys, dt);

    int n = sys->num_particles;
    int c_dim = sys->dim_c;
    int v_dim = tr->vocab_size;

    float* pooled = (float*)alloca(c_dim * sizeof(float));
    for (int c = 0; c < c_dim; ++c) pooled[c] = 0.0f;

    float total_w = 0.0f;
    for (int i = 0; i < n; ++i) {
        DHVNParticle* p = &sys->particles[i];
        float w = p->mass * p->density;
        total_w += w;
        for (int c = 0; c < c_dim; ++c) pooled[c] += p->charge[c] * w;
    }
    if (total_w > 0.0f) {
        float inv_w = 1.0f / total_w;
        for (int c = 0; c < c_dim; ++c) pooled[c] *= inv_w;
    }

    float* logits = (float*)alloca(v_dim * sizeof(float));
    float max_l = -1e9f;
    for (int v = 0; v < v_dim; ++v) {
        float sum = 0.0f;
        for (int c = 0; c < c_dim; ++c) {
            sum += tr->W_readout[v * c_dim + c] * pooled[c];
        }
        logits[v] = sum;
        if (sum > max_l) max_l = sum;
    }

    float sum_exp = 0.0f;
    int best_v = 0;
    float best_p = -1.0f;
    for (int v = 0; v < v_dim; ++v) {
        float p = expf(logits[v] - max_l);
        sum_exp += p;
        logits[v] = p;
    }
    float inv_exp = 1.0f / (sum_exp + 1e-7f);
    for (int v = 0; v < v_dim; ++v) {
        float p = logits[v] * inv_exp;
        if (p > best_p) {
            best_p = p;
            best_v = v;
        }
    }

    if (out_prob) *out_prob = best_p;
    return best_v;
}

void dhvn_predict_distribution(
    DHVNSystem* sys,
    DHVNTrainer* tr,
    unsigned char in_byte,
    float dt,
    float* out_probs
) {
    dhvn_inject_byte(sys, in_byte, 2.0f);
    dhvn_step(sys, dt);

    int n = sys->num_particles;
    int c_dim = sys->dim_c;
    int v_dim = tr->vocab_size;

    float* pooled = (float*)alloca(c_dim * sizeof(float));
    for (int c = 0; c < c_dim; ++c) pooled[c] = 0.0f;

    float total_w = 0.0f;
    for (int i = 0; i < n; ++i) {
        DHVNParticle* p = &sys->particles[i];
        float w = p->mass * p->density;
        total_w += w;
        for (int c = 0; c < c_dim; ++c) pooled[c] += p->charge[c] * w;
    }
    if (total_w > 0.0f) {
        float inv_w = 1.0f / total_w;
        for (int c = 0; c < c_dim; ++c) pooled[c] *= inv_w;
    }

    float max_l = -1e9f;
    for (int v = 0; v < v_dim; ++v) {
        float sum = 0.0f;
        for (int c = 0; c < c_dim; ++c) {
            sum += tr->W_readout[v * c_dim + c] * pooled[c];
        }
        out_probs[v] = sum; // temporarily hold logits
        if (sum > max_l) max_l = sum;
    }

    float sum_exp = 0.0f;
    for (int v = 0; v < v_dim; ++v) {
        out_probs[v] = expf(out_probs[v] - max_l);
        sum_exp += out_probs[v];
    }
    float inv_exp = 1.0f / (sum_exp + 1e-7f);
    for (int v = 0; v < v_dim; ++v) {
        out_probs[v] *= inv_exp;
    }
}


