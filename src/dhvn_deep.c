#include "dhvn_deep.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Activation function: GELU approximation (0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))))
static inline float gelu_act(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
}

static inline float gelu_grad(float x) {
    float x3 = x * x * x;
    float inner = 0.79788456f * (x + 0.044715f * x3);
    float t = tanhf(inner);
    float dt = (1.0f - t * t) * 0.79788456f * (1.0f + 0.134145f * x * x);
    return 0.5f * (1.0f + t) + 0.5f * x * dt;
}

static inline void adam_update(
    float* w, float* grad, float* m, float* v,
    int size, float lr, int t,
    float beta1, float beta2, float eps
) {
    float alpha_t = lr * sqrtf(1.0f - powf(beta2, (float)t)) / (1.0f - powf(beta1, (float)t));
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; ++i) {
        float g = grad[i];
        if (g > 5.0f) g = 5.0f;
        if (g < -5.0f) g = -5.0f;

        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
        w[i] -= alpha_t * m[i] / (sqrtf(v[i]) + eps);
        grad[i] = 0.0f;
    }
}

static void compute_pooled_charge(DHVNSystem* s, float* pooled) {
    int dim = s->dim_c;
    for (int c = 0; c < dim; ++c) pooled[c] = 0.0f;
    int np = s->num_particles;
    for (int i = 0; i < np; ++i) {
        for (int c = 0; c < dim; ++c) {
            pooled[c] += s->particles[i].charge[c];
        }
    }
    float inv_n = 1.0f / (float)np;
    for (int c = 0; c < dim; ++c) pooled[c] *= inv_n;
}

DeepDHVN* deep_dhvn_create_tri(
    int vocab_size,
    int np1, int dim_c1,
    int np2, int dim_c2,
    int np3, int dim_c3,
    int dim_hid
) {
    DeepDHVN* m = (DeepDHVN*)malloc(sizeof(DeepDHVN));
    m->vocab_size = vocab_size;
    m->dim_c1 = dim_c1;
    m->dim_c2 = dim_c2;
    m->dim_c3 = dim_c3;
    m->num_inputs = 2 * DHVN_MAX_HARMONICS;
    m->t_step = 0;
    m->num_substeps = 1;
    m->dim_hid = (dim_hid > 0) ? dim_hid : 256;

    // Chamber 1: Fast acoustic void (micro-eddies, low viscosity)
    m->chamber_char = dhvn_create(np1, dim_c1, 2.0f, 0.7f);
    m->chamber_char->viscosity = 0.02f;
    m->chamber_char->stiffness_B = 1.0f;

    // Chamber 2: Lexical syntactic void (macro-vortices, medium viscosity)
    m->chamber_sem = dhvn_create(np2, dim_c2, 2.5f, 0.85f);
    m->chamber_sem->viscosity = 0.12f;
    m->chamber_sem->stiffness_B = 2.2f;

    // Chamber 3: Propositional discourse void (stable anchor vortices, high viscosity)
    m->chamber_disc = dhvn_create(np3, dim_c3, 3.0f, 0.95f);
    m->chamber_disc->viscosity = 0.25f;
    m->chamber_disc->stiffness_B = 3.5f;

    int v = vocab_size;
    int c1 = dim_c1;
    int c2 = dim_c2;
    int c3 = dim_c3;
    int in_dim = m->num_inputs;
    int dh = m->dim_hid;

    // Waveguide 1: Chamber 1 -> Chamber 2
    m->W_waveguide = (float*)malloc(c2 * c1 * sizeof(float));
    m->grad_W_waveguide = (float*)calloc(c2 * c1, sizeof(float));
    m->m_W_waveguide = (float*)calloc(c2 * c1, sizeof(float));
    m->v_W_waveguide = (float*)calloc(c2 * c1, sizeof(float));

    // Waveguide 2: Chamber 2 -> Chamber 3
    m->W_waveguide2 = (float*)malloc(c3 * c2 * sizeof(float));
    m->grad_W_waveguide2 = (float*)calloc(c3 * c2, sizeof(float));
    m->m_W_waveguide2 = (float*)calloc(c3 * c2, sizeof(float));
    m->v_W_waveguide2 = (float*)calloc(c3 * c2, sizeof(float));

    // Hierarchical residual linear readouts (Chambers 1, 2, 3)
    m->W_out1 = (float*)malloc(v * c1 * sizeof(float));
    m->grad_W_out1 = (float*)calloc(v * c1, sizeof(float));
    m->m_W_out1 = (float*)calloc(v * c1, sizeof(float));
    m->v_W_out1 = (float*)calloc(v * c1, sizeof(float));

    m->W_out2 = (float*)malloc(v * c2 * sizeof(float));
    m->grad_W_out2 = (float*)calloc(v * c2, sizeof(float));
    m->m_W_out2 = (float*)calloc(v * c2, sizeof(float));
    m->v_W_out2 = (float*)calloc(v * c2, sizeof(float));

    m->W_out3 = (float*)malloc(v * c3 * sizeof(float));
    m->grad_W_out3 = (float*)calloc(v * c3, sizeof(float));
    m->m_W_out3 = (float*)calloc(v * c3, sizeof(float));
    m->v_W_out3 = (float*)calloc(v * c3, sizeof(float));

    m->grad_W_charge1 = (float*)calloc(c1 * in_dim, sizeof(float));
    m->m_W_charge1 = (float*)calloc(c1 * in_dim, sizeof(float));
    m->v_W_charge1 = (float*)calloc(c1 * in_dim, sizeof(float));

    // Non-linear Resonant MLP Readout Head
    int c_in = c1 + c2 + c3 + in_dim;
    m->W_hid = (float*)malloc(dh * c_in * sizeof(float));
    m->b_hid = (float*)calloc(dh, sizeof(float));
    m->grad_W_hid = (float*)calloc(dh * c_in, sizeof(float));
    m->grad_b_hid = (float*)calloc(dh, sizeof(float));
    m->m_W_hid = (float*)calloc(dh * c_in, sizeof(float));
    m->v_W_hid = (float*)calloc(dh * c_in, sizeof(float));
    m->m_b_hid = (float*)calloc(dh, sizeof(float));
    m->v_b_hid = (float*)calloc(dh, sizeof(float));

    m->W_head = (float*)malloc(v * dh * sizeof(float));
    m->b_head = (float*)calloc(v, sizeof(float));
    m->grad_W_head = (float*)calloc(v * dh, sizeof(float));
    m->grad_b_head = (float*)calloc(v, sizeof(float));
    m->m_W_head = (float*)calloc(v * dh, sizeof(float));
    m->v_W_head = (float*)calloc(v * dh, sizeof(float));
    m->m_b_head = (float*)calloc(v, sizeof(float));
    m->v_b_head = (float*)calloc(v, sizeof(float));

    // Initialize deterministic weights
    srand(101);
    float sc_w1 = sqrtf(2.0f / (float)(c1 + c2));
    for (int i = 0; i < c2 * c1; ++i) {
        m->W_waveguide[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * sc_w1;
    }
    float sc_w2 = sqrtf(2.0f / (float)(c2 + c3));
    for (int i = 0; i < c3 * c2; ++i) {
        m->W_waveguide2[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * sc_w2;
    }

    float sc_o1 = sqrtf(2.0f / (float)(c1 + v));
    for (int i = 0; i < v * c1; ++i) {
        m->W_out1[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * sc_o1;
    }
    float sc_o2 = sqrtf(2.0f / (float)(c2 + v));
    for (int i = 0; i < v * c2; ++i) {
        m->W_out2[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * sc_o2;
    }
    float sc_o3 = sqrtf(2.0f / (float)(c3 + v));
    for (int i = 0; i < v * c3; ++i) {
        m->W_out3[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * sc_o3;
    }

    float sc_hid = sqrtf(2.0f / (float)c_in);
    for (int i = 0; i < dh * c_in; ++i) {
        m->W_hid[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * sc_hid;
    }
    float sc_head = sqrtf(2.0f / (float)dh);
    for (int i = 0; i < v * dh; ++i) {
        m->W_head[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * sc_head;
    }

    return m;
}

DeepDHVN* deep_dhvn_create(int vocab_size, int np1, int dim_c1, int np2, int dim_c2) {
    return deep_dhvn_create_tri(vocab_size, np1, dim_c1, np2, dim_c2, np2, dim_c2, 256);
}

void deep_dhvn_free(DeepDHVN* m) {
    if (!m) return;
    dhvn_free(m->chamber_char);
    dhvn_free(m->chamber_sem);
    dhvn_free(m->chamber_disc);

    free(m->W_waveguide);
    free(m->grad_W_waveguide);
    free(m->m_W_waveguide);
    free(m->v_W_waveguide);

    free(m->W_waveguide2);
    free(m->grad_W_waveguide2);
    free(m->m_W_waveguide2);
    free(m->v_W_waveguide2);

    free(m->W_out1);
    free(m->grad_W_out1);
    free(m->m_W_out1);
    free(m->v_W_out1);

    free(m->W_out2);
    free(m->grad_W_out2);
    free(m->m_W_out2);
    free(m->v_W_out2);

    free(m->W_out3);
    free(m->grad_W_out3);
    free(m->m_W_out3);
    free(m->v_W_out3);

    free(m->grad_W_charge1);
    free(m->m_W_charge1);
    free(m->v_W_charge1);

    free(m->W_hid);
    free(m->b_hid);
    free(m->grad_W_hid);
    free(m->grad_b_hid);
    free(m->m_W_hid);
    free(m->v_W_hid);
    free(m->m_b_hid);
    free(m->v_b_hid);

    free(m->W_head);
    free(m->b_head);
    free(m->grad_W_head);
    free(m->grad_b_head);
    free(m->m_W_head);
    free(m->v_W_head);
    free(m->m_b_head);
    free(m->v_b_head);

    free(m);
}

void deep_dhvn_set_substeps(DeepDHVN* model, int substeps) {
    if (model && substeps >= 1) model->num_substeps = substeps;
}

// Option 1: Topological Memory Crystallization ("Fluid Sleep" Routine)
void deep_dhvn_crystallize_memory(DeepDHVN* m, int sleep_cycles, float cooling_rate) {
    if (!m) return;
    float dt = 0.03f;
    float damp = 1.0f - cooling_rate;
    if (damp < 0.5f) damp = 0.5f;

    for (int cycle = 0; cycle < sleep_cycles; ++cycle) {
        // 1. Cool velocities across all 3 chambers
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m->chamber_char->num_particles; ++i) {
            for (int d = 0; d < 4; ++d) m->chamber_char->particles[i].vel[d] *= damp;
        }
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m->chamber_sem->num_particles; ++i) {
            for (int d = 0; d < 4; ++d) m->chamber_sem->particles[i].vel[d] *= damp;
        }
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m->chamber_disc->num_particles; ++i) {
            for (int d = 0; d < 4; ++d) m->chamber_disc->particles[i].vel[d] *= damp;
        }

        // 2. Step internal Navier-Stokes dynamics without sensory influx (allows solitons to crystallize)
        dhvn_step(m->chamber_char, dt);
        dhvn_step(m->chamber_sem, dt);
        dhvn_step(m->chamber_disc, dt);
    }
}

// Option 2: Continuous Analog Audio Waveform Ingestion
void deep_dhvn_inject_waveform(DeepDHVN* m, const float* audio_pcm, int num_samples, float dt) {
    if (!m || !audio_pcm || num_samples <= 0) return;

    // Windowed Fibonacci Fourier packet projection
    float packet[2 * DHVN_MAX_HARMONICS];
    for (int k = 0; k < 2 * DHVN_MAX_HARMONICS; ++k) packet[k] = 0.0f;

    int win = (num_samples < 512) ? num_samples : 512;
    for (int k = 0; k < DHVN_MAX_HARMONICS; ++k) {
        float f_k = (float)DHVN_FIBONACCI_HARMONICS[k];
        float sum_c = 0.0f, sum_s = 0.0f;
        for (int n = 0; n < win; ++n) {
            float t = (float)n / (float)win;
            float w = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * t)); // Hanning window
            float s = audio_pcm[n] * w;
            sum_c += s * cosf(2.0f * (float)M_PI * f_k * t);
            sum_s += s * sinf(2.0f * (float)M_PI * f_k * t);
        }
        packet[2 * k] = sum_c / (float)win;
        packet[2 * k + 1] = sum_s / (float)win;
    }

    // Influx directly into Chamber 1
    int c1 = m->dim_c1;
    int in_dim = m->num_inputs;
    float* j_in = (float*)alloca(c1 * sizeof(float));
    for (int c = 0; c < c1; ++c) {
        float sum = 0.0f;
        for (int k = 0; k < in_dim; ++k) sum += m->chamber_char->W_charge[c * in_dim + k] * packet[k];
        j_in[c] = sum;
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m->chamber_char->num_particles; ++i) {
        DHVNParticle* p = &m->chamber_char->particles[i];
        for (int c = 0; c < c1; ++c) {
            p->charge[c] = p->charge[c] * 0.9f + 0.1f * j_in[c];
        }
    }
    dhvn_step(m->chamber_char, dt);
}

// Multi-Step BPTT Sequence Training across 3 Chambers
float deep_dhvn_train_tokens(
    DeepDHVN* m,
    const int* in_tokens,
    const int* target_tokens,
    int length,
    float lr,
    float dt
) {
    int v_dim = m->vocab_size;
    int c1 = m->dim_c1;
    int c2 = m->dim_c2;
    int c3 = m->dim_c3;
    int in_dim = m->num_inputs;
    int dh = m->dim_hid;

    float* rec_pooled1 = (float*)malloc(length * c1 * sizeof(float));
    float* rec_pooled2 = (float*)malloc(length * c2 * sizeof(float));
    float* rec_pooled3 = (float*)malloc(length * c3 * sizeof(float));
    float* rec_phi = (float*)malloc(length * in_dim * sizeof(float));
    float* rec_probs = (float*)malloc(length * v_dim * sizeof(float));
    float* rec_h = (float*)malloc(length * dh * sizeof(float));
    float* rec_h_pre = (float*)malloc(length * dh * sizeof(float));

    float total_loss = 0.0f;

    // 1. FORWARD PASS
    for (int t = 0; t < length; ++t) {
        int t_in = in_tokens[t];
        float theta = 2.0f * (float)M_PI * ((float)(t_in % v_dim) / (float)v_dim);
        float* phi_t = &rec_phi[t * in_dim];
        dhvn_compute_fourier_packet(theta, phi_t);

        int n_sub = (m->num_substeps >= 1) ? m->num_substeps : 1;
        float sub_dt = dt / (float)n_sub;

        // --- Chamber 1: Acoustic Influx ---
        dhvn_inject_token(m->chamber_char, t_in, v_dim, 2.0f);
        for (int s = 0; s < n_sub; ++s) dhvn_step(m->chamber_char, sub_dt);
        float* pooled1_t = &rec_pooled1[t * c1];
        compute_pooled_charge(m->chamber_char, pooled1_t);

        // --- Waveguide 1: Chamber 1 -> Chamber 2 ---
        float* j_1to2 = (float*)alloca(c2 * sizeof(float));
        float norm_j1 = 0.0f;
        for (int i = 0; i < c2; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < c1; ++j) sum += m->W_waveguide[i * c1 + j] * pooled1_t[j];
            j_1to2[i] = sum;
            norm_j1 += sum * sum;
        }
        float sc_j1 = 1.0f / (sqrtf(norm_j1) + 1e-6f);
        for (int i = 0; i < c2; ++i) j_1to2[i] *= sc_j1;

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m->chamber_sem->num_particles; ++i) {
            DHVNParticle* p2 = &m->chamber_sem->particles[i];
            float r = sqrtf(p2->pos[0]*p2->pos[0] + p2->pos[1]*p2->pos[1] +
                            p2->pos[2]*p2->pos[2] + p2->pos[3]*p2->pos[3]);
            if (r > m->chamber_sem->radius * 0.7f) {
                for (int c = 0; c < c2; ++c) p2->charge[c] = p2->charge[c] * 0.85f + 0.15f * j_1to2[c];
            }
        }
        for (int s = 0; s < n_sub; ++s) dhvn_step(m->chamber_sem, sub_dt);
        float* pooled2_t = &rec_pooled2[t * c2];
        compute_pooled_charge(m->chamber_sem, pooled2_t);

        // --- Waveguide 2: Chamber 2 -> Chamber 3 ---
        float* j_2to3 = (float*)alloca(c3 * sizeof(float));
        float norm_j2 = 0.0f;
        for (int i = 0; i < c3; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < c2; ++j) sum += m->W_waveguide2[i * c2 + j] * pooled2_t[j];
            j_2to3[i] = sum;
            norm_j2 += sum * sum;
        }
        float sc_j2 = 1.0f / (sqrtf(norm_j2) + 1e-6f);
        for (int i = 0; i < c3; ++i) j_2to3[i] *= sc_j2;

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m->chamber_disc->num_particles; ++i) {
            DHVNParticle* p3 = &m->chamber_disc->particles[i];
            float r = sqrtf(p3->pos[0]*p3->pos[0] + p3->pos[1]*p3->pos[1] +
                            p3->pos[2]*p3->pos[2] + p3->pos[3]*p3->pos[3]);
            if (r > m->chamber_disc->radius * 0.7f) {
                for (int c = 0; c < c3; ++c) p3->charge[c] = p3->charge[c] * 0.85f + 0.15f * j_2to3[c];
            }
        }
        for (int s = 0; s < n_sub; ++s) dhvn_step(m->chamber_disc, sub_dt);
        float* pooled3_t = &rec_pooled3[t * c3];
        compute_pooled_charge(m->chamber_disc, pooled3_t);

        // --- Non-linear Resonant MLP Readout Head ---
        int c_in = c1 + c2 + c3 + in_dim;
        float* x_in = (float*)alloca(c_in * sizeof(float));
        memcpy(x_in, pooled1_t, c1 * sizeof(float));
        memcpy(x_in + c1, pooled2_t, c2 * sizeof(float));
        memcpy(x_in + c1 + c2, pooled3_t, c3 * sizeof(float));
        memcpy(x_in + c1 + c2 + c3, phi_t, in_dim * sizeof(float));

        float* h_t = &rec_h[t * dh];
        float* h_pre_t = &rec_h_pre[t * dh];
        for (int k = 0; k < dh; ++k) {
            float val = m->b_hid[k];
            for (int j = 0; j < c_in; ++j) val += m->W_hid[k * c_in + j] * x_in[j];
            h_pre_t[k] = val;
            h_t[k] = gelu_act(val);
        }

        float* logits = (float*)alloca(v_dim * sizeof(float));
        float max_l = -1e9f;
        float scale_res = (1.0f / sqrtf((float)(c1 + c2 + c3))) * 0.4f;
        float sc_head = 1.0f / sqrtf((float)dh);

        #pragma omp parallel for schedule(static)
        for (int v = 0; v < v_dim; ++v) {
            float sum_head = m->b_head[v];
            for (int k = 0; k < dh; ++k) sum_head += m->W_head[v * dh + k] * h_t[k];

            float s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            for (int c = 0; c < c1; ++c) s1 += m->W_out1[v * c1 + c] * pooled1_t[c];
            for (int c = 0; c < c2; ++c) s2 += m->W_out2[v * c2 + c] * pooled2_t[c];
            for (int c = 0; c < c3; ++c) s3 += m->W_out3[v * c3 + c] * pooled3_t[c];

            logits[v] = sum_head * sc_head + (s1 + s2 + s3) * scale_res;
        }

        for (int v = 0; v < v_dim; ++v) {
            if (logits[v] > max_l) max_l = logits[v];
        }

        // Softmax
        float sum_exp = 0.0f;
        float* prob_t = &rec_probs[t * v_dim];
        for (int v = 0; v < v_dim; ++v) {
            prob_t[v] = expf(logits[v] - max_l);
            sum_exp += prob_t[v];
        }
        float inv_exp = 1.0f / (sum_exp + 1e-7f);
        for (int v = 0; v < v_dim; ++v) prob_t[v] *= inv_exp;

        int target_idx = target_tokens[t];
        if (target_idx >= 0 && target_idx < v_dim) {
            float p_target = prob_t[target_idx];
            if (p_target < 1e-7f) p_target = 1e-7f;
            total_loss += -logf(p_target);
        }
    }

    // 2. BACKWARD PASS (Multi-Stage Temporal Adjoint Momentum BPTT)
    float* temporal_adj1 = (float*)calloc(c1, sizeof(float));

    for (int t = length - 1; t >= 0; --t) {
        int target_idx = target_tokens[t];
        float* prob_t = &rec_probs[t * v_dim];
        float* pooled1_t = &rec_pooled1[t * c1];
        float* pooled2_t = &rec_pooled2[t * c2];
        float* pooled3_t = &rec_pooled3[t * c3];
        float* phi_t = &rec_phi[t * in_dim];

        float* d_logits = (float*)alloca(v_dim * sizeof(float));
        for (int v = 0; v < v_dim; ++v) d_logits[v] = prob_t[v];
        if (target_idx >= 0 && target_idx < v_dim) d_logits[target_idx] -= 1.0f;

        float* dh_grad = (float*)alloca(dh * sizeof(float));
        for (int k = 0; k < dh; ++k) dh_grad[k] = 0.0f;

        float* h_t = &rec_h[t * dh];
        float* h_pre_t = &rec_h_pre[t * dh];

        float sc_head = 1.0f / sqrtf((float)dh);
        for (int v = 0; v < v_dim; ++v) {
            float err = d_logits[v] * sc_head;
            m->grad_b_head[v] += err;
            for (int k = 0; k < dh; ++k) {
                m->grad_W_head[v * dh + k] += err * h_t[k];
                dh_grad[k] += m->W_head[v * dh + k] * err;
            }
        }

        float* dh_pre = (float*)alloca(dh * sizeof(float));
        for (int k = 0; k < dh; ++k) {
            dh_pre[k] = dh_grad[k] * gelu_grad(h_pre_t[k]);
            m->grad_b_hid[k] += dh_pre[k];
        }

        int c_in = c1 + c2 + c3 + in_dim;
        float* dx_in = (float*)alloca(c_in * sizeof(float));
        for (int j = 0; j < c_in; ++j) dx_in[j] = 0.0f;

        float* x_in = (float*)alloca(c_in * sizeof(float));
        memcpy(x_in, pooled1_t, c1 * sizeof(float));
        memcpy(x_in + c1, pooled2_t, c2 * sizeof(float));
        memcpy(x_in + c1 + c2, pooled3_t, c3 * sizeof(float));
        memcpy(x_in + c1 + c2 + c3, phi_t, in_dim * sizeof(float));

        for (int k = 0; k < dh; ++k) {
            float err_h = dh_pre[k];
            for (int j = 0; j < c_in; ++j) {
                m->grad_W_hid[k * c_in + j] += err_h * x_in[j];
                dx_in[j] += m->W_hid[k * c_in + j] * err_h;
            }
        }

        float scale_res = (1.0f / sqrtf((float)(c1 + c2 + c3))) * 0.4f;
        for (int v = 0; v < v_dim; ++v) {
            float err = d_logits[v] * scale_res;
            for (int c = 0; c < c1; ++c) m->grad_W_out1[v * c1 + c] += err * pooled1_t[c];
            for (int c = 0; c < c2; ++c) m->grad_W_out2[v * c2 + c] += err * pooled2_t[c];
            for (int c = 0; c < c3; ++c) m->grad_W_out3[v * c3 + c] += err * pooled3_t[c];
        }

        // Adjoint to Chamber 3
        float* d_pooled3 = (float*)alloca(c3 * sizeof(float));
        for (int c = 0; c < c3; ++c) {
            float sum = dx_in[c1 + c2 + c];
            for (int v = 0; v < v_dim; ++v) sum += m->W_out3[v * c3 + c] * d_logits[v] * scale_res;
            d_pooled3[c] = sum;
        }

        // Gradients to Waveguide 2 (Chamber 2 -> Chamber 3)
        for (int i = 0; i < c3; ++i) {
            float err3 = d_pooled3[i];
            for (int j = 0; j < c2; ++j) m->grad_W_waveguide2[i * c2 + j] += err3 * pooled2_t[j];
        }

        // Adjoint to Chamber 2
        float* d_pooled2 = (float*)alloca(c2 * sizeof(float));
        for (int c = 0; c < c2; ++c) {
            float sum = dx_in[c1 + c];
            for (int v = 0; v < v_dim; ++v) sum += m->W_out2[v * c2 + c] * d_logits[v] * scale_res;
            for (int i = 0; i < c3; ++i) sum += m->W_waveguide2[i * c2 + c] * d_pooled3[i];
            d_pooled2[c] = sum;
        }

        // Gradients to Waveguide 1 (Chamber 1 -> Chamber 2)
        for (int i = 0; i < c2; ++i) {
            float err2 = d_pooled2[i];
            for (int j = 0; j < c1; ++j) m->grad_W_waveguide[i * c1 + j] += err2 * pooled1_t[j];
        }

        // Adjoint to Chamber 1
        float* d_step1 = (float*)alloca(c1 * sizeof(float));
        for (int j = 0; j < c1; ++j) {
            float sum = dx_in[j];
            for (int v = 0; v < v_dim; ++v) sum += m->W_out1[v * c1 + j] * d_logits[v] * scale_res;
            for (int i = 0; i < c2; ++i) sum += m->W_waveguide[i * c1 + j] * d_pooled2[i];
            d_step1[j] = sum;
        }

        for (int j = 0; j < c1; ++j) temporal_adj1[j] = 0.70f * temporal_adj1[j] + d_step1[j];

        for (int c = 0; c < c1; ++c) {
            float eff_c = temporal_adj1[c] * 0.1f;
            for (int j = 0; j < in_dim; ++j) m->grad_W_charge1[c * in_dim + j] += eff_c * phi_t[j];
        }
    }

    free(temporal_adj1);

    // 3. UNIFIED ADAM OPTIMIZATION STEP
    m->t_step++;
    adam_update(m->W_waveguide, m->grad_W_waveguide, m->m_W_waveguide, m->v_W_waveguide,
                c2 * c1, lr, m->t_step, 0.9f, 0.999f, 1e-8f);
    adam_update(m->W_waveguide2, m->grad_W_waveguide2, m->m_W_waveguide2, m->v_W_waveguide2,
                c3 * c2, lr, m->t_step, 0.9f, 0.999f, 1e-8f);

    adam_update(m->W_out1, m->grad_W_out1, m->m_W_out1, m->v_W_out1,
                v_dim * c1, lr * 0.2f, m->t_step, 0.9f, 0.999f, 1e-8f);
    adam_update(m->W_out2, m->grad_W_out2, m->m_W_out2, m->v_W_out2,
                v_dim * c2, lr * 0.2f, m->t_step, 0.9f, 0.999f, 1e-8f);
    adam_update(m->W_out3, m->grad_W_out3, m->m_W_out3, m->v_W_out3,
                v_dim * c3, lr * 0.2f, m->t_step, 0.9f, 0.999f, 1e-8f);

    float lr_mlp = lr * 0.15f;
    int c_in = c1 + c2 + c3 + in_dim;
    adam_update(m->W_hid, m->grad_W_hid, m->m_W_hid, m->v_W_hid,
                dh * c_in, lr_mlp, m->t_step, 0.9f, 0.999f, 1e-8f);
    adam_update(m->b_hid, m->grad_b_hid, m->m_b_hid, m->v_b_hid,
                dh, lr_mlp, m->t_step, 0.9f, 0.999f, 1e-8f);
    adam_update(m->W_head, m->grad_W_head, m->m_W_head, m->v_W_head,
                v_dim * dh, lr_mlp, m->t_step, 0.9f, 0.999f, 1e-8f);
    adam_update(m->b_head, m->grad_b_head, m->m_b_head, m->v_b_head,
                v_dim, lr_mlp, m->t_step, 0.9f, 0.999f, 1e-8f);
    adam_update(m->chamber_char->W_charge, m->grad_W_charge1, m->m_W_charge1, m->v_W_charge1,
                c1 * in_dim, lr * 0.5f, m->t_step, 0.9f, 0.999f, 1e-8f);

    free(rec_pooled1);
    free(rec_pooled2);
    free(rec_pooled3);
    free(rec_phi);
    free(rec_probs);
    free(rec_h);
    free(rec_h_pre);

    return total_loss / (float)length;
}

float deep_dhvn_train_sequence(
    DeepDHVN* model,
    const unsigned char* in_bytes,
    const unsigned char* target_bytes,
    int length,
    float lr,
    float dt
) {
    int* in_tok = (int*)malloc(length * sizeof(int));
    int* tgt_tok = (int*)malloc(length * sizeof(int));
    for (int i = 0; i < length; ++i) {
        in_tok[i] = (int)in_bytes[i];
        tgt_tok[i] = (int)target_bytes[i];
    }
    float loss = deep_dhvn_train_tokens(model, in_tok, tgt_tok, length, lr, dt);
    free(in_tok);
    free(tgt_tok);
    return loss;
}

void deep_dhvn_predict_token_distribution(
    DeepDHVN* m,
    int in_token,
    float dt,
    float* out_probs
) {
    int v_dim = m->vocab_size;
    int c1 = m->dim_c1;
    int c2 = m->dim_c2;
    int c3 = m->dim_c3;
    int in_dim = m->num_inputs;
    int dh = m->dim_hid;

    float theta = 2.0f * (float)M_PI * ((float)(in_token % v_dim) / (float)v_dim);
    float* phi_t = (float*)alloca(in_dim * sizeof(float));
    dhvn_compute_fourier_packet(theta, phi_t);

    int n_sub = (m->num_substeps >= 1) ? m->num_substeps : 1;
    float sub_dt = dt / (float)n_sub;

    // Chamber 1
    dhvn_inject_token(m->chamber_char, in_token, v_dim, 2.0f);
    for (int s = 0; s < n_sub; ++s) dhvn_step(m->chamber_char, sub_dt);
    float* pooled1 = (float*)alloca(c1 * sizeof(float));
    compute_pooled_charge(m->chamber_char, pooled1);

    // Waveguide 1 -> Chamber 2
    float* j_1to2 = (float*)alloca(c2 * sizeof(float));
    float norm_j1 = 0.0f;
    for (int i = 0; i < c2; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < c1; ++j) sum += m->W_waveguide[i * c1 + j] * pooled1[j];
        j_1to2[i] = sum;
        norm_j1 += sum * sum;
    }
    float sc_j1 = 1.0f / (sqrtf(norm_j1) + 1e-6f);
    for (int i = 0; i < c2; ++i) j_1to2[i] *= sc_j1;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m->chamber_sem->num_particles; ++i) {
        DHVNParticle* p2 = &m->chamber_sem->particles[i];
        float r = sqrtf(p2->pos[0]*p2->pos[0] + p2->pos[1]*p2->pos[1] +
                        p2->pos[2]*p2->pos[2] + p2->pos[3]*p2->pos[3]);
        if (r > m->chamber_sem->radius * 0.7f) {
            for (int c = 0; c < c2; ++c) p2->charge[c] = p2->charge[c] * 0.85f + 0.15f * j_1to2[c];
        }
    }
    for (int s = 0; s < n_sub; ++s) dhvn_step(m->chamber_sem, sub_dt);
    float* pooled2 = (float*)alloca(c2 * sizeof(float));
    compute_pooled_charge(m->chamber_sem, pooled2);

    // Waveguide 2 -> Chamber 3
    float* j_2to3 = (float*)alloca(c3 * sizeof(float));
    float norm_j2 = 0.0f;
    for (int i = 0; i < c3; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < c2; ++j) sum += m->W_waveguide2[i * c2 + j] * pooled2[j];
        j_2to3[i] = sum;
        norm_j2 += sum * sum;
    }
    float sc_j2 = 1.0f / (sqrtf(norm_j2) + 1e-6f);
    for (int i = 0; i < c3; ++i) j_2to3[i] *= sc_j2;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m->chamber_disc->num_particles; ++i) {
        DHVNParticle* p3 = &m->chamber_disc->particles[i];
        float r = sqrtf(p3->pos[0]*p3->pos[0] + p3->pos[1]*p3->pos[1] +
                        p3->pos[2]*p3->pos[2] + p3->pos[3]*p3->pos[3]);
        if (r > m->chamber_disc->radius * 0.7f) {
            for (int c = 0; c < c3; ++c) p3->charge[c] = p3->charge[c] * 0.85f + 0.15f * j_2to3[c];
        }
    }
    for (int s = 0; s < n_sub; ++s) dhvn_step(m->chamber_disc, sub_dt);
    float* pooled3 = (float*)alloca(c3 * sizeof(float));
    compute_pooled_charge(m->chamber_disc, pooled3);

    // Resonant Readout
    int c_in = c1 + c2 + c3 + in_dim;
    float* x_in = (float*)alloca(c_in * sizeof(float));
    memcpy(x_in, pooled1, c1 * sizeof(float));
    memcpy(x_in + c1, pooled2, c2 * sizeof(float));
    memcpy(x_in + c1 + c2, pooled3, c3 * sizeof(float));
    memcpy(x_in + c1 + c2 + c3, phi_t, in_dim * sizeof(float));

    float* h = (float*)alloca(dh * sizeof(float));
    for (int k = 0; k < dh; ++k) {
        float val = m->b_hid[k];
        for (int j = 0; j < c_in; ++j) val += m->W_hid[k * c_in + j] * x_in[j];
        h[k] = gelu_act(val);
    }

    float* logits = (float*)alloca(v_dim * sizeof(float));
    float max_l = -1e9f;
    float scale_res = (1.0f / sqrtf((float)(c1 + c2 + c3))) * 0.4f;
    float sc_head = 1.0f / sqrtf((float)dh);

    for (int v = 0; v < v_dim; ++v) {
        float sum_head = m->b_head[v];
        for (int k = 0; k < dh; ++k) sum_head += m->W_head[v * dh + k] * h[k];

        float s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (int c = 0; c < c1; ++c) s1 += m->W_out1[v * c1 + c] * pooled1[c];
        for (int c = 0; c < c2; ++c) s2 += m->W_out2[v * c2 + c] * pooled2[c];
        for (int c = 0; c < c3; ++c) s3 += m->W_out3[v * c3 + c] * pooled3[c];

        float logit = sum_head * sc_head + (s1 + s2 + s3) * scale_res;
        logits[v] = logit;
        if (logit > max_l) max_l = logit;
    }

    float sum_exp = 0.0f;
    for (int v = 0; v < v_dim; ++v) {
        out_probs[v] = expf(logits[v] - max_l);
        sum_exp += out_probs[v];
    }
    float inv_exp = 1.0f / (sum_exp + 1e-7f);
    for (int v = 0; v < v_dim; ++v) out_probs[v] *= inv_exp;
}

void deep_dhvn_predict_distribution(DeepDHVN* model, unsigned char in_byte, float dt, float* out_probs) {
    deep_dhvn_predict_token_distribution(model, (int)in_byte, dt, out_probs);
}

// Active Inference Reflex Loop: Self-referential internal contemplation & counterfactual testing
void deep_dhvn_active_inference_step(
    DeepDHVN* m,
    int in_token,
    int ponder_steps,
    float reflex_strength,
    float dt,
    float* out_probs,
    float* out_free_energy
) {
    if (!m || !out_probs) return;
    int v_dim = m->vocab_size;
    int c2 = m->dim_c2;
    int c3 = m->dim_c3;

    // Step 0: Initial direct prediction
    deep_dhvn_predict_token_distribution(m, in_token, dt, out_probs);

    // Initial Free Energy
    float max_p = 0.0f;
    for (int v = 0; v < v_dim; ++v) if (out_probs[v] > max_p) max_p = out_probs[v];
    float initial_fe = -logf(max_p + 1e-7f);
    if (out_free_energy) out_free_energy[0] = initial_fe;

    if (ponder_steps <= 0) return;

    for (int step = 0; step < ponder_steps; ++step) {
        // Synthesize counterfactual efference copy phi_pred
        float* phi_pred2 = (float*)calloc(c2, sizeof(float));
        float* phi_pred3 = (float*)calloc(c3, sizeof(float));

        for (int v = 0; v < v_dim; ++v) {
            float p = out_probs[v];
            if (p > 0.005f) {
                for (int c = 0; c < c2; ++c) phi_pred2[c] += p * m->W_out2[v * c2 + c];
                for (int c = 0; c < c3; ++c) phi_pred3[c] += p * m->W_out3[v * c3 + c];
            }
        }

        // Normalize counterfactual copies
        float norm2 = 0.0f, norm3 = 0.0f;
        for (int c = 0; c < c2; ++c) norm2 += phi_pred2[c] * phi_pred2[c];
        for (int c = 0; c < c3; ++c) norm3 += phi_pred3[c] * phi_pred3[c];
        float sc2 = reflex_strength / (sqrtf(norm2) + 1e-6f);
        float sc3 = reflex_strength / (sqrtf(norm3) + 1e-6f);
        for (int c = 0; c < c2; ++c) phi_pred2[c] *= sc2;
        for (int c = 0; c < c3; ++c) phi_pred3[c] *= sc3;

        // Feedback counterfactual copy into Chamber 2
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m->chamber_sem->num_particles; ++i) {
            DHVNParticle* p = &m->chamber_sem->particles[i];
            for (int c = 0; c < c2; ++c) {
                p->charge[c] = p->charge[c] * (1.0f - 0.2f * reflex_strength) + 0.2f * phi_pred2[c];
            }
        }

        // Feedback counterfactual copy into Chamber 3
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m->chamber_disc->num_particles; ++i) {
            DHVNParticle* p = &m->chamber_disc->particles[i];
            for (int c = 0; c < c3; ++c) {
                p->charge[c] = p->charge[c] * (1.0f - 0.2f * reflex_strength) + 0.2f * phi_pred3[c];
            }
        }

        free(phi_pred2);
        free(phi_pred3);

        // Relax fluid chambers by sub_dt
        dhvn_step(m->chamber_sem, dt * 0.5f);
        dhvn_step(m->chamber_disc, dt * 0.5f);

        // Re-read distribution
        float* pooled1 = (float*)alloca(m->dim_c1 * sizeof(float));
        float* pooled2 = (float*)alloca(c2 * sizeof(float));
        float* pooled3 = (float*)alloca(c3 * sizeof(float));
        compute_pooled_charge(m->chamber_char, pooled1);
        compute_pooled_charge(m->chamber_sem, pooled2);
        compute_pooled_charge(m->chamber_disc, pooled3);

        int in_dim = m->num_inputs;
        float theta = 2.0f * (float)M_PI * ((float)(in_token % v_dim) / (float)v_dim);
        float* phi_t = (float*)alloca(in_dim * sizeof(float));
        dhvn_compute_fourier_packet(theta, phi_t);

        int c1 = m->dim_c1;
        int dh = m->dim_hid;
        int c_in = c1 + c2 + c3 + in_dim;
        float* x_in = (float*)alloca(c_in * sizeof(float));
        memcpy(x_in, pooled1, c1 * sizeof(float));
        memcpy(x_in + c1, pooled2, c2 * sizeof(float));
        memcpy(x_in + c1 + c2, pooled3, c3 * sizeof(float));
        memcpy(x_in + c1 + c2 + c3, phi_t, in_dim * sizeof(float));

        float* h = (float*)alloca(dh * sizeof(float));
        for (int k = 0; k < dh; ++k) {
            float val = m->b_hid[k];
            for (int j = 0; j < c_in; ++j) val += m->W_hid[k * c_in + j] * x_in[j];
            h[k] = gelu_act(val);
        }

        float max_l = -1e9f;
        float scale_res = (1.0f / sqrtf((float)(c1 + c2 + c3))) * 0.4f;
        float sc_head = 1.0f / sqrtf((float)dh);

        float* logits = (float*)alloca(v_dim * sizeof(float));
        for (int v = 0; v < v_dim; ++v) {
            float sum_head = m->b_head[v];
            for (int k = 0; k < dh; ++k) sum_head += m->W_head[v * dh + k] * h[k];

            float s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            for (int c = 0; c < c1; ++c) s1 += m->W_out1[v * c1 + c] * pooled1[c];
            for (int c = 0; c < c2; ++c) s2 += m->W_out2[v * c2 + c] * pooled2[c];
            for (int c = 0; c < c3; ++c) s3 += m->W_out3[v * c3 + c] * pooled3[c];

            float logit = sum_head * sc_head + (s1 + s2 + s3) * scale_res;
            logits[v] = logit;
            if (logit > max_l) max_l = logit;
        }

        float sum_exp = 0.0f;
        for (int v = 0; v < v_dim; ++v) {
            out_probs[v] = expf(logits[v] - max_l);
            sum_exp += out_probs[v];
        }
        float inv_exp = 1.0f / (sum_exp + 1e-7f);
        max_p = 0.0f;
        for (int v = 0; v < v_dim; ++v) {
            out_probs[v] *= inv_exp;
            if (out_probs[v] > max_p) max_p = out_probs[v];
        }

        if (out_free_energy) {
            out_free_energy[step + 1] = -logf(max_p + 1e-7f);
        }
    }
}

size_t deep_dhvn_get_memory_footprint(DeepDHVN* m) {
    if (!m) return 0;
    size_t sz = sizeof(DeepDHVN);
    sz += dhvn_get_memory_footprint(m->chamber_char);
    sz += dhvn_get_memory_footprint(m->chamber_sem);
    sz += dhvn_get_memory_footprint(m->chamber_disc);

    int v = m->vocab_size;
    int c1 = m->dim_c1;
    int c2 = m->dim_c2;
    int c3 = m->dim_c3;
    int in_dim = m->num_inputs;
    int dh = m->dim_hid;
    int c_in = c1 + c2 + c3 + in_dim;

    sz += 4 * (size_t)(c2 * c1) * sizeof(float);
    sz += 4 * (size_t)(c3 * c2) * sizeof(float);

    sz += 4 * (size_t)(v * c1) * sizeof(float);
    sz += 4 * (size_t)(v * c2) * sizeof(float);
    sz += 4 * (size_t)(v * c3) * sizeof(float);

    sz += 3 * (size_t)(c1 * in_dim) * sizeof(float);

    sz += 4 * (size_t)(dh * c_in) * sizeof(float);
    sz += 4 * (size_t)dh * sizeof(float);
    sz += 4 * (size_t)(v * dh) * sizeof(float);
    sz += 4 * (size_t)v * sizeof(float);

    return sz;
}

int deep_dhvn_save_weights(DeepDHVN* m, const char* filepath) {
    if (!m || !filepath) return 0;
    FILE* fp = fopen(filepath, "wb");
    if (!fp) return 0;

    int magic = 0x4448564E; // "DHVN"
    int version = 3;        // Tri-Chamber Version 3
    fwrite(&magic, sizeof(int), 1, fp);
    fwrite(&version, sizeof(int), 1, fp);
    fwrite(&m->vocab_size, sizeof(int), 1, fp);
    fwrite(&m->dim_c1, sizeof(int), 1, fp);
    fwrite(&m->dim_c2, sizeof(int), 1, fp);
    fwrite(&m->dim_c3, sizeof(int), 1, fp);
    fwrite(&m->dim_hid, sizeof(int), 1, fp);

    int v = m->vocab_size;
    int c1 = m->dim_c1;
    int c2 = m->dim_c2;
    int c3 = m->dim_c3;
    int in_dim = m->num_inputs;
    int dh = m->dim_hid;
    int c_in = c1 + c2 + c3 + in_dim;

    fwrite(m->W_waveguide, sizeof(float), c2 * c1, fp);
    fwrite(m->W_waveguide2, sizeof(float), c3 * c2, fp);
    fwrite(m->W_out1, sizeof(float), v * c1, fp);
    fwrite(m->W_out2, sizeof(float), v * c2, fp);
    fwrite(m->W_out3, sizeof(float), v * c3, fp);
    fwrite(m->W_hid, sizeof(float), dh * c_in, fp);
    fwrite(m->b_hid, sizeof(float), dh, fp);
    fwrite(m->W_head, sizeof(float), v * dh, fp);
    fwrite(m->b_head, sizeof(float), v, fp);
    fwrite(m->chamber_char->W_charge, sizeof(float), c1 * in_dim, fp);

    fclose(fp);
    return 1;
}

int deep_dhvn_load_weights(DeepDHVN* m, const char* filepath) {
    if (!m || !filepath) return 0;
    FILE* fp = fopen(filepath, "rb");
    if (!fp) return 0;

    int magic = 0, version = 0;
    if (fread(&magic, sizeof(int), 1, fp) != 1) { fclose(fp); return 0; }
    if (fread(&version, sizeof(int), 1, fp) != 1) { fclose(fp); return 0; }

    int f_vocab = 0, f_c1 = 0, f_c2 = 0, f_c3 = 0, f_dh = 0;
    if (version == 3) {
        fread(&f_vocab, sizeof(int), 1, fp);
        fread(&f_c1, sizeof(int), 1, fp);
        fread(&f_c2, sizeof(int), 1, fp);
        fread(&f_c3, sizeof(int), 1, fp);
        fread(&f_dh, sizeof(int), 1, fp);
    } else if (version == 2) {
        fread(&f_vocab, sizeof(int), 1, fp);
        fread(&f_c1, sizeof(int), 1, fp);
        fread(&f_c2, sizeof(int), 1, fp);
        fread(&f_dh, sizeof(int), 1, fp);
        f_c3 = f_c2;
    } else {
        fclose(fp);
        return 0;
    }

    if (f_vocab != m->vocab_size || f_c1 != m->dim_c1 || f_c2 != m->dim_c2) {
        fclose(fp);
        return 0;
    }

    int v = m->vocab_size;
    int c1 = m->dim_c1;
    int c2 = m->dim_c2;
    int c3 = m->dim_c3;
    int in_dim = m->num_inputs;
    int dh = m->dim_hid;

    if (version == 3) {
        int c_in = c1 + c2 + c3 + in_dim;
        fread(m->W_waveguide, sizeof(float), c2 * c1, fp);
        fread(m->W_waveguide2, sizeof(float), c3 * c2, fp);
        fread(m->W_out1, sizeof(float), v * c1, fp);
        fread(m->W_out2, sizeof(float), v * c2, fp);
        fread(m->W_out3, sizeof(float), v * c3, fp);
        fread(m->W_hid, sizeof(float), dh * c_in, fp);
        fread(m->b_hid, sizeof(float), dh, fp);
        fread(m->W_head, sizeof(float), v * dh, fp);
        fread(m->b_head, sizeof(float), v, fp);
        fread(m->chamber_char->W_charge, sizeof(float), c1 * in_dim, fp);
    } else {
        // Migration from version 2 dual-chamber checkpoint
        int c_in_v2 = c1 + c2 + in_dim;
        fread(m->W_waveguide, sizeof(float), c2 * c1, fp);
        fread(m->W_out1, sizeof(float), v * c1, fp);
        fread(m->W_out2, sizeof(float), v * c2, fp);
        // zero/copy out3
        memcpy(m->W_out3, m->W_out2, v * c3 * sizeof(float));
        fread(m->W_hid, sizeof(float), dh * c_in_v2, fp);
        fread(m->b_hid, sizeof(float), dh, fp);
        fread(m->W_head, sizeof(float), v * dh, fp);
        fread(m->b_head, sizeof(float), v, fp);
        fread(m->chamber_char->W_charge, sizeof(float), c1 * in_dim, fp);
    }

    fclose(fp);
    return 1;
}
