#ifndef DHVN_DEEP_H
#define DHVN_DEEP_H

#include "dhvn_core.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    // 3-Stage Neocortical Void Chambers
    DHVNSystem* chamber_char;   // Stage 1: Fast acoustic void (micro-eddies, phonotactics)
    DHVNSystem* chamber_sem;    // Stage 2: Lexical syntactic void (phrasing, grammar)
    DHVNSystem* chamber_disc;   // Stage 3: Propositional discourse void (conversational intent, topic)
    
    int vocab_size;
    int dim_c1;
    int dim_c2;
    int dim_c3;
    int num_inputs;
    
    // Waveguide coupling: Stage 1 -> Stage 2, Stage 2 -> Stage 3
    float* W_waveguide;        // dim_c2 x dim_c1 (wg1)
    float* grad_W_waveguide;
    float* m_W_waveguide;
    float* v_W_waveguide;

    float* W_waveguide2;       // dim_c3 x dim_c2 (wg2)
    float* grad_W_waveguide2;
    float* m_W_waveguide2;
    float* v_W_waveguide2;
    
    // Hierarchical Readout Head (Linear residual paths from all 3 chambers)
    float* W_out1;             // vocab_size x dim_c1
    float* W_out2;             // vocab_size x dim_c2
    float* W_out3;             // vocab_size x dim_c3
    float* grad_W_out1;
    float* grad_W_out2;
    float* grad_W_out3;
    float* m_W_out1;
    float* v_W_out1;
    float* m_W_out2;
    float* v_W_out2;
    float* m_W_out3;
    float* v_W_out3;

    // High-Capacity Non-Linear Resonant MLP Readout Head
    int dim_hid;               // Hidden dimension (e.g. 256)
    float* W_hid;              // dim_hid x (dim_c1 + dim_c2 + dim_c3 + num_inputs)
    float* b_hid;              // dim_hid
    float* grad_W_hid;
    float* grad_b_hid;
    float* m_W_hid;
    float* v_W_hid;
    float* m_b_hid;
    float* v_b_hid;

    float* W_head;             // vocab_size x dim_hid
    float* b_head;             // vocab_size
    float* grad_W_head;
    float* grad_b_head;
    float* m_W_head;
    float* v_W_head;
    float* m_b_head;
    float* v_b_head;

    // Unified Adam buffers for Chamber 1 Influx
    float* grad_W_charge1;
    float* m_W_charge1;
    float* v_W_charge1;

    int t_step;
    int num_substeps;
} DeepDHVN;

// Tri-chamber creation with custom dimensions
DeepDHVN* deep_dhvn_create_tri(
    int vocab_size,
    int np1, int dim_c1,
    int np2, int dim_c2,
    int np3, int dim_c3,
    int dim_hid
);

// Backward-compatible dual-chamber wrapper
DeepDHVN* deep_dhvn_create(int vocab_size, int np1, int dim_c1, int np2, int dim_c2);

void deep_dhvn_free(DeepDHVN* model);
void deep_dhvn_set_substeps(DeepDHVN* model, int substeps);

// Multi-Step BPTT sequence training with 3-chamber temporal adjoint momentum
float deep_dhvn_train_tokens(
    DeepDHVN* model,
    const int* in_tokens,
    const int* target_tokens,
    int length,
    float lr,
    float dt
);

float deep_dhvn_train_sequence(
    DeepDHVN* model,
    const unsigned char* in_bytes,
    const unsigned char* target_bytes,
    int length,
    float lr,
    float dt
);

// Forward distribution prediction with tri-chamber hierarchical readout
void deep_dhvn_predict_token_distribution(
    DeepDHVN* model,
    int in_token,
    float dt,
    float* out_probs
);

void deep_dhvn_predict_distribution(
    DeepDHVN* model,
    unsigned char in_byte,
    float dt,
    float* out_probs
);

// Active Inference Reflex Loop: Counterfactual pondering across Chamber 2 and Chamber 3
void deep_dhvn_active_inference_step(
    DeepDHVN* model,
    int in_token,
    int ponder_steps,
    float reflex_strength,
    float dt,
    float* out_probs,
    float* out_free_energy
);

// Option 1: Topological Memory Crystallization ("Fluid Sleep" Routine)
// Relaxes active turbulence into permanent vortex crystals without gradient drift
void deep_dhvn_crystallize_memory(
    DeepDHVN* model,
    int sleep_cycles,
    float cooling_rate
);

// Option 2: Continuous Analog Audio Waveform Ingestion
// Windowed Fourier decomposition injected directly into Chamber 1 particles
void deep_dhvn_inject_waveform(
    DeepDHVN* model,
    const float* audio_pcm,
    int num_samples,
    float dt
);

size_t deep_dhvn_get_memory_footprint(DeepDHVN* model);

int deep_dhvn_save_weights(DeepDHVN* model, const char* filepath);
int deep_dhvn_load_weights(DeepDHVN* model, const char* filepath);

#ifdef __cplusplus
}
#endif

#endif // DHVN_DEEP_H
