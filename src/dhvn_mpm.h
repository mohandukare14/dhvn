#ifndef DHVN_MPM_H
#define DHVN_MPM_H

#include "dhvn_core.h"
#include <stddef.h>
#include <immintrin.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DHVN_GRID_DIM 8  // 8x8x8x8 = 4096 grid nodes in 4D
#define DHVN_GRID_SIZE (DHVN_GRID_DIM * DHVN_GRID_DIM * DHVN_GRID_DIM * DHVN_GRID_DIM) // 4096
#define DHVN_MPM_THREADS 12

// 4D Cartesian Background Grid for Material Point Method
typedef struct {
    int dim;            // DHVN_GRID_DIM (8)
    int total_nodes;    // 4096
    float bound;        // Spatial half-extent [-bound, +bound]
    float inv_dx;       // 1.0f / dx
    
    // Aligned 32-byte arrays for AVX2 SIMD operations
    float* mass;        // total_nodes (4096)
    float* vel;         // total_nodes x 4 (16384 floats)
    float* charge;      // total_nodes x dim_c (4096 x dim_c)
    float* div;         // total_nodes (divergence)
    float* pressure;    // total_nodes (Helmholtz scalar potential)
    float* force;       // total_nodes x 4 (viscosity + pressure force)
} DHVN4DGrid;

// Tri-Chamber MPM Model Structure
typedef struct {
    // 3 Physical Chambers with 512 particles each
    DHVNSystem* chamber_char;   // Chamber 1: Acoustic Void (512 particles, dim_c1)
    DHVNSystem* chamber_sem;    // Chamber 2: Lexical Syntactic Void (512 particles, dim_c2)
    DHVNSystem* chamber_disc;   // Chamber 3: Propositional Discourse Void (512 particles, dim_c3)
    
    // Dedicated 4D MPM Grid per chamber for cache-resident O(N) solver
    DHVN4DGrid* grid1;
    DHVN4DGrid* grid2;
    DHVN4DGrid* grid3;
    
    int vocab_size;
    int dim_c1;
    int dim_c2;
    int dim_c3;
    int num_inputs;
    
    // Waveguide coupling: C1 -> C2, C2 -> C3
    float* W_waveguide;        // dim_c2 x dim_c1
    float* grad_W_waveguide;
    float* m_W_waveguide;
    float* v_W_waveguide;

    float* W_waveguide2;       // dim_c3 x dim_c2
    float* grad_W_waveguide2;
    float* m_W_waveguide2;
    float* v_W_waveguide2;
    
    // High-Capacity Non-Linear Resonant MLP Readout Head (AVX2/FMA Accelerated)
    int dim_hid;               // 256
    float* W_hid;              // dim_hid x (dim_c1 + dim_c2 + dim_c3 + num_inputs)
    float* b_hid;              // dim_hid
    float* grad_W_hid;
    float* grad_b_hid;
    float* m_W_hid;
    float* v_W_hid;
    float* m_b_hid;
    float* v_b_hid;

    float* W_head;             // vocab_size x dim_hid (4096 x 256)
    float* b_head;             // vocab_size (4096)
    float* grad_W_head;
    float* grad_b_head;
    float* m_W_head;
    float* v_W_head;
    float* m_b_head;
    float* v_b_head;

    // Linear residual bypass weights
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

    // Adam optimizer buffers for Chamber 1 Influx
    float* grad_W_charge1;
    float* m_W_charge1;
    float* v_W_charge1;

    int t_step;
    int num_substeps;
} DeepDHVN_MPM;

// Public MPM C API
DeepDHVN_MPM* dhvn_mpm_create_tri(
    int vocab_size,
    int np1, int dim_c1,
    int np2, int dim_c2,
    int np3, int dim_c3,
    int dim_hid
);

void dhvn_mpm_free(DeepDHVN_MPM* model);
void dhvn_mpm_set_substeps(DeepDHVN_MPM* model, int substeps);

// 4D MPM Fluid Solver Steps (P2G -> Grid Navier-Stokes -> G2P)
void dhvn_mpm_solve_chamber(DHVNSystem* sys, DHVN4DGrid* grid, float dt, float viscosity);

// Forward Inference & Training Step (with AVX2/FMA 12-thread vectorization)
void dhvn_mpm_forward_token(
    DeepDHVN_MPM* model,
    int token_id,
    float* out_probs,
    float* out_hidden,
    float* out_input_features,
    float dt
);

float dhvn_mpm_train_step(
    DeepDHVN_MPM* model,
    int in_token,
    int target_token,
    float lr,
    float dt
);

// High-Speed Multi-Token Sequence Chunk Training
float dhvn_mpm_train_sequence(
    DeepDHVN_MPM* model,
    const int* tokens,
    int seq_len,
    float lr,
    float dt
);

// Option 1: Topological Memory Crystallization ("Fluid Sleep")
void dhvn_mpm_crystallize(DeepDHVN_MPM* model, int sleep_cycles, float cooling_rate);

// Option 2: Continuous Analog Audio Waveform Influx
void dhvn_mpm_inject_waveform(DeepDHVN_MPM* model, const float* audio_pcm, int pcm_len, float dt);

// Weight Serialization (Exact binary compatibility with standard checkpoints)
int dhvn_mpm_save_weights(DeepDHVN_MPM* model, const char* filepath);
int dhvn_mpm_load_weights(DeepDHVN_MPM* model, const char* filepath);

#ifdef __cplusplus
}
#endif

#endif // DHVN_MPM_H

