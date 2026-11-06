#ifndef DHVN_CORE_H
#define DHVN_CORE_H

#include <stddef.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DHVN_DIM_SPACE 4
#define DHVN_MAX_HARMONICS 16

static const float DHVN_FIBONACCI_HARMONICS[DHVN_MAX_HARMONICS] = {
    1.0f, 2.0f, 3.0f, 5.0f, 8.0f, 13.0f, 21.0f, 34.0f,
    55.0f, 89.0f, 144.0f, 233.0f, 377.0f, 610.0f, 987.0f, 1597.0f
};

static inline void dhvn_compute_fourier_packet(float theta, float* out_phi) {
    for (int k = 0; k < DHVN_MAX_HARMONICS; ++k) {
        float freq = DHVN_FIBONACCI_HARMONICS[k];
        out_phi[k * 2]     = cosf(freq * theta);
        out_phi[k * 2 + 1] = sinf(freq * theta);
    }
}

typedef struct {
    float pos[DHVN_DIM_SPACE];     // 4D coordinates
    float vel[DHVN_DIM_SPACE];     // 4D velocity
    float acc[DHVN_DIM_SPACE];     // 4D acceleration
    float* charge;                 // Semantic charge vector (dim_c)
    float density;
    float pressure;
    float mass;
    float enstrophy;
} DHVNParticle;

typedef struct {
    int num_particles;
    int dim_c;
    float radius;
    float h;
    float viscosity;
    float stiffness_B;
    float boundary_k;
    float critical_threshold;
    float recoil_strength;
    float cognitive_decay;         // Conservation factor for long-term memory (Kelvin circulation)
    
    // Boundary injection port
    float port[DHVN_DIM_SPACE];
    
    // Harmonic Fourier projections (dim_c x (2 * harmonics))
    float* W_force;   // 4 x (2 * harmonics)
    float* W_charge;  // dim_c x (2 * harmonics)
    float* W_gate;    // 1 x (2 * harmonics) for Acoustic Impedance Gating
    float gate_bias;
    
    DHVNParticle* particles;
    float* charge_pool; // Contiguous buffer for all particles

    // 4D Spatial Hash Cell Grid for O(N) Neighbor Search
    int grid_dim;         // Cells per 4D dimension K
    int grid_total_cells; // K^4
    float grid_cell_size; // Cell width (>= 3h)
    int* grid_head;       // Array of size grid_total_cells
    int* grid_next;       // Array of size num_particles
} DHVNSystem;

// Lifecycle
DHVNSystem* dhvn_create(int num_particles, int dim_c, float radius, float h);
void dhvn_free(DHVNSystem* sys);

// Execution
void dhvn_step(DHVNSystem* sys, float dt);
void dhvn_inject_byte(DHVNSystem* sys, unsigned char byte_val, float strength);
void dhvn_inject_token(DHVNSystem* sys, int token_id, int vocab_size, float strength);
int dhvn_process_stream(DHVNSystem* sys, const unsigned char* bytes, int length, float dt, int* out_emissions, int max_emissions);

// Telemetry
void dhvn_get_center_of_mass(DHVNSystem* sys, float* out_cm);
float dhvn_get_kinetic_energy(DHVNSystem* sys);
size_t dhvn_get_memory_footprint(DHVNSystem* sys);
void dhvn_get_mean_charge(DHVNSystem* sys, float* out_charge);
float dhvn_compute_resonance(DHVNSystem* sys, const unsigned char* target_bytes, int target_len);
float dhvn_compute_gate(DHVNSystem* sys, unsigned char byte_val);
float dhvn_compute_token_gate(DHVNSystem* sys, int token_id, int vocab_size);
void dhvn_configure_selective_gate(DHVNSystem* sys, const unsigned char* salience_keys, int count);
int dhvn_probe_retrieval(DHVNSystem* sys, const unsigned char* query_bytes, int query_len, unsigned char* out_decoded, int max_len);

#ifdef __cplusplus
}
#endif

#endif // DHVN_CORE_H
