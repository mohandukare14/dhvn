#include "dhvn_core.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 4D Quintic Spline normalization: 1 / (64 * pi^2)
static const float ALPHA_4 = 1.0f / (64.0f * (float)(M_PI * M_PI));

static inline float kernel_4d(float r, float h) {
    float q = r / h;
    if (q >= 3.0f) return 0.0f;
    float factor = ALPHA_4 / (h * h * h * h);
    float val;
    if (q < 1.0f) {
        float t3 = 3.0f - q;
        float t2 = 2.0f - q;
        float t1 = 1.0f - q;
        val = (t3*t3*t3*t3*t3) - 6.0f * (t2*t2*t2*t2*t2) + 15.0f * (t1*t1*t1*t1*t1);
    } else if (q < 2.0f) {
        float t3 = 3.0f - q;
        float t2 = 2.0f - q;
        val = (t3*t3*t3*t3*t3) - 6.0f * (t2*t2*t2*t2*t2);
    } else {
        float t3 = 3.0f - q;
        val = (t3*t3*t3*t3*t3);
    }
    return factor * val;
}

static inline void grad_kernel_4d(const float* xi, const float* xj, float h, float* grad, float* out_r) {
    float dx0 = xi[0] - xj[0];
    float dx1 = xi[1] - xj[1];
    float dx2 = xi[2] - xj[2];
    float dx3 = xi[3] - xj[3];
    float r = sqrtf(dx0*dx0 + dx1*dx1 + dx2*dx2 + dx3*dx3);
    *out_r = r;
    if (r < 1e-6f || r >= 3.0f * h) {
        grad[0] = 0.0f; grad[1] = 0.0f; grad[2] = 0.0f; grad[3] = 0.0f;
        return;
    }
    float q = r / h;
    float factor = ALPHA_4 / (h * h * h * h * h);
    float dq;
    if (q < 1.0f) {
        float t3 = 3.0f - q;
        float t2 = 2.0f - q;
        float t1 = 1.0f - q;
        dq = -5.0f*(t3*t3*t3*t3) + 30.0f*(t2*t2*t2*t2) - 75.0f*(t1*t1*t1*t1);
    } else if (q < 2.0f) {
        float t3 = 3.0f - q;
        float t2 = 2.0f - q;
        dq = -5.0f*(t3*t3*t3*t3) + 30.0f*(t2*t2*t2*t2);
    } else {
        float t3 = 3.0f - q;
        dq = -5.0f*(t3*t3*t3*t3);
    }
    float dW_dr = factor * dq;
    float inv_r = 1.0f / r;
    grad[0] = dW_dr * dx0 * inv_r;
    grad[1] = dW_dr * dx1 * inv_r;
    grad[2] = dW_dr * dx2 * inv_r;
    grad[3] = dW_dr * dx3 * inv_r;
}

DHVNSystem* dhvn_create(int num_particles, int dim_c, float radius, float h) {
    DHVNSystem* sys = (DHVNSystem*)malloc(sizeof(DHVNSystem));
    sys->num_particles = num_particles;
    sys->dim_c = dim_c;
    sys->radius = radius;
    sys->h = h;
    sys->viscosity = 0.08f;
    sys->stiffness_B = 1.5f;
    sys->boundary_k = 25.0f;
    sys->critical_threshold = 0.04f;
    sys->recoil_strength = 0.8f;
    sys->cognitive_decay = 0.99995f; // Bounded Kelvin circulation conservation

    sys->port[0] = radius * 0.8f;
    sys->port[1] = 0.0f;
    sys->port[2] = 0.0f;
    sys->port[3] = 0.0f;

    int num_inputs = 2 * DHVN_MAX_HARMONICS;
    sys->W_force = (float*)malloc(4 * num_inputs * sizeof(float));
    sys->W_charge = (float*)malloc(dim_c * num_inputs * sizeof(float));
    sys->W_gate = (float*)malloc(num_inputs * sizeof(float));
    sys->gate_bias = 0.5f;

    // Initialize deterministic weights
    srand(1337);
    for (int i = 0; i < 4 * num_inputs; ++i) {
        sys->W_force[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 1.2f;
    }
    for (int i = 0; i < dim_c * num_inputs; ++i) {
        sys->W_charge[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 1.2f;
    }
    for (int i = 0; i < num_inputs; ++i) {
        sys->W_gate[i] = 0.0f; // Default neutral gate
    }

    sys->particles = (DHVNParticle*)malloc(num_particles * sizeof(DHVNParticle));
    sys->charge_pool = (float*)calloc(num_particles * dim_c, sizeof(float));

    // Initialize particles uniformly in 4D ball
    for (int i = 0; i < num_particles; ++i) {
        DHVNParticle* p = &sys->particles[i];
        p->charge = &sys->charge_pool[i * dim_c];
        
        float u[4];
        float norm_sq = 0.0f;
        for (int k = 0; k < 4; ++k) {
            u[k] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f;
            norm_sq += u[k] * u[k];
        }
        float norm = sqrtf(norm_sq) + 1e-6f;
        float r = powf((float)rand() / (float)RAND_MAX, 0.25f) * (radius * 0.85f);
        for (int k = 0; k < 4; ++k) {
            p->pos[k] = (u[k] / norm) * r;
            p->vel[k] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.04f;
            p->acc[k] = 0.0f;
        }
        for (int c = 0; c < dim_c; ++c) {
            p->charge[c] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.02f;
        }
        p->density = 1.0f;
        p->pressure = 0.0f;
        p->mass = 1.0f;
        p->enstrophy = 0.0f;
    }

    // Setup 4D Spatial Hash Cell Grid for O(N) neighbor lookup
    float R_bound = radius * 1.2f;
    float L = 2.0f * R_bound;
    float s_cell = 3.0f * h;
    int K = (int)ceilf(L / s_cell);
    if (K < 3) K = 3;
    if (K > 8) K = 8;
    sys->grid_dim = K;
    sys->grid_cell_size = L / (float)K;
    sys->grid_total_cells = K * K * K * K;
    sys->grid_head = (int*)malloc(sys->grid_total_cells * sizeof(int));
    sys->grid_next = (int*)malloc(num_particles * sizeof(int));

    return sys;
}

void dhvn_free(DHVNSystem* sys) {
    if (!sys) return;
    free(sys->W_force);
    free(sys->W_charge);
    free(sys->W_gate);
    free(sys->charge_pool);
    free(sys->particles);
    free(sys->grid_head);
    free(sys->grid_next);
    free(sys);
}

float dhvn_compute_token_gate(DHVNSystem* sys, int token_id, int vocab_size) {
    if (vocab_size <= 0) vocab_size = 256;
    float theta = 2.0f * (float)M_PI * ((float)(token_id % vocab_size) / (float)vocab_size);
    float phi[2 * DHVN_MAX_HARMONICS];
    dhvn_compute_fourier_packet(theta, phi);
    float logit = sys->gate_bias;
    for (int j = 0; j < 2 * DHVN_MAX_HARMONICS; ++j) {
        logit += sys->W_gate[j] * phi[j];
    }
    // Sigmoid acoustic permeability: g in (0, 1)
    return 1.0f / (1.0f + expf(-logit));
}

float dhvn_compute_gate(DHVNSystem* sys, unsigned char byte_val) {
    return dhvn_compute_token_gate(sys, (int)byte_val, 256);
}

void dhvn_inject_token(DHVNSystem* sys, int token_id, int vocab_size, float strength) {
    if (vocab_size <= 0) vocab_size = 256;
    float gate = dhvn_compute_token_gate(sys, token_id, vocab_size);
    // Acoustic reflection: if gate is closed, reflect wave without perturbing fluid
    if (gate < 0.05f) return;

    float theta = 2.0f * (float)M_PI * ((float)(token_id % vocab_size) / (float)vocab_size);
    float phi[2 * DHVN_MAX_HARMONICS];
    dhvn_compute_fourier_packet(theta, phi);

    // Dynamic 4D Boundary Port: Distribute influx across 4D hypersphere surface based on frequency
    float port_pos[4];
    float r_port = sys->radius * 0.85f;
    port_pos[0] = r_port * cosf(theta);
    port_pos[1] = r_port * sinf(theta);
    port_pos[2] = r_port * cosf(2.0f * theta);
    port_pos[3] = r_port * sinf(2.0f * theta);

    int num_inputs = 2 * DHVN_MAX_HARMONICS;
    float force[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int d = 0; d < 4; ++d) {
        for (int j = 0; j < num_inputs; ++j) {
            force[d] += sys->W_force[d * num_inputs + j] * phi[j];
        }
    }

    float* d_charge = (float*)alloca(sys->dim_c * sizeof(float));
    for (int c = 0; c < sys->dim_c; ++c) {
        d_charge[c] = 0.0f;
        for (int j = 0; j < num_inputs; ++j) {
            d_charge[c] += sys->W_charge[c * num_inputs + j] * phi[j];
        }
    }

    float h_port = sys->h * 1.2f;
    float gated_strength = strength * gate;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < sys->num_particles; ++i) {
        DHVNParticle* p = &sys->particles[i];
        float dx0 = p->pos[0] - port_pos[0];
        float dx1 = p->pos[1] - port_pos[1];
        float dx2 = p->pos[2] - port_pos[2];
        float dx3 = p->pos[3] - port_pos[3];
        float r = sqrtf(dx0*dx0 + dx1*dx1 + dx2*dx2 + dx3*dx3);
        float w = kernel_4d(r, h_port);
        if (w > 0.0f) {
            // Topological Soliton Shielding:
            // High-enstrophy vortex cores are shielded from external boundary wash
            float shield = 1.0f / (1.0f + p->enstrophy * 5.0f);
            float eff = gated_strength * w * shield;

            p->vel[0] += eff * force[0] * 0.1f;
            p->vel[1] += eff * force[1] * 0.1f;
            p->vel[2] += eff * force[2] * 0.1f;
            p->vel[3] += eff * force[3] * 0.1f;
            for (int c = 0; c < sys->dim_c; ++c) {
                p->charge[c] += eff * d_charge[c] * 0.2f;
            }
        }
    }
}

void dhvn_inject_byte(DHVNSystem* sys, unsigned char byte_val, float strength) {
    dhvn_inject_token(sys, (int)byte_val, 256, strength);
}

static inline int get_cell_coord(float x, float R_bound, float inv_s, int K) {
    int c = (int)((x + R_bound) * inv_s);
    if (c < 0) c = 0;
    if (c >= K) c = K - 1;
    return c;
}

static inline void dhvn_build_spatial_grid(DHVNSystem* sys) {
    int total_cells = sys->grid_total_cells;
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < total_cells; ++c) {
        sys->grid_head[c] = -1;
    }

    float R_bound = sys->radius * 1.2f;
    float inv_s = 1.0f / sys->grid_cell_size;
    int K = sys->grid_dim;

    for (int i = 0; i < sys->num_particles; ++i) {
        float* p = sys->particles[i].pos;
        int c0 = get_cell_coord(p[0], R_bound, inv_s, K);
        int c1 = get_cell_coord(p[1], R_bound, inv_s, K);
        int c2 = get_cell_coord(p[2], R_bound, inv_s, K);
        int c3 = get_cell_coord(p[3], R_bound, inv_s, K);
        int cell = ((c0 * K + c1) * K + c2) * K + c3;

        sys->grid_next[i] = sys->grid_head[cell];
        sys->grid_head[cell] = i;
    }
}

static inline int dhvn_get_neighbor_cells(
    const float* pos,
    float R_bound,
    float inv_s,
    int K,
    int* out_neighbor_cells
) {
    int c0 = get_cell_coord(pos[0], R_bound, inv_s, K);
    int c1 = get_cell_coord(pos[1], R_bound, inv_s, K);
    int c2 = get_cell_coord(pos[2], R_bound, inv_s, K);
    int c3 = get_cell_coord(pos[3], R_bound, inv_s, K);

    int count = 0;
    for (int d0 = -1; d0 <= 1; ++d0) {
        int n0 = c0 + d0; if (n0 < 0 || n0 >= K) continue;
        for (int d1 = -1; d1 <= 1; ++d1) {
            int n1 = c1 + d1; if (n1 < 0 || n1 >= K) continue;
            for (int d2 = -1; d2 <= 1; ++d2) {
                int n2 = c2 + d2; if (n2 < 0 || n2 >= K) continue;
                for (int d3 = -1; d3 <= 1; ++d3) {
                    int n3 = c3 + d3; if (n3 < 0 || n3 >= K) continue;
                    out_neighbor_cells[count++] = ((n0 * K + n1) * K + n2) * K + n3;
                }
            }
        }
    }
    return count;
}

void dhvn_step(DHVNSystem* sys, float dt) {
    int n = sys->num_particles;
    float h = sys->h;
    float B = sys->stiffness_B;
    float visc = sys->viscosity;
    float bk = sys->boundary_k;
    float rad_limit = sys->radius * 0.9f;

    // Build 4D Spatial Hash Cell Grid in O(N) linear time
    dhvn_build_spatial_grid(sys);

    float r_cut_sq = 9.0f * h * h;
    float R_bound = sys->radius * 1.2f;
    float inv_s = 1.0f / sys->grid_cell_size;
    int K = sys->grid_dim;

    // 1. Density & Pressure estimation via 4D Spatial Hash Grid
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        DHVNParticle* pi = &sys->particles[i];
        float rho = pi->mass * kernel_4d(0.0f, h);

        int nbr_cells[81];
        int n_nbr_cells = dhvn_get_neighbor_cells(pi->pos, R_bound, inv_s, K, nbr_cells);

        for (int nc = 0; nc < n_nbr_cells; ++nc) {
            int j = sys->grid_head[nbr_cells[nc]];
            while (j != -1) {
                if (i != j) {
                    float dx0 = pi->pos[0] - sys->particles[j].pos[0];
                    float dx1 = pi->pos[1] - sys->particles[j].pos[1];
                    float dx2 = pi->pos[2] - sys->particles[j].pos[2];
                    float dx3 = pi->pos[3] - sys->particles[j].pos[3];
                    float r_sq = dx0*dx0 + dx1*dx1 + dx2*dx2 + dx3*dx3;
                    if (r_sq < r_cut_sq) {
                        float r = sqrtf(r_sq);
                        rho += sys->particles[j].mass * kernel_4d(r, h);
                    }
                }
                j = sys->grid_next[j];
            }
        }
        pi->density = rho < 0.1f ? 0.1f : rho;
        float ratio = pi->density / 1.0f; // rho_0 = 1.0
        float p = B * (powf(ratio, 7.0f) - 1.0f);
        pi->pressure = p < 0.0f ? 0.0f : p;
    }

    // Temporary buffer for semantic charge derivatives
    float* dc_dt = (float*)calloc(n * sys->dim_c, sizeof(float));

    // 2. Accelerations & Charge transfer via 4D Spatial Hash Grid
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        DHVNParticle* pi = &sys->particles[i];
        float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        int nbr_cells[81];
        int n_nbr_cells = dhvn_get_neighbor_cells(pi->pos, R_bound, inv_s, K, nbr_cells);

        for (int nc = 0; nc < n_nbr_cells; ++nc) {
            int j = sys->grid_head[nbr_cells[nc]];
            while (j != -1) {
                if (i != j) {
                    DHVNParticle* pj = &sys->particles[j];
                    float dx0 = pi->pos[0] - pj->pos[0];
                    float dx1 = pi->pos[1] - pj->pos[1];
                    float dx2 = pi->pos[2] - pj->pos[2];
                    float dx3 = pi->pos[3] - pj->pos[3];
                    float r_sq = dx0*dx0 + dx1*dx1 + dx2*dx2 + dx3*dx3;
                    if (r_sq < r_cut_sq && r_sq >= 1e-12f) {
                        float grad[4];
                        float r;
                        grad_kernel_4d(pi->pos, pj->pos, h, grad, &r);

                        // Pressure force
                        float p_term = (pi->pressure / (pi->density * pi->density)) + 
                                       (pj->pressure / (pj->density * pj->density));
                        float p_fac = pj->mass * p_term;
                        a[0] -= p_fac * grad[0];
                        a[1] -= p_fac * grad[1];
                        a[2] -= p_fac * grad[2];
                        a[3] -= p_fac * grad[3];

                        // Viscous shear with Radial Stratification
                        float r_i = sqrtf(pi->pos[0]*pi->pos[0] + pi->pos[1]*pi->pos[1] + 
                                          pi->pos[2]*pi->pos[2] + pi->pos[3]*pi->pos[3]);
                        float depth_ratio = r_i / sys->radius;
                        float effective_visc = visc * (0.01f + 2.0f * depth_ratio * depth_ratio);

                        float dot_x_grad = dx0*grad[0] + dx1*grad[1] + dx2*grad[2] + dx3*grad[3];
                        float denom = (r * r) + 0.01f * (h * h);
                        float v_fac = effective_visc * (pj->mass / pj->density) * (dot_x_grad / denom);
                        a[0] += v_fac * (pj->vel[0] - pi->vel[0]);
                        a[1] += v_fac * (pj->vel[1] - pi->vel[1]);
                        a[2] += v_fac * (pj->vel[2] - pi->vel[2]);
                        a[3] += v_fac * (pj->vel[3] - pi->vel[3]);

                        // Gauge Semantic coupling
                        float charge_dot = 0.0f;
                        for (int c = 0; c < sys->dim_c; ++c) {
                            charge_dot += pi->charge[c] * pj->charge[c];
                        }
                        float g_fac = (pj->mass / pj->density) * tanhf(charge_dot) * 0.2f;
                        a[0] += g_fac * grad[0];
                        a[1] += g_fac * grad[1];
                        a[2] += g_fac * grad[2];
                        a[3] += g_fac * grad[3];

                        // Charge transfer with Hebbian Vorticity Strain Plasticity:
                        // Co-circulating particles in the same vortex eddy strengthen semantic coupling
                        float vel_dot = pi->vel[0]*pj->vel[0] + pi->vel[1]*pj->vel[1] + 
                                        pi->vel[2]*pj->vel[2] + pi->vel[3]*pj->vel[3];
                        float vi_sq = pi->vel[0]*pi->vel[0] + pi->vel[1]*pi->vel[1] + 
                                      pi->vel[2]*pi->vel[2] + pi->vel[3]*pi->vel[3];
                        float vj_sq = pj->vel[0]*pj->vel[0] + pj->vel[1]*pj->vel[1] + 
                                      pj->vel[2]*pj->vel[2] + pj->vel[3]*pj->vel[3];
                        float v_norm = sqrtf(vi_sq * vj_sq) + 1e-6f;
                        float cos_v = vel_dot / v_norm;
                        float hebb = (cos_v > 0.0f) ? (1.0f + 2.5f * cos_v) : 0.2f;

                        float w = kernel_4d(r, h);
                        float trans_fac = 0.02f * (pj->mass / pj->density) * w * hebb;
                        for (int c = 0; c < sys->dim_c; ++c) {
                            dc_dt[i * sys->dim_c + c] += trans_fac * (pj->charge[c] - pi->charge[c]);
                        }
                    }
                }
                j = sys->grid_next[j];
            }
        }

        // Boundary containment & Core sink for coherent vortices
        float dist_sq = pi->pos[0]*pi->pos[0] + pi->pos[1]*pi->pos[1] + 
                        pi->pos[2]*pi->pos[2] + pi->pos[3]*pi->pos[3];
        float dist = sqrtf(dist_sq);
        
        // Coherent vortex centripetal pull (solitons contract toward the protected core)
        if (pi->enstrophy > 0.1f && dist > 0.2f) {
            float sink_fac = -0.5f * pi->enstrophy / (dist + 1e-6f);
            a[0] += sink_fac * pi->pos[0];
            a[1] += sink_fac * pi->pos[1];
            a[2] += sink_fac * pi->pos[2];
            a[3] += sink_fac * pi->pos[3];
        }

        if (dist > rad_limit) {
            float pen = dist - rad_limit;
            float b_fac = -bk * pen / (dist + 1e-6f);
            a[0] += b_fac * pi->pos[0];
            a[1] += b_fac * pi->pos[1];
            a[2] += b_fac * pi->pos[2];
            a[3] += b_fac * pi->pos[3];
        }

        pi->acc[0] = a[0];
        pi->acc[1] = a[1];
        pi->acc[2] = a[2];
        pi->acc[3] = a[3];
    }

    // 3. Velocity-Verlet Integration
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        DHVNParticle* p = &sys->particles[i];
        for (int d = 0; d < 4; ++d) {
            p->pos[d] += p->vel[d] * dt + 0.5f * p->acc[d] * (dt * dt);
            p->vel[d] = (p->vel[d] + p->acc[d] * dt) * 0.98f; // Kinetic stabilization drag
        }
        for (int c = 0; c < sys->dim_c; ++c) {
            p->charge[c] = (p->charge[c] + dc_dt[i * sys->dim_c + c] * dt) * sys->cognitive_decay;
        }
    }

    free(dc_dt);
}

int dhvn_process_stream(
    DHVNSystem* sys,
    const unsigned char* bytes,
    int length,
    float dt,
    int* out_emissions,
    int max_emissions
) {
    int emission_count = 0;

    for (int t = 0; t < length; ++t) {
        dhvn_inject_byte(sys, bytes[t], 1.5f);
        // Integrate 2 ticks per incoming byte
        dhvn_step(sys, dt);
        dhvn_step(sys, dt);

        // Compute 4D generalized enstrophy for readout
        int critical_idx = -1;
        float max_val = -1.0f;
        int n = sys->num_particles;
        float h = sys->h;

        for (int i = 0; i < n; ++i) {
            DHVNParticle* pi = &sys->particles[i];
            float Omega[4][4] = {{0}};
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                DHVNParticle* pj = &sys->particles[j];
                float grad[4];
                float r;
                grad_kernel_4d(pi->pos, pj->pos, h, grad, &r);
                if (r >= 3.0f * h || r < 1e-6f) continue;
                float mass_rho = pj->mass / pj->density;
                float v_diff[4] = {pj->vel[0]-pi->vel[0], pj->vel[1]-pi->vel[1],
                                   pj->vel[2]-pi->vel[2], pj->vel[3]-pi->vel[3]};
                for (int mu = 0; mu < 4; ++mu) {
                    for (int nu = mu + 1; nu < 4; ++nu) {
                        Omega[mu][nu] += 0.5f * mass_rho * (v_diff[mu]*grad[nu] - v_diff[nu]*grad[mu]);
                    }
                }
            }
            float enstrophy = 0.0f;
            for (int mu = 0; mu < 4; ++mu) {
                for (int nu = mu + 1; nu < 4; ++nu) {
                    enstrophy += Omega[mu][nu] * Omega[mu][nu];
                }
            }
            pi->enstrophy = enstrophy;
            float val = enstrophy * pi->density;
            if (val > sys->critical_threshold && val > max_val) {
                max_val = val;
                critical_idx = i;
            }
        }

        // Phase transition readout event
        if (critical_idx != -1 && emission_count < max_emissions) {
            DHVNParticle* p_star = &sys->particles[critical_idx];
            // Droplet pooling
            float pooled[32] = {0};
            float total_w = 0.0f;
            (void)total_w;
            for (int j = 0; j < n; ++j) {
                DHVNParticle* pj = &sys->particles[j];
                float dx0 = pj->pos[0] - p_star->pos[0];
                float dx1 = pj->pos[1] - p_star->pos[1];
                float dx2 = pj->pos[2] - p_star->pos[2];
                float dx3 = pj->pos[3] - p_star->pos[3];
                float r = sqrtf(dx0*dx0 + dx1*dx1 + dx2*dx2 + dx3*dx3);
                if (r < h) {
                    float w = pj->mass * pj->density;
                    total_w += w;
                    for (int c = 0; c < sys->dim_c && c < 32; ++c) {
                        pooled[c] += pj->charge[c] * w;
                    }
                    // Apply cavitation recoil
                    float inv_r = 1.0f / (r + 1e-4f);
                    pj->vel[0] += sys->recoil_strength * dx0 * inv_r;
                    pj->vel[1] += sys->recoil_strength * dx1 * inv_r;
                    pj->vel[2] += sys->recoil_strength * dx2 * inv_r;
                    pj->vel[3] += sys->recoil_strength * dx3 * inv_r;
                    pj->enstrophy = 0.0f;
                }
            }

            // Approximate character emission
            int emitted_byte = 65 + (abs((int)(pooled[0] * 1000.0f)) % 26);
            if (out_emissions) {
                out_emissions[emission_count] = emitted_byte;
            }
            emission_count++;
        }
    }

    return emission_count;
}

void dhvn_get_center_of_mass(DHVNSystem* sys, float* out_cm) {
    out_cm[0] = 0.0f; out_cm[1] = 0.0f; out_cm[2] = 0.0f; out_cm[3] = 0.0f;
    float total_mass = 0.0f;
    for (int i = 0; i < sys->num_particles; ++i) {
        DHVNParticle* p = &sys->particles[i];
        total_mass += p->mass;
        out_cm[0] += p->mass * p->pos[0];
        out_cm[1] += p->mass * p->pos[1];
        out_cm[2] += p->mass * p->pos[2];
        out_cm[3] += p->mass * p->pos[3];
    }
    if (total_mass > 0.0f) {
        out_cm[0] /= total_mass;
        out_cm[1] /= total_mass;
        out_cm[2] /= total_mass;
        out_cm[3] /= total_mass;
    }
}

float dhvn_get_kinetic_energy(DHVNSystem* sys) {
    float ke = 0.0f;
    for (int i = 0; i < sys->num_particles; ++i) {
        DHVNParticle* p = &sys->particles[i];
        float v_sq = p->vel[0]*p->vel[0] + p->vel[1]*p->vel[1] + 
                     p->vel[2]*p->vel[2] + p->vel[3]*p->vel[3];
        ke += 0.5f * p->mass * v_sq;
    }
    return ke;
}

size_t dhvn_get_memory_footprint(DHVNSystem* sys) {
    size_t sz = sizeof(DHVNSystem);
    sz += sys->num_particles * sizeof(DHVNParticle);
    sz += sys->num_particles * sys->dim_c * sizeof(float);
    sz += (4 + sys->dim_c) * (2 * DHVN_MAX_HARMONICS) * sizeof(float);
    sz += sys->grid_total_cells * sizeof(int);
    sz += sys->num_particles * sizeof(int);
    return sz;
}

void dhvn_get_mean_charge(DHVNSystem* sys, float* out_charge) {
    for (int c = 0; c < sys->dim_c; ++c) {
        out_charge[c] = 0.0f;
    }
    for (int i = 0; i < sys->num_particles; ++i) {
        DHVNParticle* p = &sys->particles[i];
        for (int c = 0; c < sys->dim_c; ++c) {
            out_charge[c] += p->charge[c];
        }
    }
    if (sys->num_particles > 0) {
        float inv = 1.0f / (float)sys->num_particles;
        for (int c = 0; c < sys->dim_c; ++c) {
            out_charge[c] *= inv;
        }
    }
}

float dhvn_compute_resonance(DHVNSystem* sys, const unsigned char* target_bytes, int target_len) {
    int num_inputs = 2 * DHVN_MAX_HARMONICS;
    float* expected_charge = (float*)calloc(sys->dim_c, sizeof(float));

    // Construct target composite semantic charge
    for (int t = 0; t < target_len; ++t) {
        float theta = 2.0f * (float)M_PI * ((float)target_bytes[t] / 256.0f);
        float phi[2 * DHVN_MAX_HARMONICS];
        dhvn_compute_fourier_packet(theta, phi);
        for (int c = 0; c < sys->dim_c; ++c) {
            for (int j = 0; j < num_inputs; ++j) {
                expected_charge[c] += sys->W_charge[c * num_inputs + j] * phi[j];
            }
        }
    }

    // Enstrophy-weighted / localized vortex core sampling
    // Compute local enstrophy for each particle
    float* fluid_charge = (float*)calloc(sys->dim_c, sizeof(float));
    float total_vort_weight = 0.0f;
    int n = sys->num_particles;
    float h = sys->h;

    for (int i = 0; i < n; ++i) {
        DHVNParticle* pi = &sys->particles[i];
        float Omega[4][4] = {{0}};
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            DHVNParticle* pj = &sys->particles[j];
            float grad[4];
            float r;
            grad_kernel_4d(pi->pos, pj->pos, h, grad, &r);
            if (r >= 3.0f * h || r < 1e-6f) continue;
            float mass_rho = pj->mass / pj->density;
            float v_diff[4] = {pj->vel[0]-pi->vel[0], pj->vel[1]-pi->vel[1],
                               pj->vel[2]-pi->vel[2], pj->vel[3]-pi->vel[3]};
            for (int mu = 0; mu < 4; ++mu) {
                for (int nu = mu + 1; nu < 4; ++nu) {
                    Omega[mu][nu] += 0.5f * mass_rho * (v_diff[mu]*grad[nu] - v_diff[nu]*grad[mu]);
                }
            }
        }
        float enstrophy = 0.0f;
        for (int mu = 0; mu < 4; ++mu) {
            for (int nu = mu + 1; nu < 4; ++nu) {
                enstrophy += Omega[mu][nu] * Omega[mu][nu];
            }
        }
        pi->enstrophy = enstrophy;

        // Weight particle by its coherent circulation and core depth (protected from surface noise)
        float r_i = sqrtf(pi->pos[0]*pi->pos[0] + pi->pos[1]*pi->pos[1] + 
                          pi->pos[2]*pi->pos[2] + pi->pos[3]*pi->pos[3]);
        float core_depth = 1.0f - (r_i / (sys->radius + 1e-6f));
        if (core_depth < 0.0f) core_depth = 0.0f;
        float w = (enstrophy * pi->density + 0.01f) * (core_depth * core_depth * core_depth);
        total_vort_weight += w;
        for (int c = 0; c < sys->dim_c; ++c) {
            fluid_charge[c] += pi->charge[c] * w;
        }
    }

    if (total_vort_weight > 0.0f) {
        float inv = 1.0f / total_vort_weight;
        for (int c = 0; c < sys->dim_c; ++c) {
            fluid_charge[c] *= inv;
        }
    }

    // Compute cosine similarity between fluid state and target resonance
    float dot = 0.0f;
    float norm_e = 0.0f;
    float norm_f = 0.0f;
    for (int c = 0; c < sys->dim_c; ++c) {
        dot += expected_charge[c] * fluid_charge[c];
        norm_e += expected_charge[c] * expected_charge[c];
        norm_f += fluid_charge[c] * fluid_charge[c];
    }

    free(expected_charge);
    free(fluid_charge);

    float denom = sqrtf(norm_e) * sqrtf(norm_f);
    if (denom < 1e-7f) return 0.0f;
    return dot / denom;
}

void dhvn_configure_selective_gate(DHVNSystem* sys, const unsigned char* salience_keys, int count) {
    // Trainable / Calibrated acoustic impedance filter:
    // Aligns gate weights with salient patterns while suppressing repetitive background noise
    int num_inputs = 2 * DHVN_MAX_HARMONICS;
    for (int j = 0; j < num_inputs; ++j) {
        sys->W_gate[j] = 0.0f;
    }
    sys->gate_bias = -1.2f; // Default barrier (reflects ambient noise)

    for (int i = 0; i < count; ++i) {
        float theta = 2.0f * (float)M_PI * ((float)salience_keys[i] / 256.0f);
        for (int k = 1; k <= DHVN_MAX_HARMONICS; ++k) {
            float c = cosf((float)k * theta);
            float s = sinf((float)k * theta);
            sys->W_gate[(k - 1) * 2]     += 1.5f * c;
            sys->W_gate[(k - 1) * 2 + 1] += 1.5f * s;
        }
    }
    // Normalize gate weights
    float norm = 0.0f;
    for (int j = 0; j < num_inputs; ++j) {
        norm += sys->W_gate[j] * sys->W_gate[j];
    }
    if (norm > 0.0f) {
        float inv = 2.5f / sqrtf(norm);
        for (int j = 0; j < num_inputs; ++j) {
            sys->W_gate[j] *= inv;
        }
    }
}

int dhvn_probe_retrieval(DHVNSystem* sys, const unsigned char* query_bytes, int query_len, unsigned char* out_decoded, int max_len) {
    // Inject query wavefront with full resonance power
    for (int t = 0; t < query_len; ++t) {
        dhvn_inject_byte(sys, query_bytes[t], 3.0f);
        dhvn_step(sys, 0.02f);
    }

    // Step fluid to allow constructive interference
    for (int s = 0; s < 5; ++s) {
        dhvn_step(sys, 0.02f);
    }

    // Detect critical vortex focus
    int critical_idx = -1;
    float max_enstrophy = 0.0f;
    for (int i = 0; i < sys->num_particles; ++i) {
        if (sys->particles[i].enstrophy > max_enstrophy) {
            max_enstrophy = sys->particles[i].enstrophy;
            critical_idx = i;
        }
    }

    if (critical_idx == -1) return 0;

    // Readout droplet pooled charge
    DHVNParticle* p_star = &sys->particles[critical_idx];
    float pooled[32] = {0};
    float total_w = 0.0f;
    for (int j = 0; j < sys->num_particles; ++j) {
        DHVNParticle* pj = &sys->particles[j];
        float dx0 = pj->pos[0] - p_star->pos[0];
        float dx1 = pj->pos[1] - p_star->pos[1];
        float dx2 = pj->pos[2] - p_star->pos[2];
        float dx3 = pj->pos[3] - p_star->pos[3];
        float r = sqrtf(dx0*dx0 + dx1*dx1 + dx2*dx2 + dx3*dx3);
        if (r < sys->h) {
            float w = pj->mass * pj->density;
            total_w += w;
            for (int c = 0; c < sys->dim_c && c < 32; ++c) {
                pooled[c] += pj->charge[c] * w;
            }
        }
    }

    if (total_w > 0.0f) {
        float inv = 1.0f / total_w;
        for (int c = 0; c < sys->dim_c && c < 32; ++c) {
            pooled[c] *= inv;
        }
    }

    // Decode top resonant ASCII bytes
    int decoded_count = 0;
    int num_inputs = 2 * DHVN_MAX_HARMONICS;

    for (int b = 32; b <= 126 && decoded_count < max_len; ++b) {
        float theta = 2.0f * (float)M_PI * ((float)b / 256.0f);
        float phi[2 * DHVN_MAX_HARMONICS];
        dhvn_compute_fourier_packet(theta, phi);
        float score = 0.0f;
        for (int c = 0; c < sys->dim_c; ++c) {
            float expected_c = 0.0f;
            for (int j = 0; j < num_inputs; ++j) {
                expected_c += sys->W_charge[c * num_inputs + j] * phi[j];
            }
            score += pooled[c] * expected_c;
        }
        if (score > 0.1f) {
            out_decoded[decoded_count++] = (unsigned char)b;
        }
    }

    return decoded_count;
}

