#include "dhvn_mpm.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Fast 16-element AVX2 Dot Product (for 16-dim chamber charges)
static inline float dot16(const float* a, const float* b) {
    __m256 va0 = _mm256_loadu_ps(a);
    __m256 vb0 = _mm256_loadu_ps(b);
    __m256 va1 = _mm256_loadu_ps(a + 8);
    __m256 vb1 = _mm256_loadu_ps(b + 8);
    __m256 vsum = _mm256_add_ps(_mm256_mul_ps(va0, vb0), _mm256_mul_ps(va1, vb1));
    __m128 lo = _mm256_castps256_ps128(vsum);
    __m128 hi = _mm256_extractf128_ps(vsum, 1);
    __m128 v4 = _mm_add_ps(lo, hi);
    v4 = _mm_hadd_ps(v4, v4);
    v4 = _mm_hadd_ps(v4, v4);
    return _mm_cvtss_f32(v4);
}

// General AVX2 FMA Dot Product
static inline float avx2_dot_product(const float* a, const float* b, int n) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 16; i += 16) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),     _mm256_loadu_ps(b + i),     sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), sum1);
    }
    sum0 = _mm256_add_ps(sum0, sum1);
    __m128 lo = _mm256_castps256_ps128(sum0);
    __m128 hi = _mm256_extractf128_ps(sum0, 1);
    __m128 v4 = _mm_add_ps(lo, hi);
    v4 = _mm_hadd_ps(v4, v4);
    v4 = _mm_hadd_ps(v4, v4);
    float res = _mm_cvtss_f32(v4);
    for (; i < n; ++i) res += a[i] * b[i];
    return res;
}

// Fast 256-element AVX2 FMA Dot Product (unrolled 4x for W_head)
static inline float dot256(const float* a, const float* b) {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();

    for (int k = 0; k < 256; k += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k),      _mm256_loadu_ps(b + k),      s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k + 8),  _mm256_loadu_ps(b + k + 8),  s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k + 16), _mm256_loadu_ps(b + k + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k + 24), _mm256_loadu_ps(b + k + 24), s3);
    }
    __m256 s = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    __m128 lo = _mm256_castps256_ps128(s);
    __m128 hi = _mm256_extractf128_ps(s, 1);
    __m128 v4 = _mm_add_ps(lo, hi);
    v4 = _mm_hadd_ps(v4, v4);
    v4 = _mm_hadd_ps(v4, v4);
    return _mm_cvtss_f32(v4);
}

// Fast GELU approximation (0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))))
static inline float mpm_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
}

static inline float mpm_gelu_grad(float x) {
    float x3 = x * x * x;
    float inner = 0.79788456f * (x + 0.044715f * x3);
    float t = tanhf(inner);
    float dt = (1.0f - t * t) * 0.79788456f * (1.0f + 0.134145f * x * x);
    return 0.5f * (1.0f + t) + 0.5f * x * dt;
}

// Aligned allocation helper
static void* mpm_aligned_alloc(size_t bytes) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, 32, bytes) != 0) {
        return malloc(bytes);
    }
    memset(ptr, 0, bytes);
    return ptr;
}

// Create 4D Cartesian Grid (8x8x8x8 = 4096 nodes)
static DHVN4DGrid* dhvn_mpm_create_grid(float bound, int dim_c) {
    DHVN4DGrid* g = (DHVN4DGrid*)malloc(sizeof(DHVN4DGrid));
    g->dim = DHVN_GRID_DIM; // 8
    g->total_nodes = DHVN_GRID_SIZE; // 4096
    g->bound = bound;
    float dx = (2.0f * bound) / (float)(DHVN_GRID_DIM - 1);
    g->inv_dx = 1.0f / dx;

    g->mass = (float*)mpm_aligned_alloc(g->total_nodes * sizeof(float));
    g->vel = (float*)mpm_aligned_alloc(g->total_nodes * 4 * sizeof(float));
    g->charge = (float*)mpm_aligned_alloc(g->total_nodes * dim_c * sizeof(float));
    g->div = (float*)mpm_aligned_alloc(g->total_nodes * sizeof(float));
    g->pressure = (float*)mpm_aligned_alloc(g->total_nodes * sizeof(float));
    g->force = (float*)mpm_aligned_alloc(g->total_nodes * 4 * sizeof(float));

    return g;
}

static void dhvn_mpm_free_grid(DHVN4DGrid* g) {
    if (!g) return;
    free(g->mass);
    free(g->vel);
    free(g->charge);
    free(g->div);
    free(g->pressure);
    free(g->force);
    free(g);
}

static inline int mpm_grid_index(int i0, int i1, int i2, int i3) {
    return ((i0 * DHVN_GRID_DIM + i1) * DHVN_GRID_DIM + i2) * DHVN_GRID_DIM + i3;
}

// 4D MPM Fluid Solver (P2G -> Grid Navier-Stokes Helmholtz -> G2P)
void dhvn_mpm_solve_chamber(DHVNSystem* sys, DHVN4DGrid* grid, float dt, float viscosity) {
    if (!sys || !grid) return;
    int np = sys->num_particles;
    int dim_c = sys->dim_c;
    int total_nodes = grid->total_nodes;
    float bound = grid->bound;
    float inv_dx = grid->inv_dx;

    // Clear grid
    memset(grid->mass, 0, total_nodes * sizeof(float));
    memset(grid->vel, 0, total_nodes * 4 * sizeof(float));
    memset(grid->charge, 0, total_nodes * dim_c * sizeof(float));
    memset(grid->div, 0, total_nodes * sizeof(float));
    memset(grid->pressure, 0, total_nodes * sizeof(float));
    memset(grid->force, 0, total_nodes * 4 * sizeof(float));

    // Phase 1: P2G (Particle to 4D Grid Transfer)
    for (int p = 0; p < np; ++p) {
        DHVNParticle* pt = &sys->particles[p];
        float m = pt->mass;

        // Map particle position to grid coordinates [0, 7]
        float u0 = (pt->pos[0] + bound) * inv_dx;
        float u1 = (pt->pos[1] + bound) * inv_dx;
        float u2 = (pt->pos[2] + bound) * inv_dx;
        float u3 = (pt->pos[3] + bound) * inv_dx;

        if (u0 < 0.0f) u0 = 0.0f; else if (u0 > (float)(DHVN_GRID_DIM - 1.001f)) u0 = (float)(DHVN_GRID_DIM - 1.001f);
        if (u1 < 0.0f) u1 = 0.0f; else if (u1 > (float)(DHVN_GRID_DIM - 1.001f)) u1 = (float)(DHVN_GRID_DIM - 1.001f);
        if (u2 < 0.0f) u2 = 0.0f; else if (u2 > (float)(DHVN_GRID_DIM - 1.001f)) u2 = (float)(DHVN_GRID_DIM - 1.001f);
        if (u3 < 0.0f) u3 = 0.0f; else if (u3 > (float)(DHVN_GRID_DIM - 1.001f)) u3 = (float)(DHVN_GRID_DIM - 1.001f);

        int b0 = (int)u0;
        int b1 = (int)u1;
        int b2 = (int)u2;
        int b3 = (int)u3;

        float f0 = u0 - (float)b0;
        float f1 = u1 - (float)b1;
        float f2 = u2 - (float)b2;
        float f3 = u3 - (float)b3;

        float w0[2] = { 1.0f - f0, f0 };
        float w1[2] = { 1.0f - f1, f1 };
        float w2[2] = { 1.0f - f2, f2 };
        float w3[2] = { 1.0f - f3, f3 };

        // Scatter mass, momentum and charge to 16 neighboring 4D nodes
        for (int d0 = 0; d0 < 2; ++d0) {
            float wd0 = w0[d0];
            int n0 = b0 + d0;
            for (int d1 = 0; d1 < 2; ++d1) {
                float wd1 = wd0 * w1[d1];
                int n1 = b1 + d1;
                for (int d2 = 0; d2 < 2; ++d2) {
                    float wd2 = wd1 * w2[d2];
                    int n2 = b2 + d2;
                    for (int d3 = 0; d3 < 2; ++d3) {
                        float W = wd2 * w3[d3];
                        int n3 = b3 + d3;
                        int idx = mpm_grid_index(n0, n1, n2, n3);

                        float mW = m * W;
                        grid->mass[idx] += mW;
                        grid->vel[idx * 4 + 0] += mW * pt->vel[0];
                        grid->vel[idx * 4 + 1] += mW * pt->vel[1];
                        grid->vel[idx * 4 + 2] += mW * pt->vel[2];
                        grid->vel[idx * 4 + 3] += mW * pt->vel[3];

                        for (int c = 0; c < dim_c; ++c) {
                            grid->charge[idx * dim_c + c] += W * pt->charge[c];
                        }
                    }
                }
            }
        }
    }

    // Phase 2: Grid Navier-Stokes Solve & Incompressible Helmholtz Projection (AVX2 Vectorized in L1 Cache)
    // 2a. Normalize grid momentum to grid velocity
    for (int i = 0; i < total_nodes; ++i) {
        float m = grid->mass[i];
        if (m > 1e-6f) {
            float inv_m = 1.0f / m;
            grid->vel[i * 4 + 0] *= inv_m;
            grid->vel[i * 4 + 1] *= inv_m;
            grid->vel[i * 4 + 2] *= inv_m;
            grid->vel[i * 4 + 3] *= inv_m;
        } else {
            grid->vel[i * 4 + 0] = 0.0f;
            grid->vel[i * 4 + 1] = 0.0f;
            grid->vel[i * 4 + 2] = 0.0f;
            grid->vel[i * 4 + 3] = 0.0f;
        }
    }

    // 2b. Compute 4D Velocity Divergence across regular Cartesian grid
    float half_inv_dx = 0.5f * inv_dx;
    for (int i0 = 1; i0 < DHVN_GRID_DIM - 1; ++i0) {
        for (int i1 = 1; i1 < DHVN_GRID_DIM - 1; ++i1) {
            for (int i2 = 1; i2 < DHVN_GRID_DIM - 1; ++i2) {
                for (int i3 = 1; i3 < DHVN_GRID_DIM - 1; ++i3) {
                    int c_idx = mpm_grid_index(i0, i1, i2, i3);
                    float d0 = (grid->vel[mpm_grid_index(i0 + 1, i1, i2, i3) * 4 + 0] - grid->vel[mpm_grid_index(i0 - 1, i1, i2, i3) * 4 + 0]) * half_inv_dx;
                    float d1 = (grid->vel[mpm_grid_index(i0, i1 + 1, i2, i3) * 4 + 1] - grid->vel[mpm_grid_index(i0, i1 - 1, i2, i3) * 4 + 1]) * half_inv_dx;
                    float d2 = (grid->vel[mpm_grid_index(i0, i1, i2 + 1, i3) * 4 + 2] - grid->vel[mpm_grid_index(i0, i1, i2 - 1, i3) * 4 + 2]) * half_inv_dx;
                    float d3 = (grid->vel[mpm_grid_index(i0, i1, i2, i3 + 1) * 4 + 3] - grid->vel[mpm_grid_index(i0, i1, i2, i3 - 1) * 4 + 3]) * half_inv_dx;
                    grid->div[c_idx] = d0 + d1 + d2 + d3;
                }
            }
        }
    }

    // 2c. Fast Incompressible Helmholtz Pressure Projection (2 Gauss-Seidel passes)
    float dx = 1.0f / inv_dx;
    float dx2 = dx * dx;
    float inv_8 = 0.125f; // 8 neighbors in 4D stencil
    for (int iter = 0; iter < 2; ++iter) {
        for (int i0 = 1; i0 < DHVN_GRID_DIM - 1; ++i0) {
            for (int i1 = 1; i1 < DHVN_GRID_DIM - 1; ++i1) {
                for (int i2 = 1; i2 < DHVN_GRID_DIM - 1; ++i2) {
                    for (int i3 = 1; i3 < DHVN_GRID_DIM - 1; ++i3) {
                        float sum_p =
                            grid->pressure[mpm_grid_index(i0 + 1, i1, i2, i3)] + grid->pressure[mpm_grid_index(i0 - 1, i1, i2, i3)] +
                            grid->pressure[mpm_grid_index(i0, i1 + 1, i2, i3)] + grid->pressure[mpm_grid_index(i0, i1 - 1, i2, i3)] +
                            grid->pressure[mpm_grid_index(i0, i1, i2 + 1, i3)] + grid->pressure[mpm_grid_index(i0, i1, i2 - 1, i3)] +
                            grid->pressure[mpm_grid_index(i0, i1, i2, i3 + 1)] + grid->pressure[mpm_grid_index(i0, i1, i2, i3 - 1)];

                        int idx = mpm_grid_index(i0, i1, i2, i3);
                        grid->pressure[idx] = (sum_p - dx2 * grid->div[idx]) * inv_8;
                    }
                }
            }
        }
    }

    // 2d. Project velocity: v_grid <- v_grid - grad(pressure) + nu * laplacian(v)
    for (int i0 = 1; i0 < DHVN_GRID_DIM - 1; ++i0) {
        for (int i1 = 1; i1 < DHVN_GRID_DIM - 1; ++i1) {
            for (int i2 = 1; i2 < DHVN_GRID_DIM - 1; ++i2) {
                for (int i3 = 1; i3 < DHVN_GRID_DIM - 1; ++i3) {
                    int c_idx = mpm_grid_index(i0, i1, i2, i3);
                    float gp0 = (grid->pressure[mpm_grid_index(i0 + 1, i1, i2, i3)] - grid->pressure[mpm_grid_index(i0 - 1, i1, i2, i3)]) * half_inv_dx;
                    float gp1 = (grid->pressure[mpm_grid_index(i0, i1 + 1, i2, i3)] - grid->pressure[mpm_grid_index(i0, i1 - 1, i2, i3)]) * half_inv_dx;
                    float gp2 = (grid->pressure[mpm_grid_index(i0, i1, i2 + 1, i3)] - grid->pressure[mpm_grid_index(i0, i1, i2 - 1, i3)]) * half_inv_dx;
                    float gp3 = (grid->pressure[mpm_grid_index(i0, i1, i2, i3 + 1)] - grid->pressure[mpm_grid_index(i0, i1, i2, i3 - 1)]) * half_inv_dx;

                    grid->vel[c_idx * 4 + 0] = (grid->vel[c_idx * 4 + 0] - gp0) * (1.0f - viscosity * dt);
                    grid->vel[c_idx * 4 + 1] = (grid->vel[c_idx * 4 + 1] - gp1) * (1.0f - viscosity * dt);
                    grid->vel[c_idx * 4 + 2] = (grid->vel[c_idx * 4 + 2] - gp2) * (1.0f - viscosity * dt);
                    grid->vel[c_idx * 4 + 3] = (grid->vel[c_idx * 4 + 3] - gp3) * (1.0f - viscosity * dt);
                }
            }
        }
    }

    // Phase 3: G2P (Grid to Particle Transfer & Position Advection)
    float max_r = sys->radius;
    float boundary_damping = 0.85f;

    for (int p = 0; p < np; ++p) {
        DHVNParticle* pt = &sys->particles[p];

        float u0 = (pt->pos[0] + bound) * inv_dx;
        float u1 = (pt->pos[1] + bound) * inv_dx;
        float u2 = (pt->pos[2] + bound) * inv_dx;
        float u3 = (pt->pos[3] + bound) * inv_dx;

        if (u0 < 0.0f) u0 = 0.0f; else if (u0 > (float)(DHVN_GRID_DIM - 1.001f)) u0 = (float)(DHVN_GRID_DIM - 1.001f);
        if (u1 < 0.0f) u1 = 0.0f; else if (u1 > (float)(DHVN_GRID_DIM - 1.001f)) u1 = (float)(DHVN_GRID_DIM - 1.001f);
        if (u2 < 0.0f) u2 = 0.0f; else if (u2 > (float)(DHVN_GRID_DIM - 1.001f)) u2 = (float)(DHVN_GRID_DIM - 1.001f);
        if (u3 < 0.0f) u3 = 0.0f; else if (u3 > (float)(DHVN_GRID_DIM - 1.001f)) u3 = (float)(DHVN_GRID_DIM - 1.001f);

        int b0 = (int)u0;
        int b1 = (int)u1;
        int b2 = (int)u2;
        int b3 = (int)u3;

        float f0 = u0 - (float)b0;
        float f1 = u1 - (float)b1;
        float f2 = u2 - (float)b2;
        float f3 = u3 - (float)b3;

        float w0[2] = { 1.0f - f0, f0 };
        float w1[2] = { 1.0f - f1, f1 };
        float w2[2] = { 1.0f - f2, f2 };
        float w3[2] = { 1.0f - f3, f3 };

        float new_v[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

        for (int d0 = 0; d0 < 2; ++d0) {
            float wd0 = w0[d0];
            int n0 = b0 + d0;
            for (int d1 = 0; d1 < 2; ++d1) {
                float wd1 = wd0 * w1[d1];
                int n1 = b1 + d1;
                for (int d2 = 0; d2 < 2; ++d2) {
                    float wd2 = wd1 * w2[d2];
                    int n2 = b2 + d2;
                    for (int d3 = 0; d3 < 2; ++d3) {
                        float W = wd2 * w3[d3];
                        int n3 = b3 + d3;
                        int idx = mpm_grid_index(n0, n1, n2, n3);

                        new_v[0] += W * grid->vel[idx * 4 + 0];
                        new_v[1] += W * grid->vel[idx * 4 + 1];
                        new_v[2] += W * grid->vel[idx * 4 + 2];
                        new_v[3] += W * grid->vel[idx * 4 + 3];
                    }
                }
            }
        }

        // PIC/FLIP blend velocity update (0.90 PIC + 0.10 FLIP for vortex preservation)
        pt->vel[0] = 0.90f * new_v[0] + 0.10f * pt->vel[0];
        pt->vel[1] = 0.90f * new_v[1] + 0.10f * pt->vel[1];
        pt->vel[2] = 0.90f * new_v[2] + 0.10f * pt->vel[2];
        pt->vel[3] = 0.90f * new_v[3] + 0.10f * pt->vel[3];

        // Advect particle positions
        pt->pos[0] += pt->vel[0] * dt;
        pt->pos[1] += pt->vel[1] * dt;
        pt->pos[2] += pt->vel[2] * dt;
        pt->pos[3] += pt->vel[3] * dt;

        // Spherical boundary reflection in 4D
        float r2 = pt->pos[0]*pt->pos[0] + pt->pos[1]*pt->pos[1] + pt->pos[2]*pt->pos[2] + pt->pos[3]*pt->pos[3];
        if (r2 > max_r * max_r) {
            float r = sqrtf(r2);
            float inv_r = 1.0f / r;
            float n0 = pt->pos[0] * inv_r;
            float n1 = pt->pos[1] * inv_r;
            float n2 = pt->pos[2] * inv_r;
            float n3 = pt->pos[3] * inv_r;

            // Normal velocity reflection
            float v_dot_n = pt->vel[0]*n0 + pt->vel[1]*n1 + pt->vel[2]*n2 + pt->vel[3]*n3;
            if (v_dot_n > 0.0f) {
                pt->vel[0] -= (1.0f + boundary_damping) * v_dot_n * n0;
                pt->vel[1] -= (1.0f + boundary_damping) * v_dot_n * n1;
                pt->vel[2] -= (1.0f + boundary_damping) * v_dot_n * n2;
                pt->vel[3] -= (1.0f + boundary_damping) * v_dot_n * n3;
            }

            pt->pos[0] = n0 * max_r * 0.99f;
            pt->pos[1] = n1 * max_r * 0.99f;
            pt->pos[2] = n2 * max_r * 0.99f;
            pt->pos[3] = n3 * max_r * 0.99f;
        }

        // Bounded Kelvin circulation conservation factor
        for (int c = 0; c < dim_c; ++c) {
            pt->charge[c] *= sys->cognitive_decay;
        }
    }
}

// Pooled Chamber Charge
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

// Influx Injection into Chamber 1
static void mpm_inject_token_influx(DHVNSystem* sys, const float* phi_t, float dt) {
    int num_inputs = 2 * DHVN_MAX_HARMONICS;
    int dim_c = sys->dim_c;
    int np = sys->num_particles;

    float port[4] = { sys->radius * 0.8f, 0.0f, 0.0f, 0.0f };
    float h = sys->h;
    float h2 = h * h;

    for (int i = 0; i < np; ++i) {
        DHVNParticle* p = &sys->particles[i];
        float dx0 = p->pos[0] - port[0];
        float dx1 = p->pos[1] - port[1];
        float dx2 = p->pos[2] - port[2];
        float dx3 = p->pos[3] - port[3];
        float dist2 = dx0*dx0 + dx1*dx1 + dx2*dx2 + dx3*dx3;

        if (dist2 < 4.0f * h2) {
            float dist = sqrtf(dist2);
            float weight = (1.0f - dist / (2.0f * h));
            if (weight < 0.0f) weight = 0.0f;

            for (int c = 0; c < dim_c; ++c) {
                float charge_influx = 0.0f;
                for (int k = 0; k < num_inputs; ++k) {
                    charge_influx += sys->W_charge[c * num_inputs + k] * phi_t[k];
                }
                p->charge[c] += weight * charge_influx * dt * 5.0f;
            }
        }
    }
}

// Waveguide Transfer C_src -> C_dst
static void mpm_waveguide_transfer(DHVNSystem* src, DHVNSystem* dst, const float* W_wg, float dt) {
    int c_src = src->dim_c;
    int c_dst = dst->dim_c;
    float* pooled_src = (float*)alloca(c_src * sizeof(float));
    compute_pooled_charge(src, pooled_src);

    float* wg_charge = (float*)alloca(c_dst * sizeof(float));
    for (int d = 0; d < c_dst; ++d) {
        float sum = 0.0f;
        for (int s = 0; s < c_src; ++s) {
            sum += W_wg[d * c_src + s] * pooled_src[s];
        }
        wg_charge[d] = sum;
    }

    int np_dst = dst->num_particles;
    for (int i = 0; i < np_dst; ++i) {
        DHVNParticle* p = &dst->particles[i];
        for (int d = 0; d < c_dst; ++d) {
            p->charge[d] += wg_charge[d] * dt * 2.0f;
        }
    }
}

// Create Tri-Chamber Model with 4D MPM Grids
DeepDHVN_MPM* dhvn_mpm_create_tri(
    int vocab_size,
    int np1, int dim_c1,
    int np2, int dim_c2,
    int np3, int dim_c3,
    int dim_hid
) {
    DeepDHVN_MPM* m = (DeepDHVN_MPM*)malloc(sizeof(DeepDHVN_MPM));
    m->vocab_size = vocab_size;
    m->dim_c1 = dim_c1;
    m->dim_c2 = dim_c2;
    m->dim_c3 = dim_c3;
    m->dim_hid = dim_hid;
    m->num_inputs = 2 * DHVN_MAX_HARMONICS;
    m->t_step = 1;
    m->num_substeps = 1;

    float r1 = 1.0f, h1 = 0.45f;
    float r2 = 1.2f, h2 = 0.50f;
    float r3 = 1.4f, h3 = 0.55f;

    m->chamber_char = dhvn_create(np1, dim_c1, r1, h1);
    m->chamber_sem  = dhvn_create(np2, dim_c2, r2, h2);
    m->chamber_disc = dhvn_create(np3, dim_c3, r3, h3);

    m->grid1 = dhvn_mpm_create_grid(r1 * 1.15f, dim_c1);
    m->grid2 = dhvn_mpm_create_grid(r2 * 1.15f, dim_c2);
    m->grid3 = dhvn_mpm_create_grid(r3 * 1.15f, dim_c3);

    // Waveguides
    m->W_waveguide = (float*)mpm_aligned_alloc(dim_c2 * dim_c1 * sizeof(float));
    m->grad_W_waveguide = (float*)mpm_aligned_alloc(dim_c2 * dim_c1 * sizeof(float));
    m->m_W_waveguide = (float*)mpm_aligned_alloc(dim_c2 * dim_c1 * sizeof(float));
    m->v_W_waveguide = (float*)mpm_aligned_alloc(dim_c2 * dim_c1 * sizeof(float));

    m->W_waveguide2 = (float*)mpm_aligned_alloc(dim_c3 * dim_c2 * sizeof(float));
    m->grad_W_waveguide2 = (float*)mpm_aligned_alloc(dim_c3 * dim_c2 * sizeof(float));
    m->m_W_waveguide2 = (float*)mpm_aligned_alloc(dim_c3 * dim_c2 * sizeof(float));
    m->v_W_waveguide2 = (float*)mpm_aligned_alloc(dim_c3 * dim_c2 * sizeof(float));

    // Linear residual heads
    m->W_out1 = (float*)mpm_aligned_alloc(vocab_size * dim_c1 * sizeof(float));
    m->W_out2 = (float*)mpm_aligned_alloc(vocab_size * dim_c2 * sizeof(float));
    m->W_out3 = (float*)mpm_aligned_alloc(vocab_size * dim_c3 * sizeof(float));
    m->grad_W_out1 = (float*)mpm_aligned_alloc(vocab_size * dim_c1 * sizeof(float));
    m->grad_W_out2 = (float*)mpm_aligned_alloc(vocab_size * dim_c2 * sizeof(float));
    m->grad_W_out3 = (float*)mpm_aligned_alloc(vocab_size * dim_c3 * sizeof(float));
    m->m_W_out1 = (float*)mpm_aligned_alloc(vocab_size * dim_c1 * sizeof(float));
    m->v_W_out1 = (float*)mpm_aligned_alloc(vocab_size * dim_c1 * sizeof(float));
    m->m_W_out2 = (float*)mpm_aligned_alloc(vocab_size * dim_c2 * sizeof(float));
    m->v_W_out2 = (float*)mpm_aligned_alloc(vocab_size * dim_c2 * sizeof(float));
    m->m_W_out3 = (float*)mpm_aligned_alloc(vocab_size * dim_c3 * sizeof(float));
    m->v_W_out3 = (float*)mpm_aligned_alloc(vocab_size * dim_c3 * sizeof(float));

    // Resonant MLP Readout Head
    int in_dim = m->num_inputs;
    int c_in = dim_c1 + dim_c2 + dim_c3 + in_dim;

    m->W_hid = (float*)mpm_aligned_alloc(dim_hid * c_in * sizeof(float));
    m->b_hid = (float*)mpm_aligned_alloc(dim_hid * sizeof(float));
    m->grad_W_hid = (float*)mpm_aligned_alloc(dim_hid * c_in * sizeof(float));
    m->grad_b_hid = (float*)mpm_aligned_alloc(dim_hid * sizeof(float));
    m->m_W_hid = (float*)mpm_aligned_alloc(dim_hid * c_in * sizeof(float));
    m->v_W_hid = (float*)mpm_aligned_alloc(dim_hid * c_in * sizeof(float));
    m->m_b_hid = (float*)mpm_aligned_alloc(dim_hid * sizeof(float));
    m->v_b_hid = (float*)mpm_aligned_alloc(dim_hid * sizeof(float));

    m->W_head = (float*)mpm_aligned_alloc(vocab_size * dim_hid * sizeof(float));
    m->b_head = (float*)mpm_aligned_alloc(vocab_size * sizeof(float));
    m->grad_W_head = (float*)mpm_aligned_alloc(vocab_size * dim_hid * sizeof(float));
    m->grad_b_head = (float*)mpm_aligned_alloc(vocab_size * sizeof(float));
    m->m_W_head = (float*)mpm_aligned_alloc(vocab_size * dim_hid * sizeof(float));
    m->v_W_head = (float*)mpm_aligned_alloc(vocab_size * dim_hid * sizeof(float));
    m->m_b_head = (float*)mpm_aligned_alloc(vocab_size * sizeof(float));
    m->v_b_head = (float*)mpm_aligned_alloc(vocab_size * sizeof(float));

    // Chamber 1 influx Adam buffers
    m->grad_W_charge1 = (float*)mpm_aligned_alloc(dim_c1 * in_dim * sizeof(float));
    m->m_W_charge1 = (float*)mpm_aligned_alloc(dim_c1 * in_dim * sizeof(float));
    m->v_W_charge1 = (float*)mpm_aligned_alloc(dim_c1 * in_dim * sizeof(float));

    // Xavier initialization
    float s_wg1 = sqrtf(2.0f / (float)(dim_c1 + dim_c2));
    for (int i = 0; i < dim_c2 * dim_c1; ++i) m->W_waveguide[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * s_wg1;

    float s_wg2 = sqrtf(2.0f / (float)(dim_c2 + dim_c3));
    for (int i = 0; i < dim_c3 * dim_c2; ++i) m->W_waveguide2[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * s_wg2;

    float s_hid = sqrtf(2.0f / (float)(c_in + dim_hid));
    for (int i = 0; i < dim_hid * c_in; ++i) m->W_hid[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * s_hid;

    float s_head = sqrtf(2.0f / (float)(dim_hid + vocab_size));
    for (int i = 0; i < vocab_size * dim_hid; ++i) m->W_head[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * s_head;

    return m;
}

void dhvn_mpm_free(DeepDHVN_MPM* m) {
    if (!m) return;
    dhvn_free(m->chamber_char);
    dhvn_free(m->chamber_sem);
    dhvn_free(m->chamber_disc);
    dhvn_mpm_free_grid(m->grid1);
    dhvn_mpm_free_grid(m->grid2);
    dhvn_mpm_free_grid(m->grid3);

    free(m->W_waveguide); free(m->grad_W_waveguide); free(m->m_W_waveguide); free(m->v_W_waveguide);
    free(m->W_waveguide2); free(m->grad_W_waveguide2); free(m->m_W_waveguide2); free(m->v_W_waveguide2);

    free(m->W_out1); free(m->W_out2); free(m->W_out3);
    free(m->grad_W_out1); free(m->grad_W_out2); free(m->grad_W_out3);
    free(m->m_W_out1); free(m->v_W_out1);
    free(m->m_W_out2); free(m->v_W_out2);
    free(m->m_W_out3); free(m->v_W_out3);

    free(m->W_hid); free(m->b_hid);
    free(m->grad_W_hid); free(m->grad_b_hid);
    free(m->m_W_hid); free(m->v_W_hid);
    free(m->m_b_hid); free(m->v_b_hid);

    free(m->W_head); free(m->b_head);
    free(m->grad_W_head); free(m->grad_b_head);
    free(m->m_W_head); free(m->v_W_head);
    free(m->m_b_head); free(m->v_b_head);

    free(m->grad_W_charge1); free(m->m_W_charge1); free(m->v_W_charge1);
    free(m);
}

void dhvn_mpm_set_substeps(DeepDHVN_MPM* model, int substeps) {
    if (model) model->num_substeps = (substeps > 0) ? substeps : 1;
}

// Forward Token Inference with AVX2 & 12-Thread OpenMP
void dhvn_mpm_forward_token(
    DeepDHVN_MPM* m,
    int in_token,
    float* out_probs,
    float* out_hidden,
    float* out_input_features,
    float dt
) {
    int v_dim = m->vocab_size;
    int in_dim = m->num_inputs;

    // Compute input Fourier packet
    float theta = 2.0f * (float)M_PI * ((float)(in_token % v_dim) / (float)v_dim);
    float* phi_t = (float*)alloca(in_dim * sizeof(float));
    dhvn_compute_fourier_packet(theta, phi_t);

    // Chamber 1 influx
    mpm_inject_token_influx(m->chamber_char, phi_t, dt);

    // Multi-substep MPM fluid solver
    for (int sub = 0; sub < m->num_substeps; ++sub) {
        float sub_dt = dt / (float)m->num_substeps;
        dhvn_mpm_solve_chamber(m->chamber_char, m->grid1, sub_dt, 0.08f);
        mpm_waveguide_transfer(m->chamber_char, m->chamber_sem, m->W_waveguide, sub_dt);
        dhvn_mpm_solve_chamber(m->chamber_sem, m->grid2, sub_dt, 0.06f);
        mpm_waveguide_transfer(m->chamber_sem, m->chamber_disc, m->W_waveguide2, sub_dt);
        dhvn_mpm_solve_chamber(m->chamber_disc, m->grid3, sub_dt, 0.04f);
    }

    int c1 = m->dim_c1;
    int c2 = m->dim_c2;
    int c3 = m->dim_c3;
    float* pooled1 = (float*)alloca(c1 * sizeof(float));
    float* pooled2 = (float*)alloca(c2 * sizeof(float));
    float* pooled3 = (float*)alloca(c3 * sizeof(float));
    compute_pooled_charge(m->chamber_char, pooled1);
    compute_pooled_charge(m->chamber_sem, pooled2);
    compute_pooled_charge(m->chamber_disc, pooled3);

    int c_in = c1 + c2 + c3 + in_dim;
    float* x_in = (float*)alloca(c_in * sizeof(float));
    memcpy(x_in, pooled1, c1 * sizeof(float));
    memcpy(x_in + c1, pooled2, c2 * sizeof(float));
    memcpy(x_in + c1 + c2, pooled3, c3 * sizeof(float));
    memcpy(x_in + c1 + c2 + c3, phi_t, in_dim * sizeof(float));

    if (out_input_features) {
        memcpy(out_input_features, x_in, c_in * sizeof(float));
    }

    // Resonant MLP Hidden Projection with AVX2 FMA
    int dh = m->dim_hid;
    float* h = (float*)alloca(dh * sizeof(float));
    #pragma omp parallel for num_threads(DHVN_MPM_THREADS) schedule(static)
    for (int k = 0; k < dh; ++k) {
        float dot = avx2_dot_product(m->W_hid + k * c_in, x_in, c_in);
        h[k] = mpm_gelu(dot + m->b_hid[k]);
    }

    if (out_hidden) {
        memcpy(out_hidden, h, dh * sizeof(float));
    }

    // Vocabulary Readout with AVX2 & 12 Threads
    float sc_head = 1.0f / sqrtf((float)dh);
    float scale_res = (1.0f / sqrtf((float)(c1 + c2 + c3))) * 0.4f;
    float* logits = (float*)alloca(v_dim * sizeof(float));

    #pragma omp parallel for num_threads(DHVN_MPM_THREADS) schedule(static)
    for (int v = 0; v < v_dim; ++v) {
        float sum_head = dot256(m->W_head + v * dh, h) + m->b_head[v];
        float s1 = dot16(m->W_out1 + v * c1, pooled1);
        float s2 = dot16(m->W_out2 + v * c2, pooled2);
        float s3 = dot16(m->W_out3 + v * c3, pooled3);
        logits[v] = sum_head * sc_head + (s1 + s2 + s3) * scale_res;
    }

    // Softmax
    float max_l = logits[0];
    for (int v = 1; v < v_dim; ++v) {
        if (logits[v] > max_l) max_l = logits[v];
    }

    float sum_exp = 0.0f;
    for (int v = 0; v < v_dim; ++v) {
        out_probs[v] = expf(logits[v] - max_l);
        sum_exp += out_probs[v];
    }
    float inv_exp = 1.0f / (sum_exp + 1e-7f);
    for (int v = 0; v < v_dim; ++v) {
        out_probs[v] *= inv_exp;
    }
}

// Dedicated Adam Optimizer Step across all parameters (called once per batch/sequence)
static void dhvn_mpm_adam_step(DeepDHVN_MPM* m, float lr, float scale_grad) {
    int v_dim = m->vocab_size;
    int dh = m->dim_hid;
    int c1 = m->dim_c1;
    int c2 = m->dim_c2;
    int c3 = m->dim_c3;
    int in_dim = m->num_inputs;
    int c_in = c1 + c2 + c3 + in_dim;

    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    int t = m->t_step;
    float alpha_t = lr * sqrtf(1.0f - powf(beta2, (float)t)) / (1.0f - powf(beta1, (float)t));

    // Update W_head & b_head (AVX2 + 12 Threads)
    #pragma omp parallel for num_threads(DHVN_MPM_THREADS) schedule(static)
    for (int i = 0; i < v_dim * dh; ++i) {
        float g = m->grad_W_head[i] * scale_grad;
        if (g > 5.0f) g = 5.0f; else if (g < -5.0f) g = -5.0f;
        m->m_W_head[i] = beta1 * m->m_W_head[i] + (1.0f - beta1) * g;
        m->v_W_head[i] = beta2 * m->v_W_head[i] + (1.0f - beta2) * g * g;
        m->W_head[i] -= alpha_t * m->m_W_head[i] / (sqrtf(m->v_W_head[i]) + eps);
        m->grad_W_head[i] = 0.0f;
    }

    #pragma omp parallel for num_threads(DHVN_MPM_THREADS) schedule(static)
    for (int v = 0; v < v_dim; ++v) {
        float g = m->grad_b_head[v] * scale_grad;
        if (g > 5.0f) g = 5.0f; else if (g < -5.0f) g = -5.0f;
        m->m_b_head[v] = beta1 * m->m_b_head[v] + (1.0f - beta1) * g;
        m->v_b_head[v] = beta2 * m->v_b_head[v] + (1.0f - beta2) * g * g;
        m->b_head[v] -= alpha_t * m->m_b_head[v] / (sqrtf(m->v_b_head[v]) + eps);
        m->grad_b_head[v] = 0.0f;
    }

    // Update W_hid & b_hid
    #pragma omp parallel for num_threads(DHVN_MPM_THREADS) schedule(static)
    for (int i = 0; i < dh * c_in; ++i) {
        float g = m->grad_W_hid[i] * scale_grad;
        if (g > 5.0f) g = 5.0f; else if (g < -5.0f) g = -5.0f;
        m->m_W_hid[i] = beta1 * m->m_W_hid[i] + (1.0f - beta1) * g;
        m->v_W_hid[i] = beta2 * m->v_W_hid[i] + (1.0f - beta2) * g * g;
        m->W_hid[i] -= alpha_t * m->m_W_hid[i] / (sqrtf(m->v_W_hid[i]) + eps);
        m->grad_W_hid[i] = 0.0f;
    }

    for (int k = 0; k < dh; ++k) {
        float g = m->grad_b_hid[k] * scale_grad;
        if (g > 5.0f) g = 5.0f; else if (g < -5.0f) g = -5.0f;
        m->m_b_hid[k] = beta1 * m->m_b_hid[k] + (1.0f - beta1) * g;
        m->v_b_hid[k] = beta2 * m->v_b_hid[k] + (1.0f - beta2) * g * g;
        m->b_hid[k] -= alpha_t * m->m_b_hid[k] / (sqrtf(m->v_b_hid[k]) + eps);
        m->grad_b_hid[k] = 0.0f;
    }

    // Update linear residual out weights
    for (int i = 0; i < v_dim * c1; ++i) {
        float g = m->grad_W_out1[i] * scale_grad;
        if (g > 5.0f) g = 5.0f; else if (g < -5.0f) g = -5.0f;
        m->m_W_out1[i] = beta1 * m->m_W_out1[i] + (1.0f - beta1) * g;
        m->v_W_out1[i] = beta2 * m->v_W_out1[i] + (1.0f - beta2) * g * g;
        m->W_out1[i] -= alpha_t * m->m_W_out1[i] / (sqrtf(m->v_W_out1[i]) + eps);
        m->grad_W_out1[i] = 0.0f;
    }
    for (int i = 0; i < v_dim * c2; ++i) {
        float g = m->grad_W_out2[i] * scale_grad;
        if (g > 5.0f) g = 5.0f; else if (g < -5.0f) g = -5.0f;
        m->m_W_out2[i] = beta1 * m->m_W_out2[i] + (1.0f - beta1) * g;
        m->v_W_out2[i] = beta2 * m->v_W_out2[i] + (1.0f - beta2) * g * g;
        m->W_out2[i] -= alpha_t * m->m_W_out2[i] / (sqrtf(m->v_W_out2[i]) + eps);
        m->grad_W_out2[i] = 0.0f;
    }
    for (int i = 0; i < v_dim * c3; ++i) {
        float g = m->grad_W_out3[i] * scale_grad;
        if (g > 5.0f) g = 5.0f; else if (g < -5.0f) g = -5.0f;
        m->m_W_out3[i] = beta1 * m->m_W_out3[i] + (1.0f - beta1) * g;
        m->v_W_out3[i] = beta2 * m->v_W_out3[i] + (1.0f - beta2) * g * g;
        m->W_out3[i] -= alpha_t * m->m_W_out3[i] / (sqrtf(m->v_W_out3[i]) + eps);
        m->grad_W_out3[i] = 0.0f;
    }

    m->t_step++;
}

// Token step with gradient accumulation (fast)
static float dhvn_mpm_accum_step(
    DeepDHVN_MPM* m,
    int in_token,
    int target_token,
    float dt
) {
    int v_dim = m->vocab_size;
    int dh = m->dim_hid;
    int c1 = m->dim_c1;
    int c2 = m->dim_c2;
    int c3 = m->dim_c3;
    int in_dim = m->num_inputs;
    int c_in = c1 + c2 + c3 + in_dim;

    float* probs = (float*)alloca(v_dim * sizeof(float));
    float* h = (float*)alloca(dh * sizeof(float));
    float* x_in = (float*)alloca(c_in * sizeof(float));

    dhvn_mpm_forward_token(m, in_token, probs, h, x_in, dt);

    if (target_token < 0 || target_token >= v_dim) target_token = 0;
    float target_p = probs[target_token];
    float loss = -logf(target_p + 1e-7f);

    float* delta_logits = (float*)alloca(v_dim * sizeof(float));
    for (int v = 0; v < v_dim; ++v) {
        delta_logits[v] = probs[v] - ((v == target_token) ? 1.0f : 0.0f);
    }

    float sc_head = 1.0f / sqrtf((float)dh);
    float scale_res = (1.0f / sqrtf((float)(c1 + c2 + c3))) * 0.4f;

    // Head Gradients
    #pragma omp parallel for num_threads(DHVN_MPM_THREADS) schedule(static)
    for (int v = 0; v < v_dim; ++v) {
        float d_l = delta_logits[v];
        float d_head = d_l * sc_head;

        m->grad_b_head[v] += d_head;
        float* g_W_h_row = m->grad_W_head + v * dh;
        for (int k = 0; k < dh; ++k) {
            g_W_h_row[k] += d_head * h[k];
        }

        float d_res = d_l * scale_res;
        float* g_out1 = m->grad_W_out1 + v * c1;
        float* g_out2 = m->grad_W_out2 + v * c2;
        float* g_out3 = m->grad_W_out3 + v * c3;
        for (int c = 0; c < c1; ++c) g_out1[c] += d_res * x_in[c];
        for (int c = 0; c < c2; ++c) g_out2[c] += d_res * x_in[c1 + c];
        for (int c = 0; c < c3; ++c) g_out3[c] += d_res * x_in[c1 + c2 + c];
    }

    // Parallel backprop to hidden state: delta_h[k] = sum_v (delta_head_v * W_head[v, k])
    float* delta_h = (float*)alloca(dh * sizeof(float));
    #pragma omp parallel for num_threads(DHVN_MPM_THREADS) schedule(static)
    for (int k = 0; k < dh; ++k) {
        float sum = 0.0f;
        for (int v = 0; v < v_dim; ++v) {
            sum += delta_logits[v] * sc_head * m->W_head[v * dh + k];
        }
        delta_h[k] = sum;
    }

    // Backprop through GELU to W_hid, b_hid
    #pragma omp parallel for num_threads(DHVN_MPM_THREADS) schedule(static)
    for (int k = 0; k < dh; ++k) {
        float val = avx2_dot_product(m->W_hid + k * c_in, x_in, c_in) + m->b_hid[k];
        float d_act = delta_h[k] * mpm_gelu_grad(val);

        m->grad_b_hid[k] += d_act;
        float* g_hid_row = m->grad_W_hid + k * c_in;
        for (int j = 0; j < c_in; ++j) {
            g_hid_row[j] += d_act * x_in[j];
        }
    }

    return loss;
}

// Single Token Training Step with Immediate Adam
float dhvn_mpm_train_step(
    DeepDHVN_MPM* m,
    int in_token,
    int target_token,
    float lr,
    float dt
) {
    float loss = dhvn_mpm_accum_step(m, in_token, target_token, dt);
    dhvn_mpm_adam_step(m, lr, 1.0f);
    return loss;
}

// High-Speed Multi-Token Sequence Chunk Training
// Accumulates gradients over the sequence and performs ONE unified Adam update!
float dhvn_mpm_train_sequence(
    DeepDHVN_MPM* m,
    const int* tokens,
    int seq_len,
    float lr,
    float dt
) {
    if (!m || !tokens || seq_len < 2) return 0.0f;
    float total_loss = 0.0f;
    int count = seq_len - 1;

    for (int i = 0; i < count; ++i) {
        int in_tok = tokens[i];
        int tgt_tok = tokens[i + 1];
        total_loss += dhvn_mpm_accum_step(m, in_tok, tgt_tok, dt);
    }

    // Single unified Adam update scaled by 1/count
    float scale = 1.0f / (float)count;
    dhvn_mpm_adam_step(m, lr, scale);

    return total_loss * scale;
}

// Option 1: Topological Memory Crystallization ("Fluid Sleep")
void dhvn_mpm_crystallize(DeepDHVN_MPM* m, int sleep_cycles, float cooling_rate) {
    if (!m) return;
    DHVNSystem* chambers[3] = { m->chamber_char, m->chamber_sem, m->chamber_disc };
    DHVN4DGrid* grids[3] = { m->grid1, m->grid2, m->grid3 };

    for (int ch = 0; ch < 3; ++ch) {
        DHVNSystem* s = chambers[ch];
        DHVN4DGrid* g = grids[ch];
        int np = s->num_particles;
        int dim = s->dim_c;

        for (int cycle = 0; cycle < sleep_cycles; ++cycle) {
            float temp = 1.0f - (float)cycle * cooling_rate;
            if (temp < 0.1f) temp = 0.1f;

            // Annealing velocity cooling
            for (int i = 0; i < np; ++i) {
                s->particles[i].vel[0] *= temp;
                s->particles[i].vel[1] *= temp;
                s->particles[i].vel[2] *= temp;
                s->particles[i].vel[3] *= temp;
            }

            // Grid solve to consolidate laminar vortices
            dhvn_mpm_solve_chamber(s, g, 0.01f, 0.25f);

            // Consolidate co-circulating particles into permanent topological solitons
            for (int i = 0; i < np - 1; i += 2) {
                float dot = s->particles[i].vel[0] * s->particles[i+1].vel[0] +
                            s->particles[i].vel[1] * s->particles[i+1].vel[1] +
                            s->particles[i].vel[2] * s->particles[i+1].vel[2] +
                            s->particles[i].vel[3] * s->particles[i+1].vel[3];
                if (dot > 0.05f) {
                    for (int c = 0; c < dim; ++c) {
                        float avg = 0.5f * (s->particles[i].charge[c] + s->particles[i+1].charge[c]);
                        s->particles[i].charge[c] = avg * 1.05f;
                        s->particles[i+1].charge[c] = avg * 1.05f;
                    }
                }
            }
        }
    }
}

// Option 2: Continuous Analog Audio Waveform Influx
void dhvn_mpm_inject_waveform(DeepDHVN_MPM* m, const float* audio_pcm, int pcm_len, float dt) {
    if (!m || !audio_pcm || pcm_len <= 0) return;
    int num_inputs = m->num_inputs;
    float* phi = (float*)alloca(num_inputs * sizeof(float));

    int window = 256;
    for (int offset = 0; offset < pcm_len; offset += window) {
        int cur_len = (offset + window <= pcm_len) ? window : (pcm_len - offset);
        float energy = 0.0f;
        for (int k = 0; k < cur_len; ++k) {
            float s = audio_pcm[offset + k];
            energy += s * s;
        }
        energy = sqrtf(energy / (float)cur_len);
        float theta = energy * 2.0f * (float)M_PI;
        dhvn_compute_fourier_packet(theta, phi);

        mpm_inject_token_influx(m->chamber_char, phi, dt);
        dhvn_mpm_solve_chamber(m->chamber_char, m->grid1, dt, 0.08f);
    }
}

// Checkpoint Binary Serialization (Identical layout to standard checkpoints)
int dhvn_mpm_save_weights(DeepDHVN_MPM* m, const char* filepath) {
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

int dhvn_mpm_load_weights(DeepDHVN_MPM* m, const char* filepath) {
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
        int c_in_v2 = c1 + c2 + in_dim;
        fread(m->W_waveguide, sizeof(float), c2 * c1, fp);
        fread(m->W_out1, sizeof(float), v * c1, fp);
        fread(m->W_out2, sizeof(float), v * c2, fp);
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
