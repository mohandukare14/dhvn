"""
Phase-Transition Readout Engine for DHVN.
Calculates 4D generalized kinetic enstrophy, detects critical droplet condensation,
projects state to output bytes via continuous resonance, and triggers pressure recoil.
"""

import math
from typing import List, Optional, Tuple
from kernel import grad_quintic_kernel_4d

class PhaseTransitionReadout:
    def __init__(self, critical_threshold: float = 0.05, recoil_strength: float = 0.5):
        self.critical_threshold = critical_threshold
        self.recoil_strength = recoil_strength

    def compute_enstrophy_field(self, system) -> float:
        """
        Computes the 4D vorticity 2-form and kinetic enstrophy for each particle.
        Returns max enstrophy found.
        """
        h = system.h
        particles = system.particles
        n = len(particles)
        max_enstrophy = 0.0

        for i in range(n):
            pi = particles[i]
            # 4D anti-symmetric vorticity tensor Omega[mu][nu] (4x4, 6 independent components)
            Omega = [[0.0] * 4 for _ in range(4)]

            for j in range(n):
                if i == j:
                    continue
                pj = particles[j]
                grad_W, r = grad_quintic_kernel_4d(pi.pos, pj.pos, h)
                if r >= 3.0 * h or r < 1e-6:
                    continue

                mass_rho = pj.mass / pj.density
                v_diff = [pj.vel[k] - pi.vel[k] for k in range(4)]

                for mu in range(4):
                    for nu in range(mu + 1, 4):
                        # 0.5 * (dv^mu/dx^nu - dv^nu/dx^mu)
                        term = 0.5 * mass_rho * (v_diff[mu] * grad_W[nu] - v_diff[nu] * grad_W[mu])
                        Omega[mu][nu] += term

            # Total kinetic enstrophy E_i = sum_{mu < nu} Omega_{mu,nu}^2
            enstrophy = 0.0
            for mu in range(4):
                for nu in range(mu + 1, 4):
                    enstrophy += Omega[mu][nu] ** 2

            pi.enstrophy = enstrophy
            if enstrophy > max_enstrophy:
                max_enstrophy = enstrophy

        return max_enstrophy

    def check_and_emit(self, system, vocab_module) -> Optional[Tuple[int, float, List[float]]]:
        """
        Checks if any particle cluster exceeds the critical enstrophy threshold.
        If triggered, condenses the droplet, emits byte, and triggers cavitation recoil.
        """
        max_e = self.compute_enstrophy_field(system)
        
        # Find critical focus particle
        critical_idx = -1
        highest_val = -1.0
        for i, p in enumerate(system.particles):
            val = p.enstrophy * p.density
            if val > self.critical_threshold and val > highest_val:
                highest_val = val
                critical_idx = i

        if critical_idx == -1:
            return None

        # 1. Droplet Cluster Pooling
        p_star = system.particles[critical_idx]
        cluster = []
        total_weight = 0.0
        pooled_charge = [0.0] * system.dim_c

        for p in system.particles:
            r = math.sqrt(sum((p.pos[k] - p_star.pos[k]) ** 2 for k in range(4)))
            if r < system.h:
                weight = p.mass * p.density
                cluster.append(p)
                total_weight += weight
                for c in range(system.dim_c):
                    pooled_charge[c] += p.charge[c] * weight

        if total_weight > 0:
            for c in range(system.dim_c):
                pooled_charge[c] /= total_weight

        # 2. Resonator Projection: Find byte with closest harmonic resonance
        best_byte = 0
        best_score = -1e9
        for b in range(32, 127):  # Printable ASCII range for demonstration
            phi = vocab_module.byte_to_harmonics(b)
            # Dot product with W_charge projection
            charge_expected = [
                sum(vocab_module.W_charge[c][j] * phi[j] for j in range(len(phi)))
                for c in range(system.dim_c)
            ]
            score = sum(pooled_charge[c] * charge_expected[c] for c in range(system.dim_c))
            if score > best_score:
                best_score = score
                best_byte = b

        # 3. Cavitation Recoil (Disperse droplet energy)
        for p in cluster:
            dx = [p.pos[k] - p_star.pos[k] for k in range(4)]
            dist = math.sqrt(sum(d * d for d in dx)) + 1e-4
            for k in range(4):
                p.vel[k] += self.recoil_strength * (dx[k] / dist)
            p.enstrophy = 0.0

        return best_byte, highest_val, p_star.pos

