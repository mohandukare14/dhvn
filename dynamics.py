"""
Dynamics Engine for DHVN.
Implements 4D SPH Navier-Stokes, Tait Equation of State, Viscous Shear,
Gauge Coupling, and Symplectic Velocity-Verlet Integration.
"""

import math
from typing import List
from kernel import quintic_kernel_4d, grad_quintic_kernel_4d

class SPHDynamics4D:
    def __init__(
        self,
        viscosity: float = 0.05,
        stiffness_B: float = 1.0,
        gamma: float = 7.0,
        rho_0: float = 1.0,
        boundary_k: float = 20.0
    ):
        self.viscosity = viscosity
        self.stiffness_B = stiffness_B
        self.gamma = gamma
        self.rho_0 = rho_0
        self.boundary_k = boundary_k

    def compute_density_and_pressure(self, system):
        """Computes local fluid density rho_i and Tait pressure P_i."""
        h = system.h
        particles = system.particles
        n = len(particles)

        for i in range(n):
            pi = particles[i]
            rho = pi.mass * quintic_kernel_4d(0.0, h)
            for j in range(n):
                if i == j:
                    continue
                pj = particles[j]
                r = math.sqrt(sum((pi.pos[k] - pj.pos[k]) ** 2 for k in range(4)))
                if r < 3.0 * h:
                    rho += pj.mass * quintic_kernel_4d(r, h)
            
            pi.density = max(0.1, rho)
            # Tait Equation of State: P = B * ((rho / rho_0)^gamma - 1)
            ratio = pi.density / self.rho_0
            pi.pressure = self.stiffness_B * (math.pow(ratio, self.gamma) - 1.0)
            pi.pressure = max(0.0, pi.pressure)

    def compute_accelerations_and_charge_transfer(self, system) -> List[List[float]]:
        """
        Computes 4D Navier-Stokes acceleration:
        a_i = a_pressure + a_viscosity + a_gauge + a_boundary
        and semantic charge derivative dc_i/dt.
        """
        h = system.h
        particles = system.particles
        n = len(particles)
        
        # Reset accelerations and charge deltas
        dc_dt = [[0.0] * system.dim_c for _ in range(n)]

        for i in range(n):
            pi = particles[i]
            # Reset acceleration
            a_p = [0.0, 0.0, 0.0, 0.0]
            a_v = [0.0, 0.0, 0.0, 0.0]
            a_g = [0.0, 0.0, 0.0, 0.0]

            for j in range(n):
                if i == j:
                    continue
                pj = particles[j]
                grad_W, r = grad_quintic_kernel_4d(pi.pos, pj.pos, h)
                if r >= 3.0 * h or r < 1e-6:
                    continue

                # 1. Symmetric Pressure Acceleration
                p_term = (pi.pressure / (pi.density ** 2)) + (pj.pressure / (pj.density ** 2))
                for d in range(4):
                    a_p[d] -= pj.mass * p_term * grad_W[d]

                # 2. Viscous Shear Diffusion
                dx = [pi.pos[k] - pj.pos[k] for k in range(4)]
                dot_x_grad = sum(dx[k] * grad_W[k] for k in range(4))
                denom = (r ** 2) + 0.01 * (h ** 2)
                v_diff = [pj.vel[k] - pi.vel[k] for k in range(4)]
                for d in range(4):
                    a_v[d] += self.viscosity * (pj.mass / pj.density) * v_diff[d] * (dot_x_grad / denom)

                # 3. Neural Gauge Semantic Interaction
                # Dot product of semantic charges acts as attraction/repulsion potential
                charge_dot = sum(pi.charge[c] * pj.charge[c] for c in range(system.dim_c))
                gauge_factor = math.tanh(charge_dot) * 0.2
                for d in range(4):
                    a_g[d] += (pj.mass / pj.density) * gauge_factor * grad_W[d]

                # 4. Semantic Charge Transport
                W_val = quintic_kernel_4d(r, h)
                for c in range(system.dim_c):
                    # Exchange features smoothly with neighboring particles
                    dc_dt[i][c] += 0.5 * (pj.mass / pj.density) * (pj.charge[c] - pi.charge[c]) * W_val

            # 5. Boundary Elastic Barrier (Keeps particles within R_0)
            dist_origin = math.sqrt(sum(pi.pos[k] ** 2 for k in range(4)))
            a_b = [0.0, 0.0, 0.0, 0.0]
            if dist_origin > system.radius * 0.9:
                pen = dist_origin - (system.radius * 0.9)
                for d in range(4):
                    a_b[d] = -self.boundary_k * pen * (pi.pos[d] / (dist_origin + 1e-6))

            # Total acceleration
            for d in range(4):
                pi.acc[d] = a_p[d] + a_v[d] + a_g[d] + a_b[d]

        return dc_dt

    def step(self, system, dt: float = 0.02):
        """Single symplectic Velocity-Verlet integration step."""
        # 1. Update density and pressure from current positions
        self.compute_density_and_pressure(system)

        # 2. Compute accelerations
        dc_dt = self.compute_accelerations_and_charge_transfer(system)

        # 3. Integrate positions and velocities (Velocity-Verlet)
        for i, p in enumerate(system.particles):
            for d in range(4):
                p.pos[d] += p.vel[d] * dt + 0.5 * p.acc[d] * (dt ** 2)
                p.vel[d] += p.acc[d] * dt
                # Damping drag to maintain kinetic stability
                p.vel[d] *= 0.98

            # Update semantic charges with continuous decay
            for c in range(system.dim_c):
                p.charge[c] += dc_dt[i][c] * dt
                p.charge[c] *= 0.99  # subtle cognitive decay

