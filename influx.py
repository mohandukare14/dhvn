"""
Dynamic Fluid Vocabulary Module (DHVN).
No static token lookup tables: translates continuous character/byte streams into
harmonic boundary impulses that perturb the 4D fluid substrate.
"""

import math
from typing import List, Tuple
from kernel import quintic_kernel_4d

class DynamicFluidVocabulary:
    def __init__(self, harmonics: int = 4, dim_c: int = 8, h_port: float = 0.8):
        self.harmonics = harmonics
        self.dim_c = dim_c
        self.h_port = h_port
        
        # Pseudo-random fixed projection weights for harmonic projection (zero static token table)
        # 2 * harmonics -> 4D force and dim_c charge
        num_inputs = 2 * harmonics
        import random
        rng = random.Random(1337)
        
        self.W_force = [[rng.gauss(0.0, 0.5) for _ in range(num_inputs)] for _ in range(4)]
        self.W_charge = [[rng.gauss(0.0, 0.5) for _ in range(num_inputs)] for _ in range(dim_c)]

    def byte_to_harmonics(self, byte_val: int) -> List[float]:
        """Continuous Fourier harmonic encoding of a byte (0-255)."""
        theta = 2.0 * math.pi * (byte_val / 256.0)
        phi = []
        for k in range(1, self.harmonics + 1):
            phi.append(math.cos(k * theta))
            phi.append(math.sin(k * theta))
        return phi

    def compute_impulse(self, byte_val: int) -> Tuple[List[float], List[float]]:
        """Maps byte directly to 4D mechanical force and semantic perturbation."""
        phi = self.byte_to_harmonics(byte_val)
        
        # 4D Force: W_force * phi
        force = [0.0] * 4
        for i in range(4):
            force[i] = sum(self.W_force[i][j] * phi[j] for j in range(len(phi)))
            
        # Semantic charge: W_charge * phi
        d_charge = [0.0] * self.dim_c
        for i in range(self.dim_c):
            d_charge[i] = sum(self.W_charge[i][j] * phi[j] for j in range(len(phi)))
            
        return force, d_charge

    def inject_byte(self, fluid_system, byte_val: int, port_pos: List[float], strength: float = 1.2):
        """
        Injects the byte impulse into the 4D void at the specified boundary port.
        Only particles within support distance h_port receive the impulse.
        """
        force, d_charge = self.compute_impulse(byte_val)

        for p in fluid_system.particles:
            r = math.sqrt(sum((p.pos[d] - port_pos[d]) ** 2 for d in range(4)))
            weight = quintic_kernel_4d(r, self.h_port)
            if weight > 0.0:
                # Add kinetic momentum
                for d in range(4):
                    p.vel[d] += strength * force[d] * weight * 0.1
                # Perturb semantic charge
                for c in range(fluid_system.dim_c):
                    p.charge[c] += strength * d_charge[c] * weight * 0.2

