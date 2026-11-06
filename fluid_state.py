"""
Fluid State container for 4D Euclidean SPH particles in DHVN.
"""

from dataclasses import dataclass, field
from typing import List
import random
import math

@dataclass
class Particle:
    id: int
    pos: List[float]       # x_i in R^4
    vel: List[float]       # v_i in R^4
    acc: List[float]       # a_i in R^4
    charge: List[float]    # c_i in R^{d_c} (semantic state vector)
    density: float = 1.0   # rho_i
    pressure: float = 0.0  # P_i
    mass: float = 1.0      # m_i
    enstrophy: float = 0.0 # E_i (vorticity concentration)

class FluidSystem4D:
    def __init__(self, num_particles: int = 128, dim_c: int = 8, radius: float = 2.0, h: float = 0.6):
        self.num_particles = num_particles
        self.dim_c = dim_c
        self.radius = radius
        self.h = h
        self.particles: List[Particle] = []
        self._initialize_particles()

    def _initialize_particles(self):
        """Uniformly distributes particles inside a 4D hypersphere of radius self.radius."""
        random.seed(42)
        self.particles = []
        for i in range(self.num_particles):
            # Sample random point in 4D ball via normal distribution + radius scaling
            u = [random.gauss(0.0, 1.0) for _ in range(4)]
            norm = math.sqrt(sum(k * k for k in u))
            r = (random.random() ** (1.0 / 4.0)) * (self.radius * 0.85)
            pos = [(k / norm) * r for k in u]

            # Initial thermal velocities (low baseline jitter)
            vel = [random.gauss(0.0, 0.02) for _ in range(4)]
            acc = [0.0, 0.0, 0.0, 0.0]

            # Semantic charge initialization
            charge = [random.gauss(0.0, 0.01) for _ in range(self.dim_c)]

            p = Particle(
                id=i,
                pos=pos,
                vel=vel,
                acc=acc,
                charge=charge,
                density=1.0,
                pressure=0.0,
                mass=1.0,
                enstrophy=0.0
            )
            self.particles.append(p)

