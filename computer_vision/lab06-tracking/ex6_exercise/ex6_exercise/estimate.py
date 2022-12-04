import numpy as np

def estimate(particles, particles_w):
    print(particles.shape)
    print(particles_w.shape)
    return np.sum(particles * particles_w)