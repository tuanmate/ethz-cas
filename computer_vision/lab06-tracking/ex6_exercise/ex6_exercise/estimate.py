import numpy as np

def estimate(particles, particles_w):
    est = np.sum(particles * particles_w, axis=0)
    return est