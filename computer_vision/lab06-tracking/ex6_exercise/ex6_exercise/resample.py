import numpy as np

rng = np.random.default_rng()

def resample(particles, particles_w):
    p = np.concatenate((particles, particles_w), axis=1)
    p_new = rng.choice(p, size=particles.shape[0], replace=True, p=particles_w.reshape(-1), axis=0)
    n_dim = particles.shape[1]
    sampled_particles = p_new[:, :n_dim]
    sampled_weights = p_new[:, n_dim]
    sampled_weights = sampled_weights / np.sum(sampled_weights)
    sampled_weights = sampled_weights.reshape(-1,1)

    return sampled_particles, sampled_weights