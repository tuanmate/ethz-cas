import numpy as np

rng = np.random.default_rng()

def resample(particles, particles_w):
    #print("Resampling...", end='')
    #print(particles.shape)
    #print(particles_w.shape)
    #print("============")
    sampled_particles = rng.choice(particles, size=particles.shape[0], replace=True, p=particles_w.reshape(-1), axis=0)
    #print(sampled_particles)
    #print(sampled_particles.shape)
    #print("++++++++++++")

    #print("done")
    return sampled_particles, particles_w