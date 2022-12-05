import numpy as np

def estimate(particles, particles_w):
    #print("Estimating...", end='')
    est = np.sum(particles * particles_w, axis=0)
    #print("done")
    if(False):
        print("DEBUG-------------------------")
        print(particles)
        print(particles_w)
        print(est)
        print(particles.shape, particles_w.shape, est.shape, np.multiply(particles, particles_w).shape, (particles*particles_w).shape)
        print("DEBUG-------------------------")
    return est