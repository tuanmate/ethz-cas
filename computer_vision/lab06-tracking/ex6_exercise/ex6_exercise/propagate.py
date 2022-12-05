import numpy as np

def propagate(particles, frame_height, frame_width, params):
    #print("Propagating...", end='')

    n_particles = particles.shape[0]
    pos_noise = np.random.normal(0, params["sigma_position"], (n_particles, 2))
    if params["model"] == 0:
        # no motion
        A = np.array([[1,0],
                      [0,1]])
        noise = pos_noise
    else:
        # constant velocity
        A = np.array([[1,0,1,0,],
                      [0,1,0,1,],
                      [0,0,1,0,],
                      [0,0,0,1,]])
        vel_noise = np.random.normal(0, params["sigma_velocity"], (n_particles, 2))
        noise = np.concatenate((pos_noise, vel_noise), axis=1)
    
    #print("    A: ", A.shape)
    #print(" part: ", particles.shape)
    #print("noise: ", noise.shape)

    new_particles = np.dot(A, particles.transpose()).transpose() + noise
    #print(("++++++++++++++++++++++"))
    #print("New particles: ")
    #print(new_particles.transpose())
    new_particles[new_particles<0] = 0
    xs = new_particles[:,0] >= frame_width
    ys = new_particles[:,1] >= frame_height
    new_particles[xs, 0] = frame_width-1
    new_particles[ys, 1] = frame_height-1
    #print("Updated:")
    #print(new_particles.transpose())
    #print(("++++++++++++++++++++++"))

    #print("done") 
    return new_particles
    