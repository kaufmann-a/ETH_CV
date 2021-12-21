import cv2
import os
import numpy as np
from color_histogram import color_histogram
from propagate import propagate
from estimate import estimate
from observe import observe
from chi2_cost import chi2_cost
import scipy.stats

def resample(particles, particles_w):
    list_range = list(range(particles.shape[0]))
    rearanged_idx = np.random.choice(list_range, size=particles.shape[0], replace=True, p=particles_w)
    return particles[rearanged_idx, :], particles_w[rearanged_idx]/sum(particles_w[rearanged_idx])


if __name__ == "__main__":
    model = 0
    params = {
        "draw_plots": 1,
        "hist_bin": 16,
        "alpha": 0,
        "sigma_observe": 0.1,
        "model": 0,
        "num_particles": 300,
        "sigma_position": 15,
        "sigma_velocity": 1,
        "initial_velocity": (1, 10)
    }

    video_name = "video3.avi"
    data_dir = './data/'
    video_path = os.path.join(data_dir, video_name)
    first_frame = 1
    last_frame = 60
    vidcap = cv2.VideoCapture(video_path)
    vidcap.set(1, first_frame)
    ret, first_image = vidcap.read()
    first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
    hist = color_histogram(30, 30, 60, 60, first_image, 16)
    state_length = 2
    frame_height = first_image.shape[0]
    frame_width = first_image.shape[1]


    state_length = 2
    if model == 1:
        state_length = 4

    mean_state_a_priori = np.zeros([last_frame - first_frame + 1, state_length])
    mean_state_a_posteriori = np.zeros([last_frame - first_frame + 1, state_length])
    # bounding box centre
    mean_state_a_priori[0, 0:2] = [(30 + 60)/2., (30 + 60)/2.]

    if model == 1:
        # use initial velocity
        mean_state_a_priori[0, 2:4] = (1, 10)

    # Initialize Particles
    particles = np.tile(mean_state_a_priori[0], (300, 1))
    particles_w = np.ones((300,)) * 1. / 300
    print("end")
    for i in range(last_frame - first_frame + 1):
        t = i + first_frame

        # Propagate particles
        # === Implement function propagate() ===
        particles = propagate(particles, frame_height, frame_width, params)
        mean_state_a_priori[i, :] = estimate(particles, particles_w)
        ret, frame = vidcap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    particles_w = observe(particles, frame, 50, 50, params["hist_bin"], hist, params["sigma_observe"])

    # Update estimation
    mean_state_a_posteriori[i, :] = estimate(particles, particles_w)

    # Update histogram color model
    hist_crrent = color_histogram(min(max(0, round(mean_state_a_posteriori[i, 0] - 0.5 * 60)), frame_width - 1),
                                  min(max(0, round(mean_state_a_posteriori[i, 1] - 0.5 * 60)), frame_height - 1),
                                  min(max(0, round(mean_state_a_posteriori[i, 0] + 0.5 * 60)), frame_width - 1),
                                  min(max(0, round(mean_state_a_posteriori[i, 1] + 0.5 * 60)), frame_height - 1),
                                  frame, params["hist_bin"])

    hist = (1 - params["alpha"]) * hist + params["alpha"] * hist_crrent



    # RESAMPLE PARTICLES
    # === Implement function resample() ===
    particles, particles_w = resample(particles, particles_w)
    # =====================================