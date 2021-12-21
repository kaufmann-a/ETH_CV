import cv2
import os
import numpy as np
from color_histogram import color_histogram
from propagate import propagate
from estimate import estimate
from chi2_cost import chi2_cost
import scipy.stats

def observe(particles, frame, bbox_height, bbox_width, hist_bin, hist, sigma_observe):



    particles_w = np.zeros(particles.shape[0])
    for i in range(particles.shape[0]):
        xMin = min(max(0, round(particles[i, 0] - 0.5 * bbox_width)), frame.shape[1] - 1)
        yMin = min(max(0, round(particles[i, 1] - 0.5 * bbox_height)), frame.shape[0] - 1)
        xMax = min(max(0, round(particles[i, 0] + 0.5 * bbox_width)), frame.shape[1] - 1)
        yMax = min(max(0, round(particles[i, 1] + 0.5 * bbox_height)), frame.shape[0] - 1)
        hist_i = color_histogram(xMin, yMin, xMax, yMax, frame, hist_bin)

        chi = chi2_cost(hist, hist_i)
        particles_w[i] = 1/(np.sqrt(2*np.pi)*sigma_observe)*np.exp(- np.power(chi, 2)/(2*np.power(sigma_observe, 2)))
        # print(particles_w[i])
    return particles_w / sum(particles_w)



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
    particles_w = np.ones([300, 1]) * 1. / 300
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
