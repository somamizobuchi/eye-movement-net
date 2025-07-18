import numpy as np
import os
from tqdm import tqdm
from utils import implay
from fixation_utils import pink_noise_gray_image
from em_utils import generate_brownian_motion, generate_saccade
from scipy.stats import gamma
from scipy.ndimage import convolve1d
from skimage.transform import rescale
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import subprocess
import os
from utils import generate_rgc_impulse_response, generate_rgc_spatial_rf
import torch.nn.functional as F
import torch

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def save_array_as_mp4_opencv(array_3d, output_path, fps=30):
    """
    Save 3D float array as MP4 using OpenCV
    
    Parameters:
    array_3d: numpy array of shape (frames, height, width) with float values
    output_path: string, path to save the MP4 file
    fps: int, frames per second
    """
    # Normalize array to 0-255 range
    array_normalized = ((array_3d - array_3d.min()) / 
                       (array_3d.max() - array_3d.min()) * 255).astype(np.uint8)
    
    frames, height, width = array_normalized.shape
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    
    # Write frames
    for i in range(frames):
        out.write(array_normalized[i])
    
    out.release()
    print(f"Video saved as {output_path} using OpenCV")

if __name__ == "__main__":
    # Parameters
    num_movies = 1  # Number of images to generate
    NX = 128  # Size of each image
    NT = 1024
    D = 20 / 3600
    alpha = 1.0  # Pink noise parameter
    fs = 240
    ppd = 30
    l_filt = 24

    NT_PAD = NT + l_filt - 1

    generate_video_file = True  # Whether or not to save an mp4 of a sample video
    save_data = False            # Whether or not to save npy file containing all data
    plot_figures = False
    play_video = True

    cell_type = "M"
    eccentricity = 1.0

    _, _, hx = generate_rgc_spatial_rf(cell_type=cell_type, eccentricity=eccentricity, resolution=ppd, size_samples=24)
    _, ht = generate_rgc_impulse_response(cell_type=cell_type, num_samples=l_filt, fs=fs)


    fig, ax = plt.subplots(2, 1)
    ax[0].imshow(hx, cmap="viridis")
    ax[0].axis("off")
    ax[1].plot(ht)
    fig.savefig("figures/filters.png")



    # Fixation duration distr params
    alpha = 1.5
    beta = 0.08

    alpha_sacc = 1.41
    beta_sacc = 4.87


    imsize = 2048

    input = np.zeros([num_movies, NT, NX, NX])
    output = np.zeros([num_movies, NT, NX, NX])
    em = []
    fixation_durations = []
    saccade_amps = []

    em_carryover = False
    # Generate and save images
    for i in tqdm(range(num_movies), desc="Generating eye movement videos"):
        # Generate image every 10 EMs
        if i % 10 == 0:
            img = pink_noise_gray_image(imsize)
            img_filt = cv2.filter2D(img, -1, hx)

        if not em_carryover:
            # Choose a random point in image to start
            eye_idx = np.random.randint((NX/2, imsize-3*NX/2), size=(2, 1))

        while eye_idx.shape[1] < NT_PAD:
            # Generate drift
            while True:
                drift_dur = gamma.rvs(alpha, loc=0, scale=beta)
                drift_samples = (drift_dur * fs).astype(int)
                if drift_samples < 1:
                    continue
                drift = generate_brownian_motion(D, fs, drift_samples)
                drift = drift * ppd + eye_idx[:,-1:]
                if np.min(drift) >= 0 and np.max(drift + NX) < imsize:
                    break; 
            fixation_durations.append(drift_dur)
            eye_idx = np.concat((eye_idx, drift.round().astype(int)), axis=1)

            # Generate saccade
            a = saccade_amp = gamma.rvs(alpha_sacc, scale=beta_sacc)
            saccade_amps.append(a)
            while True:
                theta = np.random.rand() * 360.0
                t, sacc_x, sacc_y, _ = generate_saccade(a, theta, fs)
                sacc = np.stack((sacc_x, sacc_y))
                sacc = sacc * ppd + eye_idx[:,-1:]
                if np.min(sacc) >= 0 and np.max(sacc + NX) < imsize:
                    break; 

            eye_idx = np.concat((eye_idx, sacc.round().astype(int)), axis=1)

        # Trim to fit and save carryover
        if eye_idx.shape[1] > NT_PAD:
            eye_res = eye_idx[:,NT_PAD:]
            eye_idx = eye_idx[:,:NT_PAD:]
            em_carryover = True
        else:
            em_carryover = False

        em.append(eye_idx)

        input_tmp = np.zeros([NT_PAD, NX, NX])
        output_tmp = np.zeros([NT_PAD, NX, NX])

        for fi in range(NT_PAD):
            input_tmp[fi,:,:] = img[eye_idx[1,fi]:eye_idx[1,fi]+NX, eye_idx[0,fi]:eye_idx[0,fi]+NX]
            output_tmp[fi,:,:] = img_filt[eye_idx[1,fi]:eye_idx[1,fi]+NX, eye_idx[0,fi]:eye_idx[0,fi]+NX]

        input[i,:] = input_tmp[l_filt//2-1:-l_filt//2,:]
        output[i,:] = F.softplus(torch.tensor(convolve1d(output_tmp, ht, 0)[:NT,:])).numpy()
        
        if em_carryover:
            eye_idx = eye_res
    

    if plot_figures:
        # Create a figure with 2 rows and 1 column of subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        x_drift_dur = np.linspace(0, 1, 1000)
        p_drift_dur = gamma.pdf(x_drift_dur, alpha, scale=beta)
        x_sacc_amp = np.linspace(0, 25, 1000)
        p_sacc_amp = gamma.pdf(x_sacc_amp, alpha_sacc, scale=beta_sacc)


        # Plot fixation duration distr
        ax1.plot(x_drift_dur, p_drift_dur, 'b-', linewidth=2, label="Theoretical")
        ax1.hist(fixation_durations, bins=20, density=True, label="Data")
        ax1.set_title(f"Drift duration distribution (alpha={alpha}, beta={beta})")
        ax1.set_xlabel('Drift duration [s]')
        ax1.set_ylabel('Probability')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot data on the second subplot
        ax2.plot(x_sacc_amp, p_sacc_amp, 'r-', linewidth=2, label="Theoretical")
        ax2.hist(saccade_amps, bins=20, density=True, label="Data")
        ax2.set_title(f"Saccade amplitude distribution (alpha={alpha_sacc}, beta={beta_sacc})")
        ax2.set_xlabel('Saccade amplitude [deg]')
        ax2.set_ylabel('Probability')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Adjust spacing between subplots
        plt.tight_layout()

        # Display the plot
        # plt.show()

        fig.savefig("figures/distributions.png")


    if plot_figures:
        fig = plt.figure()

        t = np.arange(0, NT) / fs

        axl = plt.gca()
        axr = axl.twinx()

        xline, = axl.plot(t, em[-1][0,:], color='blue', label="x")
        yline, = axr.plot(t, em[-1][1,:], color='red', label="y")

        axl.set_xlabel("Time [s]")
        axl.set_ylabel("Position Horizontal [px]")
        axr.set_ylabel("Position Vertical [px]")

        # Combine lines and labels
        lines = [xline, yline]
        labels = [line.get_label() for line in lines]

        # Create a single legend
        plt.legend(lines, labels)
        # plt.show()
        fig.savefig("figures/sample_trace.png")

        fig = plt.figure()
        plt.imshow(img, cmap='gray')
        plt.plot(em[-1][0,:], em[-1][1,:], color='red')
        plt.tight_layout()
        rect = Rectangle((em[-1][0,0], em[-1][1,0]), NX, NX, linewidth=1, edgecolor='w', facecolor='none')
        plt.gca().add_patch(rect)

        fig.savefig("figures/trace_image_overlay.png")


    if play_video:
        implay(input[-1].transpose(1, 2, 0), interval=10, repeat=True)
        # implay(output[-1].transpose(1, 2, 0), interval=10, repeat=True)

    if generate_video_file:
        out_vid = np.concat((NormalizeData(input[-1]), NormalizeData(output[-1])), axis=2)
        input_file = "data/retinal_input_filt.mp4"
        output_file = "data/retinal_input_filt_web.mp4"
        save_array_as_mp4_opencv(out_vid, input_file, fps=30)

        # Convert codec
        if not os.path.exists(input_file):
            print(f"Error: Input file not found at '{input_file}'")

        command = [
            "ffmpeg",
            "-i", input_file,
            "-c:v", "libx264",
            "-y",
            output_file
        ]
        subprocess.run(command, check=True) # check=True will raise CalledProcessError on non-zero exit code

    if save_data:
        np.save("data/em_videos_raw.npy", input)
        np.save("data/em_videos_filt.npy", output)
        print("Done! All images generated and saved.")

