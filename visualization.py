"""Utilities for visualizing convolution outputs
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def visualize_conv1d_gif(f, g, response, filepath):
    """Plots gif of convolution output over each timestep

    WARNING: This function can be very taxing on your computer

    Args:
        f (np.array): one of the signals
        g (np.array): the other signal
        response (np.array): convolution output of f and g
        filepath (string): filepath for saving the gif file
    """
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 8))

    # plot appearance
    fig.suptitle("Convolution of $f$ and $g$", size=20, y=0.95)
    for i, label in enumerate(("$f$", "$g$", "$f*g$")):
        axes[i].set_ylabel(label, size=15)
        axes[i].set_xlim(0, len(response))
        axes[i].set_xlim(0, len(response))
    axes[-1].set_xlabel("t")

    # pad input signals
    length = len(response)
    x = range(length)
    f_rem = length - len(f)
    g_rem = length - len(g)

    f_padding = (0, f_rem)
    g_padding = (g_rem, 0)

    f_pad = np.pad(f, f_padding)
    g_pad = np.pad(g[::-1], g_padding)

    # init animation
    f_line, = axes[0].plot(x, f_pad)
    g_line, = axes[1].plot(x, np.zeros(length))
    fg_line, = axes[-1].plot(x, np.zeros(length))
    lines = [f_line, g_line, fg_line]

    # plot convolution
    def animate(i):

        # translate g to the right for the gif animation
        g_subset = g_pad[-(i + 1):]
        if length - len(g_subset) >= 0:
            g_subset = np.pad(g_subset, (0, length - len(g_subset)))
        else:
            g_subset = g_subset[:length]
        lines[1].set_data(x, g_subset)

        # plot the response
        r_subset = response[:(i + 1)]
        r_padded = np.pad(r_subset, (0, length - len(r_subset)))
        lines[-1].set_data(x, r_padded)

        return lines

    axes[0].set_ylim(f.min(), f.max())
    axes[1].set_ylim(g.min(), g.max())
    axes[-1].set_ylim(response.min(), response.max())
    anim = animation.FuncAnimation(fig, animate, frames=np.arange(length), interval=40, blit=True)
    anim.save(filepath, dpi=150, writer="imagemagick")
    plt.close()


def visualize_conv2d(img, kernel, response):
    """Visualizes the convolution of an image with a kernel

    Args:
        img (np.ndarray): grayscale image
        kernel (np.ndarray): filter or kernel
        response (np.ndarray): convolution output
    """
    fig, axes = plt.subplots(
        nrows=1, 
        ncols=3, 
        gridspec_kw={
            'width_ratios': [4, 1, 4]
        },
        figsize=(10, 5)
    )
    plt.rcParams['image.cmap'] = 'bone'
        
    # plot images
    axes[0].set_title("Original Image")
    axes[0].imshow(img)
    axes[0].axis('off')
    
    axes[1].set_title("Filter")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].imshow(kernel, vmin=min(kernel.min(), 0), vmax=kernel.max())
    
    axes[2].set_title("Filtered Image")
    axes[2].imshow(response)
    axes[2].axis('off')
    
    plt.show()
    plt.close()

