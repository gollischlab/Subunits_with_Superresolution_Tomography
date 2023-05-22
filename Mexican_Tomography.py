""" This script implements a tomographic presentation of bars with a Mexican
hat profile to detect subunits in a retinal ganglion cell model."""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from skimage.transform import iradon
from scipy.ndimage import gaussian_filter
import pathlib
import Subunit_Model


###############################################################################
# Global parameters
###############################################################################
SCENARIO = 'realistic gauss'            # Model subunit layout type.
NUM_SUBUNITS = 10                       # Model number of subunits.
RNG_SEED = 100                          # Model random seed used for generating the subunit layout (and potentially number).
SUBUNIT_NL = 'threshold-linear'         # Model nonlinearity of the subunits.
SYNAPTIC_WEIGHTS = 'equal'              # Model weights of the individual subunits.
RGC_NL = None                           # Model output nonlinearity.
RESOLUTION = 40                         # Width and height of the simulated area in pixels.
NUM_POSITIONS = 60                      # Number of bar positions.
NUM_ANGLES = 36                         # Number of bar angles.
HAT_STD = (RESOLUTION/40) * 2.5         # Standard deviation sigma of the bar profile. The prefactor makes it independent of the resolution.
SURROUND_FACTOR = 2.5                   # Factor that strengthens the suppressive surround of the stimulus.


###############################################################################
# Functions
###############################################################################
def Plot_stimulus(resolution, hat_std, surround_factor, savepath=None):
    """ Plot an examplary stimulus with the given characteristics.

    Parameters
    ----------
    resolution : int
        Resolution/size of the stimulus as measured along an edge.
    hat_std : float
        Standard deviation sigma of the Mexican bar profile.
    surround_factor : float
        Multiplicative factor applied to the sidebands of the Mexican bar.
    savepath : string, optional
        If provided, the plot is not shown in terminal but saved at the given
        location.
    """

    stimulus = Create_stimulus(resolution, hat_std, 0, resolution/2,
                               surround_factor)
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.suptitle("Stimulus")
    ax.imshow(np.transpose(stimulus), origin='lower',
              cmap='gray', vmin=-1, vmax=1)
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, dpi=300)
    plt.clf()
    plt.close('all')


def Plot_sinogram(sinogram, savepath=None):
    """ Plot a sinogram.

    Parameters
    ----------
    sinogram : ndarray
        2D array containing the sinogram. First index denotes position of the
        bar, second denotes angle of the bar.
    savepath : string, optional
        If provided, the plot is not shown in terminal but saved at the given
        location.
    """

    num_angles = sinogram.shape[1]
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.suptitle("Sinogram")
    ax.set_xlabel("Position")
    ax.set_ylabel("Angle")
    ax.set_yticks([0, num_angles/4, num_angles/2, num_angles*3/4])
    ax.set_yticklabels(["0°", "45°", "90°", "135°"],
                       rotation='vertical', va='center')
    im = ax.imshow(np.transpose(sinogram), origin='lower', aspect='auto')
    cax = inset_axes(ax,
                     width="2%",
                     height="100%",
                     loc='lower left',
                     bbox_to_anchor=(1.02, 0., 1, 1),
                     bbox_transform=ax.transAxes,
                     borderpad=0)
    cbar = fig.colorbar(im, cax=cax,
                        ticks=[np.nanmin(sinogram), np.nanmax(sinogram)])
    cbar.ax.tick_params(labelsize=5)
    cbar.ax.set_yticklabels([f"{np.nanmin(sinogram):.3g}",
                             f"{np.nanmax(sinogram):.3g}"])
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, dpi=300)
    plt.clf()
    plt.close('all')


def Plot_reconstruction(reconstruction, savepath=None):
    """ Plot a reconstruction of the subunit layout.

    Parameters
    ----------
    reconstruction : ndarray
        2D array containing the reconstruction.
    savepath : string, optional
        If provided, the plot is not shown in terminal but saved at the given
        location.
    """

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.suptitle("Reconstruction")
    ax.imshow(np.transpose(reconstruction), origin='lower')
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, dpi=300)
    plt.clf()
    plt.close('all')


def Mexican_hat(pos, std, surround_factor):
    """ Compute the value of a Mexican hat at the given location.

    Hat is centered at 0 and normalized to a maximum value of 1.

    Parameters
    ----------
    pos : float
        Location at which the Mexican Hat is evaluated.
    std : float
        Standard deviation sigma of the Hat.
    surround_factor : float, optional
        Multiplicative factor applied to the surround part of the hat. The
        resulting values of the hat will be clipped to the range [-1, 1].

    Returns
    -------
    float
        Value of the Mexican Hat at *pos*.
    """

    value = (1-np.square(pos/std)) * np.exp(-np.square(pos/std)/2)
    value[value <= 0] *= surround_factor
    value[value < -1] = -1
    value[value > 1] = 1

    return value


def Create_stimulus(resolution, std, angle, position, surround_factor):
    """ Return a Mexican bar stimulus with the given specifications.

    Parameters
    ----------
    resolution : int
        Size of the image as measured along an edge.
    std : float
        Standard deviation of the Mexican bar profile.
    angle : float
        Rotational angle of the bar in radiance. 0 yields a vertical bar,
        higher values turn the bar clockwise.
    position : int
        Defines the position of the bar. Should be between 0 (inclusive) and
        *resolution* (exclusive). Low values mean bar is placed at the left
        (for a vertical bar).
    surround_factor : float
        Multiplicative factor applied to the sidebands of the Mexican bar.

    Returns
    -------
    ndarray
        2D containing the stimulus.
    """

    #  1st index of any array is x-coordinate, 2nd is y-coordinate
    yy, xx = np.array(np.meshgrid(np.arange(-resolution/2, resolution/2) + 0.5,
                                  np.arange(-resolution/2, resolution/2) + 0.5))
    # Applying the rotation
    xx_rot = np.cos(angle) * xx - np.sin(angle) * yy
    # Applying the shift
    xx_rot += resolution/2 - 0.5 - position
    # Calculating the rotated and shifted image
    image = Mexican_hat(xx_rot, std, surround_factor)

    return image


def Measure_sinogram(rgc, num_positions, num_angles, hat_std, surround_factor):
    """ Compute the sinogram of the given cell using stimuli with the given
    properties.

    Parameters
    ----------
    rgc : Subunit_Model
        Object of the class Subunit_Model from Subunit_Model.py that is
        investigated.
    num_positions : int
        Number of bar positions to be measured.
    num_angles : int
        Number of bar angles to be measured.
    hat_std : float
        Standard deviation sigma of the Mexican bar profile.
    surround_factor : float
        Multiplicative factor applied to the sidebands of the Mexican bar.

    Returns
    -------
    ndarray
        2D array containing the sinogram. First index denotes position of the
        bar, second denotes angle of the bar.
    """

    resolution = rgc.resolution
    responses = np.empty((num_positions, num_angles))
    for counter_a, angle in enumerate(np.linspace(0, np.pi, num=num_angles,
                                                  endpoint=False)):
        for counter_p, position in enumerate(np.linspace(0, resolution,
                                                         num=num_positions,
                                                         endpoint=False)):
            stimulus = Create_stimulus(resolution, hat_std, angle, position,
                                       surround_factor)
            responses[counter_p, counter_a] = rgc.response_to_flash(stimulus)

    return responses


###############################################################################
# Main program
###############################################################################
# Creating a folder for the plots
pathlib.Path("Mexican Tomography").mkdir(parents=True, exist_ok=True)

# Set the model up
rgc = Subunit_Model.Subunit_Model(resolution=RESOLUTION, scenario=SCENARIO,
                                  subunit_nonlinearity=SUBUNIT_NL,
                                  subunit_weights=SYNAPTIC_WEIGHTS,
                                  rgc_nonlinearity=RGC_NL, rgc_spiking=None,
                                  num_subunits=NUM_SUBUNITS, rng_seed=RNG_SEED)
rgc.plot_subunit_ellipses("Mexican Tomography/1 Subunit Layout")
rgc.plot_receptive_field("Mexican Tomography/2 Receptive Field")

# Plotting an exemplary stimulus
Plot_stimulus(RESOLUTION, HAT_STD, SURROUND_FACTOR,
              "Mexican Tomography/3 Stimulus")

# Doing the tomographic scan
sinogram = Measure_sinogram(rgc, NUM_POSITIONS, NUM_ANGLES, HAT_STD,
                            SURROUND_FACTOR)
Plot_sinogram(sinogram, "Mexican Tomography/4 Sinogram")

# Doing the same for a spiking neuron
rgc.set_spiking('poisson', coefficient='realistic')
sinogram_sp = Measure_sinogram(rgc, NUM_POSITIONS, NUM_ANGLES, HAT_STD,
                               SURROUND_FACTOR)
sinogram_sp_f = gaussian_filter(sinogram_sp, sigma=((NUM_POSITIONS/60)*2,
                                                    (NUM_ANGLES/36)*0.5))
Plot_sinogram(sinogram_sp, "Mexican Tomography/5 Sinogram Spiking")
Plot_sinogram(sinogram_sp_f, "Mexican Tomography/6 Sinogram Spiking Filtered")

# Reconstructing using filtered back projection (transposing due to different
# angle conventions)
fbp = np.transpose(iradon(sinogram, circle=True, filter_name='hamming'))
fbp_sp = np.transpose(iradon(sinogram_sp, circle=True, filter_name='hamming'))
fbp_sp_f = np.transpose(iradon(sinogram_sp_f, circle=True, filter_name='hamming'))

# Plotting the FBPs
Plot_reconstruction(fbp, "Mexican Tomography/7 FBP")
Plot_reconstruction(fbp_sp, "Mexican Tomography/8 FBP Spiking")
Plot_reconstruction(fbp_sp_f, "Mexican Tomography/9 FBP Spiking Filtered")
