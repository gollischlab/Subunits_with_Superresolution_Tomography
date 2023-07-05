""" This script implements a tomographic presentation of bars with a Mexican
hat profile to detect subunits in a retinal ganglion cell model."""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from skimage.transform import iradon
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
import pathlib
import Subunit_Model


###############################################################################
# Global parameters
###############################################################################
SCENARIO = 'realistic gauss'            # Model subunit layout type.
NUM_SUBUNITS = 10                       # Model number of subunits.
LAYOUT_SEED = 100                       # Model random seed used for generating the subunit layout (and potentially number).
SUBUNIT_NL = 'threshold-linear'         # Model nonlinearity of the subunits.
SYNAPTIC_WEIGHTS = 'equal'              # Model weights of the individual subunits.
RGC_NL = None                           # Model output nonlinearity.
POISSON_SEED = None                     # Model random seed used for generating spikes in the poisson process.
RESOLUTION = 40                         # Width and height of the simulated area in pixels.
NUM_POSITIONS = 60                      # Number of bar positions.
NUM_ANGLES = 36                         # Number of bar angles.
HAT_STD = (RESOLUTION/40) * 2.5         # Standard deviation sigma of the bar profile. The prefactor makes it independent of the resolution.
SURROUND_FACTOR = 2.0                   # Factor that strengthens the suppressive surround of the stimulus.
SMOOTHING = (NUM_POSITIONS/60 * 1.5,    # Gaussian sigma for smoothing the sinograms. First value is in position-,
             NUM_ANGLES/36 * 1.0)       # second in angle-direction. In units of sinogram-pixels.


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


def Plot_reconstruction(reconstruction, savepath=None, coordinates=None,
                        f_score=None):
    """ Plot a reconstruction of the subunit layout.

    Parameters
    ----------
    reconstruction : ndarray
        2D array containing the reconstruction.
    savepath : string, optional
        If provided, the plot is not shown in terminal but saved at the given
        location.
    coordinates : ndarray, optional
        If provided, the coordinates contained in this 2D array are marked in
        the plot.
    f_score : float, optional
        If provided, the F-score will be written in the plot.
    """

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.suptitle("Reconstruction")
    ax.imshow(np.transpose(reconstruction), origin='lower')
    if coordinates is not None:
        ax.scatter(coordinates[:, 0], coordinates[:, 1], c='red', marker='x')
    if f_score is not None:
        fig.text(0.05, 0.01, f"F-score of hotspots: {f_score:.2f}", size=6)
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


def Reconstruction(sinogram, sigma, return_smoothed=False):
    """ Reconstructs the subunit layout from the given sinogram using filtered
    back-projection (FBP).

    Parameters
    ----------
    sinogram : ndarray
        2D array containing the sinogram. First index denotes position of the
        bar, second denotes angle of the bar.
    sigma : tuple
        Contains the standard deviations of the Gaussian smoothing of the
        sinogram in position-direction and in angle-direction in units of
        elements.
    return_smoothed : bool, optional
        If True, returns the smoothed sinogram.

    Returns
    -------
    ndarray
        2D array containing the FBP.
    ndarray, optional
        2D array containing the smoothed sinogram. Only provided if
        return_smoothed is True.
    """

    smoothed = gaussian_filter(sinogram, sigma=sigma)
    # Transposing the FBP due to different angle conventions
    fbp = np.transpose(iradon(smoothed, circle=True))

    if return_smoothed:
        return fbp, smoothed
    else:
        return fbp


def Find_hotspots(fbp):
    """ Localizes hotspots in the FBP by finding local maxima higher than 30%
    of the global maximum.

    Parameters
    ----------
    fbp : ndarray
        2D array containing the FBP.

    Returns
    -------
    ndarray
        2D array containing the x- and y-coordinates of all identified
        hotspots.
    """

    global_max = np.max(fbp)
    coordinates = peak_local_max(fbp, min_distance=1)
    coordinates = [coord for coord in coordinates
                     if fbp[coord[0], coord[1]] >= 0.3*global_max]
    coordinates = np.array(coordinates)

    return coordinates


def F_score(rgc, locations, num_positions):
    """ Calculates the F-score for the detected subunit locations.

    Detected subunits are determined by checking if a detected location falls
    within the 0.75 sigma ellipse of the real subunit.

    Parameters
    ----------
    rgc : Subunit_Model
        Object of the class Subunit_Model from Subunit_Model.py that is
        investigated.
    locations : ndarray
        2D array containing the x- and y-coordinates of all locations that are
        tested against the subunit locations.
    num_positions : int
        Number of bar positions that were measured. This is equal to the edge
        size of the reconstruction.

    Returns
    -------
    float
        F-score of *locations* as detections of subunits in *rgc*.

    """

    # First create subunit arrays with the parameters of the rgc's subunits,
    # but with the array size of the reconstruction, i.e. scaled versions of
    # rgc.subunits.
    scaling = num_positions / rgc.resolution
    subunits = [Subunit_Model.Gaussian_array(num_positions,
                                             params[0]*scaling,
                                             params[1]*scaling,
                                             params[2]*scaling,
                                             params[3]*scaling,
                                             params[4])
                for params in rgc.subunit_params]
    subunits = np.array(subunits)
    # Calculate the amplitude of the subunit gaussians.
    amplitudes = 1/(2*np.pi*rgc.subunit_params[:, 2]*rgc.subunit_params[:, 3]
                    *scaling**2)
    # Calculate what the values of Gaussian with those amplitudes at 0.75 sigma
    # are.
    thresholds = np.exp((-0.75**2)/2) * amplitudes
    # Convert the subunit arrays into arrays that are true at all locations
    # within 0.75 sigma.
    within = np.transpose(np.transpose(subunits) >= thresholds)
    # Make sure that the 0.75 sigma ellipses don't overlap
    if np.max(np.sum(within, axis=0) > 1):
        raise Exception("Subunits overlap too strongly to calculate F-score.")
    # Use the locations as indices to find out which subunits each location has
    # hit
    detections_idxb = [within[:, loc[0], loc[1]] for loc in locations]
    # Count which subunits have been detected while avoiding redundant
    # detections
    true_positives = np.sum(np.any(np.array(detections_idxb), axis=0))
    # Calculate the F-score
    f_score = 2*true_positives/(locations.shape[0] + rgc.num_subunits)

    return f_score


def Tomographic_analysis(rgc, num_positions, num_angles, hat_std,
                         surround_factor, smoothing, known_sinogram=None):
    """ Using the tomographic method to analyze a model cell.

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
    smoothing : tuple
        Contains the standard deviations of the Gaussian smoothing of the
        sinogram in position-direction and in angle-direction in units of
        elements.
    known_sinogram : ndarray, optional
        If the sinogram of the cell is already known and measuring it again
        should be avoided (e.g. for performance reasons), the sinogram can be
        passed here and only the rest of the analysis will be performed.
        *known_sinogram* should the be like *sinogram* and will also be
        returned as such.

    Returns
    -------
    sinogram : ndarray
        2D array containing the sinogram. First index denotes position of the
        bar, second denotes angle of the bar.
    smoothed : ndarray
        2D array containing the smoothed sinogram.
    fbp : ndarray
        2D array containing the FBP.
    hotspots : ndarray
        Coordinates of the hotspots in the FBP.
    f_score : float
        F-score of how well the detected hotspots correspond to the subunits of
        the model.
    """

    if known_sinogram is None:
        sinogram = Measure_sinogram(rgc, num_positions, num_angles, hat_std,
                                    surround_factor)
    else:
        sinogram = known_sinogram
    fbp, smoothed = Reconstruction(sinogram, smoothing, return_smoothed=True)
    hotspots = Find_hotspots(fbp)
    f_score = F_score(rgc, hotspots, num_positions)

    return sinogram, smoothed, fbp, hotspots, f_score

###############################################################################
# Main program
###############################################################################
if __name__ == '__main__':

    # Creating a folder for the plots
    pathlib.Path("Mexican Tomography").mkdir(parents=True, exist_ok=True)

    # Set the model up
    rgc = Subunit_Model.Subunit_Model(resolution=RESOLUTION, scenario=SCENARIO,
                                      subunit_nonlinearity=SUBUNIT_NL,
                                      subunit_weights=SYNAPTIC_WEIGHTS,
                                      rgc_nonlinearity=RGC_NL,
                                      rgc_spiking=None,
                                      num_subunits=NUM_SUBUNITS,
                                      layout_seed=LAYOUT_SEED)

    # Applying the tomographic method
    temp = Tomographic_analysis(rgc, NUM_POSITIONS, NUM_ANGLES,
                                HAT_STD, SURROUND_FACTOR, (0, 0))
    sinogram, _, fbp, hotspots, f_score = temp

    # Doing the same for a spiking neuron with and without smoothing
    rgc.set_spiking('poisson', spiking_coefficient='realistic',
                    poisson_seed=POISSON_SEED)
    temp = Tomographic_analysis(rgc, NUM_POSITIONS, NUM_ANGLES, HAT_STD,
                                SURROUND_FACTOR,
                                (0, 0))
    sinogram_sp, _, fbp_sp, hotspots_sp, f_score_sp = temp
    temp = Tomographic_analysis(rgc, NUM_POSITIONS, NUM_ANGLES, HAT_STD,
                                SURROUND_FACTOR,
                                SMOOTHING)
    _, sinogram_sp_f, fbp_sp_f, hotspots_sp_f, f_score_sp_f = temp

    # Plots
    rgc.plot_subunit_ellipses("Mexican Tomography/1 Subunit Layout")
    rgc.plot_receptive_field("Mexican Tomography/2 Receptive Field")
    Plot_stimulus(RESOLUTION, HAT_STD, SURROUND_FACTOR,
                  "Mexican Tomography/3 Stimulus")
    Plot_sinogram(sinogram, "Mexican Tomography/4 Sinogram")
    Plot_sinogram(sinogram_sp, "Mexican Tomography/5 Sinogram Spiking")
    Plot_sinogram(sinogram_sp_f,
                  "Mexican Tomography/6 Sinogram Spiking Filtered")
    Plot_reconstruction(fbp, "Mexican Tomography/7 FBP", coordinates=hotspots,
                        f_score=f_score)
    Plot_reconstruction(fbp_sp, "Mexican Tomography/8 FBP Spiking",
                        coordinates=hotspots_sp, f_score=f_score_sp)
    Plot_reconstruction(fbp_sp_f, "Mexican Tomography/9 FBP Spiking Filtered",
                        coordinates=hotspots_sp_f, f_score=f_score_sp_f)
