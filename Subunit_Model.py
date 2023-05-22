""" This script provides a class for subunit models"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy.optimize as opt


###############################################################################
# Dictionary storing the default parameters of the class
###############################################################################
DEFAULT_PARAMS = {"num_subunits" : None,                            # Planned number of subunits of the model. Note that this might be overruled by the subunit scenario. If None, this option has no effect.
                  "rng_seed" : None,                                # Non-negative seed of the random number generator in the subunit scenario 'realistic gauss'.  If None, a random seed is used.
                  "overlap_factor" : 1.35,                          # Regulates the size of the subunits in the scenario 'realistic gauss' without changing their spacing, thereby adjusting their overlap. Note that the value is not related to any meaningful measure.
                  "irregularity" : 3,                               # Determines how strongly the subunit layout in the scenario 'realistic gauss' deviates from a hexagonal grid.
                  "swap_gauss_for_cosine" : False,                  # If True, subunit layouts which normally use Gaussian subunits, will instead use cosine-shaped subunits.
                  "weights_gauss_std" : 0.12,                       # Standard deviation of the Gaussian used to set the subunit weights if they are option 'gauss'. In units of *resolution*, i.e. simulation area size.
                  "poisson_coefficient" : 'realistic'}              # If the spiking process is set to 'poisson', this defines the coefficient between the RGC response and the expected value of the Poisson distribution. 'realistic' means a realistic coefficient based on the response to locally sparse noise is calculated.


###############################################################################
# Helper functions
###############################################################################
def Circle(resolution, center, radius):
    """ Create a circle in an ndarray.

    Parameters
    ----------
    resolution : int
        Width and height of the array.
    center : (float, float)
        X- and y-position of the center of the circle (counting from 0).
    radius : float
        Radius of the circle.

    Returns
    -------
    circle : ndarray
        2D array containing the circle. Entries 1 mean circle, 0 mean no
        circle.
    """

    xx, yy = np.mgrid[:resolution, :resolution]
    distance_squared = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    circle = (distance_squared <= (radius ** 2))

    return circle.astype(int)


def Gaussian_array(resolution, x0, y0, sigma_x, sigma_y, theta):
    """ Create a Gaussian distribution in an ndarray.

    Gaussian is normalised to a volume of 1.

    Parameters
    ----------
    resolution : int
        Width and height of the array.
    x0 : float
        X-position of the center of the Gaussian.
    y0 : float
        Y-position of the center of the Gaussian.
    sigma_x : float
        Standard deviation in x-direction.
    sigma_y : float
        Standard deviation in y-direction.
    theta : float
        Rotation angle in radians.

    Returns
    -------
    gaussian : ndarray
        2D array containing the Gaussian.
    """

    xx, yy = np.mgrid[:resolution, :resolution]
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    amplitude = 1/(2*np.pi*sigma_x*sigma_y)
    gaussian = amplitude*np.exp(-(a*((xx-x0)**2) + 2*b*(xx-x0)*(yy-y0)
                                  + c*((yy-y0)**2)))

    return gaussian


def Cosine_spot_array(resolution, x0, y0, size_x, size_y, theta):
    """ Create an elliptical spot with a cosine profile in an ndarray.

    Spot is normalised to a volume of 1.

    Parameters
    ----------
    resolution : int
        Width and height of the array.
    x0 : float
        X-position of the center of the Spot.
    y0 : float
        Y-position of the center of the Spot.
    size_x : float
        For convenience, this is not the radius in x-direction of the cosine
        spot, but the standard deviation in x-direction a Gaussian would have
        if it was fitted to the spot. This makes this function more analagous
        in use to *Gaussian_array()*.
    size_y : float
        Same as *size_x*, but in y-direction.
    theta : float
        Rotation angle in radians.

    Returns
    -------
    ndarray
        2D array containing the spot.
    """

    xx, yy = np.mgrid[:resolution, :resolution]
    xx = xx - x0
    yy = yy - y0
    xx, yy = (np.cos(theta)*xx - np.sin(theta)*yy,
              np.sin(theta)*xx + np.cos(theta)*yy)
    xx /= 2.22*size_x
    yy /= 2.22*size_y
    rr = np.linalg.norm(np.stack([xx, yy]), axis=0)
    in_idxb = (rr <= 1)
    spot = np.zeros((resolution, resolution))
    spot[in_idxb] = np.cos(rr[in_idxb]*(np.pi/2))
    spot /= np.sum(spot)
    return spot


def TwoD_Gaussian(data_tuple, x0, y0, sigma_x, sigma_y, theta):
    """ Calculate the values of a 2D Gaussian with the specified parameters at
    the given positions.

    Gaussian is normalised to a volume of 1.

    Parameters
    ----------
    data_tuple : ndarray
        Locations at which the Gaussian should be evaluated. 2D with first
        column denoting x and second column denoting y.
    x0 : float
        X-position of the center of the Gaussian.
    y0 : float
        Y-position of the center of the Gaussian.
    sigma_x : float
        Standard deviation in x-direction.
    sigma_y : float
        Standard deviation in y-direction.
    theta : float
        Rotation angle in radians.

    Returns
    -------
    ndarray
        1D array of the values of the gaussian at the specified positions.
    """

    x = data_tuple[:, 0]
    y = data_tuple[:, 1]
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    amplitude = 1/(2*np.pi*sigma_x*sigma_y)
    g = amplitude*np.exp(-(a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2)))

    return g.ravel()


###############################################################################
# Functions for the subunit scenarios
###############################################################################
def Scenario_basic_squares(resolution):
    """ Create a 3D-array containing subunits according to scenario
    'basic squares'.

    'basic sqaures' is a 2x2 layout of quadratic non-overlapping subunits.

    Parameters
    ----------
    resolution : int
        Width and height of the square receptive area of the RGC in pixels.

    Returns
    -------
    subunits : ndarray
        3D ndarray containing the subunits.
    """

    subunits = np.zeros((4, resolution, resolution))
    center = int(resolution/2)
    quarter = int(round(resolution/4))
    subunits[0, quarter:center, quarter:center] = 1
    subunits[1, center:-quarter, quarter:center] = 1
    subunits[2, quarter:center, center:-quarter] = 1
    subunits[3, center:-quarter, center:-quarter] = 1

    return subunits


def Scenario_basic_squares_overlap(resolution):
    """ Create a 3D-array containing subunits according to scenario
    'basic squares overlap'.

    'basic squares overlap' is a 2x2 layout of quadratic non-overlapping
    subunits plus one quadratic subunit in the center overlapping with all
    others.

    Parameters
    ----------
    resolution : int
        Width and height of the square receptive area of the RGC in pixels.

    Returns
    -------
    subunits : ndarray
        3D ndarray containing the subunits.
    """

    subunits = np.zeros((5, resolution, resolution))
    center = int(resolution/2)
    quarter = int(round(resolution/4))
    subunits[0, quarter:center, quarter:center] = 1
    subunits[1, center:-quarter, quarter:center] = 1
    subunits[2, quarter:center, center:-quarter] = 1
    subunits[3, center:-quarter, center:-quarter] = 1
    three_eight = int(round(3/8*resolution))
    subunits[4, three_eight:-three_eight, three_eight:-three_eight] = 1

    return subunits


def Scenario_grid(resolution, num_subunits):
    """ Create a 3D-array containing subunits according to scenario
    'grid'.

    'grid' is a square grid layout of quadratic non-overlapping
    subunits.

    Parameters
    ----------
    resolution : int
        Width and height of the square receptive area of the RGC in pixels.
    num_subunits : int
        Number of subunits in the grid. Will be rounded down to the next square
        number.

    Returns
    -------
    subunits : ndarray
        3D ndarray containing the subunits.
    """

    N = int(np.sqrt(num_subunits))
    subunits = np.zeros((N*N, resolution, resolution))
    quarter = int(round(resolution/4))
    for row in range(N):
        for col in range(N):
            index = col + N*row
            start_x = int(quarter + col * (resolution - 2*quarter)/N)
            end_x = int(quarter + (col + 1) * (resolution - 2*quarter)/N)
            start_y = int(quarter + row * (resolution - 2*quarter)/N)
            end_y = int(quarter + (row + 1) * (resolution - 2*quarter)/N)
            subunits[index, start_x:end_x, start_y:end_y] = 1

    return subunits


def Scenario_basic_squares_full(resolution):
    """ Create a 3D-array containing subunits according to scenario
    'basic squares full'.

    'basic sqaures full' is a 2x2 layout of quadratic non-overlapping subunits
    without any empty space in the outer part of layout.

    Parameters
    ----------
    resolution : int
        Width and height of the square receptive area of the RGC in pixels.

    Returns
    -------
    subunits : ndarray
        3D ndarray containing the subunits.
    """

    subunits = np.zeros((4, resolution, resolution))
    center = int(resolution/2)
    subunits[0, :center, :center] = 1
    subunits[1, center:, :center] = 1
    subunits[2, :center, center:] = 1
    subunits[3, center:, center:] = 1

    return subunits


def Scenario_basic_circles(resolution):
    """ Create a 3D-array containing subunits according to scenario
    'basic circles'.

    'basic circles' is a 2x2 layout of circular non-overlapping subunits.

    Parameters
    ----------
    resolution : int
        Width and height of the square receptive area of the RGC in pixels.

    Returns
    -------
    subunits : ndarray
        3D ndarray containing the subunits.
    """

    subunits = np.zeros((4, resolution, resolution))
    three_eight = 3*resolution/8
    radius = resolution/8
    subunits[0] = Circle(resolution,
                         (three_eight-0.5, three_eight-0.5),
                         radius)
    subunits[1] = Circle(resolution,
                         (resolution-three_eight-0.5, three_eight-0.5),
                         radius)
    subunits[2] = Circle(resolution,
                         (three_eight-0.5, resolution-three_eight-0.5),
                         radius)
    subunits[3] = Circle(resolution,
                         (resolution-three_eight-0.5,
                          resolution-three_eight-0.5),
                         radius)

    return subunits


def Scenario_basic_circles_overlap(resolution):
    """ Create a 3D-array containing subunits according to scenario
    'basic circles overlap'.

    'basic circles overlap' is a 2x2 layout of circular non-overlapping
    subunits plus one quadratic subunit in the center overlapping with all
    others.

    Parameters
    ----------
    resolution : int
        Width and height of the square receptive area of the RGC in pixels.

    Returns
    -------
    subunits : ndarray
        3D ndarray containing the subunits.
    """

    subunits = np.zeros((5, resolution, resolution))
    three_eight = 3*resolution/8
    radius = resolution/8
    subunits[0] = Circle(resolution,
                         (three_eight-0.5, three_eight-0.5),
                         radius)
    subunits[1] = Circle(resolution,
                         (resolution-three_eight-0.5, three_eight-0.5),
                         radius)
    subunits[2] = Circle(resolution,
                         (three_eight-0.5, resolution-three_eight-0.5),
                         radius)
    subunits[3] = Circle(resolution,
                         (resolution-three_eight-0.5,
                          resolution-three_eight-0.5),
                         radius)
    center = resolution/2
    subunits[4] = Circle(resolution,
                         (center-0.5, center-0.5),
                         radius)

    return subunits


def Scenario_basic_gauss(resolution, swap_gauss_for_cosine=False):
    """ Create a 3D-array containing subunits according to scenario
    'basic gauss'.

    'basic gauss' is a 2x2 layout of gaussian non-overlapping subunits.

    Parameters
    ----------
    resolution : int
        Width and height of the square receptive area of the RGC in pixels.
    swap_gauss_for_cosine : bool, optional
        If True, creates cosine-shaped subunits instead of Gaussians. Default
        is False.

    Returns
    -------
    subunits : ndarray
        3D ndarray containing the subunits.
    subunit_params : ndarray
        2D array containing parameters of the gaussian subunits. First index
        denotes subunit, second denotes: x-position, y-position, sigma_x,
        sigma_y, angle (radians).
    """

    pos_3_8 = 3*resolution/8 - 0.5
    pos_5_8 = 5*resolution/8 - 0.5
    sigma = resolution/10
    gauss_params = np.array([[pos_3_8, pos_3_8, sigma, sigma, 0],
                             [pos_3_8, pos_5_8, sigma, sigma, 0],
                             [pos_5_8, pos_3_8, sigma, sigma, 0],
                             [pos_5_8, pos_5_8, sigma, sigma, 0]])
    subunits = np.zeros((gauss_params.shape[0], resolution, resolution))
    for i in range(gauss_params.shape[0]):
        if swap_gauss_for_cosine:
            subunits[i] = Cosine_spot_array(resolution, *gauss_params[i])
        else:
            subunits[i] = Gaussian_array(resolution, *gauss_params[i])

    return subunits, gauss_params


def Scenario_basic_gauss_overlap(resolution, swap_gauss_for_cosine=False):
    """ Create a 3D-array containing subunits according to scenario
    'basic gauss overlap'.

    'basic gauss overlap' is a 2x2 layout of gaussian non-overlapping subunits
    plus one gaussian subunit in the center overlapping with all others.

    Parameters
    ----------
    resolution : int
        Width and height of the square receptive area of the RGC in pixels.
    swap_gauss_for_cosine : bool, optional
        If True, creates cosine-shaped subunits instead of Gaussians. Default
        is False.

    Returns
    -------
    subunits : ndarray
        3D ndarray containing the subunits.
    subunit_params : ndarray
        2D array containing parameters of the gaussian subunits. First index
        denotes subunit, second denotes: x-position, y-position, sigma_x,
        sigma_y, angle (radians).
    """

    pos_3_8 = 3*resolution/8 - 0.5
    pos_5_8 = 5*resolution/8 - 0.5
    pos_1_2 = resolution/2 - 0.5
    sigma = resolution/10
    gauss_params = np.array([[pos_3_8, pos_3_8, sigma, sigma, 0],
                             [pos_3_8, pos_5_8, sigma, sigma, 0],
                             [pos_5_8, pos_3_8, sigma, sigma, 0],
                             [pos_5_8, pos_5_8, sigma, sigma, 0],
                             [pos_1_2, pos_1_2, sigma, sigma, 0]])
    subunits = np.zeros((gauss_params.shape[0], resolution, resolution))
    for i in range(gauss_params.shape[0]):
        if swap_gauss_for_cosine:
            subunits[i] = Cosine_spot_array(resolution, *gauss_params[i])
        else:
            subunits[i] = Gaussian_array(resolution, *gauss_params[i])

    return subunits, gauss_params


def Scenario_realistic_gauss(resolution, num_subunits, rng_seed, overlap_factor,
                             irregularity, swap_gauss_for_cosine=False):
    """ Create a 3D-array containing subunits according to scenario
    'realistic gauss'.

    'realistic gauss' is a layout of 4-12 gaussian subunits with variable
    standard deviations and variable rotation angles.

    Parameters
    ----------
    resolution : int
        Width and height of the square receptive area of the RGC in pixels.
    num_subunits : int
        If not None, specifies the number of subunits.
    rng_seed : int
        Non-negative seed used for randomly generating the subunit layout. If
        None, a random seed is used.
    overlap_factor : float
        Regulates the size of the subunits without changing their spacing,
        thereby adjusting their overlap. Note that the value is not related to
        any meaningful measure.
    irregularity : float
        Determines how strongly the layout deviates from a hexagonal grid.
    swap_gauss_for_cosine : bool, optional
        If True, creates cosine-shaped subunits instead of Gaussians. Default
        is False.

    Returns
    -------
    subunits : ndarray
        3D ndarray containing the subunits.
    subunit_params : ndarray
        2D array containing parameters of the gaussian subunits. First index
        denotes subunit, second denotes: x-position, y-position, sigma_x,
        sigma_y, angle (radians).
    """

    # Setting up random generator
    rng = np.random.default_rng(seed=rng_seed)

    # local constants
    size = 100
    sqrt_num_points = 8
    if num_subunits is None:
        num_subunits = rng.integers(4, 13)

    # Generating a perturbed hexagonal grid
    xx, yy = np.mgrid[0:size:sqrt_num_points*1j, 0:size:sqrt_num_points*1j]
    points = np.transpose(np.vstack([xx.ravel(), yy.ravel()]))
    points[::2, 0] += size/(sqrt_num_points-1)/2
    points[:, 1] *= np.sqrt(3)/2
    points = points + rng.normal(scale=irregularity, size=points.shape)

    # Calculating the voronoi sets
    xx, yy = np.mgrid[0:size, 0:size]
    pixels = np.transpose(np.vstack([xx.ravel(), yy.ravel()]))
    closest = np.empty(pixels.shape[0])
    for counter, pixel in enumerate(pixels):
        closest[counter] = np.argmin(np.sum(np.square(points - pixel), axis=1))
    voronoi = np.reshape(closest, (size, size))

    # Calculate the center of masses of the voronoi sets
    centers = np.empty_like(points)
    for i in range(points.shape[0]):
        centers[i] = np.mean(pixels[voronoi.ravel() == i], axis=0)

    # Choosing only the N sets that are closest to the screen center
    distances = np.sum(np.square(centers - size/2), axis=1)
    subunits_idxe = np.argsort(distances)[:num_subunits]
    subunits = np.empty((num_subunits, size, size))
    for counter, idxe in enumerate(subunits_idxe):
        subunits[counter] = (voronoi == idxe).astype(int)
    centers = centers[subunits_idxe]

    # Fitting Gaussians to the selected Voronoi sets
    gauss_params = np.empty((num_subunits, 5))
    for i in range(num_subunits):
        initial_guess = (centers[i, 0], centers[i, 1], 1, 1, 0)
        gauss_params[i], _ = opt.curve_fit(TwoD_Gaussian, pixels,
                                           (subunits[i].ravel()
                                            / np.sum(subunits[i])),
                                           p0=initial_guess)

    # Adjusting the standard deviation of the Gaussians to be more realistic
    gauss_params[:, 2:4] *= overlap_factor

    # Rescaling everything to a constant RGC size
    gauss_params[:, :2] = size/2 + ((gauss_params[:, :2] - size/2)
                                    * 3 / np.sqrt(num_subunits))
    gauss_params[:, 2:4] *= 3 / np.sqrt(num_subunits)

    # Rescaling to the resolution of the model
    gauss_params[:, :4] *= resolution/size

    # Creating the subunit array
    subunits = np.zeros((num_subunits, resolution, resolution))
    for i in range(gauss_params.shape[0]):
        if swap_gauss_for_cosine:
            subunits[i] = Cosine_spot_array(resolution, *gauss_params[i])
        else:
            subunits[i] = Gaussian_array(resolution, *gauss_params[i])

    return subunits, gauss_params


###############################################################################
# Functions for the subunit nonlinearity
###############################################################################
def Subunit_nl_none(signal):
    """ No subunit nonlinearity.

    Parameters
    ----------
    signal : ndarray
        Contains all subunit signals.

    Returns
    -------
    response : ndarray
        Contains the result of the nonlinearity applied elementwise to
        *signal*.
    """

    return signal


def Subunit_nl_threshold_cubic(signal):
    """ Threshold cubic nonlinearity for subunits.

    Parameters
    ----------
    signal : ndarray
        Contains all subunit signals.

    Returns
    -------
    response : ndarray
        Contains the result of the nonlinearity applied elementwise to
        *signal*.
    """

    response = np.zeros_like(signal)
    response[signal > 0] = np.power(signal[signal > 0], 3)

    return response


def Subunit_nl_threshold_quadratic(signal):
    """ Threshold quadratic nonlinearity for subunits.

    Parameters
    ----------
    signal : ndarray
        Contains all subunit signals.

    Returns
    -------
    response : ndarray
        Contains the result of the nonlinearity applied elementwise to
        *signal*.
    """

    response = np.zeros_like(signal)
    response[signal > 0] = np.square(signal[signal > 0])

    return response


def Subunit_nl_threshold_linear(signal):
    """ Threshold linear (relu) nonlinearity for subunits.

    Parameters
    ----------
    signal : ndarray
        Contains all subunit signals.

    Returns
    -------
    response : ndarray
        Contains the result of the nonlinearity applied elementwise to
        *signal*.
    """

    response = np.zeros_like(signal)
    response[signal > 0] = signal[signal > 0]

    return response


def Subunit_nl_threshold_sqrt(signal):
    """ Threshold square root nonlinearity for subunits.

    Parameters
    ----------
    signal : ndarray
        Contains all subunit signals.

    Returns
    -------
    response : ndarray
        Contains the result of the nonlinearity applied elementwise to
        *signal*.
    """

    response = np.zeros_like(signal)
    response[signal > 0] = np.sqrt(signal[signal > 0])

    return response


def Subunit_nl_exponential(signal):
    """ Exponential nonlinearity for subunits.

    Parameters
    ----------
    signal : ndarray
        Contains all subunit signals.

    Returns
    -------
    response : ndarray
        Contains the result of the nonlinearity applied elementwise to
        *signal*.
    """

    response = np.exp(signal)

    return response


def Subunit_nl_linear_linear(signal):
    """ Linear-linear nonlinearity for subunits.

    Parameters
    ----------
    signal : ndarray
        Contains all subunit signals.

    Returns
    -------
    response : ndarray
        Contains the result of the nonlinearity applied elementwise to
        *signal*.
    """

    response = signal
    response[signal < 0] /= 2

    return response


def Subunit_nl_softplus(signal):
    """ Softplus nonlinearity for subunits.

    Parameters
    ----------
    signal : ndarray
        Contains all subunit signals.

    Returns
    -------
    response : ndarray
        Contains the result of the nonlinearity applied elementwise to
        *signal*.
    """

    sharpness = 30
    response = np.log(1 + np.exp(sharpness*signal))/sharpness

    return response


###############################################################################
# Functions for the connection weights of subunits to the RGC
###############################################################################
def Weights_equal(num_subunits):
    """ Create equal connections weights between subunits and RGC.

    Weights are normalised to have a sum of 1.

    Parameters
    ----------
    num_subunits : int
        Number of subunits.

    Returns
    -------
    weights : ndarray
        Contains the connection weights.
    """

    return np.ones(num_subunits) / num_subunits


def Weights_gauss(resolution, subunit_locs, gauss_std):
    """ Create connection weights between subunits and RGC according to a
    Gaussian located at the center of the receptive area.

    Weights are normalised to have a sum of 1.

    Parameters
    ----------
    resolution : int
        Width and height of the square receptive area of the RGC in pixels.
    subunit_locs : int
        Locations of the subunits.
    gauss_std : float
        Standard deviation of the Gaussian used to set the subunit weights in
        units of *resolution*, i.e. simulation area size.

    Returns
    -------
    weights : ndarray
        Contains the connection weights.
    """

    weights = TwoD_Gaussian(subunit_locs, (resolution-1)/2, (resolution-1)/2,
                            gauss_std*resolution, gauss_std*resolution, 0)

    return weights / np.sum(weights)


###############################################################################
# Functions for the output nonlinearity
###############################################################################
def Output_nl_none(signal):
    """ No output nonlinearity.

    Parameters
    ----------
    signal : float
        Contains the RGC signal.

    Returns
    -------
    response : float
        Contains the result of the nonlinearity applied to *signal*.
    """

    return signal


def Output_nl_threshold_quadratic(signal):
    """ Threshold-quadratic output nonlinearity.

    Parameters
    ----------
    signal : float
        Contains the RGC signal.

    Returns
    -------
    response : float
        Contains the result of the nonlinearity applied to *signal*.
    """

    if signal <= 0:
        response = 0
    else:
        response = np.square(signal)

    return response


def Output_nl_threshold_linear(signal):
    """ Threshold-linear output nonlinearity.

    Parameters
    ----------
    signal : float
        Contains the RGC signal.

    Returns
    -------
    response : float
        Contains the result of the nonlinearity applied to *signal*.
    """

    if signal <= 0:
        response = 0
    else:
        response = signal

    return response


def Output_nl_linear_linear(signal):
    """ Linear-linear output nonlinearity.

    Parameters
    ----------
    signal : float
        Contains the RGC signal.

    Returns
    -------
    response : float
        Contains the result of the nonlinearity applied to *signal*.
    """

    if signal <= 0:
        response = 0.5 * signal
    else:
        response = signal

    return response


def Output_nl_softplus(signal):
    """ Softplus output nonlinearity.

    Parameters
    ----------
    signal : float
        Contains the RGC signal.

    Returns
    -------
    response : float
        Contains the result of the nonlinearity applied to *signal*.
    """

    sharpness = 1000
    response = np.log(1 + np.exp(sharpness*signal))/sharpness

    return response


###############################################################################
# Functions for the spiking process
###############################################################################
def Spiking_none(response):
    """ No spiking process. Input is returned.

    Parameters
    ----------
    response : float
        Contains the RGC response.

    Returns
    -------
    spikes : float
        Contains the result of the spiking process applied to *response*.
    """

    return response


def Spiking_poisson(coefficient, shift):
    """ Poisson spiking process.

    Parameters
    ----------
    coefficient : float
        Coefficient between *response* and expected value of spikes.
    shift : float
        Shift to be added to *response* before multiplication with
        *coefficient*. Should be used to prevent negative Poisson rates.
        Thus corresponds to the response to full-field black.

    Returns
    -------
    Spikes : function
        Function that returns the number of spikes for a given response.

        Parameters

        response : float
            Contains the RGC response.
        Returns

        float
            The result of the spiking process applied to *response*.
    """

    rng = np.random.default_rng()

    def Spikes(response):
        return rng.poisson(coefficient * (response + shift))

    return Spikes


###############################################################################
# Subunit Class
###############################################################################
class Subunit_Model:
    """ This class implements a model of a retinal ganglion cell (RGC) based
    on an LNLN structure with linear subunits, a nonlinearity, a linear
    ganglion cell and an output nonlinearity. It can be used to simulate
    responses to arbitrary spatial or spatiotemporal stimuli, in the latter
    case with optional feedback mechanisms. Most stages of the model are not
    required and have multiple options, such that the desired model can be set
    up in a sandbox-principle.

    Attributes
    ----------
    resolution : int
        Edge length of the simulation area in pixels.
    scenario : string
        Name of the subunit layout scenario.
    subunit_nonlinearity : string
        Name of the subunit nonlinearity.
    subunit_feedback : string
        Name of the subunit/local feedback mechanism.
    subunit_weights : string
        Name of the choice for the weights of the subunits.
    rgc_nonlinearity : string
        Name of the RGC/output nonlinearity.
    rgc_feedback : string
        Name of the RGC/global feedback mechanism.
    rgc_spiking : string
        Name of the spike generation process.
    num_subunits : int
        Current number of subunits of the model. In contrast to the keyword
        argument *num_subunits* (stored in *params*), the attribute always
        contains the correct number and thus also never None.
    params : dict
        Contains the parameters of the model object. Refer to the global
        variable *DEFAULT_PARAMS* for more info.

    Methods
    -------
    set_subunits(scenario, **kwargs)
        Set the subunit scenario.
    set_temporal_filter(length=21, amplitude_1=1, mean_1=2, sigma_1=0.8,
                        amplitude_2=-0.3, mean_2=6, sigma_2=2.5,
                        flip_on_off=False)
        Set the temporal filter of the subunits.
    set_subunit_nl(subunit_nonlinearity)
        Set the subunit nonlinearity.
    set_local_feedback(subunit_feedback, length=100, exp_time_constant=15,
                       sig_middle=10, sig_steepness=0.3)
        Set the local multiplicative feedback of the model.
    set_weights(subunit_weights, **kwargs)
        Set the subunit to RGC connection weights.
    set_output_nl(rgc_nonlinearity)
        Set the RGC output nonlinearity.
    set_global_feedback(rgc_feedback, length=100, exp_time_constant=15,
                        sig_middle=10, sig_steepness=0.3)
        Set the global multiplicative feedback of the model.
    set_spiking(rgc_spiking, **kwargs)
        Set the spiking process.
    set_resolution(new_resolution)
        Change the resolution without changing anything else.
    set_circular()
        Change all Gaussian ellipses to be circular.
    set_elliptic()
        Resets the Gaussian ellipses to their state before *set_circular*
        was called.
    clear_history()
        Clears all stimulus-dependent history of the model.
    get_receptive_field()
        Compute the receptive field of the model.
    plot_subunits(savepath=None)
        Plot the subunits of the model in separate plots.
    plot_subunit_ellipses(savepath=None)
        Plot the subunit ellipses in one plot.
    plot_receptive_field(savepath=None)
        Plot the receptive field of the model.
    plot_temporal_filter(savepath=None)
        Plot the temporal filter of the subunits.
    plot_local_feedback_filter(savepath=None)
        Plot the local feedback filter of the model affecting the subunits.
    plot_global_feedback_filter(savepath=None)
        Plot the global feedback filter of the RGC.
    response_to_flash(image)
        Calculate the response of the model to a flash of the stimulus.
    response_to_frame(frame)
        Calculate the response of the model during the next frame of the
        stimulus.

    Parameters for initialization
    -----------------------------
    resolution : int, optional
        Width and height in pixels of the square area in which the receptive
        field of the RGC lies. Default is 100. From *resolution* a physical
        pixel size can be deduced using the 1.5-sigma ellipse diameter for
        marmoset Off parasol cells of about 120 microns and assuming that the
        other parameters that influence the size of the modelled receptive
        field are left at their defaults. *resolution* 120 would correspond to
        pixels of 2.5 microns, *resolution* 40 would correspond to 7.5 micron
        pixels, and *resolution* 20 to 15 microns.
    scenario : string, optional
        Defines the subunit layout. Note that some options overrule the
        keyword argument *num_subunits*. Options:
            'basic squares': 2x2 layout of quadratic non-overlapping
            subunits.

            'basic squares overlap': like 'basic squares' but with one
            additional quadratic subunit in the center overlapping all
            others.

            'grid': NxN layout of quadratic non-overlapping subunits.

            'basic squares full': like 'basic squares' but without empty
            space in the outer parts of the area.

            'basic circles': like 'basic squares' but with circular
            subunits.

            'basic circles overlap': like 'basic squares overlap' but with
            circular subunits.

            'basic gauss': like 'basic squares' but with Gaussian subunits.
            Gaussians are not truncated, so they do slightly overlap.

            'basic gauss overlap': like 'basic squares overlap' but with
            Gaussian subunits.

            'realistic gauss' (default): layout of 4-12 gaussian subunits with
            variable standard deviations and variable rotation angles.
    subunit_nonlinearity : string, optional
        Nonlinearity of the subunits. Options:
            None.

            'threshold-cubic'.

            'threshold-quadratic' (default).

            'threshold-linear': relu.

            'threshold-sqrt'.

            'exponential'.

            'linear-linear': Linear relation with a corner at 0 changing
            the steepness.

            'softplus'.
    subunit_feedback : string, optional
        Multiplicative feedback mechanism of the subunits. Options:
            None (default): No local feedback.

            'exp_sig': Exponentially decaying feedback filter combined with
            a sigmoidal feedback nonlinearity.
    subunit_weights : string, optional
        Weights connecting the subunits with the RGC. Options:
            'equal' (default): All subunits have equal weights.

            'gauss': Weights correspond to a 2D Gaussian located at the
            receptive area's center.
    rgc_nonlinearity : string, optional
        Output nonlinearity of the RGC. Options:
            None (default).

            'threshold-quadratic'.

            'threshold-linear'.

            'linear-linear': Linear relation with a corner at 0 changing
            the steepness.

            'softplus'.
    rgc_feedback : string, optional
        Multiplicative feedback mechanism of the RGC. Options:
            None (default): No global feedback.

            'exp_sig': Exponentially decaying feedback filter combined with
            a sigmoidal feedback nonlinearity.
    rgc_spiking : string, optional
        Spiking process of the RGC. Options:
            None (default): No random spiking process, model outputs firing
            rate.

            'poisson': Spiking via Poisson distribution. Model outputs
            number of spikes.
    **kwargs
        Additional keyword arguments. Check the global variable
        *DEFAULT_PARAMS* for more information.
    """


    def __init__(self, resolution=100,
                 scenario='realistic gauss',
                 subunit_nonlinearity='threshold-quadratic',
                 subunit_feedback=None,
                 subunit_weights='equal',
                 rgc_nonlinearity=None,
                 rgc_feedback=None,
                 rgc_spiking=None,
                 **kwargs):
        """ Create an RGC model containing subunits.

        Parameters
        ----------
        resolution : int, optional
            Width and height in pixels of the square area in which the receptive
            field of the RGC lies. Default is 100. From *resolution* a physical
            pixel size can be deduced using the 1.5-sigma ellipse diameter for
            marmoset Off parasol cells of about 120 microns and assuming that the
            other parameters that influence the size of the modelled receptive
            field are left at their defaults. *resolution* 120 would correspond to
            pixels of 2.5 microns, *resolution* 40 would correspond to 7.5 micron
            pixels, and *resolution* 20 to 15 microns.
        scenario : string, optional
            Defines the subunit layout. Note that some options overrule the
            keyword argument *num_subunits*. Options:
                'basic squares': 2x2 layout of quadratic non-overlapping
                subunits.

                'basic squares overlap': like 'basic squares' but with one
                additional quadratic subunit in the center overlapping all
                others.

                'grid': NxN layout of quadratic non-overlapping subunits.

                'basic squares full': like 'basic squares' but without empty
                space in the outer parts of the area.

                'basic circles': like 'basic squares' but with circular
                subunits.

                'basic circles overlap': like 'basic squares overlap' but with
                circular subunits.

                'basic gauss': like 'basic squares' but with Gaussian subunits.
                Gaussians are not truncated, so they do slightly overlap.

                'basic gauss overlap': like 'basic squares overlap' but with
                Gaussian subunits.

                'realistic gauss' (default): layout of 4-12 gaussian subunits with
                variable standard deviations and variable rotation angles.
        subunit_nonlinearity : string, optional
            Nonlinearity of the subunits. Options:
                None.

                'threshold-cubic'.

                'threshold-quadratic' (default).

                'threshold-linear': relu.

                'threshold-sqrt'.

                'exponential'.

                'linear-linear': Linear relation with a corner at 0 changing
                the steepness.

                'softplus'.
        subunit_feedback : string, optional
            Multiplicative feedback mechanism of the subunits. Options:
                None (default): No local feedback.

                'exp_sig': Exponentially decaying feedback filter combined with
                a sigmoidal feedback nonlinearity.
        subunit_weights : string, optional
            Weights connecting the subunits with the RGC. Options:
                'equal' (default): All subunits have equal weights.

                'gauss': Weights correspond to a 2D Gaussian located at the
                receptive area's center.
        rgc_nonlinearity : string, optional
            Output nonlinearity of the RGC. Options:
                None (default).

                'threshold-quadratic'.

                'threshold-linear'.

                'linear-linear': Linear relation with a corner at 0 changing
                the steepness.

                'softplus'.
        rgc_feedback : string, optional
            Multiplicative feedback mechanism of the RGC. Options:
                None (default): No global feedback.

                'exp_sig': Exponentially decaying feedback filter combined with
                a sigmoidal feedback nonlinearity.
        rgc_spiking : string, optional
            Spiking process of the RGC. Options:
                None (default): No random spiking process, model outputs firing
                rate.

                'poisson': Spiking via Poisson distribution. Model outputs
                number of spikes.
        weights_gauss_std : float, optional
            Standard deviation of the Gaussian used to set the subunit weights
            if *subunit_weights* is 'gauss'. In units of *resolution*, i.e.
            simulation area size. Default is 0.12.
        **kwargs
            Additional keyword arguments. Check the global variable
            *DEFAULT_PARAMS* for more information.
        """

        # Store status of initialization
        self.initialized = False

        # Preprocessing parameters
        self.resolution = resolution
        self.scenario = scenario
        self.params = DEFAULT_PARAMS.copy()
        self.params.update(kwargs)

        # Setting up the subunit scenario
        self.set_subunits(scenario)

        # Setting up the temporal filter of the subunits
        self.set_temporal_filter()

        # Defining the subunit nonlinearity
        self.set_subunit_nl(subunit_nonlinearity)

        # Defining the local feedback
        self.set_local_feedback(subunit_feedback)

        # Defining the subunit to RGC connection weights
        self.set_weights(subunit_weights)

        # Defining the RGC output nonlinearity
        self.set_output_nl(rgc_nonlinearity)

        # Defining the global feedback
        self.set_global_feedback(rgc_feedback)

        # Defining the spiking process
        self.set_spiking(rgc_spiking)

        # Initialize variables that store stimulus-dependent history
        self.clear_history()

        # Initialization completed
        self.initialized = True


    def set_subunits(self, scenario, **kwargs):
        """ Set the subunit scenario.

        Parameters
        ----------
        scenario : string
            Defines the subunit layout. Note that some options overrule the
            keyword argument *num_subunits*. Options:
                'basic squares': 2x2 layout of quadratic non-overlapping
                subunits.

                'basic squares overlap': like 'basic squares' but with one
                additional quadratic subunit in the center overlapping all
                others.

                'grid': NxN layout of quadratic non-overlapping subunits.

                'basic squares full': like 'basic squares' but without empty
                space in the outer parts of the area.

                'basic circles': like 'basic squares' but with circular
                subunits.

                'basic circles overlap': like 'basic squares overlap' but with
                circular subunits.

                'basic gauss': like 'basic squares' but with Gaussian subunits.
                Gaussians are not truncated, so they do slightly overlap.

                'basic gauss overlap': like 'basic squares overlap' but with
                Gaussian subunits.

                'realistic gauss': layout of 4-12 gaussian subunits with
                variable standard deviations and variable rotation angles.
        **kwargs
            Additional keyword arguments. Check the global variable
            *DEFAULT_PARAMS* for more information.
        """

        self.params.update(kwargs)

        # Setting up the subunit scenario
        if scenario == 'basic squares':
            self.subunits = Scenario_basic_squares(self.resolution)
        elif scenario == 'basic squares overlap':
            self.subunits = Scenario_basic_squares_overlap(self.resolution)
        elif scenario == 'grid':
            self.subunits = Scenario_grid(self.resolution,
                                          self.params['num_subunits'])
        elif scenario == 'basic squares full':
            self.subunits = Scenario_basic_squares_full(self.resolution)
        elif scenario == 'basic circles':
            self.subunits = Scenario_basic_circles(self.resolution)
        elif scenario == 'basic circles overlap':
            self.subunits = Scenario_basic_circles_overlap(self.resolution)
        elif scenario == 'basic gauss':
            self.subunits, self.subunit_params = Scenario_basic_gauss(self.resolution,
                                                                      self.params['swap_gauss_for_cosine'])
        elif scenario == 'basic gauss overlap':
            self.subunits, self.subunit_params = Scenario_basic_gauss_overlap(self.resolution,
                                                                              self.params['swap_gauss_for_cosine'])
        elif scenario == 'realistic gauss':
            self.subunits, self.subunit_params = Scenario_realistic_gauss(self.resolution,
                                                                          self.params['num_subunits'],
                                                                          self.params['rng_seed'],
                                                                          self.params['overlap_factor'],
                                                                          self.params['irregularity'],
                                                                          self.params['swap_gauss_for_cosine'])
        self.num_subunits = self.subunits.shape[0]

        # If this function has been used to change (i.e. not initialize) the
        # subunits, then the history of subunit views has to be cleared
        if self.initialized:
            self.clear_history()



    def set_temporal_filter(self, length=21, amplitude_1=1, mean_1=2,
                            sigma_1=0.8, amplitude_2=-0.3, mean_2=6,
                            sigma_2=2.5, flip_on_off=False):
        """ Set the temporal filter of the subunits.

        First value in filter is weight of current frame, each value
        corresponds to one frame. Filter is generated from difference (sum to
        be precise) of Gaussians.

        Parameters
        ----------
        length : int, optional
            Length of the temporal filter in frames.
        amplitude_1 : float, optional
            Amplitude of the first Gaussian.
        mean_1 : float, optional
            Mean of the first Gaussian.
        sigma_1 : float, optional
            Standard deviation of the first Gaussian.
        amplitude_2 : float, optional
            Amplitude of the second Gaussian.
        mean_2 : float, optional
            Mean of the second Gaussian.
        sigma_2 : float, optional
            Standard deviation of the second Gaussian.
        flip_on_off : bool, optional
            If True, the polarity of the temporal filter is flipped, thereby
            changing an On/Off filter to an Off/On filter.
        """

        x = np.arange(length)
        self.temporal_filter = (amplitude_1
                                * np.exp(-1/2*np.square((x - mean_1)/sigma_1))
                                + amplitude_2
                                * np.exp(-1/2*np.square((x - mean_2)/sigma_2)))
        if flip_on_off:
            self.temporal_filter *= -1

        # If this function has been used to change (i.e. not initialize) the
        # filter, then the history of subunit views has to be cleared
        if self.initialized:
            self.clear_history()


    def set_subunit_nl(self, subunit_nonlinearity):
        """ Set the subunit nonlinearity.

        Parameters
        ----------
        subunit_nonlinearity : string
            Nonlinearity of the subunits. Options:
                None.

                'threshold-cubic'.

                'threshold-quadratic'.

                'threshold-linear': relu.

                'threshold-sqrt'.

                'exponential'.

                'linear-linear': Linear relation with a corner at 0 changing
                the steepness.

                'softplus'.
        """

        if subunit_nonlinearity is None:
            self.subunit_nl = Subunit_nl_none
        elif subunit_nonlinearity == 'threshold-cubic':
            self.subunit_nl = Subunit_nl_threshold_cubic
        elif subunit_nonlinearity == 'threshold-quadratic':
            self.subunit_nl = Subunit_nl_threshold_quadratic
        elif subunit_nonlinearity == 'threshold-linear':
            self.subunit_nl = Subunit_nl_threshold_linear
        elif subunit_nonlinearity == 'threshold-sqrt':
            self.subunit_nl = Subunit_nl_threshold_sqrt
        elif subunit_nonlinearity == 'exponential':
            self.subunit_nl = Subunit_nl_exponential
        elif subunit_nonlinearity == 'linear-linear':
            self.subunit_nl = Subunit_nl_linear_linear
        elif subunit_nonlinearity == 'softplus':
            self.subunit_nl = Subunit_nl_softplus


    def set_local_feedback(self, subunit_feedback, length=100,
                            exp_time_constant=15, sig_middle=10,
                            sig_steepness=0.3):
        """ Set the local multiplicative feedback of the model.

        This kind of adaptation will affect the subunit stage.

        Parameters
        ----------
        subunit_feedback : string
            Multiplicative feedback mechanism of the subunits. Options:
                None: No local feedback.

                'exp_sig': Exponentially decaying feedback filter combined with
                a sigmoidal feedback nonlinearity.
        length : int
            Lenght of the local feedback filter in frames.
        exp_time_constant : float, optional
            If *subunit_feedback* is 'exp_sig', this parameter defines the time
            constant of the exponential decay in terms of frames.
        sig_middle : float, optional
            If *subunit_feedback* is 'exp_sig', this parameters defines the
            middle point of the sigmoidal feedback nonlinearity.
        sig_steepness : float, optional
            If *subunit_feedback* is 'exp_sig', this parameter defines the
            steepness of the sigmoidal feedback nonlinearity.
        """

        if subunit_feedback is None:
            self.local_feedback_filter = np.zeros(length)

            def nonlinearity(signal):
                return 1

            self.local_feedback_nonlinearity = nonlinearity
        elif subunit_feedback == 'exp_sig':
            self.local_feedback_filter = np.exp(-np.arange(length)
                                                / exp_time_constant)

            def nonlinearity(signal):
                return 1 / (1 + np.exp(sig_steepness * (signal - sig_middle)))

            self.local_feedback_nonlinearity = nonlinearity

        # If this function has been used to change (i.e. not initialize) the
        # feedback, then the stimulus-dependent history has to be cleared
        if self.initialized:
            self.clear_history()


    def set_weights(self, subunit_weights, **kwargs):
        """ Set the subunit to RGC connection weights.

        Parameters
        ----------
        subunit_weights : string
            Weights connecting the subunits with the RGC. Options:
                'equal': All subunits have equal weights.

                'gauss': Weights correspond to a 2D Gaussian located at the
                receptive area's center.
        **kwargs
            Additional keyword arguments. Check the global variable
            *DEFAULT_PARAMS* for more information.
        """

        self.params.update(kwargs)

        if subunit_weights == 'equal':
            self.weights = Weights_equal(self.num_subunits)
        elif subunit_weights == 'gauss':
            self.weights = Weights_gauss(self.resolution,
                                         self.subunit_params[:, :2],
                                         self.params['weights_gauss_std'])


    def set_output_nl(self, rgc_nonlinearity):
        """ Set the RGC output nonlinearity.

        Parameters
        ----------
        rgc_nonlinearity : string
            Output nonlinearity of the RGC. Options:
                None.

                'threshold-quadratic'.

                'threshold-linear'.

                'linear-linear': Linear relation with a corner at 0 changing
                the steepness.

                'softplus'.
        """

        if rgc_nonlinearity is None:
            self.output_nl = Output_nl_none
        elif rgc_nonlinearity == 'threshold-quadratic':
            self.output_nl = Output_nl_threshold_quadratic
        elif rgc_nonlinearity == 'threshold-linear':
            self.output_nl = Output_nl_threshold_linear
        elif rgc_nonlinearity == 'linear-linear':
            self.output_nl = Output_nl_linear_linear
        elif rgc_nonlinearity == 'softplus':
            self.output_nl = Output_nl_softplus


    def set_global_feedback(self, rgc_feedback, length=100,
                            exp_time_constant=15, sig_middle=10,
                            sig_steepness=0.3):
        """ Set the global multiplicative feedback of the model.

        This kind of adaptation will affect the rgc stage.

        Parameters
        ----------
        rgc_feedback : string, optional
            Multiplicative feedback mechanism of the RGC. Options:
                None: No global feedback.

                'exp_sig': Exponentially decaying feedback filter combined with
                a sigmoidal feedback nonlinearity.
        length : int
            Lenght of the global feedback filter in frames.
        exp_time_constant : float, optional
            If *rgc_feedback* is 'exp_sig', this parameter defines the time
            constant of the exponential decay in terms of frames.
        sig_middle : float, optional
            If *rgc_feedback* is 'exp_sig', this parameters defines the middle
            point of the sigmoidal feedback nonlinearity.
        sig_steepness : float, optional
            If *rgc_feedback* is 'exp_sig', this parameter defines the
            steepness of the sigmoidal feedback nonlinearity.
        """

        if rgc_feedback is None:
            self.global_feedback_filter = np.zeros(length)

            def nonlinearity(signal):
                return 1

            self.global_feedback_nonlinearity = nonlinearity
        elif rgc_feedback == 'exp_sig':
            self.global_feedback_filter = np.exp(-np.arange(length)
                                                 / exp_time_constant)

            def nonlinearity(signal):
                return 1 / (1 + np.exp(sig_steepness * (signal - sig_middle)))

            self.global_feedback_nonlinearity = nonlinearity

        # If this function has been used to change (i.e. not initialize) the
        # feedback, then the stimulus-dependent history has to be cleared
        if self.initialized:
            self.clear_history()


    def set_spiking(self, rgc_spiking, **kwargs):
        """ Set the spiking process.

        Parameters
        ----------
        rgc_spiking : string
            Spiking process of the RGC. Options:
                None: No random spiking process, model outputs firing rate.

                'poisson': Spiking via Poisson distribution. Model outputs
                number of spikes.
        **kwargs
            Additional keyword arguments. Check the global variable
            *DEFAULT_PARAMS* for more information.
        """

        self.params.update(kwargs)

        self.spiking = Spiking_none
        if rgc_spiking == 'poisson':
            # First find out how negative the response to full-field black is
            # so that negative Poisson rates can be prevented
            stim = -np.ones((self.resolution, self.resolution))
            shift = np.abs(self.response_to_flash(stim))
            # Next find out the response to a white flash in order to calibrate
            # the number of spikes
            if self.params['poisson_coefficient'] == 'realistic':
                stim = np.zeros((self.resolution, self.resolution))
                quarter = int(round(self.resolution/4))
                stim[quarter:-quarter, quarter:-quarter] = 1
                resp = self.response_to_flash(stim)
                coefficient = 50 / (resp + shift)
                self.spiking = Spiking_poisson(coefficient, shift)
            else:
                self.spiking = Spiking_poisson(self.params['poisson_coefficient'],
                                               shift)


    def set_resolution(self, new_resolution):
        """ Change the resolution without changing anything else.

        Parameters
        ----------
        new_resolution : int
            New width and height in pixels of the square area in which the
            receptive field of the RGC lies.
        """

        if self.scenario == 'realistic gauss':
            # Rescaling subunit parameters
            self.subunit_params[:, :4] *= new_resolution/self.resolution
            # Creating the subunit arrays again
            self.subunits = np.zeros((self.num_subunits, new_resolution,
                                      new_resolution))
            for i in range(self.num_subunits):
                self.subunits[i] = Gaussian_array(new_resolution,
                                                  *self.subunit_params[i])
            self.resolution = new_resolution
            # Clear the stimulus history
            self.clear_history()
        # All other subunit layouts can just be called again as they are
        # deterministic
        else:
            self.resolution = new_resolution
            self.set_subunits(self.scenario)


    def set_circular(self):
        """ Change all Gaussian ellipses to be circular.

        Ellipse axes are set to the mean of the previous axes' length.
        Positions are not changed.
        """

        # Store previous ellipse parameters so they can be reset
        self.previous_ellipse = np.copy(self.subunit_params)
        means = np.mean(self.subunit_params[:, 2:4], axis=1)
        self.subunit_params[:, 2] = means
        self.subunit_params[:, 3] = means
        self.subunits = np.zeros((self.num_subunits, self.resolution,
                                  self.resolution))
        for i in range(self.num_subunits):
            self.subunits[i] = Gaussian_array(self.resolution,
                                              *self.subunit_params[i])

        # Clear the stimulus history
        self.clear_history()


    def set_elliptic(self):
        """ Resets the Gaussian ellipses to their state before *set_circular*
        was called.
        """

        self.subunit_params = self.previous_ellipse
        self.subunits = np.zeros((self.num_subunits, self.resolution,
                                  self.resolution))
        for i in range(self.num_subunits):
            self.subunits[i] = Gaussian_array(self.resolution,
                                              *self.subunit_params[i])

        # Clear the stimulus history
        self.clear_history()


    def clear_history(self):
        """ Clears all stimulus-dependent history of the model."""

        # This variable stores the multiplication of the subunits with the
        # stimulus history to avoid computing this multiple times.
        self.sub_views_history = np.zeros((self.num_subunits,
                                           self.temporal_filter.size))
        # This variable stores the RGC's response history
        self.rgc_response_history = np.zeros(self.global_feedback_filter.size)
        # This variable stores the subunit's response history
        self.subunit_response_history = np.zeros((self.num_subunits,
                                                  self.local_feedback_filter.size))


    def get_receptive_field(self):
        """ Compute the receptive field of the model.

        Receptive field is computed by flashing a white pixel at each location
        and taking the spike-free response as the receptive field strength at
        that location.

        Returns
        -------
        ndarray
            2D array containing the receptive field. Shape is *self.resolution*
            x *self.resolution*.
        ndarray
            1D array containing the parameters of a Gaussian fitted to the RF
            in the order x-position, y-position, sigma_x, sigma_y, angle
            (radians).
        """

        # Spiking is deactivated to get a noise-free RF
        spiking = self.spiking
        self.set_spiking(None)
        # Test RF
        rf = np.empty((self.resolution, self.resolution))
        for x in range(self.resolution):
            for y in range(self.resolution):
                stimulus = np.zeros((self.resolution, self.resolution))
                stimulus[x, y] = 1
                rf[x, y] = self.response_to_flash(stimulus)
        # Fit ellipse to RF
        xx, yy = np.mgrid[0:self.resolution, 0:self.resolution]
        pixels = np.transpose(np.vstack([xx.ravel(), yy.ravel()]))
        initial_guess = (self.resolution/2, self.resolution/2,
                         self.resolution/6, self.resolution/6, 0)
        rf_params, _ = opt.curve_fit(TwoD_Gaussian, pixels,
                                     rf.ravel()/np.sum(rf), p0=initial_guess)
        # Reset spiking
        self.spiking = spiking

        return rf, rf_params


    def plot_subunits(self, savepath=None):
        """ Plot the subunits of the model in separate plots.

        Parameters
        ----------
        savepath : string, optional
            If provided, the plot is not shown in terminal but saved at the
            given location.
        """

        num_edge = int(np.ceil(np.sqrt(self.num_subunits)))
        fig, axs = plt.subplots(nrows=num_edge, ncols=num_edge,
                                subplot_kw={'xticks': [], 'yticks': []})
        for i, ax in enumerate(np.ravel(axs)):
            if i < self.num_subunits:
                ax.imshow(np.transpose(self.subunits[i]), origin='lower',
                          cmap='gray_r')
            else:
                ax.axis('off')
        if savepath is None:
            plt.show()
        else:
            plt.savefig(savepath, dpi=300)
        plt.clf()
        plt.close('all')


    def plot_subunit_ellipses(self, savepath=None):
        """ Plot the 1.5-sigma subunit ellipses in one plot.

        Only available if subunits are Gaussians.

        Parameters
        ----------
        savepath : string, optional
            If provided, the plot is not shown in terminal but saved at the
            given location.
        """

        fig, ax = plt.subplots(figsize=(4, 4))
        fig.suptitle("Subunit layout")
        ax.set_xlim((0, self.resolution-1))
        ax.set_ylim((0, self.resolution-1))
        for i in range(self.num_subunits):
            rf_ellipse = Ellipse(self.subunit_params[i, 0:2],
                                 self.subunit_params[i, 2]*2*1.5,
                                 self.subunit_params[i, 3]*2*1.5,
                                 -self.subunit_params[i, 4]*180/np.pi,
                                 fill=False)
            ax.add_patch(rf_ellipse)
        _, rf_params = self.get_receptive_field()
        rf_ellipse = Ellipse(rf_params[0:2],
                             rf_params[2]*2*1.5,
                             rf_params[3]*2*1.5,
                             -rf_params[4]*180/np.pi,
                             fill=False, color='red')
        ax.add_patch(rf_ellipse)
        if savepath is None:
            plt.show()
        else:
            plt.savefig(savepath, dpi=300)
        plt.clf()
        plt.close('all')


    def plot_receptive_field(self, savepath=None):
        """ Plot the receptive field of the model.

        Parameters
        ----------
        savepath : string, optional
            If provided, the plot is not shown in terminal but saved at the
            given location.
        """

        rf, rf_params = self.get_receptive_field()
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.suptitle("Receptive field of RGC")
        ax.imshow(np.transpose(rf), origin='lower', cmap='gray_r')
        rf_ellipse = Ellipse(rf_params[0:2],
                             rf_params[2]*2*1.5,
                             rf_params[3]*2*1.5,
                             -rf_params[4]*180/np.pi,
                             fill=False, color='red')
        ax.add_patch(rf_ellipse)
        if savepath is None:
            plt.show()
        else:
            plt.savefig(savepath, dpi=300)
        plt.clf()
        plt.close('all')


    def plot_temporal_filter(self, savepath=None):
        """ Plot the temporal filter of the subunits.

        Parameters
        ----------
        savepath : string, optional
            If provided, the plot is not shown in terminal but saved at the
            given location.
        """

        t_filter = self.temporal_filter
        fig, ax = plt.subplots()
        fig.suptitle("Temporal filter of subunits")
        ax.set_ylabel("Stimulus")
        ax.set_xlabel("Frames in the past")
        ax.set_ylim(-1.1*np.max(np.abs(t_filter)),
                    1.1*np.max(np.abs(t_filter)))
        ax.set_xlim(t_filter.size - 1, 0)
        ax.grid()
        ax.plot(t_filter)
        if savepath is None:
            plt.show()
        else:
            plt.savefig(savepath, dpi=300)
        plt.clf()
        plt.close('all')


    def plot_local_feedback_filter(self, savepath=None):
        """ Plot the local feedback filter of the model affecting the subunits.

        Parameters
        ----------
        savepath : string, optional
            If provided, the plot is not shown in terminal but saved at the
            given location.
        """

        fb_filter = self.local_feedback_filter
        fig, ax = plt.subplots()
        fig.suptitle("Local feedback filter of model")
        ax.set_ylabel("Weight")
        ax.set_xlabel("Frames in the past")
        ax.set_ylim(-1.1*np.max(np.abs(fb_filter)),
                    1.1*np.max(np.abs(fb_filter)))
        ax.set_xlim(fb_filter.size - 1, 0)
        ax.grid()
        ax.plot(fb_filter)
        if savepath is None:
            plt.show()
        else:
            plt.savefig(savepath, dpi=300)
        plt.clf()
        plt.close('all')


    def plot_global_feedback_filter(self, savepath=None):
        """ Plot the global feedback filter of the RGC.

        Parameters
        ----------
        savepath : string, optional
            If provided, the plot is not shown in terminal but saved at the
            given location.
        """

        fb_filter = self.global_feedback_filter
        fig, ax = plt.subplots()
        fig.suptitle("Global feedback filter of model")
        ax.set_ylabel("Weight")
        ax.set_xlabel("Frames in the past")
        ax.set_ylim(-1.1*np.max(np.abs(fb_filter)),
                    1.1*np.max(np.abs(fb_filter)))
        ax.set_xlim(fb_filter.size - 1, 0)
        ax.grid()
        ax.plot(fb_filter)
        if savepath is None:
            plt.show()
        else:
            plt.savefig(savepath, dpi=300)
        plt.clf()
        plt.close('all')


    def response_to_flash(self, image):
        """ Calculate the response of the model to a flash of the stimulus.

        In this response, any feedback mechanisms and the response history are
        irrelevant.

        Parameters
        ----------
        image : ndarray
            Stimulus to be flashed. Must have the same size as given in
            initialization of the model. Entry values -1 mean full black, 0
            means grey, +1 means full white.

        Returns
        -------
        response
            Response strength to the stimulus. Proportional to firing rate.
        """

        sub_sums = np.sum(np.multiply(self.subunits, image), axis=(1, 2))
        sub_responses = self.subunit_nl(sub_sums)
        rgc_sum = np.sum(sub_responses * self.weights)
        rgc_response = self.output_nl(rgc_sum)
        spikes = self.spiking(rgc_response)

        return spikes


    def response_to_frame(self, frame):
        """ Calculate the response of the model during the next frame of the
        stimulus.

        Grey screen is assumed at the beginning of the simulation. Feedback
        mechanisms and response history play a role.

        Parameters
        ----------
        frame : ndarray
            Stimulus in the next frame. Must have the same size as given in
            initialization of the model. Entry values -1 mean full black, 0
            means grey, +1 means full white.

        Returns
        -------
        response
            Response strength during this frame. Proportional to firing rate.
        """

        # Compute subunit views on new frame
        sub_views = np.sum(np.multiply(self.subunits, frame), axis=(1, 2))
        # Update the sub_views history
        self.sub_views_history[:, 1:] = self.sub_views_history[:, :-1]
        self.sub_views_history[:, 0] = sub_views
        # Apply temporal filter
        sub_sums = np.sum(np.multiply(self.sub_views_history,
                                      self.temporal_filter),
                          axis=1)
        # Compute and apply the local feedback
        local_feedback = np.sum(np.multiply(self.local_feedback_filter,
                                            self.subunit_response_history),
                                axis=1)
        local_feedback = self.local_feedback_nonlinearity(local_feedback)
        sub_sums *= local_feedback
        # Apply subunit nonlinearity
        sub_responses = self.subunit_nl(sub_sums)
        # Update subunit response history
        self.subunit_response_history[:, 1:] = self.subunit_response_history[:, :-1]
        self.subunit_response_history[:, 0] = sub_responses
        # Sum subunit responses
        rgc_sum = np.sum(sub_responses * self.weights)
        # Compute and apply the global feedback
        global_feedback = np.sum(self.global_feedback_filter
                                 * self.rgc_response_history)
        global_feedback = self.global_feedback_nonlinearity(global_feedback)
        rgc_sum *= global_feedback
        # Apply RGC nonlinearity
        rgc_response = self.output_nl(rgc_sum)
        # Update RGC response history
        self.rgc_response_history[1:] = self.rgc_response_history[:-1]
        self.rgc_response_history[0] = rgc_response
        # Apply spiking process
        spikes = self.spiking(rgc_response)

        return spikes


###############################################################################
# Testing
###############################################################################
if __name__ == '__main__':
    # Instantiation of a Subunit_Model object. For reference, all available
    # parameters are listed in this command. Check the documentation for infos
    # about them. Additionally, there are keyword arguments. Check the global
    # variable *DEFAULT_PARAMS* for more info.
    rgc = Subunit_Model(resolution=100, scenario='realistic gauss',
                        subunit_nonlinearity='threshold-quadratic',
                        subunit_feedback=None, subunit_weights='equal',
                        rgc_nonlinearity=None, rgc_feedback=None,
                        rgc_spiking=None)
    # Plotting the receptive field of the model.
    rgc.plot_receptive_field("Receptive field")
    # Plotting the 1.5-sigma subunit ellipses of the model.
    rgc.plot_subunit_ellipses("Subunits")
