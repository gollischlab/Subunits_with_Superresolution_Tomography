# Subunits_with_Superresolution_Tomography
Simulations of subunit model of a retinal ganglion cell in response to Mexican-hat-like bar stimuli and reconstruction of subunit layout via inverse Radon transform.

Mexican_Tomography.py can be run to perform a simulation of a subunit model under stimulation with bars (with sidebands) at different positions and different angles. This yields a sinogram and, subsequently, a reconstruction via filtered backprojection.

Running the file creates a subfolder "Mexican Tomography" with a bunch of plots, showing the layout of subunits, the simulated receptive field, an example of the spatial stimulus layout (for one position and one angle), sinograms computed without spiking stochasticity (= infinite data), computed from simulated spike counts (Poisson statistics), and the latter filtered (smoothed) across positions. And then FBP reconstructions from all three cases.

Subunit_Model.py contains the code for the model simulations that are called from Mexican_Tomography.py. A variety of options are supplied to use specific layouts of subunits, subunit shapes, subunit nonlinearities etc.
