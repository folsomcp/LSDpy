# LSDpy
A Least Squares Deconvolution in Python for analysis of stellar spectra.

## Basic use
The main executable is lsdpy.py, there are a set of control parameters specified in the file inlsd.dat

To use the code you can edit the file inlsd.dat, specify the observation file to process, the line mask to use, and any other parameters you need to modify.  Then run lsdpy.py and it should generate an output LSD profile called prof.dat.  If all went well the prof.dat file will have columns of velocity Stokes I, error on Stokes I, Stokes V, error on Stokes V, the null (N1), and errors on the null.

You can also run the code from the command line like:
```
python lsdpy.py [observation_file] [output_profile] -m [line_mask]
```
Any other parameters will be still read from the inlsd.dat file.  The observation and line mask can optionally be specified with inlsd.dat rather than the command line.

## Input parameters
These are specified in the inlsd.dat file.

### Input observation
The input observation file name, this can include a path to the file.
The observation file is assumed to have columns of either: wavelength, I/Ic, V/Ic, Null1/Ic, Null2/Ic, uncertainty, or alternately: wavelength, I/Ic, uncertainty.  All normalized to the continuum.  This follows the output of the Libre-ESPRIT data reduction pipeline.  The second polarimetric Null is usually not used.  The uncertainties should be 1 sigma, ideally from propagating uncertainties through the data reduction.

### Line mask
The line mask file name, this can include a path to the file. 
This should have columns of wavelength, species code (atomic number + ionization state/100, where neutrals have an ionization of 0), line depth at the line center (depth from the continuum), excitation potential of the lower level (in eV), effective Lande factor, and a flag for whether the line is used (1=use, 0=don't).  The code expects to have 1 line of header, usually the number of lines in the mask, but that is not currently read or used for anything.  The excitation potential is never actually used.  The observation wavelength needs to be in the same units as the line mask wavelengths. 

### Start and end velocity (km/s) for LSD profile
The range that the LSD profile will span, in velocity units.  

### Pixel size in velocity (km/s)
The size of a pixel in the LSD profile, in velocity units.
Usually a good choice is the average pixel size of the observation (in km/s), which the code will print to terminal as a diagnostic. 

Choosing a pixel size that is too small may cause oscillations in the LSD profile, usually when the pixel size is smaller than the mean observation pixel size.  This is essentially a sampling rate problem.  Since the observed spectrum pixels are roughly (although not exactly) the same size in velocity, sampling it with model LSD pixels that are too small effectively aliases some power into high frequencies, creating that oscillation.

Using LSD pixels larger than the observed pixels generally seems to be safe, although I recommend multiples of the observed pixel size when larger LSD pixels are desired.

### Mask/profile normalization parameters
Normalizing line depth, Lande factor, and wavelength (nm)

These parameters scale the LSD profile, or equivalently normalize the line mask, by those values.  Just depth (the 1st value) is used for Stokes I, and (depth)x(Lande_factor)x(wavelength) for Stokes V.  The LSD profile amplitudes depend on the scaling of the weights in the LSD mask, and mathematically that scaling is arbitrary.  These parameters control that scaling explicitly.  For a detailed explanation see Kochukhov et al. (2010, A&A 524, A5), particularly Sect 2.5 and 2.6.

The most important point is that the LSD profile should theoretically behave like a line with these parameters for Lande factor, wavelength, and central depth (of an unbroadened line), to the extent that the approximations in LSD hold true.  A popular choice is to set those parameters to the average from the line mask, but one can also use some round 'reasonable' values.  Just remember what you used when you measure longitudinal fields or attempt ZDI.  

### Remove continuum polarization
This removes continuum polarization in the Stokes V profile by forcing the average of Stokes V to be 0.  Set the flag to 1 to enable this, 0 to disable.  In a well behaved observation this should not be necessary, but instrumental or data reduction errors may introduce an offset in Stokes V.  Be a bit careful with this, as it is usually an imperfect correction for an error that occurred elsewhere.

### Sigma clip to reject bad pixels
A sigma clipping routine can be applied to the observation, if the number of iterations is 0 it is disable entirely.  Uses the sigma value (number of sigma discrepancy between a model and observed pixel) beyond which a pixel is rejected, and and a number of iterations to repeat the sigma clipping.  

The sigma clipping routine may help improve the line profile shape in Stokes I.  However, the more iterations the slower the code runs.  And if too many points in the observation get rejected in the sigma clipping (too small a sigma clipping limit) it reduces the SNR in Stokes V.

For most issues it is better to fine tune the line mask (e.g. rejecting problematic lines, or lines in problem regions of the observation) rather than relying on this sigma clipping routine.  If you do need to use it, it is often good to start with the sigma clipping turned off, then turn it on to fine tune the profile.  The LSD code prints how many pixels it has reject to the terminal.  Often rejecting some pixels helps, but rejecting more than 5% of pixels may be a bad idea (or a sign that you need a better line mask).  

### Interpolation mode
Flag for how the model is interpolated onto observed wavelengths, 1=linear interpolation (recommended!), 0=nearest neighbor interpolation.

A somewhat experimental feature to change how the model LSD spectrum is interpolated on to the wavelengths of the observed spectrum.  The code can linearly interpolate between two model points to get the exact wavelength for an observed point (this is the usual way of doing things).  But the code can also just take the nearest model point in wavelength to the observed point ('nearest neighbor interpolation').  Doing that seems to increase the effective SNR of the LSD profile, but also seems to smear the profile by a fraction of a pixel.  Unless you are interested in experimenting, just leave the flag at 1 and stick with linear interpolation.

If you set the "interpolation mode" flag to 0 (for nearest neighbor), that can mitigate the problem of oscillations in the profile due to too small LSD pixels.  But it doesn't completely solve it, so choosing a better pixel size is usually the better approach.

### Save LSD model spectrum?
Controls if and where to save the LSD fit to the observation, 1 = save 0 = don't save, followed by a file name.

The model spectrum produced by LSD is the convolution of the LSD profile and the line mask.  It is the best fit in a least squares sense, although generally does a poor job of fully reproducing the Stokes I spectrum.  The file has columns of wavelength, I/Ic, V/Ic, Null1/Ic.  This can be useful for evaluating the quality of a line mask by comparing this to the observation.

### Plot LSD profile?
The code can optionally plot the LSD profile with matplotlib.  The values are a flag to plot the profile (1 = yes, 0 = no), a flag to optionally save the plot (1 = yes, 0 = no), and a file name for the saved plot (only used if save plot = 1).  The save plot file name needs to end with a suffix recognized by matplotlib (usually .png .eps or .pdf).  If matplotlib is not installed, the code should still run if you set the plotting to 0.  


## Terminal output
(This may be subject to change as the code evolves!)
A brief description of some of the text output to the terminal.  The output is mostly self explanatory, I hope.

The average pixel sized of the observed spectrum, in velocity units, is provided.

The mean mask parameters output are simple unweighted averages.

For each iteration of the sigma clipping routine the number of rejected pixels and number of considered pixels is printed (sigma clip rejecting xxx points of yyy)

Stokes I, V, and N error bars are re-scaled by the square root of the reduced chi^2 of the fit to the observation, if this would increase the errorbar (if the reduced chi^2 is > 1).  The reduced chi^2 and re-scaling value are printed to the terminal ("Rescaling error bars by: ", or "Not re-scaling error bars" if the scaling value is less than 1.0).  This typically leads to substantially increased error bars for Stokes I.

The continuum polarization level is printed for V and N "note, profile continuum pol:"

An estimate of the velocity range that the line spans is printed "line range estimate".  It is based on Stokes I, if the line shape or continuum are particularly strange then the estimate may be wrong.  The line range is taken to be where the profile drops 4 sigma below the continuum level.  The continuum level is estimated from the first and last 20 points in the profile.  The line range is used to compute detections statistics.

The detections statistics are printed for Stokes V, and then N.  The chi^2 of the null model (a flat line) fit to the observation, for the portion inside the line, is printed.  The code also prints the detection probability, and false alarm probability, for that chi^2.  The detection probability is the probability that the null model disagrees with the observation.  Then the code prints the same information calculated outside the line, as a diagnostic test.  This is then repeated for the null.  When operating well, there would be a detection inside the line for Stokes V, but no detection outside the line in V, inside the line in N or outside the line in N.  Currently the code uses detection probabilities of 0.0-0.99 as "non-detection", 0.99-0.9999 as "marginal detections", and > 0.9999 as "definite detections". 
