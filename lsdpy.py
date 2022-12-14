#!/usr/bin/python3
#
# Main LSD code

__version__ = "0.4.3"

import numpy as np
import lsdpFunc as lsdpFunc

import scipy.constants
c = scipy.constants.c*1e-3

def main(observation=None, mask=None, outName='prof.dat',
         velStart=None, velEnd=None, velPixel=None, 
         normDepth=None, normLande=None, normWave=None,
         removeContPol=None, trimMask=None, sigmaClipIter=None, sigmaClip=None, 
         interpMode=None, fSaveModelS=None, outModelName='',
         fLSDPlotImg=None, fSavePlotImg=None, outPlotImgName=None):
    """Run the LSD code.  
    
    Any arguments not specified will be read from the file inlsd.dat.
    The file inlsd.dat is optional, but if the file dose not exist and 
    any arguments are 'None', the program will error and halt.
    Some arguments have default values, which will be used if they are not
    explicitly specified and if the inlsd.dat file is missing.
    
    Arguments are: 
    :param observation:   name of the input observed spectrum file
    :param mask:          name of the input LSD mask file
    :param outName:       name of the output LSD profile (Default = 'prof.dat')
    :param velStart:      float, starting velocity for the LSD profile (km/s)
    :param velEnd:        float, ending  velocity (km/s)
    :param velPixel:      float, velocity pixel size (km/s)
    :param normDepth:     float, normalizing line depth
    :param normLande:     float, normalizing effective Lande factor
    :param normWave:      float, normalizing wavelength
    :param removeContPol: int, flag for whether continuum polarization is 
                          subtracted from the LSD profile (0=no, 1=yes)
                          (Default = 1)
    :param trimMask:      int, flag for whether very closely spaced lines 
                          should be removed from the line mask (0=no, 1=yes)
                          (Default = 0)
    :param sigmaClipIter: int, number of iterations for sigma clipping, 
                          rejecting possible bad pixels based on the fit to
                          Stokes I. Set to 0 for no sigma clipping.
                          (Default = 0, no sigma clipping)
    :param sigmaClip:     float, if sigma clipping, reject pixels where the
                          observation differs from the model by more than this
                          number of sigma.  Should be a large value so only very
                          bad pixels are rejected.
                          (Default = 500.)
    :param interpMode:    int, mode for interpolating the model on to the
                          observation during LSD 0 = nearest neighbour,
                          1 = linear interpolation.
                          (Default = 1)
    :param fSaveModelS:   int, flag for whether to save a copy of the 'model'
                          spectrum from LSD (i.e. the line mask convolved with
                          the LSD profile).  If != 0 this will also return the
                          'model' spectrum from the function call.
                          (Default = 0)
    :param outModelName:  name of the file for the output model spectrum 
                          (if saved) If this is '' and fSaveModelS != 0 a model 
                          will be generated but not saved to file.
                          (Default = '')
    :param fLSDPlotImg:   int, flag for whether to plot the LSD profile
                          (using matplotlib) (0=no, 1=yes)
                          (Default = 1)
    :param fSavePlotImg:  int, flag for whether to save the plot of the 
                          LSD profile (0=no, 1=yes)
                          Default = 0)
    :param outPlotImgName: name of the plotted figure of the LSD profile (if saved)
                          (Default = 'figProf.pdf')
    :rtype: Returns the calculated LSD profile, as a tuple of numpy arrays
            (velocity, Stokes I, error on I, Stokes V, error on V,
            Null 1, error on Null 1).
    """
    
    #Read input data
    params = lsdpFunc.paramsLSD('inlsd.dat')
    
    #Use passed parameters, if they exist
    if(observation != None):    params.inObs = observation
    if(mask != None):           params.inMask = mask
    if(outName != None):        outputProfName = outName
    if(velStart != None):       params.velStart = velStart
    if(velEnd != None):         params.velEnd = velEnd
    if(velPixel != None):       params.pixVel = velPixel
    if(normDepth != None):      params.normDepth = normDepth
    if(normLande != None):      params.normLande = normLande
    if(normWave != None):       params.normWave = normWave
    if(removeContPol != None):  params.removeContPol = removeContPol
    if(trimMask != None):       params.trimMask = trimMask
    if(sigmaClipIter != None):  params.sigmaClipIter = sigmaClipIter
    if(sigmaClip != None):      params.sigmaClip = sigmaClip
    if(interpMode != None):     params.interpMode = interpMode
    if(fSaveModelS != None):    params.fSaveModelSpec = fSaveModelS
    if(outModelName != None):   params.outModelSpecName = outModelName
    if(fLSDPlotImg != None):    params.fLSDPlotImg = fLSDPlotImg
    if(fSavePlotImg != None):   params.fSavePlotImg = fSavePlotImg
    if(outPlotImgName != None): params.outPlotImgName = outPlotImgName

    #Check if any important parameters are missing
    if(params.inObs == None or params.inMask == None or params.velStart == None
       or params.velEnd == None or params.pixVel == None
       or params.normDepth == None or params.normLande == None
       or params.normWave == None):
        print('WARNING: missing inlsd.dat!')
        print('ERROR: missing a required input value in lsdpy.main()!')
        print('Halting...')
        import sys
        sys.exit()
    
    #Read the observation
    obs = lsdpFunc.observation(params.inObs)
    # Work in shifted I units (i.e. 1-I => continuum @ 0)
    obs.specI = 1.-obs.specI
    
    #Read the line mask
    mask = lsdpFunc.mask(params.inMask)
    #remove very closely spaced lines in the mask
    if params.trimMask != 0: mask.removePoorLines(params)
    #calculate and normalize line mask weights
    mask.setWeights(params)
    
    #initial setup
    prof = lsdpFunc.prof(params)
    
    #trim the observation's wavelength range
    obs.setInRange(mask, prof)
    
    #Get average observed wavelength spacing (spending a bit of memory for speed)
    wlStep = obs.wlOrig[1:] - obs.wlOrig[:-1]
    indWlSteps = np.where( (wlStep > 0.) & (wlStep < 0.01))[0]
    obsAvgVel = np.average( wlStep[indWlSteps]/obs.wlOrig[indWlSteps]*c )
    print('Average observed spec velocity spacing: {:.6f} km/s'.format(
        obsAvgVel))
    
    
    #Print some information about the LSD profile to be calculated
    #And output some basic error checking
    print('using a {:n} point profile with {:.6f} km/s pixels'.format(
        prof.npix, params.pixVel))
    
    if(obsAvgVel*0.9 > params.pixVel):
        print('warning: profile velocity spacing small - profile may be poorly constrained')
    if(obsAvgVel*2.0 < params.pixVel):
        print('warning: profile velocity spacing large - profile may be under sampling')
    
    nLinesTooLow = np.where(mask.wl < obs.wl[0])[0].shape[0]
    nLinesTooHigh =  np.where(mask.wl > obs.wl[-1])[0].shape[0]
    if(nLinesTooLow > 0):
        print('WARNING: {:n} lines in mask falls below observed range'.format(
            nLinesTooLow))
    if(nLinesTooHigh > 0):
        print('WARNING: {:n} lines in mask falls above observed range'.format(
            nLinesTooHigh))
    
    #Print average line mask information
    useMask = np.where(mask.iuse != 0)
    print('mean mask depth {:.6f} wl {:.3f} Lande {:.6f} (from {:n} lines)'.format(
        np.average(mask.depth[useMask]), np.average(mask.wl[useMask]),
        np.average(mask.lande[useMask]), mask.wl.shape[0]))
    print('mean mask norm weightI {:.6f} weightV {:.6f}'.format(
        np.average(mask.weightI[useMask]), np.average(mask.weightV[useMask])))
    
    #Run the actual fitting
    #This is the major function, which does the fitting and takes most time
    chi2I, chi2V, chi2N1, modelSpec = lsdpFunc.lsdFitSigmaClip(obs, mask, prof, params)
    
    #Print chi^2, scale error bars if chi^2 is larger,
    #and remove continuum polarization if desired
    #May be statistically more accurate to use obs.nPixUsed than obs.wl.shape[0]
    print('I reduced chi2 {:.4f} (chi2 {:.2f} constraints {:n} dof {:n})'.format(
        chi2I/(obs.wl.shape[0]-prof.npix), chi2I, obs.wl.shape[0], prof.npix))
    lsdpFunc.scaleErr(prof.specSigI, chi2I, obs.wl.shape[0], prof.npix)
    
    print('V reduced chi2 {:.4f} (chi2 {:.2f} constraints {:n} dof {:n})'.format(
        chi2V/(obs.wl.shape[0]-prof.npix), chi2V, obs.wl.shape[0], prof.npix))
    lsdpFunc.scaleErr(prof.specSigV, chi2V, obs.wl.shape[0], prof.npix)
    lsdpFunc.zeroProf(prof.specV, prof.specSigV, params.removeContPol)
    
    print('N1 reduced chi2 {:.4f} (chi2 {:.2f} constraints {:n} dof {:n})'.format(
        chi2N1/(obs.wl.shape[0]-prof.npix), chi2N1, obs.wl.shape[0], prof.npix))
    lsdpFunc.scaleErr(prof.specSigN1, chi2N1, obs.wl.shape[0], prof.npix)
    lsdpFunc.zeroProf(prof.specN1, prof.specSigN1, params.removeContPol)
    
    #check for detections
    lsdpFunc.nullTest(prof)
    
    prof.save(outputProfName, obs.header, params)
    if(params.fLSDPlotImg != 0):
        prof.lsdplot(params.outPlotImgName)
        
    if(params.fSaveModelSpec != 0):
            return (prof.vel, 1.-prof.specI, prof.specSigI, prof.specV, 
                    prof.specSigV, prof.specN1, prof.specSigN1, modelSpec)

    return (prof.vel, 1.-prof.specI, prof.specSigI, prof.specV, prof.specSigV,
            prof.specN1, prof.specSigN1)


# Boilerplate for running the main function #
if __name__ == "__main__":

    #Read command line arguments (optional)
    import argparse
    parser = argparse.ArgumentParser(description='Run Least Squares Deconvolution, most input parameters are in read from the file inlsd.dat.')
    parser.add_argument("observation", nargs='?', default='', help='Observed spectrum file. If none is given, defaults to the value specified in inlsd.dat')
    parser.add_argument("output", nargs='?', default='prof.dat', help='Output file name for the LSD profile. If none is given, defaults to prof.dat')
    parser.add_argument("-m", "--mask",dest='mask',  default='', help='Mask for the LSD calculation. If none is given, defaults to the value specified in inlsd.dat')
    args = parser.parse_args()

    observation = None
    if args.observation != '':
        observation = args.observation
    outName = args.output
    mask = None
    if args.mask != '':
        mask = args.mask
    
    #Run the LSD code
    main(observation=observation, outName=outName, mask=mask)
