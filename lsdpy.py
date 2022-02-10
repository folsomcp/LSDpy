#!/usr/bin/python3
#
# Main LSD code

__version__ = "0.3.3"

import numpy as np
#from scipy.sparse import diags, issparse, csr_matrix, csc_matrix
#from numpy.linalg import inv
import lsdpFunc as lsdpFunc

import scipy.constants
c = scipy.constants.c*1e-3

#Read command line arguments (optional)
import argparse
parser = argparse.ArgumentParser(description='Run Least Squares Deconvolution, most input parameters are in read from the file inlsd.dat.')
parser.add_argument("observation", nargs='?', default='', help='Observed spectrum file. If none is given, defaults to the value specificed in inlsd.dat')
parser.add_argument("output", nargs='?', default='prof.dat', help='Output file name for the LSD profile. If none is given, defaults to prof.dat')
parser.add_argument("-m", "--mask",dest='mask',  default='', help='Mask for the LSD calculation. If none is given, defaults to the value specificed in inlsd.dat')
args = parser.parse_args()

#Read input data and observation
params = lsdpFunc.paramsLSD('inlsd.dat')

if args.observation != '':
    params.inObs = args.observation
outputProfName = args.output
if args.mask != '':
    params.inMask = args.mask

obs = lsdpFunc.observation(params.inObs)
# Work in shifted I units (i.e. 1-I => continuum @ 0)
obs.specI = 1.-obs.specI

mask = lsdpFunc.mask(params.inMask)
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
print('Average observed spec velocity spacing: {:.6f} km/s'.format(obsAvgVel))



#Print some information about the LSD profile to be calculated
#And output some basic error checking
print('using a {:n} point profile with {:.6f} km/s pixels'.format(prof.npix, params.pixVel))

if(obsAvgVel*0.9 > params.pixVel):
    print('warning: profile velocity spacing small - profile may be poorly constrained')
if(obsAvgVel*2.0 < params.pixVel):
    print('warning: profile velocity spacing large - profile may be under sampling')

nLinesTooLow = np.where(mask.wl < obs.wl[0])[0].shape[0]
nLinesTooHigh =  np.where(mask.wl > obs.wl[-1])[0].shape[0]
if(nLinesTooLow > 0):
    print('WARNING: {:n} lines in mask falls below observed range'.format(nLinesTooLow))
if(nLinesTooHigh > 0):
    print('WARNING: {:n} lines in mask falls above observed range'.format(nLinesTooHigh))

#Print average line mask information
useMask = np.where(mask.iuse != 0)
print('mean mask depth {:.6f} wl {:.3f} Lande {:.6f} (from {:n} lines)'.format(np.average(mask.depth[useMask]), np.average(mask.wl[useMask]), np.average(mask.lande[useMask]), mask.wl.shape[0]))
print('mean mask norm weightI {:.6f} weightV {:.6f}'.format(np.average(mask.weightI[useMask]), np.average(mask.weightV[useMask])))

#Run the actual fitting
#This is the major function, which does the fitting and takes most time
chi2I, chi2V, chi2N1 = lsdpFunc.lsdFitSigmaClip(obs, mask, prof, params)

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

prof.save(outputProfName)
if(params.fLSDPlotImg != 0):
    prof.lsdplot(params.outPlotImgName)



