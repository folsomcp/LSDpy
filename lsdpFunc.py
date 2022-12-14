#LSD helper functions
import numpy as np
import scipy.special as specialf
from scipy.sparse import diags, spdiags, dia_matrix, csr_matrix, csc_matrix, issparse, coo_matrix, dok_matrix, lil_matrix
from numpy.linalg import inv

import scipy.constants
c = scipy.constants.c*1e-3

class paramsLSD:
    def __init__(self, fname):
        #Read in most information controlling how the program runs

        #Check if the input file exists, if it doesn't then return some
        #reasonable default values where possible.
        #(Other values are intended to be set by the call to lsdpy.main() )
        try:
            infile = open(fname, 'r')
        except FileNotFoundError:
            self.inObs = None
            self.inMask = None
            self.velStart = None
            self.velEnd = None
            self.pixVel = None
            self.normDepth = None
            self.normLande = None
            self.normWave = None
            self.removeContPol = 1
            self.trimMask = 0
            self.sigmaClipIter = 0
            self.sigmaClip = 500.
            self.interpMode = 1
            self.fSaveModelSpec = 0
            self.outModelSpecName = 'outSpec.dat',
            self.fLSDPlotImg = 1
            self.fSavePlotImg = 0
            self.outPlotImgName = 'figProf.pdf'
            return
        
        i = 0
        for line in infile:
            if (len(line) <= 1):
                continue
            if (line.strip()[0] != '#'):
                if (i==0):
                    self.inObs = line.strip().split()[0]
                elif(i==1):
                    self.inMask = line.strip().split()[0]
                elif(i==2):
                    self.velStart = float(line.split()[0])
                    self.velEnd = float(line.split()[1])
                elif(i==3):
                    self.pixVel = float(line.split()[0])
                elif(i==4):
                    self.normDepth = float(line.split()[0])
                    self.normLande = float(line.split()[1])
                    self.normWave  = float(line.split()[2])
                elif(i==5):
                    self.removeContPol = int(line.split()[0])
                elif(i==6):
                    self.trimMask = int(line.split()[0])
                elif(i==7):
                    self.sigmaClip = float(line.split()[0])
                    self.sigmaClipIter = int(line.split()[1])
                elif(i==8):
                    self.interpMode = int(line.split()[0])
                elif(i==9):
                    self.fSaveModelSpec = int(line.split()[0])
                    if(self.fSaveModelSpec == 1):
                        self.outModelSpecName = line.split()[1]
                    else:
                        self.outModelSpecName = ''
                elif(i==10):
                    self.fLSDPlotImg = int(line.split()[0])
                    self.fSavePlotImg = int(line.split()[1])
                    if(self.fSavePlotImg == 1):
                        self.outPlotImgName = line.split()[2]
                    else:
                        self.outPlotImgName = ''
                else:
                    print('end of parameters file reached, error?')
                i += 1
            
        infile.close()
        if(i < 5):
            print('ERROR: incomplete information read from {:}'.format(fname))
        

class observation:
    def __init__(self, fname):
        #Read in the observed spectrum and save it
        ## Reading manually is ~4 time faster than np.loadtxt for a large files 
        fObs = open(fname, 'r')
        nLines = 0
        #Check if the file starts with data or a header (assume it is two lines)
        line = fObs.readline()
        words = line.split()
        try:
            float(words[0])
            float(words[1])
            float(words[2])
            self.header = '#\n'
            fObs.seek(0)
        except ValueError:
            self.header = line
            fObs.readline()
        
        for line in fObs:
            words = line.split()
            if(nLines == 0):
                ncolumns = len(words)
                if (ncolumns != 6):
                    if(ncolumns == 3):
                        print('Apparent Stokes I only spectrum')
                        print('Generating place holder V and N columns')
                    else:
                        print('{:} column spectrum: unknown format!'.format(ncolumns))
                        import sys
                        sys.exit()
            if len(words) == ncolumns:
                if ncolumns == 6:
                    if(float(words[1]) > 0. and float(words[5]) > 0.):
                        nLines += 1
                elif ncolumns == 3:
                    if(float(words[1]) > 0. and float(words[2]) > 0.):
                        nLines += 1
            else:
                print('ERROR: reading observation, line {:}, {:} columns :\n{:}'.format(nLines, len(words), line))

        self.wlOrig = np.zeros(nLines)
        self.specIOrig = np.zeros(nLines)
        self.specVOrig = np.zeros(nLines)
        self.specN1Orig = np.zeros(nLines)
        self.specN2Orig = np.zeros(nLines)
        self.specSigOrig = np.zeros(nLines)
        
        i = 0
        #rewind to start then advance the file pointer 2 lines
        fObs.seek(0)
        if self.header != '#\n':
            fObs.readline()
            fObs.readline()
        for line in fObs:
            words = line.split()
            if (len(words) == ncolumns and ncolumns == 6):
                if(float(words[1]) > 0. and float(words[5]) > 0.):
                    self.wlOrig[i] = float(words[0])
                    self.specIOrig[i] = float(words[1])
                    self.specVOrig[i] = float(words[2])
                    self.specN1Orig[i] = float(words[3])
                    self.specN2Orig[i] = float(words[4])
                    self.specSigOrig[i] = float(words[5])
                    i += 1
            elif (len(words) == ncolumns and ncolumns == 3):
                if(float(words[1]) > 0. and float(words[2]) > 0.):
                    self.wlOrig[i] = float(words[0])
                    self.specIOrig[i] = float(words[1])
                    self.specSigOrig[i] = float(words[2])
                    self.specVOrig[i] = 0.
                    self.specN1Orig[i] = 0.
                    self.specN2Orig[i] = 0.
                    i += 1
                
        fObs.close()
        
        #Optionally deal with order overlap, or do any trimming?
        
        #Sort the observation so wavelength is always increasing
        self.ind = np.argsort(self.wlOrig)

        self.wl = self.wlOrig[self.ind]
        self.specI = self.specIOrig[self.ind]
        self.specV = self.specVOrig[self.ind]
        self.specN1 = self.specN1Orig[self.ind]
        self.specN2 = self.specN2Orig[self.ind]
        self.specSig = self.specSigOrig[self.ind]
        #Save the number of observed spectral pixels used in the LSD profile
        self.nPixUsed = 0


    def setInRange(self, mask, prof):
        #Get the set of observed pixels in range of lines in the LSD mask.
        
        #add an extra 1 LSD pixel buffer to the range we extract, just in case.
        velStart = prof.vel[0] + (prof.vel[0] - prof.vel[1])
        velEnd = prof.vel[-1] + (prof.vel[-1] - prof.vel[-2])

        wlSort = np.sort(mask.wl)
        maskWlLow = wlSort + velStart/c*wlSort
        maskWlHigh = wlSort + velEnd/c*wlSort

        #For each line in the mask check which observed pixels are in range
        #This is logicaly simpler, but much slower:
        #Set array of bool for pixels in range of a line in the mask (starts False)
        #maskuse = np.zeros_like(self.wl, dtype='bool')
        #for i in range(mask.wl.shape[0]):
        #    maskuse += ( (self.wl >= maskWlLow[i]) & (self.wl <= maskWlHigh[i]) )
        
        #for each observed point, get the nearest line profile (line mask wavelength +/- profile size) start (blue edge)
        #(actually gets where each observed point would fit in the ordered list of line profile starts)
        #(maskWlLow must be sorted.  side only matters if some wl are identical or maskWlLow=wl)
        indNearestStart =  np.searchsorted(maskWlLow, self.wl, side='left') 
        #for each observed point, get the nearest line profile end 
        indNearestEnd =  np.searchsorted(maskWlHigh, self.wl, side='right')
        #observed pixel is only in range of a mask line if it is between the line profile start and end wavelengths
        #i.e. if indNearestStart has incremented for the line but indNearestEnd has not yet incremented
        maskuse = (indNearestStart > indNearestEnd)
                
        self.wl = self.wl[maskuse]
        self.specI = self.specI[maskuse]
        self.specV = self.specV[maskuse]
        self.specN1 = self.specN1[maskuse]
        self.specN2 = self.specN2[maskuse]
        self.specSig = self.specSig[maskuse]

        if(self.wl.shape[0] <= 0):
            print('ERROR: no lines in mask in wavelength range of observation!')
        
        return

    def sigmaClipI(self, prof, MI, sigmaLim):

        ptsBefore = self.wl.shape[0]
        modelSpecI = MI.dot(prof.specI)
        maskuse = (np.abs(self.specI-modelSpecI)/self.specSig < sigmaLim)
        #maskuse = np.where(np.abs(self.specI-modelSpecI)/self.specSig < sigmaLim) #alternate version

        self.wl = self.wl[maskuse]
        self.specI = self.specI[maskuse]
        self.specV = self.specV[maskuse]
        self.specN1 = self.specN1[maskuse]
        self.specN2 = self.specN2[maskuse]
        self.specSig = self.specSig[maskuse]
        ptsAfter = self.wl.shape[0]

        print('sigma clip rejecting {:n} points of {:n}'.format(ptsBefore-ptsAfter, ptsBefore))
        
        return

        
class mask:
    def __init__(self, fname):

        #Columns should be wavelength (nm), element+ionization*0.01, line depth,
        #excitation potential of the lower level, effective Lande factor,
        #and a flag for whether the line is used (1=use, 0=skip).
        #self.wl, self.element, self.depth, self.excite, self.lande, tmpiuse \
        #    = np.loadtxt(fname, skiprows=1, unpack=True)
        tmpMask = np.loadtxt(fname, skiprows=1, unpack=True)

        #Sort the line mask so wavelength is always increasing
        self.ind = np.argsort(tmpMask[0,:])
        
        self.wl = tmpMask[0, self.ind]
        self.element = tmpMask[1, self.ind]
        self.depth = tmpMask[2, self.ind]
        self.excite = tmpMask[3, self.ind]
        self.lande = tmpMask[4, self.ind]
        self.iuse = tmpMask[5, self.ind].astype(int)

        #For speed just reduce the mask to the mask.iuse != 0 parts here
        self.prune()

    def prune(self):
        #Restrict the mask to only lines flagged to be used
        ind2 = np.where(self.iuse != 0)
        self.wl = self.wl[ind2]
        self.element = self.element[ind2]
        self.depth = self.depth[ind2]
        self.excite = self.excite[ind2]
        self.lande = self.lande[ind2]
        self.iuse = self.iuse[ind2]

        
    def setWeights(self, params):
        self.weightI = self.depth / params.normDepth
        self.weightV = self.depth*self.wl*self.lande / (params.normDepth*params.normWave*params.normLande)
        return
    
    def removePoorLines(self, params, fracPix = 1.0, sumDepths=True):
        #Remove nearly digenerate lines from the mask.
        #Reject lines seperated by less than fracPix of an LSD (velocity) pixel.
        depthCutoff = 0.6
        nTrimmed = 0
        for l in range(1,self.wl.shape[0]):
            #This is relativly inefficent but handels unsorted line masks
            #and lines with multiple bad blends.
            if self.iuse[l] == 1:
                deltas = np.abs(self.wl[l] - self.wl)/self.wl[l]*c
                iClose = np.nonzero((deltas < params.pixVel*fracPix) & (self.iuse == 1))[0]
                if iClose.shape[0] > 1:
                    #If other lines are too close to the current line
                    self.iuse[iClose] = 0
                    deepestLine = np.argmax(self.depth[iClose])
                    self.iuse[iClose[deepestLine]] = 1
                    
                    #If we want to sum line depths, limit the maximum depth lines 
                    #can sum to, as a very rough approximation for saturation.
                    if sumDepths:
                        summedDepth = np.sum(self.depth[iClose])
                        if summedDepth < depthCutoff:
                            self.depth[iClose[deepestLine]] = summedDepth
                        else:
                            self.depth[iClose[deepestLine]] = max(depthCutoff, self.depth[iClose[deepestLine]])
                    nTrimmed += np.count_nonzero(iClose) - 1
        if nTrimmed > 0:
            print('Modified line mask, removed {:n} too closely spaced lines'.format(nTrimmed))

        #Apply the changes to the line mask
        self.prune()

        return
    
    
class prof:
    def __init__(self, params):
        self.vel = np.arange(params.velStart, params.velEnd+params.pixVel, params.pixVel, )
        #Alternate wl pixel scheme, gets the velocity range exact, but changes pixel size:
        #self.npix = np.ceil((params.velEnd - params.velStart)/params.pixVel)
        #self.vel = np.linspace(params.velStart, params.velEnd, self.npix, endpoint=True)
        self.npix = self.vel.shape[0]
        self.specI = np.ones(self.npix)
        self.specSigI = np.zeros(self.npix)
        self.specV = np.zeros(self.npix)
        self.specSigV = np.zeros(self.npix)
        self.specN1 = np.zeros(self.npix)
        self.specSigN1 = np.zeros(self.npix)
        self.specN2 = np.zeros(self.npix)
        self.specSigN2 = np.zeros(self.npix)

    def save(self, fname, header=None, params=None):
        #Save the LSD profile to a file.
        #finally convert I from 1-I to full I/Ic units at output
        oFile = open(fname, 'w')

        #Add the mask normalizing parameters to the header
        if params == None:
            headerAdd = ''
        else:
            sFormat = ' normalizing: d={:5.3f} lande={:5.3f} wl={:6.1f} (I norm weight {:5.3f}, V norm weight {:7.3f})\n'
            headerAdd = sFormat.format(params.normDepth, params.normLande,
                                       params.normWave, params.normDepth,
                                       params.normDepth*params.normLande
                                       *params.normWave)
        
        if header == None:
            #The observation reading function will usually provide a placeholder
            #header of '#\n' if the file had no header, so this may be unused. 
            oFile.write('***LSD profile' + headerAdd)
        else:
            oFile.write(header.strip() + headerAdd)
        
        oFile.write(' {:d} 6\n'.format(self.npix))
        for i in range(self.npix):
            oFile.write('{:>12.6f} {:>13.6e} {:>13.6e} {:>13.6e} {:>13.6e} {:>13.6e} {:>13.6e}\n'.format(
                self.vel[i], 1.-self.specI[i], self.specSigI[i], self.specV[i],
                self.specSigV[i], self.specN1[i], self.specSigN1[i]))
        oFile.close()

    def lsdplot(self,fname):
        import matplotlib.pyplot as plt
        
        # Set up axes and put y-axis labels on the right
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, gridspec_kw = {'height_ratios':[1, 1, 3]})
        ax1.yaxis.set_label_position("right")
        ax2.yaxis.set_label_position("right")
        ax3.yaxis.set_label_position("right")

        # y-axis limits for V and N so on same scale - 1.05 time biggest divergence from zero
        plotVLims = np.max([np.abs(1.05*np.min(self.specV)),np.abs(1.05*np.max(self.specV)),np.abs(1.05*np.min(self.specN1)),np.abs(1.05*np.max(self.specN1)), 1.05*np.max(self.specSigV)])

        # Plot V - errorbars and data.  Setting limits to max/min vel and y-axis limit above
        ax1.errorbar(self.vel, self.specV, yerr=self.specSigV, fmt='none', ecolor='r', alpha=0.4)
        ax1.scatter(self.vel, self.specV, marker='.', c='r' )
        ax1.plot([np.min(self.vel),np.max(self.vel)],[0.,0.], 'k--', alpha=0.5)
        ax1.set_xlim(xmin=np.min(self.vel),xmax=np.max(self.vel))
        ax1.set_ylim(ymin=-plotVLims,ymax=plotVLims)
        ax1.set_ylabel('$V/I_c$')

        # Plot N1 - errorbars and data.  Setting limits to max/min vel and y-axis limit above
        ax2.errorbar(self.vel, self.specN1, yerr=self.specSigN1, fmt='none', ecolor='m', alpha=0.4)
        ax2.scatter(self.vel, self.specN1, marker='.', c='m' )
        ax2.plot([np.min(self.vel),np.max(self.vel)],[0.,0.], 'k--', alpha=0.5)
        ax2.set_ylabel('$N/I_c$')
        ax2.set_xlim(xmin=np.min(self.vel),xmax=np.max(self.vel))
        ax2.set_ylim(ymin=-plotVLims,ymax=plotVLims)

        #Optionally, plot smoothed versions of V and N1
        try:
            # Import smoothing filter
            from scipy.signal import savgol_filter
            # Apply Savitzky-Golay smoothing filter to V & N (window=9, order=5 - just random!)
            plotVhat = savgol_filter(self.specV, 9, 5)
            plotNhat = savgol_filter(self.specN1, 9, 5)
            ax1.plot(self.vel, plotVhat, 'r', lw=1.2,
                     label='Circular Polarisation')
            ax2.plot(self.vel, plotNhat, 'm', lw=1.2,
                     label='Null Polarisation Check')
        except:
            plotVhat = self.specV
            plotNhat = self.specN1
        
        # Plot I - errorbars and data.  Only setting x-limits
        ax3.errorbar(self.vel, 1.-self.specI, yerr=self.specSigI, fmt='none', ecolor='b', alpha=0.4)
        ax3.plot(self.vel, 1.-self.specI, 'b', lw=1.2, label='Unpolarised Line Profile')
        ax3.set_ylabel('$I/I_c$')
        ax3.set_xlabel('Velocity $(km s^{-1})$')
        ax3.plot([np.min(self.vel),np.max(self.vel)],[1.,1.], 'k--', alpha=0.5)
        ax3.set_xlim(xmin=np.min(self.vel),xmax=np.max(self.vel))
        
        # fig.tight_layout()
        if fname != '':
            plt.savefig(fname)
        plt.show()
        
    

def buildInvSig2(obs):
    #construct the diagonal matrix of 1/sigma^2, dimension of observation X observation
    #Use a sparse matrix for the nobs x nobs array (otherwise its several Gb!)
    tmp = obs.specSig**(-2)
    sparseS2 = scipy.sparse.diags(tmp, offsets=0)
    
    return sparseS2


def buildM(obs, mask, prof, interpMode):
    #Build the nObsPix x nProfPix matrix of weights connecting LSD pixels to observed pixels
    #Builds I and V matrices at once (more efficient)
    
    #outer loop over lines in the mask version, 2-3x faster
    maskMI = np.zeros((obs.wl.shape[0], prof.npix))
    maskMV = np.zeros((obs.wl.shape[0], prof.npix))
    #Sparse matrices are generally slower here due to the overhead accessing entries

    #calculate wavelengths for the profile at each line in the mask here, since it is reusable
    #wlProf = prof.vel/c*mask.wl[l] + mask.wl[l]
    wlProfA = np.outer(prof.vel/c, mask.wl) + np.tile(mask.wl, (prof.npix,1))  #(prof.npix, mask.wl.shape)

    obs.nPixUsed = 0
    #Nearest neighbor 'interpolation' of model spec on to observed spec
    if(interpMode == 0):
        for l in range(mask.wl.shape[0]):
            #Get observation points in range of this line in the mask
            iObsRange = np.where( (wlProfA[0,l] < obs.wl[:]) & (wlProfA[-1,l] > obs.wl[:]) )
        
            #set up nProf x nObsUsed matrices, one of the used observed pixels repeated for each nProf
            obsWl2 = np.tile(obs.wl[iObsRange], (wlProfA[:,l].shape[0], 1))
            #and one of the wavelengths of the profile (at this line), repeated for each nObsUsed
            wlProf2 = np.tile(wlProfA[:,l,np.newaxis], (1, iObsRange[0].shape[0]))
            #The get the profile pixel closest in wavelength to the used observed pixel, for each nObsUsed
            iProf = np.argmin(np.abs(wlProf2 - obsWl2), axis=0)
            
            maskMI[iObsRange,iProf] += mask.weightI[l]
            maskMV[iObsRange,iProf] += mask.weightV[l]
        
            #Slower by 2x version with a loop
            #for i in iObsRange[0]:
            #    wlProf = wlProfA[:,l]  
            #    
            #    #Use the nearest neighbor model point (column in M) for the observed point (row in M)
            #    iProf = np.argmin(np.abs(wlProf - obs.wl[i]))
            #    
            #    maskMI[i,iProf] += mask.weightI[l]
            #    maskMV[i,iProf] += mask.weightV[l]

    #Linear interpolation of model spec on to observed spec
    elif(interpMode == 1):
        for l in range(mask.wl.shape[0]):            
            #Get observation points in range of this line in the mask
            iObsRange = np.where( (wlProfA[0,l] < obs.wl[:]) & (wlProfA[-1,l] > obs.wl[:]) )

            #set up nProf x nObsUsed matrices, one of the used observed pixels repeated for each nProf
            obsWl2 = np.tile(obs.wl[iObsRange], (wlProfA[:,l].shape[0], 1))
            #and one of the wavelengths of the profile (at this line), repeated for each nObsUsed
            wlProf2 = np.tile(wlProfA[:,l,np.newaxis], (1, iObsRange[0].shape[0]))
            #Get the point in the profile with a wavelength (for this line in the mask) just beyond this observed point, for each nObsUsed
            iProf = np.argmax(wlProf2 > obsWl2, axis=0)  #generates array of bool, returns 1st true

            wlWeight = (obs.wl[iObsRange] - wlProfA[iProf-1,l])/(wlProfA[iProf,l]-wlProfA[iProf-1,l])
            
            maskMI[iObsRange,iProf-1] += mask.weightI[l]*(1.-wlWeight)
            maskMI[iObsRange,iProf] += mask.weightI[l]*wlWeight
            
            maskMV[iObsRange,iProf-1] += mask.weightV[l]*(1.-wlWeight)
            maskMV[iObsRange,iProf] += mask.weightV[l]*wlWeight
            
            #Slower by 2x version with a loop
            #for i in iObsRange[0]:
            #    wlProf = wlProfA[:,l]
            #    
            #    #Linearly interpolate between two model points (columns in M) for the observed point (row in M)
            #    #Get the point in the profile with a wavelength (for this line in the masK) just beyond this observed point
            #    #iProf = np.where(wlProf > obs.wl[i])[0][0] 
            #    #iProf = np.argmax(wlProf > obs.wl[i])  #generates array of bool, returns 1st true
            #    iProf = np.searchsorted(wlProf, obs.wl[i], side='right') #relies on ordered wlProf but is faster
            #    
            #    wlWeight = (obs.wl[i] - wlProf[iProf-1])/(wlProf[iProf]-wlProf[iProf-1])
            #    
            #    maskMI[i,iProf-1] += mask.weightI[l]*(1.-wlWeight)
            #    maskMI[i,iProf] += mask.weightI[l]*wlWeight
            #    
            #    maskMV[i,iProf-1] += mask.weightV[l]*(1.-wlWeight)
            #    maskMV[i,iProf] += mask.weightV[l]*wlWeight


    return maskMI, maskMV


def getChi2(Yo, sparseM, sparseS2, Z):
    # Model the spectrum Y as a convolution of a line pattern (mask) M,
    # and a mean line profile Z.  With matrix multiplication Y = M.Z
    # then: chi^2 = (Yo - M.Z)^T.S^2.(Yo - M.Z)
    # For an observation Yo with S being a square diagonal with 1/sigma errors

    tmpChi = (Yo - sparseM.dot(Z))
    #tmpChi is not sparse, but can still use this syntax to be safe
    chi2 = tmpChi.T.dot(sparseS2.dot(tmpChi))

    return chi2

def saveModelSpec(outModelSpecName, prof, MI, MV, obsWl):
    #Save the LSD model spectrum (i.e. convolution of line mask and LSD profile)
    #for pixels in the observation used
    # From below, model the spectrum as a convolution of a line mask M 
    # and a line profile Z:  Y = M.Z

    specI = 1. - MI.dot(prof.specI)
    specV = MV.dot(prof.specV)
    specN1 = MV.dot(prof.specN1)
        
    if (outModelSpecName != ''):
        outFile = open(outModelSpecName, 'w')
        for i in range(specI.shape[0]):
            outFile.write('{:10f} {:11e} {:11e} {:11e}\n'.format(obsWl[i], specI[i], specV[i], specN1[i]))
    return (obsWl, specI, specV, specN1)


def lsdFit(obs, mask, prof, interpMode):
    # Model the spectrum as a convolution of a line pattern (mask) M,
    # and a mean line profile Z.  Y = conv(M, Z)
    # With matrix multiplication Y = M.Z
    # where Y has n wavelength elements, Z has m profile elements, and M is nxm strengths (with diagonals encoding positions)
    
    # Define a chi^2, for an observed spectrum Yo (n long)
    # chi^2 = (Yo - M.Z)^T.S^2.(Yo - M.Z)
    # where S is the square diagonal matrix of inverse errors: S_(i,i) = 1/sigma_i
    # and ^T is the transpose of a matrix (and . is a dot product)
    # For a linear least squares solution, where the derivative of free parameters=0
    # 0 = (-M)^T.S^2.(Yo - M.Z)*2
    # Z = (M^T.S^2.M)^(-1).M^T.S^2.Yo
    # Where ^(-1) is the matrix inverse
    
    # Here M^T.S^2.Yo is is effectively the cross-correlation between the mask and the observation.  M^T.S^2.M is effectively the auto-correlation matrix.
    # Uncertainties can be estimated from the diagonal of (M^T.S^2.M)^(-1)
    # (M^T.S^2.M)^(-1) is the covariance matrix C, and sigma^2(Z_(i)) = C_(i,i)
    
    # For liner interpolation from the model on to the observed wavelength pixels,
    # the matrix M needs to have approximately double diagonals.
    # (not exactly diagonal, there will at least be glitches!)
    # Line l in the mask contributes to M by:
    # M_(i,j) = w_l * (v_(j+1) - v_(i))/(v_(j+1) - v_(j))
    # M_(i,j+1) = w_l * (v_(i) - v_(j))/(v_(j+1) - v_(j))
    # or: M_(i,j) = w_l * (1 - (v_(i) - v_(j))/(v_(j+1) - v_(j)) )
    # where w_l is the weigh of line l, and v_i is calculated as
    # v_i = c*(lambda_i - lambda_l)/lambda_l  and v_(j) < v_(i) < v_(j+1)
    # and lambda_l, lambda_i are the wavelengths of line l and pixel i
    # i runs over observed wavelengths, j runs over velocity pixels in the LSD profile
    
    # In the classic linear least squares setup of a.x=b, then minimize (a.x-b)^2
    # effectively x = Z, a = M, and b = Yo, solving for x
    # This could be solved by numpy.linalg.lstsq or scipy.optimize.lsq_linear
    # but that would ignore uncertainties. (Maybe divide a and b by sigma?)
    #
    # so here beta = M^T.S^2.Yo, alpha = (M^T.S^2.M), and covar = alpha^(-1)
    # 

    sparseS2 = buildInvSig2(obs)
    
    M4I, M4V = buildM(obs, mask, prof, interpMode)
    MI = csr_matrix(M4I)
    MV = csr_matrix(M4V)

    #Use the sparse matrix 'dot with a vector' function for correct efficient calculation
    #(the sparse matrix dot product with a regular dense matrix seems to return a regular matrix)
    betaI = MI.T.dot( sparseS2.dot(obs.specI))
    betaV = MV.T.dot( sparseS2.dot(obs.specV))
    betaN1 = MV.T.dot( sparseS2.dot(obs.specN1))

    #Use the sparse matrix 'dot with a vector' function for correct efficient calculation
    #(the sparse matrix dot product with a sparse matrix seems to return a sparse matrix)
    alphaI = MI.T.dot(sparseS2.dot(MI))
    alphaV = MV.T.dot(sparseS2.dot(MV))
    #alpha is probably sparse, but with very few zeros here
    covarI = inv(alphaI.toarray())
    covarV = inv(alphaV.toarray())

    prof.specI = np.dot(covarI, betaI)
    prof.specV =  np.dot(covarV, betaV)
    prof.specN1 =  np.dot(covarV, betaN1)
    prof.specSigI =  np.sqrt(np.diag(covarI))
    prof.specSigV =  np.sqrt(np.diag(covarV))
    prof.specSigN1 =  np.sqrt(np.diag(covarV))
        
    return MI, MV, sparseS2


def lsdFitSigmaClip(obs, mask, prof, params):
    #Fit within sigma clipping loop.
    #Calls the main LSD fitting function and several suport functions

    #simple error checking
    if(params.sigmaClipIter < 0):
        print('WARNING: sigma clipping iterations < 1 found ({:}) assuming 1 fit desired'.format(params.sigmaClipIter))
        params.sigmaClipIter = 0
    
    #major sigma clipping loop
    i = 0
    while(i < params.sigmaClipIter + 1):
        
        MI, MV, sparseS2 = lsdFit(obs, mask, prof, params.interpMode)
        
        chi2I = getChi2(obs.specI, MI, sparseS2, prof.specI)
        chi2V = getChi2(obs.specV, MV, sparseS2, prof.specV)
        chi2N1 = getChi2(obs.specN1, MV, sparseS2, prof.specN1)

        if(i < params.sigmaClipIter):
            obs.sigmaClipI(prof, MI, params.sigmaClip)
        
        #print 'tlsdFit', tlsdFit-tStart, 'tChi2', tChi2-tlsdFit, 'tsigmaClipI', tsigmaClipI-tChi2
        i += 1

    #Optionally save the model after sigma clipping is done
    if(params.fSaveModelSpec != 0):
        print('saving model spectrum to {:} ...'.format(params.outModelSpecName))
        modelSpec = saveModelSpec(params.outModelSpecName, prof, MI, MV, obs.wl)
        return chi2I, chi2V, chi2N1, modelSpec
    
    return chi2I, chi2V, chi2N1, None


def scaleErr(profErr, chi2, obsUsed, profNpix):
    # Re-scales the error bars of an LSD profile by the root of the reduced chi^2
    # this ensures the reduced chi^2 is always ~1 
    # useful for cases in which nose from the LSD reconstruction dominates photon noise
    # Note: this takes a calculated chi^2 value, assumed to be determined in the fitting routine
    
    scale = np.sqrt(chi2/(obsUsed-profNpix))
    if(scale > 1.):
        print(' Rescaling error bars by: {:.6f}'.format(scale))
        profErr *= scale
    else:
        print(' Not rescaling error bars (scale {:.6f})'.format(scale))

    return scale


def zeroProf(prof, profErr, iContCorr):
    #Simple subroutine to make sure the average supplied profile is zero
    #this hopes to normalize out continuum polarization
    #Note: this may not always be desirable!

    avgP = np.average(prof)
    avgErr = np.average(profErr)
    rmsErr = np.sum(profErr**2)
    avgPerr = rmsErr/prof.shape[0]
    
    if(iContCorr != 0):
        print(' removing profile continuum pol: {:.4e} +/- {:.4e} (avg err {:.4e})'.format(avgP, avgPerr, avgErr))
        prof -= avgP
    else:        
        print(' note, profile continuum pol: {:.4e} +/- {:.4e} (avg err {:.4e})'.format(avgP, avgPerr, avgErr))

    return


def estimateLineRange(profI, profSigI):
    #estimate the continuum from a 20 point average, using either end of the profile
    pad = 2
    contPix = 20
    approxCont = np.average((profI[pad:contPix+pad], profI[-contPix-pad:-pad]))
    approxErr = np.std((profI[pad:contPix+pad], profI[-contPix-pad:-pad]))
    meanErr = np.average((profSigI[pad:contPix+pad], profSigI[-contPix-pad:-pad]))
    scaleErr = 1.0
    if(approxErr > 1.1*meanErr):
        print('(possible Stokes I uncertainty underestimate {:.4e} vs {:.4e})'.format(approxErr, meanErr))
        scaleErr = approxErr/meanErr
    
    #Get 4 sigma below (above) continuum points
    iTheorIn = np.where(profI[pad:-pad] > approxCont + 4.*scaleErr*profSigI[pad:-pad])[0] + pad
    iTheorOut = np.where(profI[pad:-pad] <= approxCont + 4.*scaleErr*profSigI[pad:-pad])[0] + pad
    
    return iTheorIn, iTheorOut

def nullTest(prof):
    #Check for a magnetic detection in V or N
    iTheorIn, iTheorOut = estimateLineRange(prof.specI, prof.specSigI)

    if(iTheorIn.shape[0] > 0):
        print('line range estimate {:} {:} km/s'.format(prof.vel[iTheorIn[0]], prof.vel[iTheorIn[-1]]))
    else:
        print('ERROR: could not find line range!  (using full profile)')
        iTheorIn = iTheorOut

    #import matplotlib.pyplot as plt
    #plt.plot(prof.vel, prof.specI)
    #plt.plot(prof.vel[iTheorIn], prof.specI[iTheorIn], '.')
    #plt.show()
    
    #'fitting' the flat line (essentially an average weighted by 1/sigma^2)
    contV = np.sum(prof.specV[iTheorIn]/prof.specSigV[iTheorIn]**2) / np.sum(1./prof.specSigV[iTheorIn]**2)
    contN1 = np.sum(prof.specN1[iTheorIn]/prof.specSigN1[iTheorIn]**2) / np.sum(1./prof.specSigN1[iTheorIn]**2)
    
    
    chi2Vin = np.sum(((prof.specV[iTheorIn] - contV)/prof.specSigV[iTheorIn])**2)
    chi2Vout = np.sum(((prof.specV[iTheorOut] - contV)/prof.specSigV[iTheorOut])**2)
    chi2N1in = np.sum(((prof.specN1[iTheorIn] - contN1)/prof.specSigN1[iTheorIn])**2)
    chi2N1out = np.sum(((prof.specN1[iTheorOut] - contN1)/prof.specSigN1[iTheorOut])**2)
    
    approxDOFin = (iTheorIn.shape[0]-1.)
    approxDOFout = (iTheorOut.shape[0]-1.)
    
    probVIn = specialf.gammainc(approxDOFin/2., chi2Vin/2.)
    probVOut = specialf.gammainc(approxDOFout/2., chi2Vout/2.)
    probN1In = specialf.gammainc(approxDOFin/2., chi2N1in/2.)
    probN1Out = specialf.gammainc(approxDOFout/2., chi2N1out/2.)

    print('V in line reduced chi^2 {:8f} (chi2 {:10f}) \n detect prob {:6f} (fap {:12.6e})'.format(
        chi2Vin/approxDOFin, chi2Vin, probVIn, 1.-probVIn))
    if(probVIn > 0.9999):
        print(' Detection! V (fap {:12.6e})'.format(1.-probVIn))
    elif(probVIn > 0.99):
        print(' Marginal detection V (fap {:12.6e})'.format(1.-probVIn))
    else:
        print(' Non-detection V (fap {:12.6e})'.format(1.-probVIn))
    print(' V outside line reduced chi^2 {:8f} (chi2 {:10f}) \n detect prob {:6f} (fap {:12.6e})'.format(
        chi2Vout/approxDOFout, chi2Vout, probVOut, 1.-probVOut))
    
    print('N1 in line reduced chi^2 {:8f} (chi2 {:10f}) \n detect prob {:6f} (fap {:12.6e})'.format(
        chi2N1in/approxDOFin, chi2N1in, probN1In, 1.-probN1In))
    if(probN1In > 0.9999):
        print(' Detection! N1 (fap {:12.6e})'.format(1.-probN1In))
    elif(probN1In > 0.99):
        print(' Marginal detection N1 (fap {:12.6e})'.format(1.-probN1In))
    else:
        print(' Non-detection N1 (fap {:12.6e})'.format(1.-probN1In))
    print(' N1 outside line reduced chi^2 {:8f} (chi2 {:10f}) \n detect prob {:6f} (fap {:12.6e})'.format(
        chi2N1out/approxDOFout, chi2N1out, probN1Out, 1.-probN1Out))
    
    return
