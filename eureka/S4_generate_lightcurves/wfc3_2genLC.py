#! /usr/bin/env python

import numpy as np
import scipy.interpolate as spi
import matplotlib.pyplot as plt
from ..lib import manageevent as me
import sys, time
from ..lib import smooth
#import hst_scan as hst
import scipy.ndimage.interpolation as spni
#import imp
#reload(smooth)
#reload(hst)

def lcWFC3(eventname, eventdir, nchan, madVariable, madVarSet, wmin=1.125, wmax=1.65, expand=1, smooth_len=None, correctDrift=True, isplots=True):
    '''
    Compute photometric flux over specified range of wavelengths

    Parameters
    ----------
    eventname   : Unique identifier for these data
    eventdir    : Location of save file
    nchan       : Number of spectrophotometric channels
    wmin        : minimum wavelength
    wmax        : maximum wavelength
    expand      : expansion factor
    isplots     : Set True to produce plots

    Returns
    -------
    None

    History
    -------
    Written by Kevin Stevenson      June 2012

    '''

    # Load saved data
    print("Loading saved data...")
    try:
        ev              = me.loadevent(eventdir + '/d-' + eventname + '-w2')
        print('W2 data loaded\n')
    except:
        ev              = me.loadevent(eventdir + '/d-' + eventname + '-w0')
        print('W0 data loaded\n')
    aux             = me.loadevent(eventdir + '/d-' + eventname + '-data')
    ev.spectra      = aux.spectra
    specerr         = aux.specerr
    data_mhdr       = aux.data_mhdr

    #Replace NaNs with zero
    ev.spectra[np.where(np.isnan(ev.spectra))] = 0

    # Determine wavelength bins
    binsize     = (wmax - wmin)/nchan
    wave_low    = np.round([i for i in np.linspace(wmin, wmax-binsize, nchan)],3)
    wave_hi     = np.round([i for i in np.linspace(wmin+binsize, wmax, nchan)],3)
    #binwave     = (wave_low + wave_hi)/2.

    # Increase resolution of spectra
    nx  = ev.spectra.shape[-1]
    if expand > 1:
        print("Increasing spectra resolution...")
        #hdspectra = np.zeros((ev.n_files,ev.n_reads-1,expand*nx))
        #hdspecerr = np.zeros((ev.n_files,ev.n_reads-1,expand*nx))
        hdspectra = spni.zoom(ev.spectra,zoom=[1,1,expand])
        hdspecerr = spni.zoom(specerr,zoom=[1,1,expand])*np.sqrt(expand)
        hdwave    = np.zeros((ev.n_img,ev.n_spec,expand*nx))
        for j in range(ev.n_img):
            hdwave[j] = spni.zoom(ev.wave[j][0],zoom=expand)
        ev.spectra  = hdspectra
        specerr     = hdspecerr
        ev.wave     = hdwave
        nx         *= expand

    # Smooth spectra
    if smooth_len != None:
        for m in range(ev.n_files):
            for n in range(ev.n_reads-1):
                ev.spectra[m,n] = smooth.smooth(ev.spectra[m,n],smooth_len,'flat')
    """
    # First read is bad for IMA files
    if ev.n_reads > 2:
        print('WARNING: Marking all first reads as bad.')
        istart = 1
    else:
        print('Using first reads.')
        istart = 0
    """
    print('Using first reads.')
    istart = 0

    if correctDrift == True:
        #Shift 1D spectra
        #Calculate drift over all frames and non-destructive reads
        print('Applying drift correction...')
        ev.drift, ev.goodmask = hst.drift_fit2(ev, preclip=0, postclip=None, width=5*expand, deg=2, validRange=11*expand, istart=istart, iref=ev.iref[0])
        # Correct for drift
        for m in range(ev.n_files):
            for n in range(istart,ev.n_reads-1):
                spline            = spi.UnivariateSpline(np.arange(nx), ev.spectra[m,n], k=3, s=0)
                #ev.spectra[m,n,p] = spline(np.arange(nx)+ev.drift_model[n,m,p])
                #if m==13:
                #    ev.drift[n,m,p] -= 0.476
                #Using measured drift, not model fit
                ev.spectra[m,n] = spline(np.arange(nx)+ev.drift[m,n])
    '''
    # Look for bad columns
    igoodcol    = np.ones(nx)
    normspec    = ev.spectra/np.mean(ev.spectra,axis=2)[:,:,np.newaxis]
    sumspec     = np.sum(normspec,axis=1)/(ev.n_reads-istart-1)
    stdsumspec  = np.std(sumspec, axis=0)
    igoodcol[np.where(stdsumspec > 0.007)] = 0  #FINDME: hard coded
    '''

    print("Generating light curves...")
    ev.eventname2 = ev.eventname
    for i in range(nchan):
        ev.wave_low  = wave_low[i]
        ev.wave_hi   = wave_hi[i]
        print("Bandpass = %.3f - %.3f" % (ev.wave_low,ev.wave_hi))
        # Calculate photometric flux for each spectrum
        ev.photflux     = np.zeros((ev.n_spec, ev.n_files, np.max((1,ev.n_reads-1-istart))))
        ev.photfluxerr  = np.zeros((ev.n_spec, ev.n_files, np.max((1,ev.n_reads-1-istart))))
        #ev.wave         = []
        if ev.detector == 'IR':
            #Compute common wavelength and indeces to apply over all observations
            wave = np.zeros(nx)
            for j in range(ev.n_img):
                wave += ev.wave[j][0]
            wave /= ev.n_img
            #index = np.where(np.bitwise_and(wave >= wave_low[i], wave <= wave_hi[i]))[0]
            index = np.where((wave >= wave_low[i])*(wave <= wave_hi[i]))[0]
            #define numgoodcol, totcol
        else:
            # UVIS: Use all pixels for aperture photometry
            index = range(len(ev.spectra[0,0,0]))
        for m in range(ev.n_files):
            '''
            #Select appropriate orbit-dependent wavelength
            if ev.n_img == (np.max(ev.orbitnum)+1):
                j = int(ev.orbitnum[m])
            else:
                j = 0
            #Method 1
            ev.wave.append(np.mean(ev.wavegrid[j][n],axis=0))
            index = np.where(np.bitwise_and(ev.wave[n] >= wave_low, ev.wave[n] <= wave_hi))[0]
            #Method 2
            index = np.where(np.bitwise_and(ev.wave[j][n] >= wave_low, ev.wave[j][n] <= wave_hi))[0]
            '''
            ev.photflux[0,m]    = np.sum(ev.spectra[m,istart:,index],axis=0)
            ev.photfluxerr[0,m] = np.sqrt(np.sum(specerr[m,istart:,index]**2,axis=0))

        # Save results
        ev.eventname  = ev.eventname2 + '_' + str(int(ev.wave_low*1e3)) + '_' + str(int(ev.wave_hi*1e3))
        #me.saveevent(ev, eventdir + '/d-' + ev.eventname + '-w3', delete=['data_mhdr', 'spectra', 'specerr'])
        me.saveevent(ev, eventdir + '/d-' + ev.eventname + '-w3')

        # Produce plot
        if isplots == True:
            plt.figure(3000+i, figsize=(10,8))
            plt.clf()
            plt.suptitle('Wavelength range: ' + str(wave_low[i]) + '-' + str(wave_hi[i]))
            ax = plt.subplot(111)
            #for n in range(ev.n_spec):
            #plt.subplot(ev.n_spec,1,1)
            #plt.title('Star ' + str(n))
            #igood   = np.where(ev.goodmask[0])[0]
            iscan0  = np.where(ev.scandir == 0)[0]
            iscan1  = np.where(ev.scandir == 1)[0]
            mjd     = np.floor(ev.bjdtdb[0])
            flux0   = np.sum(ev.photflux[0][iscan0],axis=1)/np.sum(ev.photflux[0,[iscan0[-1]]]) # forward scan
            #err  = np.sqrt(1 / np.sum(1/ev.photfluxerr[0]**2,axis=1))/np.sum(ev.photflux[0,-1])
            try:
                err0    = np.sqrt(np.sum(ev.photfluxerr[0][iscan0]**2,axis=1))/np.sum(ev.photflux[0,[iscan0[-1]]])
            except:
                err0    = 0
                #err1    = 0
            plt.errorbar(ev.bjdtdb[iscan0]-mjd, flux0, err0, fmt='bo')
            plt.text(0.05, 0.1, "MAD = "+str(np.round(1e6*np.median(np.abs(np.ediff1d(flux0)))))+" ppm", transform=ax.transAxes, color='b')
            #print(len(iscan1))
            flux1 = 0

            if len(iscan1) > 0:
                flux1   = np.sum(ev.photflux[0][iscan1],axis=1)/np.sum(ev.photflux[0,[iscan0[-1]]]) # reverse scan
                err1    = np.sqrt(np.sum(ev.photfluxerr[0][iscan1]**2,axis=1))/np.sum(ev.photflux[0,[iscan0[-1]]])
                plt.errorbar(ev.bjdtdb[iscan1]-mjd, flux1, err1, fmt='ro')
                plt.text(0.05, 0.05, "MAD = "+str(np.round(1e6*np.median(np.abs(np.ediff1d(flux1)))))+" ppm", transform=ax.transAxes, color='r')
            plt.ylabel('Normalized Flux')
            plt.xlabel('Time [MJD + ' + str(mjd) + ']')

            plt.subplots_adjust(left=0.10,right=0.95,bottom=0.10,top=0.90,hspace=0.20,wspace=0.3)
            plt.savefig(eventdir + '/figs/Fig' + str(3000+i) + '-' + ev.eventname + '.png')
            #plt.pause(0.1)

            # f = open('2017-07-15-w1_spec_width_20/W5_MAD_'+ev.madVarStr+'_1D.txt','a+')
            # fooTemp = getattr(ev,madVariable)
            # print('W5: ' + ev.madVarStr + ' = ' + str(fooTemp))
            # f.write(str(fooTemp) + ',' + str(np.round(1e6*np.median(np.abs(np.ediff1d(flux0))))) + ',' + str(np.round(1e6*np.median(np.abs(np.ediff1d(flux1))))) +'\n')
            # f.close()
            # print('W5_MAD_'+ ev.madVarStr +'_1D.txt saved\n')

    if (isplots >= 1) and (ev.detector == 'IR'):
        # Drift
        plt.figure(3100, figsize=(10,8))
        plt.clf()
        plt.subplot(211)
        for j in range(istart,ev.n_reads-1):
            plt.plot(ev.drift2D[:,j,1],'.')
        plt.ylabel('Spectrum Drift Along y')
        plt.subplot(212)
        for j in range(istart,ev.n_reads-1):
            plt.plot(ev.drift2D[:,j,0]+ev.drift[:,j],'.')
        plt.ylabel('Spectrum Drift Along x')
        plt.xlabel('Frame Number')
        plt.savefig(eventdir + '/figs/fig3100-Drift.png')

        # 2D light curve with drift correction
        plt.figure(3200, figsize=(7.85,ev.n_files/20.+0.8))
        plt.clf()
        vmin        = 0.98
        vmax        = 1.01
        #FINDME
        normspec    = np.zeros((ev.n_files,ev.spectra.shape[2]))
        for p in range(2):
            iscan   = np.where(ev.scandir == p)[0]
            if len(iscan) > 0:
                normspec[iscan] = np.mean(ev.spectra[iscan],axis=1)/ \
                                  np.mean(ev.spectra[iscan[ev.inormspec[0]:ev.inormspec[1]]],axis=(0,1))
                #normspec[iscan] = np.mean(ev.spectra[iscan],axis=1)/np.mean(ev.spectra[ev.iref[p]],axis=0)
        #normspec    = np.mean(ev.spectra[:,istart:],axis=1)/np.mean(ev.spectra[ev.inormspec[0]:ev.inormspec[1],istart:],axis=(0,1))
        ediff       = np.zeros(ev.n_files)
        iwmin       = np.where(ev.wave[0][0]>wmin)[0][0]
        iwmax       = np.where(ev.wave[0][0]>wmax)[0][0]
        for m in range(ev.n_files):
            ediff[m]    = 1e6*np.median(np.abs(np.ediff1d(normspec[m,iwmin:iwmax])))
            plt.scatter(ev.wave[0][0], np.zeros(normspec.shape[-1])+m, c=normspec[m],
                        s=14,linewidths=0,vmin=vmin,vmax=vmax,marker='s',cmap=plt.cm.RdYlBu_r)
        plt.title("MAD = "+str(np.round(np.mean(ediff),0)) + " ppm")
        plt.xlim(wmin,wmax)
        if nchan > 1:
            xticks  = np.round([i for i in np.linspace(wmin, wmax, nchan+1)],3)
            plt.xticks(xticks,xticks)
            plt.vlines(xticks,0,ev.n_files,'k','dashed')
        plt.ylim(0,ev.n_files)
        plt.ylabel('Frame Number')
        plt.xlabel(r'Wavelength ($\mu m$)')
        plt.xticks(rotation=30)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(eventdir+'/figs/fig3200-'+str(nchan)+'-2D_LC.png')
        #plt.savefig(eventdir+'/figs/fig3200-'+str(nchan)+'-2D_LC_'+madVariable+'_'+str(madVarSet)+'.png')

        #ev.mad5 = np.round(np.mean(ediff),0)
        # f = open('2017-07-15-w1_spec_width_20/W5_MAD_'+ev.madVarStr+'.txt','a+')
        # fooTemp = getattr(ev,madVariable)
        # print('W5: ' + ev.madVarStr + ' = ' + str(fooTemp))
        # f.write(str(fooTemp) + ',' + str(np.round(np.mean(ediff),0)) + '\n')
        # f.close()
        # print('W5_MAD_'+ ev.madVarStr +'.txt saved\n')

    if (isplots >= 3) and (ev.detector == 'IR'):
        # Plot individual non-destructive reads
        vmin        = 0.97
        vmax        = 1.03
        iwmin       = np.where(ev.wave[0][0]>wmin)[0][0]
        iwmax       = np.where(ev.wave[0][0]>wmax)[0][0]
        #FINDME
        normspec    = ev.spectra[:,istart:]/np.mean(ev.spectra[ev.inormspec[0]:ev.inormspec[1],istart:],axis=0)
        for n in range(ev.n_reads-1):
            plt.figure(3300+n, figsize=(8,ev.n_files/20.+0.8))
            plt.clf()
            ediff       = np.zeros(ev.n_files)
            for m in range(ev.n_files):
                ediff[m]    = 1e6*np.median(np.abs(np.ediff1d(normspec[m,n,iwmin:iwmax])))
                plt.scatter(ev.wave[0][0], np.zeros(normspec.shape[-1])+m, c=normspec[m,n],
                            s=14,linewidths=0,vmin=vmin,vmax=vmax,marker='s',cmap=plt.cm.RdYlBu_r)
            plt.title("MAD = "+str(np.round(np.mean(ediff),0)) + " ppm")
            plt.xlim(wmin,wmax)
            plt.ylim(0,ev.n_files)
            plt.ylabel('Frame Number')
            plt.xlabel(r'Wavelength ($\mu m$)')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(ev.eventdir+'/figs/fig'+str(3300+n)+'-2D_LC.png')
        """
        # Aligned 1D spectra
        plt.figure(3300, figsize=(8,6.5))
        plt.clf()
        #istart=0
        #normspec    = ev.spectra[:,istart:]/np.mean(ev.spectra[:,istart:],axis=2)[:,:,np.newaxis]
        normspec    = ev.spectra[:,:,1:]/np.mean(ev.spectra[:,:,1:],axis=2)[:,:,np.newaxis]
        wave        = ev.wave[0][0][1:]
        sumspec     = np.sum(normspec,axis=1)/(ev.n_reads-istart-1)
        for m in range(10,16):
            plt.plot(wave,sumspec[m],'r-')
        for m in range(7,10):
            plt.plot(wave,sumspec[m],'.k-')
        """



    return ev

#lcGMOS(isplots=True)

'''
flux1 = np.sum(ev.photflux[0:6],axis=0)
flux2 = np.sum(ev.photflux[6:12],axis=0)
flux3 = np.sum(ev.photflux[12:18],axis=0)
diffflux = flux2/(flux1+flux3)
normflux = diffflux/np.median(diffflux)
plt.figure(1)
plt.clf()
plt.suptitle('WASP-12 Light Curve')
plt.plot(ev.bjd_tdb-ev.bjd_tdb[0], normflux, 'o')
plt.xlabel('Time (Days)')
plt.ylabel('Normalized Flux')
plt.savefig('figs/' + ev.eventname + '-LC.png')
'''
