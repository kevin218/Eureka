import numpy as np
import matplotlib.pyplot as plt


#SET PLOTTING FORMAT
ebfmt   = ['bo',  'go',  'ro',  'co',  'mo',  'yo',  
           'bo',  'go',  'ro',  'co',  'mo',  'yo',  
           'bo',  'go',  'ro',  'co',  'mo',  'yo',  
           'bo',  'go',  'ro',  'co',  'mo',  'yo',   
           'bo',  'go',  'ro',  'co',  'mo',  'yo',  
           'bo',  'go',  'ro',  'co',  'mo',  'yo',  
           'bo',  'go',  'ro',  'co',  'mo',  'yo',  
           'bo',  'go',  'ro',  'co',  'mo',  'yo',  
           'bo',  'go',  'ro',  'co',  'mo',  'yo',  
           'bo',  'go',  'ro',  'co',  'mo',  'yo', 
           'bo',  'go',  'ro',  'co',  'mo',  'yo',   
           'bo',  'go',  'ro',  'co',  'mo',  'yo',  
           'bo',  'go',  'ro',  'co',  'mo',  'yo',  
           'bo',  'go',  'ro',  'co',  'mo',  'yo',   
           'bo',  'go',  'ro',  'co',  'mo',  'yo',  
           'bo',  'go',  'ro',  'co',  'mo',  'yo',  
           'bo',  'go',  'ro',  'co',  'mo',  'yo',  
           'bo',  'go',  'ro',  'co',  'mo',  'yo',  
           'bo',  'go',  'ro',  'co',  'mo',  'yo',  
           'bo',  'go',  'ro',  'co',  'mo',  'yo', 
           'bo',  'go',  'ro',  'co',  'mo',  'yo', 
           'bo',  'go',  'ro',  'co',  'mo',  'yo']
pltfmt  = ['b-',  'g-',  'r-',  'c-',  'm-',  'y-',  
           'b-',  'g-',  'r-',  'c-',  'm-',  'y-',  
           'b-',  'g-',  'r-',  'c-',  'm-',  'y-',  
           'b-',  'g-',  'r-',  'c-',  'm-',  'y-',  
           'b-',  'g-',  'r-',  'c-',  'm-',  'y-',  
           'b-',  'g-',  'r-',  'c-',  'm-',  'y-',  
           'b-',  'g-',  'r-',  'c-',  'm-',  'y-',  
           'b-',  'g-',  'r-',  'c-',  'm-',  'y-',  
           'b-',  'g-',  'r-',  'c-',  'm-',  'y-', 
           'b-',  'g-',  'r-',  'c-',  'm-',  'y-',  
           'b-',  'g-',  'r-',  'c-',  'm-',  'y-',   
           'b-',  'g-',  'r-',  'c-',  'm-',  'y-',  
           'b-',  'g-',  'r-',  'c-',  'm-',  'y-',  
           'b-',  'g-',  'r-',  'c-',  'm-',  'y-',  
           'b-',  'g-',  'r-',  'c-',  'm-',  'y-',  
           'b-',  'g-',  'r-',  'c-',  'm-',  'y-',  
           'b-',  'g-',  'r-',  'c-',  'm-',  'y-',  
           'b-',  'g-',  'r-',  'c-',  'm-',  'y-',  
           'b-',  'g-',  'r-',  'c-',  'm-',  'y-', 
           'b-',  'g-',  'r-',  'c-',  'm-',  'y-',  
           'b-',  'g-',  'r-',  'c-',  'm-',  'y-', 
           'b-',  'g-',  'r-',  'c-',  'm-',  'y-']
pltfmt2 = ['b--', 'g--', 'r--', 'c--', 'm--', 'y--', 
           'b--', 'g--', 'r--', 'c--', 'm--', 'y--', 
           'b--', 'g--', 'r--', 'c--', 'm--', 'y--', 
           'b--', 'g--', 'r--', 'c--', 'm--', 'y--', 
           'b--', 'g--', 'r--', 'c--', 'm--', 'y--', 
           'b--', 'g--', 'r--', 'c--', 'm--', 'y--', 
           'b--', 'g--', 'r--', 'c--', 'm--', 'y--', 
           'b--', 'g--', 'r--', 'c--', 'm--', 'y--', 
           'b--', 'g--', 'r--', 'c--', 'm--', 'y--', 
           'b--', 'g--', 'r--', 'c--', 'm--', 'y--', 
           'b--', 'g--', 'r--', 'c--', 'm--', 'y--', 
           'b--', 'g--', 'r--', 'c--', 'm--', 'y--', 
           'b--', 'g--', 'r--', 'c--', 'm--', 'y--', 
           'b--', 'g--', 'r--', 'c--', 'm--', 'y--', 
           'b--', 'g--', 'r--', 'c--', 'm--', 'y--', 
           'b--', 'g--', 'r--', 'c--', 'm--', 'y--', 
           'b--', 'g--', 'r--', 'c--', 'm--', 'y--', 
           'b--', 'g--', 'r--', 'c--', 'm--', 'y--', 
           'b--', 'g--', 'r--', 'c--', 'm--', 'y--', 
           'b--', 'g--', 'r--', 'c--', 'm--', 'y--', 
           'b--', 'g--', 'r--', 'c--', 'm--', 'y--', 
           'b--', 'g--', 'r--', 'c--', 'm--', 'y--']


# Plot binned data and best fit
def binlc(event, fit, fignum, savefile=None, istitle=True, j=0):
    plt.rcParams.update({'legend.fontsize':13})
    plt.figure(fignum)
    plt.clf()
    if istitle == True:
        plt.suptitle(event.eventname + ' Binned Data With Best Fit',size=16)
        plt.title(fit.model, size=10)
    elif istitle == False:
        pass
    else:
        plt.suptitle(istitle, size=16)
    plt.errorbar(fit.abscissauc, fit.binfluxuc, fit.binstduc, fmt='ko', 
                     ms=4, linewidth=1, label='Binned Data', zorder=3)
    plt.plot(fit.abscissa, fit.binnoecl, 'k-', label='No Eclipse', zorder=1)
    #plt.plot(fit.abscissa, fit.binmedianfit, pltfmt2[j], label='Median Fit')
    plt.plot(fit.abscissa, fit.binbestfit,    pltfmt[j], label='Best Fit', zorder=5)
    plt.xticks(size=13)
    plt.yticks(size=13)
    plt.xlabel(fit.xlabel,size=14)
    plt.ylabel('Flux',size=14)
    plt.legend(loc='best')
    if savefile != None:
        plt.savefig(savefile)
    return

# Plot normalized data, best fit, and residuals
def normlc(event, fit, fignum, savefile=None, istitle=True, j=0, interclip=None):
    plt.rcParams.update({'legend.fontsize':13})
    plt.figure(fignum, figsize=(8,6))
    plt.clf()
    # Normalized subplot
    a = plt.axes([0.15,0.35,0.8,0.55])
    a.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%0.4f'))
    if istitle == True:
        plt.suptitle(event.eventname + ' Normalized Binned Data With Best Fit',size=16)
        plt.title(fit.model, size=10)
    elif istitle == False:
        pass
    else:
        plt.suptitle(istitle, size=16)
    plt.errorbar(fit.abscissauc,fit.normbinfluxuc,fit.normbinsduc,fmt='ko',ms=4,lw=1, label='Binned Data', zorder=1)
    plt.plot(fit.timeunit, fit.normbestfit,pltfmt[j], label='Best Fit', lw=2, zorder=3)
    if interclip != None:
        for i in range(len(interclip)):
            ind0 = np.argmin(np.abs(fit.timeunit-fit.timeunituc[interclip[i][0]]))
            ind1 = np.argmin(np.abs(fit.timeunit-fit.timeunituc[interclip[i][1]]))
            #ind0 = ind1-1
            plt.plot([fit.timeunit[ind0],fit.timeunit[ind1]], 
                     [fit.normbestfit[ind0],fit.normbestfit[ind1]], '-w', lw=3, zorder=5)
            #plt.plot(fit.tuall[interclip[i][0]:interclip[i][1]], np.ones(interclip[i][1]-interclip[i][0]), '-w', lw=3)
    plt.setp(a.get_xticklabels(), visible = False)
    plt.yticks(size=13)
    plt.ylabel('Normalized Flux',size=14)
    plt.legend(loc='best')
    xmin, xmax = plt.xlim()
    plt.axes([0.15,0.1,0.8,0.2])
    # Residuals subplot
    #fit.bestlinear  = np.polyfit(fit.timeunit, fit.residuals, 1)
    #fit.binresfit   = fit.bestlinear[0]*fit.abscissa + fit.bestlinear[1]
    #plt.errorbar(fit.binphase,fit.binres,fit.binresstd,fmt='ko',ms=4,linewidth=1)
    flatline = np.zeros(len(fit.abscissa))
    #plt.errorbar(fit.abscissa,fit.binres/fit.mflux,fit.binresstd/fit.mflux,fmt='ko',ms=4,lw=1)
    if hasattr(fit, 'normbinresuc'):
        #plt.plot(fit.abscissa,fit.normbinres,'ko',ms=4)
        plt.errorbar(fit.abscissauc,fit.normbinresuc-1.,fit.normbinsduc,fmt='ko',ms=4)
    else:
        plt.plot(fit.abscissa,fit.binres/fit.mflux,'ko',ms=4)
    plt.plot(fit.abscissa, flatline,'k-',lw=2)
    plt.xlim(xmin,xmax)
    plt.xticks(size=13)
    plt.yticks(size=13)
    plt.xlabel(fit.xlabel,size=14)
    plt.ylabel('Residuals',size=14)
    if savefile != None:
        plt.savefig(savefile)
    return

def Znormlc(event, fit, fignum, savefile=None, istitle=True, j=0, interclip=None):
    plt.rcParams.update({'legend.fontsize':13})
    plt.figure(fignum, figsize=(8,6))
    plt.clf()
    # Normalized subplot
    a = plt.axes([0.15,0.35,0.8,0.55])
    a.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%0.4f'))
    if istitle == True:
        plt.suptitle(event.eventname + ' Zoomed Normalized Binned Data With Best Fit',size=16)
        plt.title(fit.model, size=10)
    elif istitle == False:
        pass
    else:
        plt.suptitle(istitle, size=16)
    #### calc amp and offset
    con=np.nanmean((fit.bestecl*fit.bestsin)/(fit.normbestfit))
    zrenorm=1.0
    if 'mandelecl' in fit.model:
        zper=fit.bestp[(np.where(np.array(fit.parname)=='Period'))[0][0]]
        zt_ecl=fit.bestp[(np.where(np.array(fit.parname)=='Eclipse Phase'))[0][0]]
        zrenorm=(fit.normbestfit)[np.argmin(np.abs(fit.timeunit-zt_ecl))]
        zmax=np.argmax(fit.bestsin)
        zmin=np.argmin(fit.bestsin)
        
        zoff=360.*(fit.timeunit[zmax]-zt_ecl)/zper
        zoff=np.nanmin([np.abs(zoff-360.),np.abs(zoff)])
        zamp=0.5*(fit.bestsin[zmax]-fit.bestsin[zmin])/zrenorm
    else:
        zrenorm=1.0
        zoff=0.0
        zoff=0.0
        zamp=0.0
    #####
    plt.errorbar(fit.abscissauc,fit.normbinfluxuc/zrenorm,fit.normbinsduc,fmt='ko',mfc='none',ms=8,lw=1, label='Binned Data', zorder=1)
    plt.plot(fit.timeunit, fit.normbestfit/zrenorm,ls='-', label='Best Fit', lw=4, c='hotpink',zorder=3)
    plt.plot(fit.timeunit, fit.bestsin/zrenorm,ls='-', lw=2, c='crimson',zorder=4)
    #plt.plot(fit.timeunit, fit.bestsin,ls='-', label='Phase Only', lw=2, c='darkred',zorder=4)
    plt.axhline(y=1.0,ls='--',c='darkslategrey',lw=2)
    plt.axvline(x=fit.timeunit[np.argmax(fit.bestsin)],ls='--',c='darkslategrey',lw=1)
    if interclip != None:
        for i in range(len(interclip)):
            ind0 = np.argmin(np.abs(fit.timeunit-fit.timeunituc[interclip[i][0]]))
            ind1 = np.argmin(np.abs(fit.timeunit-fit.timeunituc[interclip[i][1]]))
            #ind0 = ind1-1
            plt.plot([fit.timeunit[ind0],fit.timeunit[ind1]],
                     [fit.normbestfit[ind0]/zrenorm,fit.normbestfit[ind1]/zrenorm], '-w', lw=3, zorder=5)
    if 'mmbilinint' in fit.model:
        map_inds = np.where(fit.mastermapovrF == 0)[0]
        plttime = np.copy(fit.timeunit)
        plttime[map_inds] = np.nan
        plt.plot(plttime, np.ones_like(fit.timeunit), '-', color = 'crimson', alpha = 0.3, lw = 20, zorder  = 0)
    #plt.plot(fit.tuall[interclip[i][0]:interclip[i][1]], np.ones(interclip[i][1]-interclip[i][0]), '-w', lw=3)
    plt.setp(a.get_xticklabels(), visible = False)
    plt.yticks(size=13)
    plt.ylabel('Normalized Flux',size=14)
    plt.legend(loc='upper left')
    xmin, xmax = plt.xlim()
    plt.ylim(ymin=0.99700)
    plt.figtext(0.93,0.43,'BIC: '+str(np.round(fit.bic,2)),ha='right',va='center',fontproperties='bold',fontsize=15)
    plt.figtext(0.93,0.38,'SDNR [ppm]: '+str(np.round(fit.sdnr*10**6.,2)),ha='right',va='center',fontproperties='bold',fontsize=15)
    plt.figtext(0.17,0.43,'Hotspot: '+str(np.round(zoff,2))+' $^{\circ}$',ha='left',va='center',fontproperties='bold',fontsize=15)
    plt.figtext(0.17,0.38,'Amplitude [ppm]: '+str(np.round(zamp*10**6.,2)),ha='left',va='center',fontproperties='bold',fontsize=15)
    plt.axes([0.15,0.1,0.8,0.2])
    # Residuals subplot
    #fit.bestlinear  = np.polyfit(fit.timeunit, fit.residuals, 1)
    #fit.binresfit   = fit.bestlinear[0]*fit.abscissa + fit.bestlinear[1]
    #plt.errorbar(fit.binphase,fit.binres,fit.binresstd,fmt='ko',ms=4,linewidth=1)
    flatline = np.zeros(len(fit.abscissa))
        #plt.errorbar(fit.abscissa,fit.binres/fit.mflux,fit.binresstd/fit.mflux,fmt='ko',ms=4,lw=1)
    if hasattr(fit, 'normbinresuc'):
        #plt.plot(fit.abscissa,fit.normbinres,'ko',ms=4)
        plt.errorbar(fit.abscissauc,fit.normbinresuc-1.,fit.normbinsduc,fmt='ko',ms=4)
    else:
        plt.plot(fit.abscissa,fit.binres/fit.mflux,'ko',ms=4)
    plt.plot(fit.abscissa, flatline,'k-',lw=2)
    plt.xlim(xmin,xmax)
    plt.xticks(size=13)
    plt.yticks(size=13)
    plt.xlabel(fit.xlabel,size=14)
    plt.ylabel('Residuals',size=14)
    if savefile != None:
        plt.savefig(savefile)
    return
    
# Trace plots
def trace(event, fit, fignum, savefile=None, allparams=None, parname=None, iparams=None, stepsize=None, istitle=True):
    if stepsize == None:
        try:
            stepsize = event.params.stepsize
        except:
            stepsize = 10
    if allparams == None:
        allparams = fit.allparams
    if parname == None:
        parname   = fit.parname
    if iparams == None:
        nonfixedpars = fit.nonfixedpars
    else:
        nonfixedpars = range(allparams.shape[0])
        if parname == None:
            parname   = np.array(fit.parname)[iparams]
    plt.figure(fignum, figsize=(8,8))
    plt.clf()
    if istitle:
        plt.suptitle(event.eventname + ' Trace Plots',size=16)
    numfp     = len(nonfixedpars)
    plt.subplots_adjust(left=0.15,right=0.95,bottom=0.10,top=0.90,hspace=0.20,wspace=0.20)
    k = 1
    for i in nonfixedpars:
        a = plt.subplot(numfp,1,k)
        #if parname[i].startswith('System Flux'):
        #    a.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%0.0f'))
        plt.plot(allparams[i],',')
        s = parname[i].replace(',','\n')
        plt.ylabel(s,size=10, multialignment='center')
        plt.yticks(size=10)
        if i == nonfixedpars[-1]:
            plt.xticks(size=10)
            plt.xlabel('MCMC step number',size=10)
        else:
            plt.xticks(visible=False)
        k += 1
    
    if savefile != None:
        plt.savefig(savefile)
    return

# Autocorrelation plots of trace values
def autocorr(event, fit, fignum, savefile=None, allparams=None, parname=None, iparams=None, stepsize=None, istitle=True):
    if stepsize == None:
        try:
            stepsize = event.params.stepsize
        except:
            stepsize = 10
    if allparams == None:
        allparams = fit.allparams
    if parname == None:
        parname   = fit.parname
    if iparams == None:
        nonfixedpars = fit.nonfixedpars
    else:
        nonfixedpars = range(allparams.shape[0])
        if parname == None:
            parname   = np.array(fit.parname)[iparams]
    plt.figure(fignum, figsize=(8,8))
    plt.clf()
    if istitle:
        plt.suptitle(event.eventname + ' Autocorrelation Plots',size=16)
    numfp     = len(nonfixedpars)
    plt.subplots_adjust(left=0.15,right=0.95,bottom=0.10,top=0.90,hspace=0.20,wspace=0.20)
    k = 0
    for i in nonfixedpars:
        a = plt.subplot(numfp,1,k+1)
        plt.plot(fit.autocorr[k],'-', lw=2)
        line = np.zeros(len(fit.autocorr[k]))
        plt.plot(line, 'k-')
        s = parname[i].replace(',','\n')
        plt.ylabel(s,size=10, multialignment='center')
        plt.yticks(size=10)
        if i == nonfixedpars[-1]:
            plt.xticks(size=10)
            plt.xlabel('MCMC step number (up to first 2,000)',size=10)
        else:
            plt.xticks(visible=False)
        k += 1
    
    if savefile != None:
        plt.savefig(savefile)
    return

# Correlation plots with 2D histograms
def hist2d(event, fit, fignum, savefile=None, allparams=None, parname=None, iparams=None, stepsize=None, istitle=True):
    if stepsize == None:
        try:
            stepsize = event.params.stepsize
        except:
            stepsize = 10
    if allparams == None:
        allparams = fit.allparams
    if parname == None:
        parname   = fit.parname
    if iparams == None:
        nonfixedpars = fit.nonfixedpars
    else:
        nonfixedpars = range(allparams.shape[0])
        parname   = np.array(parname)[iparams]
    #palette = plt.matplotlib.colors.LinearSegmentedColormap('jet2',plt.cm.datad['jet'],65536)
    palette = plt.matplotlib.colors.LinearSegmentedColormap('YlOrRd2',plt.cm.datad['YlOrRd'],65536)
    palette.set_under(color='w')
    plt.figure(fignum, figsize=(8,8))
    plt.clf()
    if istitle:
        plt.suptitle(event.eventname + ' Correlation Plots with 2D Histograms',size=16)
    numfp     = len(nonfixedpars)
    paramcorr = np.corrcoef(allparams)
    h     = 1
    m     = 1
    plt.subplots_adjust(left=0.15,right=0.95,bottom=0.15,top=0.95,hspace=0.15,wspace=0.15)
    for i in nonfixedpars[1:numfp]:
        n     = 0
        for k in nonfixedpars[0:numfp-1]:
            if i > k:
                a = plt.subplot(numfp-1,numfp-1,h)
                #a.set_axis_bgcolor(plt.cm.YlOrRd(np.abs(paramcorr[m,n])))
                #if parname[i].startswith('System Flux'):
                #    a.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%0.0f'))
                #if parname[k].startswith('System Flux'):
                #    a.xaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%0.0f'))
                if k == nonfixedpars[0]:
                    plt.yticks(size=11)
                    s = parname[i].replace(',','\n')
                    plt.ylabel(s, size = 12, multialignment='center')
                else:
                    a = plt.yticks(visible=False)
                if i == nonfixedpars[numfp-1]:
                    plt.xticks(size=11,rotation=90)
                    s = parname[k].replace(',','\n')
                    plt.xlabel(s, size = 12)
                else:
                    a = plt.xticks(visible=False)
                hist2d, xedges, yedges = np.histogram2d(allparams[k,0::int(stepsize)],
                                                        allparams[i,0::int(stepsize)],20,density=True)
                vmin = np.min(hist2d[np.where(hist2d > 0)])
                #largerhist = np.zeros((22,22))
                #largerhist[1:-1,1:-1] = hist2d
                a = plt.imshow(hist2d.T,extent=(xedges[0],xedges[-1],yedges[0],yedges[-1]), #cmap=palette, 
                               vmin=vmin, aspect='auto', origin='lower') #,interpolation='bicubic')
            h += 1
            n +=1
        m +=1
    
    if numfp > 2:
        a = plt.subplot(numfp-1, numfp-1, numfp-1, frameon=False)
        a.yaxis.set_visible(False)
        a.xaxis.set_visible(False)
        a = plt.imshow([[0,1],[0,0]], cmap=plt.cm.YlOrRd, visible=False)
        a = plt.text(1.4, 0.5, 'Normalized Point Density', rotation='vertical', ha='center', va='center')
        a = plt.colorbar()
    else:
        a = plt.colorbar()
    if savefile != None:
        plt.savefig(savefile)
    return

# 1D histogram plots
def histograms(event, fit, fignum, savefile=None, allparams=None, parname=None, iparams=None, stepsize=None, istitle=True):
    if stepsize == None:
        try:
            stepsize = event.params.stepsize
        except:
            stepsize = 10
    if allparams == None:
        allparams = fit.allparams
    if parname == None:
        parname   = fit.parname
    if iparams == None:
        nonfixedpars = fit.nonfixedpars
    else:
        nonfixedpars = range(allparams.shape[0])
        if parname == None:
            parname   = np.array(fit.parname)[iparams]
    j          = 1
    numfp      = len(nonfixedpars)
    histheight = np.min((int(4*np.ceil(numfp/3.)),8))
    if histheight == 4:
        bottom = 0.23
        hspace = 0.40
    elif histheight == 8:
        bottom = 0.13
        hspace = 0.40
    else:
        bottom = 0.12
        hspace = 0.65
    plt.figure(fignum, figsize=(8,histheight))
    plt.clf()
    if istitle:
        a = plt.suptitle(event.eventname + ' Histograms', size=16)
    for i in nonfixedpars:
        a = plt.subplot(np.ceil(numfp/3.),3,j)
        #if parname[i].startswith('System Flux'):
        #    a.xaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%0.0f'))
        plt.xticks(size=12,rotation=90)
        plt.yticks(size=12)
        #plt.axvline(x=fit.meanp[i,0])
        plt.xlabel(parname[i], size=14)
        a  = plt.hist(allparams[i,0::int(stepsize)], 20, density=False, label=str(fit.meanp[i,0]))
        j += 1
    plt.subplots_adjust(left=0.07,right=0.95,bottom=bottom,top=0.95,hspace=hspace,wspace=0.25)
    if savefile != None:
        plt.savefig(savefile)
    return

# Projections of position sensitivity along x and y
def ipprojections(event, fit, fignum, savefile=None, istitle=True):
    plt.rcParams.update({'legend.fontsize':11})
    plt.figure(fignum, figsize=(8,4))
    plt.clf()
    if istitle:
        a = plt.suptitle(event.eventname + ' Projections of Position Sensitivity', size=16)
    yround = fit.yuc[0] - fit.y[0]
    xround = fit.xuc[0] - fit.x[0]
    plt.subplot(1,2,1)
    plt.errorbar(yround+fit.binyy, fit.binyflux/fit.binybestgw, fit.binyflstd, fmt='ro', label='Binned Flux', zorder=1)
    plt.plot(yround+fit.binyy, fit.binybestip/fit.binybestgw, 'k-', lw=2, label='BLISS Map', zorder=3)
    plt.xlabel('Pixel Postion in y', size=14)
    plt.ylabel('Normalized Flux', size=14)
    plt.xticks(rotation=90)
    plt.legend(loc='best')
    plt.subplot(1,2,2)
    plt.errorbar(xround+fit.binxx, fit.binxflux/fit.binxbestgw, fit.binxflstd, fmt='bo', label='Binned Flux', zorder=1)
    plt.plot(xround+fit.binxx, fit.binxbestip/fit.binxbestgw, 'k-', lw=2, label='BLISS Map', zorder=3)
    plt.xlabel('Pixel Postion in x', size=14)
    plt.xticks(rotation=90)
    #a = plt.ylabel('Normalized Flux', size=14)
    a = plt.legend(loc='best')
    plt.subplots_adjust(left=0.11,right=0.97,bottom=0.20,top=0.90,wspace=0.20)
    if savefile != None:
        plt.savefig(savefile)
    return

# Projections of position sensitivity along x and y
def ipprojections_GWo(event, fit, fignum, savefile=None, istitle=True):
    plt.rcParams.update({'legend.fontsize':11})
    plt.figure(fignum, figsize=(8,4))
    plt.clf()
    if istitle:
        a = plt.suptitle(event.eventname + ' Projections of Gaussian Half-Width Fits', size=16)
    yround = fit.syuc[0] - fit.sy[0]
    xround = fit.sxuc[0] - fit.sx[0]
    sxbestbm = fit.binsxbestbm
    sybestbm = fit.binsybestbm
    plt.subplot(1,2,1)
    plt.errorbar(yround+fit.binsyy, fit.binsyflux/sybestbm, fit.binsyflstd, fmt='ro', label='Binned Flux - BM', zorder=1)
    plt.plot(yround+fit.binsyy, fit.binsybestgw, 'k-', lw=2, label='cubicgw y-fit', zorder=3)
    plt.xlabel('Gaussian Width in y', size=14)
    plt.ylabel('Normalized Flux', size=14)
    plt.xticks(rotation=90)
    plt.legend(loc='best')
    plt.subplot(1,2,2)
    plt.errorbar(xround+fit.binsxx, fit.binsxflux/sxbestbm, fit.binsxflstd, fmt='bo', label='Binned Flux - BM', zorder=1)
    plt.plot(xround+fit.binsxx, fit.binsxbestgw, 'k-', lw=2, label='cubicgw x-fit', zorder=3)
    plt.xlabel('Gaussian Width in x', size=14)
    plt.xticks(rotation=90)
    #a = plt.ylabel('Normalized Flux', size=14)
    a = plt.legend(loc='best')
    plt.subplots_adjust(left=0.11,right=0.97,bottom=0.20,top=0.90,wspace=0.20)
    if savefile != None:
        plt.savefig(savefile)
    return

# BLISS map
def blissmap(event, fit, fignum, savefile=None, istitle=True, minnumpts=1, srcest=None):
    palette   = plt.cm.terrain#plt.matplotlib.colors.LinearSegmentedColormap('jet3',plt.cm.datad['jet'],16384)
    palette.set_under(alpha=0.0, color='w')
    # Determine size of non-zero region
    vmin = fit.binipflux[np.where(fit.binipflux > 0)].min()
    vmax = fit.binipflux.max()
    yround = fit.yuc[0] - fit.y[0]
    xround = fit.xuc[0] - fit.x[0]
    xmin = fit.xygrid[0][np.where(fit.numpts>=minnumpts)].min() + xround
    xmax = fit.xygrid[0][np.where(fit.numpts>=minnumpts)].max() + xround
    ymin = fit.xygrid[1][np.where(fit.numpts>=minnumpts)].min() + yround
    ymax = fit.xygrid[1][np.where(fit.numpts>=minnumpts)].max() + yround
    ixmin = np.where(fit.xygrid[0] + xround == xmin)[1][0]
    ixmax = np.where(fit.xygrid[0] + xround == xmax)[1][0]
    iymin = np.where(fit.xygrid[1] + yround == ymin)[0][0]
    iymax = np.where(fit.xygrid[1] + yround == ymax)[0][0]
    # Plot
    plt.figure(fignum, figsize=(8,6))
    plt.clf()
    if istitle:
        a = plt.suptitle(event.eventname + ' BLISS Map', size=16)
    if fit.model.__contains__('nnint'):
        interp = 'nearest'
    else:
        interp = 'bilinear'
    #MAP
    a = plt.axes([0.11,0.10,0.75,0.80])
    if 'mmbilinint' in fit.model:
        plt.imshow(fit.mastermapFI.reshape(fit.xygrid[0].shape)[iymin:iymax+1,ixmin:ixmax+1], cmap=plt.cm.Greys, vmin=vmin, vmax=vmax, origin='lower',
               extent=(xmin,xmax,ymin,ymax), aspect='auto', interpolation=interp,alpha=0.3,zorder=0)
    plt.imshow(fit.binipflux[iymin:iymax+1,ixmin:ixmax+1], cmap=palette, vmin=vmin, vmax=vmax, origin='lower', 
               extent=(xmin,xmax,ymin,ymax), aspect='auto',zorder=1)#, interpolation=interp)
    plt.ylabel('Pixel Position in y', size=14)
    plt.xlabel('Pixel Position in x', size=14)
    #print(xmin,xmax,ymin,ymax)
    if srcest == None:
        #Spitzer
        if ymin < -0.5+yround:
            plt.hlines(-0.5+yround, xmin, xmax, 'k')
        if ymax >  0.5+yround:
            plt.hlines( 0.5+yround, xmin, xmax, 'k')
        if xmin < -0.5+xround:
            plt.vlines(-0.5+xround, ymin, ymax, 'k')
        if xmax >  0.5+xround:
            plt.vlines( 0.5+xround, ymin, ymax, 'k')
    else:
        #K2
        if ymin < -0.5+srcest[0]:
            plt.hlines(-0.5+srcest[0], xmin, xmax, 'k')
        if ymax >  0.5+srcest[0]:
            plt.hlines( 0.5+srcest[0], xmin, xmax, 'k')
        if xmin < -0.5+srcest[1]:
            plt.vlines(-0.5+srcest[1], ymin, ymax, 'k')
        if xmax >  0.5+srcest[1]:
            plt.vlines( 0.5+srcest[1], ymin, ymax, 'k')
    #COLORBAR
    a = plt.axes([0.90,0.10,0.01,0.8], frameon=False)
    a.yaxis.set_visible(False)
    a.xaxis.set_visible(False)
    a = plt.imshow([[vmin,vmax],[vmin,vmax]], cmap=plt.cm.terrain, aspect='auto', visible=False)
    plt.colorbar(a, fraction=3.0)
    if savefile != None:
        plt.savefig(savefile)
    return


# Pointing Histogram
def pointingHist(event, fit, fignum, savefile=None, istitle=True, minnumpts=1, srcest=None):
    palette   = plt.matplotlib.colors.LinearSegmentedColormap('jet3',plt.cm.datad['jet'],16384)
    palette.set_under(alpha=0.0, color='w')
    # Determine size of non-zero region
    vmin = 1
    vmax = fit.numpts.max()
    yround = fit.yuc[0] - fit.y[0]
    xround = fit.xuc[0] - fit.x[0]
    xmin = fit.xygrid[0][np.where(fit.numpts>=minnumpts)].min() + xround
    xmax = fit.xygrid[0][np.where(fit.numpts>=minnumpts)].max() + xround
    ymin = fit.xygrid[1][np.where(fit.numpts>=minnumpts)].min() + yround
    ymax = fit.xygrid[1][np.where(fit.numpts>=minnumpts)].max() + yround
    ixmin = np.where(fit.xygrid[0] + xround == xmin)[1][0]
    ixmax = np.where(fit.xygrid[0] + xround == xmax)[1][0]
    iymin = np.where(fit.xygrid[1] + yround == ymin)[0][0]
    iymax = np.where(fit.xygrid[1] + yround == ymax)[0][0]
    # Plot
    plt.figure(fignum, figsize=(8,6))
    plt.clf()
    if istitle:
        a = plt.suptitle(event.eventname + ' Pointing Histogram', size=16)
    #MAP
    a = plt.axes([0.11,0.10,0.75,0.80])
    plt.imshow(fit.numpts[iymin:iymax+1,ixmin:ixmax+1], cmap=palette, vmin=vmin, vmax=vmax,
               origin='lower', extent=(xmin,xmax,ymin,ymax), aspect='auto', interpolation='nearest')
    plt.ylabel('Pixel Position in y', size=14)
    plt.xlabel('Pixel Position in x', size=14)
    if srcest == None:
        #Spitzer
        #print(yround,xround)
        if ymin < -0.5+yround:
            plt.hlines(-0.5+yround, xmin, xmax, 'k')
        if ymax >  0.5+yround:
            plt.hlines( 0.5+yround, xmin, xmax, 'k')
        if xmin < -0.5+xround:
            plt.vlines(-0.5+xround, ymin, ymax, 'k')
        if xmax >  0.5+xround:
            plt.vlines( 0.5+xround, ymin, ymax, 'k')
    else:
        #K2
        if ymin < -0.5+srcest[0]:
            plt.hlines(-0.5+srcest[0], xmin, xmax, 'k')
        if ymax >  0.5+srcest[0]:
            plt.hlines( 0.5+srcest[0], xmin, xmax, 'k')
        if xmin < -0.5+srcest[1]:
            plt.vlines(-0.5+srcest[1], ymin, ymax, 'k')
        if xmax >  0.5+srcest[1]:
            plt.vlines( 0.5+srcest[1], ymin, ymax, 'k')
    #COLORBAR
    a = plt.axes([0.90,0.10,0.01,0.8], frameon=False)
    a.yaxis.set_visible(False)
    a.xaxis.set_visible(False)
    a = plt.imshow([[vmin,vmax],[vmin,vmax]], cmap=plt.cm.jet, aspect='auto', visible=False)
    plt.colorbar(a, fraction=3.0)
    if savefile != None:
        plt.savefig(savefile)
    return

# PRF WIDTH
def prfghw(event, fit, fignum, savefile=None, istitle=True, srcest=None, axis='y'):
    if axis == 'y':
        vmin = event.sy[np.where(event.x>0)].min()
        vmax = event.sy[np.where(event.x>0)].max()
    elif axis == 'x':
        vmin = event.sx[np.where(event.x>0)].min()
        vmax = event.sx[np.where(event.x>0)].max()
    else:
        print('Specify axis as y or x')
        return
    yround = fit.yuc[0] - fit.y[0]
    xround = fit.xuc[0] - fit.x[0]
    # Determine size of non-zero region
    xmin = event.x[np.where(event.x>0)].min()
    xmax = event.x.max()
    ymin = event.y[np.where(event.y>0)].min()
    ymax = event.y.max()
    # Plot
    plt.figure(fignum, figsize=(8,6))
    plt.clf()
    if istitle:
        a = plt.suptitle(event.eventname + ' PRF Gaussian Half-Width in ' + axis, size=16)
    #MAP
    a = plt.axes([0.11,0.10,0.75,0.80])
    if axis == 'y':
        plt.scatter(event.x, event.y, c=event.sy, vmin=vmin, vmax=vmax)
    elif axis == 'x':
        plt.scatter(event.x, event.y, c=event.sx, vmin=vmin, vmax=vmax)
    plt.ylabel('Pixel Position in y', size=14)
    plt.xlabel('Pixel Position in x', size=14)
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    #print(xmin,xmax,ymin,ymax)
    if srcest == None:
        #Spitzer
        if ymin < -0.5+yround:
            plt.hlines(-0.5+yround, xmin, xmax, 'k')
        if ymax >  0.5+yround:
            plt.hlines( 0.5+yround, xmin, xmax, 'k')
        if xmin < -0.5+xround:
            plt.vlines(-0.5+xround, ymin, ymax, 'k')
        if xmax >  0.5+xround:
            plt.vlines( 0.5+xround, ymin, ymax, 'k')
    else:
        #K2
        if ymin < -0.5+srcest[0]:
            plt.hlines(-0.5+srcest[0], xmin, xmax, 'k')
        if ymax >  0.5+srcest[0]:
            plt.hlines( 0.5+srcest[0], xmin, xmax, 'k')
        if xmin < -0.5+srcest[1]:
            plt.vlines(-0.5+srcest[1], ymin, ymax, 'k')
        if xmax >  0.5+srcest[1]:
            plt.vlines( 0.5+srcest[1], ymin, ymax, 'k')
    #COLORBAR
    a = plt.axes([0.90,0.10,0.01,0.8], frameon=False)
    a.yaxis.set_visible(False)
    a.xaxis.set_visible(False)
    a = plt.imshow([[vmin,vmax],[vmin,vmax]], aspect='auto', visible=False)
    plt.colorbar(a, fraction=3.0)
    if savefile != None:
        plt.savefig(savefile)
    return

# Plot RMS vs. bin size looking for time-correlated noise
def rmsplot(event, fit, fignum, savefile=None, istitle=True, stderr=None, normfactor=None):
    if stderr == None:
        stderr = fit.stderr
    if normfactor == None:
        normfactor = 1e-6
    plt.rcParams.update({'legend.fontsize':11})
    plt.figure(fignum, figsize=(8,6))
    plt.clf()
    if istitle:
        a = plt.suptitle(event.eventname + ' Correlated Noise', size=16)
    plt.loglog(fit.binsz, fit.rms/normfactor, color='black', lw=1.5, label='Fit RMS', zorder=3)    # our noise
    plt.loglog(fit.binsz, stderr/normfactor, color='red', ls='-', lw=2, label='Std. Err.', zorder=1) # expected noise
    plt.xlim(0, fit.binsz[-1]*2)
    plt.ylim(stderr[-1]/normfactor/2., stderr[0]/normfactor*2.)
    plt.xlabel("Bin Size", fontsize=14)
    plt.ylabel("RMS (ppm)", fontsize=14)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.legend()
    if savefile != None:
        plt.savefig(savefile)
    return

# Plot RMS vs. bin size (with RMS uncertainties) looking for time-correlated noise
def rmsploterr(event, fit, fignum, savefile=None, istitle=True, stderr=None, normfactor=None):
    if stderr == None:
        stderr = fit.stderr
    if normfactor == None:
        normfactor = stderr[0]
    plt.rcParams.update({'legend.fontsize':11})
    plt.figure(fignum, figsize=(8,6))
    plt.clf()
    if istitle:
        a = plt.suptitle(event.eventname + ' Correlated Noise', size=16)
    plt.loglog(fit.binsz, fit.rms/normfactor, color='black', lw=1.5, label='Fit RMS')    # our noise
    plt.loglog(fit.binsz, stderr/normfactor, color='red', ls='-', lw=2, label='Std. Err.') # expected noise
    plt.xlim(0, fit.binsz[-1]*2)
    plt.ylim(stderr[-1]/normfactor/2., stderr[0]/normfactor*2.)
    plt.xlabel("Bin Size", fontsize=14)
    plt.ylabel("Normalized RMS", fontsize=14)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.legend()
    if savefile != None:
        plt.savefig(savefile)
    return
