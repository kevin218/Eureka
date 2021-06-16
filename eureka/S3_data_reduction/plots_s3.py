import numpy as np
import matplotlib.pyplot as plt


def image_and_background(ev, intstart, n, subdata, submask, subbg):
    plt.figure(3301)
    plt.clf()
    plt.suptitle(str(intstart + n))
    plt.subplot(211)
    max = np.max(subdata[n] * submask[n])
    plt.imshow(subdata[n] * submask[n], origin='lower', aspect='auto', vmin=0, vmax=max / 10)
    # plt.imshow(subdata[n], origin='lower', aspect='auto', vmin=0, vmax=10000)
    plt.subplot(212)
    # plt.imshow(submask[i], origin='lower', aspect='auto', vmax=1)
    median = np.median(subbg[n])
    std = np.std(subbg[n])
    plt.imshow(subbg[n], origin='lower', aspect='auto', vmin=median - 3 * std, vmax=median + 3 * std)
    # plt.imshow(submask[n], origin='lower', aspect='auto', vmin=0, vmax=1)
    plt.savefig(ev.workdir + '/figs/fig3301-' + str(intstart + n) + '-Image+Background.png')
    # plt.pause(0.1)

def optimal_spectrum(ev, intstart, n, subnx, stdspec, optspec, opterr):
    plt.figure(3302)
    plt.clf()
    plt.suptitle(str(intstart + n))
    plt.plot(range(subnx), stdspec[n], '-', color='C1', label='Std Spec')
    # plt.errorbar(range(subnx), stdspec[n], yerr=np.sqrt(stdvar[n]), fmt='-', color='C1', ecolor='C0', label='Std Spec')
    plt.errorbar(range(subnx), optspec[n], opterr[n], fmt='-', color='C2', ecolor='C2', label='Optimal Spec')
    plt.legend(loc='best')
    plt.savefig(ev.workdir + '/figs/fig3302-' + str(intstart + n) + '-Spectrum.png')
    # plt.pause(0.1)

def lc_nodriftcorr(ev):
    plt.figure(3101, figsize=(8, 8))  # ev.n_files/20.+0.8))
    plt.clf()
    wmin = ev.wave_1d.min()
    wmax = ev.wave_1d.max()
    n_int, nx = ev.optspec.shape
    # iwmin       = np.where(ev.wave[src_ypos]>wmin)[0][0]
    # iwmax       = np.where(ev.wave[src_ypos]>wmax)[0][0]
    vmin = 0.97
    vmax = 1.03
    # normspec    = np.mean(ev.optspec,axis=1)/np.mean(ev.optspec[ev.inormspec[0]:ev.inormspec[1]],axis=(0,1))
    normspec = ev.optspec / np.mean(ev.optspec, axis=0)
    plt.imshow(normspec, origin='lower', aspect='auto', extent=[wmin, wmax, 0, n_int], vmin=vmin, vmax=vmax,
               cmap=plt.cm.RdYlBu_r)
    ediff = np.zeros(n_int)
    for m in range(n_int):
        ediff[m] = 1e6 * np.median(np.abs(np.ediff1d(normspec[m])))
        # plt.scatter(ev.wave[src_ypos], np.zeros(nx)+m, c=normspec[m],
        #             s=14,linewidths=0,vmin=vmin,vmax=vmax,marker='s',cmap=plt.cm.RdYlBu_r)
    plt.title("MAD = " + str(np.round(np.mean(ediff), 0).astype(int)) + " ppm")
    # plt.xlim(wmin,wmax)
    # plt.ylim(0,n_int)
    plt.ylabel('Integration Number')
    plt.xlabel(r'Wavelength ($\mu m$)')
    plt.colorbar(label='Normalized Flux')
    plt.tight_layout()
    plt.savefig(ev.workdir + '/figs/fig3101-2D_LC.png')
