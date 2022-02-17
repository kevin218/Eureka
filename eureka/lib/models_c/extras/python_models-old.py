import numpy as np
import scipy.interpolate as spi
import sys, re, os

try:
    import models_c as mc
except:
    import models as mc
#reload(mc)
import orbit
import smoothing
#reload(smoothing)

def setupmodel(model, ind):
   """
   This function sets up the indeces and parameter names for the given model, also defined in this file.

  Parameters
  ----------
    model:  Array of model names
    ind:    Empty object for parameter indeces

  Notes
  -----
    FUNCTYPE = 'ecl/tr'     : eclipse model function
               'ramp'       : ramp model function
               'ippoly'     : intra-pixel model function
               'posoffset'  : position offset model function
               'vissen'     : visit sensitivity function
               'flatf'      : flat field correction
               'ipmap'      : Interpolated intra-pixel model function (BLISS map)

  Returns
  -------
    This function returns:
    myfuncs:    List of functions
    functype:   List of function types
    parname:    List parameter names
    ind:        Object containing parameter indices
    saveext:    Save filename extension

   """
   
   parname  = []
   myfuncs  = []
   saveext  = []
   functype = []
   ind.expramplist = []
   for i in range(len(model)):
    if   model[i] == 'mandelecl':
        #DEFINE INDICES
        ind.midpt = ind.size
        ind.width = ind.size + 1
        ind.depth = ind.size + 2
        ind.t12   = ind.size + 3
        ind.t34   = ind.size + 4
        ind.flux  = ind.size + 5
        ind.size += 6
        #DEFINE NAMES
        parname.insert(ind.midpt,'Eclipse Phase')
        parname.insert(ind.width,'Eclipse Duration')
        parname.insert(ind.depth,'Eclipse Depth')
        parname.insert(ind.t12,  'Ingress Time')
        parname.insert(ind.t34,  'Egress Time')
        parname.insert(ind.flux,r'System Flux ($\mu Jy$)')
        #DEFINE ECLIPSE MODEL
        myfuncs.append(mc.mandelecl)
        saveext.append('m1')
        functype.append('ecl/tr')
    elif model[i] == 'mandelecl2':
        #DEFINE INDICES
        ind.midpt2 = ind.size
        ind.width2 = ind.size + 1
        ind.depth2 = ind.size + 2
        ind.t122   = ind.size + 3
        ind.t342   = ind.size + 4
        ind.flux2  = ind.size + 5
        ind.size  += 6
        #DEFINE NAMES
        parname.insert(ind.midpt2,'Eclipse Phase')
        parname.insert(ind.width2,'Eclipse Duration')
        parname.insert(ind.depth2,'Eclipse Depth')
        parname.insert(ind.t122,  'Ingress Time')
        parname.insert(ind.t342,  'Egress Time')
        parname.insert(ind.flux2,r'System Flux ($\mu Jy$)')
        #DEFINE ECLIPSE MODEL
        myfuncs.append(mc.mandelecl)
        saveext.append('m2')
        functype.append('ecl/tr')
    elif model[i] == 'mandelecl3':
        #DEFINE INDICES
        ind.midpt3 = ind.size
        ind.width3 = ind.size + 1
        ind.depth3 = ind.size + 2
        ind.t123   = ind.size + 3
        ind.t343   = ind.size + 4
        ind.flux3  = ind.size + 5
        ind.size  += 6
        #DEFINE NAMES
        parname.insert(ind.midpt3,'Eclipse Phase')
        parname.insert(ind.width3,'Eclipse Duration')
        parname.insert(ind.depth3,'Eclipse Depth')
        parname.insert(ind.t123,  'Ingress Time')
        parname.insert(ind.t343,  'Egress Time')
        parname.insert(ind.flux3,r'System Flux ($\mu Jy$)')
        #DEFINE ECLIPSE MODEL
        myfuncs.append(mc.mandelecl)
        saveext.append('m3')
        functype.append('ecl/tr')
    elif model[i] == 'mandeltr':
        #DEFINE INDICES
        ind.trmidpt = ind.size
        ind.trrprs  = ind.size + 1
        ind.trcosi  = ind.size + 2
        ind.trars   = ind.size + 3
        ind.trflux  = ind.size + 4
        ind.trper   = ind.size + 5
        ind.size  += 6
        #DEFINE NAMES
        parname.insert(ind.trmidpt,  'Transit Midpoint')
        parname.insert(ind.trrprs,   'Radius Ratio')
        parname.insert(ind.trcosi,   'Cos Inclination')
        parname.insert(ind.trars,    'a/Rs')
        parname.insert(ind.trflux,  r'System Flux ($\mu Jy$)')
        parname.insert(ind.trper,    'Period')
        #DEFINE ECLIPSE MODEL
        myfuncs.append(mc.mandeltr)
        saveext.append('t1')
        functype.append('ecl/tr')
    elif model[i] == 'mandeltr2':
        #DEFINE INDICES
        ind.trmidpt2 = ind.size
        ind.trrprs2  = ind.size + 1
        ind.trcosi2  = ind.size + 2
        ind.trars2   = ind.size + 3
        ind.trflux2  = ind.size + 4
        ind.trper2   = ind.size + 5
        ind.size  += 6
        #DEFINE NAMES
        parname.insert(ind.trmidpt2,  'Transit Midpoint')
        parname.insert(ind.trrprs2,   'Radius Ratio')
        parname.insert(ind.trcosi2,   'Cos Inclination')
        parname.insert(ind.trars2,    'a/Rs')
        parname.insert(ind.trflux2,  r'System Flux ($\mu Jy$)')
        parname.insert(ind.trper2,    'Period')
        #DEFINE ECLIPSE MODEL
        myfuncs.append(mc.mandeltr)
        saveext.append('t2')
        functype.append('ecl/tr')
    elif model[i] == 'trquad':
        #DEFINE INDICES
        ind.trqmid  = ind.size
        ind.trqrprs = ind.size + 1
        ind.trqcosi = ind.size + 2
        ind.trqars  = ind.size + 3
        ind.trqf    = ind.size + 4
        ind.trqp    = ind.size + 5
        ind.trqc1   = ind.size + 6
        ind.trqc2   = ind.size + 7
        ind.size   += 8
        #DEFINE NAMES
        parname.insert(ind.trqmid,  'Transit Midpoint')
        parname.insert(ind.trqrprs, 'Radius Ratio')
        parname.insert(ind.trqcosi, 'Cos Inclination')
        parname.insert(ind.trqars,  'a/Rs')
        parname.insert(ind.trqf,    'System Flux')
        parname.insert(ind.trqp,    'Period')
        parname.insert(ind.trqc1,   'Limb-Darkening, c1')
        parname.insert(ind.trqc2,   'Limb-Darkening, c2')
        #DEFINE ECLIPSE MODEL
        myfuncs.append(mc.trquad)
        saveext.append('trq1')
        functype.append('ecl/tr')
    elif model[i] == 'trquad2':
        #DEFINE INDICES
        ind.trq2mid  = ind.size
        ind.trq2rprs = ind.size + 1
        ind.trq2cosi = ind.size + 2
        ind.trq2ars  = ind.size + 3
        ind.trq2f    = ind.size + 4
        ind.trq2p    = ind.size + 5
        ind.trq2c1   = ind.size + 6
        ind.trq2c2   = ind.size + 7
        ind.size    += 8
        #DEFINE NAMES
        parname.insert(ind.trq2mid,  'Transit Midpoint')
        parname.insert(ind.trq2rprs, 'Radius Ratio')
        parname.insert(ind.trq2cosi, 'Cos Inclination')
        parname.insert(ind.trq2ars,  'a/Rs')
        parname.insert(ind.trq2f,    'System Flux')
        parname.insert(ind.trq2p,    'Period')
        parname.insert(ind.trq2c1,   'Limb-Darkening, c1')
        parname.insert(ind.trq2c2,   'Limb-Darkening, c2')
        #DEFINE ECLIPSE MODEL
        myfuncs.append(mc.trquad)
        saveext.append('trq2')
        functype.append('ecl/tr')
    elif model[i] == 'trnlldsp':
        #DEFINE INDICES
        ind.trspmid = ind.size
        ind.rprs    = ind.size + 1
        ind.cosi    = ind.size + 2
        ind.ars     = ind.size + 3
        ind.trspf   = ind.size + 4
        ind.trspp   = ind.size + 5
        ind.trspc1  = ind.size + 6
        ind.trspc2  = ind.size + 7
        ind.trspc3  = ind.size + 8
        ind.trspc4  = ind.size + 9
        ind.size  += 10
        #DEFINE NAMES
        parname.insert(ind.trspmid,'Transit Midpoint')
        parname.insert(ind.rprs,   'Radius Ratio')
        parname.insert(ind.cosi,   'Cos Inclination')
        parname.insert(ind.ars,    'a/Rs')
        parname.insert(ind.trspf,  'System Flux')
        parname.insert(ind.trspp,  'Period')
        parname.insert(ind.trspc1, 'Limb-Darkening, c1')
        parname.insert(ind.trspc2, 'Limb-Darkening, c2')
        parname.insert(ind.trspc3, 'Limb-Darkening, c3')
        parname.insert(ind.trspc4, 'Limb-Darkening, c4')
        #DEFINE ECLIPSE MODEL
        myfuncs.append(mc.trnlldsp)
        saveext.append('tsp1')
        functype.append('ecl/tr')
    elif model[i] == 'trnlldsp2':
        #DEFINE INDICES
        ind.trspmid2 = ind.size
        ind.rprs2    = ind.size + 1
        ind.cosi2    = ind.size + 2
        ind.ars2     = ind.size + 3
        ind.trspf2   = ind.size + 4
        ind.trspp2   = ind.size + 5
        ind.trspc12  = ind.size + 6
        ind.trspc22  = ind.size + 7
        ind.trspc32  = ind.size + 8
        ind.trspc42  = ind.size + 9
        ind.size  += 10
        #DEFINE NAMES
        parname.insert(ind.trspmid2,'Transit Midpoint')
        parname.insert(ind.rprs2,   'Radius Ratio')
        parname.insert(ind.cosi2,   'Cos Inclination')
        parname.insert(ind.ars2,    'a/Rs')
        #parname.insert(ind.trspf2, r'System Flux ($\mu Jy$)')
        parname.insert(ind.trspf2,  'System Flux')
        parname.insert(ind.trspp2,  'Period')
        parname.insert(ind.trspc12, 'Limb-Darkening, c1')
        parname.insert(ind.trspc22, 'Limb-Darkening, c2')
        parname.insert(ind.trspc32, 'Limb-Darkening, c3')
        parname.insert(ind.trspc42, 'Limb-Darkening, c4')
        #DEFINE ECLIPSE MODEL
        myfuncs.append(mc.trnlldsp)
        saveext.append('tsp2')
        functype.append('ecl/tr')
    elif model[i] == 'mandelgeom':
        #DEFINE INDICES
        ind.midpt3 = ind.size
        ind.width3 = ind.size + 1
        ind.rp_rs = ind.size + 2
        ind.b3   = ind.size + 3
        ind.flux  = ind.size + 4
        ind.size  += 5
        #DEFINE NAMES
        parname.insert(ind.midpt3,'Center')
        parname.insert(ind.width3,'Duration')
        parname.insert(ind.rp_rs,r'$R_p/R_s$')
        parname.insert(ind.b3,  'Impact Parameter')
        parname.insert(ind.flux3,r'System Flux ($\mu Jy$)')
        #DEFINE ECLIPSE MODEL
        myfuncs.append(mandel_geom)
        saveext.append('mg')
        functype.append('ecl/tr')
    elif model[i] == 'mandelorbit':
        #DEFINE INDICES
        ind.e = ind.size
        ind.omega = ind.size + 1
        ind.i = ind.size + 2
        ind.rplanet   = ind.size + 3
        ind.rstar  = ind.size + 4
        ind.mstar = ind.size + 5
        ind.depth = ind.size + 6
        ind.flux4 = ind.size + 7
        ind.size  += 7
        #DEFINE NAMES
        parname.insert(ind.e,'Eccentricity')
        parname.insert(ind.omega,'Argument of Periapsis')
        parname.insert(ind.i,'Inclination')
        parname.insert(ind.rplanet,  r'Planet Radius ($R_J$)')
        parname.insert(ind.rstar, r'Star Radius ($R_\odot$)')
        parname.insert(ind.mstar, r'Star Mass ($M_\odot$)')
        parname.insert(ind.depth, 'Eclipse Depth')
        parname.insert(ind.flux4, r'System Flux ($\mu Jy$)')
        #DEFINE ECLIPSE MODEL
        myfuncs.append(mandelecl_orbit)
        saveext.append('mo')
        functype.append('ecl/tr')
    elif model[i] == 'ortho':
        #DEFINE INDICES
        ind.orthop = ind.size
        ind.size  += 1
        #DEFINE NAMES
        parname.insert(ind.orthop,   'Ortho, none')
        #DEFINE INTRA-PIXEL MODEL
        myfuncs.append(mc.orthoInvTrans)
        saveext.append('O')
        functype.append('ortho')
    elif model[i] == 'rednoise':
        #DEFINE INDICES
        ind.whinoise = ind.size
        ind.rednose  = ind.size + 1
        ind.gamma    = ind.size + 2
        ind.size  += 3
        #DEFINE NAMES
        parname.insert(ind.whinoise, 'White Noise')
        parname.insert(ind.rednose,  'Red Noise')
        parname.insert(ind.gamma,    'gamma')
        #DEFINE INTRA-PIXEL MODEL
        myfuncs.append(mc.rednoise)
        saveext.append('RN')
        functype.append('noise')
    elif model[i] == 'hook':
        #DEFINE INDICES
        ind.hgoal  = ind.size
        ind.h0     = ind.size + 1
        ind.h1     = ind.size + 2
        ind.hpm    = ind.size + 3
        ind.size  += 4
        #DEFINE NAMES
        parname.insert(ind.hgoal, 'Hook, Ideal Flux')
        parname.insert(ind.h0,    'Hook, h0')
        parname.insert(ind.h1,    'Hook, h1')
        parname.insert(ind.hpm,   'Hook, +/-')
        #DEFINE RAMP MODEL
        myfuncs.append(mc.hook)
        saveext.append('h')
        functype.append('ramp')
    elif model[i] == 'hook2':
        #DEFINE INDICES
        ind.h2goal  = ind.size
        ind.h20     = ind.size + 1
        ind.h21     = ind.size + 2
        ind.h2pm    = ind.size + 3
        ind.h2per   = ind.size + 4
        ind.size  += 5
        #DEFINE NAMES
        parname.insert(ind.h2goal, 'Hook2, Ideal Flux')
        parname.insert(ind.h20,    'Hook2, h0')
        parname.insert(ind.h21,    'Hook2, h1')
        parname.insert(ind.h2pm,   'Hook2, +/-')
        parname.insert(ind.h2per,  'Hook2, Period')
        #DEFINE RAMP MODEL
        myfuncs.append(mc.hook2)
        saveext.append('h2')
        functype.append('ramp')
    elif model[i] == 'heqramp':
        #DEFINE INDICES
        ind.heqt0   = ind.size
        ind.heqr0   = ind.size + 1
        ind.heqr1   = ind.size + 2
        ind.heqr2   = ind.size + 3
        ind.heqr3   = ind.size + 4
        ind.heqpm   = ind.size + 5
        ind.heqper  = ind.size + 6
        ind.size   += 7
        #DEFINE NAMES
        parname.insert(ind.heqt0,   'Hook, t0')
        parname.insert(ind.heqr0,   'Hook, r0')
        parname.insert(ind.heqr1,   'Hook, r1')
        parname.insert(ind.heqr2,   'Hook, r2')
        parname.insert(ind.heqr3,   'Hook, r3')
        parname.insert(ind.heqpm,   'Hook, +/-')
        parname.insert(ind.heqper,  'Hook, Period')
        #DEFINE RAMP MODEL
        myfuncs.append(mc.heqramp)
        saveext.append('heq')
        functype.append('ramp')
    elif model[i] == 'seramp':
        #DEFINE INDICES
        ind.segoal  = ind.size
        ind.ser0    = ind.size + 1
        ind.ser1    = ind.size + 2
        ind.sepm    = ind.size + 3
        ind.size   += 4
        #DEFINE NAMES
        parname.insert(ind.segoal, 'Ramp, Ideal Flux')
        parname.insert(ind.ser0,   'Ramp, r0')
        parname.insert(ind.ser1,   'Ramp, r1')
        parname.insert(ind.sepm,   'Ramp, +/-')
        #DEFINE RAMP MODEL
        myfuncs.append(mc.seramp)
        saveext.append('se')
        functype.append('ramp')
    elif model[i] == 'selramp':
        #DEFINE INDICES
        ind.selgoal  = ind.size
        ind.selr0    = ind.size + 1
        ind.selr1    = ind.size + 2
        ind.selr2    = ind.size + 3
        ind.selt0    = ind.size + 4
        ind.selpm    = ind.size + 5
        ind.size    += 6
        #DEFINE NAMES
        parname.insert(ind.selgoal, 'Ramp, Ideal Flux')
        parname.insert(ind.selr0,   'Ramp, r0')
        parname.insert(ind.selr1,   'Ramp, r1')
        parname.insert(ind.selr2,   'Ramp, r2')
        parname.insert(ind.selt0,   'Ramp, t0')
        parname.insert(ind.selpm,   'Ramp, +/-')
        #DEFINE RAMP MODEL
        myfuncs.append(mc.selramp)
        saveext.append('sel')
        functype.append('ramp')
    elif model[i] == 'seqramp':
        #DEFINE INDICES
        ind.seqgoal  = ind.size
        ind.seqr0    = ind.size + 1
        ind.seqr1    = ind.size + 2
        ind.seqr2    = ind.size + 3
        ind.seqr3    = ind.size + 4
        ind.seqt0    = ind.size + 5
        ind.seqpm    = ind.size + 6
        ind.size    += 7
        #DEFINE NAMES
        parname.insert(ind.seqgoal, 'Ramp, Ideal Flux')
        parname.insert(ind.seqr0,   'Ramp, r0')
        parname.insert(ind.seqr1,   'Ramp, r1')
        parname.insert(ind.seqr2,   'Ramp, r2')
        parname.insert(ind.seqr3,   'Ramp, r3')
        parname.insert(ind.seqt0,   'Ramp, t0')
        parname.insert(ind.seqpm,   'Ramp, +/-')
        #DEFINE RAMP MODEL
        myfuncs.append(mc.seqramp)
        saveext.append('seq')
        functype.append('ramp')
    elif model[i] == 'se2ramp':
        #DEFINE INDICES
        ind.se2goal  = ind.size
        ind.se2r0    = ind.size + 1
        ind.se2r1    = ind.size + 2
        ind.se2pm0   = ind.size + 3
        ind.se2r4    = ind.size + 4
        ind.se2r5    = ind.size + 5
        ind.se2pm1   = ind.size + 6
        ind.size    += 7
        #DEFINE NAMES
        parname.insert(ind.se2goal, 'Ramp, Ideal Flux')
        parname.insert(ind.se2r0,   'Ramp, r0')
        parname.insert(ind.se2r1,   'Ramp, r1')
        parname.insert(ind.se2pm0,  'Ramp, +/-')
        parname.insert(ind.se2r4,   'Ramp, r4')
        parname.insert(ind.se2r5,   'Ramp, r5')
        parname.insert(ind.se2pm1,  'Ramp, +/-')
        #DEFINE RAMP MODEL
        myfuncs.append(mc.se2ramp)
        saveext.append('se2')
        functype.append('ramp')
    elif model[i] == 'risingexp':
        #DEFINE INDICES
        ind.regoal  = ind.size
        ind.rem     = ind.size + 1
        ind.ret0    = ind.size + 2
        ind.size   += 3
        #DEFINE NAMES
        parname.insert(ind.regoal, 'Ramp, Ideal Flux')
        parname.insert(ind.rem,    'Ramp, Curvature')
        parname.insert(ind.ret0,   'Ramp, Phase Offset')
        #DEFINE RAMP MODEL
        myfuncs.append(mc.risingexp)
        saveext.append('re')
        functype.append('ramp')
    elif model[i] == 'deramp':
        #DEFINE INDICES
        ind.expramplist.append([ind.size])
        ind.degoal  = ind.size
        ind.der0    = ind.size + 1
        ind.der1    = ind.size + 2
        ind.desint  = ind.size + 3
        ind.decost  = ind.size + 4
        ind.depm    = ind.size + 5
        ind.degb    = ind.size + 6
        ind.der0b   = ind.size + 7
        ind.der1b   = ind.size + 8
        ind.size   += 9
        #DEFINE NAMES
        parname.insert(ind.degoal, 'Ramp, Ideal Flux')
        parname.insert(ind.der0,   'Ramp, r0')
        parname.insert(ind.der1,   'Ramp, r1')
        parname.insert(ind.desint, 'Ramp, sin($\theta$)')
        parname.insert(ind.decost, 'Ramp, cos($\theta$)')
        parname.insert(ind.depm,   'Ramp, +/-')
        parname.insert(ind.degb,   'Ramp, Best Ideal Flux')
        parname.insert(ind.der0b,  'Ramp, Best r0')
        parname.insert(ind.der1b,  'Ramp, Best r1')
        #DEFINE RAMP MODEL
        myfuncs.append(mc.deramp)
        saveext.append('de')
        functype.append('ramp')
    elif model[i] == 'pcaramp':
        #DEFINE INDICES
        #ind.expramplist.append([ind.size])
        ind.pcagoal  = ind.size
        ind.pcar0    = ind.size + 1
        ind.pcar1    = ind.size + 2
        ind.pcapm    = ind.size + 3
        ind.pcagb    = ind.size + 4
        ind.pcar0b   = ind.size + 5
        ind.pcar1b   = ind.size + 6
        ind.pcait    = ind.size + 7
        ind.size   += 16
        #DEFINE NAMES
        parname.insert(ind.pcagoal, 'Ramp, Ideal Flux')
        parname.insert(ind.pcar0,   'Ramp, r0')
        parname.insert(ind.pcar1,   'Ramp, r1')
        parname.insert(ind.pcapm,   'Ramp, +/-')
        parname.insert(ind.pcagb,   'Ramp, Best Ideal Flux')
        parname.insert(ind.pcar0b,  'Ramp, Best r0')
        parname.insert(ind.pcar1b,  'Ramp, Best r1')
        #DEFINE RAMP MODEL
        myfuncs.append(mc.pcaramp)
        saveext.append('pca')
        functype.append('ramp')
    elif model[i] == 'pcaramp2':
        #DEFINE INDICES
        #ind.expramplist.append([ind.size])
        ind.pcagoal  = ind.size
        ind.pcar0    = ind.size + 1
        ind.pcar1    = ind.size + 2
        ind.pcapm    = ind.size + 3
        ind.pcagb    = ind.size + 4
        ind.pcar0b   = ind.size + 5
        ind.pcar1b   = ind.size + 6
        ind.pcait    = ind.size + 7
        ind.size   += 16
        #DEFINE NAMES
        parname.insert(ind.pcagoal, 'Ramp, Ideal Flux')
        parname.insert(ind.pcar0,   'Ramp, r0')
        parname.insert(ind.pcar1,   'Ramp, r1')
        parname.insert(ind.pcapm,   'Ramp, +/-')
        parname.insert(ind.pcagb,   'Ramp, Best Ideal Flux')
        parname.insert(ind.pcar0b,  'Ramp, Best r0')
        parname.insert(ind.pcar1b,  'Ramp, Best r1')
        #DEFINE RAMP MODEL
        myfuncs.append(mc.pcaramp2)
        saveext.append('pca')
        functype.append('ramp')
    elif model[i] == 'expramp':
        #DEFINE INDICES
        ind.regoal  = ind.size
        ind.rem     = ind.size + 1
        ind.rea     = ind.size + 2
        ind.size   += 3
        #DEFINE NAMES
        parname.insert(ind.regoal, 'Ramp, Ideal Flux')
        parname.insert(ind.rem,    'Ramp, Curvature')
        parname.insert(ind.rea,    'Ramp, Exp. Term')
        #DEFINE RAMP MODEL
        myfuncs.append(mc.expramp)
        saveext.append('er')
        functype.append('ramp')
    # FINDME ccampo added 2010-08-20
    elif model[i] == 'not0risingexp':
        #DEFINE INDICES
        ind.regoal  = ind.size
        ind.rem     = ind.size + 1
        ind.rea     = ind.size + 2
        ind.reb     = ind.size + 3
        ind.rec     = ind.size + 4
        ind.size   += 5
        #DEFINE NAMES
        parname.insert(ind.regoal, 'Ramp, Ideal Flux')
        parname.insert(ind.rem,    'Ramp, Curvature')
        parname.insert(ind.rea,    'Ramp, Offset Goal')
        parname.insert(ind.reb,    'Ramp, Offset Curvature')
        parname.insert(ind.rec,    'Ramp, Offset Offset')
        #DEFINE RAMP MODEL
        myfuncs.append(not0risingexp)
        saveext.append('not0re')
        functype.append('ramp')
    elif model[i] == 're2ramp':
        #DEFINE INDICES
        ind.re2goal  = ind.size
        ind.re2a     = ind.size + 1
        ind.re2m1    = ind.size + 2
        ind.re2t1    = ind.size + 3
        ind.re2b     = ind.size + 4
        ind.re2m2    = ind.size + 5
        ind.re2t2    = ind.size + 6
        ind.size    += 7
        #DEFINE NAMES
        parname.insert(ind.re2goal, 'Ramp, Ideal Flux')
        parname.insert(ind.re2a,    'Ramp1, Exp. Term')
        parname.insert(ind.re2m1,   'Ramp1, Curvature')
        parname.insert(ind.re2t1,   'Ramp1, Phase Offset')
        parname.insert(ind.re2b,    'Ramp2, Exp. Term')
        parname.insert(ind.re2m2,   'Ramp2, Curvature')
        parname.insert(ind.re2t2,   'Ramp2, Phase Offset')
        #DEFINE RAMP MODEL
        myfuncs.append(mc.re2ramp)
        saveext.append('re2')
        functype.append('ramp')
    elif model[i] == 'reqramp':
        #DEFINE INDICES
        ind.reqgoal  = ind.size
        ind.reqm     = ind.size + 1
        ind.reqt0    = ind.size + 2
        ind.reqa     = ind.size + 3
        ind.reqb     = ind.size + 4
        ind.reqc     = ind.size + 5
        ind.reqt1    = ind.size + 6
        ind.size   += 7
        #DEFINE NAMES
        parname.insert(ind.reqgoal, 'Ramp, Ideal Flux')
        parname.insert(ind.reqm,    'Ramp, Curvature')
        parname.insert(ind.reqt0,   'Ramp, Exp. Phase Offset')
        parname.insert(ind.reqa,    'Ramp, Quadratic Term')
        parname.insert(ind.reqb,    'Ramp, Linear Term')
        parname.insert(ind.reqc,    'Ramp, Constant Term')
        parname.insert(ind.reqt1,   'Ramp, Poly. Phase Offset')
        #DEFINE RAMP MODEL
        myfuncs.append(mc.reqramp)
        saveext.append('req')
        functype.append('ramp')
    elif model[i] == 'relramp':
        #DEFINE INDICES
        ind.relgoal  = ind.size
        ind.relm     = ind.size + 1
        ind.relt0    = ind.size + 2
        ind.rela     = ind.size + 3
        ind.relb     = ind.size + 4
        ind.relt1    = ind.size + 5
        ind.size   += 6
        #DEFINE NAMES
        parname.insert(ind.relgoal, 'Ramp, Ideal Flux')
        parname.insert(ind.relm,    'Ramp, Curvature')
        parname.insert(ind.relt0,   'Ramp, Exp. Phase Offset')
        parname.insert(ind.rela,    'Ramp, Linear Term')
        parname.insert(ind.relb,    'Ramp, Constant Term')
        parname.insert(ind.relt1,   'Ramp, Poly. Phase Offset')
        #DEFINE RAMP MODEL
        myfuncs.append(mc.relramp)
        saveext.append('rel')
        functype.append('ramp')
    elif model[i] == 'fallingexp':
        #DEFINE INDICES
        ind.fegoal  = ind.size
        ind.fem     = ind.size + 1
        ind.fet0    = ind.size + 2
        ind.size   += 3
        #DEFINE NAMES
        parname.insert(ind.fegoal, 'Ramp, Ideal Flux')
        parname.insert(ind.fem,    'Ramp, Curvature')
        parname.insert(ind.fet0,   'Ramp, Phase Offset')
        #DEFINE RAMP MODEL
        myfuncs.append(mc.fallingexp)
        saveext.append('fe')
        functype.append('ramp')
    elif model[i] == 'felramp':
        #DEFINE INDICES
        ind.felgoal  = ind.size
        ind.felm     = ind.size + 1
        ind.felt0    = ind.size + 2
        ind.fela     = ind.size + 3
        ind.felt1    = ind.size + 4
        ind.size   += 5
        #DEFINE NAMES
        parname.insert(ind.felgoal, 'Ramp, Ideal Flux')
        parname.insert(ind.felm,    'Ramp, Curvature')
        parname.insert(ind.felt0,   'Ramp, Exp. Phase Offset')
        parname.insert(ind.fela,    'Ramp, Linear Term')
        parname.insert(ind.felt1,   'Ramp, Poly. Phase Offset')
        #DEFINE RAMP MODEL
        myfuncs.append(mc.felramp)
        saveext.append('fel')
        functype.append('ramp')
    elif model[i] == 'linramp':
        #DEFINE INDICES
        ind.linr2    = ind.size
        ind.linc     = ind.size + 1
        ind.lint0    = ind.size + 2
        ind.size    += 3
        #DEFINE NAMES
        parname.insert(ind.linr2,   'Ramp, Linear Term')
        parname.insert(ind.linc,    'Ramp, Constant Term')
        parname.insert(ind.lint0,   'Ramp, Phase Offset')
           #DEFINE RAMP MODEL
        myfuncs.append(mc.linramp)
        saveext.append('ln')
        functype.append('ramp')
    elif model[i] == 'quadramp':
        #DEFINE INDICES
        ind.qrr3    = ind.size
        ind.qrr2    = ind.size + 1
        ind.qrc     = ind.size + 2
        ind.qrt0    = ind.size + 3
        ind.size   += 4
        #DEFINE NAMES
        parname.insert(ind.qrr3,   'Ramp, Quadratic Term')
        parname.insert(ind.qrr2,   'Ramp, Linear Term')
        parname.insert(ind.qrc,    'Ramp, Constant Term')
        parname.insert(ind.qrt0,   'Ramp, Phase Offset')
           #DEFINE RAMP MODEL
        myfuncs.append(mc.quadramp)
        saveext.append('qd')
        functype.append('ramp')
    elif model[i] == 'llramp':
        #DEFINE INDICES
        ind.llt0    = ind.size
        ind.llr6    = ind.size + 1
        ind.llr2    = ind.size + 2
        ind.llc     = ind.size + 3
        ind.llt1    = ind.size + 4
        ind.size   += 5
        #DEFINE NAMES
        parname.insert(ind.llt0,   'Ramp, Log Phase Offset')
        parname.insert(ind.llr6,   'Ramp, Log Term')
        parname.insert(ind.llr2,   'Ramp, Linear Term')
        parname.insert(ind.llc,    'Ramp, Constant Term')
        parname.insert(ind.llt1,   'Ramp, Poly. Phase Offset')
           #DEFINE RAMP MODEL
        myfuncs.append(mc.llramp)
        saveext.append('ll')
        functype.append('ramp')
    elif model[i] == 'lqramp':
        #DEFINE INDICES
        ind.lqt0    = ind.size
        ind.lqr6    = ind.size + 1
        ind.lqr7    = ind.size + 2
        ind.lqr2    = ind.size + 3
        ind.lqc     = ind.size + 4
        ind.lqt1    = ind.size + 5
        ind.size   += 6
        #DEFINE NAMES
        parname.insert(ind.lqt0,   'Ramp, Log Phase Offset')
        parname.insert(ind.lqr6,   'Ramp, Log Term')
        parname.insert(ind.lqr7,   'Ramp, Quadratic Term')
        parname.insert(ind.lqr2,   'Ramp, Linear Term')
        parname.insert(ind.lqc,    'Ramp, Constant Term')
        parname.insert(ind.lqt1,   'Ramp, Poly. Phase Offset')
           #DEFINE RAMP MODEL
        myfuncs.append(mc.lqramp)
        saveext.append('lq')
        functype.append('ramp')
    elif model[i] == 'logramp':
        #DEFINE INDICES
        ind.logt0    = ind.size
        ind.logr9    = ind.size + 1
        ind.logr8    = ind.size + 2
        ind.logr7    = ind.size + 3
        ind.logr6    = ind.size + 4
        ind.logc     = ind.size + 5
        ind.size    += 6
        #DEFINE NAMES
        parname.insert(ind.logt0,   'Ramp, Phase Offset')
        parname.insert(ind.logr9,   'Ramp, Quartic Term')
        parname.insert(ind.logr8,   'Ramp, Cubic Term')
        parname.insert(ind.logr7,   'Ramp, Quadratic Term')
        parname.insert(ind.logr6,   'Ramp, Linear Term')
        parname.insert(ind.logc,    'Ramp, Constant Term')
           #DEFINE RAMP MODEL
        myfuncs.append(mc.logramp)
        saveext.append('lg')
        functype.append('ramp')
    elif model[i] == 'log4qramp':
        #DEFINE INDICES
        ind.l4qt0      = ind.size
        ind.l4qr9      = ind.size + 1
        ind.l4qr8      = ind.size + 2
        ind.l4qr7      = ind.size + 3
        ind.l4qr6      = ind.size + 4
        ind.l4qr3      = ind.size + 5
        ind.l4qr2      = ind.size + 6
        ind.l4qc       = ind.size + 7
        ind.l4qt1      = ind.size + 8
        ind.size    += 9
        #DEFINE NAMES
        parname.insert(ind.l4qt0,     'Ramp, Log Phase Offset')
        parname.insert(ind.l4qr9,     'Ramp, Quartic Log')
        parname.insert(ind.l4qr8,     'Ramp, Cubic Log')
        parname.insert(ind.l4qr7,     'Ramp, Quadratic Log')
        parname.insert(ind.l4qr6,     'Ramp, Linear Log')
        parname.insert(ind.l4qr3,     'Ramp, Quadratic Term')
        parname.insert(ind.l4qr2,     'Ramp, Linear Term')
        parname.insert(ind.l4qc,      'Ramp, Constant Term')
        parname.insert(ind.l4qt1,     'Ramp, Poly. Phase Offset')
           #DEFINE RAMP MODEL
        myfuncs.append(mc.log4qramp)
        saveext.append('l4q')
        functype.append('ramp')
    elif model[i] == 'sindecay':
        #DEFINE INDICES
        ind.sdx0    = ind.size
        ind.sda     = ind.size + 1
        ind.sdb     = ind.size + 2
        ind.sdc     = ind.size + 3
        ind.sdd     = ind.size + 4
        ind.size   += 5
        #DEFINE NAMES
        parname.insert(ind.sdx0,   'Decay, Phase Offset')
        parname.insert(ind.sda,    'Decay, Amplitude')
        parname.insert(ind.sdb,    'Decay, Exp Term')
        parname.insert(ind.sdc,    'Decay, Period')
        parname.insert(ind.sdd,    'Decay, Constant Term')
           #DEFINE RAMP MODEL
        myfuncs.append(mc.sindecay)
        saveext.append('sd')
        functype.append('sinusoidal')
    elif model[i] == 'sincos':
        #DEFINE INDICES
        ind.sca     = ind.size
        ind.scp1    = ind.size + 1
        ind.sct1    = ind.size + 2
        ind.scb     = ind.size + 3
        ind.scp2    = ind.size + 4
        ind.sct2    = ind.size + 5
        ind.scc     = ind.size + 6
        ind.size   += 7
        #DEFINE NAMES
        parname.insert(ind.sca,   'Sine, Amplitude')
        parname.insert(ind.scp1,  'Sine, Period')
        parname.insert(ind.sct1,  'Sine, Phase Offset')
        parname.insert(ind.scb,   'Cosine, Amplitude')
        parname.insert(ind.scp2,  'Cosine, Period')
        parname.insert(ind.sct2,  'Cosine, Phase Offset')
        parname.insert(ind.scc,   'Sin/Cos, Constant Term')
           #DEFINE RAMP MODEL
        myfuncs.append(mc.sincos)
        saveext.append('sc')
        functype.append('sinusoidal')
    elif model[i] == 'sincos2':
        #DEFINE INDICES
        ind.sc2c1a  = ind.size
        ind.sc2c1o  = ind.size + 1
        ind.sc2c2a  = ind.size + 2
        ind.sc2c2o  = ind.size + 3
        ind.sc2s1a  = ind.size + 4
        ind.sc2s1o  = ind.size + 5
        ind.sc2s2a  = ind.size + 6
        ind.sc2s2o  = ind.size + 7
        ind.sc2p    = ind.size + 8
        ind.sc2c    = ind.size + 9
        ind.sc2midpt= ind.size + 10
        ind.sc2t14  = ind.size + 11
        ind.sc2t12  = ind.size + 12
        ind.size   += 13
        #DEFINE NAMES
        parname.insert(ind.sc2c1a,  'Cosine 1, Amplitude')
        parname.insert(ind.sc2c1o,  'Cosine 1, Phase Offset')
        parname.insert(ind.sc2c2a,  'Cosine 2, Amplitude')
        parname.insert(ind.sc2c2o,  'Cosine 2, Phase Offset')
        parname.insert(ind.sc2s1a,  'Sine 1, Amplitude')
        parname.insert(ind.sc2s1o,  'Sine 1, Phase Offset')
        parname.insert(ind.sc2s2a,  'Sine 1, Amplitude')
        parname.insert(ind.sc2s2o,  'Sine 1, Phase Offset')
        parname.insert(ind.sc2p,    'Sin/Cos, Period')
        parname.insert(ind.sc2c,    'Sin/Cos, Constant Term')
        parname.insert(ind.sc2midpt,'Sin/Cos, Ecl. Midpt')
        parname.insert(ind.sc2t14,  'Sin/Cos, Ecl. Width')
        parname.insert(ind.sc2t12,  'Sin/Cos, Ecl. Ingress')
           #DEFINE RAMP MODEL
        myfuncs.append(mc.sincos2)
        saveext.append('sc2')
        functype.append('sinusoidal')
    elif model[i] == 'cosine8':
        #DEFINE INDICES
        ind.c8a1    = ind.size
        ind.c8p1    = ind.size + 1
        ind.c8t1    = ind.size + 2
        ind.c8a2    = ind.size + 3
        ind.c8p2    = ind.size + 4
        ind.c8t2    = ind.size + 5
        ind.c8a3    = ind.size + 6
        ind.c8p3    = ind.size + 7
        ind.c8t3    = ind.size + 8
        ind.c8a4    = ind.size + 9
        ind.c8p4    = ind.size + 10
        ind.c8t4    = ind.size + 11
        ind.c8a5    = ind.size + 12
        ind.c8p5    = ind.size + 13
        ind.c8t5    = ind.size + 14
        ind.c8a6    = ind.size + 15
        ind.c8p6    = ind.size + 16
        ind.c8t6    = ind.size + 17
        ind.c8a7    = ind.size + 18
        ind.c8p7    = ind.size + 19
        ind.c8t7    = ind.size + 20
        ind.c8a8    = ind.size + 21
        ind.c8p8    = ind.size + 22
        ind.c8t8    = ind.size + 23
        ind.c8c     = ind.size + 24
        ind.size   += 25
        #DEFINE NAMES
        parname.insert(ind.c8a1,  'Cosine, Amplitude 1')
        parname.insert(ind.c8p1,  'Cosine, Period 1')
        parname.insert(ind.c8t1,  'Cosine, Phase Offset 1')
        parname.insert(ind.c8a2,  'Cosine, Amplitude 2')
        parname.insert(ind.c8p2,  'Cosine, Period 2')
        parname.insert(ind.c8t2,  'Cosine, Phase Offset 2')
        parname.insert(ind.c8a3,  'Cosine, Amplitude 3')
        parname.insert(ind.c8p3,  'Cosine, Period 3')
        parname.insert(ind.c8t3,  'Cosine, Phase Offset 3')
        parname.insert(ind.c8a4,  'Cosine, Amplitude 4')
        parname.insert(ind.c8p4,  'Cosine, Period 4')
        parname.insert(ind.c8t4,  'Cosine, Phase Offset 4')
        parname.insert(ind.c8a5,  'Cosine, Amplitude 5')
        parname.insert(ind.c8p5,  'Cosine, Period 5')
        parname.insert(ind.c8t5,  'Cosine, Phase Offset 5')
        parname.insert(ind.c8a6,  'Cosine, Amplitude 6')
        parname.insert(ind.c8p6,  'Cosine, Period 6')
        parname.insert(ind.c8t6,  'Cosine, Phase Offset 6')
        parname.insert(ind.c8a7,  'Cosine, Amplitude 7')
        parname.insert(ind.c8p7,  'Cosine, Period 7')
        parname.insert(ind.c8t7,  'Cosine, Phase Offset 7')
        parname.insert(ind.c8a8,  'Cosine, Amplitude 8')
        parname.insert(ind.c8p8,  'Cosine, Period 8')
        parname.insert(ind.c8t8,  'Cosine, Phase Offset 8')
        parname.insert(ind.c8c,   'Cosine, Constant Term')
           #DEFINE RAMP MODEL
        myfuncs.append(mc.cosine8)
        saveext.append('c8')
        functype.append('sinusoidal')
    elif model[i] == 'gp_exp2':
        #DEFINE INDICES
        ind.gpexp2a  = ind.size
        ind.gpexp2b  = ind.size + 1
        ind.gpexp2n  = ind.size + 2
        ind.size    += 3
        #DEFINE NAMES
        parname.insert(ind.gpexp2a,    'GP, Max Amplitude')
        parname.insert(ind.gpexp2b,    'GP, Scale Length')
        parname.insert(ind.gpexp2n,    'GP, Number of Samples')
           #DEFINE RAMP MODEL
        myfuncs.append(mc.gp_exp2)
        saveext.append('gpexp2')
        functype.append('gp')
    elif model[i] == 'rotation':
        #DEFINE INDICES
        ind.rota     = ind.size
        ind.rotb     = ind.size + 1
        ind.roto     = ind.size + 2
        ind.size    += 3
        #DEFINE NAMES
        parname.insert(ind.rota,    'Rotation, Airmass Multiplier')
        parname.insert(ind.rotb,    'Rotation, Cosine Multiplier')
        parname.insert(ind.roto,    'Rotation, Angular Offset')
           #DEFINE RAMP MODEL
        myfuncs.append(mc.rotation)
        saveext.append('rot')
        functype.append('instrument')
    elif model[i] == 'humidity':
        #DEFINE INDICES
        ind.rha      = ind.size
        ind.rhb      = ind.size + 1
        ind.size    += 2
        #DEFINE NAMES
        parname.insert(ind.rha,    'Humidity, Offset')
        parname.insert(ind.rhb,    'Humidity, Multiplier')
           #DEFINE RAMP MODEL
        myfuncs.append(mc.humidity)
        saveext.append('rh')
        functype.append('instrument')
    elif model[i] == 'quadip':
        #DEFINE INDICES
        ind.qy2    = ind.size
        ind.qx2    = ind.size + 1
        ind.qxy    = ind.size + 2
        ind.qy     = ind.size + 3
        ind.qx     = ind.size + 4
        ind.qipc   = ind.size + 5
        ind.size  += 6
        #DEFINE NAMES
        parname.insert(ind.qy2,    'Intra-pixel, Quadratic Term in y')
        parname.insert(ind.qx2,    'Intra-pixel, Quadratic Term in x')
        parname.insert(ind.qxy,    'Intra-pixel, Cross Term')
        parname.insert(ind.qy,     'Intra-pixel, Linear Term in y')
        parname.insert(ind.qx,     'Intra-pixel, Linear Term in x')
        parname.insert(ind.qipc,   'Intra-pixel, Constant Term')
        #DEFINE INTRA-PIXEL MODEL
        myfuncs.append(mc.quadip)
        saveext.append('qip')
        functype.append('ippoly')
    elif model[i] == 'quadip4':
        #DEFINE INDICES
        ind.q0y2   = ind.size
        ind.q0x2   = ind.size + 1
        ind.q0xy   = ind.size + 2
        ind.q0y    = ind.size + 3
        ind.q0x    = ind.size + 4
        ind.q0c    = ind.size + 5
        ind.q1y2   = ind.size + 6
        ind.q1x2   = ind.size + 7
        ind.q1xy   = ind.size + 8
        ind.q1y    = ind.size + 9
        ind.q1x    = ind.size + 10
        ind.q1c    = ind.size + 11
        ind.q2y2   = ind.size + 12
        ind.q2x2   = ind.size + 13
        ind.q2xy   = ind.size + 14
        ind.q2y    = ind.size + 15
        ind.q2x    = ind.size + 16
        ind.q2c    = ind.size + 17
        ind.q3y2   = ind.size + 18
        ind.q3x2   = ind.size + 19
        ind.q3xy   = ind.size + 20
        ind.q3y    = ind.size + 21
        ind.q3x    = ind.size + 22
        ind.q3c    = ind.size + 23
        ind.size  += 24
        #DEFINE NAMES
        parname.insert(ind.q0y2,    'Intra-pixel, Q0, Quad Term in y')
        parname.insert(ind.q0x2,    'Intra-pixel, Q0, Quad Term in x')
        parname.insert(ind.q0xy,    'Intra-pixel, Q0, Cross Term')
        parname.insert(ind.q0y,     'Intra-pixel, Q0, Linear Term in y')
        parname.insert(ind.q0x,     'Intra-pixel, Q0, Linear Term in x')
        parname.insert(ind.q0c,     'Intra-pixel, Q0, Constant Term')
        parname.insert(ind.q1y2,    'Intra-pixel, Q1, Quad Term in y')
        parname.insert(ind.q1x2,    'Intra-pixel, Q1, Quad Term in x')
        parname.insert(ind.q1xy,    'Intra-pixel, Q1, Cross Term')
        parname.insert(ind.q1y,     'Intra-pixel, Q1, Linear Term in y')
        parname.insert(ind.q1x,     'Intra-pixel, Q1, Linear Term in x')
        parname.insert(ind.q1c,     'Intra-pixel, Q1, Constant Term')
        parname.insert(ind.q2y2,    'Intra-pixel, Q2, Quad Term in y')
        parname.insert(ind.q2x2,    'Intra-pixel, Q2, Quad Term in x')
        parname.insert(ind.q2xy,    'Intra-pixel, Q2, Cross Term')
        parname.insert(ind.q2y,     'Intra-pixel, Q2, Linear Term in y')
        parname.insert(ind.q2x,     'Intra-pixel, Q2, Linear Term in x')
        parname.insert(ind.q2c,     'Intra-pixel, Q2, Constant Term')
        parname.insert(ind.q3y2,    'Intra-pixel, Q3, Quad Term in y')
        parname.insert(ind.q3x2,    'Intra-pixel, Q3, Quad Term in x')
        parname.insert(ind.q3xy,    'Intra-pixel, Q3, Cross Term')
        parname.insert(ind.q3y,     'Intra-pixel, Q3, Linear Term in y')
        parname.insert(ind.q3x,     'Intra-pixel, Q3, Linear Term in x')
        parname.insert(ind.q3c,     'Intra-pixel, Q3, Constant Term')
        #DEFINE INTRA-PIXEL MODEL
        myfuncs.append(mc.quadip4)
        saveext.append('qip4')
        functype.append('ippoly')
    elif model[i] == 'cubicip':
        #DEFINE INDICES
        ind.cy3    = ind.size
        ind.cx3    = ind.size + 1
        ind.cy2x   = ind.size + 2
        ind.cyx2   = ind.size + 3
        ind.cy2    = ind.size + 4
        ind.cy2    = ind.size + 5
        ind.cx2    = ind.size + 6
        ind.cxy    = ind.size + 7
        ind.cy     = ind.size + 8
        ind.cx     = ind.size + 9
        ind.cipc   = ind.size + 10
        ind.size  += 11
        #DEFINE NAMES
        parname.insert(ind.cy3,    'Intra-pixel, Cubic Term in y')
        parname.insert(ind.cx3,    'Intra-pixel, Cubic Term in x')
        parname.insert(ind.cy2x,   'Intra-pixel, y^2*x Cross Term')
        parname.insert(ind.cyx2,   'Intra-pixel, y*x^2 Cross Term')
        parname.insert(ind.cy2,    'Intra-pixel, Quadratic Term in y')
        parname.insert(ind.cx2,    'Intra-pixel, Quadratic Term in x')
        parname.insert(ind.cxy,    'Intra-pixel, y*x Cross Term')
        parname.insert(ind.cy,     'Intra-pixel, Linear Term in y')
        parname.insert(ind.cx,     'Intra-pixel, Linear Term in x')
        parname.insert(ind.cipc,   'Intra-pixel, Constant Term')
        #DEFINE INTRA-PIXEL MODEL
        myfuncs.append(mc.cubicip)
        saveext.append('cip')
        functype.append('ippoly')
    elif model[i] == 'sexticip':
        #DEFINE INDICES
        ind.sy6    = ind.size
        ind.sx6    = ind.size + 1
        ind.sy5    = ind.size + 2
        ind.sx5    = ind.size + 3
        ind.sy4    = ind.size + 4
        ind.sx4    = ind.size + 5
        ind.sy3    = ind.size + 6
        ind.sx3    = ind.size + 7
        ind.sy2    = ind.size + 8
        ind.sx2    = ind.size + 9
        ind.sy     = ind.size + 10
        ind.sx     = ind.size + 11
        ind.sipc   = ind.size + 12
        ind.size  += 13
        #DEFINE NAMES
        parname.insert(ind.sy6,    'Intra-pixel, Sextic Term in y')
        parname.insert(ind.sx6,    'Intra-pixel, Sextic Term in x')
        parname.insert(ind.sy5,    'Intra-pixel, Quintic Term in y')
        parname.insert(ind.sx5,    'Intra-pixel, Quintic Term in x')
        parname.insert(ind.sy4,    'Intra-pixel, Quartic Term in y')
        parname.insert(ind.sx4,    'Intra-pixel, Quartic Term in x')
        parname.insert(ind.sy3,    'Intra-pixel, Cubic Term in y')
        parname.insert(ind.sx3,    'Intra-pixel, Cubic Term in x')
        parname.insert(ind.sy2,    'Intra-pixel, Quadratic Term in y')
        parname.insert(ind.sx2,    'Intra-pixel, Quadratic Term in x')
        parname.insert(ind.sy,     'Intra-pixel, Linear Term in y')
        parname.insert(ind.sx,     'Intra-pixel, Linear Term in x')
        parname.insert(ind.sipc,   'Intra-pixel, Constant Term')
        #DEFINE INTRA-PIXEL MODEL
        myfuncs.append(sexticip)
        saveext.append('6ip')
        functype.append('ippoly')
    elif model[i] == 'sexticipc':
        #DEFINE INDICES
        ind.sy6    = ind.size
        ind.sx6    = ind.size + 1
        ind.sy5    = ind.size + 2
        ind.sx5    = ind.size + 3
        ind.sy4    = ind.size + 4
        ind.sx4    = ind.size + 5
        ind.sy3    = ind.size + 6
        ind.sx3    = ind.size + 7
        ind.sy2x   = ind.size + 8
        ind.sx2y   = ind.size + 9
        ind.sy2    = ind.size + 10
        ind.sx2    = ind.size + 11
        ind.sxy    = ind.size + 12
        ind.sy     = ind.size + 13
        ind.sx     = ind.size + 14
        ind.sipc   = ind.size + 15
        ind.size  += 16
        #DEFINE NAMES
        parname.insert(ind.sy6,    'Intra-pixel, Sextic Term in y')
        parname.insert(ind.sx6,    'Intra-pixel, Sextic Term in x')
        parname.insert(ind.sy5,    'Intra-pixel, Quintic Term in y')
        parname.insert(ind.sx5,    'Intra-pixel, Quintic Term in x')
        parname.insert(ind.sy4,    'Intra-pixel, Quartic Term in y')
        parname.insert(ind.sx4,    'Intra-pixel, Quartic Term in x')
        parname.insert(ind.sy3,    'Intra-pixel, Cubic Term in y')
        parname.insert(ind.sx3,    'Intra-pixel, Cubic Term in x')
        parname.insert(ind.sy2x,   'Intra-pixel, y^2*x Cross Term')
        parname.insert(ind.sx2y,   'Intra-pixel, y*x^2 Cross Term')
        parname.insert(ind.sy2,    'Intra-pixel, Quadratic Term in y')
        parname.insert(ind.sx2,    'Intra-pixel, Quadratic Term in x')
        parname.insert(ind.sxy,    'Intra-pixel, y*x Cross Term')
        parname.insert(ind.sy,     'Intra-pixel, Linear Term in y')
        parname.insert(ind.sx,     'Intra-pixel, Linear Term in x')
        parname.insert(ind.sipc,   'Intra-pixel, Constant Term')
        #DEFINE INTRA-PIXEL MODEL
        myfuncs.append(mc.sexticipc)
        saveext.append('6ipc')
        functype.append('ippoly')
    elif model[i] == 'cubicgw':
        #DEFINE INDICES
        ind.tx1    = ind.size
        ind.tx2    = ind.size + 1
        ind.tx3    = ind.size + 2
        ind.ty1    = ind.size + 3
        ind.ty2    = ind.size + 4
        ind.ty3    = ind.size + 5
        ind.tc     = ind.size + 6
        ind.size  += 7
        #DEFINE NAMES
        parname.insert(ind.tx1,    'Intra-pixel, Linear Term in x')
        parname.insert(ind.tx2,    'Intra-pixel, Quadratic Term in x')
        parname.insert(ind.tx3,    'Intra-pixel, Cubic Term in x')
        parname.insert(ind.ty1,    'Intra-pixel, Linear Term in y')
        parname.insert(ind.ty2,    'Intra-pixel, Quadratic Term in y')
        parname.insert(ind.ty3,    'Intra-pixel, Cubic Term in y')
        parname.insert(ind.tc,    'Intra-pixel, Constant Term')
        #DEFINE INTRA-PIXEL MODEL
        myfuncs.append(mc.cubicgw)
        saveext.append('3gw')
        functype.append('ippoly')
    elif model[i] == 'ballardip':
        #DEFINE INDICES
        ind.sigmay    = ind.size
        ind.sigmax    = ind.size + 1
        ind.nbins     = ind.size + 2
        ind.size     += 3
        #DEFINE NAMES
        parname.insert(ind.sigmay,    'Intra-pixel, Sigma y')
        parname.insert(ind.sigmax,    'Intra-pixel, Sigma x')
        parname.insert(ind.nbins,     'Intra-pixel, # of Bins')
        #DEFINE INTRA-PIXEL MODEL
        myfuncs.append(mc.ballardip)
        saveext.append('bip')
        functype.append('ballardip')
    elif model[i] == 'medianip':
        #DEFINE INDICES
        ind.rad    = ind.size
        ind.size  += 1
        #DEFINE NAMES
        parname.insert(ind.rad,    'Intra-pixel, Radius')
        #DEFINE INTRA-PIXEL MODEL
        myfuncs.append(medianip)
        saveext.append('mip')
        functype.append('ipmap')
    elif model[i] == 'nnint':
        #DEFINE INDICES
        ind.nnip   = ind.size
        ind.size  += 1
        #DEFINE NAMES
        parname.insert(ind.nnip,   'Interpolation, min # pts')
        #DEFINE INTRA-PIXEL MODEL
        myfuncs.append(mc.nnint)
        saveext.append('nni')
        functype.append('ipmap')
    elif model[i] == 'bilinint':
        #DEFINE INDICES
        ind.blip   = ind.size
        ind.size  += 1
        #DEFINE NAMES
        parname.insert(ind.blip,   'Interpolation, min # pts')
        #DEFINE INTRA-PIXEL MODEL
        myfuncs.append(mc.bilinint)
        saveext.append('bli')
        functype.append('ipmap')
    elif model[i] == 'ipspline':
        #DEFINE INDICES
        ind.ipsx0   = ind.size
        ind.ipsy0   = ind.size + 1
        ind.size   += 2
        #DEFINE NAMES
        parname.insert(ind.ipsx0,    'Intra-pixel, Knot x0')
        parname.insert(ind.ipsy0,    'Intra-pixel, Knot y0')
        #DEFINE INTRA-PIXEL MODEL
        myfuncs.append(mc.ipspline)
        saveext.append('ipspl')
        functype.append('ippoly')
    elif model[i] == 'unispline':
        #DEFINE INDICES
        ind.usnk   = ind.size
        ind.usk    = ind.size + 1
        ind.size  += 2
        #DEFINE NAMES
        parname.insert(ind.usnk,   '# of Knots')
        parname.insert(ind.usk,    'Degree Poll')
        #DEFINE INTRA-PIXEL MODEL
        myfuncs.append(mc.unispline)
        saveext.append('uspl')
        functype.append('spline')
    elif model[i] == 'posflux':
        #DEFINE INDICES
        ind.p0    = ind.size
        ind.p1    = ind.size + 1
        ind.p2    = ind.size + 2
        ind.p3    = ind.size + 3
        ind.p4    = ind.size + 4
        ind.p5    = ind.size + 5
        ind.p6    = ind.size + 6
        ind.p7    = ind.size + 7
        ind.p8    = ind.size + 8
        ind.size  += 9
        #DEFINE NAMES
        parname.insert(ind.p0,    'Position 0')
        parname.insert(ind.p1,    'Position 1')
        parname.insert(ind.p2,    'Position 2')
        parname.insert(ind.p3,    'Position 3')
        parname.insert(ind.p4,    'Position 4')
        parname.insert(ind.p5,    'Position 5')
        parname.insert(ind.p6,    'Position 6')
        parname.insert(ind.p7,    'Position 7')
        parname.insert(ind.p8,    'Position 8')
        #DEFINE INTRA-PIXEL MODEL
        myfuncs.append(mc.posflux)
        saveext.append('pf')
        functype.append('posoffset')
    elif model[i] == 'posflux2':
        #DEFINE INDICES
        ind.p0    = ind.size
        ind.p1    = ind.size + 1
        ind.p2    = ind.size + 2
        ind.p3    = ind.size + 3
        ind.p4    = ind.size + 4
        ind.p5    = ind.size + 5
        ind.p6    = ind.size + 6
        ind.p7    = ind.size + 7
        ind.p8    = ind.size + 8
        ind.size  += 9
        #DEFINE NAMES
        parname.insert(ind.p0,    'Position 0')
        parname.insert(ind.p1,    'Position 1')
        parname.insert(ind.p2,    'Position 2')
        parname.insert(ind.p3,    'Position 3')
        parname.insert(ind.p4,    'Position 4')
        parname.insert(ind.p5,    'Position 5')
        parname.insert(ind.p6,    'Position 6')
        parname.insert(ind.p7,    'Position 7')
        parname.insert(ind.p8,    'Position 8')
        #DEFINE INTRA-PIXEL MODEL
        myfuncs.append(mc.posflux2)
        saveext.append('pf2')
        functype.append('posoffset')
    elif model[i] == 'posfluxlinip':
        #DEFINE INDICES
        ind.p0    = ind.size
        ind.p1    = ind.size + 1
        ind.p2    = ind.size + 2
        ind.p3    = ind.size + 3
        ind.p4    = ind.size + 4
        ind.p5    = ind.size + 5
        ind.p6    = ind.size + 6
        ind.p7    = ind.size + 7
        ind.p8    = ind.size + 8
        ind.y0    = ind.size + 9
        ind.x0    = ind.size + 10
        ind.y1    = ind.size + 11
        ind.x1    = ind.size + 12
        ind.y2    = ind.size + 13
        ind.x2    = ind.size + 14
        ind.y3    = ind.size + 15
        ind.x3    = ind.size + 16
        ind.y4    = ind.size + 17
        ind.x4    = ind.size + 18
        ind.y5    = ind.size + 19
        ind.x5    = ind.size + 20
        ind.y6    = ind.size + 21
        ind.x6    = ind.size + 22
        ind.y7    = ind.size + 23
        ind.x7    = ind.size + 24
        ind.y8    = ind.size + 25
        ind.x8    = ind.size + 26
        ind.size  += 27
        #DEFINE NAMES
        parname.insert(ind.p0,    'Position 0')
        parname.insert(ind.p1,    'Position 1')
        parname.insert(ind.p2,    'Position 2')
        parname.insert(ind.p3,    'Position 3')
        parname.insert(ind.p4,    'Position 4')
        parname.insert(ind.p5,    'Position 5')
        parname.insert(ind.p6,    'Position 6')
        parname.insert(ind.p7,    'Position 7')
        parname.insert(ind.p8,    'Position 8')
        parname.insert(ind.y0,    'Intra-pixel, y0 Linear Term')
        parname.insert(ind.x0,    'Intra-pixel, x0 Linear Term')
        parname.insert(ind.y1,    'Intra-pixel, y1 Linear Term')
        parname.insert(ind.x1,    'Intra-pixel, x1 Linear Term')
        parname.insert(ind.y2,    'Intra-pixel, y2 Linear Term')
        parname.insert(ind.x2,    'Intra-pixel, x2 Linear Term')
        parname.insert(ind.y3,    'Intra-pixel, y3 Linear Term')
        parname.insert(ind.x3,    'Intra-pixel, x3 Linear Term')
        parname.insert(ind.y4,    'Intra-pixel, y4 Linear Term')
        parname.insert(ind.x4,    'Intra-pixel, x4 Linear Term')
        parname.insert(ind.y5,    'Intra-pixel, y5 Linear Term')
        parname.insert(ind.x5,    'Intra-pixel, x5 Linear Term')
        parname.insert(ind.y6,    'Intra-pixel, y6 Linear Term')
        parname.insert(ind.x6,    'Intra-pixel, x6 Linear Term')
        parname.insert(ind.y7,    'Intra-pixel, y7 Linear Term')
        parname.insert(ind.x7,    'Intra-pixel, x7 Linear Term')
        parname.insert(ind.y8,    'Intra-pixel, y8 Linear Term')
        parname.insert(ind.x8,    'Intra-pixel, x8 Linear Term')
        #DEFINE INTRA-PIXEL MODEL
        myfuncs.append(mc.posfluxlinip)
        saveext.append('pflip')
        functype.append('ippoly')
    elif model[i] == 'linip':
        #DEFINE INDICES
        ind.y0    = ind.size
        ind.x0    = ind.size + 1
        ind.y1    = ind.size + 2
        ind.x1    = ind.size + 3
        ind.y2    = ind.size + 4
        ind.x2    = ind.size + 5
        ind.y3    = ind.size + 6
        ind.x3    = ind.size + 7
        ind.y4    = ind.size + 8
        ind.x4    = ind.size + 9
        ind.y5    = ind.size + 10
        ind.x5    = ind.size + 11
        ind.y6    = ind.size + 12
        ind.x6    = ind.size + 13
        ind.y7    = ind.size + 14
        ind.x7    = ind.size + 15
        ind.y8    = ind.size + 16
        ind.x8    = ind.size + 17
        ind.size  += 18
        #DEFINE NAMES
        parname.insert(ind.y0,    'Intra-pixel, y0 Linear Term')
        parname.insert(ind.x0,    'Intra-pixel, x0 Linear Term')
        parname.insert(ind.y1,    'Intra-pixel, y1 Linear Term')
        parname.insert(ind.x1,    'Intra-pixel, x1 Linear Term')
        parname.insert(ind.y2,    'Intra-pixel, y2 Linear Term')
        parname.insert(ind.x2,    'Intra-pixel, x2 Linear Term')
        parname.insert(ind.y3,    'Intra-pixel, y3 Linear Term')
        parname.insert(ind.x3,    'Intra-pixel, x3 Linear Term')
        parname.insert(ind.y4,    'Intra-pixel, y4 Linear Term')
        parname.insert(ind.x4,    'Intra-pixel, x4 Linear Term')
        parname.insert(ind.y5,    'Intra-pixel, y5 Linear Term')
        parname.insert(ind.x5,    'Intra-pixel, x5 Linear Term')
        parname.insert(ind.y6,    'Intra-pixel, y6 Linear Term')
        parname.insert(ind.x6,    'Intra-pixel, x6 Linear Term')
        parname.insert(ind.y7,    'Intra-pixel, y7 Linear Term')
        parname.insert(ind.x7,    'Intra-pixel, x7 Linear Term')
        parname.insert(ind.y8,    'Intra-pixel, y8 Linear Term')
        parname.insert(ind.x8,    'Intra-pixel, x8 Linear Term')
        #DEFINE INTRA-PIXEL MODEL
        myfuncs.append(mc.linip)
        saveext.append('lip')
        functype.append('ippoly')
    elif model[i] == 'vsll':
        #DEFINE INDICES
        ind.vsx0    = ind.size
        ind.vsa    = ind.size + 1
        ind.vsb    = ind.size + 2
        ind.vsc    = ind.size + 3
        ind.vsx1   = ind.size + 4
        ind.size  += 5
        #DEFINE NAMES
        parname.insert(ind.vsx0,   'Visit Sensitivity, Log Phase Offset')
        parname.insert(ind.vsa,    'Visit Sensitivity, Log Term')
        parname.insert(ind.vsb,    'Visit Sensitivity, Linear Term')
        parname.insert(ind.vsc,    'Visit Sensitivity, Constant Term')
        parname.insert(ind.vsx1,   'Visit Sensitivity, Poly. Phase Offset')
        #DEFINE INTRA-PIXEL MODEL
        myfuncs.append(mc.vsll)
        saveext.append('vsll')
        functype.append('vissen')
    elif model[i] == 'vsspline':
        #DEFINE INDICES
        ind.vss0    = ind.size
        ind.vss1    = ind.size + 1
        ind.vss2    = ind.size + 2
        ind.vss3    = ind.size + 3
        ind.vss4    = ind.size + 4
        ind.vss5    = ind.size + 5
        ind.vss6    = ind.size + 6
        ind.vss7    = ind.size + 7
        ind.vss8    = ind.size + 8
        ind.vss9    = ind.size + 9
        ind.vss10   = ind.size + 10
        ind.vss11   = ind.size + 11
        ind.size  += 12
        #DEFINE NAMES
        parname.insert(ind.vss0,   'Visit Sensitivity, Knot 0')
        parname.insert(ind.vss1,   'Visit Sensitivity, Knot 1')
        parname.insert(ind.vss2,   'Visit Sensitivity, Knot 2')
        parname.insert(ind.vss3,   'Visit Sensitivity, Knot 3')
        parname.insert(ind.vss4,   'Visit Sensitivity, Knot 4')
        parname.insert(ind.vss5,   'Visit Sensitivity, Knot 5')
        parname.insert(ind.vss6,   'Visit Sensitivity, Knot 6')
        parname.insert(ind.vss7,   'Visit Sensitivity, Knot 7')
        parname.insert(ind.vss8,   'Visit Sensitivity, Knot 8')
        parname.insert(ind.vss9,   'Visit Sensitivity, Knot 9')
        parname.insert(ind.vss10,  'Visit Sensitivity, Knot 10')
        parname.insert(ind.vss11,  'Visit Sensitivity, Knot 11')
        #DEFINE INTRA-PIXEL MODEL
        myfuncs.append(mc.vsspline)
        saveext.append('vss')
        functype.append('vissen')
    elif model[i] == 'flatfield3':
        #DEFINE INDICES
        ind.ff30    = ind.size
        ind.ff31    = ind.size + 1
        ind.ff32    = ind.size + 2
        ind.ff33    = ind.size + 3
        ind.ff34    = ind.size + 4
        ind.ff35    = ind.size + 5
        ind.ff36    = ind.size + 6
        ind.ff37    = ind.size + 7
        ind.ff38    = ind.size + 8
        ind.flux4   = ind.size + 9
        ind.size   += 10
        #DEFINE NAMES
        parname.insert(ind.ff30,   'Flat Field, Pixel 0')
        parname.insert(ind.ff31,   'Flat Field, Pixel 1')
        parname.insert(ind.ff32,   'Flat Field, Pixel 2')
        parname.insert(ind.ff33,   'Flat Field, Pixel 3')
        parname.insert(ind.ff34,   'Flat Field, Pixel 4')
        parname.insert(ind.ff35,   'Flat Field, Pixel 5')
        parname.insert(ind.ff36,   'Flat Field, Pixel 6')
        parname.insert(ind.ff37,   'Flat Field, Pixel 7')
        parname.insert(ind.ff38,   'Flat Field, Pixel 8')
        parname.insert(ind.flux4, r'System Flux ($\mu Jy$)')
        #DEFINE INTRA-PIXEL MODEL
        myfuncs.append(mc.flatfield3)
        saveext.append('ff3')
        functype.append('flatf')
    else:
        print("Error: " + str(model[i]) + " model not found.")
        myfuncs.append(-1)
        functype[i] = -1

   return myfuncs, functype, parname, ind, "".join(saveext)

def chisq(ymodel, y, sigma):
    return sum((ymodel - y)**2 / sigma**2)

'''
elif model[i] == 'pixlevdec':
    #DEFINE INDICES
    ind.pldc0   = ind.size
    ind.pldc1   = ind.size + 1
    ind.pldc2   = ind.size + 2
    ind.pldc3   = ind.size + 3
    ind.pldc4   = ind.size + 4
    ind.pldc5   = ind.size + 5
    ind.pldc6   = ind.size + 6
    ind.pldc7   = ind.size + 7
    ind.pldc8   = ind.size + 8
    ind.size   += 9
    #DEFINE NAMES
    parname.insert(ind.pldc0,   'PLD, c0')
    parname.insert(ind.pldc1,   'PLD, c1')
    parname.insert(ind.pldc2,   'PLD, c2')
    parname.insert(ind.pldc3,   'PLD, c3')
    parname.insert(ind.pldc4,   'PLD, c4')
    parname.insert(ind.pldc5,   'PLD, c5')
    parname.insert(ind.pldc6,   'PLD, c6')
    parname.insert(ind.pldc7,   'PLD, c7')
    parname.insert(ind.pldc8,   'PLD, c8')
    #DEFINE INTRA-PIXEL MODEL
    myfuncs.append(mc.pixlevdec)
    saveext.append('pld')
    functype.append('ippld')
'''

def mandel_geom(params, x, etc = []):
    """
  This function computes a transit shape using equations provided by Mandel & Agol (2002).

  Parameters
  ----------
    midpt:  Center of eclipse
    width:  Eclipse duration from contacts 1 to 4
    rp_rs:  Planet-star radius ratio
    b:        Impact parameter
    flux:   Stellar flux
    x:        Array of phase points

  Returns
  -------
    This function returns the flux for each point in x.

  Revisions
  ---------
  2009-12-10    Ryan A. Hardy, UCF  
            hardy.r@gmail.com
        Original version
    """
    midpt, width, rp_rs, b, flux = params
    ingress = orbit.limbtime(b, width, 1, rp_rs)[0]
    trpars = np.array([midpt, width, rp_rs**2, ingress, ingress, flux])
    return mandelecl(trpars, x)

def mandelecl_orbit(params, x, etc = []):
    """
  This function computes the transit shape using equations provided by Mandel & Agol (2002)

  Parameters
  ----------
    midpt:  Center of eclipse
    width:  Eclipse duration from contacts 1 to 4
    rp_rs:  Planet-star radius ratio
    b:        Impact parameter
    flux:   Stellar flux
    x:        Array of phase points

  Returns
  -------
    This function returns the flux for each point in x.

  Revisions
  ---------
  2009-12-10    Ryan A. Hardy, UCF      
            hardy.r@gmail.com
        Original version
    """
    G = 6.674e-11
    msun = 1.98892e30
    rsun = 6.955e8
    rjupiter = 71492000
    e, omega, i, period, rplanet, rstar, mstar, ecldepth, flux = params
    rplanet *= rjupiter
    rstar *= rsun
    mstar *= msun
    rp_rs = rplanet/rstar
    period*=86400
    a = (G*mstar*(period/(2*np.pi))**2)**(1/3.0)
    r = a*(1-e**2)/(1+e*np.cos(np.pi*(90-omega)/180.0))
    btr = r*np.cos(np.pi/180.0*i)/(rstar)
    trdur = orbit.duration(e, period/86400., omega, mstar/msun, rstar, rplanet, i*np.pi/180.)/(1440.*period/86400)
    #Fold data
    x = np.abs(x % 1)
    trlimb = orbit.limbtime(btr, trdur, 1, rp_rs)[0]
    ecldur, becl, ecllimb, midpt = orbit.scaled_eclipse(e, omega, trdur, rp_rs, btr)
    trpars = np.array([0, trdur, rp_rs**2, trlimb, trlimb, flux])
    eclpars = np.array([midpt, ecldur, ecldepth, ecllimb, ecllimb, flux])
    eclipse = mandelecl(eclpars, x)
    transit = mandelecl(trpars, x)
    return eclipse*transit/flux

#no
def medianip(ipparams, posflux, etc = [], retbinflux = False):
    """
  This function fits the intra-pixel sensitivity effect using the median 
   within a given radius of the current position.

  Parameters
  ----------
    ipparams :  tuple
                unused
    y :         1D array, size = # of measurements
                Pixel position along y
    x :         1D array, size = # of measurements
                Pixel position along x
    flux :      1D array, size = # of measurements
                Observed flux at each position
    wherebinflux :  1D array, size = # of bins    
    
  Returns
  -------
    1D array, size = # of measurements
    Intra-pixel-corrected flux multiplier

  Revisions
  ---------
  2010-06-06    Kevin Stevenson, UCF  
            kevin218@knights.ucf.edu
        Original version
    """
    
    r          = ipparams[0]
    y, x, flux, whereltrad = posflux[0:4]
    
    fluxcorr   = np.ones(flux.size)
    print(len(flux), len(etc))
    fluxip     = flux / etc
    for i in range(flux.size):
        #if i%10000 == 0:
            #print(i)
        fluxcorr[i] = np.median(fluxip[whereltrad[i]])
    
    if retbinflux == False:
        return fluxcorr
    else:
        return [fluxcorr, -1]

#no
def flatfield(ffparams, ap, etc = []):
    """
  This function perform flat field correction on a n-by-n pixel aperture.


  Parameters
  ----------
    ff#:    Flux modifier, starting with upper left pixel, reads normally (shape = n*n or n,n)
    flux:   Array of flux values (shape = #images)
    ap:     3D array of flux values within the aperture (shape = n,n,#images)

  Returns
  -------
    This function returns an array of new flux values.

  Notes
  -----
    Apertures for each image must be the same size as ffparams. (ap[0].size=ffparams.size)

  References
  ----------
    See SI from Harrinton et al. (2007)

  Revisions
  ---------
  2010-05-10     Kevin Stevenson, UCF
            kevin218@knights.ucf.edu
        Original version
    """
    ff   = np.array(ffparams[:-1])
    flux = ffparams[-1]
    #if ff.ndim == 1:
    #    ff = np.reshape(ff, (ap.shape[1],ap.shape[2]))
    ap = np.reshape(ap, (-1, ff.size))

    #Test which method is faster
    #deltaflux = np.sum(np.sum((ffparams*ap - ap), axis=1), axis=1)
    #ne.evaluate('flux + deltaflux')
    return (np.sum((ff*ap - ap), axis=1) + flux)

#no
def not0risingexp(params, x, etc=[]):
   """
   Rising exponential model for wa012bs11 with x0 in terms of m.
   """
   goal  = params[0]
   m     = params[1]
   a     = params[2]  # x0(m) fitted goal
   b     = params[3]  # x0(m) fitted curvature
   c     = params[4]  # x0(m) fitted offset

   # parameters for x0 related to m (from leastsq)
#   x0par = np.array([a, b, c])
#  This should not be passed as a numpy array
   x0par = [a, b, c]
   x0    = risingexp(x0par, m)
   
   return ne.evaluate('goal*(1 - exp(-m*(x - x0)))')
#no
def not0ramp(params, t, etc):
    """
    Rising exponential-like model with t0 in terms of m.
    """
    #goal  = params[0]
    #m     = params[1]
    #a     = params[2]  # x0(m) fitted goal
    #b     = params[3]  # x0(m) fitted curvature
    #c     = params[4]  # x0(m) fitted offset
    
    modelfunc = etc[0]
    t0func    = etc[1]
    
    # parameters for t0 related to m (from leastsq)
    params[2] = t0func(etc[2], params[1])
    
    return modelfunc(params, t)
