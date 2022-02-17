from numpy import size,zeros,where,arccos,sqrt,pi,log,sin,cos,bitwise_and
try:
    from models_c import occultquad as occ #ellke, ellpic_bulirsch
except:
    pass

# Computes Hasting's polynomial approximation for the complete
# elliptic integral of the first (ek) and second (kk) kind
"""
def ellke(k):
    #print('ellke:',k.shape)
    m1=1.-k**2
    logm1 = log(m1)

    a1=0.44325141463
    a2=0.06260601220
    a3=0.04757383546
    a4=0.01736506451
    b1=0.24998368310
    b2=0.09200180037
    b3=0.04069697526
    b4=0.00526449639
    ee1=1.+m1*(a1+m1*(a2+m1*(a3+m1*a4)))
    ee2=m1*(b1+m1*(b2+m1*(b3+m1*b4)))*(-logm1)
    ek = ee1+ee2
        
    a0=1.38629436112
    a1=0.09666344259
    a2=0.03590092383
    a3=0.03742563713
    a4=0.01451196212
    b0=0.5
    b1=0.12498593597
    b2=0.06880248576
    b3=0.03328355346
    b4=0.00441787012
    ek1=a0+m1*(a1+m1*(a2+m1*(a3+m1*a4)))
    ek2=(b0+m1*(b1+m1*(b2+m1*(b3+m1*b4))))*logm1
    kk = ek1-ek2
    
    return [ek,kk]
"""

# Computes the complete elliptical integral of the third kind using
# the algorithm of Bulirsch (1965):
"""
def ellpic_bulirsch(n,k):
    #print('bulirsch:',k.shape, n.shape)
    kc=sqrt(1.-k**2); p=n+1.
    if(min(p) < 0.):
        print('Negative p')
    m0=1.; c=1.; p=sqrt(p); d=1./p; e=kc
    while 1:
        f = c; c = d/p+c; g = e/p; d = 2.*(f*g+d)
        p = g + p; g = m0; m0 = kc + m0
        if max(abs(1.-kc/g)) > 1.e-8:
            kc = 2*sqrt(e); e=kc*m0
        else:
            return 0.5*pi*(c*m0+d)/(m0*(m0+p))
"""

#   Python translation of IDL code.
#   This routine computes the lightcurve for occultation of a
#   quadratically limb-darkened source without microlensing.  Please
#   cite Mandel & Agol (2002) and Eastman & Agol (2008) if you make use
#   of this routine in your research.  Please report errors or bugs to
#   jdeast@astronomy.ohio-state.edu
def trquad(params, t, etc):
    '''
    Parameters
    ----------
    midpt:  Center of eclipse
    rprs:   Planet radius / stellar radius
    cosi:   Cosine of the inclination
    ars:    Semi-major axis / stellar radius
    flux:   Flux offset from 0
    p:      Period in same units as t
    u#:     Limb-darkening coefficients
    t:	    Array of phase/time points
    
    Returns
    -------
    muo1    fraction of flux at each z0 for a limb-darkened source
    mu0     fraction of flux at each z0 for a uniform source
    
    Revisions
    ---------
    2012-08-13	Kevin Stevenson, UChicago  
                kbs@uchicago.edu
                Modified from Jason Eastman's version
    '''
    #DEFINE PARAMETERS
    midpt, rprs, cosi, ars, flux, p, u1, u2 = params
    
    #COMPUTE z(t) FOR TRANSIT ONLY (NOT ECLIPSE)
    #NOTE: z(t) ASSUMES A CIRCULAR ORBIT
    z  = ars*sqrt(sin(2*pi*(t-midpt)/p)**2 + (cosi*cos(2*pi*(t-midpt)/p))**2)
    z[where(bitwise_and((t-midpt)%p > p/4.,(t-midpt)%p < p*3./4))] = ars
    
    try:
        model = occ(z, u1, u2, abs(rprs), len(z))
    except:
        model = occultquad(z, u1, u2, abs(rprs), len(z))
    
    if rprs < 0:
        model = 2 - model
    
    return flux*model
    
def occultquad(z, u1, u2, rprs, lenz):
    '''
    
    '''
    nz = size(z)
    lambdad = zeros(nz)
    etad = zeros(nz)
    lambdae = zeros(nz)
    omega=1.-u1/3.-u2/6.

    ## tolerance for double precision equalities
    ## special case integrations
    tol = 1e-14

    rprs = abs(rprs)
    
    z = where(abs(rprs-z) < tol,rprs,z)
    z = where(abs((rprs-1)-z) < tol,rprs-1.,z)
    z = where(abs((1-rprs)-z) < tol,1.-rprs,z)
    z = where(z < tol,0.,z)
               
    x1=(rprs-z)**2.
    x2=(rprs+z)**2.
    x3=rprs**2.-z**2.
    
    ## trivial case of no planet
    if rprs <= 0.:
        muo1 = zeros(nz) + 1. 
        mu0  = zeros(nz) + 1.
        #return [muo1,mu0]
        return muo1

    ## Case 1 - the star is unocculted:
    ## only consider points with z lt 1+rprs
    notusedyet = where( z < (1. + rprs) )
    notusedyet = notusedyet[0]
    if size(notusedyet) == 0:
        muo1 =1.-((1.-u1-2.*u2)*lambdae+(u1+2.*u2)*(lambdad+2./3.*(rprs > z))+ \
                  u2*etad)/omega
        mu0=1.-lambdae
        #return [muo1,mu0]
        return muo1

    # Case 11 - the  source is completely occulted:
    if rprs >= 1.:
        occulted = where(z[notusedyet] <= rprs-1.)#,complement=notused2)
        if size(occulted) != 0:
            ndxuse = notusedyet[occulted]
            etad[ndxuse] = 0.5 # corrected typo in paper
            lambdae[ndxuse] = 1.
            # lambdad = 0 already
            notused2 = where(z[notusedyet] > rprs-1)
            if size(notused2) == 0:
                muo1 =1.-((1.-u1-2.*u2)*lambdae+(u1+2.*u2)*(lambdad+2./3.* \
                                                 (rprs > z))+u2*etad)/omega
                mu0=1.-lambdae
                #return [muo1,mu0]
                return muo1
            notusedyet = notusedyet[notused2]
                
    # Case 2, 7, 8 - ingress/egress (uniform disk only)
    inegressuni = where((z[notusedyet] >= abs(1.-rprs)) & (z[notusedyet] < 1.+rprs))
    if size(inegressuni) != 0:
        ndxuse = notusedyet[inegressuni]
        tmp = (1.-rprs**2.+z[ndxuse]**2.)/2./z[ndxuse]
        tmp = where(tmp > 1.,1.,tmp)
        tmp = where(tmp < -1.,-1.,tmp)
        kap1 = arccos(tmp)
        tmp = (rprs**2.+z[ndxuse]**2-1.)/2./rprs/z[ndxuse]
        tmp = where(tmp > 1.,1.,tmp)
        tmp = where(tmp < -1.,-1.,tmp)
        kap0 = arccos(tmp)
        tmp = 4.*z[ndxuse]**2-(1.+z[ndxuse]**2-rprs**2)**2
        tmp = where(tmp < 0,0,tmp)
        lambdae[ndxuse] = (rprs**2*kap0+kap1 - 0.5*sqrt(tmp))/pi
        # eta_1
        etad[ndxuse] = 1./2./pi*(kap1+rprs**2*(rprs**2+2.*z[ndxuse]**2)*kap0- \
           (1.+5.*rprs**2+z[ndxuse]**2)/4.*sqrt((1.-x1[ndxuse])*(x2[ndxuse]-1.)))
    
    # Case 5, 6, 7 - the edge of planet lies at origin of star
    ocltor = where(z[notusedyet] == rprs)#, complement=notused3)
    t = where(z[notusedyet] == rprs)
    if size(ocltor) != 0:
        ndxuse = notusedyet[ocltor] 
        if rprs < 0.5:
            # Case 5
            q=2.*rprs  # corrected typo in paper (2k -> 2rprs)
            Ek,Kk = ellke(q)
            # lambda_4
            lambdad[ndxuse] = 1./3.+2./9./pi*(4.*(2.*rprs**2-1.)*Ek+\
                                              (1.-4.*rprs**2)*Kk)
            # eta_2
            etad[ndxuse] = rprs**2/2.*(rprs**2+2.*z[ndxuse]**2)        
            lambdae[ndxuse] = rprs**2 # uniform disk
        elif rprs > 0.5:
            # Case 7
            q=0.5/rprs # corrected typo in paper (1/2k -> 1/2rprs)
            Ek,Kk = ellke(q)
            # lambda_3
            lambdad[ndxuse] = 1./3.+16.*rprs/9./pi*(2.*rprs**2-1.)*Ek-\
                              (32.*rprs**4-20.*rprs**2+3.)/9./pi/rprs*Kk
            # etad = eta_1 already
        else:
            # Case 6
            lambdad[ndxuse] = 1./3.-4./pi/9.
            etad[ndxuse] = 3./32.
        notused3 = where(z[notusedyet] != rprs)
        if size(notused3) == 0:
            muo1 =1.-((1.-u1-2.*u2)*lambdae+(u1+2.*u2)*\
                      (lambdad+2./3.*(rprs > z))+u2*etad)/omega
            mu0=1.-lambdae
            #return [muo1,mu0]
            return muo1
        notusedyet = notusedyet[notused3]

    # Case 2, Case 8 - ingress/egress (with limb darkening)
    inegress = where( ((z[notusedyet] > 0.5+abs(rprs-0.5)) & \
                       (z[notusedyet] < 1.+rprs))  | \
                      ( (rprs > 0.5) & (z[notusedyet] > abs(1.-rprs)) & \
                        (z[notusedyet] < rprs)) )#, complement=notused4)
    if size(inegress) != 0:

        ndxuse = notusedyet[inegress]
        q=sqrt((1.-x1[ndxuse])/(x2[ndxuse]-x1[ndxuse]))
        Ek,Kk = ellke(q)
        n=1./x1[ndxuse]-1.

        # lambda_1:
        lambdad[ndxuse]=2./9./pi/sqrt(x2[ndxuse]-x1[ndxuse])*\
                         (((1.-x2[ndxuse])*(2.*x2[ndxuse]+x1[ndxuse]-3.)-\
                           3.*x3[ndxuse]*(x2[ndxuse]-2.))*Kk+(x2[ndxuse]-\
                           x1[ndxuse])*(z[ndxuse]**2+7.*rprs**2-4.)*Ek-\
                          3.*x3[ndxuse]/x1[ndxuse]*ellpic_bulirsch(n,q))

        notused4 = where( ( (z[notusedyet] <= 0.5+abs(rprs-0.5)) | \
                            (z[notusedyet] >= 1.+rprs) ) & ( (rprs <= 0.5) | \
                            (z[notusedyet] <= abs(1.-rprs)) | \
                            (z[notusedyet] >= rprs) ))
        if size(notused4) == 0:
            muo1 =1.-((1.-u1-2.*u2)*lambdae+(u1+2.*u2)*(lambdad+2./3.*\
                                                     (rprs > z))+u2*etad)/omega
            mu0=1.-lambdae
            #return [muo1,mu0]
            return muo1
        notusedyet = notusedyet[notused4]

    # Case 3, 4, 9, 10 - planet completely inside star
    if rprs < 1.:
        inside = where(z[notusedyet] <= (1.-rprs))#, complement=notused5)
        if size(inside) != 0:
            ndxuse = notusedyet[inside]

            ## eta_2
            etad[ndxuse] = rprs**2/2.*(rprs**2+2.*z[ndxuse]**2)

            ## uniform disk
            lambdae[ndxuse] = rprs**2

            ## Case 4 - edge of planet hits edge of star
            edge = where(z[ndxuse] == 1.-rprs)#, complement=notused6)
            if size(edge[0]) != 0:
                ## lambda_5
                lambdad[ndxuse[edge]] = 2./3./pi*arccos(1.-2.*rprs)-\
                                      4./9./pi*sqrt(rprs*(1.-rprs))*(3.+2.*rprs-8.*rprs**2)
                if rprs > 0.5:
                    lambdad[ndxuse[edge]] -= 2./3.
                notused6 = where(z[ndxuse] != 1.-rprs)
                if size(notused6) == 0:
                    muo1 =1.-((1.-u1-2.*u2)*lambdae+(u1+2.*u2)*\
                              (lambdad+2./3.*(rprs > z))+u2*etad)/omega
                    mu0=1.-lambdae
                    #return [muo1,mu0]
                    return muo1
                ndxuse = ndxuse[notused6[0]]

            ## Case 10 - origin of planet hits origin of star
            origin = where(z[ndxuse] == 0)#, complement=notused7)
            if size(origin) != 0:
                ## lambda_6
                lambdad[ndxuse[origin]] = -2./3.*(1.-rprs**2)**1.5
                notused7 = where(z[ndxuse] != 0)
                if size(notused7) == 0:
                    muo1 =1.-((1.-u1-2.*u2)*lambdae+(u1+2.*u2)*\
                              (lambdad+2./3.*(rprs > z))+u2*etad)/omega
                    mu0=1.-lambdae
                    #return [muo1,mu0]
                    return muo1
                ndxuse = ndxuse[notused7[0]]
   
            q=sqrt((x2[ndxuse]-x1[ndxuse])/(1.-x1[ndxuse]))
            n=x2[ndxuse]/x1[ndxuse]-1.
            Ek,Kk = ellke(q)    

            ## Case 3, Case 9 - anywhere in between
            ## lambda_2
            lambdad[ndxuse] = 2./9./pi/sqrt(1.-x1[ndxuse])*\
                              ((1.-5.*z[ndxuse]**2+rprs**2+x3[ndxuse]**2)*Kk+\
                               (1.-x1[ndxuse])*(z[ndxuse]**2+7.*rprs**2-4.)*Ek-\
                               3.*x3[ndxuse]/x1[ndxuse]*ellpic_bulirsch(n,q))

        ## if there are still unused elements, there's a bug in the code
        ## (please report it)
        notused5 = where(z[notusedyet] > (1.-rprs))
        if notused5[0] != 0:
            print("ERROR: the following values of z didn't fit into a case:")
            return [-1,-1]

        muo1 =1.-((1.-u1-2.*u2)*lambdae+(u1+2.*u2)*(lambdad+2./3.*(rprs > z))+\
                  u2*etad)/omega
        mu0=1.-lambdae
    return muo1

