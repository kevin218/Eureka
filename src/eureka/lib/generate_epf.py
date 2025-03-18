"""
EPF Generator for Eureka!

Description:
-------------
This script generates an Exoplanet Parameter File (EPF) to aid in modeling
exoplanetary systems.

Using input data from STSCI's ExoMAST database, the script populates an EPF
with relevant parameters for an exoplanet and its orbit.

Inputs:
-------
- planet_data: A dictionary containing relevant parameters for the exoplanet
               from ExoMAST.

Required keys: 'Rp/Rs', 'orbital_period', 'transit_time', 'inclination',
               'a/Rs', 'eccentricity', 'omega', 'Rp_lower', 'Rp_upper',
               'orbital_period_lower', 'orbital_period_upper',
               'transit_time_upper', 'inclination_upper', and 'a/Rs_upper'.

- file_path: The path to the output file where the generated EPF will be saved.
             Default is 'output_file.txt'.

- [Additional Parameters]: The function accepts a many additional parameters
                           customized for modeling purposes. These parameters
                           include values, boundaries, and prior types for
                           different characteristics like the planet's radius,
                           orbital period, transit time, inclination, etc.

Outputs:
--------
- An EPF written to the provided file_path. This file will contain
  transit/eclipse parameters for the given exoplanet, with values, freedom of
  parameters (fixed/free), prior
  parameters, and prior types.

Dependencies:
-------------
- External data source: ExoMAST


Author: Reza Ashtari
Date: 08/22/2023
"""


# Generate EPF - Starter EPF populated using ExoMAST. Some free params.
def start(planet_data, file_path='output_file.txt',
          rp_val=0, rp_free='free', rp_pp1=0, rp_pp2=0, rp_pt='U',
          per_val=0, per_free='fixed', per_pp1=0, per_pp2=0, per_pt='U',
          t0_val=0, t0_free='free', t0_pp1=0, t0_pp2=0.5, t0_pt='N',
          inc_val=0, inc_free='free', inc_pp1=0, inc_pp2=5, inc_pt='N',
          a_val=0, a_free='free', a_pp1=0, a_pp2=0.5, a_pt='N',
          ecc_val=0.0, ecc_free='fixed', ecc_pp1=0, ecc_pp2=1, ecc_pt='U',
          w_val=90, w_free='fixed', w_pp1=0, w_pp2=360, w_pt='U',
          ld_type='quadratic', ld_free='independent',
          u1_val=0.3, u1_free='fixed', u1_pp1=0, u1_pp2=1, u1_pt='U',
          u2_val=0.1, u2_free='fixed', u2_pp1=0, u2_pp2=1, u2_pt='U',
          c0_val=1, c0_free='free', c0_pp1=1, c0_pp2=0.1, c0_pt='N',
          c1_val=0, c1_free='free', c1_pp1=-1, c1_pp2=1, c1_pt='U',
          scatter_mult_val=1, scatter_mult_free='free', scatter_mult_pp1=0.8,
          scatter_mult_pp2=1.2, scatter_mult_pt='U'):
    """
    Exoplanet Parameter File (EPF) Generator for Eureka!

    Description:
    This function populates an Exoplanet Parameter File (EPF) tailored to
    facilitate modeling of exoplanetary systems.
    It gleans its data from STSCI's ExoMAST database, and subsequently,
    furnishes an EPF, complete with requisite parameters for the exoplanet and
    its orbit.

    Parameters:
    planet_data: A dictionary
    The main source of parameters for the exoplanet sourced from ExoMAST.
    Essential keys: 'Rp/Rs', 'orbital_period', 'transit_time', 'inclination',
    'a/Rs',
    'eccentricity', 'omega', 'Rp_lower', 'Rp_upper', 'orbital_period_lower',
    'orbital_period_upper', 'transit_time_upper', 'inclination_upper', and
    'a/Rs_upper'.

    file_path: str
    The specified pathway to the desired output file where the newly-generated
    EPF will be archived.
    Default: 'output_file.txt'.

    Additional Parameters:
    Multiple auxiliary parameters facilitate tailored modeling. They include
    specific values, boundary constraints,
    and prior types for various characteristics such as the planet's radius,
    its orbital period, transit timing, inclination, among others.

    Returns:
    An EPF written to the prescribed file_path.
    This file enumerates transit/eclipse parameters for the featured exoplanet.
    It lists values, parameter flexibility (whether they are fixed or free),
    parameters for priors, and the types of these priors.
    Dependencies:
    External data: This function hinges on data pulled from the ExoMAST
    platform.

    Notes:
    The function initially generates an EPF with some free parameters.
    Orbital parameters in the EPF are later overwritten with actual data from
    ExoMAST, optimizing accuracy.
    For some parameters, if certain values are not available, default values or
    calculations are used to fill in gaps.
    Setting bounds for parameters, like eccentricity or inclination, ensures
    physical plausibility in the modeling process.

    Author:
    Reza Ashtari
    Date:
    08/22/2023
    """

    # Overwrite EPF orbital params using exoMAST data
    # rp_free = 'free'
    rp_free = 'fixed'
    rp_pt = "U"
    rp_val = planet_data['Rp/Rs']

    if planet_data['Rp_lower'] is None:
        rp_pp1 = 0
    else:
        rp_pp1 = planet_data['Rp/Rs']
        - (planet_data['Rp_lower'] / planet_data['Rs'])  # Lower Bound

    if planet_data['Rp_upper'] is None:
        rp_pp2 = 1
    else:
        rp_pp2 = planet_data['Rp/Rs']
        + (planet_data['Rp_upper'] / planet_data['Rs'])  # Upper Bound

    per_free = 'fixed'
    per_pt = "U"
    per_val = planet_data['orbital_period']

    per_pp1 = planet_data['orbital_period']
    - planet_data['orbital_period_lower']  # Lower Bound

    per_pp2 = planet_data['orbital_period']
    + planet_data['orbital_period_upper']  # Upper Bound

    # t0_free = 'free'
    # t0_free = 'fixed'
    t0_pt = "N"
    t0_val = planet_data['transit_time']
    t0_pp1 = planet_data['transit_time']  # Mean
    t0_pp2 = planet_data['transit_time_upper']  # Standard Deviation

    # if planet_data['canonical_name'] == 'HD 86226 c':
    #     per_val = 3.9846654
        
    # if planet_data['canonical_name'] == 'HD 86226 c':
    #     t0_val = 2458698.6559
    #     t0_pp1 = 2458698.6559   # Mean
    #     t0_pp2 = 0.001  # Standard Deviation

    inc_free = 'free'
    inc_pt = "N"
    inc_val = planet_data['inclination']
    inc_pp1 = planet_data['inclination']  # Mean
    inc_pp2 = planet_data['inclination_upper']  # Standard Deviation

    a_free = 'free'
    a_pt = "N"
    a_val = planet_data['a/Rs']
    if a_val is None:
        a_val = 0.1
    a_pp1 = a_val  # Mean
    a_pp2 = planet_data['a/Rs_upper']  # Standard Deviation
    if a_pp2 is None:
        a_pp2 = planet_data['a/Rs_lower']
        if a_pp2 is None:
            a_pp2 = a_val * 0.1

    ecc_free = 'fixed'
    ecc_pt = "U"
    ecc_val = planet_data['eccentricity']
    if (ecc_val is None):
        ecc_val = 0.0
    ecc_pp1 = 0.0  # Lower Bound
    ecc_pp2 = 1  # Upper Bound

    w_free = 'fixed'
    w_pt = "U"
    w_val = planet_data['omega']
    if (w_val is None):
        w_val = 90.0
    w_pp1 = 0.0  # Lower Bound
    w_pp2 = 359.9  # Upper Bound

    content = \
        f'''
        # Stage 5 Fit Parameters Documentation:
        # https://eurekadocs.readthedocs.io/en/latest/ecf.html#stage-5-fit-parameters
        # Name  Value    Free?   PriorPar1   PriorPar2   PriorType
        # "Free?" can be free, fixed, white_free, white_fixed, shared, or
        # independent
        # PriorType can be U (Uniform), LU (Log Uniform), or N (Normal).
        # If U/LU, PriorPar1 and PriorPar2 represent upper and lower limits of
        # the parameter/log(the parameter).
        # If N, PriorPar1 is the mean and PriorPar2 is the standard deviation
        # of a Gaussian prior.
        #-------------------------------------------------------------------------------------------------------
        #
        # ------------------
        # ** Transit/eclipse parameters **
        # ------------------
        rp           {rp_val}       '{rp_free}'   {rp_pp1}     {rp_pp2}      {rp_pt}
        per          {per_val}      '{per_free}'  {per_pp1}    {per_pp2}     {per_pt}
        t0           {t0_val}       '{t0_free}'   {t0_pp1}     {t0_pp2}      {t0_pt}
        time_offset  2400000.5      'independent'
        inc          {inc_val}      '{inc_free}'  {inc_pp1}    {inc_pp2}     {inc_pt}
        a            {a_val}        '{a_free}'    {a_pp1}      {a_pp2}       {a_pt}
        ecc          {ecc_val}      '{ecc_free}'  {ecc_pp1}    {ecc_pp2}     {ecc_pt}
        w            {w_val}        '{w_free}'    {w_pp1}      {w_pp2}       {w_pt}
        limb_dark    '{ld_type}'    '{ld_free}'
        u1           {u1_val}       '{u1_free}'   {u1_pp1}     {u1_pp2}      {u1_pt}
        u2           {u2_val}       '{u2_free}'   {u2_pp1}     {u2_pp2}      {u2_pt}
        c0           {c0_val}       '{c0_free}'   {c0_pp1}     {c0_pp2}      {c0_pt}
        c1           {c1_val}       '{c1_free}'   {c1_pp1}     {c1_pp2}      {c1_pt}
        scatter_mult {scatter_mult_val} '{scatter_mult_free}' {scatter_mult_pp1} {scatter_mult_pp2} {scatter_mult_pt}

        '''

        # xpos         {xpos_val}     '{xpos_free}' {xpos_pp1}   {xpos_pp2}    {xpos_pt}
        # ypos         {ypos_val}     '{ypos_free}' {ypos_pp1}   {ypos_pp2}    {ypos_pt}

    if planet_data['canonical_name'] == 'HD 86226 c':        
        content = \
            f'''
            # Stage 5 Fit Parameters Documentation:
            # https://eurekadocs.readthedocs.io/en/latest/ecf.html#stage-5-fit-parameters
            # Name  Value    Free?   PriorPar1   PriorPar2   PriorType
            # "Free?" can be free, fixed, white_free, white_fixed, shared, or
            # independent
            # PriorType can be U (Uniform), LU (Log Uniform), or N (Normal).
            # If U/LU, PriorPar1 and PriorPar2 represent upper and lower limits of
            # the parameter/log(the parameter).
            # If N, PriorPar1 is the mean and PriorPar2 is the standard deviation
            # of a Gaussian prior.
            #-------------------------------------------------------------------------------------------------------
            #
            # ------------------
            # ** Transit/eclipse parameters **
            # ------------------
            rp           0.01962646833343065     'fixed'   0     1      U
            per          3.984651184041696  'fixed'  3.9846654    0.0000092     N
            t0           58698.15683206442      'fixed'   58698.1559     0.00065      N
            time_offset  2400000.5  'independent'
            inc          86.5      'fixed'  86.5    0.16     N
            a            10.11        'fixed'    7.1      1.0034833931292224       N
            ecc          0.075      'fixed'  0    1     U
            w            196.0      'fixed'    0      360       U
            limb_dark    'quadratic'    'independent'
            u1           0.15651933602854776     'fixed'   0.15651933602854776     0.01      N
            u2           0.236344988380854      'fixed'   0.2367380655160314     0.01      N
            c0           0.9985409602491301       'free'   1     0.01      N
            c1           -0.015325138271693704       'free'   -1     1      U
            scatter_mult 1.19999999     'fixed'   0.8 1.2 U 

            '''

    with open(file_path, 'w') as f:
        f.write(content)


# Generate EPF - Stage 5 Optimization
def Stage_5(planet_data, file_path='output_file.txt',
            rp_val=0, rp_free='free', rp_pp1=0, rp_pp2=0, rp_pt='U',
            per_val=0, per_free='fixed', per_pp1=0, per_pp2=0, per_pt='U',
            t0_val=0, t0_free='free', t0_pp1=0, t0_pp2=0.5, t0_pt='N',
            inc_val=0, inc_free='free', inc_pp1=0, inc_pp2=5, inc_pt='N',
            a_val=0, a_free='free', a_pp1=0, a_pp2=0.5, a_pt='N',
            ecc_val=0.0, ecc_free='fixed', ecc_pp1=0, ecc_pp2=1, ecc_pt='U',
            w_val=90, w_free='fixed', w_pp1=0, w_pp2=360, w_pt='U',
            ld_type='quadratic', ld_free='independent',
            u1_val=0.3, u1_free='fixed', u1_pp1=0, u1_pp2=1, u1_pt='U',
            u2_val=0.1, u2_free='fixed', u2_pp1=0, u2_pp2=1, u2_pt='U',
            c0_val=1, c0_free='free', c0_pp1=1, c0_pp2=0.1, c0_pt='N',
            c1_val=0, c1_free='free', c1_pp1=-1, c1_pp2=1, c1_pt='U',
            scatter_mult_val=1, scatter_mult_free='free', scatter_mult_pp1=0.8, scatter_mult_pp2=1.2, scatter_mult_pt='U'):
    """
    Exoplanet Parameter File (EPF) Generator for Eureka!

    Description:
    This function populates an Exoplanet Parameter File (EPF) tailored to
    facilitate modeling of exoplanetary systems.
    It gleans its data from STSCI's ExoMAST database, and subsequently,
    furnishes an EPF, complete with requisite parameters for the exoplanet and
    its orbit.

    Parameters:
    planet_data: A dictionary
    The main source of parameters for the exoplanet sourced from ExoMAST.
    Essential keys: 'Rp/Rs', 'orbital_period', 'transit_time', 'inclination',
    'a/Rs',
    'eccentricity', 'omega', 'Rp_lower', 'Rp_upper', 'orbital_period_lower',
    'orbital_period_upper', 'transit_time_upper', 'inclination_upper', and
    'a/Rs_upper'.

    file_path: str
    The specified pathway to the desired output file where the newly-generated
    EPF will be archived.
    Default: 'output_file.txt'.

    Additional Parameters:
    Multiple auxiliary parameters facilitate tailored modeling. They include
    specific values, boundary constraints,
    and prior types for various characteristics such as the planet's radius,
    its orbital period, transit timing, inclination, among others.

    Returns:
    An EPF written to the prescribed file_path.
    This file enumerates transit/eclipse parameters for the featured exoplanet.
    It lists values, parameter flexibility (whether they are fixed or free),
    parameters for priors, and the types of these priors.
    Dependencies:
    External data: This function hinges on data pulled from the ExoMAST
    platform.

    Notes:
    The function initially generates an EPF with some free parameters.
    Optimized values from the "best" dictionary, containing optimized ECF and
    EPF inputs, are used for the parameter values
    For some parameters, if certain values are not available, default values or
    calculations are used to fill in gaps.
    Setting bounds for parameters, like eccentricity or inclination, ensures
    physical plausibility in the modeling process.

    Author:
    Reza Ashtari
    Date:
    08/22/2023
    """

    # Overwrite EPF orbital params using exoMAST data
    # rp_free = 'free'
    # rp_pt = "U"

    rp_val = planet_data['Rp/Rs']

    rp_pp1 = planet_data['Rp/Rs'] - (planet_data['Rp_lower']
                                     / planet_data['Rs'])  # Lower Bound
    rp_pp2 = planet_data['Rp/Rs'] + (planet_data['Rp_upper']
                                     / planet_data['Rs'])  # Upper Bound
        
    # if planet_data['canonical_name'] == 'HD 86226 c':
    #     rp_pp1 = 0.01942
    #     rp_pp2 = 0.020961
    # if planet_data['canonical_name'] == 'HD-86226 c':
    #     rp_pp1 = 0.01942
    #     rp_pp2 = 0.020961

    # rp_pp1 = 0  # Lower Bound
    # rp_pp2 = 1  # Upper Bound

    # per_free = 'fixed'
    # per_pt = "U"
    per_val = planet_data['orbital_period']

    per_pp1 = planet_data['orbital_period']
    - planet_data['orbital_period_lower']  # Lower Bound

    per_pp2 = planet_data['orbital_period']
    + planet_data['orbital_period_upper']  # Upper Bound

    # t0_free = 'free'
    # t0_pt = "N"
    t0_val = planet_data['transit_time']
    t0_pp1 = planet_data['transit_time']  # Mean
    t0_pp2 = planet_data['transit_time_upper']  # Standard Deviation

    # if planet_data['canonical_name'] == 'HD 86226 c':
    #     per_val = 3.9846654
        
    # if planet_data['canonical_name'] == 'HD 86226 c':
    #     t0_val = 2458698.6559
    #     t0_pp1 = 2458698.6559   # Mean
    #     t0_pp2 = 0.001  # Standard Deviation

    # inc_free = 'fixed'
    # inc_pt = "N"
    inc_val = planet_data['inclination']
    inc_pp1 = planet_data['inclination']  # Mean
    inc_pp2 = planet_data['inclination_upper']  # Standard Deviation

    inc_val = planet_data['inclination']
    inc_pp1 = planet_data['inclination']  # Mean
    inc_pp2 = planet_data['inclination_upper']  # Standard Deviation

    a_free = 'free'
    a_pt = "N"
    a_val = planet_data['a/Rs']
    if a_val is None:
        a_val = 0.1
    a_pp1 = a_val  # Mean
    a_pp2 = planet_data['a/Rs_upper']  # Standard Deviation
    if a_pp2 is None:
        a_pp2 = planet_data['a/Rs_lower']
        if a_pp2 is None:
            a_pp2 = a_val * 0.1

    ecc_free = 'fixed'
    ecc_pt = "U"
    ecc_val = planet_data['eccentricity']
    if (ecc_val is None):
        ecc_val = 0.0
    ecc_pp1 = 0.0  # Lower Bound
    ecc_pp2 = 1  # Upper Bound

    w_free = 'fixed'
    w_pt = "U"
    w_val = planet_data['omega']
    if (w_val is None):
        w_val = 90.0
    w_pp1 = 0.0  # Lower Bound
    w_pp2 = 359.9  # Upper Bound

    content = \
        f'''
        # Stage 5 Fit Parameters Documentation:
        # https://eurekadocs.readthedocs.io/en/latest/ecf.html#stage-5-fit-parameters
        # Name  Value    Free?   PriorPar1   PriorPar2   PriorType
        # "Free?" can be free, fixed, white_free, white_fixed, shared, or
        # independent
        # PriorType can be U (Uniform), LU (Log Uniform), or N (Normal).
        # If U/LU, PriorPar1 and PriorPar2 represent upper and lower limits of
        # the parameter/log(the parameter).
        # If N, PriorPar1 is the mean and PriorPar2 is the standard deviation
        # of a Gaussian prior.
        #-------------------------------------------------------------------------------------------------------
        #
        # ------------------
        # ** Transit/eclipse parameters **
        # ------------------
        rp           {rp_val}       '{rp_free}'   {rp_pp1}     {rp_pp2}      {rp_pt}
        per          {per_val}      '{per_free}'  {per_pp1}    {per_pp2}     {per_pt}
        t0           {t0_val}       '{t0_free}'   {t0_pp1}     {t0_pp2}      {t0_pt}
        time_offset  2400000.5      'independent'
        inc          {inc_val}      '{inc_free}'  {inc_pp1}    {inc_pp2}     {inc_pt}
        a            {a_val}        '{a_free}'    {a_pp1}      {a_pp2}       {a_pt}
        ecc          {ecc_val}      '{ecc_free}'  {ecc_pp1}    {ecc_pp2}     {ecc_pt}
        w            {w_val}        '{w_free}'    {w_pp1}      {w_pp2}       {w_pt}
        limb_dark    '{ld_type}'    '{ld_free}'
        u1           {u1_val}       '{u1_free}'   {u1_pp1}     {u1_pp2}      {u1_pt}
        u2           {u2_val}       '{u2_free}'   {u2_pp1}     {u2_pp2}      {u2_pt}
        c0           {c0_val}       '{c0_free}'   {c0_pp1}     {c0_pp2}      {c0_pt}
        c1           {c1_val}       '{c1_free}'   {c1_pp1}     {c1_pp2}      {c1_pt}
        scatter_mult {scatter_mult_val} '{scatter_mult_free}' {scatter_mult_pp1} {scatter_mult_pp2} {scatter_mult_pt}

        '''
    
    if planet_data['canonical_name'] == 'HD 86226 c':        
        content = \
            f'''
            # Stage 5 Fit Parameters Documentation:
            # https://eurekadocs.readthedocs.io/en/latest/ecf.html#stage-5-fit-parameters
            # Name  Value    Free?   PriorPar1   PriorPar2   PriorType
            # "Free?" can be free, fixed, white_free, white_fixed, shared, or
            # independent
            # PriorType can be U (Uniform), LU (Log Uniform), or N (Normal).
            # If U/LU, PriorPar1 and PriorPar2 represent upper and lower limits of
            # the parameter/log(the parameter).
            # If N, PriorPar1 is the mean and PriorPar2 is the standard deviation
            # of a Gaussian prior.
            #-------------------------------------------------------------------------------------------------------
            #
            # ------------------
            # ** Transit/eclipse parameters **
            # ------------------
            rp           0.01962646833343065     'fixed'   0     1      U
            per          3.984651184041696  'fixed'  3.9846654    0.0000092     N
            t0           58698.15683206442      'fixed'   58698.1559     0.00065      N
            time_offset  2400000.5  'independent'
            inc          86.5      'fixed'  86.5    0.16     N
            a            10.11        'fixed'    7.1      1.0034833931292224       N
            ecc          0.075      'fixed'  0    1     U
            w            196.0      'fixed'    0      360       U
            limb_dark    'quadratic'    'independent'
            u1           0.15651933602854776     'fixed'   0.15651933602854776     0.01      N
            u2           0.236344988380854      'fixed'   0.2367380655160314     0.01      N
            c0           0.9985409602491301       'free'   1     0.01      N
            c1           -0.015325138271693704       'free'   -1     1      U
            scatter_mult 1.19999999     'fixed'   0.8 1.2 U 

            '''

    with open(file_path, 'w') as f:
        f.write(content)


# Generate EPF - Fixed Params
def fixed(best, file_path='output_file.txt',
          rp_val=None, rp_free='fixed', rp_pp1=0, rp_pp2=1, rp_pt='N',
          per_val=None, per_free='fixed', per_pp1=0, per_pp2=1, per_pt='N',
          t0_val=None, t0_free='fixed', t0_pp1=0, t0_pp2=1, t0_pt='N',
          inc_val=None, inc_free='fixed', inc_pp1=0, inc_pp2=1, inc_pt='N',
          a_val=None, a_free='fixed', a_pp1=0, a_pp2=1, a_pt='N',
          ecc_val=None, ecc_free='fixed', ecc_pp1=0, ecc_pp2=1, ecc_pt='N',
          w_val=None, w_free='fixed', w_pp1=0, w_pp2=1, w_pt='N',
          u1_val=None, u1_free='fixed', u1_pp1=0, u1_pp2=1, u1_pt='N',
          u2_val=None, u2_free='fixed', u2_pp1=0, u2_pp2=1, u2_pt='N',
          c0_val=None, c0_free='fixed', c0_pp1=0, c0_pp2=1, c0_pt='N',
          c1_val=None, c1_free='fixed', c1_pp1=0, c1_pp2=1, c1_pt='N',
          scatter_mult_val=None, scatter_mult_free='free', scatter_mult_pp1=0.8, scatter_mult_pp2=1.2, scatter_mult_pt='U'):

    """
    Exoplanet Parameter File (EPF) Generator for Eureka!

    Description:
    This function populates an Exoplanet Parameter File (EPF) tailored to
    facilitate modeling of exoplanetary systems.
    It gleans its data from STSCI's ExoMAST database, and subsequently,
    furnishes an EPF, complete with requisite parameters for the exoplanet and
    its orbit.

    Parameters:
    planet_data: A dictionary
    The main source of parameters for the exoplanet sourced from ExoMAST.
    Essential keys: 'Rp/Rs', 'orbital_period', 'transit_time', 'inclination',
    'a/Rs',
    'eccentricity', 'omega', 'Rp_lower', 'Rp_upper', 'orbital_period_lower',
    'orbital_period_upper', 'transit_time_upper', 'inclination_upper', and
    'a/Rs_upper'.

    file_path: str
    The specified pathway to the desired output file where the newly-generated
    EPF will be archived.
    Default: 'output_file.txt'.

    Additional Parameters:
    Multiple auxiliary parameters facilitate tailored modeling. They include
    specific values, boundary constraints,
    and prior types for various characteristics such as the planet's radius,
    its orbital period, transit timing, inclination, among others.

    Returns:
    An EPF written to the prescribed file_path.
    This file enumerates transit/eclipse parameters for the featured exoplanet.
    It lists values, parameter flexibility (whether they are fixed or free),
    parameters for priors, and the types of these priors.
    Dependencies:
    External data: This function hinges on data pulled from the ExoMAST
    platform.

    Notes:
    The function generates an EPF with only fixed parameters.
    Optimized values from the "best" dictionary, containing optimized ECF and
    EPF inputs, are used for the parameter values

    Author:
    Reza Ashtari
    Date:
    08/22/2023
    """

    # Check and set values from `best` if the parameters are None
    if rp_val is None:
        rp_val = best['rp']
    if per_val is None:
        per_val = best['per']
    if t0_val is None:
        t0_val = best['t0']
    if inc_val is None:
        inc_val = best['inc']
    if a_val is None:
        a_val = best['a']
    if ecc_val is None:
        ecc_val = best['ecc']
    if w_val is None:
        w_val = best['omega']
    if u1_val is None:
        u1_val = best['u1']
    if u2_val is None:
        u2_val = best['u2']
    if c0_val is None:
        c0_val = best['c0']
    if c1_val is None:
        c1_val = best['c1']
    if scatter_mult_val is None:
        scatter_mult_val = best['scatter_mult']

    content = \
        f'''
        # Stage 5 Fit Parameters Documentation:
        # https://eurekadocs.readthedocs.io/en/latest/ecf.html#stage-5-fit-parameters
        # Name  Value    Free?   PriorPar1   PriorPar2   PriorType
        # "Free?" can be free, fixed, white_free, white_fixed, shared, or
        # independent
        # PriorType can be U (Uniform), LU (Log Uniform), or N (Normal).
        # If U/LU, PriorPar1 and PriorPar2 represent upper and lower limits of
        # the parameter/log(the parameter).
        # If N, PriorPar1 is the mean and PriorPar2 is the standard deviation
        # of a Gaussian prior.
        #-------------------------------------------------------------------------------------------------------
        #
        # ------------------
        # ** Transit/eclipse parameters **
        # ------------------
        rp           {rp_val}       '{rp_free}'   {rp_pp1}     {rp_pp2}      {rp_pt}
        per          {per_val}      '{per_free}'  {per_pp1}    {per_pp2}     {per_pt}
        t0           {t0_val}       '{t0_free}'   {t0_pp1}     {t0_pp2}      {t0_pt}
        time_offset  2400000.5      'independent'
        inc          {inc_val}      '{inc_free}'  {inc_pp1}    {inc_pp2}     {inc_pt}
        a            {a_val}        '{a_free}'    {a_pp1}      {a_pp2}       {a_pt}
        ecc          {ecc_val}      '{ecc_free}'  {ecc_pp1}    {ecc_pp2}     {ecc_pt}
        w            {w_val}        '{w_free}'    {w_pp1}      {w_pp2}       {w_pt}
        limb_dark    'quadratic'    'independent'
        u1           {u1_val}       '{u1_free}'   {u1_pp1}     {u1_pp2}      {u1_pt}
        u2           {u2_val}       '{u2_free}'   {u2_pp1}     {u2_pp2}      {u2_pt}
        c0           {c0_val}       '{c0_free}'   {c0_pp1}     {c0_pp2}      {c0_pt}
        c1           {c1_val}       '{c1_free}'   {c1_pp1}     {c1_pp2}      {c1_pt}
        scatter_mult {scatter_mult_val} '{scatter_mult_free}' {scatter_mult_pp1} {scatter_mult_pp2} {scatter_mult_pt}

        '''
    
        # xpos         {xpos_val}     '{xpos_free}' {xpos_pp1}   {xpos_pp2}    {xpos_pt}
        # ypos         {ypos_val}     '{ypos_free}' {ypos_pp1}   {ypos_pp2}    {ypos_pt}

    with open(file_path, 'w') as f:
        f.write(content)


def Stage_6(best, planet_data, file_path='output_file.txt',
            rp_val=None, rp_free='free', rp_pp1=0, rp_pp2=0, rp_pt='U',
            per_val=None, per_free='fixed', per_pp1=0, per_pp2=0.1, per_pt='U',
            t0_val=None, t0_free='fixed', t0_pp1=0, t0_pp2=0.5, t0_pt='N',
            inc_val=None, inc_free='fixed', inc_pp1=0, inc_pp2=5, inc_pt='N',
            a_val=None, a_free='fixed', a_pp1=0, a_pp2=0.5, a_pt='N',
            ecc_val=None, ecc_free='fixed', ecc_pp1=0, ecc_pp2=1, ecc_pt='U',
            w_val=None, w_free='fixed', w_pp1=0, w_pp2=360, w_pt='U',
            ld_type='quadratic', ld_free='independent',
            u1_val=None, u1_free='fixed', u1_pp1=0, u1_pp2=1, u1_pt='U',
            u2_val=None, u2_free='fixed', u2_pp1=0, u2_pp2=1, u2_pt='U',
            c0_val=None, c0_free='free', c0_pp1=1, c0_pp2=0.1, c0_pt='N',
            c1_val=None, c1_free='free', c1_pp1=-1, c1_pp2=1, c1_pt='U',
            scatter_mult_val=None, scatter_mult_free='free', scatter_mult_pp1=0.8, scatter_mult_pp2=1.2, scatter_mult_pt='U'):

    """
    Exoplanet Parameter File (EPF) Generator for Eureka!

    Description:
    This function populates an Exoplanet Parameter File (EPF) tailored to
    facilitate modeling of exoplanetary systems.
    It gleans its data from STSCI's ExoMAST database, and subsequently,
    furnishes an EPF, complete with requisite parameters for the exoplanet and
    its orbit.

    Parameters:
    planet_data: A dictionary
    The main source of parameters for the exoplanet sourced from ExoMAST.
    Essential keys: 'Rp/Rs', 'orbital_period', 'transit_time', 'inclination',
    'a/Rs',
    'eccentricity', 'omega', 'Rp_lower', 'Rp_upper', 'orbital_period_lower',
    'orbital_period_upper', 'transit_time_upper', 'inclination_upper', and
    'a/Rs_upper'.

    file_path: str
    The specified pathway to the desired output file where the newly-generated
    EPF will be archived.
    Default: 'output_file.txt'.

    Additional Parameters:
    Multiple auxiliary parameters facilitate tailored modeling. They include
    specific values, boundary constraints,
    and prior types for various characteristics such as the planet's radius,
    its orbital period, transit timing, inclination, among others.

    Returns:
    An EPF written to the prescribed file_path.
    This file enumerates transit/eclipse parameters for the featured exoplanet.
    It lists values, parameter flexibility (whether they are fixed or free),
    parameters for priors, and the types of these priors.
    Dependencies:
    External data: This function hinges on data pulled from the ExoMAST
    platform.

    Notes:
    The function generates an EPF with some free parameters.
    Optimized values from the "best" dictionary, containing optimized ECF and
    EPF inputs, are used for the parameter values
    For some parameters, if certain values are not available, default values or
    calculations are used to fill in gaps.
    Setting bounds for parameters, like eccentricity or inclination, ensures
    physical plausibility in the modeling process.

    Author:
    Reza Ashtari
    Date:
    08/22/2023
    """

    # Defaulting to values from the `best` dictionary if None
    if rp_val is None:
        rp_val = best['rp']
    if per_val is None:
        per_val = best['per']
    if t0_val is None:
        t0_val = best['t0']
    if inc_val is None:
        inc_val = best['inc']
    if a_val is None:
        a_val = best['a']
    if ecc_val is None:
        ecc_val = best['ecc']
    if w_val is None:
        w_val = best['omega']
    if u1_val is None:
        u1_val = best['u1']
    if u2_val is None:
        u2_val = best['u2']
    if c0_val is None:
        c0_val = best['c0']
    if c1_val is None:
        c1_val = best['c1']
    if scatter_mult_val is None:
        scatter_mult_val = best['scatter_mult']

    # Overwrite EPF orbital params using exoMAST data
    # rp_free = 'free'
    # rp_pt = "U"
    # rp_val = planet_data['Rp/Rs']
        
    rp_pp1 = planet_data['Rp/Rs'] - (planet_data['Rp_lower']
                                     / planet_data['Rs'])  # Lower Bound
    rp_pp2 = planet_data['Rp/Rs'] + (planet_data['Rp_upper']
                                     / planet_data['Rs'])  # Upper Bound

    # if planet_data['canonical_name'] == 'HD 86226 c':
    #     rp_pp1 = 0.01942
    #     rp_pp2 = 0.020961
    # if planet_data['canonical_name'] == 'HD-86226 c':
    #     rp_pp1 = 0.01942
    #     rp_pp2 = 0.020961

    # rp_pp1 = 0  # Lower Bound
    # rp_pp2 = 1  # Upper Bound

    # per_free = 'fixed'
    # per_pt = "U"
    # per_val = planet_data['orbital_period']
    per_pp1 = planet_data['orbital_period']
    - planet_data['orbital_period_lower']  # Lower Bound
    per_pp2 = planet_data['orbital_period']
    + planet_data['orbital_period_upper']  # Upper Bound

    # t0_free = 'free'
    # t0_pt = "N"
    # t0_val = planet_data['transit_time']
    t0_pp1 = planet_data['transit_time']  # Mean
    t0_pp2 = planet_data['transit_time_upper']  # Standard Deviation

    # inc_free = 'fixed'
    # inc_pt = "N"
    # inc_val = planet_data['inclination']
    inc_pp1 = planet_data['inclination']  # Mean
    inc_pp2 = planet_data['inclination_upper']  # Standard Deviation

    # a_free = 'fixed'
    # a_pt = "N"
    # a_val = planet_data['a/Rs']
    a_pp1 = planet_data['a/Rs']  # Mean
    a_pp2 = planet_data['a/Rs_upper']  # Standard Deviation
    if (a_pp2 is None):
        a_pp2 = planet_data['a/Rs_lower']
        if (a_pp2 is None):
            a_pp2 = a_val * 0.1

    # ecc_free = 'fixed'
    # ecc_pt = "U"
    # ecc_val = planet_data['eccentricity']
    # if(ecc_val is None):
    #     ecc_val = 0.0
    # ecc_pp1 = 0.0 # Lower Bound
    # ecc_pp2 = 1 # Upper Bound

    # w_free = 'fixed'
    # w_pt = "U"
    # w_val = planet_data['omega']
    # if(w_val is None):
    #     w_val = 90.0
    # w_pp1 = 0.0 # Lower Bound
    # w_pp2 = 359.9 # Upper Bound

    content = \
        f'''
        # Stage 5 Fit Parameters Documentation:
        # https://eurekadocs.readthedocs.io/en/latest/ecf.html#stage-5-fit-parameters
        # Name  Value    Free?   PriorPar1   PriorPar2   PriorType
        # "Free?" can be free, fixed, white_free, white_fixed, shared, or
        # independent
        # PriorType can be U (Uniform), LU (Log Uniform), or N (Normal).
        # If U/LU, PriorPar1 and PriorPar2 represent upper and lower limits of
        # the parameter/log(the parameter).
        # If N, PriorPar1 is the mean and PriorPar2 is the standard deviation
        # of a Gaussian prior.
        #-------------------------------------------------------------------------------------------------------
        #
        # ------------------
        # ** Transit/eclipse parameters **
        # ------------------
        rp           {rp_val}       '{rp_free}'   {rp_pp1}     {rp_pp2}      {rp_pt}
        per          {per_val}      '{per_free}'  {per_pp1}    {per_pp2}     {per_pt}
        t0           {t0_val}       '{t0_free}'   {t0_pp1}     {t0_pp2}      {t0_pt}
        time_offset  2400000.5      'independent'
        inc          {inc_val}      '{inc_free}'  {inc_pp1}    {inc_pp2}     {inc_pt}
        a            {a_val}        '{a_free}'    {a_pp1}      {a_pp2}       {a_pt}
        ecc          {ecc_val}      '{ecc_free}'  {ecc_pp1}    {ecc_pp2}     {ecc_pt}
        w            {w_val}        '{w_free}'    {w_pp1}      {w_pp2}       {w_pt}
        limb_dark    '{ld_type}'    '{ld_free}'
        u1           {u1_val}       '{u1_free}'   {u1_pp1}     {u1_pp2}      {u1_pt}
        u2           {u2_val}       '{u2_free}'   {u2_pp1}     {u2_pp2}      {u2_pt}
        c0           {c0_val}       '{c0_free}'   {c0_pp1}     {c0_pp2}      {c0_pt}
        c1           {c1_val}       '{c1_free}'   {c1_pp1}     {c1_pp2}      {c1_pt}
        scatter_mult {scatter_mult_val} '{scatter_mult_free}' {scatter_mult_pp1} {scatter_mult_pp2} {scatter_mult_pt}

        '''

    if planet_data['canonical_name'] == 'HD 86226 c':        
        content = \
            f'''
            # Stage 5 Fit Parameters Documentation:
            # https://eurekadocs.readthedocs.io/en/latest/ecf.html#stage-5-fit-parameters
            # Name  Value    Free?   PriorPar1   PriorPar2   PriorType
            # "Free?" can be free, fixed, white_free, white_fixed, shared, or
            # independent
            # PriorType can be U (Uniform), LU (Log Uniform), or N (Normal).
            # If U/LU, PriorPar1 and PriorPar2 represent upper and lower limits of
            # the parameter/log(the parameter).
            # If N, PriorPar1 is the mean and PriorPar2 is the standard deviation
            # of a Gaussian prior.
            #-------------------------------------------------------------------------------------------------------
            #
            # ------------------
            # ** Transit/eclipse parameters **
            # ------------------
            rp           0.01962646833343065     'free'   0     1      U
            per          3.984651184041696  'fixed'  3.9846654    0.0000092     N
            t0           58698.15683206442      'fixed'   58698.1559     0.00065      N
            time_offset  2400000.5  'independent'
            inc          86.5      'fixed'  86.5    0.16     N
            a            10.11        'fixed'    7.1      1.0034833931292224       N
            ecc          0.075      'fixed'  0    1     U
            w            196.0      'fixed'    0      360       U
            limb_dark    'quadratic'    'independent'
            u1           0.15651933602854776     'fixed'   0.15651933602854776     0.01      N
            u2           0.236344988380854      'fixed'   0.2367380655160314     0.01      N
            c0           0.9985409602491301       'free'   1     0.01      N
            c1           -0.015325138271693704       'free'   -1     1      U
            scatter_mult 1.19999999     'fixed'   0.8 1.2 U 

            '''

        # xpos         {xpos_val}     '{xpos_free}' {xpos_pp1}   {xpos_pp2}    {xpos_pt}
        # ypos         {ypos_val}     '{ypos_free}' {ypos_pp1}   {ypos_pp2}    {ypos_pt}

    with open(file_path, 'w') as f:
        f.write(content)
