# $Author: patricio $
# $Revision: 304 $
# $Date: 2010-07-13 11:36:20 -0400 (Tue, 13 Jul 2010) $
# $HeadURL: file:///home/esp01/svn/code/python/branches/patricio/photpipe/lib/err_fasym_c.py $
# $Id: err_fasym_c.py 304 2010-07-13 15:36:20Z patricio $

import numpy as np
from scipy import optimize
from numpy import sum
#from uwv import var
from scipy.ndimage.interpolation import map_coordinates
#from make_asym import make_asym


def gaussian(height, center_x, center_y, width_x, width_y,offset):
    """
      Returns a gaussian function with the given parameters.
    """
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)+offset

def moments(data):
    """
      Returns (height, x, y, width_x, width_y,offset)
      The gaussian parameters of a 2D distribution by calculating its
      moments.
    """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    #col = data[:, int(y)]
    #width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    #row = data[int(x), :]
    #width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    firstq = np.median(data[data<np.median(data)])
    thirdq   = np.median(data[data>np.median(data)])
    offset = np.median(data[np.where(np.bitwise_and(data>firstq,data<thirdq))])
    places = np.where((data-offset) > 4*np.std(data[np.where(np.bitwise_and(data>firstq,data<thirdq))]))
    width_y = np.std(places[0])
    width_x = np.std(places[1])
    # These if statements take into account there might only be one significant point above the background

    # when that is the case it is assumend the width of the gaussian must be smaller than one pixel
    if width_y == 0.0:
        width_y = 0.5
    if width_x == 0.0:
        width_x = 0.5

    height -= offset
    return height, x, y, width_x, width_y,offset


def fitgaussian(data, weights=False):
    """
      Returns (height, x, y, width_x, width_y) the gaussian parameters
      of a 2D distribution found by a fit.  Weights must be the same
      size as the data, but every point contains the value of the
      weight of the pixel
    """
    if type(weights) == type(False):
        weights = np.ones(data.shape,dtype=float)
    elif weights.dtype != np.dtype('float'):
        weights = np.array(weights,dtype(float))
    params = moments(data)
    errorfunction = lambda p: np.ravel((gaussian(*p)(*np.indices(data.shape)) -
                                 data)*weights)
    p, success = optimize.leastsq(errorfunction, params)
    return p


def col(data, weights=False):
    if type(weights) == type(False):
        weights = np.ones(data.shape,dtype=float)
    elif weights.dtype != np.dtype('float'):
        weights = np.array(weights,dtype(float))
    ny,nx = np.indices(data.shape)
    return [sum(weights*ny*data)/sum(weights*data),\
            sum(weights*nx*data)/sum(weights*data)]

#def make_asym(data,dis,weights):
    # This function generates the asym value associated a given frame and distance array (aka raidail profile)
    # It is intended to be used entirely internally and never called from the outside
    # it implements sum of the variance squared at a given radius times the number of points at that radius
 #   assert(data.shape == dis.shape)
 #   temp = map(lambda r: (var(data[dis == r],\
 #                                      weights[dis ==r])),dis.flatten())
    #temp = []
    #for r in dis.flatten():
     #   temp.append(var(data[dis == r],weights[dis==r]))
 #   temp = np.array(temp)
    #uncoment these lines if you plan on using weights
    #if np.sum(weights) == weights.size:
 #   return np.sum(temp)
    #else:
     #   return np.sum(temp**2)


def actr(data, yxguess, asym_rad=8, asym_size=5, maxcounts=2,
         method='gaus', half_pix=False, resize=False, weights=False):
    """

    Calculate the center of an input array by first switching the
    array into asymmetry space and finding the minimum

    This function takes in an input two dimentional array `data`, with
    a certain distribution of values, such as the flux from a
    star. The center of this distribution of values is based on the
    idea that the center will be the point of minimum asymmetry. To
    convert to asymmetry space, the asymmetry of a radial profile
    about a particular pixel is calculated according to
    sum(var(r)*Npts(r)), the sum of the variance about a particular
    radius times the number of points at that radius. The outter
    radius of consideration is set by `asym_rad`. The number of points
    that are converted to asymmetry space is governed by `asym_size`
    producing a shape `asym_size`*2+1 by `asym_size`*2+1. This
    asymmetry space is recalculated and moved succesively until the
    point of minimum asymmetry is in the center of the array or
    `maxcounts` is reached. Traditional Gaussian or center of light
    centering is then used in the asymmetry space to find the
    sub-pixel point of minimum asymmetry.

    Parameters:
    -----------
    data: 2D array
          This is the data to be worked on, the radius of the array
          should at minimum be asym_size times 2. radius being defined
          as the cut size, i.e.:
          data[yxguess[0]-rad:yxguess[0]+rad+1,yxguess[1]-rad:yxguess[1]+rad+1]
          a recomendation would be asym_size times 3 or 4, to account
          for the searching of the algorithm

    yxguess:   tuple
               This contains a tuple, or 1x2 array, that contains the
               guess of the center, this is used as the starting
               location of the walk

    asym_rad:  int
               The integer used to define the span of the radial
               profile used in the asym calculation

    asym_size: int
               This sets the radius of the asym space that is used to
               determine the center. The larger the radius of the star
               in flux space, the larger this variable should be

    maxcounts: int
               An int to set the number of times the routine is
               allowed to walk trying to put the point of minimum
               asymmetry in the center of the array

    method:    string
               Must be 'gaus' to used gaussian for sub pixel, or 'col'
               for center of light sub pixel

    half_pix:  Bool
               Default false, if set to true asymmetry will be
               calculated for all the half pixel location between the
               integers contained in the asym array. This may be more
               accurate, but much slower.  If set, it is recommended
               to use a larger value for `asym_rad`.

    resize:    float
               Though by default False, set to a float which will be
               the factor the asym the array is resized before
               centering. Reccomend scale factors 5 or
               below. Resizeing introduces a bit of error in the
               center by itself, but may be combensated by the gains
               in percision.  test with care. This WILL slow the
               function down noticably.  weights: array This is an
               array the same size as the input data, each point
               contains the weighting that each point should
               recive. low number is less weight, should be type
               float, if none is given the weights are set to one.

    Returns:
    --------
    numpy array:     A 1x2 array containing the found center of the data

    Raises:
    -------
    Possibly an assertaion error, it the size of the radial profile is
    different than a given view of the data.  This is most likely due
    to a boundary issue, aka the routine is trying to move out of the
    boundary of the input data. This can be caused by incorrect sizes
    for asym_rad & asym_size, or becuase the routine is trying to walk
    off the edge of data searching for the point of minimum
    asymmetry. If this happens, try reducing asym_size, or maxcounts
    """
    #initialize the boolean that determines if there are weights present or not
    #This is used in the make asym array to square the variance, to provide
    #larger contrast when using weights
    w_truth = 1
    #create the weights array if one is not passed in, and set w_truth to 0
    if type(weights) == type(False):
        weights = np.ones(data.shape,dtype=float)
        w_truth = 0
    elif weights.dtype != np.dtype('float'):
        #cast to a float if it is not already float
        weights = np.array(weights,dtype=float)
    if data.dtype != 'float64':
        data = data.astype('float64')

    x_guess = yxguess[1]
    y_guess = yxguess[0]

    #create the array indexes
    ny,nx = np.indices((data.shape))

    #create the indices for the radial profile
    ryind, rxind = np.indices((asym_rad*2+1,asym_rad*2+1))

    #for the course pixel asym location we will reuse the same radial profile, generate that now
    dis = ((ryind-asym_rad)**2+(rxind-asym_rad)**2)**0.5

    #make the view for the positions to calculate an asymmetry value, this may be unneeded look at removing
    #this for optimization, save the shape for later shape restoration

    suby = ny[y_guess - asym_size:y_guess +asym_size+1,x_guess - asym_size:x_guess + asym_size+1]
    shape_save = suby.shape
    suby = suby.flatten()
    subx = nx[y_guess - asym_size:y_guess +asym_size+1,x_guess - asym_size:x_guess + asym_size+1].flatten()

    #set up the range statement, as to not recreate it every loop, same with len
    len_y = len(suby)
    itterator = np.arange(len_y)
    ones_len  = np.ones(len_y)
    middle    = (len_y - 1)/2.

    #set a counter for the number of times that the routine has moved pixel space
    counter = 0

    #define the lambda function used to generate the views outside of the loop
    view_maker = lambda frame,y_lamb,x_lamb,rng: frame[y_lamb-rng:y_lamb+rng+1,x_lamb-rng:x_lamb+rng+1]


    # start a while loop, to be broken when the maximum number of steps is reached, or when the minimum
    #asymmetry value is in the center of the array
    while counter <= maxcounts:
        # This will make heave use of python generators ie (x for x in range(10)), does not make the list
        # all at once, but instead creates something that will create the each element when itterated over
        # this saves memory and time, an generator needs to be recreated after each use however. The map
        # function will be used alot as well. This is like a for loop but one which runs faster.

        # create a generator for the views ahead of time that will be needed
        views = (view_maker(data,suby[i],subx[i],asym_rad) for i in itterator)

        # create a generator for the view on the weights ahead of time
        lb_views = (view_maker(weights,suby[i],subx[i],asym_rad) for i in itterator)

        # create a generator to duplicate the distance array the required number of times for the map function
        dis_dup = (dis *i for i in ones_len)

        #greate a generator duplicate for the state of w_truth
        w_truth_dup = (w_truth*1 for i in ones_len)

        # now create the actual asym array
        asym = np.array(map(make_asym,views,dis_dup,lb_views,w_truth_dup))

        # need to find out if the minimum is in the center of the array if it is, break

        if asym.argmin() == middle:
            break

        # if not, move the array index locations and itterate counter, the while loop then repeats, delete variable
        # to make the garbage collector work less hard
        else:
            suby    += (suby[asym.argmin()]-y_guess)
            subx    += (subx[asym.argmin()]-x_guess)
            counter += 1
            del views,dis_dup,asym

    if counter <= maxcounts:
        # now we must find the sub pixel persision and related options
        # First reshape the array to the saved shape
        asym = asym.reshape(shape_save)



        # set a constant used in the center calculation, it is one if half_pix is unset, it becomes 2 if it is set
        divisor = 1.0

        #set a center offset, this is zero normally, but is set to a non zero value if the array is zoomed.
        zoffset = 0

        # This gives the option to include the half pixel values
        if half_pix:
            # First create the new container array for the asymmetry, double the dimentions of the orrigional
            new_asym = np.zeros(np.array(asym.shape)*2,dtype=float)

            # Next assign the old asym array to the be the top left corners in the new 4x4 pixels that represent
            # one orrigional one
            new_asym[::2,::2] = asym.copy()

            # Now create all of the edges, this represnts the upper right box in the 4x4, or the edges between
            # pixels going to the right.

            # make the new distance array
            dis = ((ryind-asym_rad)**2+(rxind-(asym_rad+0.5))**2)**0.5

            # create the generator for the views
            views = (view_maker(data,suby[i],subx[i],asym_rad) for i in itterator)

            # create the generator to duplicate the distance array
            dis_dup = (dis *i for i in ones_len)

            # generate the asymmetry values and save them
            new_asym[::2,1::2] = np.array(map(make_asym,views,dis_dup)).reshape(new_asym[::2,1::2].shape)

            # create all of the bottoms, representing the lower left box in 4x4, the edge between pixels going
            # down

            # make the new distance array
            dis = ((ryind-(asym_rad+0.5))**2+(rxind-asym_rad)**2)**0.5

            # create the generator for the views
            views = (view_maker(data,suby[i],subx[i],asym_rad) for i in itterator)

            # create the generator to duplicate the distance array
            dis_dup = (dis *i for i in ones_len)

            # generate the asymmetry values and save them
            new_asym[1::2,::2] = np.array(map(make_asym,views,dis_dup)).reshape(new_asym[1::2,::2].shape)

            # create all of the corners, like above

            # make the new distance array
            dis = ((ryind-(asym_rad+0.5))**2+(rxind-(asym_rad+0.5))**2)**0.5

            # create the generator for the views
            views = (view_maker(data,suby[i],subx[i],asym_rad) for i in itterator)

            # create the generator to duplicate the distance array
            dis_dup = (dis *i for i in ones_len)

            # generate the asymmetry values and save them
            new_asym[1::2,1::2] = np.array(map(make_asym,views,dis_dup)).reshape(new_asym[1::2,1::2].shape)

            # set the new_asym to the old one
            asym = new_asym

            # set the divisor varible, representing that every pixel now represents a half step
            divisor = 2.0


        # next invert the asym space so the minimum point is now the maximum
        asym = -1.0*asym


        # insert any additional steps that would modify the asym array, such as interpolating, or adding
        # more asym values.
        if resize != False:
            fact = 1/float(resize)
            asym = map_coordinates(asym,np.mgrid[0:asym.shape[1]-1+fact:fact,0:asym.shape[1]-1+fact:fact])
            divisor = resize

        #now find the sub pixel position using the given method
        if method == 'col':
            return ((np.array(col(asym)))/divisor - asym_size)+np.array((suby[middle],subx[middle]),dtype=float)
        if method == 'gaus':
            return ((np.array(fitgaussian(asym)[[1,2]]))/divisor -asym_size)+np.array((suby[middle],subx[middle]),dtype=float)

    else:
        # return this as an error code if the function waled more times than allowed, ie not finding a center
        return np.array([-1.0,-1.0])

def tactr(data,yxguess,asym_rad=8,asym_size=5,maxcounts=2,method='gaus',half_pix=False,resize=False,weights=False):
    try:
        temp = actr(data,yxguess,asym_rad,asym_size,maxcounts,method,half_pix,resize,weights)
    except:
        temp = [yxguess[0]+3,yxguess[1]+3]
    stop = 0
    subs = -1
    subr = 1

    while (np.abs(temp[0]-yxguess[0]) > 1.5 or np.abs(temp[1]-yxguess[1]) >1.5):
        asym_size += subs
        asym_rad  += subr
        try:
            temp = actr(data,yxguess,asym_rad,asym_size,maxcounts,method,half_pix,resize,weights)
        except:
            temp = [yxguess[0]+3,yxguess[1]+3]
        stop +=1
        if stop == 3:
            asym_size -= 4*subs
            asym_rad  -= 4*subr
            subs = 1
            subr = -1
        if stop == 6:
            temp = yxguess
            break
    return temp
