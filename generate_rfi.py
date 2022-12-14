import numpy as np
from math import *
import matplotlib.pyplot as plt



def create_struct(image_shape, number=3, width=0.3, period=1./2., amp=[]):
    """
    Create small periodic rfi structure.

    Parameters
    ----------
    image_shape : tuple(int)
        Dimension of the image that will be use to create the rfi
    number : int, optional
        Number of rfi to generate
    width : float, optional
        Percent of the image width used to create the rfi structure
    period : float, optional
        Spatial period of the rfi
    amp : array(float), optional
        Noise level of the structure 

    Returns
    -------

    rfi_map : array
        The binary structures of RFI created
    amp(rfi_map) : array
        The RFI structures with noise level
    """
    rfi_map = np.zeros(image_shape)
    for i in range(number):
        #compute position of the left top corner of the rfi
        size_y = np.random.randint(3,int(image_shape[0]*width))
        size_x = np.random.randint(3,int(image_shape[1]*4.*width)) 
        pos_y = np.random.randint(image_shape[0]-size_y)
        pos_x = np.random.randint(image_shape[1]-size_x)

        #create the rfi square
        tmp_rfi = np.zeros((size_y,size_x))
        step = int(period+1)
        tmp_rfi[...,::step] = 1.
        #add variability to that rfi by deleting some pixel 
        tmp = np.random.randint(0,2,tmp_rfi.shape)
        #put the rfi structure into the map
        rfi_map[pos_y:pos_y+size_y, pos_x:pos_x+size_x] = np.multiply(tmp, tmp_rfi)
    return rfi_map, add_amp(rfi_map, amp)


def create_spot(rfi_map, coeff = 15, width=1., amp=[]):
    """
    Given a rfi_map add some random spot.

    Parameters
    ----------

    rfi_map : array
        rfi_map to add some random hot spot, must be 2D array
    coeff : float, optional
        defines the number of hotpost to add, for a 2D array of shape u,v it is u*v/(coeff**2)
    width : float, optional
        the width of the hotspot
    amp : array(float), optional
        Noise level of the structure

    Returns
    -------

    rfi_ : array
        the new rfi_map with hotspot added (from a copy of the input rfi_map)
    amp(rfi_) : array
        the new rfi map with hotspot and noise level applied
    """
    rfi_ = np.copy(rfi_map)
    #create the spot index 
    index_y = np.random.randint(0, rfi_map.shape[0],size = int(rfi_map.shape[0]*rfi_map.shape[1]/coeff**2))

    index_x = np.random.randint(0, rfi_map.shape[1], size = int(rfi_map.shape[0]*rfi_map.shape[1]/coeff**2))
    for i in range(len(index_y)):
        if np.random.randint(0,2):
            rfi_[index_y[i], index_x[i]:index_x[i]+int(width)]= 1.
        else:
             rfi_[index_y[i]:index_y[i]+int(width), index_x[i]]= 1.
    
    return rfi_, add_amp(rfi_, amp)

def create_hrfi(rfi_map, number = 3, width = 3., amp=[]):
    """
    Create horizontal RFI

    Parameters
    ----------
    rfi_map : array
        rfi_map to add some random horizontal lines, must be 2D array
    number : int, optional
        defines the number of lines to add
    width : float, optional
        the width of the lines
    amp : array(float), optional
        Noise level of the structure

    Returns
    -------

    rfi_ : array
        the new rfi_map with horizontal lines added (from a copy of the input rfi_map)
    amp(rfi_) : array
        the new rfi map with horizontal lines and noise level applied
    """
    rfi_ = np.copy(rfi_map)
    for i in range(number):
        #compute position and the width of the horizontal line 

        size_y = np.random.randint(1,int(width)) 
        pos_y = np.random.randint(rfi_map.shape[0]-size_y)


        #create the rfi line 
        tmp_rfi = np.ones((size_y,rfi_map.shape[1]))

        #add variability to that rfi by deleting some pixel 
        if np.random.randint(0,2) and size_y>1:
            tmp = np.random.randint(0,2,size=(1,rfi_map.shape[1]))
            if np.random.randint(0,2):
                index = 0
            else:
                index = tmp_rfi.shape[0]-1
            tmp_rfi[index,:] = np.abs(np.subtract(tmp, tmp_rfi[index,:]))
        #put the rfi structure into the map
        rfi_[pos_y:pos_y+size_y,...] =  tmp_rfi
    
    return rfi_, add_amp(rfi_, amp)

def create_vrfi(rfi_map, period = 1./3., size_w = 0.2, vsize=0, number = 10, rfi_w=1, amp=[], destr=True):
    """
    Create vertical RFI with periodic pattern

    Parameters
    ----------
    rfi_map : array
        rfi_map to add some random horizontal lines, must be 2D array
    period : float, optional
        spatial period of the vertical pattern
    size_w : float, optional
        defines the maximal size of the periodic pattern, it will be randomly drawn between 1 and shape[0]*size_w (or shape[1]*size_w)
    vsize : int, optional
        change the position and the size of the vertical pattern 
    number : int, optional
        defines the number of lines to add
    rfi_w : float, optional
        the width of the lines
    amp : array(float), optional
        Noise level of the structure

    Returns
    -------

    rfi_ : array
        the new rfi_map with horizontal lines added (from a copy of the input rfi_map)
    amp(rfi_) : array
        the new rfi map with horizontal lines and noise level applied
    """
    rfi_ = np.copy(rfi_map)
    for i in range(number):
        #compute position of the left top corner of the rfi
        size_y = np.random.randint(1,int(rfi_map.shape[0]*size_w))
        size_x = np.random.randint(1,int(rfi_map.shape[1]*size_w))
        pos_y = np.random.randint(rfi_map.shape[0]-size_y)
        pos_x = np.random.randint(rfi_map.shape[1]-size_x)

        if vsize:
            size_y = rfi_map.shape[0]
            pos_y  = 0
            size_x = np.random.randint(1,4)
            pos_x = np.random.randint(rfi_map.shape[1]-size_x)
        #create the rfi square
        tmp_rfi = np.zeros((size_y,size_x))
        step = int(rfi_w/period)
        for i in range(0,size_x,rfi_w+1):
            tmp_rfi[...,i*int(rfi_w):(1+i)*int(rfi_w)]=1
        #add variability to that rfi by deleting some pixel 
        #put the rfi structure into the map
        rfi_[pos_y:pos_y+size_y, pos_x:pos_x+size_x] = np.logical_or(tmp_rfi,rfi_[pos_y:pos_y+size_y, pos_x:pos_x+size_x])
    if destr:
       return destruction(rfi_), add_amp(destruction(rfi_), amp)
    return rfi_, add_amp(rfi_,amp)

def background(image_shape, nb_band, bw, amp=[]):
    """
    Create a background noise.

    Parameters
    ----------
    image_shape : tuple(int)
        dimension of the image that will be use to create the rfi
    nb_band : int
        number of horizontal noise bands
    bw : int 
        width of the horizontal noise bands
    amp : array, optional
        contains mean and std of the bg noise level, should be at least 2D array

    Returns
    -------

    bg_map : array
        the background map containing the background noise
    """
    ##add the way to cover the full observation (not only some band)  
    bg_map_  = np.zeros((image_shape))
    #create the rfi band
    #apply a random amplitude between 0 and 1 for the band

    for i in range(int(nb_band)):
        nlvl = np.random.rand()
        pos_y = np.random.randint(image_shape[0]-bw)
        tmp = np.repeat([0.,1.,0.], int(bw/3))
        win = np.hanning(int(bw/3))
        tmp_bg = np.convolve(tmp,win, mode='same')
        tmp_bg /= np.max(tmp_bg)
        bg_map_[pos_y:pos_y+len(tmp_bg),:] = nlvl*tmp_bg.T[...,np.newaxis]*np.ones((len(tmp_bg), image_shape[1]))
    # create the 3 channels images
    if not isinstance(amp,np.ndarray):
        amp = np.ones((3,2))
    bg_map =    np.zeros((image_shape[0],image_shape[1],3))
    bg_map[...,0] = np.abs(bg_map_ * ( 2.* amp[0,0] *np.random.rand() + (amp[0,0] - amp[0,1])))
    bg_map[...,1] = np.abs(bg_map_ * ( 2.* amp[1,0] *np.random.rand() + (amp[1,0] - amp[1,1])))
    bg_map[...,2] = np.abs(bg_map_ * ( 2.* amp[2,0] *np.random.rand() + (amp[2,0] - amp[2,1])))

    return bg_map



def add_amp(rfi_struc, amp_values):
    """
        Add noise levels to a binary mask of RFI

    Parameters
    ----------

    rfi_struct: array
        a binary map of the RFI to modulate by noise level
    amp_values : array
        a 2D array of shape 3x2 containing the values of noise to apply

    Returns
    -------

    amp : array
        contains the rfi_struct with noise amplitudes
    """
    if not isinstance(amp_values,np.ndarray):
        amp_values = np.ones((3,2))
    tmp = np.random.random()
    amp = np.zeros((rfi_struc.shape[0], rfi_struc.shape[1],3))
    amp[rfi_struc==1.,0] = tmp * (amp_values[0,0] + amp_values[0,1]) 
    amp[rfi_struc==1.,1] = tmp * (amp_values[1,0] + amp_values[1,1]) 
    amp[rfi_struc==1.,2] = tmp * (amp_values[2,0] + amp_values[2,1]) 
    return amp


def destruction(rfi_map):
    """
    IT WOULD BE A SHAME IF SOMEONE DESTROYED YOUR BEAUTIFUL RFI.
    Add flaws to the perfectly generated RFI by mutipliying with random binary matrices.

    Parameters
    ----------

    rfi_map : array
        2D binary array that need to be a little bit destroyed. 
    
    
    Returns
    -------

    rfi_map : array
        imperfect RFI binary maps.
    """
    tmp = np.random.random(rfi_map.shape)
    tmp += np.random.random(rfi_map.shape)
    tmp += np.random.random(rfi_map.shape)
    tmp += np.random.random(rfi_map.shape)
    tmp[tmp > 1.] = 1.
    tmp[tmp < 1.] = 0.
    return np.logical_and(tmp, rfi_map)


 

def create_RFIs(rfi_shape, rfi_amp=[], bg_amp=[], nb_h=1, nb_v=1, nb_s=15, size_w=0.2, hw=3, period=2, rfi_w=1, nb_vl=3):
    """
    Create RFI map with vertical rfi, hot spot and horizontal ones.

    Parameters
    ----------
    
    rfi_shape : tuple(int)
        shape of the rfi map we want to create
    rfi_amp   : array(tuple(float,float)), optional
        size of (3,2) containing mean and sigma of rfi amplitude to use
    bg_amp    : array(tuple(float,float)), optional
        size of (3,2) containing mean and sigma of background amplitude to use
    nb_h      : int, optional
        number of horizontal RFI to create
    nb_v      : int, optional
        number of vertical RFI to create
    nb_s      : int, optional
        ratio to create hot spot. It will create rfi_shape[0]xrfi_shape[1]/nb_s**2 hot spots. 
    size_w    : float, optional
        ercent of the y or x dimension to use as a maximum width/height for rfi structure
    hw        : int, optional
        maximum width of the horizontal RFI
    period    : int, optional
        spatial period of the periodic RFI 
    rfi_w     : int, optional
        width of the vertical RFI
    nb_vl     : int, optional
        number of vertical line to put inside the rfi
                
    """    
    nb_band     = int(rfi_shape[0]/3) 
    bg_bw       = 50
    #create the background noise
    bg          =   background(rfi_shape, nb_band, bg_bw, bg_amp)
    #create the rfi_map canvas 
    rfi, amp    =   create_struct(rfi_shape, number=nb_h, width=size_w, period=period, amp=rfi_amp)
       
    
    vrfi, vamp  =   create_vrfi(rfi, number=nb_v, size_w=size_w, period=period, rfi_w=rfi_w, amp=rfi_amp)
    mask        =   np.logical_or(vrfi,rfi)
    vline, vlamp=   create_vrfi(rfi, number=nb_vl, size_w=0.2, period=1, rfi_w=1, amp=rfi_amp, vsize=rfi_shape[0])
    mask        =   np.logical_or(mask,vline)
    srfi, samp  =   create_spot(rfi, coeff=nb_s, width=2,amp=rfi_amp)
    mask        =   np.logical_or(mask,srfi)
    hrfi, hamp  =   create_hrfi(rfi, number=nb_h, width=hw,amp=rfi_amp)
    mask        =   np.logical_or(mask,hrfi)

    #combinining with different level of amplitude (bc some RFI types are stronger than other)
    #and make sure 
    rfi_map            =   bg / 4.  + (vamp/3. + hamp/3. + amp/2. + samp + vlamp/4.)
    rfi_map[...,0]    +=   np.min(rfi_map[...,0])
    rfi_map[...,1]    +=   np.min(rfi_map[...,1])
    rfi_map[...,2]    +=   np.min(rfi_map[...,2])
    return rfi_map, mask
    
