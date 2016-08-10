# cython: profile=True
'''Code to downsample an array by binning with a non-integer ratio of old 
size to new size.


Alan Kastengren, XSD, APS

Started: August 26, 2015
'''
import numpy as np
cimport numpy as np
float_DTYPE = np.float64
int_DTYPE = np.int
ctypedef np.int_t int_DTYPE_t
ctypedef np.float_t float_DTYPE_t
import time

def fbin_array(np.ndarray[float_DTYPE_t,ndim=1] input_array, double delta_ratio):
    '''Bins an array by ratio delta_ratio.
    The idea is that for the input array, each point has a bin in the new
    array to which it is supposed to go.  Breakpoints occur between the points
    in the old array.  The integration is set up so that the first breakpoint is
    coincident with the first point (sampling occurs exactly at t = 0).
    Linear interpolation is used to find value at the breakpoint.
    Trapezoidal rule integration is used to sum values into each bin.
    
    Inputs:
    input_array: numpy array with old values
    delta_ratio: # of points from old array for each new array ValueError
    Output:
    output_array: numpy array with binned values
    '''
    #Make an array saying which new point each point should go to
    cdef np.ndarray[int_DTYPE_t,ndim=1] mapping_array = (np.arange(input_array.shape[0],dtype=float_DTYPE) / delta_ratio).astype(int_DTYPE)
    cdef long i
    #Set the size of the output array
    cdef np.ndarray[np.float_t,ndim=1] output_array = np.zeros(mapping_array[-1] - 1,dtype=float_DTYPE)
    #Define floats we will be using
    cdef double fraction = 0.0
    cdef double alt_fraction = 1.0
    #Loop across the input array
    cdef np.ndarray[int_DTYPE_t,ndim=1] loop_array = np.arange(input_array.shape[0],dtype=int_DTYPE)
    for i in loop_array:
        #Deal with the very first point
        if i == 0:
            output_array[0] += input_array[0]/2.0
            continue
        #Make sure we don't overrun in the input array
        if i == input_array.shape[0] - 2:
            break
        #If the next breakpoint isn't between this point and the next one ...
        if mapping_array[i+1] == mapping_array[i]:
            #If the breakpoint wasn't just passed, add the full value: trap integration
            if mapping_array[i-1] == mapping_array[i]:
                output_array[mapping_array[i]] += input_array[i]
            #If we just past breakpoint, add only half of value: trap integration
            else:
                output_array[mapping_array[i]] += input_array[i] / 2.0
        #Breakpoint is between this point and the next one
        else:
            #How far away from current point is the breakpoint
            fraction = mapping_array[i+1] * delta_ratio - i
            #Add in the 1/2 of the current value for the region before the current value,
            #but only if the current value isn't between two breakpoints: trap integration
            if mapping_array[i-1] == mapping_array[i]: 
                output_array[mapping_array[i]] += input_array[i] / 2.0
            #Add the area between current value and breakpoint, by linear interpolation
#             print output_array[mapping_array[i]],fraction / 2.0 * (fraction * input_array[i+1] + (2.0 - fraction) * input_array[i])
            output_array[mapping_array[i]] += fraction / 2.0 * (fraction * input_array[i+1] + (2.0 - fraction) * input_array[i])
            #If we would go past the end of the output array, break
            if mapping_array[i+1] >= output_array.size:
                break
            #Add the area between breakpoint and next point by linear interpolation
            alt_fraction = 1.0 - fraction
            output_array[mapping_array[i+1]] += alt_fraction / 2.0 * (alt_fraction * input_array[i] + (2.0 - alt_fraction) * input_array[i+1])
#             print output_array[mapping_array[i+1]]
    return output_array

def ffind_breakpoints_peaked(np.ndarray[float_DTYPE_t,ndim=1]input_array,int pulse_points,int search_region=5):
    """Find the breakpoints between peaks in a periodic peaked signal.
    Inputs:
    input_array: float array holding the input data values
    pulse_points: the nominal number of points between pulses
    search_region: number of points over which to search for each peak
    """
    #Figure out the stride we will use between peaks
    cdef int stride = int(np.floor(pulse_points))
    print "Stride length = ", stride, " points."
    #Make an empty array to hold the peak positions.  Pre-allocate to attempt to speed up
    cdef long peak_array_size = int(float(input_array.shape[0]) * 1.05) / pulse_points
    cdef np.ndarray[int_DTYPE_t,ndim=1] peak_positions = np.zeros(peak_array_size,dtype=int_DTYPE)
    #Set starting position
    cdef long num_peaks = 1
    peak_positions[0] = np.argmax(input_array[:stride+1])
    #Reduce stride by half of search region size to make math simpler during actual striding
    stride -= search_region//2
    cdef long current_position = peak_positions[0] + stride
    #Keep striding until we are within two strides of the end
    while current_position + 2 * stride < input_array.shape[0]:
        #Find the maximum value in the search region around nominal peak
        current_position = current_position + cython_argmax(input_array[current_position:current_position+search_region])
        #Add this peak position to the peak positions list and increment counter
        peak_positions[num_peaks] = current_position
        num_peaks += 1
        #Stride
        current_position += stride
        #Print about progress
#         if not(num_peaks%500000):
    return peak_positions[:num_peaks]

cdef cython_argmax(np.ndarray[float_DTYPE_t,ndim=1] input_array):
    '''Attempt to write an argmax function.
    '''
    cdef int i
    cdef double max_value 
    cdef int max_loc
    i = 1
    max_value = input_array[0]
    max_loc = 0
    while i < input_array.shape[0]:
        if input_array[i] > max_value:
            max_loc = i 
            max_value = input_array[i]
        i += 1
    return max_loc