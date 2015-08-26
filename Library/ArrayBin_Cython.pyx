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

def fbin_array(np.ndarray[float_DTYPE_t,ndim=1] input_array, float delta_ratio):
    '''Bins an array by ration delta_ratio
    Inputs:
    input_array: numpy array with old values
    delta_ratio: # of points from old array for each new array ValueError
    Output:
    output_array: numpy array with binned values
    '''
    #Make sure delta_ratio is a float
    print "Delta Ratio = " + str(delta_ratio)
#     print input_array.shape
    #Make an array saying which new point each point should go to
    cdef np.ndarray[float_DTYPE_t,ndim=1] temp_array = np.arange(input_array.shape[0]) / delta_ratio
    cdef np.ndarray[int_DTYPE_t,ndim=1] mapping_array = np.zeros(temp_array.size,dtype=int_DTYPE)
    cdef i
    for i in range(temp_array.size):
        mapping_array[i] = int(temp_array[i])
    #Set the size of the output array
    cdef np.ndarray[np.float_t,ndim=1] output_array = np.zeros(mapping_array[-1] - 1)
    cdef fraction = 0.0
    cdef alt_fraction = 1.0
#     plt.plot(np.arange(input_array.shape[0]),mapping_array)
#     plt.show()
    for i in np.arange(input_array.shape[0]):
        #Deal with the very first point
        if i == 0:
            output_array[0] += input_array[0]/2.0
            continue
        #Make sure we don't overrun in the input array
        if i == input_array.shape[0] - 2:
            break
        #If the next breakpoint isn't between this point and the next one
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
#             print i, delta_ratio, mapping_array[i], fraction
            #Add in the 1/2 of the current value for the region before the current value: trap integration
            if mapping_array[i-1] == mapping_array[i]: 
                output_array[mapping_array[i]] += input_array[i] / 2.0
            #Add the area between current value and breakpoint, by linear interpolation
            output_array[mapping_array[i]] += fraction / 2.0 * (fraction * input_array[i+1] + (2.0 - fraction) * input_array[i])
            #If we would go past the end of the output array, break
            if mapping_array[i+1] >= output_array.size:
                break
            #Add the area between breakpoint and next point by linear interpolation
            alt_fraction = 1.0 - fraction
            output_array[mapping_array[i+1]] += alt_fraction / 2.0 * (alt_fraction * input_array[i] + (2.0 - alt_fraction) * input_array[i+1])
    return output_array