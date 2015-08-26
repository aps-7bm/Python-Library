'''Code to downsample an array by binning with a non-integer ratio of old 
size to new size.

Alan Kastengren, XSD, APS

Started: August 26, 2015
'''
import numpy as np
import matplotlib.pyplot as plt

def fbin_array(input_array,delta_ratio):
    '''Bins an array by ration delta_ratio
    Inputs:
    input_array: numpy array with old values
    delta_ratio: # of points from old array for each new array ValueError
    Output:
    output_array: numpy array with binned values
    '''
    #Make sure delta_ratio is a float
    delta_ratio = float(delta_ratio)
    print "Delta Ratio = " + str(delta_ratio)
    print input_array.shape
    #Make an array saying which new point each point should go to
    mapping_array = np.floor(np.arange(float(input_array.shape[0]))/delta_ratio).astype(np.int32)
    #Set the size of the output array
    output_array = np.zeros(mapping_array[-1] - 1)
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


if __name__ == '__main__':
    test_array = np.pi * np.ones(120)
    test_output = fbin_array(test_array,np.sqrt(2.0))
    plt.plot(test_output/np.pi/np.sqrt(2.0))
    plt.show()
    assert np.allclose(np.ones_like(test_output)*np.sqrt(2),test_output/np.pi,1e-7,1e-7)