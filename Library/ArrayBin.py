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

def ffind_breakpoints_peaked(input_array,pulse_points,search_region=5):
    """Find the breakpoints between peaks in a periodic peaked signal.
    Inputs:
    input_array: float array holding the input data values
    pulse_points: the nominal number of points between pulses
    search_region: number of points over which to search for each peak
    """
    #Figure out the stride we will use between peaks
    stride = int(np.floor(pulse_points))
    print "Stride length = ", stride, " points."
    #Make an empty array to hold the peak positions.  Pre-allocate to attempt to speed up
    peak_positions = np.zeros(int(float(input_array.shape[0]) * 1.05 / pulse_points),dtype='i8')
    #Set starting position
    num_peaks = 1
    peak_positions[0] = np.argmax(input_array[:stride+1])
    #Reduce stride by half of search region size to make math simpler during actual striding
    stride -= search_region//2
    current_position = peak_positions[0] + stride
    #Keep striding until we are within two strides of the end
    while current_position + 2 * stride < input_array.shape[0]:
        #Find the maximum value in the search region around nominal peak
        current_position = current_position + np.argmax(input_array[current_position:current_position+search_region])
        #Add this peak position to the peak positions list and increment counter
        peak_positions[num_peaks] = current_position
        num_peaks += 1
        #Stride
        current_position += stride
        #Print about progress
#         if not(num_peaks%500000):
    print "Total number of peaks = " + str(num_peaks) + "."
    return peak_positions[:num_peaks]



if __name__ == '__main__':
    #Unit test for fbin_array
    test_array = np.pi * np.ones(120)
    test_output = fbin_array(test_array,np.sqrt(2.0))
#     plt.plot(test_output/np.pi/np.sqrt(2.0))
#     plt.show()
    assert np.allclose(np.ones_like(test_output)*np.sqrt(2),test_output/np.pi,1e-7,1e-7)
    #Unit test for ffind_breakpoints_peaked
    input_angles = np.pi * np.arange(1e6) / 1000
    input_array = np.sin(input_angles)
    pulse_output = ffind_breakpoints_peaked(input_array,2000)
#     plt.plot(input_angles,input_array)
#     plt.plot(pulse_output * np.pi / 1000,np.ones_like(pulse_output),'r.')
#     plt.show()
    assert np.allclose(np.arange(pulse_output.shape[0])*2000 + 500,pulse_output,1e-7,1e-7)