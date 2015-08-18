'''Library module with some useful signal processing functions.

Alan Kastengren, XSD, APS

Created: August 18, 2015
'''
import numpy as np
import scipy.signal


def ffilter_signal(data_array,delta_t,filter_cutoff=100,order=4):
    """Function to filter a signal at a requested frequency and order.
    Uses the builtin functions from scipy for a Butterworth filter.
    Inputs:
    data_array: signal to be processed.  Assumed to be acquired at fixed delta_t
    delta_t: time step between data points
    filter_cutoff: cutoff frequency of filter in Hz.
    order: order of the Butterworth filter.
    
    Output:
    filtered array, with same length as the input data_array
    """
    #Compute parameters to design filter
    nyquist = 1.0 / 2.0 / float(delta_t)
    cutoff_norm = filter_cutoff / nyquist
    print "Normalized cutoff frequency = ", cutoff_norm
    (b,a) = scipy.signal.butter(order,cutoff_norm)
    return scipy.signal.filtfilt(b,a,data_array)

def fcompute_autocorrelation_direct(input_array,max_lag):
    '''Used to directly (not FFT) compute the autocorrelation.
    Inputs:
    input_array: array to be autocorrelated
    max_lag: maximum record shift to consider
    Output:
    array max_lag long with correlation coefficient
    '''
    output_array = np.zeros(max_lag)
    output_array[0] = np.var(input_array)
    input_array -= np.mean(input_array)
    for i in range(1,max_lag):
        output_array[i] = np.sum(input_array[:-i] * input_array[i:]) / (input_array.size - i)
    return output_array / np.var(input_array)
