#Module with methods to compute autospectral density
import numpy as np
import scipy.fftpack
import math
def fcompute_hanning_window(num_points,truncate=False):
    """Compute the Hanning window for this number of
    points, returning it to the caller.
    truncate option of True transforms to greatest power of 2 less than
    input array size.  
    """
    #Define prefactor that accounts of loss due to windowing.  Follows Bendat and Piersol    
    if truncate:    
        num_points = int(math.pow(2,math.floor(math.log(num_points,2))))
        print "Number of points to transform = ", num_points
    prefactor = math.sqrt(8/3)
    multiplier = math.pi/num_points     #Internal to window calculation
    #Define a numpy array with requisite number of points
    output_window = np.zeros(num_points)
    #Create an array of points for use in window array
    x = np.arange(0.5*multiplier,(0.5+num_points)*multiplier,multiplier)
    #Fill window array with points
    output_window = prefactor*(np.ones(num_points)-np.power(np.cos(x),2))
    return output_window

def fcompute_autospectrum(data,window_array,truncate=False,delta_t=1):
    """Compute the autospectrum of a chunk of data.

    Input window array is applied to data.
    Computes the one-sided autospectral density.
    Assumes the the frequency of sampling is unity.
    Truncate option of True transforms to greatest power of 2 less than
    input array size.
    """
    #If necessary, truncate now
    new_size = data.size
    if truncate:
        new_size = int(math.pow(2,math.floor(math.log(data.size,2))))
    #Remove the mean component
    data = data - np.mean(data[:new_size])
    print "Data SD = ", np.std(data[:new_size],ddof=1)
    print "Data mean = ", np.mean(data[:new_size])
    data = np.multiply(data[:new_size], window_array)
    print "Data SD = ", np.std(data[:new_size],ddof=1)
    #Give DFT: have to include delta_t to make this correct, per Bendat and Piersol
    data_fft = scipy.fftpack.fft(data[:new_size])*delta_t
    output_autospectrum = np.zeros(len(data_fft)/2)
    output_autospectrum = 2.0/(len(data_fft)*delta_t)*np.power(np.absolute(data_fft[:len(data_fft)/2]),2)
    print "Autospectrum sum = ",np.sum(output_autospectrum)
    return output_autospectrum

def fcompute_autospectrum_ensemble(input_data,chunks=8,delta_t=1,window=True,truncate=True):
    """Compute the autospectrum of a dataset using ensemble averaging.
    
    Compute the autospectrum by splitting the data into chunks pieces,
    windowing (if requested), and averaging autospectra.  Return values
    are the autospectrum and a numpy array of frequencies.
    """
    #If requested, truncate to the next lowest power of 2
    data = np.zeros(10)
    if truncate:
        truncated_size = int(math.pow(2,math.floor(math.log(input_data.size,2))))
        data = input_data[:truncated_size]
    else:
        data = input_data
    #Remove the mean component
    data = data - np.mean(data)
    print "Data variance = ", np.std(data,ddof=1)**2
    print "Data mean = ", np.mean(data)
    #Find the length of the chunks that will be used.
    chunk_length = data.size/chunks
    #Find the windowing function that will be applied
    window = fcompute_hanning_window(chunk_length)
    #Loop through the data, computing autospectrum of each chunk.
    output_autospectrum = np.zeros(chunk_length/2)
    for i in range(chunks):
        chunk_data = np.multiply(data[i*chunk_length:(i+1)*chunk_length], window)
        print "Data variance = ", np.std(chunk_data,ddof=1)**2
        #Give DFT: have to include delta_t to make this correct, per Bendat and Piersol
        data_fft = scipy.fftpack.fft(chunk_data)*delta_t
        output_autospectrum += 2.0/(len(chunk_data)*delta_t)*np.power(np.absolute(data_fft[:chunk_length/2]),2)
    output_autospectrum /= float(chunks)
    #Compute frequency array
    delta_f = 1.0/delta_t/float(chunk_length)
    frequencies = np.linspace(0,chunk_length/2.0*delta_f,chunk_length/2,endpoint=False)
    print "Autospectrum integral = ",np.sum(output_autospectrum)*delta_f
    return frequencies,output_autospectrum
