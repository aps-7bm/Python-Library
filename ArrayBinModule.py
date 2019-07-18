'''Module of routines for binning an array by a non-integer number of steps.
Routines are included for binning pulsed data for each pulse, as well as 
using a prespecified time for each bin. 

Alan Kastengren, XSD, APS

Started: June 8, 2016
'''
import ArrayBin_Cython as arc
import numpy as np
import scipy.stats
import scipy.signal
import matplotlib.pyplot as plt
bin_time = 153e-9         #Time between bunches in 24 bunch mode at APS as of 2015
bunches = 24

def fbin_signal_by_pulse(input_array,delta_t,pulse_time=None,repeat_num=1,start_time=0):
    '''Bin a peaked signal with a regular period by each peak.  
    
    Code finds the peak spacing.  Peaks need not be at integer spacing.
    Inputs:
    input array: array of peaked input data.
    delta_t: time period between measurement points in the original array.
    pulse_time: Initial estimate of the time between pulses.
    repeat_num: if the pulses have a repeating pattern, how many pulses per period.
    start_time: where should integration begin.  Only here to match fbin_signal_fixed_time
    Outputs:
    output_data: binned signal 
    pulse_time: updated measurement of the time between pulses
    start_point * delta_t: where did we really begin integration
    '''
      
    #Set pulse time to default if it isn't given
    if not pulse_time:
        pulse_time = bin_time
    #Set the number of sub-bunches for pulses with a repeating pattern
    if not repeat_num:
        repeat_num = bunches
    #Find the integration parameters
    pulse_duration_points,start_point = ffind_integration_parameters(input_array,pulse_time,delta_t)
    #Compute the real pulse time
    pulse_time = pulse_duration_points * delta_t
    #Perform integration with Cython code
    output_data = arc.fbin_array(input_array[start_point:],pulse_duration_points)
    #Divide the output data by the pulse duration to get an average voltage back
    output_data = output_data / pulse_duration_points
    #Attempt to remove the impact of bunch charge variations.
    if repeat_num > 1:
        output_data = fremove_bunch_charge_variations(output_data,repeat_num)
    #Return both the binned array and the pulse time
    return (output_data,pulse_time,start_point*delta_t)

def ffind_integration_parameters(input_data,pulse_time,delta_t):
    '''Find the correct start point and number of points per bin to integrate a
    peaked signal whose repetition rate does not match the delta_t of the
    input data.
    
    Inputs:
    input_data: peaked signal to be integrated
    pulse_time: estimate of the time between pulses
    delta_t: time period for each measurement point
    Outputs:
    actual_pulse_time: pulse time derived from data, assuming delta_t is right
    start_point: the point at which to start integration
    '''
    #Use Cython code to find the peaks
    peak_positions = arc.ffind_breakpoints_peaked(input_data,int(pulse_time / delta_t))
    #Compute the actual average pulse duration and add to coordinate header
    actual_pulse_duration,first_peak = scipy.stats.linregress(np.arange(peak_positions.size),peak_positions)[:2]
    #Find the actual number of points before the peak that we should use
    points_before_peak = ffind_points_before_peak(input_data,peak_positions,actual_pulse_duration)
    #Give the start point for the integration
    if first_peak > points_before_peak:
        start_point = int(np.rint(first_peak - points_before_peak))
    else:
        start_point = int(np.rint(first_peak + actual_pulse_duration - points_before_peak))
    return actual_pulse_duration,start_point

def ffind_points_before_peak(data_array,peak_positions,pulse_duration):
    '''Figure out how many points before the peak we should go to perform integration.
    
    Inputs:
    data_array: actual data values
    peak_positions: peak points for each bunch.
    pulse_duration: average pulse duration in time steps
    Outputs:
    points_before_peak: how many time steps before peak to use for integration
    '''
    #Look over first one hundred peaks
    points_before_peak_array = np.zeros(100)
    back_step = int(2*pulse_duration/3)
    for i in range(1,101):
        minimum_point = np.argmin(data_array[peak_positions[i]-back_step:peak_positions[i]])
        points_before_peak_array[i-1] = float(back_step - minimum_point)
    return np.mean(points_before_peak_array)

def fremove_bunch_charge_variations(input_array,bunches):
    '''Removes the bunch charge variations from the data.
    '''
    output_array = np.zeros(np.size(input_array))
    multiplier = np.zeros(bunches)
    for i in range(bunches):
        multiplier[i] = np.sum(input_array[i::bunches],dtype=np.float64)
    #Divide through by the mean multiplier
    multiplier /= np.mean(multiplier,dtype=np.float64)
    for i in range(bunches):
        output_array[i::bunches] = input_array[i::bunches] / multiplier[i]
    return output_array

def fbin_signal_fixed_time(input_array,delta_t,pulse_time=None,repeat_num=None,start_time=0):
    """Function to read in signal, binning by fixed bin time bin_time.
    
    Inputs:
    input_array: array of input data.
    delta_t: time period between measurement points in the original array.
    pulse_time: Initial estimate of the time between pulses.
    repeat_num: if the pulses have a repeating pattern, how many pulses per period.
    start_point: where should integration begin.
    Outputs:
    output_data: binned signal 
    pulse_time: updated measurement of the time between pulses
    start_point * delta_t: where did we really begin integration
    """   
    #Set pulse time to default if it isn't given
    if not pulse_time:
        pulse_time = bin_time
    #Set the number of sub-bunches for pulses with a repeating pattern
    if not repeat_num:
        repeat_num = bunches
    #Compute the number of points (float) per bin
    pulse_duration_points = pulse_time / delta_t
    #Compute the start point for the integration
    start_point = np.rint(start_time / delta_t)
    #Perform integration with Cython code
    output_data = arc.fbin_array(input_array[start_point:],pulse_duration_points)
    #Attempt to remove the impact of bunch charge variations.
    if repeat_num > 1:
        output_data = fremove_bunch_charge_variations(output_data,repeat_num)
    #Return both the binned array and the pulse time
    return (output_data,pulse_time,start_time)

def frefine_time_estimate(input_array,delta_t,pulse_time):
    '''Takes a periodic signal and computes an accurate time per pulse.
    Based on cross-correlation of signal at beginning to the end.
    Inputs:
    input_array: numpy array of the input periodic signal
    delta_t: time period between measurement points in the original array.
    pulse_time: Initial estimate of the time between pulses.
    '''
    #Compute the number of points (float) per cycle of the periodic signal
    pulse_points = pulse_time / delta_t
    #Perform cross-correlation between initial pulse_points points and final 2 * pulse_points
    correlation_matrix = scipy.signal.correlate(input_array[-2 * int(pulse_points):],input_array[:int(pulse_points)], 'valid')
    #Compute where this means the best overlap is found
    points_between = input_array.shape[0] - 2 * int(pulse_points) + np.argmax(correlation_matrix)
    #Refine the time estimate
    num_cycles = np.rint(points_between / pulse_points)
    return float(points_between) / float(num_cycles) * float(delta_t)

def frefine_time_estimate_staged(input_array,delta_t,pulse_time,fraction=0.02):
    '''Takes a periodic signal and computes an accurate time per pulse.
    Based on cross-correlation of signal at beginning to the end.
    Performs frefine_time_estimate twice, once on a short snippet of the signal,
    once on a longer snippet in case number of pulses is greater than
    the fractional error in the time estimate, such that we might be fooled
    about how many pulses are in the signal.
    Inputs:
    input_array: numpy array of the input periodic signal
    delta_t: time period between measurement points in the original array.
    pulse_time: Initial estimate of the time between pulses.
    fraction: fraction of the signal to use for the first iteration.
    '''
    num_initial_points = int(input_array.shape[0] * fraction)
    new_pulse_time = frefine_time_estimate(input_array[:num_initial_points],delta_t,pulse_time)
    return frefine_time_estimate(input_array,delta_t,new_pulse_time)

def fread_signal_direct(input_array,delta_t,pulse_time=None,repeat_num=None,start_time=0):
    """Function to read in data directly
    Inputs:
    input_array: numpy array with raw data
    pulse_time,repeat_num: dummy inputs to match other methods in this module.
    start_time: time of first point to be used in the new array.
    Outputs:
    tuple of numpy array of data starting at start_point, pulse_time, start_point
    """   
    #Compute the start point for the integration
    start_point = np.rint(start_time / delta_t)
    #Return the input array 
    return (input_array[start_point:],pulse_time,start_point)