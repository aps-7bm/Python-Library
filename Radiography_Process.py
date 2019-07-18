''' Code to turn x-ray intensity into absorption in rational units.
Alan Kastengren, 7-BM Beamline, APS
Created: August 14, 2013
April 23, 2014: Add the fcompute_extinction_lengths function.

Edits
May 5, 2014: Clean up and reorder fluorescence corrections.  Use masked arrays
            for the extinction correction calculations.
August 10, 2016: Correct bug in fcompute_transmission so it works if 
            clear_ratio == 0 and transmission is multidimensional
August 10, 2016: Allow correction attenuation in fluorescence to be
            either assuming colocation or fluorescence in the middle.
'''
import numpy as np
from scipy.optimize import brentq
#
def fcompute_transmission(PIN, reference, clear_ratio = 0, PIN_dark=0, ref_dark = 0):
    '''Compute the transmission of a sample.
    Variables:
    PIN: measured x-ray intensity after the sample
    reference: measured x-ray intensity before the sample
    clear_ratio: ratio between PIN and reference when there is no sample.
                If zero (default), use one of the highest transmission
                points as a proxy for no sample present.
    PIN_dark, ref_dark: dark currents, i.e., signal with no beam.
    '''
    #Divide PIN by reference, subtracting off dark currents
    transmission = (PIN - PIN_dark)/(reference - ref_dark)
    #If a clear_ratio value is given, use it to normalize
    if clear_ratio:
        transmission /= clear_ratio
    else:
        #Divide by the second to max value of converted PIN to get transmission
        #Use second-highest in case highest is an outlier
        transmission = transmission/sorted(transmission.flatten())[-2 if len(transmission.flatten())>1 else -1]
    return transmission

def fcompute_extinction_lengths(PIN, reference, clear_ratio = 0, PIN_dark=0, ref_dark = 0,abs_value=True):
    '''Compute the number of extinction lengths of material in the beam.
    Use absolute value so we don't get NAN errors.  Not strictly correct, but if 
    the transmission is negative, the calculations are invalid anyway.
    '''
    if abs_value: 
        return -np.log(np.abs(fcompute_transmission(PIN,reference,clear_ratio,PIN_dark,ref_dark)))
    else:
        return -np.log(fcompute_transmission(PIN,reference,clear_ratio,PIN_dark,ref_dark))

def fcompute_radiography_density(PIN, reference, clear_ratio = 0, PIN_dark=0, ref_dark = 0, abs_coeff=1):
    '''Compute the density (mass/area or pathlength, depending on units of the absorption
    coefficient) from radiography.  Simply calls fcompute_extinction_lengths and divides
    by the absorption coefficient.  Return none for abs_coeff = 0; would get NAN errors.
    '''
    if abs_coeff:
        return fcompute_extinction_lengths(PIN, reference, clear_ratio, PIN_dark, ref_dark, True)/abs_coeff
    else:
        return None

def fcompute_fluorescence_fit_fast(raw_fluorescence, slow_events, fast_events, radiography, reference,
                          integration_time = 1, abs_coeff = 0, baseline = False, time_const = 3.13e-7,
                          colocated=True):
    '''Basic fluorescence data processing.  These are corrections that should
    stay the same for all fluorescence peaks.  They include, in order of correction:
    *Dead time.  This is detector-based, and should be first.  This is based
                on slow/fast events, with fitting performed on the fast
    *Incoming intensity variations.
    *Extinction of the incident beam in the sample.
    *If requested, baselining.
    '''
    #Correct for dead time
    final_fluorescence = fcorrect_deadtime(raw_fluorescence,slow_events,fast_events,
                      integration_time,time_const)
    #
    #Correct for the variations in the incoming intensity.  Normalize the
    #reference by its mean so we are still basically in detector counts.
    final_fluorescence /= (reference / np.mean(reference))
    #
    #Convert radiography signal to extinction lengths.  If abs_coeff = 0, assume 
    #signal is transmission.
    if abs_coeff:
        ext_lengths = radiography * abs_coeff
    else:
        ext_lengths = -np.log(radiography)
    #
    #Form correction for nonuniform intensity along beam path.  
    #If we assume absorption and fluorescence are colocated, correct the data.
    if colocated:
        final_fluorescence /= fcorrect_incident_intensity_colocated(ext_lengths)
    #Otherwise, if not colocated, assume fluorescence is in the middle.
    else:
        final_fluorescence /= fcorrect_incident_intensity_middle(ext_lengths)
    #
    #Subtract the minimum value from the raw_fluorescence to remove background
    if baseline:
        final_fluorescence = final_fluorescence - np.min(final_fluorescence)
    #
    return final_fluorescence

def fcorrect_deadtime(raw_fluorescence,slow_events,fast_events,
                      integration_time,fast_filter_time_constant=3.15e-7):
    '''Corrects for dead time.  Uses Eq. 1 from NIM 2010 Walko et al.
    paper to correct for dead time in the fast filter.  Thereafter,
    use corrected fast events/slow as a constant multiplier to correct for
    dead time.
    '''
    #Convert fast counts into a countrate
    fast_countrate = fast_events / integration_time
    #Perform fitting to correct fast countrate.  Do it on the flattened
    #array to retain generality if the array is multidimensional.
    reshaped_fast = fast_countrate.flatten()
    fitted_fast_countrate = np.zeros_like(reshaped_fast)
    for i in range(len(reshaped_fast)):
        fitted_fast_countrate[i] = brentq(fdead_time_zero_function,reshaped_fast[i]-50000,reshaped_fast[i]+1e5,
                    (reshaped_fast[i],fast_filter_time_constant))
    fitted_fast_countrate = fitted_fast_countrate.reshape(fast_countrate.shape)
    #Compute live time from the slow and fast events.
    #Take care of NAN values, which may happen when fast_events=0.
    #For these points, assume dead time = 0, since there must be little or no flux.
    live_time = np.nan_to_num(slow_events/fitted_fast_countrate/integration_time)
    #
    #Divide the fluorescence by live_time to correct.
    return raw_fluorescence/live_time

def fdead_time_zero_function(actual_countrate,observed_countrate,time_constant):
    '''Used for finding actual countrate based on dead time function.
    '''
    return actual_countrate * np.exp(-time_constant * actual_countrate) - observed_countrate

def fcompute_fluorescence(raw_fluorescence, slow_events, fast_events, radiography, reference,
                          abs_coeff = 0, baseline = False, colocated=True):
    '''Basic fluorescence data processing.  These are corrections that should
    stay the same for all fluorescence peaks.  
    They include, in order of correction:
    *Dead time.  This is detector-based, and should be first.  This is based 
                only on fast/slow events, assuming fast filter is 
                infinitely fast.  This may not be accurate at high countrates.
    *Incoming intensity variations.
    *Extinction of the incident beam in the sample.
    *If requested, baselining.
    '''
    #Compute live time from the slow and fast events.
    #Take care of NAN values, which may happen when fast_events=0.
    #For these points, assume dead time = 0, since there must be little or no flux.
    live_time = np.nan_to_num(slow_events/fast_events)
    #
    #Divide the fluorescence by live_time to correct.
    final_fluorescence = raw_fluorescence/live_time
    #
    #Correct for the variations in the incoming intensity.  Normalize the
    #reference by its mean so we are still basically in detector counts.
    final_fluorescence /= (reference / np.mean(reference))
    #
    #Convert radiography signal to extinction lengths.  If abs_coeff = 0, assume 
    #signal is transmission.
    if abs_coeff:
        ext_lengths = radiography * abs_coeff
    else:
        ext_lengths = -np.log(radiography)
    #
    #Form correction for nonuniform intensity along beam path.  
    #If we assume absorption and fluorescence are colocated, correct the data.
    if colocated:
        final_fluorescence /= fcorrect_incident_intensity_colocated(ext_lengths)
    #Otherwise, if not colocated, assume fluorescence is in the middle.
    else:
        final_fluorescence /= fcorrect_incident_intensity_middle(ext_lengths)
    #
    #Subtract the minimum value from the raw_fluorescence to remove background
    if baseline:
        final_fluorescence = final_fluorescence - np.min(final_fluorescence)
    #
    return final_fluorescence

def fcorrect_incident_intensity_colocated(ext_lengths):
    '''Formulate correction to the fluorescence due to the attenuation of the 
    incident beam in the sample.  Assume fluorescence and radiography are 
    co-located.
    '''
    intensity_correction = np.ones_like(ext_lengths)
    is_sig = ext_lengths > 1e-5
    intensity_correction[is_sig] = (1-np.exp(-ext_lengths[is_sig]))/ext_lengths[is_sig]
    return intensity_correction

def fcorrect_incident_intensity_middle(ext_lengths):
    '''Formulate correction to the fluorescence due to the attenuation of the 
    incident beam in the sample.  Assume fluorescence is in the middle of the
    absorption.
    '''
    return np.exp(-ext_lengths/2.0)