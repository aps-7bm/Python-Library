''' Code to turn x-ray intensity into absorption in rational units.
Alan Kastengren, 7-BM Beamline, APS
Created: August 14, 2013
April 23, 2014: Add the fcompute_extinction_lengths function.

Edits
May 5, 2014: Clean up and reorder fluorescence corrections.  Use masked arrays
            for the extinction correction calculations.
'''
import numpy as np
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
        transmission = transmission/sorted(transmission)[-2 if len(transmission)>1 else -1]
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

def fcompute_fluorescence(raw_fluorescence, slow_events, fast_events, radiography, reference,
                          abs_coeff = 0, baseline = False):
    '''Basic fluorescence data processing.  These are corrections that should
    stay the same for all fluorescence peaks.  They include, in order of correction:
    *Dead time.  This is detector-based, and should be first.
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
    #Correct for the variations in the incoming intensity
    final_fluorescence /= reference
    #
    #Correction for attenuation of incident beam in the sample.
    #This correction assumes that absorption and fluorescence are co-located
    #along the beam path.
    #Convert radiography signal to extinction lengths.  If abs_coeff = 0, assume 
    #signal is transmission.
    if abs_coeff:
        ext_lengths = radiography * abs_coeff
    else:
        ext_lengths = -np.log(radiography)
    #
    #Form correction for nonuniform intensity through sample.  Avoid points with very low absorption
    intensity_correction = np.ones_like(ext_lengths)
    is_sig = ext_lengths > 1e-5
    intensity_correction[is_sig] = (1-np.exp(-ext_lengths[is_sig]))/ext_lengths[is_sig]
    #
    #Add this correction to the fluorescence
    final_fluorescence /= intensity_correction
    #
    #Subtract the minimum value from the raw_fluorescence to remove background
    if baseline:
        final_fluorescence = final_fluorescence - np.min(final_fluorescence)
    #
    return final_fluorescence