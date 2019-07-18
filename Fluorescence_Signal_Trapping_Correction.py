'''Script to perform self-absorption correction for the water doped with Zn and Ni.

Alan Kastengren, XSD, Argonne

Started: April 28, 2014
'''
import h5py
import numpy as np
import os
import ALK_Utilities as ALK

def fcorrect_signal_trapping_fitted(hdf_file,dataset_names,line_energies,x_name='7bmb1:m26.VAL',
                                    output_name=None,ref_r0=0.0,fitting=True,degree=2):
    '''Perform signal trapping correction to the dataset.
    Use two fluorescence lines to find signal trapping.
    If fitting=True (default), will perform a fit of degree degree to ratio between lines.
    '''
    #Make sure that we actually have appropriate inputs
    if len(line_energies) != 2 or len(dataset_names) != 2:
        print "Invalid input"
        return
    #Find the exponent for the correction
    exponent = 1.0 / (1.0 - (line_energies[0]/line_energies[1])**3)
    print "Exponent = " + str(exponent)
    #Extract arrays for high and low Z materials
    high_Z_array = hdf_file.get(dataset_names[0])[...]
    low_Z_array = hdf_file.get(dataset_names[1])[...]
    #If either of these arrays is None, or sizes don't match, exit
    if not high_Z_array or not low_Z_array or np.size(high_Z_array) != np.size(low_Z_array) or np.any(np.isnan(high_Z_array)) or np.any(np.isnan(low_Z_array)):
        print "The arrays for the two fluorescence lines did not read in properly."
        print "Exiting without doing anything."
        return
    #Don't want to correct where the signal is nearly zero.  Find the mean value and set
    #threshold at 25% of way between mean and min.  A bit abritrary, but better than 
    #previous method, which relied on the peak.
    threshold_ratio = 0.25
    threshold_value = np.mean(high_Z_array)*threshold_ratio - np.min(high_Z_array)*(1-threshold_ratio)
    #Make a boolean array to mask the 
    above_threshold = high_Z_array>threshold_value
    #Form a ratios array
    ratio = np.ones_like(high_Z_array)
    ratio[above_threshold] = np.nan_to_num(low_Z_array[above_threshold] / high_Z_array[above_threshold])
    print ratio
    #If there is a reference value input, use it.  Otherwise, use max value we found before.
    r0 = ref_r0 if ref_r0 else np.max(ratio[above_threshold])
    print "Reference ratio between ROIs = " + str(r0)
    #Normalize ratio by this max value.  Only for values with a good amount of fluorescence
    ratio[above_threshold] /= r0
    #Perform correction
    if fitting:
        new_ratio = ffit_ratio(hdf_file[x_name][...],ratio,above_threshold,degree)
    else:
        new_ratio = ratio
    output_array = high_Z_array*new_ratio**exponent
    if not output_name:
        output_name = dataset_names[0] + "_Sig_Trap_Corrected"
    ALK.fwrite_HDF_dataset(hdf_file,output_name,output_array)
    print "File " + hdf_file.filename + " processed successfully."
    
def fcorrect_signal_trapping_filename(filename,path='/home/beams/AKASTENGREN/SprayData/Cycle_2014_1/Radke/Time_Averaged_Fluorescence/',
                                    dataset_names,line_energies,x_name='7bmb1:m26.VAL',
                                    output_name=None,ref_r0=0.0,fitting=True,degree=2):
    '''Convert DataGrabber file to HDF5 if the input is the name of the output
    file rather than an HDF5 object.  Simply makes an HDF5 File object
    and calls the converter based on the HDF5 File object.
    '''
    #Make up an approrpiate output file name if none is given
    if not os.path.isfile(path+filename):
        print "File " + filename + "does not exist.  Exiting without doing anything."
        return
    #Open an HDF file and call the converter based on having an HDF5 File object
    with h5py.File(path+filename,'r+') as write_file:
        fcorrect_signal_trapping_fitted(write_file,dataset_names,line_energies,x_name,
                                    output_name,ref_r0,fitting,degree)
    
def ffit_ratio(x,ratio,mask=None,degree=2):
    '''Take the ratio points and perform a least-squares fit.
    Experience shows that the raw ratio can be too noisy.  Use the fit to make
    the results cleaner.
    Mask is a boolean numpy mask array to show the points to be fit.
    Default is none, which will cause all points to be fit.
    '''
    #If mask is None, make it to allow all values to be permitted
    if not mask:
        mask = ratio != np.NAN
    #Perform fit on points allowed by mask
    fit_results = np.polyfit(x[mask],ratio[mask],degree)
    #Form a new ratio formed by the linearized results from the input.  Return this.
    new_ratio = np.ones_like(ratio)
    new_ratio[mask] = 0
    for j in range(degree+1):
        new_ratio[mask] = new_ratio[mask] * x[mask] + fit_results[j] 
    return new_ratio
    