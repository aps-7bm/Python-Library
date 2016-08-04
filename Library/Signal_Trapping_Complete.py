'''Module to compute the signal trapping corrections.
This code is a combination of the previous Signal_Trapping_Fitted
and Signal_Trapping_Functions codes.

This code is aimed for reconstructions where the radiography signal can simply
be scaled to get the absorption profile for the fluorescence signal
trapping.  If, for example, a significant amount of the absorption in certain
regions is due to the fluorescent species, such that he shape of the 
absorption profile would change between the incident and emitted photon
energies, this approach must be altered.

In this code, the beam is assumed to propagate along the columns 
(i.e., across rows).  This is the z direction.  The x direction
is along rows (across columns).

Alan Kastengren, XSD, APS

Started: August 3, 2016
'''
import matplotlib.pyplot as plt
import h5py
import numpy as np
import Projection_Fit as pf
from scipy import optimize
from scipy import interpolate
import scipy.stats
import scipy.integrate
import ALK_Utilities as ALK

#Set some globals to clean up 

def ffit_projection(proj_fit_function,x,projection_data,parameter_guesses,
                      abs_ratio=1,display=False):
    '''Computes a fit to the projection data.
    If desired, the project data are rescaled by the abs_ratio.
    Variables:
    projection_data: data for the projection
    proj_fit_function: function to be used to fit the projection
    x: x values for projection_data points
    parameter_guesses: guesses for proj_fit_function parameters
    abs_ratio: ratio by which to scale the projection data.  Useful for energy transform
    Returns: fitted parameters
    '''
    fit_parameters,covariance = optimize.curve_fit(proj_fit_function,x,projection_data*abs_ratio,parameter_guesses)
    if display:
        print fit_parameters
        print covariance
    return fit_parameters

def fapply_signal_trapping_transmission(data_x,data_signal,sig_trap_x,sig_trap_trans):
    '''Applies signal trapping transmission to data.
    Includes back-interpolating from signal trapping grid to data grid.
    Inputs:
    data_x: x values for the data
    data_signal: signal values for data, to which signal trapping is applied
    sig_trap_x: x values for signal trapping transmission
    sig_trap_trans: signal trapping transmission
    Outputs:
    data signal with signal trapping applied
    '''
    interpolated_transmission = interpolate.interp1d(sig_trap_x,sig_trap_trans)(data_x)
    return data_signal / interpolated_transmission, interpolated_transmission

def fwrite_corrected_fluorescence(hdf_file,x_dataset_name,old_dataset_name,new_dataset_name,sig_trap_x,sig_trap_trans):
    '''Corrects for signal trapping and writes to file.
    Variables:
    hdf_file: h5py File object for file we are examining.
    x_dataset_name: dataset with x values for the data
    old_dataset_name: name of dataset with data before signal trapping correction
    new_dataset_name: name for corrected dataset
    sig_trap_x: x values for signal trapping array
    sig_trap_trans: amount of signal trapping
    Returns: None
    '''
    #Interpolate the sig_trap array to the actual x points where the data are taken.
    corr_data,corr_trans = fapply_signal_trapping_transmission(hdf_file[x_dataset_name][...],
                                                               hdf_file[old_dataset_name][...],
                                                               sig_trap_x,sig_trap_trans)
    ALK.fwrite_HDF_dataset(hdf_file, new_dataset_name, corr_data)
    ALK.fwrite_HDF_dataset(hdf_file,old_dataset_name+'SignalTrappingTransmission',corr_trans)
    return

def fcreate_arrays(x_lims, z_lims,num_x,num_z):
    '''Creates a 2D array for performing signal trapping corrections.
    Assumes that the beam propagates along the z directiion.
    Parameters:
    x_lim: limits of the array in the x (row) direction as a two-entry list
    z_lim: limits of the array in the z (column) direction as a two-entry list
    num_x, num_z: number of array elements in the x and z directions, respectively.
    '''
    x_vals = np.linspace(x_lims[0],x_lims[1],num_x)
    z_vals = np.linspace(z_lims[0],z_lims[1],num_z)
    return x_vals, z_vals

def fdistribution_from_projection(x_vals, z_vals, proj_x, proj_data, 
                                  proj_fit_function, proj_param_guesses,
                                  dist_fit_function, 
                                  center_index=None,scaling=1):
    '''Populate a 2D array to be used for signal trapping corrections
    based on a fit to the projection data.
    Inputs:
    x_vals,z_vals: coordinates for 2D array.
    proj_x: x coordinates for the projection data
    proj_data: projection data to be fit.
    proj_fit_function: function to be used to fit the projection data.
    proj_param_guesses: initial parameter guesses for projection fit.
    dist_fit_function: function to be used to fill the 2D distribution
    center_index: parameter # that represents the x center
    scaling: scaling factor to apply to projection data.  Useful for 
            scaling absorption from incident to emission photon energy
    Output:
    2D array for distribution implied by projection fit.
    '''
    #Perform a fit to the projection data
    proj_fit_params = ffit_projection(proj_fit_function,proj_x,proj_data,
                                      proj_param_guesses,scaling)
    #Find the x center from these parameters, or set to zero
    if center_index == None:
        x_center = 0
    else:
        x_center = proj_fit_params[center_index]
    #Make a mesh and find the distances, then run through the fit function
    xmesh,zmesh = np.meshgrid(x_vals,z_vals)
    r = np.hypot(xmesh-x_center,zmesh)
    return dist_fit_function(r,proj_fit_params)

def fcalculate_signal_trapping_2D(x_vals,z_vals,density_2D,detector_negative=True,display=True):
    '''Calculate the total signal trapping for each point in the 2D array.
    The result is in extinction lengths (e**-EL needed to get transmission).
    Integrates along each row (across the columns).
    detector_negative controls whether integration is to the negative x end of
    the row or to the +x end.  If True, integrate from -x to each point.
    Inputs:
    x_vals,z_vals: coordinates of 2D arrays
    density_2D: 2D array of absorption.
    detector_negative: controls whether integration is to the negative x end of
            the row or to the +x end.  If True, integrate from -x to each point.
    display: if True, plot the distribution of signal trapping.
    Output
    2D array of extinction lengths of signal trapping for each point.
    '''
    #Define output array: 2D map of ext lengths to the detector.
    signal_trapping_array = np.zeros_like(density_2D)
    #Do an integration along axis 1: 
    if detector_negative:
        signal_trapping_array = scipy.integrate.cumtrapz(density_2D,x_vals,initial=0)
    else:
        #Since we are doing this in reverse order, we need a negative sign
        signal_trapping_array[:,::-1] = -scipy.integrate.cumtrapz(density_2D[:,::-1],x_vals[::-1],initial=0)
    if display:
        plt.contourf(x_vals,z_vals,signal_trapping_array)
        plt.colorbar()
        plt.figure()
    return signal_trapping_array

def fcalculate_attenuation_2D(x_vals,z_vals,density_2D,detector_negative=True,display=True):
    '''Calculate the incident beam attenuation for each point in the 2D array.
    The result is in extinction lengths (e**-EL needed to get transmission).
    Integrates along beam path (along the columns).
    detector_negative controls whether integration is to the negative z end of
    the row or to the +z end.  If True, integrate from -z to each point.
    Inputs:
    x_vals,z_vals: coordinates of 2D arrays
    density_2D: 2D array of absorption.
    detector_negative: controls whether integration is to the negative z end of
            the row or to the +z end.  If True, integrate from -z to each point.
    display: if True, plot the distribution of signal trapping.
    Output
    2D array of extinction lengths of attenuation for each point.
    '''
    #Define output array: 2D map of ext lengths to the detector.
    signal_trapping_array = np.zeros_like(density_2D)
    #Do an integration along axis 1: 
    if detector_negative:
        signal_trapping_array = scipy.integrate.cumtrapz(density_2D,z_vals,initial=0,axis=0)
    else:
        #Since we are doing this in reverse order, we need a negative sign
        signal_trapping_array[::-1,:] = -scipy.integrate.cumtrapz(density_2D[::-1,:],z_vals[::-1],initial=0,axis=0)
    if display:
        plt.contourf(x_vals,z_vals,signal_trapping_array,101)
        plt.colorbar()
    return signal_trapping_array

def fprojection_weighted_average(x_vals,z_vals,data_2D,signal_trapping_2D,tol=0):
    '''Calculates the projection data subject to signal trapping corrections.
    Makes an average signal trapping transmission weighted by
    the fluorescence distribution along each projection.
    Parameters:
    x_vals, z_vals: coordinates
    data_2D: distribution of fluorescence
    signal_trapping_2D: signal trapping at each point of data_2D
    tol: minimum fluorescence signal on a projection for which to calculate signal trapping. 
        Used to avoid divide by zero problems.
    Returns:
    x_vals: x values for the array, since it won't necessarily correspond to original data points.
    averaged_signal_trapping: weighted average transmission of fluorescence for each projection.
    '''
    #Convert this to transmission at each point
    signal_trapping_array = np.exp(-signal_trapping_2D)
    #Compute weighted average transmission.  Handle case where data_2D integral might be zero by making trans=1
    denominator = np.ones_like(x_vals)
    mask =  (np.abs(scipy.integrate.simps(data_2D,z_vals,axis=0)) > tol)
    denominator[mask] = scipy.integrate.simps(data_2D[:,mask],z_vals,axis=0)
    #Integrate along beam direction, dividing by total fluorescence signal for a weighted average
    averaged_signal_trapping = scipy.integrate.simps(signal_trapping_array * data_2D,z_vals,axis=0)/denominator
    return x_vals, averaged_signal_trapping

def fcompute_signal_trapping_simple(x_lims, z_lims,num_x,num_z,
                                    rad_x,rad_data,rad_fit_function,
                                    rad_param_guesses,rad_dist_function,
                                    fluor_x,fluor_data,fluor_fit_function,
                                    fluor_param_guesses,fluor_dist_function,
                                    detector_negative=True,abs_scaling=1.0,
                                    rad_center_index=None,fluor_center_index=None):
    '''Function to perform a simple signal trapping correction.
    This correction assumes that the absorption scales the same with energy at all
    locations.  For example, we don't have a large amount of absorption of the
    incident beam from the fluorescent element and a different element.
    This code also assumes that we have only a single projection for the 
    absorption and fluorescence data.  We will perform fitting and assume
    axisymmetry to get the 2D distribution.
    
    Algorithm:
    Create appropriate arrays to perform computations.
    Perform fitting of the absorption data and scale to the correct energy.
    Calculate signal trapping on a 2D slice, assuming axisymmetry.
    Return revised estimate of fluorescence signal.
    Variables:
    x_lims, z_lims, num_x, num_z: size and shape of 2D arrays for computations
    rad_x, rad_data: projection x and extinction lengths.
    fluor_x, fluor_data: fluorescence x and extinction lengths.
    rad_fit_function: function used to fit absorption, in terms of extinction lengths
    rad_param_guesses: initial guess parameters for absorption fit
    fluor_fit_function: function used to fit fluorescence
    fluor_param_guesses: initial guess parameters for fluorescence fit
    rad(fluor)_dist_function: function used to turn projection parameters into 2D distribution
    detector_negative: if True, detector is on -x side of the experiment.
    abs_scaling: scaling factor between incident and fluorescence energy for absorption
    rad_center_index,fluor_center_index: parameter for fit giving center of projection.
    Returns:
    x_vals: x values for the array, since it won't necessarily correspond to original data points.
    signal_trap_proj_trans: effective signal trapping transmission for each x.
    '''
    #Create the arrays we will use for the 
    x_vals, z_vals = fcreate_arrays(x_lims, z_lims,num_x,num_z)
    #Fit projections of radiography and fluorescence
    absorption_array = fdistribution_from_projection(x_vals, z_vals, rad_x, rad_data, 
                                  rad_fit_function, rad_param_guesses,
                                  rad_dist_function, 
                                  rad_center_index,abs_scaling)
    fluor_array = fdistribution_from_projection(x_vals, z_vals, fluor_x, fluor_data, 
                                  fluor_fit_function, fluor_param_guesses,
                                  fluor_dist_function, 
                                  fluor_center_index,1.0)
    #Compute the 2D distribution of signal trapping, converting to transmission
    signal_trapping_array = fcalculate_signal_trapping_2D(x_vals,z_vals,absorption_array,detector_negative)
    #Average this for each projection
    x,signal_trap_proj_trans = fprojection_weighted_average(x_vals,z_vals,fluor_array,signal_trapping_array)
    
    print signal_trap_proj_trans.shape
    return x,signal_trap_proj_trans

def fcompute_attenuation_correction(x_lims, z_lims,num_x,num_z,
                                    rad_x,rad_data,rad_fit_function,
                                    rad_param_guesses,rad_dist_function,
                                    fluor_x,fluor_data,fluor_fit_function,
                                    fluor_param_guesses,fluor_dist_function,
                                    detector_negative=True,abs_scaling=1.0,
                                    rad_center_index=None,fluor_center_index=None):
    '''Function to perform a correction
    of fluorescence due to attenuation within the sample.
    This codeassumes that we have only a single projection for the 
    absorption and fluorescence data.  We will perform fitting and assume
    axisymmetry to get the 2D distribution.
    
    Algorithm:
    Create appropriate arrays to perform computations.
    Perform fitting of the absorption data.
    Calculate attenuation on a 2D slice, assuming axisymmetry.
    Return estimated effective attenuation.
    Variables:
    x_lims, z_lims, num_x, num_z: size and shape of 2D arrays for computations
    rad_x, rad_data: projection x and extinction lengths.
    fluor_x, fluor_data: fluorescence x and extinction lengths.
    rad_fit_function: function used to fit absorption, in terms of extinction lengths
    rad_param_guesses: initial guess parameters for absorption fit
    fluor_fit_function: function used to fit fluorescence
    fluor_param_guesses: initial guess parameters for fluorescence fit
    rad(fluor)_dist_function: function used to turn projection parameters into 2D distribution
    detector_negative: if True, detector is on -x side of the experiment.
    abs_scaling: scaling factor between incident and fluorescence energy for absorption
    rad_center_index,fluor_center_index: parameter for fit giving center of projection.
    Returns:
    x_vals: x values for the array, since it won't necessarily correspond to original data points.
    atten_corr_proj_trans: effective intensity for each projection.
    '''
    #Create the arrays we will use for the 
    x_vals, z_vals = fcreate_arrays(x_lims, z_lims,num_x,num_z)
    #Fit projections of radiography at incident and fluorescence photon energies
    absorption_array = fdistribution_from_projection(x_vals, z_vals, rad_x, rad_data, 
                                  rad_fit_function, rad_param_guesses,
                                  rad_dist_function, 
                                  rad_center_index,1.0)
    #Find distribution of fluorescence
    fluor_array = fdistribution_from_projection(x_vals, z_vals, fluor_x, fluor_data, 
                                  fluor_fit_function, fluor_param_guesses,
                                  fluor_dist_function, 
                                  fluor_center_index,1.0)

    #Compute the 2D distribution of attenuation for correcting the fluorescence
    attenuation_corr_array = fcalculate_attenuation_2D(x_vals,z_vals,absorption_array,detector_negative)
    #Average this for each projection
    x,atten_corr_proj_trans = fprojection_weighted_average(x_vals,z_vals,fluor_array,attenuation_corr_array)
    return x,atten_corr_proj_trans

def ftest_code(abs_peak = 0.4,sigma=0.3,fluor_peak=100000,fluor_sigma=0.01):
    '''Code to test whether I am calculating things correctly.
    '''
    plt.figure()
    plt.title('Signal Trapping')
    
    rad_x = np.linspace(-3,3,1201)
    fluor_x = rad_x
    rad_data = pf.fgauss_no_offset(rad_x, np.sqrt(2.0*np.pi)*sigma*abs_peak, sigma, 0)
    fluor_data = pf.fgauss_no_offset(fluor_x, np.sqrt(2.0*np.pi)*fluor_sigma*fluor_peak, fluor_sigma, 0)
    x,signal_trap_proj_trans = fcompute_signal_trapping_simple([-1,3],[-2,3],1201,1201,
                                    rad_x,rad_data,pf.fgauss_no_offset,
                                    [1.0,1.0,0],pf.fgauss_no_offset_unproject,
                                    fluor_x,fluor_data,pf.fgauss_no_offset,
                                    [1e6,1.0,0],pf.fgauss_no_offset_unproject,
                                    detector_negative=True,abs_scaling=1.0,
                                    rad_center_index=None,fluor_center_index=None) 
    plt.plot(x,signal_trap_proj_trans,'r.')
    #Cross-check
    plt.plot(x,1.0-scipy.stats.norm.cdf(x,0,sigma)*abs_peak,'g')
    plt.xlim(-1,1)
    plt.ylim(1-abs_peak,1)
    plt.show()
    #Now, check the attenuation correction
    z,atten_proj_trans = fcompute_attenuation_correction([-1,3],[-2,3],1201,1201,
                                    rad_x,rad_data,pf.fgauss_no_offset,
                                    [1.0,1.0,0],pf.fgauss_no_offset_unproject,
                                    fluor_x,fluor_data,pf.fgauss_no_offset,
                                    [1e6,1.0,0],pf.fgauss_no_offset_unproject,
                                    detector_negative=True,abs_scaling=1.0,
                                    rad_center_index=None,fluor_center_index=None) 
    plt.figure()
    plt.plot(z,atten_proj_trans,'r.')
    #Cross-check
    plt.plot(z,1.0-scipy.stats.norm.cdf(z,0,sigma)*abs_peak,'g')
    plt.xlim(-1,1)
    plt.ylim(1-abs_peak,1)
    plt.show()

#Test code
if __name__ == '__main__':
    ftest_code()