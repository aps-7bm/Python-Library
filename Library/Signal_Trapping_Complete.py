'''Module to compute the signal trapping corrections.
This code is a combination of the previous Signal_Trapping_Fitted
and Signal_Trapping_Functions codes.

This code is aimed for reconstructions where the radiography signal can simply
be scaled to get the absorption profile for the fluorescence signal
trapping.  If, for example, a significant amount of the absorption in certain
regions is due to the fluorescent species, such that the shape of the 
absorption profile would change between the incident and emitted photon
energies, this approach must be altered.

In this code, the beam is assumed to propagate along the columns 
(i.e., across rows).  This is the z direction.  The x direction
is along rows (across columns).

Alan Kastengren, XSD, APS

Started: August 3, 2016

February 14, 2017: Major reworking of code.  Make objects to hold the
x, y, projection function, distribution function, and parameter guesses
for radiography and fluorescence to make code more readable.
'''

import matplotlib.pyplot as plt
import numpy as np
import Projection_Fit as pf
from scipy import optimize
from scipy import interpolate
import scipy.stats
import scipy.integrate
import ALK_Utilities as ALK

class SigTrapData():
    '''Class to hold the raw and fitting data for a dataset.
    This works for either radiography or fluorescence data.
    Main fields
    x: transverse location of projection points
    y: projection data
    proj_func: function to be used for fitting the projection data.
    dist_func: function to create axisymmetric density distribution from projection
    abs_ratio: ratio of attenuation of fluorescence
    center_index: which parameter gives the center of the distribution.
    '''
    def __init__(self,x,y,proj_func,dist_func,param_guesses,abs_ratio = 1,center_index=None):
        self.x = x
        self.y = y
        self.proj_func = proj_func
        self.dist_func = dist_func
        self.param_guesses = param_guesses
        self.abs_ratio = abs_ratio
        self.center_index = center_index
        self.fit_params = np.zeros_like(param_guesses)
        
    def ffit_projection(self):
        '''Computes a fit to the projection data.
        '''
        self.fit_params,__ = optimize.curve_fit(self.proj_func,self.x,
                                                self.y,self.param_guesses)
        
    def fdist_from_proj(self,x_vals,z_vals):
        '''Populate a 2D array to be used for signal trapping corrections
        based on a fit to the projection data.
        Inputs:
        x_vals,z_vals: coordinates for 2D array.
        scaling: scaling factor to apply to projection data.  Useful for 
                scaling absorption from incident to emission photon energy
        Output:
        2D array for distribution implied by projection fit.
        '''
        #Perform a fit to the projection data
        self.ffit_projection()
        #Find the x center from these parameters, or set to zero
        if self.center_index == None:
            x_center = 0
        else:
            x_center = self.fit_params[self.center_index]
        #Make a mesh and find the distances, then run through the fit function
        xmesh,zmesh = np.meshgrid(x_vals,z_vals)
        r = np.hypot(xmesh-x_center,zmesh)
        return self.dist_func(r,self.fit_params)

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
        plt.contourf(x_vals,z_vals,signal_trapping_array,101)
        plt.colorbar()
        plt.title("Local Signal Trapping")
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
        plt.title('Attenuation of Incident Beam')
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
    temp_signal_trapping = scipy.integrate.simps(signal_trapping_array * data_2D,z_vals,axis=0)/denominator
    #Mask off areas with no signal so we don't get divide by zero errors.
    averaged_signal_trapping = np.ones_like(x_vals)
    averaged_signal_trapping[mask] = temp_signal_trapping[mask]
    return x_vals, averaged_signal_trapping

def fcompute_signal_trapping_simple(x_lims, z_lims,num_x,num_z,
                                    rad_obj,fluor_obj,
                                    detector_negative=True):
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
    rad_obj: SigTrapData object holding data and fitting for radiography
    fluor_obj: SigTrapData object holding data and fitting for fluorescence
    detector_negative: if True, detector is on -x side of the experiment.
    Returns:
    x_vals: x values for the array, since it won't necessarily correspond to original data points.
    signal_trap_proj_trans: effective signal trapping transmission for each x.
    '''
    #Create the arrays we will use for the 
    x_vals, z_vals = fcreate_arrays(x_lims, z_lims,num_x,num_z)
    #Fit projections of radiography and fluorescence
    absorption_array = rad_obj.fdist_from_proj(x_vals, z_vals) * fluor_obj.abs_ratio
    fluor_array = fluor_obj.fdist_from_proj(x_vals, z_vals)
    
    #Compute the 2D distribution of signal trapping, converting to transmission
    signal_trapping_array = fcalculate_signal_trapping_2D(x_vals,z_vals,absorption_array,detector_negative)
    #Average this for each projection
    x,signal_trap_proj_trans = fprojection_weighted_average(x_vals,z_vals,fluor_array,signal_trapping_array)
    plt.plot(x,signal_trap_proj_trans)
    plt.show()
    
    return x,signal_trap_proj_trans

def fcompute_attenuation_correction(x_lims, z_lims,num_x,num_z,
                                    rad_obj,fluor_obj,
                                    detector_negative=True):
    '''Function to perform a correction
    of fluorescence due to attenuation within the sample.
    This code assumes that we have only a single projection for the 
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
    absorption_array = rad_obj.fdist_from_proj(x_vals, z_vals)
    fluor_array = fluor_obj.fdist_from_proj(x_vals, z_vals)

    #Compute the 2D distribution of attenuation for correcting the fluorescence
    attenuation_corr_array = fcalculate_attenuation_2D(x_vals,z_vals,absorption_array,detector_negative)
    #Average this for each projection
    x,atten_corr_proj_trans = fprojection_weighted_average(x_vals,z_vals,fluor_array,attenuation_corr_array)
    return x,atten_corr_proj_trans

def ftest_code(abs_peak = 0.4,orad=2.0,irad=1.0,fluor_peak=100000,fluor_rad=0.01,abs_ratio=2.0):
    '''Code to test whether I am calculating things correctly.
    '''
    #For test, radiography is an annulus 2 mm in outer radius, 1 mm inner radius
    #Keep in mind, to be truly hollow, area ~ radius**2
    x = np.linspace(-3,3,3601)
    outer_peak = abs_peak * orad / (orad - irad)
    inner_peak = -abs_peak * irad / (orad - irad)
    rad_params = [outer_peak * np.pi * orad / 2.0, orad, 0, 
                  inner_peak * np.pi * irad / 2.0, irad, 0]
    rad_proj_func = pf.fdouble_ellipse_fit_center_offset
    rad_dist_func = pf.fdouble_ellipse_fit_distribution
    rad_object = SigTrapData(x,rad_proj_func(x, *rad_params),rad_proj_func,rad_dist_func,rad_params)
    rads = np.linspace(0,3,301)
    #Plot the data
    plt.figure()    
    plt.subplot(1,2,1)
    plt.plot(rads,rad_dist_func(rads,rad_params))
    plt.title("Radiography Radial Distribution",fontsize=8)
    
    #For test, fluorescence is a constant density core
    fluor_params = [fluor_peak,fluor_rad,0]
    fluor_proj_func = pf.fellipse_fit_no_offset
    fluor_dist_func = pf.fellipse_fit_distribution
    fluor_object = SigTrapData(x,fluor_proj_func(x, *fluor_params),fluor_proj_func,fluor_dist_func,fluor_params,abs_ratio)
    plt.subplot(1,2,2)
    plt.plot(x,rad_object.y,'r',label='Radiography')
    plt.plot(x,fluor_object.y/np.max(fluor_object.y),'b',label="Scaled Fluor")
    plt.legend(loc = 'upper right',fontsize=6)
    plt.title("Input Proj Data",fontsize=8)
    plt.show()
    xs,signal_trap_proj_trans = fcompute_signal_trapping_simple([-3,3],[-3,3],1801,1801,
                                    rad_object,fluor_object,
                                    detector_negative=True) 
    #Do we get what we expected?
    x_center = np.argmin(np.abs(xs))
    print("Cross-check signal trapping in center:")
    print(np.exp(-abs_peak/2.0*abs_ratio),signal_trap_proj_trans[x_center])
    plt.figure(5)
    plt.plot(xs,signal_trap_proj_trans,'r.',label='Signal Trapping')
    plt.figure()
    #Cross-check
#     plt.plot(x,1.0-scipy.stats.norm.cdf(x,0,sigma)*abs_peak,'g')
#     plt.xlim(-1,1)
#     plt.ylim(1-abs_peak,1)
    #Now, check the attenuation correction
    z,atten_proj_trans = fcompute_attenuation_correction([-3,3],[-3,3],1801,1801,
                                    rad_object,fluor_object,
                                    detector_negative=True) 
    #Do we get what we expected?
    print("Cross-check incident attenuation in center:")
    print(np.exp(-abs_peak/2.0),atten_proj_trans[x_center])
    plt.figure(5)
    plt.plot(z,atten_proj_trans,'g.',label='Incident attenuation')
    plt.legend(loc='upper right',fontsize=6)

    plt.show()

#Test code
if __name__ == '__main__':
    ftest_code()