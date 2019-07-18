'''Module to compute the signal trapping given a fitted 2D distribution of the
absorption.  Creates a 2D array of absorption, with the beam propagation 
direction along the columns.  Assumes that the detector is at 90 degrees to 
the propagation direction, so we can just sum along rows to get the amount 
of absorption in each row.  

Alan Kastengren, XSD, APS

Started: September 15, 2014

Changes
October 26, 2014: in fcalculate_signal_trapping_2D, add negative sign for detector_negative=False
'''
import numpy as np
import scipy.integrate
import Projection_Fit as pf
import matplotlib.pyplot as plt
import scipy.stats

def fcreate_absorption_array(x_lims, z_lims,num_x,num_z):
    '''Creates a 2D array for performing signal trapping corrections.
    Assumes that the beam propagates along the z directiion.
    Parameters:
    x_lim: limits of the array in the x (row) direction as a two-entry list
    z_lim: limits of the array in the z (column) direction as a two-entry list
    num_x, num_z: number of array elements in the x and z directions, respectively.
    '''
    x_vals = np.linspace(x_lims[0],x_lims[1],num_x)
    z_vals = np.linspace(z_lims[0],z_lims[1],num_z)
    return x_vals, z_vals, np.zeros((num_z,num_x))

def fpopulate_fitted_array(x_vals, z_vals, array2D, x_center, fit_function, parameters):
    '''Populate a 2D array to be used for signal trapping corrections.
    Uses the x_vals, z_vals to find radius from (x_center,0) for each point,
    then applies the fit_function with the given parameters.
    '''
    output = np.zeros_like(array2D)
    #Loop across rows of the 2D array
    for i in range(output.shape[0]):
        #Find the radius for each point
        r = np.sqrt((x_vals - x_center)**2 + z_vals[i]**2)
        output[i] = fit_function(r, parameters)
    return output

def fcalculate_signal_trapping_2D(x_vals,z_vals,density_2D,detector_negative=True):
    '''Calculate the total signal trapping for each point in the 2D array.
    The result should be in extinction lengths (e**-EL needed to get transmission).
    Integrates along each row (across the columns).
    detector_negative controls whether integration is to the negative x end of
    the row or to the +x end.  If True, integrate from -x to each point.
    '''
    #Define output array
    signal_trapping_array = np.zeros_like(density_2D)
    print "Density peak = " + str(np.max(density_2D))
    #Loop across rows and columns
    if detector_negative:
        signal_trapping_array = scipy.integrate.cumtrapz(density_2D,x_vals,initial=0)
    else:
        #Since we are doing this in reverse order, we need a negative sign
        signal_trapping_array[:,::-1] = -scipy.integrate.cumtrapz(density_2D[:,::-1],x_vals[::-1],initial=0)
    plt.imshow(signal_trapping_array)
    return signal_trapping_array

def fcalculate_signal_trapping_projection(x_vals,z_vals,density_2D,data_2D,detector_negative=True,tol=0):
    '''Calculates the projection data subject to signal trapping corrections.
    Parameters:
    x_vals, z_vals: coordinates
    density_2D: density of the signal trapping.  Integral gives extinction 
                lengths in the beam.
    data_2D: distribution of fluorescence
    detector_negative: if True, detector is on -x side.  If False, on +x side.
    Returns:
    x_vals: x values for the array, since it won't necessarily correspond to original data points.
    scipy.integrate...: transmission of fluorescence through sample for each x.
    '''
    #Find the map of the signal trapping
    signal_trapping_array = fcalculate_signal_trapping_2D(x_vals,z_vals,density_2D,detector_negative)
    #Convert this to transmission at each point
    signal_trapping_array = np.exp(-signal_trapping_array)
#     plt.figure()
#     plt.imshow(signal_trapping_array)
#     plt.colorbar()
    #Compute weighted average transmission.  Handle case where data_2D integral might be zero by making trans=1
    denominator = np.ones_like(x_vals)
    mask =  (np.abs(scipy.integrate.simps(data_2D,z_vals,axis=0)) > tol)
    denominator[mask] = scipy.integrate.simps(data_2D[:,mask],z_vals,axis=0)
    averaged_signal_trapping = scipy.integrate.simps(signal_trapping_array * data_2D,z_vals,axis=0)/denominator
    #Now, take into account that for some fits (ellipse fit, for example), denominator would be zero.
    #Trace back to another point that does have a good signal trapping value
    
    return x_vals, averaged_signal_trapping

def fcompute_signal_trapping(x_lims, z_lims,num_x,num_z,rad_fit_function,rad_fit_args,
                             fluor_fit_function,fluor_fit_args,x_center_rad=0,
                             x_center_fluor=0,detector_negative=True):
    '''Function to put all of it together.
    Create appropriate arrays to perform computations.
    Calculate signal trapping on a 2D slice, assuming axisymmetry.
    Return revised estimate of fluorescence signal.
    Variables:
    x_lims, z_lims, num_x, num_z: size and shape of arrays for computations
    rad_fit_function: function used to fit absorption, in terms of extinction lengths
    rad_fit_args: arguments for rad_fit_function to give correct distribution
    fluor_fit_function: function used to fit fluorescence
    fluor_fit_args: arguments for fluor_fit_function to give correct distribution
    x_center_(rad,fluor): x value where the axisymmetric distribution is centered
    detector_negative: if True, detector is on -x side of the experiment.
    Returns:
    x_vals: x values for the array, since it won't necessarily correspond to original data points.
    trap: transmission of fluorescence through sample for each x.
    signal_trapping_array: 2D array with transmission for each point.
    '''
    x_vals, z_vals, array2D = fcreate_absorption_array(x_lims, z_lims,num_x,num_z)
    absorption_array = fpopulate_fitted_array(x_vals, z_vals, array2D, x_center_rad, 
                                           rad_fit_function,rad_fit_args)
    fluor_array = fpopulate_fitted_array(x_vals, z_vals, array2D, x_center_fluor,
                                         fluor_fit_function,fluor_fit_args) 
    signal_trapping_array = np.exp(-fcalculate_signal_trapping_2D(x_vals,z_vals,absorption_array,detector_negative))
    x,trap = fcalculate_signal_trapping_projection(x_vals,z_vals,absorption_array,fluor_array,detector_negative)
    return x,trap,signal_trapping_array

def ftest_code(abs_peak = 0.04,sigma=.5,fluor_peak=10000000,fluor_sigma=0.05):
    '''Code to test whether I am calculating things correctly.
    '''
    plt.title('Signal Trapping')
    x,trap,signal_trapping_array = fcompute_signal_trapping([-3,3],[-3,3],1201,1201,
                             pf.fgauss_no_offset_unproject,[abs_peak*sigma*np.sqrt(2.0*np.pi),sigma],
                             pf.fgauss_no_offset_unproject,[fluor_peak,fluor_sigma])   
    plt.plot(x,trap)
    #Cross-check
    plt.plot(x,1.0-scipy.stats.norm.cdf(x,0,sigma)*abs_peak,'g')
    plt.xlim(-1,1)
    plt.ylim(1-abs_peak,1)
    plt.figure(2)
    plt.plot(x,-np.log(signal_trapping_array)[600,:],'b')
    plt.plot(x,scipy.stats.norm.cdf(x,0,sigma)*abs_peak,'g')
    plt.show()
    
    
def fdummy_function(r,params):
    return r

#Test code
if __name__ == '__main__':
    ftest_code()
        