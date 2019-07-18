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

September 15, 2017: addition of PyAbel-based projection inversion, as well as 
many bug fixes.
'''

import matplotlib.pyplot as plt
import numpy as np
import Projection_Fit as pf
from scipy import optimize
from scipy import interpolate
import scipy.integrate
import logging
import abel


#Set up logging
logger = logging.getLogger('Signal_Trapping_Complete')
logger.addHandler(logging.NullHandler())

class SigTrapData():
    '''Class to hold the raw and fitting data for a material.
    This works for either radiography or fluorescence data.
    Note for 2D arrays: they are stored as rows being different z values
    and columns being different x values.  The numpy.meshgrid function
    calls this "Cartesian" ordering.
    Main fields
    x: transverse location of projection points
    y: projection data
    proj_func: function to be used for fitting the projection data.
    dist_func: function to create axisymmetric density distribution from projection
    abs_ratio: ratio of attenuation of this material at incident energy to
                attenuation at fluorescence energy
    center_index: which parameter gives the center of the distribution.
    abel: flag for whether to use Abel inversion (PyAbel) or fitting (False)
    density_2D: 2D array giving our best estimate of distribution of the
                material.  Coordinates are (z,x) for each point.
    sig_trap_2D: array giving the signal trapping due to this material.
    atten_2D: array giving the incident beam attenuation due to this material.
    x_vals (z_vals): x (z) values for the 2D arrays.
    '''
    def __init__(self,x,y,proj_func,dist_func,param_guesses,abs_ratio = 1,center_index=None,abel=False):
        self.x = x
        self.y = y
        self.proj_func = proj_func
        self.dist_func = dist_func
        self.param_guesses = param_guesses
        self.abs_ratio = abs_ratio
        self.center_index = center_index
        self.fit_params = np.zeros_like(param_guesses)
        self.density_2D = None
        self.sig_trap_2D = None
        self.atten_2D = None
        self.x_vals = None
        self.z_vals = None
        self.density_2D_corrected = None
        self.y_corrected = None
        self.abel = abel
        
    def ffit_projection(self):
        '''Computes a fit to the projection data.
        '''
        self.fit_params,__ = optimize.curve_fit(self.proj_func,self.x,
                                                self.y,self.param_guesses)
        
    def fdist_from_proj(self):
        if self.abel:
            self.fdist_from_proj_Abel()
        else:
            self.fdist_from_proj_fit()
    
    def fdist_from_proj_fit(self):
        '''Populate a 2D array to be used for signal trapping corrections
        based on a fit to the projection data.
        
        '''
        #Perform a fit to the projection data
        self.ffit_projection()
        #Find the x center from these parameters, or set to zero
        if self.center_index == None:
            x_center = 0
        else:
            x_center = self.fit_params[self.center_index]
        #Find the minimum spacing between x points
        min_spacing = np.min(np.diff(self.x))
        #Make arrays for the x and z values for the 2D array
        x_min = np.mean(self.x) - 2.0 * (np.mean(self.x)-np.min(self.x))
        x_max = np.mean(self.x) + 2.0 * (np.max(self.x)-np.mean(self.x))
        #Figure out how many points to use.  Use half of min spacing
        points_2D = np.rint((x_max - x_min) / min_spacing * 3.0) + 1
        self.x_vals = np.linspace(x_min,x_max,points_2D)
        self.z_vals = np.linspace(x_min,x_max,points_2D)
        #Make a mesh and find the distances, then run through the fit function
        xmesh,zmesh = np.meshgrid(self.x_vals,self.z_vals)
        r = np.hypot(xmesh-x_center,zmesh)
        self.density_2D = self.dist_func(r,self.fit_params)
        
    def fdist_from_proj_Abel(self):
        '''Populate the 2D arrays to be used for signal trapping.
        Use an Abel inversion, which removes the requirement for supplying
        fitting functions and guess coefficients.
        This does assume axisymmetry.
        Based on the PyAbel package.
        '''
        #Compute a one-sided distribution from the input projection
        x_one_side, y_one_side, com = self.faxi_data_one_size(self.x, self.y)
        dx = np.min(np.diff(x_one_side))
        #Perform the Abel inversion to get the radial profile
        radial = abel.basex.basex_transform(y_one_side,dr=dx)
        self.dist_func = scipy.interpolate.interp1d(x_one_side,radial,fill_value=0.,bounds_error=False)
        #Make arrays for the x and z values for the 2D array
        x_min = np.mean(self.x) - 2.0 * (np.mean(self.x)-np.min(self.x))
        x_max = np.mean(self.x) + 2.0 * (np.max(self.x)-np.mean(self.x))
        #Figure out how many points to use.  Use half of min spacing
        points_2D = np.rint((x_max - x_min) / dx * 3.0) + 1
        self.x_vals = np.linspace(x_min,x_max,points_2D)
        self.z_vals = np.linspace(x_min,x_max,points_2D)
        #Make a mesh and find the distances, then run through the fit function
        xmesh,zmesh = np.meshgrid(self.x_vals,self.z_vals)
        r = np.hypot(xmesh-com,zmesh)
        self.density_2D = self.dist_func(r)
    
    def faxi_data_one_size(self,x_values,y_values):
        '''Take a dataset of (x,y) values and make a one-sided transverse
        distribution assuming axisymmetry.
        Finds the center of mass and averages on either side of it.
        This code allows the center of mass to be between x values.
        '''
        #Compute the center of mass of the y data.  This is our center value.
        com = np.trapz(y_values * x_values, x_values) / np.trapz(y_values, x_values)
        center_x = np.sum(x_values) / float(x_values.shape[0])
        #Print data if the center of mass is really weird
        if np.abs(com - center_x) > (0.5 * np.max(x_values) - np.min(x_values)):
            print("Invalid center of mass!")
            logger.error("Invalid center of mass")
            com = center_x
        #Add together the right- and left-hand edges of the data.
        #Find the distance to the farthest edge
        dist_to_edge = np.max([np.max(x_values) - com, com - np.min(x_values)])
        #Make an x array at same delta x as the original x array, but centered on com
        dx = np.min(np.diff(x_values))
        if not np.all(np.isfinite(np.ceil(dist_to_edge / dx))):
            print("Invalid calculation number on each side of com.")
            logger.error("Invalid calculation number on each side of com.")
            logger.error('com = {0:f}'.format(com))
            logger.error(x_values)
            logger.error(y_values)
            logger.error("dx = {0:f}".format(dx))
            logger.error("dist_to_edge = {0:f}".format(com))
            raise ValueError
        num_each_side = int(np.ceil(dist_to_edge / dx))
        
        #Make sure we'll have an odd number of radial points in the end for Abel code
        if num_each_side % 2 == 0:
            num_each_side += 1
        try:
            new_x = np.linspace(com - num_each_side * dx, com + num_each_side * dx, 2 * num_each_side + 1)
        except ValueError:
            print("Invalid number on each side.")
            logger.error("Invalid number on each side.")
            logger.error(com)
            logger.error(x_values)
            logger.error(y_values)
            logger.error('dist to edge = {0:f}'.format(dist_to_edge))
            plt.plot(x_values,y_values)
            plt.show()
            raise ValueError
        #Interpolate from the old (x,y) onto the new x
        interp_func = scipy.interpolate.interp1d(x_values, y_values ,fill_value='extrapolate')
        new_y = interp_func(new_x)
        #Make this a one-sided transverse profile, adding the two sides
        x_one_side = new_x[num_each_side:] - com
        y_one_side = np.zeros(num_each_side + 1)
        y_one_side[0] = new_y[num_each_side]
        for i in range(1,num_each_side):
            y_one_side[i] = (new_y[num_each_side + i] + new_y[num_each_side - i]) / 2.0
        return x_one_side, y_one_side, com
        
    def fcalculate_signal_trapping_2D(self,detector_negative=True):
        '''Calculate the total signal trapping for each point in the 2D array.
        The result is in extinction lengths (e**-EL needed to get transmission).
        Integrates along each row (across the columns).
        Inputs:
        detector_negative: controls whether integration is to the negative x end of
                the row or to the +x end.  If True, integrate from -x to each point.
        display: if True, plot the distribution of signal trapping.
        
        '''
        #Define output array: 2D map of ext lengths to the detector.
        self.sig_trap_2D = np.zeros_like(self.density_2D)
        #Do an integration along axis 1: 
        if detector_negative:
            self.sig_trap_2D = scipy.integrate.cumtrapz(self.density_2D * self.abs_ratio,self.x_vals,initial=0,axis=1)
        else:
            #Since we are doing this in reverse order, we need a negative sign
            self.sig_trap_2D[:,::-1] = -scipy.integrate.cumtrapz(self.density_2D[:,::-1] * self.abs_ratio,
                                                                 self.x_vals[::-1],initial=0,axis=1)
    
    def fcalculate_attenuation_2D(self):
        '''Calculate the incident beam attenuation for each point in the 2D array.
        The result is in extinction lengths (e**-EL needed to get transmission).
        Integrates along beam path (along the columns).        
        '''
        #Define output array: 2D map of ext lengths to the detector.
        self.atten_2D = np.zeros_like(self.density_2D)
        #Do an integration along axis 0: 
        self.atten_2D = scipy.integrate.cumtrapz(self.density_2D,self.z_vals,initial=0,axis=0)

def fprojection_corrected_fluor(fluor_obj):
    '''Calculates the projection data corrected for signal trapping and incident attenuation.
    Uses linear interpolation to get data at original x points.
    This calculation finds the ratio between the original and corrected values
    as projections of the 2D density, then finds the ratio and applies this
    to the original projection data.  In this way, we aren't assuming our
    axisymmetric functional form on the corrected data, just on the 
    signal trapping correction arrays.
    Results saved in y_corrected variable of input SigTrapData object.
    
    Inputs:
    fluor_obj: the SigTrapData object for the fluorescence variable
    '''
    #Compute the ratio between the fitted projection data before and after signal trapping.
    corrected_proj_fit = scipy.integrate.trapz(fluor_obj.density_2D_corrected,fluor_obj.z_vals,axis=0)
    original_proj_fit = scipy.integrate.trapz(fluor_obj.density_2D,fluor_obj.z_vals,axis=0)
    #Look over region where the original proj fit is at least 10^-3 of max value
    mask = original_proj_fit > 1e-3 * np.max(original_proj_fit)
    proj_fit_max_loc = np.argmax(original_proj_fit)
    sig_trap_ratio = np.zeros_like(corrected_proj_fit)
    sig_trap_ratio[mask] = corrected_proj_fit[mask] / original_proj_fit[mask]
    #Take care of regions where fits might be zero
    for i in range(proj_fit_max_loc,0,-1):
        if sig_trap_ratio[i] == 0:
            sig_trap_ratio[i] = sig_trap_ratio[i+1]
    for i in range(proj_fit_max_loc,sig_trap_ratio.shape[0],1):
        if sig_trap_ratio[i] == 0:
            sig_trap_ratio[i] = sig_trap_ratio[i-1]
    interp_func = interpolate.interp1d(fluor_obj.x_vals,sig_trap_ratio)
    interp_sig_trap_ratio = interp_func(fluor_obj.x)
    fluor_obj.y_corrected = fluor_obj.y * interp_sig_trap_ratio

def fcompute_corrected_fluor_2D(fluor_obj,attenuation_objs,tol=0):
    '''Calculates the 2D fluorescence distribution after corrections.
    Takes a list of SigTrapData objects for materials that attenuate the
    incident beam and fluorescence.  Multiplies all sig_trap_2D and 
    atten_2D arrays, then applies to the fluorescence data.
    Inputs:
    fluor_obj: the SigTrapData object for the fluorescence variable
    attenuation_objs: list of SigTrapData objects for the attenuation.
    tol: minimum fluorescence signal on a projection for which to calculate signal trapping. 
        Used to avoid divide by zero problems.
    '''
    if len(attenuation_objs) < 1:
        logger.error("No attenuation objects specified.  Returning.")
        
    #Make arrays in the fluor_obj for signal trapping and attenuation
    fluor_obj.atten_2D = np.zeros_like(fluor_obj.density_2D)
    fluor_obj.sig_trap_2D = np.zeros_like(fluor_obj.density_2D)
    
    #Loop through the attenuation objects
    for at_obj in attenuation_objs:
        #Interpolate the atten_2D back onto the (x,z) grid of fluor_obj
        interp_atten = interpolate.RectBivariateSpline(at_obj.x_vals,at_obj.z_vals,at_obj.atten_2D)
        fluor_obj.atten_2D += interp_atten(fluor_obj.x_vals,fluor_obj.z_vals)
        logger.debug("Incident attenuation component added.")
        #Interpolate the sig_trap_2D back onto the (x,z) grid of fluor_obj
        interp_sig_trap = interpolate.RectBivariateSpline(at_obj.x_vals,at_obj.z_vals,at_obj.sig_trap_2D)
        fluor_obj.sig_trap_2D += interp_sig_trap(fluor_obj.x_vals,fluor_obj.z_vals)
        logger.debug("Signal trapping component added.")
    #Divide the density_2D by the attenuation and signal trapping transmission arrays
    #to correct back to what the distribution should have been.
    fluor_obj.density_2D_corrected = fluor_obj.density_2D / np.exp(-fluor_obj.sig_trap_2D - fluor_obj.atten_2D)
    
def fcorrect_fluorescence_proj_data(fluor_obj,atten_objects,detector_negative=True,abel=False):
    '''Perform the signal trapping and attenuation corrections for projection data.
    To use this code, form SigTrapData objects for the fluorescence and 
    all of the materials that will attenuate the incident beam and 
    fluorescence.  Feed these into this code.
    
    Inputs:
    fluor_obj: SigTrapData object for the fluorescence data.
    atten_obj: list of SigTrapData objects for materials that attenuate the
                incident beam and fluorescence.
    '''
    #Check that we actually have attenuation objects.
    if len(atten_objects) < 1:
        logger.error("No attenuation objects specified.  Returning.")
    #Loop through the attenuation objects
    for at_obj in atten_objects:
        #Make a distribution from the projections
        at_obj.fdist_from_proj()
        #Compute the 2D arrays for signal trapping and attenuation
        at_obj.fcalculate_attenuation_2D()
        at_obj.fcalculate_signal_trapping_2D(detector_negative)
    #For the fluorescence object, form the projection
    fluor_obj.fdist_from_proj()
    #Perform correction to find corrected 2D array of fluor
    fcompute_corrected_fluor_2D(fluor_obj,atten_objects)
    #Find the projection of the fluorescence 2D array
    fprojection_corrected_fluor(fluor_obj)

def ftest_code(abs_peak = 0.4,orad=2.0,irad=1.0,fluor_peak=100000,fluor_rad=0.01,abs_ratio=2.0,
               fluor_peak_abs=0.2,display=False,center_shift=0,abel=False):
    '''Code to test whether I am calculating things correctly.
    '''
    #For test, radiography is an annulus 2 mm in outer radius, 1 mm inner radius
    #Keep in mind, to be truly hollow, area ~ radius**2
    x = np.linspace(-orad*1.5,orad*1.5,271)
    outer_peak = abs_peak * orad / (orad - irad)
    inner_peak = -abs_peak * irad / (orad - irad)
    rad_params = [outer_peak * np.pi * orad / 2.0, orad, center_shift, 
                  inner_peak * np.pi * irad / 2.0, irad, center_shift]
    rad_proj_func = pf.fdouble_ellipse_fit_center_offset
    rad_dist_func = pf.fdouble_ellipse_fit_distribution
    rad_object = SigTrapData(x,rad_proj_func(x, *rad_params),rad_proj_func,rad_dist_func,rad_params,abs_ratio,2,abel)
    rads = np.linspace(0,3,301)
    rad_object.fdist_from_proj()
    
    #For test, fluorescence is a constant density core
    fluor_params = [fluor_peak_abs,fluor_rad,center_shift]
    fluor_proj_func = pf.fellipse_fit_no_offset
    fluor_dist_func = pf.fellipse_fit_distribution
    fluor_object = SigTrapData(x,fluor_proj_func(x, *fluor_params),fluor_proj_func,fluor_dist_func,fluor_params,abs_ratio,2,abel)
    fluor_abs_params = [0.2 * np.pi * fluor_rad / 2.0, fluor_rad, center_shift]
    fluor_abs_object = SigTrapData(x,fluor_proj_func(x, *fluor_abs_params),fluor_proj_func,fluor_dist_func,fluor_params,abs_ratio,2,abel)
    
    #Perform the signal trapping and attenuation corrections
    fcorrect_fluorescence_proj_data(fluor_object, [rad_object,fluor_abs_object],abel)
    
    #Perform calculations to make sure our values make sense.
    expected_atten = (abs_peak + fluor_peak_abs) / 2.0
    center_x = np.argmin(np.abs(fluor_object.x_vals - center_shift))
    center_z = np.argmin(np.abs(fluor_object.z_vals))
    print("Expected attenuation EL of incident beam in center = " + str(expected_atten))
    print("Actual attenuation EL of incident beam in center = " + str(fluor_object.atten_2D[center_z,center_x]))
    expected_sig_trap = (abs_peak + fluor_peak_abs) * abs_ratio / 2.0
    print("Expected signal trapping EL of fluorescence in center = " + str(expected_sig_trap))
    print("Actual signal trapping EL of fluorescence in center = " + str(fluor_object.sig_trap_2D[center_z,center_x]))
    
    if display:
        #Plot the data
        plt.figure()    
        plt.subplot(1,2,1)
        plt.plot(rads,rad_dist_func(rads,rad_params))
        plt.title("Radiography Radial Distribution",fontsize=8)
        plt.subplot(1,2,2)
        plt.plot(x,rad_object.y,'r',label='Radiography')
        plt.plot(x,fluor_object.y/np.max(fluor_object.y),'b',label="Scaled Fluor")
        plt.plot(fluor_abs_object.x,fluor_abs_object.y,'g',label='Fluor Abs')
        plt.legend(loc = 'upper right',fontsize=6)
        plt.title("Input Proj Data",fontsize=8)
        plt.figure()
        plt.contourf(fluor_object.x_vals,fluor_object.z_vals,fluor_object.atten_2D,101)
        plt.colorbar()
        plt.title("Test Attenuation Array")
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.colorbar()
        plt.figure()
        plt.plot(fluor_object.x_vals,fluor_object.z_vals)
        plt.figure()
        #plt.imshow(fluor_object.sig_trap_2D)
        plt.contourf(fluor_object.x_vals,fluor_object.z_vals,fluor_object.sig_trap_2D,101)
        plt.colorbar()
        plt.title("Test Sig Trap Array")
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.figure()
        plt.contourf(fluor_object.x_vals,fluor_object.z_vals,fluor_object.density_2D,101)
        plt.colorbar()
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.figure()
        plt.contourf(fluor_object.x_vals,fluor_object.z_vals,fluor_object.density_2D_corrected,101)
        plt.colorbar()
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.figure()
        plt.plot(fluor_object.x_vals,fluor_object.sig_trap_2D[:,center_x],'r',label='Central Sig Trap')
        plt.plot(fluor_object.x_vals,fluor_object.atten_2D[center_z,:],'b',label='Central Atten')
        plt.legend(loc='upper right')
        plt.show()

#Test code
if __name__ == '__main__':
    ftest_code(fluor_rad = 0.8,display=True,center_shift=-0.5)
    ftest_code(fluor_rad = 0.8,display=True,center_shift=-0.5,abel=True)
    ftest_code(fluor_rad = 0.3)
    ftest_code(fluor_rad = 0.3,abel=True)
    ftest_code(fluor_rad = 0.3, abs_ratio = np.pi)
    ftest_code(fluor_rad = 0.3, abs_ratio = np.pi,abel=True)
    