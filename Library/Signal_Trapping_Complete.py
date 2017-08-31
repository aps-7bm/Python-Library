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
import scipy.integrate
import logging


#Set up logging
logger = logging.getLogger('Signal_Trapping_Complete')
logger.addHandler(logging.NullHandler())

class SigTrapData():
    '''Class to hold the raw and fitting data for a material.
    This works for either radiography or fluorescence data.
    Main fields
    x: transverse location of projection points
    y: projection data
    proj_func: function to be used for fitting the projection data.
    dist_func: function to create axisymmetric density distribution from projection
    abs_ratio: ratio of attenuation of this material at incident energy to
                attenuation at fluorescence energy
    center_index: which parameter gives the center of the distribution.
    density_2D: 2D array giving our best estimate of distribution of the
                material.  Coordinates are (z,x) for each point.
    sig_trap_2D: array giving the signal trapping due to this material.
    atten_2D: array giving the incident beam attenuation due to this material.
    x_vals (z_vals): x (z) values for the 2D arrays.
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
        self.density_2D = None
        self.sig_trap_2D = None
        self.atten_2D = None
        self.x_vals = None
        self.z_vals = None
        self.density_2D_corrected = None
        self.y_corrected = None
        
    def ffit_projection(self):
        '''Computes a fit to the projection data.
        '''
        self.fit_params,__ = optimize.curve_fit(self.proj_func,self.x,
                                                self.y,self.param_guesses)
        
    def fdist_from_proj(self):
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
        self.x_vals = np.linspace(x_min,x_max,points_2D )
        self.z_vals = np.linspace(x_min,x_max,points_2D)
        #Make a mesh and find the distances, then run through the fit function
        xmesh,zmesh = np.meshgrid(self.x_vals,self.z_vals)
        r = np.hypot(xmesh-x_center,zmesh)
        self.density_2D = self.dist_func(r,self.fit_params)
    
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
    '''Calculates the projection data from the corrected 2D data.
    Uses linear interpolation to get data at original x points.
    Results saved in y_corrected variable of input SigTrapData object.
    Inputs:
    fluor_obj: the SigTrapData object for the fluorescence variable
    '''
    projected_corrected = np.sum(fluor_obj.density_2D_corrected,axis=0)
    interp_func = interpolate.interp1d(fluor_obj.x_vals,projected_corrected)
    fluor_obj.y_corrected = interp_func(fluor_obj.x)

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
    
def fcorrect_fluorescence_proj_data(fluor_obj,atten_objects):
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
        at_obj.fcalculate_signal_trapping_2D()
    
    #For the fluorescence object, form the projection
    fluor_obj.fdist_from_proj()
    
    #Perform correction to find corrected 2D array of fluor
    fcompute_corrected_fluor_2D(fluor_obj,atten_objects)
    
    #Find the projection of the fluorescence 2D array
    fprojection_corrected_fluor(fluor_obj)

def ftest_code(abs_peak = 0.4,orad=2.0,irad=1.0,fluor_peak=100000,fluor_rad=0.01,abs_ratio=2.0,
               fluor_peak_abs=0.2,display=False,center_shift=0):
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
    rad_object = SigTrapData(x,rad_proj_func(x, *rad_params),rad_proj_func,rad_dist_func,rad_params,abs_ratio,2)
    rads = np.linspace(0,3,301)
    
    
    #For test, fluorescence is a constant density core
    fluor_params = [fluor_peak_abs,fluor_rad,center_shift]
    fluor_proj_func = pf.fellipse_fit_no_offset
    fluor_dist_func = pf.fellipse_fit_distribution
    fluor_object = SigTrapData(x,fluor_proj_func(x, *fluor_params),fluor_proj_func,fluor_dist_func,fluor_params,abs_ratio,2)
    fluor_abs_params = [0.2 * np.pi * fluor_rad / 2.0, fluor_rad, center_shift]
    fluor_abs_object = SigTrapData(x,fluor_proj_func(x, *fluor_abs_params),fluor_proj_func,fluor_dist_func,fluor_params,abs_ratio,2)
    
    #Perform the signal trapping and attenuation corrections
    fcorrect_fluorescence_proj_data(fluor_object, [rad_object,fluor_abs_object])
    
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
        plt.figure()
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
        plt.plot(fluor_object.x_vals,fluor_object.sig_trap_2D[:,center_x],'r')
        plt.plot(fluor_object.x_vals,fluor_object.atten_2D[center_z,:],'b')
        plt.show()

#Test code
if __name__ == '__main__':
    ftest_code(fluor_rad = 0.8,display=True,center_shift=-0.5)
    ftest_code(fluor_rad = 0.3)
    ftest_code(fluor_rad = 0.3, abs_ratio = np.pi)