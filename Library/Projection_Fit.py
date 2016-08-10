'''Module to provide fitting of projection data to curves that can be
inverted to object space analytically.

Alan Kastengren, XSD, APS

Started: September 15, 2014

'''

import numpy as np
#import matplotlib.pyplot as plt
#import scipy.optimize as sop

def fgauss_no_offset(x_values, area, sigma, center):
    '''Computes a Gaussian curve, with no vertical offset.
    Parameters list:
    area = Area under curve
    sigma = sigma of Gaussian
    center = center offset
    '''
    return area/(sigma *np.sqrt(2*np.pi))*np.exp(-(x_values-center)**2/(2.0*sigma**2))

def fgauss_no_offset_unproject(radii, parameters):
    '''Compute the distribution, assuming axisymmetry, from a Gaussian
    fit to the projection.  Use a list of parameters, as this is the output
    of scipy.optimize.curve_fit
    parameters = [area, sigma, center]; could also handle a vertical offset
    '''
    return parameters[0] / (2.0 * np.pi * parameters[1]**2) * np.exp(
            -radii**2 / (2.0 * parameters[1]**2))

def fgauss_vertical_offset(x_values, area, sigma, center, offset):
    '''Computes a Gaussian curve, with a vertical offset.
    Parameters list:
    area = Area under curve
    sigma = sigma of Gaussian
    center = center offset
    offset = vertical offset
    '''
    return area/(sigma *np.sqrt(2*np.pi))*np.exp(-(x_values-center)**2/(2.0*sigma**2)) + offset

def fgauss_sloped_offset(x_values,area,sigma,center,offset,slope):
    '''Computes a Gaussian curve, with a vertical offset and sloped baseline.
    Parameters list:
    area = Area under curve
    sigma = sigma of Gaussian
    center = center offset
    offset = vertical offset at x = 0
    slope = slope of the baseline
    '''
    return area/(sigma *np.sqrt(2*np.pi))*np.exp(-(x_values-center)**2/(2.0*sigma**2)) + offset + x_values * slope

def fdoublegauss_no_offset(x_values, area1, sigma1, center, area2, sigma2):
    '''Computes the sum of two Gaussian curves, with no vertical offset
    and the same center point.
    Parameters list:
    area1 = Area under curve 1
    sigma1 = sigma of Gaussian 1
    center = center offset
    area2 = Area under curve 2
    sigma2 = sigma of Gaussian 2
    '''
    return (fgauss_no_offset(x_values, area1, sigma1, center) + 
            fgauss_no_offset(x_values, area2, sigma2, center))

def fdoublegauss_no_offset_unproject(radii, parameters):
    '''Compute the distribution, assuming axisymmetry, from a Gaussian
    fit to the projection.  Use a list of parameters, as this is the output
    of scipy.optimize.curve_fit
    parameters = [area1, sigma1, center, area2, sigma2]; could also handle a vertical offset
    '''
    return fgauss_no_offset_unproject(radii, parameters[:2]) + fgauss_no_offset_unproject(radii, parameters[3:])
    
def fparabolic_density_projection(x_values, area, radius, center):
    '''Computes the curve resulting from the projection of a parabolic density distribution.
    Parameters list:
    area = Area under curve 1
    radius = r where density reaches zero
    center = center offset
    '''
    #Make sure we return zeros outside of the radius
    mask = np.abs(x_values - center) < radius
    output = np.zeros_like(x_values)
    output[mask] = area * 8.0 / (3.0 * radius**4 * np.pi) * (radius**2 - (x_values[mask] - center)**2)**1.5
    return output

def fparabolic_density_unproject(radii,parameters):
    '''Compute the distribution of a parabolic density distribution.
    density = peak_value * (1 - r**2/R**2)
    '''
    output = np.zeros_like(radii)
    peak_value = 2 * parameters[0] / np.pi / parameters[1]**2
    mask = radii < parameters[1]
    output[mask] = peak_value * (1 - radii[mask]**2 / parameters[1]**2)
    return output

def fdouble_parabolic_density_projection(x_values, area1, radius1, center, area2, radius2):
    '''Computes the curve resulting from the projection of two parabolic density distributions.
    Parameters list:
    area1 = Area under curve 1
    radius1 = r where density reaches zero
    center = center offset
    area2 = Area under curve 2
    radius2 = r where density reaches zero
    '''
    #Make sure we return zeros outside of the radius
    return (fparabolic_density_projection(x_values, area1, radius1, center) +
            fparabolic_density_projection(x_values, area1, radius1, center))

def fellipse_fit_no_offset(x_values, area, radius, center):
    '''Computes the top half of an ellipse, the curve resulting from
    the projection of a circle of constant density.
    '''
    mask = np.abs(x_values - center) < radius
    output = np.zeros_like(x_values)
    output[mask] = area * 2.0 / (radius * np.pi) * np.sqrt(1 - ((x_values[mask] - center)/radius)**2)
    return output

def fdouble_ellipse_fit_no_offset(x_values,area1,radius1,center,area2,radius2):
    '''Computes the top half of two overlapping ellipses, the curves resulting from
    the projection of two circles of constant density.
    '''
    return fellipse_fit_no_offset(x_values,area1,radius1,center) + fellipse_fit_no_offset(x_values,area2,radius2,center)

def fdouble_ellipse_fit_center_offset(x_values,area1,radius1,center1,area2,radius2,center2):
    '''Computes the top half of two overlapping ellipses, the curves resulting from
    the projection of two circles of constant density.
    '''
    return fellipse_fit_no_offset(x_values,area1,radius1,center1) + fellipse_fit_no_offset(x_values,area2,radius2,center2)

def fellipse_fit_distribution(radii,parameters):
    '''Computes the density distribution at input radii from ellipse fit.
    Just a constant density distribution.
    '''
    output = np.zeros_like(radii)
    mask = radii < parameters[1]
    output[mask] = parameters[0]/(np.pi * parameters[1]**2)
    return output

def fdouble_ellipse_fit_distribution(radii,parameters):
    '''Computes the density distribution at input radii from double ellipse fit.
    Just two constant density distributions.
    '''
    return fellipse_fit_distribution(radii,parameters[:3]) + fellipse_fit_distribution(radii,(parameters[3],parameters[4],parameters[2]))

def fdribinski_distribution(radii, vertical_scale, sigma, k):
    '''Computes the distribution curve from Dribinsk, et al., RSI, v. 73, #7
    '''
    return vertical_scale * (np.e/k**2)**(k**2)*(radii/sigma)**(2 * k**2)*np.exp(-(radii/sigma)**2)

def fdribinski_distribution_unproject(radii,parameters):
    '''Computes the 2D distribution for a Dribinski fit.
    Parameters in the form [vertical_scale,sigma,center,k]
    '''
    return fdribinski_distribution(radii,parameters[0],parameters[1],parameters[2],parameters[3])

def fdribinski_projection(x_values, vert_scale, sigma, center, k, tol = 1e-6):
    '''Computes the projection curve from Dribinsk, et al., RSI, v. 73, #7.
    '''
    #Get a singularity if x_values-center == 0.  Take care of this
    x_values[np.abs(x_values - center) < tol * sigma] += tol * sigma 
    coefficient_sum = 1
    for l in range(1,k**2+1):
        product = 1
        for m in range(1,l+1):
            product *= (k**2 + 1 - m) * (m - 0.5) / m
        coefficient_sum += ((x_values-center)/sigma)**(-2 * l) * product
    output = vert_scale * 2.0 * sigma * fdribinski_distribution(x_values-center,1,sigma,k) * coefficient_sum
    return output

def fdribinski_projection_k1(x_values, vert_scale, sigma, center):
    return fdribinski_projection(x_values, vert_scale, sigma, center, 1)

def fdribinski_projection_k1_offset(x_values, vert_scale, sigma, center, offset):
    return fdribinski_projection(x_values, vert_scale, sigma, center, 1) + offset

def fdribinski_distribution_k1(radii,parameters):
    return fdribinski_distribution(radii,parameters[0],parameters[1],1)

def fdribinski_projection_k2(x_values, vert_scale, sigma, center):
    return fdribinski_projection(x_values, vert_scale, sigma, center, 2)

def fdribinski_projection_k2_offset(x_values, vert_scale, sigma, center,offset):
    return fdribinski_projection(x_values, vert_scale, sigma, center, 2) + offset

def fdribinski_distribution_k2(radii,parameters):
    return fdribinski_distribution(radii,parameters[0],parameters[1],2)

def fdribinski_projection_k3(x_values, vert_scale, sigma, center):
    return fdribinski_projection(x_values, vert_scale, sigma, center, 3)

def fdribinski_projection_k3_offset(x_values, vert_scale, sigma, center,offset):
    return fdribinski_projection(x_values, vert_scale, sigma, center, 3) + offset

def fdribinski_distribution_k3(radii,parameters):
    return fdribinski_distribution(radii,parameters[0],parameters[1],3)

def fdribinski_projection_k4(x_values, vert_scale, sigma, center):
    return fdribinski_projection(x_values, vert_scale, sigma, center, 4)

def fdribinski_projection_k4_offset(x_values, vert_scale, sigma, center,offset):
    return fdribinski_projection(x_values, vert_scale, sigma, center, 4) + offset

def fdribinski_distribution_k4(radii,parameters):
    return fdribinski_distribution(radii,parameters[0],parameters[1],4)

def fdribinski_projection_k5(x_values, vert_scale, sigma, center):
    return fdribinski_projection(x_values, vert_scale, sigma, center, 5)

def fdribinski_distribution_k5(radii,parameters):
    return fdribinski_distribution(radii,parameters[0],parameters[1],5)
def fdribinski_k5_gaussian_sum(x_values, vert_scale, sigma, center, gauss_area, gauss_sigma):
    return (fdribinski_projection(x_values, vert_scale, sigma, center, 5) +
            fgauss_no_offset(x_values, gauss_area, gauss_sigma, center)) 

def fdistribution_dribinski_k5_gaussian_sum(radii,parameters):
    '''Computes 2D distribution at given radii from a fit of the sum of a 
    Gaussian and a Dribinski k=5 curve.
    '''
    return fdribinski_distribution(radii,parameters[0],parameters[1],5) + fgauss_no_offset_unproject(radii, (parameters[3],parameters[4]))

def fdribinski_k2_gaussian_sum(x_values, vert_scale, sigma, center, gauss_area, gauss_sigma):
    return (fdribinski_projection(x_values, vert_scale, sigma, center, 2) +
            fgauss_no_offset(x_values, gauss_area, gauss_sigma, center)) 

def fdistribution_dribinski_k2_gaussian_sum(radii,parameters):
    '''Computes 2D distribution at given radii from a fit of the sum of a 
    Gaussian and a Dribinski k=5 curve.
    '''
    return fdribinski_distribution(radii,parameters[0],parameters[1],2) + fgauss_no_offset_unproject(radii, (parameters[3],parameters[4]))

def fdribinski_k1_gaussian_sum(x_values, vert_scale, sigma, center, gauss_area, gauss_sigma):
    return (fdribinski_projection(x_values, vert_scale, sigma, center, 1) +
            fgauss_no_offset(x_values, gauss_area, gauss_sigma, center)) 

def fdistribution_dribinski_k1_gaussian_sum(radii,parameters):
    '''Computes 2D distribution at given radii from a fit of the sum of a 
    Gaussian and a Dribinski k=5 curve.
    '''
    return fdribinski_distribution(radii,parameters[0],parameters[1],1) + fgauss_no_offset_unproject(radii, (parameters[3],parameters[4]))

def fdribinski_k3_gaussian_sum(x_values, vert_scale, sigma, center, gauss_area, gauss_sigma):
    return (fdribinski_projection(x_values, vert_scale, sigma, center, 3) +
            fgauss_no_offset(x_values, gauss_area, gauss_sigma, center)) 

def fdistribution_dribinski_k3_gaussian_sum(radii,parameters):
    '''Computes 2D distribution at given radii from a fit of the sum of a 
    Gaussian and a Dribinski k=5 curve.
    '''
    return fdribinski_distribution(radii,parameters[0],parameters[1],3) + fgauss_no_offset_unproject(radii, (parameters[3],parameters[4]))

def fdribinski_k2_k3_sum(x_values, vert_scale2, sigma2, center, vert_scale3, sigma3):
    return (fdribinski_projection(x_values, vert_scale2, sigma2, center, 2) +
            fdribinski_projection(x_values, vert_scale2, sigma2, center, 3))
    
def fdribinski_k2_k5_sum(x_values, vert_scale2, sigma2, center, vert_scale5, sigma5):
    return (fdribinski_projection(x_values, vert_scale2, sigma2, center, 2) +
            fdribinski_projection(x_values, vert_scale5, sigma5, center, 5))

def fgauss_ellipse_sum(x_values,area, sigma, center,area_el,radius):
    return fgauss_no_offset(x_values, area, sigma, center) + fellipse_fit_no_offset(x_values, area_el, radius, center)

def fgauss_ellipse_sum_unproject(radii,parameters):
    ellipse_params = [parameters[3],parameters[4],parameters[2]]
    return fgauss_no_offset_unproject(radii, parameters[:3]) + fellipse_fit_distribution(radii, ellipse_params)

def fellipse_parabola_sum(x_values,area, radius, center,area_p,radius_p):
    return fparabolic_density_projection(x_values, area_p, radius_p, center) + fellipse_fit_no_offset(x_values, area, radius, center)

def fellipse_parabola_sum_unproject(radii,parameters):
    para_params = [parameters[3],parameters[4],parameters[2]]
    return fellipse_fit_distribution(radii, parameters[:3]) + fparabolic_density_unproject(radii, para_params)

'''
sigma = 1.0
r = np.linspace(0,10,1001)
x = np.linspace(0,10,1001)
f = np.exp(-r**2/sigma**2)/sigma/np.pi*(r**2-sigma**2/2)
g = np.exp(-r**2/sigma**2)*r**2
#plt.plot(r,f)
#plt.plot(r,g)
#plt.figure(2)
#g_star = sigma*(x**2 + sigma**2/2.0)*np.sqrt(np.pi)*np.exp(-x**2/sigma**2)
#plt.plot(x,g_star)
plt.figure(3)
q = np.e*r**2*np.exp(-r**2)
h = (np.e/4.0)**4*r**8*np.exp(-r**2)
plt.plot(r,q,label='k=1')
plt.plot(r,h,label='k=2')
s = fdribinski_distribution(r,1,1,1)
plt.plot(r,fdribinski_distribution(r,1,1,1),label='k=1,Function')
plt.plot(r,fdribinski_distribution(r,1,1,2),label='k=2,Function')
plt.plot(r,fdribinski_distribution(r,1,1,3),label='k=3,Function')
plt.plot(r,fdribinski_distribution(r,1,1,4),label='k=4,Function')
plt.legend()
plt.title('Dribinski')
plt.figure(4)
plt.plot(x,fdribinski_projection(x,1,1,0,1),label='k=1, Function')
plt.plot(x,fdribinski_projection(x,1,1,0,2),label='k=2, Function')
plt.plot(x,fdribinski_projection(x,1,1,0,3),label='k=3, Function')
plt.plot(x,fdribinski_projection(x,1,1,0,4),label='k=4, Function')
plt.plot(x,fdribinski_projection(x,1,1,0,5),label='k=5, Function')
plt.plot(x,fdribinski_projection(x,1,1,0,6),label='k=6, Function')
plt.legend(loc = 'lower left',fontsize=8)
plt.figure(5)
plt.plot(x,fdribinski_projection(x,1,1,0,4),label='k=1, Function')
plt.plot(x,fdribinski_projection(x,1,2,0,4),label='k=2, Function')
plt.plot(x,fdribinski_projection(x,1,3,0,4),label='k=3, Function')
plt.plot(x,fdribinski_projection(x,1,4,0,4),label='k=4, Function')
plt.legend(loc = 'lower left',fontsize=8)
plt.show()'''