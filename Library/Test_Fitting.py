'''Test fitting on an AFRL scan.
'''
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
import h5py
import Projection_Fit as pf
import ALK_Utilities as ALK

file_path = ALK.fcorrect_path_start() + 'SprayData/Cycle_2014_2/AFRL_Edwards/'
file_name = '7bmb1_1128.hdf5'

with h5py.File(file_path+file_name,'r') as hdf_file:
    x_values = hdf_file['7bmb1:m26.VAL'][...]
    fluor_data = hdf_file['Kr_Kalpha'][...]
    plt.plot(x_values,fluor_data,'r.')
    #Now, fit to Dribinski curve with k = 1
    [vert_scale, sigma, center,gauss_area,gauss_sigma],param_covar = scipy.optimize.curve_fit(
                                                pf.fdribinski_k3_gaussian_sum,
                                              x_values, fluor_data,[1,4/3.0,0,5,1])
    print vert_scale, sigma, center,gauss_area,gauss_sigma
    print np.sqrt(np.diag(param_covar))
    fluor_fit = pf.fdribinski_projection_k3(x_values, vert_scale, sigma, center) + pf.fgauss_no_offset(x_values, gauss_area, gauss_sigma, center)
    plt.plot(x_values,fluor_fit,'b.')
    plt.figure(2)
    radii = np.linspace(0,np.max(np.abs(x_values)),101)
    plt.plot(radii,pf.fdribinski_distribution(radii, vert_scale, sigma, 3)+pf.fgauss_no_offset_unproject(radii, [gauss_area,gauss_sigma,center]))
    plt.show()
    