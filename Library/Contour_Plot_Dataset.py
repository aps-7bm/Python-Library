'''Script to make a contour plot of a variable from a combined HDF5 file of
several scans.  

Alan Kastengren, XSD, APS

Started: October 21, 2014
'''
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import h5py
import numpy as np

#Default_values
file_path = '/data/Data/SprayData/Cycle_2014_2/AFRL_Edwards/'

def fcontour_plot_dataset(file_path,hdf_file_name,x_variable,y_variable,z_variable,
                          clims=None,num_levels=41):
    '''Script to make a contour plot of a dataset from an HDF5 files of several
    scans combined.
    '''
    #Create a new figure
    plt.figure()
    plt.suptitle(hdf_file_name)
    #Open file
    hdf_file = h5py.File(file_path + hdf_file_name,'r')
    #Mask off any NAN entries is x; indicates scan wasn't as wide as widest scan
    mask = np.logical_and(np.isfinite(hdf_file[x_variable]),hdf_file[x_variable][...]>0.5)
    #Make triangulation for Delauney triangulation plot
    triang = tri.Triangulation(hdf_file[x_variable][mask],
                               hdf_file[y_variable][mask])
    #Create contour plot
    if clims:
        contour_levels = np.linspace(clims[0],clims[1],num_levels)
        plt.tricontourf(triang,hdf_file[z_variable][mask],contour_levels,extend='both')
    else:
        plt.tricontourf(triang,hdf_file[z_variable][mask],num_levels,extend='both')
    plt.xlabel(x_variable)
    plt.ylabel(y_variable)
    cb = plt.colorbar()
    cb.ax.set_ylabel(z_variable)
    

fcontour_plot_dataset(file_path,'J2Re10.hdf5','Axial','Transverse','Radiography',(-0.05,0.05))
fcontour_plot_dataset(file_path,'J2Re10_no_dopant.hdf5','Axial','Transverse','Radiography',(-0.05,0.05))
#fcontour_plot_dataset(file_path,'J2Re10_no_flame.hdf5','Axial','Transverse','Radiography',(-0.05,0.05))
fcontour_plot_dataset(file_path,'J2Re10.hdf5','Axial','Transverse','Raw_Ar',(0,0.2))
fcontour_plot_dataset(file_path,'J2Re10_no_dopant.hdf5','Axial','Transverse','Raw_Ar',(0,0.2))
#fcontour_plot_dataset(file_path,'J2Re10.hdf5','Axial','Transverse','Ar_Counts')
#fcontour_plot_dataset(file_path,'J2Re10_no_dopant.hdf5','Axial','Transverse','Ar_Counts')
#fcontour_plot_dataset(file_path,'J2Re10_no_flame.hdf5','Axial','Transverse','Raw_Ar',(0,0.2))
plt.show()