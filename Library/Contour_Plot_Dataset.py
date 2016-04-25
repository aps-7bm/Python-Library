'''Script to make a contour plot of a variable from a combined HDF5 file of
several scans.  

Alan Kastengren, XSD, APS

Started: October 21, 2014

Revisions:
Feb 11, 2016: add in functionality to perform nonlinear colormap.  Also revise 
                fcontour_plot_dataset to account for NaNs in the datasets.
'''
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib
import h5py
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

#Default_values
file_path = '/data/Data/SprayData/Cycle_2014_2/AFRL_Edwards/'

def fcontour_plot_dataset(file_path,hdf_file_name,x_variable,y_variable,z_variable,
                          clims=None,num_levels=41):
    '''Script to make a contour plot of a dataset from an HDF5 file of several
    scans combined.
    '''
    #Create a new figure
    plt.figure()
    plt.suptitle(hdf_file_name + ': ' + z_variable)
    #Open file
    with h5py.File(file_path + hdf_file_name,'r') as hdf_file:
        #Mask off any NAN entries is x; indicates scan wasn't as wide as widest scan
        mask = np.isfinite(hdf_file[x_variable])
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

def fcontour_plot_set(file_path,filename,dataset_names,lims_dict = None,x='Axial',y='Transvese'):
    '''Creates a contour plot of a set of datasets.
    Inputs
    file_path: path to the directory holding the files
    filename: name of HDF5 file holding the data
    dataset_names: list of names of the datasets to be plotted
    lims_dict: optional dictionary in the form dataset_name:tuple of z-variable limits
    x: name of variable with x values for each point
    y: name of variable with y values for each point
    '''
    #Close any existing figures
    plt.close('all')
    plt.clf()
    for dset in dataset_names:
        if lims_dict and lims_dict[dset]:
            fcontour_plot_dataset(file_path,filename,x,y,dset,lims_dict[dset])
        else:
            fcontour_plot_dataset(file_path,filename,x,y,dset)
        plt.subplots_adjust(right=0.8)
    plt.show()
    
class nlcmap(LinearSegmentedColormap):
    """Class to create a nonlinear colormap.
    Usage:
    In code, call, for example:
    levs = np.concatenate((np.linspace(0,0.1,21),np.linspace(0.2,0.6,5)))
    nl_cmap = nlcmap(matplotlib.cm.get_cmap('jet'),levs)
    plt.contourf(x,_y,z,levels=levs,cmap=nl_cmap)"""
    name = 'nlcmap'
    def __init__(self, cmap, levels):
        self.cmap = cmap
        # @MRR: Need to add N for backend
        self.N = cmap.N
        self.monochrome = self.cmap.monochrome
        self.levels = np.asarray(levels, dtype='float64')
        self._x = self.levels / self.levels.max()
        self._y = np.linspace(0.0, 1.0, len(self.levels))
    #@MRR Need to add **kw for 'bytes'
    def __call__(self, xi, alpha=1.0, **kw):
        """docstring for fname"""
        # @MRR: Appears broken?
        # It appears something's wrong with the
        # dimensionality of a calculation intermediate
        #yi = stineman_interp(xi, self._x, self._y)
        yi = np.interp(xi, self._x, self._y)
        return self.cmap(yi, alpha)
