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
                          grid_on = False,**kwargs):
    '''Script to make a contour plot of a dataset from an HDF5 file of several
    scans combined.
    Keyword arguments are either captured by this code or sent to the
    tricontourf function.
    Keywords handled by this code:
    z_scale: multiplicative factor by with the z variable is multiplied.
    title: title to put at the top of the figure
    xlims,ylims: set limits on x and y extent of the figure
    cticks: set colorbar ticks
    ctick_labels: set labels on the colorbar ticks
    z_label: label for the colorbar axis.  Defaults to z_variable
    '''
    #Create a new figure
    plt.figure(figsize=[4,3],dpi=300)
    if 'title' in kwargs.keys():
        plt.suptitle(kwargs['title'])
    plt.grid(grid_on)
    #Tune the size
    plt.subplots_adjust(left=0.2,right=0.82,bottom = 0.15, top = 0.95)
    #Add in a multipliciative factor, if requested
    z_scale = 1.
    if 'z_scale' in kwargs.keys():
        z_scale = float(kwargs['z_scale'])
    #Open file
    with h5py.File(file_path + hdf_file_name,'r') as hdf_file:
        #Mask off any NAN entries is x; indicates scan wasn't as wide as widest scan
        mask = np.isfinite(hdf_file[x_variable])
        #Make triangulation for Delauney triangulation plot
        triang = tri.Triangulation(hdf_file[x_variable][mask],
                                   hdf_file[y_variable][mask])
        #Create contour plot
        plt.tricontourf(triang,hdf_file[z_variable][mask]*z_scale,**kwargs)
        
        plt.xlabel(x_variable)
        plt.ylabel(y_variable)
        if 'xlims' in kwargs.keys():
            plt.xlim(kwargs['xlims'])
        if 'ylims' in kwargs.keys():
            plt.ylim(kwargs['ylims'])
            
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=12)
        if 'cticks' in kwargs.keys():
            cb.set_ticks(kwargs['cticks'])
        if 'z_label' in kwargs.keys():
            cb.ax.set_ylabel(kwargs['z_label'])
        if 'ctick_labels' in kwargs.keys():
            cb.ax.set_yticklabels(kwargs['ctick_labels'])
        else:
            cb.ax.set_ylabel(z_variable)

def fcontour_plot_set(file_path,filename,dataset_names,lims_dict = None,x='Axial',y='Transverse'):
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

def ftest_code():
    file_path = '/home/akastengren/data/Cycle_2013_2/Taitech_Reprocess/'
    file_name = 'Scans_1675_1694.hdf5'
    x_variable = 'X'
    y_variable = 'Y'
    z_variable = 'Water_Pathlength_mm'
    fcontour_plot_dataset(file_path,file_name,x_variable,y_variable,z_variable,
                          grid_on = False,z_scale=100,z_label='Scaled Water',
                          ctick_labels = ['a','b','c'])
    plt.show()
    ticks = np.concatenate((np.linspace(0,0.006,7),np.linspace(0.01,0.1,10)))
    nl_cmap = nlcmap(plt.cm.get_cmap('jet'),ticks)
    fcontour_plot_dataset(file_path,file_name,x_variable,y_variable,z_variable,
                          grid_on = False,levels = ticks,cmap=nl_cmap,extend='both',cticks=ticks,
                          xlims = [-5,5],ylims=[0,10],z_scale=100,
                          ctick_labels = ticks*100)
    
    plt.show()
    
if __name__ == '__main__':
    ftest_code()
