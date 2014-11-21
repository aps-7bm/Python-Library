'''Module to combined several HDF5 scans into one file. 
Only copies over the desired datasets.  Each dataset is placed
in a dataset one dimension bigger.  Since different scans may have different
numbers of points, fill the arrays with NAN to flag entries that aren't filled yet.

Alan Kastengren, XSD, APS

Started: October 20, 2014
'''
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def finitialize_new_hdf5(file_path,new_file_name):
    '''Initializes the new HDF5 file.
    '''
    new_file = h5py.File(file_path+new_file_name,'w')
    return new_file

def fcombine_dataset(file_path,old_file_names,new_file,
                            retrieve_dataset_method,
                            old_dataset_name,
                            new_dataset_name):
    '''Finds the values of the desired dataset and
    fills appropriate arrays in the new hdf5 file.
    Only works with 1D scan for each original HDF5 file.
    Variables:
    file_path: path to all of these files
    old_file_names: list of names of the files to be combined.
    new_file: h5py.File object for the combined file. 
    retrieve_dataset_method: function to be applied to old file to 
                                    find values of dataset.
    old_dataset_name: name of dataset in old files
    new_datset_name: name of dataset in combined file.
    '''
    #Make lists to hold the values of the dataset until we know the sizes.
    data_list = []
    #Loop through the old files, finding the positioner arrays.
    for fname in old_file_names:
        print fname
        old_file = h5py.File(fname,'r')
        data_list.append(retrieve_dataset_method(old_file,old_dataset_name))
    #Loop through these values, finding the max size from any individual data file
    max_length = 0
    for array in data_list:
        if len(array) > max_length:
            max_length = len(array)
    #Initialize the dataset in the combined file
    new_file.create_dataset(new_dataset_name,data=np.NAN*np.ones((max_length,len(data_list))))
    #Fill the combined file
    for i,array in enumerate(data_list):
        new_file[new_dataset_name][:len(array),i] = array
    return

def ftop_level_dataset(hdf_file,dataset_name):
    '''Method to retrieve a top-level dataset.
    '''
    return hdf_file[dataset_name][...]

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

def fprocess_files(file_path,old_file_names,new_file_name,dataset_method_dict,names_dict):
    '''Combine data files into one HDF5 file. 
    Variables
    file_path: path to all of the files
    old_file_names: list of names of the files to be combined.
    new_file_name: name for the combined file. 
    dataset_method_dict: dictionary in form dataset_name:retrieval_method
    names_dict: dictionary in form old_name:new:name
    '''
    #Initialize new HDF5 file
    new_file = finitialize_new_hdf5(file_path,new_file_name)
    #Loop through the datasets in the dictionaries
    for old_dataset in dataset_method_dict.keys():
        fcombine_dataset(file_path,old_file_names,new_file,
                            dataset_method_dict[old_dataset],
                            old_dataset,
                            names_dict[old_dataset])