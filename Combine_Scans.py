'''Module to combined several HDF5 scans into one file. 
Only copies over the desired datasets.  Each dataset is placed
in a dataset one dimension bigger.  Since different scans may have different
numbers of points, fill the arrays with NAN to flag entries that aren't filled yet.

Alan Kastengren, XSD, APS

Started: October 20, 2014
June 12, 2015: add functionality to sort by a given variable name.
'''
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import logging

#Set up logging
logger = logging.getLogger('Combine_Scans')
logger.addHandler(logging.NullHandler())

def finitialize_new_hdf5(file_path,new_file_name):
    '''Initializes the new HDF5 file.
    '''
    new_file = h5py.File(file_path+new_file_name,'w')
    return new_file

def fcombine_dataset(file_path,old_file_names,new_file,
                            retrieve_dataset_method,
                            old_dataset_name,
                            new_dataset_name,
                            ordering_name = None):
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
    ordering_name: name to be used to place data into datasets in ascending order.
    '''
    logger.info("Writing dataset {0:s} to new dataset name {1:s}.".format(old_dataset_name,new_dataset_name))
    print("Writing dataset {0:s} to new dataset name {1:s}.".format(old_dataset_name,new_dataset_name))
    #Make lists to hold the values of the dataset until we know the sizes.
    data_list = []
    ordering_name_list = []
    #Loop through the old files, finding the positioner arrays.
    for fname in old_file_names:
        print(fname)
        with h5py.File(fname,'r') as old_file:
            data_list.append(retrieve_dataset_method(old_file,old_dataset_name))
            #If we are ordering by a variable, make a list out of it.
            if ordering_name:
                ordering_name_list.append(np.mean(retrieve_dataset_method(old_file,ordering_name)))
    #If an ordering_name is given, sort the data_list by these values
    if ordering_name:
#         print(ordering_name)
#         print(ordering_name_list)
#         print(data_list)
        __, data_list = zip(*sorted(zip(ordering_name_list, data_list)))
    #Loop through these values, finding the max size from any individual data file
    max_length = data_list[0].shape
    for array in data_list:
        max_length = np.maximum(max_length,array.shape)
    logger.info("Max length = " + str(max_length))
    #Initialize the dataset in the combined file
    dataset_size = list(np.append(max_length,len(data_list)))
    logger.info("Final array size = " + str(dataset_size))
    new_file.create_dataset(new_dataset_name,data=np.NAN*np.ones(dataset_size))
    #Fill the combined file
    for i,array in enumerate(data_list):
        if len(array.shape) == 1:
            new_file[new_dataset_name][:len(array),i] = array
        elif len(array.shape) == 2:
            new_file[new_dataset_name][:array.shape[0],:array.shape[1],i] = array
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
    plt.show()

def fprocess_files(file_path,old_file_names,new_file_name,dataset_method_dict,names_dict,
                   ordering_name=None):
    '''Combine data files into one HDF5 file. Place in ascending order of variable ordering_name
    Variables
    file_path: path to all of the files
    old_file_names: list of names of the files to be combined.
    new_file_name: name for the combined file. 
    dataset_method_dict: dictionary in form dataset_name:retrieval_method
    names_dict: dictionary in form old_name:new:name
    ordering_name: name of a variable to be used to order the files.
    '''
    #If no dataset_method_dict is given, fill it in
    if not dataset_method_dict:
        dataset_method_dict = {}
        for key in names_dict:
            dataset_method_dict[key] = ftop_level_dataset
    #Initialize new HDF5 file
    new_file = finitialize_new_hdf5(file_path,new_file_name)
    #Loop through the datasets in the dictionaries
    for old_dataset in dataset_method_dict.keys():
        fcombine_dataset(file_path,old_file_names,new_file,
                            dataset_method_dict[old_dataset],
                            old_dataset,
                            names_dict[old_dataset],
                            ordering_name)
    new_file.close()
