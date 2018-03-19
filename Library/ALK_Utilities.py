'''Script with some handy utility functions.

Alan Kastengren, XSD, Argonne

Started: May 5, 2014

Changes:
July 31, 2014: Fix bug in fwrite_HDF_dataset for attribute writing.
November 16, 2014: Fix bug that used wrong number of digits for filename lists.
February 22, 2015: Add function for Butterworth filter from AFRL 2012-3 processing.
April 20, 2015: Add ability in fcreate_filename to handle a non-integer entry for file_num
August 18, 2015: Move filtering to a new Signal_Processing module
'''
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
from socket import gethostname
import logging

def fwrite_HDF_dataset(hdf_group,name,input_data,attributes=None,save_old_attrs=False,return_dset=False):
    '''Writes a dataset to the input HDF5 group.  Checks if the dataset already exists,
    and deletes it if it does.
    '''
    old_attrs = {}
    if hdf_group.get(name):
        if save_old_attrs:
            for key,value in hdf_group[name].attrs.items():
                old_attrs[key] = value
        del hdf_group[name]
    dset = hdf_group.create_dataset(name,data=input_data)
    if attributes:
        if old_attrs:
            for key,value in old_attrs.items():
                dset.attrs[key] = value
        for key,value in attributes.items():
            dset.attrs[key] = value
    if return_dset:
        return dset
    else:
        return
    
def fcreate_filename(file_num,filename_prefix='7bmb1_',filename_suffix='.mda',
                    digits=4):
    '''Make a filename with a number using a fixed number of digits.
    '''
    if isinstance(file_num,(int,long)):
        format_string = '{0:0'+str(digits)+'d}'
        return filename_prefix+format_string.format(file_num)+filename_suffix
    else:
        return filename_prefix+str(file_num)+filename_suffix

def fcreate_filename_list(file_nums,filename_prefix='7bmb1_',filename_suffix='.mda',
                    digits=4, path="/data/Data/SprayData/Cycle_2014_1/ISU_Point/",
                    check_exist=False):
    '''Takes a list of file number, a prefix, a suffix, path, and # of digits,
    and makes a list of the file names.
    '''
    print(path)
    filename_list = []
    #Loop through the input string numbers, converting first to int, then to
    #correct number of digits as a string.
    for i_str in file_nums:
        filename_list.append(path+fcreate_filename(i_str,filename_prefix,filename_suffix,digits))
    #If checking for existence was requested, do so
    if check_exist:
        filename_list[:] = [name for name in filename_list if os.path.isfile(name)]
    return filename_list

def fcheck_files_exist(file_nums,filename_prefix='7bmb1_',filename_suffix='.mda',
                    digits=4, path="/data/Data/SprayData/Cycle_2014_1/ISU_Point/"):
    '''Creates filenames and checks to see if they exist.
    Returns a list of those file numbers that do exist.
    '''
    return_list = []
    for i_str in file_nums:
        if os.path.isfile(path+fcreate_filename(i_str,filename_prefix,filename_suffix,digits)):
            return_list.append(i_str)
    return return_list

def fcompute_absorption_coeff(ref_coeff,energy):
    '''Computes the absorption coefficient at a given energy
    (in keV) from the reference absorption coeff at 10 keV.
    Assumes E^-3 dependence of absorption coefficient on energy.
    '''
    return ref_coeff*(10.0/energy)**3

def fcheck_file_datasets(hdf_file, names_list=[]):
    '''Checks that required datasets exist in an HDF5 file.
    '''
    #Make a list of filenames
    for value in names_list:
        if not hdf_file.get(value):
            return False
    else:
        return True

def fcorrect_path_start():
    '''Returns the correct start for different computers.
    
    It gives the parent directory of the link to SprayData.
    '''
    host_name = gethostname()
    #My new laptop
    if host_name.startswith('stokes'):
        return '/home/akastengren/data/'
    #My old laptop
    elif host_name.startswith('euler'):
        return '/data/Data'
    #Beamline workstations
    elif host_name.endswith('aps.anl.gov'):
        return '/home/'
    else:
        print("Unknown hostname.  Returning current directory.")
        return os.getcwd() + '/'
    
def fcompare_values(desired_value,search_values,tolerance=None):
    '''Returns whether search_values are within tolerance of desired_value.
    Works for both scalar numbers and numpy arrays.
    '''
    if not tolerance:
        tolerance = np.abs(desired_value/1e3)
    return np.abs(search_values - desired_value) < tolerance
    
def fcopy_group(new_group,old_group):
    '''Copies an HDF group's contents to a new group.
    Make a separate function so it can be applied recursively for
    arbitrary nested structure.
    '''
    #Copy over attributes from old to new group
    for attr_name,attr_value in old_group.attrs.items():
        new_group.attrs[attr_name] = attr_value
    #Loop through all item names in the group
    for name in old_group.keys():
        #If an item is a dataset, just copy the dataset
        if old_group.get(name,getclass=True) is h5py.Dataset:
            new_group.create_dataset(name,data=old_group[name])
            for attr_name,attr_value in old_group[name].attrs.items():
                new_group[name].attrs[attr_name] = attr_value
        #Otherwise, it is a group.  Make a new group and call this function
        #recursively.
        else:
            new_group.create_group(name)
            fcopy_group(new_group[name],old_group[name])
            
def fcompact_HDF_file(directory,existing_file_name):
    #Open both old file and a dummy file.
    with h5py.File(directory+'Dummy.hdf5','w') as new_file:
        with h5py.File(directory+existing_file_name,'r') as old_file:
            #Recursively loop, saving the data 
            fcopy_group(new_file,old_file)
    #Breaking block will force files to close.  Now, rename new file
    #to old filename to overwrite.
    os.rename(directory+'Dummy.hdf5',directory+existing_file_name)

def fplot_HDF_trace(file_path,file_name,plot_var,x_var,norms=None,
                    num_points=None,separate_figures=False,
                    y_lims = None):
    ''' Plots a list of traces from an HDF5 file.
    Input variables
    file_path and file_name: path to and name of the HDF5 file.
    plot_var: list of variables to be plotted.  Full path from
                file root.
    x_var: list of absicca for plotting
    norms: optional list of normalization factors
    num_points: optional limit on the number of points plotted. 
                Useful for very long traces
    separate_figures: plot all on the same figure (False) or in
                        separate figures (True)
    '''
    #Check that the sizes of the various input lists are the same
    if len(x_var) == 1:
        x_var = x_var * len(plot_var)
    elif not len(plot_var) == len(x_var):
        print("Mismatch in lengths of plotting and absicca variables.  Exiting.")
        return
    if norms:
        if not len(norms) == len(plot_var):
            print("Mismatch in lengths of plotting and normalization lists.")
            norms = [1.0] * len(plot_var)
    else:
        #If no normalization is given, just set it to unity.
        norms = [1.0] * len(plot_var)
    with h5py.File(file_path+file_name,'r') as hdf_file:
        for y,x,norm in zip(plot_var,x_var,norms):
            if separate_figures:
                plt.figure()
            plt.plot(hdf_file[x][...],hdf_file[y][...]/norm,label=y)
            if separate_figures:
                plt.title(y)
                plt.grid()
                if y_lims:
                    plt.ylim(y_lims)
        if not separate_figures:
            plt.legend(fontsize=8,loc='upper left')
            plt.grid()
            plt.title(file_name)
            if y_lims:
                plt.ylim(y_lims)
        plt.show()
        return

def ftest_numeric(string_input):
    '''Tests whether a string input can be cast as a float.
    '''
    try:
        float(string_input)
        return True
    except ValueError:
        return False
            
def frename_hdf_group_items(hdf_group,substitutions_dict):
    '''Renames items in an HDF group using a dictionary for substitutions.
    Saves the old name as an attribute.
    '''
    for key in hdf_group.keys():
        if key in substitutions_dict:
            hdf_group[substitutions_dict[key]] = hdf_group[key]
            hdf_group[substitutions_dict[key]].attrs['Original_Name'] = key
            del hdf_group[key]      

def ffind_parabola_peak(x_vals,y_vals):
    '''Finds the peak value of a parabola given three x and y values.
    '''
    #Make sure the lengths of the x_vals and y_vals arrays are each three
    if not len(x_vals) == 3 and len(y_vals) == 3:
        print("Method ffind_parabola_peak requires exactly three x and y values.")
        return None
    #Make up the matrices for solving the system
    A = np.zeros((3,3))
    y = np.zeros(3)
    for i in range(3):
        A[i,:] = np.array([x_vals[i]**2,x_vals[i],1])
        y[i] = y_vals[i]
    coeff = scipy.linalg.solve(A,y)
    return -coeff[1]/2.0/coeff[0]

def fstart_logging(logger_name,filename=None,logging_level=logging.DEBUG):
    '''Sets up logging in a module with a given name and log file name.
    Inputs:
    logger_name: name of this root logger
    filename: name of the log file to be used.  Default None.
    logging_level: level of logging to use
    Returns: logger instance
    '''
    #Set up logging
    logger = logging.getLogger(logger_name)
    #Set level of detail to be given
    logger.setLevel(logging_level)
    # create console handler and set level to debug
    if not filename:
        filename = logger_name + '.log'
    ch = logging.FileHandler(filename,'w')
    ch.setLevel(logging_level)
    
    # Set logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - line %(lineno)d - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
                
if __name__ == '__main__':
#     directory = fcorrect_path_start()+'SprayData/Cycle_2014_2/AFRL_Edwards/'  
#     old_name = '7bmb1_1052.hdf5'   
# #    fcompact_HDF_file(directory,old_name)        
#     fplot_HDF_trace(directory,old_name,
#                      ['Kr_Kalpha','Radiography'],
#                      ['7bmb1:m26.VAL'],
#                      [1000.0,1.0],y_lims=[0,0.01])  
    print(ffind_parabola_peak([-2.0,0,5.0],[26.0,4.0,54.0]))
    

        