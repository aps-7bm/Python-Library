'''Module of functions to be used for normalization of radiography data.

Alan Kastengren, XSD, APS

Started: February 10, 2015
'''
import ALK_Utilities as ALK
import h5py
import numpy as np

def ffind_normalization_radiography_minimums(path,ref_file_nums,dataset_name,
                                             filename_prefix,filename_suffix,
                                             digits=4,**kwargs):
    '''Loops across the files in the list, finding the minimum dataset
    value in each, and uses the median of this value as the reference.
    This is a bit ad-hoc, but avoids spurious noise in a few files.
    '''
    #Create a list of reference file names
    ref_filename_list = ALK.fcreate_filename_list(ref_file_nums, filename_prefix,filename_suffix, 
                                              digits, path, check_exist=True)
    min_values = []
    #Loop through file names.  We already checked for existence.
    for f_name in ref_filename_list:
        #Open the file
        with h5py.File(f_name,'r') as hdf_file:
            #Add minimum value to the min_values array
            min_values.append(np.min(hdf_file[dataset_name][...]))
    #Take median of the min values
    return np.median(np.array(min_values))

def ffind_normalization_radiography_outer_points(path,ref_file_nums,dataset_name,
                                             filename_prefix,filename_suffix,
                                             digits=4,**kwargs):
    '''Loops across the files in the list, finding the dataset
    values on the edges, and uses the median of these values as the reference.
    This is a bit ad-hoc, but avoids spurious noise in a few files.
    '''
    #Set the number of points based on **kwargs, or 3 if there is no valid variable in **kwargs
    num_points = 3
    if kwargs['num_points']:
        num_points = int(kwargs['num_points'])
    #Create a list of reference file names
    ref_filename_list = ALK.fcreate_filename_list(ref_file_nums, filename_prefix,filename_suffix, 
                                              digits, path, check_exist=True)
    outer_values = np.array([])
    #Loop through file names.  We already checked for existence.
    for f_name in ref_filename_list:
        #Open the file
        with h5py.File(f_name,'r') as hdf_file:
            #Get the array for the pertiment variable
            rad_data = hdf_file.get(dataset_name)[...]
            #If the dataset is shorter than 2*num_points, just include the whole thing
            if rad_data.shape[0] <= 2*num_points:
                outer_values = np.concatenate((outer_values,rad_data))
            else:
                outer_values = np.concatenate((outer_values,rad_data[:num_points]))
                outer_values = np.concatenate((outer_values,rad_data[-num_points:]))

    #Take median of the min values
    return np.median(outer_values)

def ffind_normalization_radiography_clearscan(path,ref_file_nums,dataset_name,
                                             filename_prefix,filename_suffix,
                                             digits=4,**kwargs):
    '''Loops across the files in the list, finding the overall average dataset value.
    Use if we have clear reference scans.  Just do a sum and divide manually, since
    different scans may have different numbers of points.
    '''
    #Create a list of reference file names
    ref_filename_list = ALK.fcreate_filename_list(ref_file_nums, filename_prefix,filename_suffix, 
                                              digits, path, check_exist=True)
    ref_sum=0
    num_values = 0
    #Loop through file names.  We already checked for existence.
    for f_name in ref_filename_list:
        #Open the file
        with h5py.File(f_name,'r') as hdf_file:
            #Add minimum value to the min_values array
            ref_sum += np.sum(hdf_file.get(dataset_name)[...])
            num_values += np.size(hdf_file.get(dataset_name)[...])
    #Take mean and return it
    return ref_sum/float(num_values)

def ffind_normalization_initial(array,I0_number):
    '''Normalizes a trace by the first I0_number points.
    '''
    return array/np.mean(array[:I0_number])