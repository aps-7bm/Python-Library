'''Code to combine scan data from a 2D EPICS scan converted to HDF5
into multidimensional datasets.

Alan Kastengren, XSD

Started: June 18, 2014
May 20, 2016: add fcollate_file summary function to run everything in one step.
'''
import h5py
import numpy as np
import MDA2HDF5_Fluorescence as m2h
import ALK_Utilities as ALK

def ffind_groups(hdf_group,row_var_name):
    '''Find groups that are subscans of the current group.
    Uses the row_var_name as a substring at the beginning of the group name.
    Returns a sorted list of the group names, as well as a numpy array of
    the values of the row variable.
    '''
    #Initialize variables for subgroups
    subgroup_list = []
    row_var = []
    #Loop through items in this group.  This will include datasets and groups
    for name,value in hdf_group.items():
        print name
        #Make sure this starts with the row_var_name
        if name.startswith(row_var_name+'=') :
            row_var.append(float(name[len(row_var_name+"="):]))
            subgroup_list.append(value)
    #Sort by the row_var in ascending order
    zipped_tuple = sorted(zip(row_var,subgroup_list))
    row_var,subgroup_list = zip(*zipped_tuple)
    return subgroup_list,row_var

def ffind_subgroup_datasets(subgroup_list):
    '''Loops through the subgroup list, returning a dictionary
    of dataset names and shapes.
    Returns a dictionary, which hold dataset names as keys 
    maximum size of this dataset as the value.
    Using shape tuple rather than size to allow for more generality.
    '''
    #Initialize dictionary for datasets and their maximum size
    datasets_dict = {}
    #Loop through the groups in the subgroup list
    for group in subgroup_list:
        #Loop across all items in this group
        for (name,value) in group.items():
            #See if this is a dataset.  If not, continue
#           entry_datatype = group.get(name,getclass=True) == h5py.highlevel.Dataset
            if group.get(name,getclass=True) == h5py.highlevel.Dataset:
                #If dataset name isn't in the dictionary already, add it
                if name not in datasets_dict.keys():
                    datasets_dict[name] = value.shape
                #If it is, compare size of current dataset to stored value.
                #If the current is bigger in any dimension, revise shape tuple in dictionary
                else:
                    old_shape_tuple = datasets_dict[name]
                    value_shape_tuple = value.shape
                    new_shape_tuple = ()
                    for old,current in zip(old_shape_tuple,value_shape_tuple):
                        new_shape_tuple = new_shape_tuple + (old,) if old >= current else (current,)
    return datasets_dict

def fwrite_multidimensional_datasets(subgroup_list,datasets_dict,row_var,main_group,delete_old_datasets=False):
    #For each dataset, make a zeros numpy array to go with it
    #Keep in mind that we're trying to keep this quite general, so we assume
    #the shape is a tuple; this should allow us to handle any dimensionality of
    #array
    datasets_arrays_dict = {}
    for key,value in datasets_dict.items():
        #Add a dimension to the maximum size of the 
        datasets_arrays_dict[key] = ALK.fwrite_HDF_dataset(main_group,key+"_Multidimensional",
                                                           np.zeros((len(row_var),)+value),
                                                           return_dset=True)
    #Loop across subgroup_list, adding the data to the relevant arrays
    for i,group in enumerate(subgroup_list):
        for key,value in datasets_arrays_dict.items():
            data = group[key][...]
            #Just in case size of dataset is smaller than preallocated size,
            #we have to make up an index array
            if data.shape != value[i].shape:
                pad_sequence = ()
                for data_entry,value_entry in zip(data.shape,value[i].shape):
                    pad_sequence = pad_sequence + ((0,value_entry-data_entry),)
                value[i] = np.lib.pad(data,pad_sequence)
            else:
                value[i] = data
            #If this is the first point, save the old dataset attributes
            if i==0 and group[key].attrs:
                value.attrs = group[key].attrs
            #Delete the old dataset from the group.  Leave group to maintain attributes
            del group[key]
        print "Done with group " + group.name
        if delete_old_datasets:
            group_name = group.name.split('/')[-1]
            del main_group[group_name]
        
def fcollate_file(hdf_filename,file_path,row_var_name='7bmb1:m26.VAL'):
    '''Runs all parts of code to make multidimensional arrays from
    multidimensional scan data.
    '''
    with h5py.File(file_path+hdf_filename,'r+') as hdf_file:
        subgroup_list,x_values = ffind_groups(hdf_file,row_var_name)
        dataset_dict = ffind_subgroup_datasets(subgroup_list)
        fwrite_multidimensional_datasets(subgroup_list,dataset_dict,x_values,hdf_file)
 
if __name__ == '__main__':
    file_path = '/data/Data/SprayData/Cycle_2014_2/NX_School/'
    data_file = '7bmb1_0128.hdf5'
    m2h.frun_main(data_file[:-4]+"mda", file_path, False)
    hdf_file = h5py.File(file_path+data_file,'r+')
    subgroup_list,x_values = ffind_groups(hdf_file,'7bmb1:m26.VAL')
    dataset_dict = ffind_subgroup_datasets(subgroup_list)
    fwrite_multidimensional_datasets(subgroup_list,dataset_dict,x_values,hdf_file)
