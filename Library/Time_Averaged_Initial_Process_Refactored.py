'''Script to perform initial processing for radiography and fluorescence for
time-averaged (i.e., MDA) data files.

Alan Kastengren, XSD, Argonne

Started: May 5, 2014

Change log
June 17, 2014: change frun_radiography_only to convert both regular and dark files
                to HDF5 format.  This way, we don't have to include dark files
                in list of regular files to get them converted.
June 17, 2014: add fset_variable_names function.
February 10, 2015: Major refactoring to simplify the function calls.


Instructions for use:
1.  Import module into processing code
2.  Use module attributes to set pertinent processing parameters.
3.  To process everything, simply call frun_radiography_fluorescence(file_nums,ref_file_nums,dark_compute,dark_nums,
                          abs_coeff)
'''
import h5py
import numpy as np
import os.path
import MDA2HDF5_Fluorescence as m2h
import ALK_Utilities as ALK
import Radiography_Process
import Normalization_Functions as nf

#We will likely use the path multiple times.  Make this a module variable.
file_path = "/data/Data/SprayData/Cycle_2014_7/ISU_Point/"
I0_variable = '7bmb1:scaler1.S3'
PIN_variable = '7bmb1:scaler1.S5'
slow_variable = '7bmdsp1:dxp1:Events'       #name of good events recorded by fluorescence pulse processor
fast_variable = '7bmdsp1:dxp1:Triggers'     #name of variable for triggers from fluorescence pulse processor fast filter
prefix = '7bmb1_'       #string of characters found before scan number in MDA file names.
MDA_suffix = '.mda'
HDF_suffix = '.hdf5'
digits = 4              #number of digits in file name for scan number
radiography_name = 'Radiography'            #name to be given to processed radiography data
names_dict = {}         #dictionary of fluorescence variables to be processed, in form new_name:EPICS_name
norm_function = nf.ffind_normalization_radiography_minimums     #function to be used for normalizing the radiography
norm_kwarg = {}
integration_time = 1.0
fast_time_constant = 3.13e-7    #Fast filter time constant
reprocess_existing_hdf = False  #If HDF5 file exists, should we overwrite or leave it?

def fconvert_files_to_hdf5(file_nums, mca_saving = True):
    '''Process a set of file numbers from mda to HDF5 format.
    '''
    #Create a list of file names
    filename_list = ALK.fcreate_filename_list(file_nums, prefix,
                                              MDA_suffix, digits, file_path, check_exist=True)
    #Create a list of the HDF5 file names that we would have after conversion
    hdf_filename_list = ALK.fcreate_filename_list(file_nums, prefix,
                                              HDF_suffix, digits, file_path, check_exist=False)
    #Loop through file names.  We already checked for existence of mda files.
    if not filename_list:
        print "No valid filenames found for HDF5 conversion."
    for f_name, hdf_name in zip(filename_list,hdf_filename_list):
        #Check if the HDF5 file exists if we don't want to overwrite
        if not reprocess_existing_hdf:
            if os.path.isfile(hdf_name):
                print "File " + os.path.split(hdf_name)[-1] + " exists already.  Skipping."
                continue
        print f_name
        
        m2h.frun_main(os.path.split(f_name)[-1],file_path,mca_saving)
        print "File " + os.path.split(f_name)[-1] + " converted to HDF5 successfully."
    
def fbatch_analyze_radiography(file_nums,dark_compute=False,dark_nums=[],abs_coeff=1):
    #If we are computing dark currents, do so
    dark_dict = {}
    if dark_compute:
        dark_dict = ffind_dark_current(dark_nums,(I0_variable,PIN_variable))
    else:
        dark_dict[I0_variable] = 0
        dark_dict[PIN_variable] = 0
    #Create a list of file names
    filename_list = ALK.fcreate_filename_list(file_nums, prefix,HDF_suffix, 
                                              digits, file_path, check_exist=True)
    #Loop through file names.  We already checked for existence.
    if not filename_list:
        print "No valid filenames found for radiography analysis."
    for f_name in filename_list:
        fanalyze_radiography(f_name,abs_coeff,dark_dict)
        print "File " + os.path.split(f_name)[-1] + " processed for radiography successfully."

def fanalyze_radiography(f_name,abs_coeff=1,dark_dict={},units=None):
    '''Analyze time-averaged data from an HDF5 file converted from MDA format for
    radiography.
    '''
    with h5py.File(f_name,'r+') as hdf_file:
        #Make sure these traces exist.  If they don't, return out of function.
        if ALK.fcheck_file_datasets(hdf_file,[PIN_variable,I0_variable]):
            #Get the transmitted and reference intensity traces
            I = hdf_file.get(PIN_variable)[...]
            I0 = hdf_file.get(I0_variable)[...]
            #Make sure they are both the same size and both exist.
            if np.size(I) != np.size(I0):
                print "Problem processing " + f_name + ": I and I0 sizes mismatched.  Skipping."
                return
            #If they are, go ahead and process.  Send clear_ratio = 1 so we don't normalize within the dataset.
            #We will norm properly later in the workflow.
            radiography = Radiography_Process.fcompute_radiography_density(I, I0, 1, dark_dict[PIN_variable], dark_dict[I0_variable], abs_coeff)
            ALK.fwrite_HDF_dataset(hdf_file, radiography_name, radiography,{'Absorption_Coefficient':abs_coeff,'Units':units})

def fnormalize_radiography(file_nums,ref_file_nums=None):
    '''Code to find a normalization for the radiography and apply it
    to a list of radiography files.  Flexible to allow user to give a 
    routine to find the normalization.
    '''
    #Set the reference files equal to the input files if none are explicitly given
    if not ref_file_nums:
        ref_file_nums = file_nums
    #Perform normalization routine
    ref_value = norm_function(file_path,ref_file_nums,radiography_name,
                                             prefix,HDF_suffix,
                                             digits=4,**norm_kwarg)
    #Form a list of file names
    filename_list = ALK.fcreate_filename_list(file_nums, prefix, HDF_suffix, 
                                              digits, file_path, check_exist=True)
    #Loop through file names.  We already checked for existence.
    for f_name in filename_list:
        #Open the file
        with h5py.File(f_name,'r+') as hdf_file:
            if ALK.fcheck_file_datasets(hdf_file,[radiography_name]):
                #Subtract reference value from radiography
                hdf_file[radiography_name][...] = hdf_file[radiography_name][...]-ref_value
                hdf_file[radiography_name].attrs['Normalized'] = 'True'
                hdf_file[radiography_name].attrs['Norm_value'] = ref_value
                print "File " + os.path.split(f_name)[-1] + " normalized successfully."
    
        
def ffind_dark_current(file_nums, variable_keys=None):
    '''Process a set of file numbers corresponding to dark files.
    Average the value of the variables named variable_keys over valid files.
    Returns a dictionary with the dark values.
    '''
    #Make sure we at least record I0 and PIN
    if not variable_keys:
        variable_keys = (I0_variable,PIN_variable)
        
    #Form a list of file names
    filename_list = ALK.fcreate_filename_list(file_nums, prefix, HDF_suffix, 
                                              digits, file_path, check_exist=True)
    #Set up a dictionary to hold arrays (and eventually scalars) with dark values
    dark_values = {}
    for v_name in variable_keys:
        dark_values[v_name] = []
    #Loop through the file names
    for f_name in filename_list:
        #Open the file
        with h5py.File(f_name,'r') as hdf_file:
            if ALK.fcheck_file_datasets(variable_keys):
                for key in variable_keys:
                    print key
                    dark_values[key].append(np.mean(hdf_file.get(key)[...]))
                print "File " + os.path.split(f_name)[-1] + " analyzed for dark current successfully."
    print dark_values
    #Average the values 
    for v_name in variable_keys:    
        dark_values[v_name] = np.mean(np.array(dark_values[v_name]))
    return dark_values

def fanalyze_fluorescence(file_nums):
    '''Perform initial fluorescence analysis for a set of time-averaged files.
    The input name_dict is a dictionary whose keys are the desired new names of 
    variables, the values being the existing names.
    '''
    #Form a list of file names
    filename_list = ALK.fcreate_filename_list(file_nums, prefix, HDF_suffix, 
                                              digits, file_path, check_exist=True)
    #Loop through the file names
    for f_name in filename_list:
        #Open the file
        with h5py.File(f_name,'r+') as hdf_file:
            #Check that all required datasets exist
            if ALK.fcheck_file_datasets(hdf_file,[radiography_name,I0_variable,
                                                  slow_variable,fast_variable]):
                #Extract the radiography data, I0 data, absorption coefficient, and slow and fast events
                rad_data = hdf_file.get(radiography_name)[...]
                I0_data = hdf_file.get(I0_variable)[...]
                slow_events = hdf_file.get(slow_variable)[...]
                fast_events = hdf_file.get(fast_variable)[...]
                abs_coeff = hdf_file[radiography_name].attrs['Absorption_Coefficient']
                #Loop through the items in the name_dict
                for (key,value) in names_dict.items():
                    fluor_dataset = hdf_file.get(value)
                    if not fluor_dataset:
                        print "Problem processing " + f_name + ": dataset " + value + " does not exist."
                        print "Skipping this variable."
                        continue
                    corrected_fluor = Radiography_Process.fcompute_fluorescence(fluor_dataset[...],
                                            slow_events[...], fast_events[...], rad_data[...], I0_data[...], abs_coeff)
                    corrected_fluor = Radiography_Process.fcompute_fluorescence_fit_fast(fluor_dataset, 
                                                slow_events, fast_events, rad_data, I0_data,
                                                integration_time, abs_coeff,False,fast_time_constant)
                    #Call the processing routine and write the dataset to file
                    ALK.fwrite_HDF_dataset(hdf_file,key,corrected_fluor,{"Processing":'Dead_time,I0,attenuation'})
            
                    print "File " + os.path.split(f_name)[-1] + " analyzed for fluorescence successfully."
    
def frun_radiography_only(file_nums,ref_file_nums=None,dark_compute=False,dark_nums=[],
                          abs_coeff=1):
    '''Code to run through entire chain of analysis.
    Convert MDA to HDF5.
    Dark subtract
    Calculate radiography and write to the HDF file. 
    Normalize the radiography signal.
    Variables:
        file_nums: scan numbers of files to be processed
        ref_file_nums: list of scan numbers that can be used as a reference (no absorption)
        dark_compute: boolean to determine whether to calculate dark currents and account for them.
        dark_nums: list of file numbers with no beam (used for dark currents)
        abs_coeff: absorption coefficient to be used for radiography analysis
    '''
    #Convert to HDF5.  Do both regular files and dark files
    print "Converting to HDF5"
    fconvert_files_to_hdf5(file_nums+dark_nums)
    #Analyze for radiography
    fbatch_analyze_radiography(file_nums,dark_compute,dark_nums,abs_coeff)
    fnormalize_radiography(file_nums,ref_file_nums)  

def frun_radiography_fluorescence(file_nums,ref_file_nums=None,dark_compute=False,dark_nums=[],
                          abs_coeff=1):
    '''Code to run through entire chain of analysis.
    Convert MDA to HDF5.
    Calculate radiography and write to the HDF file. 
    Normalize the radiography signal.
    Perform initial fluorescence data processing.
    Variables:
        file_nums: scan numbers of files to be processed
        ref_file_nums: list of scan numbers that can be used as a reference (no absorption)
        dark_compute: boolean to determine whether to calculate dark currents and account for them.
        dark_nums: list of file numbers with no beam (used for dark currents)
        abs_coeff: absorption coefficient to be used for radiography analysis
    '''
    #Default values evaluated when module is imported.  Following code
    #to make sure changes in default values get applied
    frun_radiography_only(file_nums,ref_file_nums,dark_compute,dark_nums,abs_coeff)
    fanalyze_fluorescence(file_nums)
    