'''Script to perform initial processing for radiography and fluorescence for
time-averaged (i.e., MDA) data files.

Alan Kastengren, XSD, Argonne

Started: May 5, 2014

Change log
June 17, 2014: change frun_radiography_only to convert both regular and dark files
                to HDF5 format.  This way, we don't have to include dark files
                in list of regular files to get them converted.
June 17, 2014: add fset_variable_names function.
'''
import h5py
import numpy as np
import os.path
import MDA2HDF5_Fluorescence as m2h
import ALK_Utilities as ALK
import Radiography_Process

#We will likely use the path multiple times.  Make this a module variable.
file_path = "/data/Data/SprayData/Cycle_2014_7/ISU_Point/"
I0_variable = '7bmb1:scaler1.S3'
PIN_variable = '7bmb1:scaler1.S5'
slow_variable = '7bmdsp1:dxp1:Events'
fast_variable = '7bmdsp1:dxp1:Triggers'
prefix = '7bmb1_'
MDA_suffix = '.mda'
HDF_suffix = '.hdf5'

def fset_I_variable(new_name):
    global PIN_variable
    PIN_variable = new_name

def fset_I0_variable(new_name):
    global I0_variable
    I0_variable = new_name

def fset_slow_variable(new_name):
    global slow_variable
    slow_variable = new_name

def fset_fast_variable(new_name):
    global fast_variable
    fast_variable = new_name

def fset_path_variable(new_name):
    global file_path
    file_path = new_name

def fconvert_files_to_hdf5(path, file_nums,filename_prefix=prefix,filename_suffix=MDA_suffix,
                    digits=4, mca_saving = True):
    '''Process a set of file numbers from mda to HDF5 format.
    '''
    #Create a list of file names
    filename_list = ALK.fcreate_filename_list(file_nums, filename_prefix,
                                              filename_suffix, digits, path, check_exist=True)
    #Loop through file names.  We already checked for existence.
    if not filename_list:
        print "No valid filenames found for HDF5 conversion."
    for f_name in filename_list:
        print f_name
        m2h.frun_main(os.path.split(f_name)[-1],path,mca_saving)
        print "File " + os.path.split(f_name)[-1] + " converted to HDF5 successfully."
    
def fbatch_analyze_radiography(path, file_nums,filename_prefix=prefix,filename_suffix=HDF_suffix,
                    digits=4, dark_compute=False,dark_nums=[],abs_coeff=1):
    #If we are computing dark currents, do so
    dark_dict = {}
    if dark_compute:
        dark_dict = ffind_dark_current(path,dark_nums,filename_prefix,filename_suffix,digits,
                           (I0_variable,PIN_variable))
    else:
        dark_dict[I0_variable] = 0
        dark_dict[PIN_variable] = 0
    #Create a list of file names
    filename_list = ALK.fcreate_filename_list(file_nums, filename_prefix,filename_suffix, 
                                              digits, path, check_exist=True)
    #Loop through file names.  We already checked for existence.
    if not filename_list:
        print "No valid filenames found for radiography analysis."
    for f_name in filename_list:
        fanalyze_radiography(f_name,abs_coeff,dark_dict)
        print "File " + os.path.split(f_name)[-1] + " processed for radiography successfully."

def fanalyze_radiography(f_name,abs_coeff=1,dark_dict={}):
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
            ALK.fwrite_HDF_dataset(hdf_file, 'Radiography', radiography,{'Absorption_Coefficient':abs_coeff})

def ffind_normalization_radiography_minimums(path,ref_file_nums,filename_prefix=prefix,filename_suffix=HDF_suffix,
                    digits=4, radiography_name='Radiography'):
    '''Loops across the files in the list, finding the minimum radiography
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
            min_values.append(np.min(hdf_file.get(radiography_name)[...]))
    #Take median of the min values
    return np.median(np.array(min_values))

def ffind_normalization_radiography_outer_points(path,ref_file_nums,filename_prefix=prefix,filename_suffix=HDF_suffix,
                    digits=4, radiography_name='Radiography',num_points = 3):
    '''Loops across the files in the list, finding the radiography
    values on the edges, and uses the median of these values as the reference.
    This is a bit ad-hoc, but avoids spurious noise in a few files.
    
    num_points: number of points in from the edge of the data to use
    '''
    #Create a list of reference file names
    ref_filename_list = ALK.fcreate_filename_list(ref_file_nums, filename_prefix,filename_suffix, 
                                              digits, path, check_exist=True)
    outer_values = np.array([])
    #Loop through file names.  We already checked for existence.
    for f_name in ref_filename_list:
        #Open the file
        with h5py.File(f_name,'r') as hdf_file:
            #Get the array for the pertiment variable
            rad_data = hdf_file.get(radiography_name)[...]
            #If the dataset is shorter than 2*num_points, just include the whole thing
            if rad_data.shape[0] <= 2*num_points:
                outer_values = np.concatenate((outer_values,rad_data))
            else:
                outer_values = np.concatenate((outer_values,rad_data[:num_points]))
                outer_values = np.concatenate((outer_values,rad_data[-num_points:]))

    #Take median of the min values
    return np.median(outer_values)

def ffind_normalization_radiography_clearscan(path,ref_file_nums,filename_prefix=prefix,filename_suffix=HDF_suffix,
                    digits=4, radiography_name='Radiography'):
    '''Loops across the files in the list, finding the overall average radiography value.
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
            ref_sum += np.sum(hdf_file.get(radiography_name)[...])
            num_values += np.size(hdf_file.get(radiography_name)[...])
    #Take mean and return it
    return ref_sum/float(num_values)

def fnormalize_radiography(path, file_nums,ref_file_nums=None,filename_prefix=prefix,filename_suffix=HDF_suffix,
                    digits=4, radiography_name='Radiography',
                    routine=ffind_normalization_radiography_minimums):
    '''Code to find a normalization for the radiography and apply it
    to a list of radiography files.  Flexible to allow user to give a 
    routine to find the normalization.
    '''
    #Find the normalization value
    if not ref_file_nums:
        ref_file_nums = file_nums
    ref_value = routine(path,ref_file_nums,filename_prefix,filename_suffix,digits,radiography_name)
    #Form a list of file names
    filename_list = ALK.fcreate_filename_list(file_nums, filename_prefix,filename_suffix, 
                                              digits, path, check_exist=True)
    #Loop through file names.  We already checked for existence.
    for f_name in filename_list:
        #Open the file
        with h5py.File(f_name,'r+') as hdf_file:
            if ALK.fcheck_file_datasets(hdf_file,[radiography_name]):
                #Subtract reference value from radiography
                hdf_file[radiography_name][...] = hdf_file[radiography_name][...]-ref_value
                print "File " + os.path.split(f_name)[-1] + " normalized successfully."
    
        
def ffind_dark_current(path, file_nums,filename_prefix=prefix,filename_suffix=HDF_suffix,
                    digits=4, variable_keys=None):
    '''Process a set of file numbers corresponding to dark files.
    Average the value of the variables named variable_keys over valid files.
    Returns a dictionary with the dark values.
    '''
    if not variable_keys:
        variable_keys = (I0_variable,PIN_variable)
    filename_list = ALK.fcreate_filename_list(file_nums, filename_prefix,
                                              filename_suffix, digits, path, check_exist=True)
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

def fanalyze_fluorescence(path,file_nums,name_dict,
                          filename_prefix=prefix,filename_suffix=HDF_suffix,
                          digits=4,radiography_name='Radiography'):
    '''Perform initial fluorescence analysis for a set of time-averaged files.
    The input name_dict is a dictionary whose keys are the desired new names of 
    variables, the values being the existing names.
    '''
    #Make a list of filenames
    filename_list = ALK.fcreate_filename_list(file_nums, filename_prefix,
                                              filename_suffix, digits, path, check_exist=True)
    #Loop through files
    #Loop through the file names
    for f_name in filename_list:
        #Open the file
        with h5py.File(f_name,'r+') as hdf_file:
            #Check that all required datasets exist
            if ALK.fcheck_file_datasets(hdf_file,[radiography_name,I0_variable,
                                                  slow_variable,fast_variable]):
                #Extract the radiography data, I0 data, absorption coefficient, and slow and fast events
                rad_data = hdf_file.get(radiography_name)
                I0_data = hdf_file.get(I0_variable)
                slow_events = hdf_file.get(slow_variable)
                fast_events = hdf_file.get(fast_variable)
                abs_coeff = hdf_file[radiography_name].attrs['Absorption_Coefficient']
                #Loop through the items in the name_dict
                for (key,value) in name_dict.items():
                    fluor_dataset = hdf_file.get(value)
                    if not fluor_dataset:
                        print "Problem processing " + f_name + ": dataset " + value + " does not exist."
                        print "Skipping this variable."
                    corrected_fluor = Radiography_Process.fcompute_fluorescence(fluor_dataset[...],
                                            slow_events[...], fast_events[...], rad_data[...], I0_data[...], abs_coeff)
                    #Call the processing routine and write the dataset to file
                    ALK.fwrite_HDF_dataset(hdf_file,key,corrected_fluor,{"Processing":'Dead_time,I0,attenuation'})
            
                    print "File " + os.path.split(f_name)[-1] + " analyzed for fluorescence successfully."
    
def frun_radiography_only(path,file_nums,ref_file_nums=None,filename_prefix=prefix,
              filename_suffix_MDA=MDA_suffix,filename_suffix_hdf=HDF_suffix,
              digits=4,dark_compute=False,dark_nums=[],abs_coeff=1,
              norm_function=ffind_normalization_radiography_minimums):
    '''Code to run through entire chain of analysis.
    Convert MDA to HDF5.
    Calculate radiography and write to the HDF file. 
    Normalize the radiography signal.
    '''
    #Convert to HDF5.  Do both regular files and dark files
    print "Converting to HDF5"
    fconvert_files_to_hdf5(path,file_nums+dark_nums,filename_prefix,filename_suffix_MDA,
                    digits)
    #Analyze for radiography
    fbatch_analyze_radiography(path,file_nums,filename_prefix,filename_suffix_hdf,
                    digits,dark_compute,dark_nums, abs_coeff)
    fnormalize_radiography(path,file_nums,ref_file_nums,filename_prefix,filename_suffix_hdf,
                    digits, 'Radiography',
                    norm_function)  

def frun_radiography_fluorescence(file_nums,names_dict,ref_file_nums=None,filename_prefix=prefix,
              filename_suffix_MDA=MDA_suffix,filename_suffix_hdf=HDF_suffix,
              digits=4,path=None,dark_compute=False,dark_nums=[],
              radiography_name='Radiography',abs_coeff=1,
              norm_function=ffind_normalization_radiography_minimums):
    '''Code to run through entire chain of analysis.
    Convert MDA to HDF5.
    Calculate radiography and write to the HDF file. 
    Normalize the radiography signal.
    Perform initial fluorescence data processing.
    Variables:
        file_nums: scan numbers of files to be processed
        names_dict: dictionary of fluorescence variables to be processed, in form new_name:EPICS_name
        ref_file_nums: list of scan numbers that can be used as a reference (no absorption)
        filename_prefix: string of characters found before scan number in MDA file names.
        filename_suffix_MDA: substring of filename after scan number for MDA files.
        filename_suffix_HDF: substring of filename after scan number for HDF5 files.
        digits: number of digits in file name for scan number
        path: path to the data files
        dark_compute: boolean to determine whether to calculate dark currents and accout for them.
        dark_nums: list of file numbers with no beam (used for dark currents)
        I_name: name of EPICS variable with transmission (I) data
        ref_name: name of EPICS variable with reference (I0) data
        slow_name: name of good events recorded by fluorescence pulse processor
        fast_name: name of variable for triggers from fluorescence pulse processor fast filter
        radiography_name: name to be given to processed radiography data
        abs_coeff: absorption coefficient to be used for radiography analysis
        norm_function: function to be used for normalizing the radiography
    '''
    #Default values evaluated when module is imported.  Following code
    #to make sure changes in default values get applied
    if not path:
        path = file_path
    frun_radiography_only(path,file_nums,ref_file_nums,filename_prefix,
              filename_suffix_MDA,filename_suffix_hdf,
              digits,dark_compute,dark_nums,abs_coeff,norm_function)
    fanalyze_fluorescence(path,file_nums,names_dict,
                          filename_prefix,filename_suffix_hdf,
                          digits, radiography_name)
    