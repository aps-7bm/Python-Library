'''Script to find the appropriate dark currents for time-averaged hdf files.

Alan Kastengren, XSD, Argonne

Started: May 5, 2014
'''
import h5py
import numpy as np
import ALK_Utilities as ALK

def ffind_dark_current(file_nums,filename_prefix='7bmb1_',filename_suffix='.hdf5',
                    digits=4, path="/data/Data/SprayData/Cycle_2014_1/ISU_Point/",
                    variable_keys = ("7bmb1:scaler1.S3","7bmb1:scaler1.S5")):
    '''Process a set of file numbers corresponding to dark files.
    Average the value of the variables named variable_keys over valid files.
    Returns a dictionary with the dark values.
    '''
    filename_list = ALK.fcreate_filename_list(file_nums, filename_prefix,
                                              filename_suffix, digits, path, check_exist=True)
    #Loop through the input string numbers, converting first to int, then to
    #correct number of digits as a string.
    for i_str in file_nums:
        i = int(i_str)
        format_string = '{0:0'+str(digits)+'d}'
        filename_list.append(path+filename_prefix+format_string.format(i)+filename_suffix)
    #Set up a dictionary to hold arrays (and eventually scalars) with dark values
    dark_values = {}
    for v_name in variable_keys:
        dark_values[v_name] = []
    #Loop through the file names
    for f_name in filename_list:
        #Open the file
        with h5py.File(f_name,'r') as hdf_file:
            for key in variable_keys:
                dark_values[key].append(np.mean(hdf_file.get(key)[...]))
        print "File " + f_name.split("//")[-1] + " processed successfully."
    print dark_values
    #Average the values 
    for v_name in variable_keys:    
        dark_values[v_name] = np.mean(np.array(dark_values[v_name]))
    return dark_values

if __name__ == '__main__':
    file_nums = [666,686,698]
    print ffind_dark_current(file_nums)