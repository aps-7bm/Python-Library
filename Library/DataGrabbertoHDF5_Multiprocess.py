#! /usr/bin/env python
"""Module to convert DataGrabber files to HDF5.

Alan Kastengren, XSD, Argonne National Laboratory
Started: July 18, 2013

Edits:
May 5, 2014: Make each channel simply a dataset with attributes, rather than
            a group.  Things were getting too nested.
May 5, 2014: Surround reading of DataGrabber file with try block in case file
            doesn't exist.
May 5, 2014: Add method for batch processing.

"""

import readdatagrabber as rdg
import h5py
import sys
import numpy as np
import multiprocessing
#
def fconvert_file(dg_filename,hdf_filename=None,
                    path="/data/Data/SprayData/Cycle_2013_3/ISU/",
                    location_keys = ("X","Y"), convert_volts=True):
    '''Convert DataGrabber file to HDF5 if the input is the name of the output
    file rather than an HDF5 object.  Simply makes an HDF5 File object
    and calls the converter based on the HDF5 File object.
    '''
    #Make up an approrpiate output file name if none is given
    if not hdf_filename:
        hdf_filename = dg_filename.split(".")[0]+".hdf5"
    #Open an HDF file and call the converter based on having an HDF5 File object
    with h5py.File(path+hdf_filename,'w') as write_file:
        fconvert_h5py_hdffile(dg_filename,write_file,path,location_keys,convert_volts)

class LocationInfo():
    '''Class to store information about a given location.  No methods.
    '''
    def __init__(self):
        self.replicates = 0             #Number of repeats of this position
        self.header_coord_objects = []    #List of the coordinates to be added to this group
        self.location_values = []           #This will be a list of the location_keys
        self.dataset_sizes = {}         #Dict with names of channels and their lengths

def fparse_headers(dg_headers,location_keys):
    '''Loops through DataGrabber headers, looking for replicates
    of the same position.  It also accounts for the fact that the
    positions may be slightly different.
    Output: LocationInfo class with appropriate data for this location. 
    '''
    location_info_list = []
    #Loop through headers
    for coord in dg_headers:
        #Find the values of the variables we want as group keys
        key_values = []
        for key in location_keys:
            key_values.append(float(coord.coordinate_header[key]))
        #Loop through the location_info_list, looking for a match
        for loc_info in location_info_list:
            #If one is found, add a replicate and the coordinate object to the
            #internal list of coordinates, then break the loop
            if np.all(np.isclose(loc_info.location_values,key_values,atol=3e-3)):
                loc_info.replicates += 1
                loc_info.header_coord_objects.append(coord)
                #Loop through the channels that are part of this coordinate
                for channel in coord.channels:
                    channel_name = str(channel.channel_header["UserDescription"])
                    record_length = int(channel.channel_header['RecordLength'])
                    #Make sure we are tracking maximum record length
                    if not loc_info.dataset_sizes[channel_name] or record_length > loc_info.dataset_sizes[channel_name]:
                        loc_info.dataset_sizes[channel_name] = record_length
                break
        #If no match is found, make a new GroupInfo object
        else:
            new_loc = LocationInfo()
            new_loc.replicates = 1
            new_loc.header_coord_objects.append(coord)
            new_loc.location_values = key_values
            #Find the channels within this group and their sizes
            for channel in coord.channels:
                channel_name = str(channel.channel_header["UserDescription"])
                new_loc.dataset_sizes[channel_name] = int(channel.channel_header['RecordLength']) 
            location_info_list.append(new_loc)
    return location_info_list  

def fparse_dictionary_numeric_string(input_dict):
    '''Returns a copy of the input_dict dictionary, but only key:value
    pairs where the value is numeric.  Returns all values as floats.
    '''
    #Loop through dictionary, trying to parse the value into a number
    num_output = []
    string_output = []
    max_string_len = 0
    for key,value in input_dict.items():
        try:
            float(value)
            num_output.append[key]
        except:
            string_output.append(key)
            if max_string_len < len(value):
                max_string_len = len(value)
    return num_output,string_output,max_string_len
            
def fconvert_h5py_hdffile(dg_filename,write_file,
                    path="/home/beams/7BMB/SprayData/Cycle_2013_3/ISU/",
                    location_keys = ("X","Y"), convert_volts=True):
    '''Convert DataGrabber file to HDF5. Input h5py object,
    not hdf file name. 
    '''
    #Read in the header information
    try:
        headers = rdg.fread_headers(path+dg_filename)
    except IOError:
        print "Problem reading file " + dg_filename
        return
    #Parse through these header data
    location_info_list = fparse_headers(headers,location_keys) 
    #Find all possible channel names as well as sizes
    channel_sizes = {}
    max_replicates = 0
    for loc_info in location_info_list:
        #Loop through the channel names:
        for dset_name,dset_size in loc_info.dataset_sizes.items():
            if dset_name not in channel_sizes or dset_size > channel_sizes[dset_name]:
                channel_sizes[dset_name] = dset_size
        #Track the maximum number of replicates as well
        if loc_info.replicates > max_replicates:
            max_replicates = loc_info.replicates 
    
    #Lets see how the calculations worked.
    for loc_info in location_info_list:
        print str(loc_info.location_values)
        print "Replicates = " + str(loc_info.replicates)
    
    #Create top-level datasets for the channels, erasing old datasets if they exist
    for channel,c_size in channel_sizes.items():
        if channel in write_file.keys():
            del write_file[channel]
        #Make 2D arrays if there are no replications.  Otherwise, make 3D arrays.
        if max_replicates > 1:
            write_file.create_dataset(channel,shape=(len(location_info_list),c_size,max_replicates), dtype='f4')
            write_file[channel].attrs['Dimension_2'] = 'Replicates'
        else:
            write_file.create_dataset(channel,shape=(len(location_info_list),c_size), dtype='f4')
        #Label the dimensions of the datasets for posterity
        write_file[channel].attrs['Dimension_0'] = 'Location'
        write_file[channel].attrs['Dimension_1'] = 'Time'
    
    #Create top_level datasets for any numeric values from the coordinate headers.  Use first coordinate as a guide.
    numeric_header_keys, string_header_keys, max_string_len = fparse_dictionary_numeric_string(location_info_list[0].header_coord_objects[0].coordinate_header)
    for numeric_key in numeric_header_keys:
        write_file.create_dataset(numeric_key,shape=(len(location_info_list),max_replicates),dtype='f4')
    #Do the same for coordinate header values that are strings
    dtype_string = 'S' + str(max_string_len)
    for string_key in string_header_keys:
        write_file.create_dataset(string_key,shape=(len(location_info_list),max_replicates),dtype=dtype_string)
    
    print "Finished writing empty datasets"
    #Loop through the group_info objects
    for i,loc_info in enumerate(location_info_list):
        print "Working on position # " + str(i) + ", position " + str(loc_info.location_values)
        #Make arrays to hold all replicate channel data
        temp_array_dict = {}
        for channel,c_size in channel_sizes.items():
            temp_array_dict[channel] = np.zeros((c_size,max_replicates))
        #Loop through the header_coordinates in each group_info object
        for replicate,coord in enumerate(loc_info.header_coord_objects):
            print "Replicate # " + str(replicate)
            #Loop through the header, writing numbers to the appropriate datasets
            for numeric_key in numeric_header_keys:
                if coord.coordinate_header[numeric_key]:
                    write_file[numeric_key][i,replicate] = float(coord.coordinate_header[numeric_key])
            #Write string datasets for string header data
            for string_key in string_header_keys:
                if coord.coordinate_header[string_key]:
                    write_file[string_key][i,replicate] = str(coord.coordinate_header[string_key])
            #Loop through the channels
            for channel in coord.channels:
                channel_name = str(channel.channel_header["UserDescription"])
                #Read in the data, converting to volts if necessary
                if convert_volts:
                    channel.fread_data_volts()
                else:
                    channel.fread_data()
                #Put data into temp array
                temp_array_dict[channel_name][:channel.data.size,replicate] = channel.data
                channel.data = ""
                #Write attributes to the dataset to save scope settings
                for key in channel.channel_header.keys():
                    write_file[channel_name].attrs[key] = channel.channel_header[key]
                
        #Write the temp data to file
        if max_replicates > 1:
            for channel_name,array in temp_array_dict.items():
                write_file[channel_name][i,:,:] = array
        else:
            for channel_name,array in temp_array_dict.items():
                write_file[channel_name][i,:] = array[:,0]
        
        #Clear out this entry in headers so we don't overload memory
        coord = ""
        loc_info = ""
    #Close the file
    write_file.close()

def fbatch_conversion(file_nums,dg_filename_prefix='Scan_',dg_filename_suffix='.dat',
                    digits=4, path="/data/Data/SprayData/Cycle_2013_3/ISU/",
                    group_keys = ("X","Y"), convert_volts=True,processors=3):
    '''Class to batch process a large number of DataGrabber files.
    Should work with strings or numbers given in file_nums list.
    '''
    #Loop through the input string numbers, converting first to int, then to
    #correct number of digits as a string.
    filenames = []
    for i_str in file_nums:
        i = int(i_str)
        format_string = '{0:0'+str(digits)+'d}'
        filenames.append(dg_filename_prefix+format_string.format(i)+dg_filename_suffix)
    #Make a JoinableQueue to hold tasks
    #Set up processes
    tasks = multiprocessing.JoinableQueue()
    processes = [MP_Process(tasks) for i in range(processors)]
    for process in processes:
        process.start()
    #Try to set up multiprocessing JoinableQueue
    for f_name in filenames:
        tasks.put(MP_Task(f_name,None,path,group_keys,convert_volts))
    for i in range(processors):
        tasks.put(None)
    tasks.join()

class MP_Task():
    def __init__(self,f_name,hdf_fname,path,group_keys,convert_volts):
        self.f_name = f_name
        self.hdf_fname = hdf_fname
        self.path = path
        self.group_keys = group_keys
        self.convert_volts = convert_volts
    
    def __call__(self):
        fconvert_file(self.f_name,self.hdf_fname,self.path,self.group_keys,self.convert_volts)
        
class MP_Process(multiprocessing.Process):
    def __init__(self,task_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
    
    def run(self):
        while True:
            next_task = self.task_queue.get()
            if not next_task:
                self.task_queue.task_done()
                break
            next_task()
            self.task_queue.task_done()
        return

def fbatch_conversion_serial(file_nums,dg_filename_prefix='Scan_',dg_filename_suffix='.dat',
                    digits=4, path="/data/Data/SprayData/Cycle_2013_3/ISU/",
                    group_keys = ("X","Y"), convert_volts=True):
    '''Class to batch process a large number of DataGrabber files.
    Should work with strings or numbers given in file_nums list.
    '''
    #Loop through the input string numbers, converting first to int, then to
    #correct number of digits as a string.
    for i_str in file_nums:
        i = int(i_str)
        format_string = '{0:0'+str(digits)+'d}'
        fconvert_file(dg_filename_prefix+format_string.format(i)+dg_filename_suffix,
                    None,path,group_keys,convert_volts)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print "Code to convert DataGrabber files to HDF5 format."
    elif len(sys.argv) == 2:
        #Split input filename on period and change extension
        fconvert_file(sys.argv[1])
