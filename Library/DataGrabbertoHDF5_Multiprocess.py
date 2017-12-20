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
February 19, 2015: Major rewrite, allowing for batch processing with multiprocessing.
February 19, 2015: Make pertinent variables module attributes, rather than function arguments.

"""
import readdatagrabber as rdg
import h5py
import sys
import numpy as np
import multiprocessing

#Set module attributes
dg_filename_prefix='Scan_'
dg_filename_suffix='.dat'
hdf_filename = None
digits=4
file_path="/data/Data/SprayData/Cycle_2013_3/ISU/"
location_keys = ("X","Y")
convert_volts=True
processors=3
location_tolerance = 3e-3   #How close must coordinates be to be considered the same.
write_datatype = 'f4'
#
def fconvert_file(dg_filename,hdf_filename=None):
    '''Convert DataGrabber file to HDF5 if the input is the name of the output
    file rather than an HDF5 object.  Simply makes an HDF5 File object
    and calls the converter based on the HDF5 File object.
    '''
    #Make up an approrpiate output file name if none is given
    if not hdf_filename:
        hdf_filename = dg_filename.split(".")[0]+".hdf5"
    #Open an HDF file and call the converter based on having an HDF5 File object
    with h5py.File(file_path+hdf_filename,'w') as write_file:
        fconvert_h5py_hdffile(dg_filename,write_file)

class LocationInfo():
    '''Class to store information about a given location.  No methods.
    '''
    def __init__(self):
        self.replicates = 0             #Number of repeats of this position
        self.header_coord_objects = []    #List of the coordinates to be added to this group
        self.location_values = []           #This will be a list of the location_keys
        self.dataset_sizes = {}         #Dict with names of channels and their lengths

def fparse_headers(dg_headers):
    '''Loops through DataGrabber headers, looking for replicates
    of the same position.  It also accounts for the fact that the
    positions may be slightly different.
    Input: header info returned from readdatagrabber.py
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
            if np.all(np.isclose(loc_info.location_values,key_values,atol=location_tolerance)):
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
        #If no match is found, make a new LocationInfo object
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
    '''Parses a dictionary into two lists of keys, one whose values cast to a
    numeric type, the other not (assumed to be just strings).  
    Also returns maximum length value length for the string keys.
    '''
    #Loop through dictionary, trying to parse the value into a number
    num_output = []
    string_output = []
    max_string_len = 0
    for key,value in input_dict.items():
        try:
            float(value)
            num_output.append(key)
        except ValueError:
            string_output.append(key)
            if max_string_len < len(value):
                max_string_len = len(value)
        except TypeError:
            print("Value = " + str(value))
            raise TypeError
    return num_output,string_output,max_string_len

def fparse_channel_names_sizes(location_info_list):
    '''Creates a list with the channel names and the maximum record length
    of these channels in the data file.
    Input:
    location_info_list: list of LocationInfo objects
    Output:
    channel_sizes: dict in form channel_name:max_record_length
    max_replicates: maximum number of times any location is replicated
    '''
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
    return channel_sizes,max_replicates

def fcreate_datasets_channels(hdf_group,channel_sizes,max_replicates,num_locations):
    '''Writes zeros datasets to HDF5 Group for each channel to preallocate.
    Inputs:
    hdf_file: h5py Group object to which to write the datasets
    channel_size: dict in form channel_name:max_record_length
    max_replicates: maximum number of times any location is replicated
    num_locations: number of locations
    '''
    #Create top-level datasets for the channels, erasing old datasets if they exist
    for channel,c_size in channel_sizes.items():
        if channel in hdf_group.keys():
            del hdf_group[channel]
        #Make 2D arrays if there are no replications.  Otherwise, make 3D arrays.
        if max_replicates > 1:
            hdf_group.create_dataset(channel,shape=(num_locations,c_size,max_replicates), dtype=write_datatype)
            hdf_group[channel].dims[2].label = 'Replicates'
        else:
            hdf_group.create_dataset(channel,shape=(num_locations,c_size), dtype=write_datatype)
        #Label the dimensions of the datasets for posterity
        hdf_group[channel].dims[0].label = 'Location'
        hdf_group[channel].dims[1].label = 'Time'
        
def fcreate_datasets_coord_header(hdf_group,coord_header,max_replicates,num_locations):
    '''Preallocates datasets for the variables saved in the coordinate headers.
    Inputs:
    hdf_file: h5py Group object to which to write the datasets
    coord_header: example readdatagrabber.py Coordinate.coordinate_header
    max_replicates: maximum number of times any location is replicated
    num_locations: number of locations
    '''
    #Create top_level datasets for any numeric values from the coordinate headers.  Use first coordinate as a guide.
    print(coord_header)
    numeric_header_keys, string_header_keys, max_string_len = fparse_dictionary_numeric_string(coord_header)
    for numeric_key in numeric_header_keys:
        hdf_group.create_dataset(numeric_key,shape=(num_locations,max_replicates),dtype=write_datatype)
    #Do the same for coordinate header values that are strings
    dtype_string = 'S' + str(max_string_len)
    for string_key in string_header_keys:
        hdf_group.create_dataset(string_key,shape=(num_locations,max_replicates),dtype=dtype_string)
    return numeric_header_keys, string_header_keys

def fconvert_h5py_hdffile(dg_filename,write_file):
    '''Convert DataGrabber file to HDF5.
    Input:
    dg_filename: name of DataGrabber file, not prepended with path.
    write_file: HDF5 Group object to which to write data
    '''
    #Read in the header information
    try:
        headers = rdg.fread_headers(file_path+dg_filename)
    except IOError:
        print("Problem reading file " + dg_filename)
        return
    #Parse through these header data to find locations and number of replicates
    location_info_list = fparse_headers(headers) 
    #Find all possible channel names as well as maximum sizes
    channel_sizes,max_replicates = fparse_channel_names_sizes(location_info_list)
        
    #Lets see how the calculations worked.
    for loc_info in location_info_list:
        print(str(loc_info.location_values))
        print("Replicates = " + str(loc_info.replicates))
    
    #Preallocate arrays in the HDF5 file from the channels
    fcreate_datasets_channels(write_file,channel_sizes,max_replicates,len(location_info_list))
    
    #Preallocate arrays in the HDF5 from the coordinate headers
    numeric_header_keys, string_header_keys = fcreate_datasets_coord_header(write_file,
                                            location_info_list[0].header_coord_objects[0].coordinate_header,
                                            max_replicates,len(location_info_list))
    print("Finished writing empty datasets")     #Feedback to user
    
    #Loop through the LocationInfo objects
    for i,loc_info in enumerate(location_info_list):
        print("Working on position # " + str(i) + ", position " + str(loc_info.location_values))
        #Make arrays to hold all replicate channel data
        temp_array_dict = {}
        for channel,c_size in channel_sizes.items():
            temp_array_dict[channel] = np.zeros((c_size,max_replicates))
        #Loop through the header_coordinates in each group_info object
        for replicate,coord in enumerate(loc_info.header_coord_objects):
            print("Replicate # " + str(replicate))
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
                
        #Write the temp data to file once we've looped through all replicates.  
        #Use one big write to speed up the file writing.
        if max_replicates > 1:
            for channel_name,array in temp_array_dict.items():
                write_file[channel_name][i,:,:] = array
        else:
            for channel_name,array in temp_array_dict.items():
                write_file[channel_name][i,:] = array[:,0]
        
        #Clear out entries in headers so we don't overload memory
        coord = ""
        loc_info = ""

def fbatch_conversion(file_nums):
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
    tasks = multiprocessing.JoinableQueue()
    #Set up processes
    processes = [MP_Process(tasks) for i in range(processors)]
    for process in processes:
        process.start()
    #Set up multiprocessing JoinableQueue
    for f_name in filenames:
        tasks.put(MP_Task(f_name,hdf_filename))
    #Add poison pills at the end of the queue so MP_Process objects know when to stop.
    for i in range(processors):
        tasks.put(None)
    #Wait for tasks to finish
    tasks.join()

class MP_Task():
    '''Object to allow for file conversion to occur in a JoinableQueue
    '''
    def __init__(self,f_name,hdf_fname):
        self.f_name = f_name
        self.hdf_fname = hdf_fname
    
    def __call__(self):
        fconvert_file(self.f_name,self.hdf_fname)
        
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

def fbatch_conversion_serial(file_nums):
    '''Class to batch process a large number of DataGrabber files.
    Should work with strings or numbers given in file_nums list.
    '''
    #Loop through the input string numbers, converting first to int, then to
    #correct number of digits as a string.
    for i_str in file_nums:
        i = int(i_str)
        format_string = '{0:0'+str(digits)+'d}'
        fconvert_file(dg_filename_prefix+format_string.format(i)+dg_filename_suffix,
                    hdf_filename)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Code to convert DataGrabber files to HDF5 format.")
    elif len(sys.argv) == 2:
        #Split input filename on period and change extension
        fconvert_file(sys.argv[1])