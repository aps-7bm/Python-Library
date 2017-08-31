#! /usr/bin/env python
"""Module to reduce highly time-resolved scope data from 7-BM radiography.

Based on processing code for AFRL, cycle 2012-3.

Workflow is to:
1. Read data from DataGrabber files.
2. Subtract dark current.
3. Normalize by I0 signal.

Processing code for AFRL single-pulse resolved measurements from late CY 2012

Alan Kastengren, XSD, Argonne National Laboratory
Started: December 4, 2012

Revisions:
February 11, 2016: add in code to allow for signals to be simply read in and 
                    normalized, even if they aren't peaked.  Also some minor 
                    refactoring.
February 12, 2016: add in ability to exclude groups from being written from
                    coordinate headers.
                    
June 9, 2016: major rewrite of how binning is handled.


"""

import numpy as np
import readdatagrabber as rdg
import h5py
import ALK_Utilities as ALK
import multiprocessing
from __builtin__ import TypeError
import os.path
import logging

read_file_path = 'home/akastengren/data/SprayData/Cycle_2015_1/Radiography/'
write_file_path = read_file_path
channels = []
pulse_time = 153e-9         #Time between bunches in 24 bunch mode at APS as of 2015
start_time = 0              #Time at which to start integrations
num_processors = 8
replicated = True
digits = 4
prefix = 'Scan_'
suffix = '.dat'
dark_file_num = None
location_tolerance = 3e-3
location_keys = ("X","Y")
write_datatype = 'f4'
exclude_groups = []
pulsed_signal = True   #If True, signals are pulsed.

#Set up logging
logger = logging.getLogger('Time_Resolved_Processing_Cython')
logger.addHandler(logging.NullHandler())

class ChannelInfo():
    '''Object to hold parameters for reading in a DataGrabber channel.
    
    Inputs:
    desc_name: the descriptor name saved by DataGrabber
    new_name: what do we want to call this
    bin_method: what method should be used to bin the data
    repeat_num: if there is a repeating pattern, how many bins does it repeat
    dark_subtract: boolean to tell if dark current should be subtracted from this channel
    dark_value: dark current value to be used for dark subtraction, if desired
    pulse_time_base: is this channel a pulsed signal that will be used as a time base?
    
    '''
    def __init__(self,desc_name,new_name,bin_method,repeat_num=1,dark_subtract=False,pulse_time_base=False):
        self.desc_name = desc_name
        self.new_name = new_name
        self.bin_method = bin_method
        self.repeat_num = repeat_num 
        self.dark_subtract = dark_subtract 
        self.dark_value = 0
        self.pulse_time_base = pulse_time_base
        self.temp_data = None
        logger.info("ChannelInfo object created with description " + desc_name)
        
class LocationInfo():
    '''Class to store information about a given location.  No methods.
    '''
    def __init__(self):
        self.replicates = 0             #Number of repeats of this position
        self.header_coord_objects = []    #List of the coordinates to be added to this group
        self.location_values = []           #This will be a list of the location_keys
        self.dataset_sizes = {}         #Dict with names of channels and their lengths
        
    def fget_num_replicates(self):
        return len(self.header_coord_objects)

def fparse_headers(dg_headers):
    '''Loops through DataGrabber headers, looking for replicates
    of the same position.  It also accounts for the fact that the
    positions may be slightly different.
    Input: header info returned from readdatagrabber.py
    Output: LocationInfo list with appropriate data for each location. 
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
            #If one is found, add the coordinate object to the
            #internal list of coordinates, then break the loop
            if np.all(np.isclose(loc_info.location_values,key_values,atol=location_tolerance)):
                loc_info.header_coord_objects.append(coord)
                #Loop through the channels that are part of this coordinate
                for channel in coord.channels:
                    channel_name = str(channel.channel_header["UserDescription"])
                    record_length = int(channel.channel_header['RecordLength'])
                    logger.debug("Channel name " + channel_name)
                    #Make sure we are tracking maximum record length
                    if not loc_info.dataset_sizes[channel_name] or record_length > loc_info.dataset_sizes[channel_name]:
                        loc_info.dataset_sizes[channel_name] = record_length
                break
        #If no match is found, make a new LocationInfo object
        else:
            new_loc = LocationInfo()
            new_loc.header_coord_objects.append(coord)
            new_loc.location_values = key_values
            logger.debug("New location found: " + str(new_loc.location_values))
            
            #Find the channels within this group and their sizes
            for channel in coord.channels:
                channel_name = str(channel.channel_header["UserDescription"])
                logger.debug("Channel name " + channel_name)
                new_loc.dataset_sizes[channel_name] = int(channel.channel_header['RecordLength']) 
            location_info_list.append(new_loc)
    return location_info_list 

def fread_coordinate_data(coordinate,update_pulse_info=False):
    '''Reads the data for a coordinate and bins.
    Inputs:
    coordinate: DataGrabberCoordinate object with data
    update_pulse_time: if True, use pulsed data to update the module pulse time variable

    '''
    #Sort the channels by the update_pulse_time variable, putting True one first
    sorted_channels = sorted(channels,key=lambda x:x.pulse_time_base,reverse=True)
    #Check to make sure we don't have more than one channel set as pulse_time_base
    num_pulse_time_base=0
    for chan in sorted_channels:
        if chan.pulse_time_base:
            num_pulse_time_base += 1
    assert(num_pulse_time_base < 2)
    #Loop through the ChannelInfoChannel objects
    for chinfochannel in sorted_channels:
        #Update the pulse time if it is desired.  Store data in ChannelInfo object for now.
        chinfochannel.temp_data = fread_channel_data(coordinate,chinfochannel,
                                            update_pulse_info and chinfochannel.pulse_time_base)
        #Subtract dark values
        chinfochannel.temp_data -= chinfochannel.dark_value
    return 

def fread_channel_data(coord_obj,channel_obj,update_pulse_info=False):
    '''Uses the ChannelInfo object channel_obj parameters
    to read the appropriate data from this channel of the input object
    coord_obj.
    Inputs:
    coord_obj: the readdatagrabber.DataGrabberCoordinate object being read
    channel_obj: the ChannelInfo object with readin parameters
    update_pulse_time: if True, update the global pulse_time based on readin
    Outputs:
    output_data: binned channel data
    new_pulse_time: time duration of the pulses for pulsed data.
    '''
    #Pick the appropriate coordinate, get channel with descriptor in it
    for channel in coord_obj.channels:
        if channel.channel_header["UserDescription"]==channel_obj.desc_name: 
            logger.debug("Processing channel " + channel_obj.desc_name)
            global pulse_time, start_time
            #Read in the channel data
            channel.fread_data_volts()
            #Get the delta_t for these data
            delta_t = float(coord_obj.fread_channel_meta_data(channel_obj.desc_name,"TimeStep"))
            #Bin the data
            output_data,new_pulse_time,new_start_time = channel_obj.bin_method(channel.data,delta_t,
                                                            pulse_time,channel_obj.repeat_num, start_time)
            #If desired, update the pulse time and start_time
            if update_pulse_info:
                logger.info("Using channel " + str(channel_obj.desc_name + " to update timing."))
                logger.info("Pulse time updated to " + str(new_pulse_time))
                logger.info("Start time updated to " + str(new_start_time))
                pulse_time = new_pulse_time
                start_time = new_start_time
            #Delete the channel data to conserve memory
            del(channel.data)
            return output_data

def ffind_dark_current(dark_num,directory=None):
    """Function to find values from a dark scan.  Save values to the
    ChannelInfo object for each channel.
    """
    #Use the default directory unless one is give
    if not directory:
        directory = read_file_path
    #If there is no valid dark file, just return
    if not dark_num:
        logger.info("No dark file number provided, so skipping dark current calculation.")
        return
    
    #Figure out if we've already processed this one.  If so, use it
    filename = ALK.fcreate_filename(dark_num,prefix,suffix,digits)
    hdf_filename = filename.split('.')[0] + '.hdf5'
    if os.path.isfile(write_file_path + hdf_filename):
        logger.info('Dark current file already processed.')
        write_hdf = h5py.File(write_file_path+hdf_filename,'r')
        for chinfochannel in channels:
            chinfochannel.dark_value = float(write_hdf.attrs[chinfochannel.desc_name])  
            print(chinfochannel.desc_name + " " + str(chinfochannel.dark_value)) 
        write_hdf.close()
        return
    #Read in headers
    filename = ALK.fcreate_filename(dark_num,prefix,suffix,digits)
    logger.info("Dark filename = " + directory + filename)
    if not os.path.isfile(directory+filename):
        logger.error("Dark file doesn't exist.  Returning.")
        return
    #Read in the header information
    try:
        header_data = rdg.fread_headers(directory+filename)
    except IOError:
        logger.error("Problem reading file " + filename)
        return
    #Loop through the coordinates measured
    for coordinate in header_data:
        #Loop through the ChannelInfo objects
        for chinfochannel in channels:
            #If this isn't one for which we are supposed to dark subtract, skip
            if not chinfochannel.dark_subtract:
                continue
            #Find the channel with the right description
            chan = coordinate.fchannel_by_name(chinfochannel.desc_name)
            #Read in data
            chan.fread_data_volts()
            chinfochannel.dark_value += np.mean(chan.data,dtype=np.float64)
            #Delete channel data to conserve memory and break loop: we found our channel
            del(chan.data)
    
    #Make an HDF file to hold these data so we don't have to constantly reprocess them
    write_file = h5py.File(write_file_path+hdf_filename,'w')
        
    #Divide by the number or coordinates to get the right dark value
    for chinfochannel in channels:
        chinfochannel.dark_value /= float(len(header_data))  
        print(chinfochannel.desc_name + " " + str(chinfochannel.dark_value)) 
        write_file.attrs[chinfochannel.desc_name] = chinfochannel.dark_value
    write_file.close()
    logger.info(str(len(header_data) + 1) + " coordinates processed for dark current.")
    return

def fcreate_datasets_channels(hdf_group,max_replicates,num_locations):
    '''Writes zeros datasets to HDF5 Group for each channel to preallocate.
    Inputs:
    hdf_file: h5py Group object to which to write the datasets
    max_replicates: maximum number of times any location is replicated
    num_locations: number of locations
    '''
    #Create top-level datasets for the channels, erasing old datasets if they exist
    for chinfochannel in channels:
        #Remove this channel from the hdf file if it already exists
        if chinfochannel.new_name in hdf_group.keys():
            del hdf_group[chinfochannel.new_name]
        #What is the length of the temporary data we've already read in
        c_size = chinfochannel.temp_data.shape[0]
        #Make 2D arrays if there are no replications.  Otherwise, make 3D arrays.
        if max_replicates > 1:
            hdf_group.create_dataset(chinfochannel.new_name,shape=(num_locations,c_size,max_replicates), dtype=write_datatype)
            hdf_group[chinfochannel.new_name].dims[2].label = 'Replicates'
            logger.info("Created 3-dimensional HDF5 dataset " + chinfochannel.new_name)
        else:
            hdf_group.create_dataset(chinfochannel.new_name,shape=(num_locations,c_size), dtype=write_datatype)
            logger.info("Created 2-dimensional HDF5 dataset " + chinfochannel.new_name)
        #Label the dimensions of the datasets
        hdf_group[chinfochannel.new_name].dims[0].label = 'Location'
        hdf_group[chinfochannel.new_name].dims[1].label = 'Time'
        
def fparse_dictionary_numeric_string(input_dict):
    '''Parses a dictionary into two lists of keys, one whose values cast to a
    numeric type, the other not (assumed to be just strings).  
    Also returns maximum length value length for the string keys.
    Inputs:
    input_dict: dictionary of key:value pairs from a header line
    Outputs:
    num_output: list of names of keys which give numeric outputs
    string_output: list of names of keys which give string outputs
    max_string_len: maximum length of strings needed for string key values
    '''
    #Loop through dictionary, trying to parse the value into a number
    num_output = []
    string_output = []
    for key,value in input_dict.items():
        if key in exclude_groups:
            continue
        try:
            float(value)
            num_output.append(key)
        except ValueError:
            string_output.append(key)
        except TypeError:
            logging.error("Problem parsing dictionary: value = " + str(value))
            raise TypeError
    #Find the maximum length of a string
    max_string_len = max([len(x) for x in string_output])
    return num_output,string_output,max_string_len

def fcreate_datasets_coord_header(hdf_group,coord_header,max_replicates,num_locations):
    '''Preallocates datasets for the variables saved in the coordinate headers.
    
    Inputs:
    hdf_group: h5py Group object to which to write the datasets
    coord_header: example readdatagrabber.py Coordinate.coordinate_header
    max_replicates: maximum number of times any location is replicated
    num_locations: number of locations
    '''
    #Create top_level datasets for any numeric values from the coordinate headers.  Use first coordinate as a guide.
    numeric_header_keys, string_header_keys, max_string_len = fparse_dictionary_numeric_string(coord_header)
    dtype_string = 'S' + str(max_string_len)
    if replicated:
        for numeric_key in numeric_header_keys:
            hdf_group.create_dataset(numeric_key,shape=(num_locations,max_replicates),dtype=write_datatype)
            logging.info("Created HDF5 group for numerical header dataset " + numeric_key)
        #Do the same for coordinate header values that are strings
        for string_key in string_header_keys:
            hdf_group.create_dataset(string_key,shape=(num_locations,max_replicates),dtype=dtype_string)
            logging.info("Created HDF5 group for string header dataset " + string_key)
    else:
        for numeric_key in numeric_header_keys:
            hdf_group.create_dataset(numeric_key,shape=(num_locations,),dtype=write_datatype)
            logging.info("Created HDF5 group for numerical header dataset " + numeric_key)
        #Do the same for coordinate header values that are strings
        for string_key in string_header_keys:
            hdf_group.create_dataset(string_key,shape=(num_locations,),dtype=dtype_string)
            logging.info("Created HDF5 group for string header dataset " + string_key)
    return numeric_header_keys, string_header_keys

def fconvert_to_hdf5_multiprocess(file_num,directory=None,write_filename=None):
    """Function to fully process PIN data at the specified coordinate.
        Writes processed PIN data to an HDF5 file using multiprocessing.
        Each position is saved as one group.  Only coordinate (not channel) header
        data saved as attributes of dataset.
    """
    #Make the filename
    filename = ALK.fcreate_filename(file_num,prefix,suffix,digits)
    #Set the directory and write_filename to reasonable defaults if they aren't specified
    if not directory:
        directory = read_file_path
    if not write_filename:
        write_filename = filename.split('.')[0] + '.hdf5'
    logger.info("Reading from DataGrabber filename " + directory + filename)
    
    logger.info("Writing to HDF5 file " + write_file_path + write_filename)
    #Bail out if the file doesn't exist
    if not os.path.isfile(directory+filename):
        logger.error("Read file " + directory + filename + " doesn't exist.")
        return
    #Open an HDF5 file to save the data.
    with h5py.File(write_file_path+write_filename,'w') as write_file:
        #Read in the header information
        try:
            headers = rdg.fread_headers(directory+filename)
        except IOError:
            logger.error("Problem reading file " + filename)
            return
        
        #Parse through these header data to find locations and number of replicates
        location_info_list = fparse_headers(headers) 
        
        #Find the maximum number of replicates
        max_replicates = max([x.fget_num_replicates() for x in location_info_list])
        global replicated
        replicated = max_replicates > 1
                    
        #Lets see how the calculations worked.
        for loc_info in location_info_list:
            print("Replicates = " + str(loc_info.fget_num_replicates()) + " at location " + str(loc_info.location_values) + ", file " + filename)
            logger.info("Replicates = " + str(loc_info.fget_num_replicates()) + " at location " + str(loc_info.location_values) + ", file " + filename)
            
        #Find dark currents: only do this once
        ffind_dark_current(dark_file_num)
   
        #Process the first coordinate to update pulse time and get the sizes of the arrays
        fread_coordinate_data(headers[0],True)
        
        #Add a time step variable to the saved datasets
        write_file.attrs['Time_Step'] = pulse_time
        
        #Preallocate arrays in the HDF5 file from the channels
        fcreate_datasets_channels(write_file,max_replicates,len(location_info_list))
        
        #Preallocate arrays in the HDF5 from the coordinate headers
        numeric_keys, string_keys = fcreate_datasets_coord_header(write_file,
                                                location_info_list[0].header_coord_objects[0].coordinate_header,
                                                max_replicates,len(location_info_list))
        logger.info("Finished writing empty datasets")     #Feedback to user
        
        #Loop through the LocationInfo objects
        for i,loc_info in enumerate(location_info_list):
            logger.info("Working on position # " + str(i) + ", position " + str(loc_info.location_values) + " file " + filename)
            for (replicate_num,coordinate) in enumerate(loc_info.header_coord_objects):
                fprocess_coordinate(coordinate,write_file,numeric_keys,
                            string_keys,i,replicate_num)
                #Clear out entries in headers so we don't overload memory
                coordinate = ""
            loc_info = ""

def fprocess_coordinate(coordinate,hdf_group,numeric_keys,string_keys,location_num,replicate):
    '''Processes one coordinate of the DataGrabber file.
    '''
    logger.debug("Replicate # " + str(replicate))
    #Read the binned data.  This also performs dark subtraction
    fread_coordinate_data(coordinate,False)
    #Start looping through the ChannelInfo objects
    for chinfochannel in channels:
        signal_name = chinfochannel.new_name
        final_data = chinfochannel.temp_data
        #Take care of case where number of read points is different from preallocated size
        max_length = hdf_group[signal_name].shape[1]
        if chinfochannel.temp_data.shape[0] < max_length:
            max_length = final_data.shape[0]
        #Put the data into the proper place in the preallocated array
        if replicated:
            hdf_group[signal_name][location_num,:max_length,replicate] = final_data[:max_length]
            #Write numeric values from headers
            for numeric_key in numeric_keys:
                if numeric_key in coordinate.coordinate_header:
                    hdf_group[numeric_key][location_num,replicate] = float(coordinate.coordinate_header[numeric_key])
            #Write string datasets for string header data
            for string_key in string_keys:
                if string_key in coordinate.coordinate_header:
                    hdf_group[string_key][location_num,replicate] = str(coordinate.coordinate_header[string_key])
        else:
            hdf_group[signal_name][location_num,:max_length] = final_data[:max_length]
        #Write numeric values from headers
        for numeric_key in numeric_keys:
            if numeric_key in coordinate.coordinate_header:
                hdf_group[numeric_key][location_num] = float(coordinate.coordinate_header[numeric_key])
        #Write string datasets for string header data
        for string_key in string_keys:
            if string_key in coordinate.coordinate_header:
                hdf_group[string_key][location_num] = str(coordinate.coordinate_header[string_key])


def fbatch_conversion_multiprocess(file_nums):
    '''Class to batch process a large number of DataGrabber files.
    Should work with strings or numbers given in file_nums list.
    '''
    print("Setting up multiprocessing of " + str(len(file_nums)) + " files.")
    #Make a JoinableQueue to hold tasks
    tasks = multiprocessing.JoinableQueue()
    #Set up processes
    processes = [MP_Process(tasks) for i in range(num_processors)]
    for process in processes:
        process.start()
    #Set up multiprocessing JoinableQueue
    for file_num in file_nums:
        tasks.put(MP_Task(file_num))
    #Add poison pills at the end of the queue so MP_Process objects know when to stop.
    for i in range(num_processors):
        tasks.put(None)
    #Wait for tasks to finish
    tasks.join()       
    
def fbatch_conversion_serial(file_nums):
    '''Class to batch process a large number of DataGrabber files.
    Should work with strings or numbers given in file_nums list.
    '''
    for file_num in file_nums:
        fconvert_to_hdf5_multiprocess(file_num)

class MP_Task():
    '''Object to allow for file conversion to occur in a JoinableQueue
    '''
    def __init__(self,file_num):
        self.file_num = file_num
    
    def __call__(self):
        fconvert_to_hdf5_multiprocess(self.file_num)

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
