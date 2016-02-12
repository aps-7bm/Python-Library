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


"""

import numpy as np
import readdatagrabber as rdg
import scipy.interpolate
import scipy.stats
import h5py
import ALK_Utilities as ALK
import multiprocessing
import ArrayBin_Cython as arc
from __builtin__ import TypeError

file_path = ALK.fcorrect_path_start() + 'SprayData/Cycle_2015_1/Radiography/'
I_name = 'PINDiode'
I0_name = 'BIM'
pulse_time = 153e-9         #Time between bunches in 24 bunch mode at APS as of 2015
bunches = 24
num_processors = 6
replicated = True
digits = 4
prefix = 'Scan_'
suffix = '.dat'
dark_file_num = None
threshold_value = 0.05
location_tolerance = 3e-3
location_keys = ("X","Y")
write_datatype = 'f4'
exclude_groups = []

def fcreate_filename(file_num):
    return ALK.fcreate_filename(file_num,prefix,suffix,digits)
    
def fread_meta_data(coord_object,descriptor,key):
    for channel in coord_object.channels:
        if channel.channel_header['UserDescription'] == descriptor:
            return channel.channel_header[key]
        
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
                    #Make sure we are tracking maximum record length
                    if not loc_info.dataset_sizes[channel_name] or record_length > loc_info.dataset_sizes[channel_name]:
                        loc_info.dataset_sizes[channel_name] = record_length
                break
        #If no match is found, make a new LocationInfo object
        else:
            new_loc = LocationInfo()
            new_loc.header_coord_objects.append(coord)
            new_loc.location_values = key_values
            #Find the channels within this group and their sizes
            for channel in coord.channels:
                channel_name = str(channel.channel_header["UserDescription"])
                new_loc.dataset_sizes[channel_name] = int(channel.channel_header['RecordLength']) 
            location_info_list.append(new_loc)
    return location_info_list 
        
def fread_bin_signal_by_bunch(coord_object,descriptor="PINDiode"):
    """Function to read in PIN data, find peak positions, and bin each peak.
    Inputs:
    coord_object: readdatagrabber.py DataGrabberCoordinate object to be processed
    descriptor: UserDescription value for the channel we want
    Outputs:
    bunch_removed_output: binned signal with bunch charge variations removed
    pulse_time: actual time between synchrotron pulses.
    """   
    #Pick the appropriate coordinate, get channel with descriptor in it
    for channel in coord_object.channels:
        if channel.channel_header["UserDescription"]==descriptor: 
            global pulse_time           #We will update this as we go along.
            #Read in data on this channel
            channel.fread_data_volts()
            #If the peak value is too low, we may have no beam.  Return nothing,
            #as our peak finding routine won't work
            if np.mean(channel.data) < threshold_value:
                channel.data = None
                return None,None
            #Find the peak positions
            delta_t = float(fread_meta_data(coord_object,descriptor,"TimeStep"))
            #Use the new Cython version of this
            peak_positions = arc.ffind_breakpoints_peaked(channel.data,int(pulse_time / delta_t))
            #Define numpy array to hold final data
            output_data = np.zeros(peak_positions.size)
            #Compute the actual average pulse duration and add to coordinate header
            peak_regression = scipy.stats.linregress(np.arange(peak_positions.size),peak_positions)
#             print peak_positions[0]
#             print peak_regression
            actual_pulse_duration = peak_regression[0]
            pulse_time = actual_pulse_duration * delta_t
            print "Pulse duration = ",pulse_time, " s."
            coord_object.coordinate_header["Pulse_Duration"] = pulse_time
            #Find the actual number of points before the peak that we should use
            points_before_peak = ffind_points_before_peak(channel.data,peak_positions,actual_pulse_duration)
#             #Make array of the breakpoints for integration between points, including one extra for end of array
#             breakpoints = np.linspace(peak_positions[0],peak_positions[-1] + actual_pulse_duration,len(peak_positions)+1) - points_before_peak
#             print "Mean of raw data = " , np.mean(channel.data[peak_positions[0]:peak_positions[-1]])
#             for counter in range(len(peak_positions)):
#                 fintegrate_peak(channel.data,output_data,counter,breakpoints)
#                 #Print out a message every 100000 peaks so one can see that it's working.
# #                 if not(counter%5000):
            #Bin the array, starting at the minimum before the first peak.  Use linear fit to find first peak location
            start_point = int(np.rint(peak_regression[1] - points_before_peak))
            if start_point < 0:
                start_point = int(np.rint(peak_regression[1] + peak_regression[0] - points_before_peak))
            output_data = arc.fbin_array(channel.data[start_point:],actual_pulse_duration)
            print "Processed ", len(peak_positions), " peaks for channel " + descriptor
            #Divide the output data by the pulse duration to get an average voltage back
            output_data = output_data / actual_pulse_duration
            print np.mean(output_data,dtype=np.float64)
            #Attempt to remove the impact of bunch charge variations.
            bunch_removed_output = fremove_bunch_charge_variations(output_data)
            assert np.allclose(np.mean(output_data,dtype=np.float64),np.mean(bunch_removed_output,dtype=np.float64),1e-7,1e-7)
            #Delete the array data from the channel to avoid maxing out memory
            channel.data = None
            return bunch_removed_output,pulse_time


def ffind_points_before_peak(data_array,peak_positions,pulse_duration):
    '''Figure out how many points before the peak we should go to perform integration.
    Inputs:
    data_array: actual data values
    peak_positions: peak points for each bunch.
    pulse_duration: average pulse duration in time steps
    Outputs:
    points_before_peak: how many time steps before peak to use for integration
    '''
    #Look over first one hundred peaks
    points_before_peak_array = np.zeros(100)
    back_step = int(2*pulse_duration/3)
    for i in range(1,101):
        minimum_point = np.argmin(data_array[peak_positions[i]-back_step:peak_positions[i]])
        points_before_peak_array[i-1] = float(back_step - minimum_point)
    return np.mean(points_before_peak_array)

def fremove_bunch_charge_variations(input_array):
    '''Removes the bunch charge variations from the data.
    '''
    output_array = np.zeros(np.size(input_array))
    multiplier = np.zeros(bunches)
    for i in range(bunches):
        multiplier[i] = np.sum(input_array[i::bunches],dtype=np.float64)
    #print multiplier
    av_multiplier = np.mean(multiplier,dtype=np.float64)    
    for i in range(bunches):
        output_array[i::bunches] = input_array[i::bunches] * av_multiplier / multiplier[i]
    return output_array

def fread_signal_direct(coord_object,descriptor="PINDiode"):
    """Function to read in data directly
    Inputs:
    coord_object: readdatagrabber.py DataGrabberCoordinate object to be processed
    descriptor: UserDescription value for the channel we want
    Outputs:
    bunch_removed_output: binned signal with bunch charge variations removed
    pulse_time: delta t between points
    """   
    #Pick the appropriate coordinate, get channel with descriptor in it
    for channel in coord_object.channels:
        if channel.channel_header["UserDescription"]==descriptor: 
            #Find the delta_t
            delta_t = float(fread_meta_data(coord_object,descriptor,"TimeStep"))
            #Read in data and copy to a new array, then delete channel.data to save memory
            channel.fread_data_volts()
            output = channel.data
            channel.data = None
            return output,delta_t
            
def fnormalize_I0_interpolate(PIN_array,PIN_time,BIM_array,BIM_time):
    """Function to normalize the PIN diode signal by BIM.
    In case the two traces are at different delta_t, use interpolation.
    Inputs:
    PIN_array, BIM_array: array of PIN diode (BIM) signal
    PIN_time, BIM_time: time array for PIN (BIM) signal
    
    """
    #Create an interpolation function
    BIM_function = scipy.interpolate.interp1d(BIM_time,BIM_array,bounds_error=False,fill_value=BIM_array[-1])
    print "Max PIN time = ",np.max(PIN_time),", Max BIM time = ", np.max(BIM_time)
    #Normalize
    return PIN_array / BIM_function(PIN_time)

def fnormalize_I_I0_colocated(PIN_array,BIM_array):
    '''Function to normalize the PIN diode signal by the BIM if they are both
    taken on the same scope simultaneously.  As such, no time shift or
    interpolation are needed.
    Inputs:
    PIN_array, BIM_array: arrays of pulse-averaged values from PIN and BIM.
    Output:
    Normalized PIN diode signal
    '''
    min_length = min(PIN_array.size,BIM_array.size)
    return PIN_array[:min_length]/BIM_array[:min_length]

def fcompute_time_array(pulse_time,data_size):
    """Computes the time arrays for a dataset.
    Returns tuple of numpy arrays (PIN_times,BIM_times).
    """
    return np.linspace(0,(data_size-1)*pulse_time,data_size)

def ffind_dark_current(dark_num,directory=None):
    """Function to find the average value of PIN and BIM from a scan.
    Used to compute the dark current from a dark scan, in particular.
    """
    #Use the default directory unless one is give
    if not directory:
        directory = file_path
    if not dark_num:
        return 0,0
    #Read in headers
    filename = fcreate_filename(dark_num)
    #Read in the header information
    try:
        header_data = rdg.fread_headers(directory+filename)
    except IOError:
        print "Problem reading file " + filename
        return
    mean_PIN = 0
    mean_BIM = 0
    #Loop through the coordinates measured
    for coordinate in header_data:
        mean_BIM += np.mean(coordinate.channels[I0_name].fread_data_volts(),dtype=np.float64)
        mean_PIN += np.mean(coordinate.channels[I_name].fread_data_volts(),dtype=np.float64)
        #Clear data so we don't soak up too much memory needlessly
        for channel in coordinate.channels:
            del(channel.data)
        print str(coordinate + 1) + " coordinates processed for dark current."
    return mean_PIN / float(len(header_data)), mean_BIM / float(len(header_data))

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
        
def fparse_dictionary_numeric_string(input_dict):
    '''Parses a dictionary into two lists of keys, one whose values cast to a
    numeric type, the other not (assumed to be just strings).  
    Also returns maximum length value length for the string keys.
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
            print "Value = " + str(value)
            raise TypeError
    #Find the maximum length of a string
    max_string_len = max([len(x) for x in string_output])
    return num_output,string_output,max_string_len

def fcreate_datasets_coord_header(hdf_group,coord_header,max_replicates,num_locations):
    '''Preallocates datasets for the variables saved in the coordinate headers.
    Inputs:
    hdf_file: h5py Group object to which to write the datasets
    coord_header: example readdatagrabber.py Coordinate.coordinate_header
    max_replicates: maximum number of times any location is replicated
    num_locations: number of locations
    '''
    #Create top_level datasets for any numeric values from the coordinate headers.  Use first coordinate as a guide.
    print coord_header
    numeric_header_keys, string_header_keys, max_string_len = fparse_dictionary_numeric_string(coord_header)
    dtype_string = 'S' + str(max_string_len)
    if replicated:
        for numeric_key in numeric_header_keys:
            hdf_group.create_dataset(numeric_key,shape=(num_locations,max_replicates),dtype=write_datatype)
        #Do the same for coordinate header values that are strings
        for string_key in string_header_keys:
            hdf_group.create_dataset(string_key,shape=(num_locations,max_replicates),dtype=dtype_string)
    else:
        for numeric_key in numeric_header_keys:
            hdf_group.create_dataset(numeric_key,shape=(num_locations,),dtype=write_datatype)
        #Do the same for coordinate header values that are strings
        for string_key in string_header_keys:
            hdf_group.create_dataset(string_key,shape=(num_locations,),dtype=dtype_string)
    return numeric_header_keys, string_header_keys

def fconvert_to_hdf5_multiprocess(file_num,directory=None,write_filename=None,peaked=True):
    """Function to fully process PIN data at the specified coordinate.
        Writes processed PIN data to an HDF5 file using multiprocessing.
        Each position is saved as one group.  Only coordinate (not channel) header
        data saved as attributes of dataset.
    """
    #Make the filename
    filename = fcreate_filename(file_num)
    print filename
    #Set the directory and write_filename to reasonable defaults if they aren't already
    if not directory:
        directory = file_path
    if not write_filename:
        write_filename = filename.split('.')[0] + '.hdf5'
    #Open an HDF5 file to save the data.
    with h5py.File(directory+write_filename,'w') as write_file:
        #Read in the header information
        try:
            headers = rdg.fread_headers(directory+filename)
        except IOError:
            print "Problem reading file " + filename
            return
        #Parse through these header data to find locations and number of replicates
        location_info_list = fparse_headers(headers) 
        #Find the maximum number of replicates
        max_replicates = max([x.fget_num_replicates() for x in location_info_list])
        global replicated
        replicated = max_replicates > 1
                    
        #Lets see how the calculations worked.
        for loc_info in location_info_list:
            print str(loc_info.location_values)
            print "Replicates = " + str(loc_info.fget_num_replicates())
            
        #Find dark currents: only do this once
        I_dark, I0_dark = ffind_dark_current(dark_file_num)
        print "Dark I = " + str(I_dark) + " V, Dark I0 = " + str(I0_dark) + " V."    
        #Process the first file to see how big the final data will be
        test_processed_data, pulse_duration_I = fread_normalize_coordinate_data(headers[0],I_name,I0_name,I_dark,I0_dark,peaked)
#         (binned_I,pulse_duration_I) = fread_bin_signal_by_bunch(headers[0],I_name)
#         (binned_I0,pulse_duration_I0) = fread_bin_signal_by_bunch(headers[0],I0_name)
#         if binned_I == None:
#             binned_I = np.zeros_like(binned_I0)
#             pulse_duration_I = pulse_duration_I0
#         else:
#             binned_I -= I_dark
#             binned_I0 -= I0_dark
#         test_processed_data = fnormalize_I_I0_colocated(binned_I,binned_I0)
        #Add a time step variable to the saved datasets
        location_info_list[0].header_coord_objects[0].coordinate_header['Time_Step'] = pulse_duration_I
        
        #Preallocate arrays in the HDF5 file from the channels
        fcreate_datasets_channels(write_file,{'Intensity':test_processed_data.shape[0]},max_replicates,len(location_info_list))
        
        #Preallocate arrays in the HDF5 from the coordinate headers
        numeric_keys, string_keys = fcreate_datasets_coord_header(write_file,
                                                location_info_list[0].header_coord_objects[0].coordinate_header,
                                                max_replicates,len(location_info_list))
        print "Finished writing empty datasets"     #Feedback to user
        
        #Loop through the LocationInfo objects
        for i,loc_info in enumerate(location_info_list):
            print "Working on position # " + str(i) + ", position " + str(loc_info.location_values)
            for (replicate_num,coordinate) in enumerate(loc_info.header_coord_objects):
                fprocess_coordinate(coordinate,write_file,I_dark,I0_dark,numeric_keys,
                            string_keys,i,replicate_num,peaked=peaked)
                #Clear out entries in headers so we don't overload memory
                coordinate = ""
            loc_info = ""

def fprocess_coordinate(coordinate,hdf_group,I_dark,I0_dark,numeric_keys,string_keys,location_num,replicate,signal_name="Intensity",peaked=True):
    '''Processes one coordinate of the DataGrabber file.
    '''
    print "Replicate # " + str(replicate)
    #Loop through the header, writing numbers to the appropriate datasets
#     for numeric_key in numeric_keys:
#         if numeric_key in coordinate.coordinate_header:
#             hdf_group[numeric_key][location_num,replicate] = float(coordinate.coordinate_header[numeric_key])
#     #Write string datasets for string header data
#     for string_key in string_keys:
#         if string_key in coordinate.coordinate_header:
#             hdf_group[string_key][location_num,replicate] = str(coordinate.coordinate_header[string_key])
    #Find the binned data
    final_data, pulse_duration = fread_normalize_coordinate_data(coordinate,I_name,I0_name,I_dark,I0_dark,peaked)
#     (binned_I,pulse_duration_I) = fread_bin_signal_by_bunch(coordinate,I_name)
#     (binned_I0,pulse_duration_I0) = fread_bin_signal_by_bunch(coordinate,I0_name)
#     #Check if we didn't have a good signal for I
#     if binned_I == None:
#         binned_I = np.zeros_like(binned_I0)
#         pulse_duration_I = pulse_duration_I0
#     else:
#         binned_I -= I_dark
#         binned_I0 -= I0_dark
#     final_data = fnormalize_I_I0_colocated(binned_I,binned_I0)
    max_length = hdf_group[signal_name].shape[1]
    if final_data.shape[0] < max_length:
        max_length = final_data.shape[0]
    if replicated:
        hdf_group[signal_name][location_num,:max_length,replicate] = final_data[:max_length]
        hdf_group['Time_Step'][location_num,replicate] = pulse_duration
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
        hdf_group['Time_Step'][location_num] = pulse_duration
        #Write numeric values from headers
        for numeric_key in numeric_keys:
            if numeric_key in coordinate.coordinate_header:
                hdf_group[numeric_key][location_num] = float(coordinate.coordinate_header[numeric_key])
        #Write string datasets for string header data
        for string_key in string_keys:
            if string_key in coordinate.coordinate_header:
                hdf_group[string_key][location_num] = str(coordinate.coordinate_header[string_key])

def fread_normalize_coordinate_data(coordinate,I_name,I0_name,I_dark=0,I0_dark=0,peaked=True):
    '''Reads the data for a coordinate and normalizes I by I0.
    '''
    if peaked:
        (binned_I,pulse_duration_I) = fread_bin_signal_by_bunch(coordinate,I_name)
        (binned_I0,pulse_duration_I0) = fread_bin_signal_by_bunch(coordinate,I0_name)
    else:
        (binned_I,pulse_duration_I) = fread_signal_direct(coordinate,I_name)
        (binned_I0,pulse_duration_I0) = fread_signal_direct(coordinate,I0_name)
    if binned_I == None:
        return np.zeros_like(binned_I0), pulse_duration_I0
    else:
        binned_I -= I_dark
        binned_I0 -= I0_dark
    return fnormalize_I_I0_colocated(binned_I,binned_I0),pulse_duration_I

def fbatch_conversion_multiprocess(file_nums,peaked=True):
    '''Class to batch process a large number of DataGrabber files.
    Should work with strings or numbers given in file_nums list.
    '''
    #Make a JoinableQueue to hold tasks
    tasks = multiprocessing.JoinableQueue()
    #Set up processes
    processes = [MP_Process(tasks) for i in range(num_processors)]
    for process in processes:
        process.start()
    #Set up multiprocessing JoinableQueue
    for file_num in file_nums:
        tasks.put(MP_Task(file_num,peaked))
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
    def __init__(self,file_num,peaked):
        self.file_num = file_num
        self.peaked = peaked
    
    def __call__(self):
        fconvert_to_hdf5_multiprocess(self.file_num,peaked=self.peaked)

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