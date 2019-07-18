'''Script to read in DataGrabber files and bin the data by a given time step.

Alan Kastengren, XSD, APS

Started: August 25, 2015
'''
import numpy as np
import readdatagrabber as rdg
import ArrayBin_Cython as ArrayBin
import h5py
import ALK_Utilities as ALK
import multiprocessing

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

def fcreate_filename(file_num):
    format_string = '{0:0'+str(digits)+'d}'
    return prefix+format_string.format(file_num)+suffix

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
        
def fcompute_time_array(pulse_time,data_size):
    """Computes the time arrays for a dataset.
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
        mean_BIM += np.mean(coordinate.channels[I0_name].fread_data_volts())
        mean_PIN += np.mean(coordinate.channels[I_name].fread_data_volts())
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
            print "Value = " + str(value)
            raise TypeError
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
    for numeric_key in numeric_header_keys:
        hdf_group.create_dataset(numeric_key,shape=(num_locations,max_replicates),dtype=write_datatype)
    #Do the same for coordinate header values that are strings
    dtype_string = 'S' + str(max_string_len)
    for string_key in string_header_keys:
        hdf_group.create_dataset(string_key,shape=(num_locations,max_replicates),dtype=dtype_string)
    return numeric_header_keys, string_header_keys

def fread_bin_signal_by_time(coord_object,delta_t,descriptor="PINDiode"):
    """Function to read in PIN data, find peak positions, and bin each peak.
    Inputs:
    coord_object: readdatagrabber.py DataGrabberCoordinate object to be processed
    delta_t: desired time spacing in the output data
    descriptor: UserDescription value for the channel we want
    Outputs:
    binned_output: binned signal 
    """   
    #Pick the appropriate coordinate, get channel with descriptor in it
    for channel in coord_object.channels:
        if channel.channel_header["UserDescription"]==descriptor: 
            #Read in data on this channel
            channel.fread_data_volts()
            #Find the time step for the data
            data_delta_t = float(fread_meta_data(coord_object,descriptor,"TimeStep"))
            print data_delta_t
            print delta_t
            #Find the number of time points for the new data
            binned_output = ArrayBin.fbin_array(channel.data,delta_t/data_delta_t)
            #Delete the array data from the channel to avoid maxing out memory
            channel.data = None
            return binned_output

def fconvert_to_hdf5_multiprocess(file_num,delta_t=3.6825e-6,channel_names=['PINDiode','BIM'],directory=None,write_filename=None):
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
        replicate_list = []
        for loc in location_info_list:
            replicate_list.append(loc.replicates)
        max_replicates = max(replicate_list)
        if max_replicates == 1:
            replicated = False
        else:
            replicated = True
                    
        #Lets see how the calculations worked.
        for loc_info in location_info_list:
            print str(loc_info.location_values)
            print "Replicates = " + str(loc_info.replicates)
            
        #Process the first file to see how big the final data will be
        channel_data = {}
        channel_sizes = {}
        for chan_name in channel_names:
            channel_data[chan_name] = fread_bin_signal_by_time(headers[0],delta_t,chan_name)
            channel_sizes[chan_name] = channel_data[chan_name].shape[0]
        #Add a time step variable to the saved datasets
        location_info_list[0].header_coord_objects[0].coordinate_header['Final_Time_Step'] = delta_t
        
        #Preallocate arrays in the HDF5 file from the channels
        fcreate_datasets_channels(write_file,channel_sizes,max_replicates,len(location_info_list))
        
        #Preallocate arrays in the HDF5 from the coordinate headers
        numeric_keys, string_keys = fcreate_datasets_coord_header(write_file,
                                                location_info_list[0].header_coord_objects[0].coordinate_header,
                                                max_replicates,len(location_info_list))
        print "Finished writing empty datasets"     #Feedback to user
        
        #Loop through the LocationInfo objects
        for i,loc_info in enumerate(location_info_list):
            print "Working on position # " + str(i) + ", position " + str(loc_info.location_values)
            for (replicate_num,coordinate) in enumerate(loc_info.header_coord_objects):
                print "Replicated = " + str(replicated)
                fprocess_coordinate_bin_by_time(coordinate,write_file,channel_names,delta_t,numeric_keys,
                            string_keys,i,replicate_num,replicated)
                
                #Clear out entries in headers so we don't overload memory
                coordinate = ""
            loc_info = ""

def fprocess_coordinate_bin_by_time(coordinate,hdf_group,channel_names,delta_t,numeric_keys,string_keys,location_num,replicate,replicated):
    '''Processes one coordinate of the DataGrabber file.
    '''
    print "Replicate # " + str(replicate)
    #Loop through the header, writing numbers to the appropriate datasets
    for numeric_key in numeric_keys:
        if numeric_key in coordinate.coordinate_header:
            hdf_group[numeric_key][location_num,replicate] = float(coordinate.coordinate_header[numeric_key])
    #Write string datasets for string header data
    for string_key in string_keys:
        if string_key in coordinate.coordinate_header:
            hdf_group[string_key][location_num,replicate] = str(coordinate.coordinate_header[string_key])
    #Find the binned data
    for chan_name in channel_names:
        channel_data = fread_bin_signal_by_time(coordinate,delta_t,chan_name)
        max_length = hdf_group[chan_name].shape[1]
    
        if channel_data.shape[0] < max_length:
            max_length = channel_data.shape[0]
        if replicated:
            hdf_group[chan_name][location_num,:max_length,replicate] = channel_data[:max_length]
            hdf_group['Final_Time_Step'][location_num,replicate] = delta_t
        else:
            hdf_group[chan_name][location_num,:max_length] = channel_data[:max_length]
            hdf_group['Final_Time_Step'][location_num] = delta_t

def fbatch_conversion_multiprocess(file_nums,delta_t):
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
        tasks.put(MP_Task(file_num,delta_t))
    #Add poison pills at the end of the queue so MP_Process objects know when to stop.
    for i in range(num_processors):
        tasks.put(None)
    #Wait for tasks to finish
    tasks.join()       

class MP_Task():
    '''Object to allow for file conversion to occur in a JoinableQueue
    '''
    def __init__(self,file_num,delta_t):
        self.file_num = file_num
        self.delta_t = delta_t
    
    def __call__(self):
        fconvert_to_hdf5_multiprocess(self.file_num,self.delta_t)
    

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