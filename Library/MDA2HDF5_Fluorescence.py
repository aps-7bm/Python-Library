""" Module to convert MDA files saved by sscan to HDF5.
Follows format specified at http://www.aps.anl.gov/bcda/synApps/sscan/saveData_fileFormat.txt
Alan Kastengren, XSD
Started February 24, 2013

Change log
April 5, 2013: change saving of positioner and detector attributes to use names when possible.
April 5, 2013: first steps at saving extra PVs.  Handles several EPICS datatypes, but not well tested yet.
April 23, 2013: Fork for saving fluorescence data from ISU 2012-3 experiments.
May 7, 2014: Minor changes to HDF file handling (using context).  
May 7, 2014: Change meta data reading functions so they always give a name.  Sim;lify other code based on this.
May 7, 2014: Save only the positioner and detector arrays up to current point.
June 18, 2014: Minor change to check the existence of the input file.
March 5, 2015: Add function frun_append to append MDA file data to an existing HDF file.
"""
#
#Imports
import h5py
import xdrlib
import numpy as np
import os.path
mca_saving = False
import logging

#Set up logging
logger = logging.getLogger('m2h')
logger.addHandler(logging.NullHandler())
#
def fparse_counted_string(data):
    '''Parses the data to get a counted string,
        which is not really in the XDR standard, but
        is used in the MDA file format.
    '''
    string_length = data.unpack_int()
    if string_length > 0:
        return data.unpack_string()
    else:
        return ""

def fstart_unpacking(input_filename,directory):
    '''Takes the input filename and converts into
    a buffer suitable for the xdrlib to get started.
    '''
    logger.info('Starting work on file ' + directory + input_filename)
    try:
        with open(directory+input_filename) as input_file:
            f = input_file.read()
            return xdrlib.Unpacker(f)
    except IOError:
        logger.error("Could not open input file: " + directory+input_filename)
        raise IOError
    
def fread_file_header(data,output_hdf):
    '''Reads the file header and adds to the attributes
    of the top level of the hdf file.
    '''
    output_hdf.attrs["Version Number"] = data.unpack_float()
    output_hdf.attrs["Scan Number"] = data.unpack_int()
    rank = data.unpack_int()
    output_hdf.attrs["Data Rank"] = rank  
    dimensions = data.unpack_farray(rank,data.unpack_int)
    is_regular = data.unpack_int()
    if is_regular:
        output_hdf.attrs["Regular_file"] = "True"
    else:
        output_hdf.attrs["Regular_file"] = "False"
    #Return the pointer to the extra PVs.  We will add them last
    #so we don't lose our place.
    return data.unpack_int()

def fread_scan(data,hdf_group):
    '''Reads through a scan, putting data vectors into datasets and all meta
    data into attributes.  Will act recursively for multi-dimensional datasets.
    '''
    #Read through scan header.
    rank = data.unpack_int()
    hdf_group.attrs["Rank"] = rank
    requested_num_points = data.unpack_int()
    current_point = data.unpack_int()
    #Save a list of the pointers for lower scans
    lower_scans_pointers = []
    if rank>1:
        for i in range(requested_num_points):
            lower_scans_pointers.append(data.unpack_int())
        #Only save up to the current_point, as others are garbage from abort
        lower_scans_pointers = lower_scans_pointers[:current_point]
    #
    #Read through the info fields
    hdf_group.attrs["Scan Name"] = fparse_counted_string(data)
    hdf_group.attrs["Time Stamp"] = fparse_counted_string(data)
    #Find the number of positioners, detectors, and triggers
    num_positioners = data.unpack_int()
    num_detectors = data.unpack_int()
    num_triggers = data.unpack_int()
    #
    #Fill up lists of dictionaries of meta data for positioners,
    #detectors, and triggers.
    positioner_meta = []
    detector_meta = []
    trigger_meta = []
    for i in range(num_positioners):
        positioner_meta.append(fread_positioner(data,i))
    for i in range(num_detectors):
        detector_meta.append(fread_detector(data,i))
    for i in range(num_triggers):
        trigger_meta.append(fread_trigger(data))
    #
    #Put these meta data as attributes of the group.
    for entry in positioner_meta:
        for key,value in entry.iteritems():
            if key != "Name":
                hdf_group.attrs[entry['Name']+"_"+str(key)] = value
    for entry in detector_meta:
        for key,value in entry.iteritems():
            if key != "Name":
                hdf_group.attrs[entry['Name']+"_"+str(key)] = value
    for i in range(num_triggers):
        for key,value in trigger_meta[i].iteritems():
            hdf_group.attrs["Trigger_"+str(i)+"_"+str(key)] = value
    #
    #Read in the actual data as datasets
    positioner_values = []
    for i in range(num_positioners):
        #Read in the data
        positioner_array = data.unpack_farray(requested_num_points,data.unpack_double)
        #Add to file and save the values
        #If there are no current points, don't write a dataset
        if current_point:
            hdf_group.create_dataset(positioner_meta[i]['Name'],data=positioner_array[:current_point])
        #Append positioner values, but only up to current point.  Others are just garbage.
        positioner_values.append(positioner_array[:current_point])
        
    #Read in detector data and write to file
    for i in range(num_detectors):
        #Same idea as positioners: give an intelligible name if we have one
        detector_array = data.unpack_farray(requested_num_points,data.unpack_float)
        #If there are no current points, don't write a dataset
        if current_point and detector_meta[i]['Name'] not in hdf_group.keys():
            logger.info('Detector name: ' + detector_meta[i]['Name'])
            hdf_group.create_dataset(detector_meta[i]['Name'],data=detector_array[:current_point])
    #
    #If this was the lowest rank, return now
    if len(lower_scans_pointers) == 0:
        return
    #Start looping through the lower dimensional scans.  Again, this
    #will end up being recursive for a multi-dimensional scan.
    #First case: want to treat lowest scan as an MCA and we're at the second to lowest scan.
    if mca_saving and hdf_group.attrs["Rank"] == 2:
        #Since we are at the lowest level, only look up the current_point
        for j in range(len(lower_scans_pointers)):
            data.set_position(lower_scans_pointers[j])
            #Call subroutine that just returns detector names and arrays of their values
            (detector_names,detector_arrays) = fread_MCA_scan(data)
            #If this is the first point, we must set up appropriate datasets in the hdf group.
            #If not, just add as appropriate columns in the existing datasets.
            for i in range(len(detector_names)):
                for name in hdf_group.keys():
                    if name == detector_names[i]:
                        break
                else:
                    hdf_group.create_dataset(detector_names[i],data=np.zeros((current_point,len(detector_arrays[i]))))
                #Add data, but only if sizes match
                if np.size(detector_arrays[i])==np.size(hdf_group[detector_names[i]][j,...]):
                    hdf_group[detector_names[i]][j,...] = detector_arrays[i]
    #All other cases: recurse.
    else:
        for j in range(len(lower_scans_pointers)):
            #Create a new subgroup
            #Give the subgroup an intelligible name if there is only one positioner.
            if num_positioners == 1 and positioner_meta[0]['Name']:
                name = positioner_meta[0]['Name'] + "=" + str(positioner_values[0][j])
            else:
                name = "Rank_"+str(rank)+"_Point_"+str(j)
            try:
                subgroup = hdf_group.create_group(name)
                print name
            except ValueError:
                logger.error("Problem making group " + str(name) + " for lower scan #" + str(j))
                raise ValueError 
            data.set_position(lower_scans_pointers[j])
            fread_scan(data,subgroup)

    return

def fread_MCA_scan(data):
    '''Reads an MCA scan.
    
    Performs similar functionality to fread_scan, but assumes that we only care
    about the array data, not the positioners and detectors.  This is a good
    assumption for MCA data (e.g., fluorescence spectra, for example).
    '''
    #Read through scan header.
    #Make sure we really are at the lowest level of the file.
    rank = data.unpack_int()
    assert rank == 1
    requested_num_points = data.unpack_int()
    current_point = data.unpack_int()
    #Read through the info fields.  Variables not used.
    scan_name = fparse_counted_string(data)
    time_stamp = fparse_counted_string(data)
    #Find the number of positioners, detectors, and triggers
    num_positioners = data.unpack_int()
    num_detectors = data.unpack_int()
    num_triggers = data.unpack_int()
    #
    #Fill up lists of dictionaries of meta data for detectors only.
    #Just skip past the positioner and trigger meta data, as we don't care.
    detector_meta = []
    detector_names = []
    detector_arrays = []
    for i in range(num_positioners):
        fread_positioner(data,i)
    for i in range(num_detectors):
        detector_meta.append(fread_detector(data,i))
    for i in range(num_triggers):
        fread_trigger(data)
    #
    #Read past the positioner data, as it isn't really meaningful for an MCA
    for i in range(num_positioners):
        data.unpack_farray(requested_num_points,data.unpack_double)
    #Read in the detector names and arrays
    for i in range(num_detectors):
        detector_names.append(detector_meta[i]["Name"])
        logger.info('MCA Detector: ' + detector_meta[i]["Name"])
        detector_arrays.append(data.unpack_farray(requested_num_points,data.unpack_float)[:current_point])
    return (detector_names, detector_arrays)
    
def fread_positioner(data,i):
    '''Reads all of the meta data for a given positioner.
    Returns a dictionary of meta data.
    '''
    meta_data = {}
#    meta_data['Number'] = data.unpack_int()
    dummy = data.unpack_int()
    #Make sure that we have a valid name
    meta_data['Name'] = fparse_counted_string(data)
    if not meta_data['Name']:
        meta_data['Name'] = 'Positioner_'+str(i)
    meta_data['Desc'] = fparse_counted_string(data)    
    meta_data['Step_Mode'] = fparse_counted_string(data) 
    meta_data['Unit'] = fparse_counted_string(data) 
    meta_data['RBV_Name'] = fparse_counted_string(data) 
    meta_data['RBV_Desc'] = fparse_counted_string(data) 
    meta_data['RBV Unit'] = fparse_counted_string(data)
    return meta_data

def fread_detector(data,i):
    '''Reads all of the meta data for a given detector.
    Returns a dictionary of meta data.
    '''
    meta_data = {}
#    meta_data['Number'] = data.unpack_int()
    dummy = data.unpack_int()
    meta_data['Name'] = fparse_counted_string(data)
    if not meta_data['Name']:
        meta_data['Name'] = 'Detector_'+str(i)
    meta_data['Desc'] = fparse_counted_string(data)    
    meta_data['Unit'] = fparse_counted_string(data) 
    return meta_data

def fread_trigger(data):
    '''Reads all of the meta data for a given trigger.
    Returns a dictionary of meta data.
    '''
    meta_data = {}
#    meta_data['Number'] = data.unpack_int()
    dummy = data.unpack_int()
    meta_data['Name'] = fparse_counted_string(data)
    meta_data['Command'] = data.unpack_float()
    return meta_data

def fread_extra_PVs(data,group):
    '''Reads in the extra PVs and puts them as attributes of the group.
    This function ihandles the most important EPICS datatypes,
    not every one.  Types other than CTRL_DOUBLE aren't tested.
    As a test, put in a new group called "Extra PVs"
    '''
    extra_PV_group = group.create_group("Extra PVs")
    num_extra_PVs = data.unpack_int()
    for i in range(num_extra_PVs):
        pv_name = fparse_counted_string(data)
        pv_desc = fparse_counted_string(data)
        pv_type = data.unpack_int()
        #Parse out the type and deal with it accordingly.  Only partially done here.
        #Only deal with types Dohn Arms deals with in mdautils, as of April 2013.  
        #Not tested on anything other than type 34 (double array).
        pv_unit = None
        pv_value = None
        if pv_type == 0:    #STRING
            pv_value = fparse_counted_string(data)
        elif pv_type == 29:       #CTRL_SHORT
            length = data.unpack_int()
            pv_unit = fparse_counted_string(data)
            pv_value = data.unpack_farray(length,data.unpack_int)
        elif pv_type == 30:       #CTRL_FLOAT
            length = data.unpack_int()
            pv_unit = fparse_counted_string(data)
            pv_value = data.unpack_farray(length,data.unpack_float)
        elif pv_type == 32:       #CTRL_CHAR
            length = data.unpack_int()
            pv_unit = fparse_counted_string(data)
            pv_value = data.unpack_farray(length,data.unpack_int)
        elif pv_type == 33:       #CTRL_LONG
            length = data.unpack_int()
            pv_unit = fparse_counted_string(data)
            pv_value = data.unpack_farray(length,data.unpack_int)
        elif pv_type == 34:      #CTRL_DOUBLE
            length = data.unpack_int()
            pv_unit = fparse_counted_string(data)
            pv_value = data.unpack_farray(length,data.unpack_double)
        #Now that we've parsed it, record in the HDF file as attributes of the group.
        prefix = "Extra_PV_" + str(i)
        if pv_name:     #Mark everything by PV name if it exists
            prefix = pv_name
        if pv_desc:
            extra_PV_group.attrs[prefix + "_Description"] = pv_desc
        if pv_unit:
            extra_PV_group.attrs[prefix + "_Unit"] = pv_unit
        if pv_value:
            extra_PV_group.attrs[prefix + "_Value"] = pv_value
    return
    
def frun_main(input_file="7bmb1_0933.mda",directory="/data/Data/SprayData/Cycle_2012_3/Time_Averaged_ISU/",mca_saving_flag=False):
    '''Function that tests the working of the code.
    '''
    #Parse the input file into the main part of the file name and the extension.
    global mca_saving
    mca_saving = mca_saving_flag
    output_filename = input_file.split(".")[0] + ".hdf5"
    #Check whether the file exists
    if not os.path.isfile(directory+input_file):
        logger.error("File " + input_file + " cannot be opened.  Exiting.")
        return
    try:
        with h5py.File(directory+output_filename,'w') as output_hdf:
            data = fstart_unpacking(input_file,directory)
            extra_PV_position = fread_file_header(data,output_hdf)
            fread_scan(data,output_hdf)
            data.set_position(extra_PV_position)
            if extra_PV_position:
                fread_extra_PVs(data,output_hdf)
    except IOError:
        logger.error("IOError: Output file could not be opened.")
        return
    except EOFError:
        logger.error("Unexpectedly reached end of file.  File may be damaged.")
        
def frun_append(input_MDA = '7bmb1_0260.mda',MDA_directory = '/data/Data/SprayData/Cycle_2015_1/Spark/MDA_Files/',
                output_HDF = 'Scan_260.hdf5',HDF_directory = '/data/Data/SprayData/Cycle_2015_1/Radiography/',
                mca_saving_flag = True):
    '''Parses data from the MDA file, adding datasets to an existing HDF5 file.
    '''
     #Parse the input file into the main part of the file name and the extension.
    global mca_saving
    mca_saving = mca_saving_flag
    #Check whether the file exists
    if not os.path.isfile(MDA_directory+input_MDA):
        logger.error("File " + input_MDA + " cannot be opened.  Exiting.")
        return
    try:
        with h5py.File(HDF_directory+output_HDF,'r+') as output_hdf:
            data = fstart_unpacking(input_MDA,MDA_directory)
            extra_PV_position = fread_file_header(data,output_hdf)
            fread_scan(data,output_hdf)
            data.set_position(extra_PV_position)
            if extra_PV_position:
                fread_extra_PVs(data,output_hdf)
    except IOError:
        logger.error("IOError: Output file could not be opened.")
        return
    except EOFError:
        logger.error("Unexpectedly reached end of file.  File may be damaged.")
        
if __name__ =="__main__":
    import sys
    import os
    print "In code"
    print sys.argv
    if len(sys.argv) == 1:
        frun_main()
    elif len(sys.argv) == 2:
        if sys.argv[1] == "-h":
            print """Program to convert sscan MDA files to HDF5. /n
                If one argument is given, assume it is a path to an MDA file. /n
                The HDF5 file will be placed in the same directory.
                Alternatively, give file and a directory in which to place HDF file.
                """
        else:
            frun_main(sys.argv[1],os.getcwd()+'/')
    elif len(sys.argv) == 3 and os.path.isdir(sys.argv[2]):
        frun_main(sys.argv[1],sys.argv[2])
    elif len(sys.argv) == 4 and os.path.isdir(sys.argv[2]):
        frun_main(sys.argv[1],sys.argv[2],sys.argv[3])
    