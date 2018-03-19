""" Module to convert MDA files saved by sscan to HDF5.
Follows format specified at http://www.aps.anl.gov/bcda/synApps/sscan/saveData_fileFormat.txt
Alan Kastengren, XSD
Started February 24, 2013

Change log
April 5, 2013: change saving of positioner and detector attributes to use names when possible.
April 5, 2013: first steps at saving extra PVs.  Handles several EPICS datatypes, but not well tested yet.
April 23, 2013: Fork for saving fluorescence data from ISU 2012-3 experiments.
May 7, 2014: Minor changes to HDF file handling (using context).  
May 7, 2014: Change meta data reading functions so they always give a name.  Simplify other code based on this.
May 7, 2014: Save only the positioner and detector arrays up to current point.
June 18, 2014: Minor change to check the existence of the input file.
March 5, 2015: Add function frun_append to append MDA file data to an existing HDF file.
September 6, 2017: rewrite to make multidimensional arrays.
February 23, 2018: refactor to better handle multidimensional scans, Python 3.  The code
	now saves all datasets as multidimensional datasets under the main file, rather than
	in nested groups.  This should be much more convenient to use.
"""
#
# Imports
import h5py
import xdrlib
import numpy as np
import os.path
import logging

# Set up logging
logger = logging.getLogger('m2h')
logger.addHandler(logging.NullHandler())


def fparse_counted_string(data):
    '''Parses the data to get a counted string,
        which is not really in the XDR standard, but
        is used in the MDA file format.
    '''
    string_length = data.unpack_int()
    if string_length > 0:
        return data.unpack_string().decode()
    else:
        return ""

def fstart_unpacking(input_filename, directory):
    '''Takes the input filename and converts into
    a buffer suitable for the xdrlib to get started.
    '''
    logger.info('Starting work on file ' + directory + input_filename)
    try:
        with open(directory + input_filename, 'rb') as input_file:
            f = input_file.read()
            return xdrlib.Unpacker(f)
    except IOError:
        logger.error("Could not open input file: " + directory + input_filename)
        raise IOError

def fread_file_header(data, output_hdf):
    '''Reads the file header and adds to the attributes
    of the top level of the hdf file.
    '''
    output_hdf.attrs["Version Number"] = float(data.unpack_float())
    output_hdf.attrs["Scan Number"] = int(data.unpack_int())
    rank = data.unpack_int()
    output_hdf.attrs["Data Rank"] = rank
    dimensions = data.unpack_farray(rank, data.unpack_int)
    is_regular = data.unpack_int()
    if is_regular:
        output_hdf.attrs["Regular_file"] = "True"
    else:
        output_hdf.attrs["Regular_file"] = "False"
    # Return the pointer to the extra PVs.  We will add them last
    # so we don't lose our place.
    return data.unpack_int()

def fread_scan(data, hdf_file, higher_dim_position=[], higher_dim_size=[]):
    '''Reads through a scan, putting data vectors into datasets and all meta
    data into attributes.  Will act recursively for multi-dimensional datasets.
    Imputs:
    data: xdrlib pointer to the data file.
    hdf_file: h5py handle to the HDF file we're writing to.
    higher_dim_position: list of the point we're at in higher dimensions
    higher_dim_size: list of the sizes of the higher dimensions above this scan, if any.
    '''
    # Read through scan header.
    rank = data.unpack_int()
    requested_num_points = data.unpack_int()
    current_point = data.unpack_int()
    # Save a list of the pointers for lower scans
    lower_scans_pointers = []
    if rank > 1:
        for i in range(requested_num_points):
            lower_scans_pointers.append(data.unpack_int())
        # Only save up to the current_point, as others are garbage from aborted scan
        lower_scans_pointers = lower_scans_pointers[:current_point]
    # Read the name of the scan record used
    hdf_file.attrs["Scan_Name_Rank_{0:d}".format(rank)] = str(fparse_counted_string(data))
    # Read the starting time stamp, if this is the first scan, and record it
    time_stamp = fparse_counted_string(data)
    if higher_dim_size == []:
        hdf_file.attrs["Time Stamp"] = time_stamp

    # Find the number of positioners, detectors, and triggers
    num_positioners = data.unpack_int()
    num_detectors = data.unpack_int()
    num_triggers = data.unpack_int()

    # Fill up lists of dictionaries of meta data for positioners,
    # detectors, and triggers.
    positioner_meta = []
    detector_meta = []
    trigger_meta = []
    for i in range(num_positioners):
        positioner_meta.append(fread_positioner(data, i))
    for i in range(num_detectors):
        detector_meta.append(fread_detector(data, i))
    for i in range(num_triggers):
        trigger_meta.append(fread_trigger(data))

    # Put these meta data as attributes of the HDF5 file, but only if
    # this is the first time we're at this dimensional level.
    if sum(higher_dim_position) == 0:
        # Handle the positioners
        for entry in (positioner_meta + detector_meta):
            dataset_name = entry['Name']
            # Get rid of the old dataset if it's there.
            # This occurs if the dataset is repeated from a higher dimension
            if dataset_name in hdf_file.keys():
                del hdf_file[dataset_name]
            # Make a new dataset
            dimensions = higher_dim_size + [requested_num_points]
            hdf_file.create_dataset(dataset_name, data=np.zeros(dimensions))
            logger.info('Dataset name: {0:s}, rank {1:d}'.format(dataset_name,rank))
            print('Dataset name: {0:s}, rank {1:d}'.format(dataset_name, rank))
            # Put in the attributes
            for key, value in entry.items():
                if key != "Name":
                    hdf_file[dataset_name].attrs[key] = value
        for i in range(num_triggers):
            for key, value in trigger_meta[i].items():
                hdf_file.attrs["Trigger_{0:d}_{1:s}_Rank{2:d}".format(i,key,rank)] = value

    # Read in the positioner data
    for i in range(num_positioners):
        positioner_array = data.unpack_farray(requested_num_points, data.unpack_double)
        pos_name = positioner_meta[i]['Name']
        logger.info("Processing positioner {0:d} of {1:d}: {2:s}".format(i+1, num_positioners, pos_name))
        fwrite_array_data(hdf_file, pos_name, higher_dim_position, positioner_array, current_point)

    # Read in the detector data
    for i in range(num_detectors):
        # Read in the data
        det_name = detector_meta[i]['Name']
        logger.info("Processing detector {0:d} of {1:d}: {2:s}".format(i+1, num_detectors, det_name))
        detector_array = data.unpack_farray(requested_num_points, data.unpack_float)
        fwrite_array_data(hdf_file, det_name, higher_dim_position, detector_array, current_point)

    # If this was the lowest rank, return now
    if len(lower_scans_pointers) == 0:
        return
    # Start looping through the lower dimensional scans.
    # This is recursive for scans with rank > 2.
    for j in range(len(lower_scans_pointers)):
        higher_dims_pos = higher_dim_position + [j]
        higher_dims_size = higher_dim_size + [requested_num_points]
        data.set_position(lower_scans_pointers[j])
        fread_scan(data, hdf_file, higher_dims_pos, higher_dims_size)
    return

def fread_positioner(data, i):
    '''Reads all of the meta data for a given positioner.
    Returns a dictionary of meta data.
    '''
    meta_data = {}
    _ = data.unpack_int()
    # Make sure that we have a valid name
    meta_data['Name'] = fparse_counted_string(data)
    if not meta_data['Name']:
        meta_data['Name'] = 'Positioner_' + str(i)
    for key in ['Desc','Step_Mode','Unit','RBV_Name','RBV_Desc','RBV_Unit']:
        meta_data[key] = fparse_counted_string(data)
    return meta_data

def fread_detector(data, i):
    '''Reads all of the meta data for a given detector.
    Returns a dictionary of meta data.
    '''
    meta_data = {}
    _ = data.unpack_int()
    meta_data['Name'] = fparse_counted_string(data)
    if not meta_data['Name']:
        meta_data['Name'] = 'Detector_' + str(i)
    meta_data['Desc'] = fparse_counted_string(data)
    meta_data['Unit'] = fparse_counted_string(data)
    return meta_data

def fread_trigger(data):
    '''Reads all of the meta data for a given trigger.
    Returns a dictionary of meta data.
    '''
    meta_data = {}
    _ = data.unpack_int()
    meta_data['Name'] = fparse_counted_string(data)
    meta_data['Command'] = data.unpack_float()
    return meta_data

def fwrite_array_data(hdf_file, dset_name, hdp, data_array, current_point):
    '''Write the positioner and detector array data to the HDF5 dataset.
    hdp = higher_dim_position
    '''
    if len(hdp) == 0:  # Top level
        hdf_file[dset_name][:current_point] = data_array[:current_point]
    elif len(hdp) == 1:
        hdf_file[dset_name][hdp[0],:current_point] = data_array[:current_point]
    elif len(hdp) == 2:
        hdf_file[dset_name][hdp[0],hdp[1],:current_point] = data_array[:current_point]
    elif len(hdp) == 3:
        hdf_file[dset_name][hdp[0],hdp[1],hdp[2],:current_point] = data_array[:current_point]

def fread_extra_PVs(data, group):
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
        pv_unit = None
        # Parse out the type and deal with it accordingly.  Only partially done here.
        # Only deal with types Dohn Arms deals with in mdautils, as of April 2013.
        # Not tested on anything other than type 34 (double array).
        if pv_type == 0:  # STRING
            pv_value = fparse_counted_string(data)
        elif pv_type == 29:  # CTRL_SHORT
            length = data.unpack_int()
            pv_unit = fparse_counted_string(data)
            pv_value = data.unpack_farray(length, data.unpack_int)
        elif pv_type == 30:  # CTRL_FLOAT
            length = data.unpack_int()
            pv_unit = fparse_counted_string(data)
            pv_value = data.unpack_farray(length, data.unpack_float)
        elif pv_type == 32:  # CTRL_CHAR
            length = data.unpack_int()
            pv_unit = fparse_counted_string(data)
            pv_value = data.unpack_farray(length, data.unpack_int)
        elif pv_type == 33:  # CTRL_LONG
            length = data.unpack_int()
            pv_unit = fparse_counted_string(data)
            pv_value = data.unpack_farray(length, data.unpack_int)
        elif pv_type == 34:  # CTRL_DOUBLE
            length = data.unpack_int()
            pv_unit = fparse_counted_string(data)
            pv_value = data.unpack_farray(length, data.unpack_double)
        else:
            logger.error('Unsupported datatype encountered, type {0:d}.  Exiting.'.format(pv_type))
            raise ValueError
        # Now that we've parsed it, record in the HDF file as attributes of the group.
        prefix = "Extra_PV_" + str(i)
        if pv_name:  # Mark everything by PV name if it exists
            prefix = pv_name
        if pv_desc:
            extra_PV_group.attrs[prefix + "_Description"] = pv_desc
        if pv_unit:
            extra_PV_group.attrs[prefix + "_Unit"] = pv_unit
        if pv_value:
            extra_PV_group.attrs[prefix + "_Value"] = pv_value
    return

def fcheck_directories(input_MDA, MDA_directory, HDF_directory):
    '''Checks that we have valid files for readin and writing.
    '''
    # If the MDA directory isn't specified, assume it's the current directory.
    if not MDA_directory:
        logger.info('No MDA_directory given.  Setting to current working directory.')
        MDA_directory = os.path.getcwd()
    # If there is no HDF_directory, make it the same as the MDA_directory.
    if not HDF_directory:
        logger.info('Setting the HDF file directory to ' + MDA_directory)
        HDF_directory = MDA_directory
    # Check whether the file exists
    if not os.path.isfile(MDA_directory + input_MDA):
        logger.error("File " + input_MDA + " cannot be opened.  Exiting.")
        raise IOError
    return(MDA_directory,HDF_directory)

def frun_main(input_file="7bmb1_0933.mda", MDA_directory=None,
              output_HDF=None, HDF_directory=None):
    '''Function that tests the working of the code.
    '''
    MDA_directory, HDF_directory = fcheck_directories(input_file, MDA_directory, HDF_directory)
    if not output_HDF:
        output_HDF = input_file.split(".")[0] + ".hdf5"
    try:
        with h5py.File(HDF_directory + output_HDF, 'w') as output_hdf:
            data = fstart_unpacking(input_file, MDA_directory)
            extra_pv_pointer = fread_file_header(data, output_hdf)
            fread_scan(data, output_hdf)
            # Put in the Extra PVs
            if extra_pv_pointer:
                data.set_position(extra_pv_pointer)
                fread_extra_PVs(data, output_hdf)
    except IOError:
        logger.error("IOError: Output file could not be opened.")
        return
    except EOFError:
        logger.error("Unexpectedly reached end of file.  File may be damaged.")


def frun_append(input_MDA='7bmb1_0260.mda', MDA_directory=None,
                output_HDF='Scan_260.hdf5', HDF_directory=None):
    '''Parses data from the MDA file, adding datasets to an existing HDF5 file.
    '''
    MDA_directory, HDF_directory = fcheck_directories(input_file, MDA_directory, HDF_directory)
    if not os.path.isfile(HDF_directory + output_HDF):
        logger.error("File " + output_HDF + " cannot be opened for appending.  Exiting.")
        raise IOError
    try:
        with h5py.File(HDF_directory + output_HDF, 'r+') as output_hdf:
            data = fstart_unpacking(input_MDA, MDA_directory)
            extra_pv_pointer = fread_file_header(data, output_hdf)
            fread_scan(data, output_hdf, [])
            # Put in the Extra PVs
            if extra_pv_pointer:
                data.set_position(extra_pv_pointer)
                fread_extra_PVs(data, output_hdf)
    except IOError:
        logger.error("IOError: Output file could not be opened.")
        return
    except EOFError:
        logger.error("Unexpectedly reached end of file.  File may be damaged.")

if __name__ == "__main__":
    import sys
    import os

    print(sys.argv)
    if len(sys.argv) == 1:
        frun_main()
    elif len(sys.argv) == 2:
        if sys.argv[1] == "-h":
            print("""Program to convert sscan MDA files to HDF5. /n
                If one argument is given, assume it is a path to an MDA file. /n
                The HDF5 file will be placed in the same directory.
                Alternatively, give file and a directory in which to place HDF file.
                """)
        else:
            frun_main(sys.argv[1], os.getcwd() + '/')
    elif len(sys.argv) == 3 and os.path.isdir(sys.argv[2]):
        frun_main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4 and os.path.isdir(sys.argv[2]):
        frun_main(sys.argv[1], sys.argv[2], sys.argv[3])
