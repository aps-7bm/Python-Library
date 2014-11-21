""" Module to convert MDA files saved by sscan to HDF5.
Follows format specified at http://www.aps.anl.gov/bcda/synApps/sscan/saveData_fileFormat.txt
Alan Kastengren, XSD
Started February 24, 2013

Change log
April 5, 2013: change saving of positioner and detector attributes to use names when possible.
April 5, 2013: first steps at saving extra PVs.  Handles several EPICS datatypes, but not well tested yet.
"""
#
#Imports
import h5py
import xdrlib
#
def fparse_counted_string(data):
    '''Parses the data to get a counted string,
        which is not really in the XDR standard, but
        is used in the MDA file format.  Actually have to get the string 
        length twice, for reasons I don't understand.
    '''
    string_length = data.unpack_int()
    if string_length > 0:
        return data.unpack_string()
    else:
        return ""
#    return data.unpack_string()

def fstart_unpacking(input_filename,directory):
    '''Takes the input filename and converts into
    a buffer suitable for the xdrlib to get started.
    '''
    with open(directory+input_filename) as input_file:
        f = input_file.read()
        return xdrlib.Unpacker(f)

def finitialize_hdf(output_filename,directory):
    '''Returns an h5py file object.
    '''
    return h5py.File(directory+output_filename,'w')
    
    
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
    print output_hdf.attrs.keys()
    print output_hdf.attrs.values()
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
    #
    print lower_scans_pointers
    #Read through the info fields
    hdf_group.attrs["Scan Name"] = fparse_counted_string(data)
    hdf_group.attrs["Time Stamp"] = fparse_counted_string(data)
    #Print out to make sure we are still working
    print hdf_group.attrs.keys()
    print hdf_group.attrs.values()
    #Find the number of positioners, detectors, and triggers
    num_positioners = data.unpack_int()
    num_detectors = data.unpack_int()
    num_triggers = data.unpack_int()
    print num_positioners
    print num_detectors
    print num_triggers
    #
    #Fill up lists of dictionaries of meta data for positioners,
    #detectors, and triggers.
    positioner_meta = []
    detector_meta = []
    trigger_meta = []
    for i in range(num_positioners):
        positioner_meta.append(fread_positioner(data))
    for i in range(num_detectors):
        detector_meta.append(fread_detector(data))
    for i in range(num_triggers):
        trigger_meta.append(fread_trigger(data))
    #
    #Put these meta data as attributes of the group.
    for dict in positioner_meta:
        for key,value in dict.iteritems():
            if dict["Name"]:
                if key != "Name":
                    hdf_group.attrs[dict['Name']+"_"+str(key)] = value
            else:
                hdf_group.attrs["Positioner_"+str(i)+"_"+str(key)] = value
    for dict in detector_meta:
        for key,value in dict.iteritems():
            if dict["Name"]:
                if key != "Name":
                    hdf_group.attrs[dict['Name']+"_"+str(key)] = value
            else:
                hdf_group.attrs["Detector_"+str(i)+"_"+str(key)] = value
    for i in range(num_triggers):
        for key,value in trigger_meta[i].iteritems():
            hdf_group.attrs["Trigger_"+str(i)+"_"+str(key)] = value
    #
    #Read in the actual data as datasets
    positioner_values = []
    for i in range(num_positioners):
        #If the positioner has a valid name, give the positioner dataset the name
        if positioner_meta[i]['Name']:
            positioner_array = data.unpack_farray(requested_num_points,data.unpack_double)
            hdf_group.create_dataset(positioner_meta[i]['Name'],data=positioner_array)
            positioner_values.append(positioner_array)
        #Otherwise, give it a generic name.
        else:
            hdf_group.create_dataset("Positioner_"+str(i),data=data.unpack_farray(requested_num_points,data.unpack_double))
    print "Done writing positioners."
    for i in range(num_detectors):
        #Same idea as positioners: give an intelligible name if we have one
        if detector_meta[i]['Name']:
            hdf_group.create_dataset(detector_meta[i]['Name'],data=data.unpack_farray(requested_num_points,data.unpack_float))
        else:
            hdf_group.create_dataset("Detector_"+str(i),data=data.unpack_farray(requested_num_points,data.unpack_float))
    #
    #Now, start looping through the lower dimensional scans.  Again, this
    #will end up being recursive for a multi-dimensional scan.
    if len(lower_scans_pointers)>0:
        for j in range(len(lower_scans_pointers)):
            #Create a new subgroup
            print "Going into a subgroup."
            #Give the subgroup an intelligible name if there is one positioner.
            if num_positioners == 1 and positioner_meta[0]['Name']:
                #subgroup = hdf_group.create_group(positioner_meta[0]['Name'] + "=" + str(positioner_values[0][j]))
                name = positioner_meta[0]['Name'] + "=" + str(positioner_values[0][j])
            else:
                name = "Rank_"+str(rank)+"_Point_"+str(j)
            subgroup = hdf_group.create_group(name)
            data.set_position(lower_scans_pointers[j])
            fread_scan(data,subgroup)
    return
    
def fread_positioner(data):
    '''Reads all of the meta data for a given positioner.
    Returns a dictionary of meta data.
    '''
    meta_data = {}
#    meta_data['Number'] = data.unpack_int()
    dummy = data.unpack_int()
    meta_data['Name'] = fparse_counted_string(data)
    meta_data['Desc'] = fparse_counted_string(data)    
    meta_data['Step_Mode'] = fparse_counted_string(data) 
    meta_data['Unit'] = fparse_counted_string(data) 
    meta_data['RBV_Name'] = fparse_counted_string(data) 
    meta_data['RBV_Desc'] = fparse_counted_string(data) 
    meta_data['RBV Unit'] = fparse_counted_string(data) 
    return meta_data

def fread_detector(data):
    '''Reads all of the meta data for a given detector.
    Returns a dictionary of meta data.
    '''
    meta_data = {}
#    meta_data['Number'] = data.unpack_int()
    dummy = data.unpack_int()
    meta_data['Name'] = fparse_counted_string(data)
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
        #print "PV Type for extra PV "+pv_name+", Description = "+ pv_desc +" = " + str(pv_type)
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
    
def frun_main(input_file="7bmb1_0958.mda",directory="/data/Data/SprayData/Cycle_2012_3/Time_Averaged_ISU/"):
    '''Function that tests the working of the code.
    '''
    #Parse the input file into the main part of the file name and the extension.
    split_name = input_file.split(".")
    print split_name
    output_filename = split_name[0] + ".hdf5"
    try:
        data = fstart_unpacking(input_file,directory)
        output_hdf = finitialize_hdf(output_filename,directory)
        extra_PV_position = fread_file_header(data,output_hdf)
        fread_scan(data,output_hdf)
        data.set_position(extra_PV_position)
        fread_extra_PVs(data,output_hdf)
    except IOError:
        print "Output file could not be opened."
        return
    finally:
        output_hdf.close()
        
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
    