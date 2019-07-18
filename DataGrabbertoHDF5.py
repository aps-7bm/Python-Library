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
#
def fconvert_and_average_file(dg_filename,hdf_filename=None,
                    path="/data/Data/SprayData/Cycle_2013_3/ISU/",
                    group_keys = ("X","Y"), convert_volts=True,average=True):
    '''Convert DataGrabber file to HDF5 if the input is the name of the output
    file rather than an HDF5 object.  Simply makes an HDF5 File object
    and calls the converter based on the HDF5 File object.
    '''
    #Make up an approrpiate output file name if none is given
    if not hdf_filename:
        hdf_filename = dg_filename.split(".")[0]+".hdf5"
    #Open an HDF file and call the converter based on having an HDF5 File object
    with h5py.File(path+hdf_filename,'w') as write_file:
        fconvert_and_average_h5py_hdffile(dg_filename,write_file,path,group_keys,convert_volts,average)

def fconvert_and_average_h5py_hdffile(dg_filename,write_file,
                    path="/home/beams/7BMB/SprayData/Cycle_2013_3/ISU/",
                    group_keys = ("X","Y"), convert_volts=True, average=True):
    '''Convert DataGrabber file to HDF5.  Input h5py object,
    not hdf file name.
    '''
    #Read in the header information
    try:
        headers = rdg.fread_headers(path+dg_filename)
    except IOError:
        print "Problem reading file " + dg_filename
        return
    if average:
        #Make lists to hold x, y, and average of scope traces
        x = []
        y = []
        scope_channel_names=[]
        #Make a list of channel names for use in averaging.
        for channel in headers[0].channels:
            scope_channel_names.append(str(channel.channel_header["UserDescription"]))
        #Make a list of lists for the average points from each of the channels
        scope_trace_averages=[[]]*len(scope_channel_names)
    #Loop through the coordinates
    for coord in headers:
        #Come up with a name for this group
        group_name = ""
        for key in group_keys:
            group_name += key + "=" + str(coord.coordinate_header[key]) + ", "
        #Leave off the trailing ", "
        group_name = group_name[:-2]
        print group_name
        #Create a group for this coordinate
        coord_group = write_file.create_group(group_name)
        #Fill in attributes for this group from the coordinate header
        for key in coord.coordinate_header.keys():
            coord_group.attrs[key] = coord.coordinate_header[key]
            if key=='X' and average:
                x.append(float(coord.coordinate_header[key]))
            if key=='Y' and average:
                y.append(float(coord.coordinate_header[key]))
        #Loop through the channels under this coordinate
        for channel in coord.channels:
            #Use UserDescription as name for new subgroup
            channel_name = str(channel.channel_header["UserDescription"])
            print channel_name
            #Read in the data for this channel
            channel.fread_data()
            #Create a dataset out of these data
            #If desired, put this in volts and save as floats (to save space)
            if convert_volts:
                channel.fread_data_volts()
                dataset = coord_group.create_dataset(channel_name,data=channel.data,dtype='f4')
            #If we aren't converting to volts, keep in the same format
            else:
                dataset = coord_group.create_dataset(channel_name,data=channel.data)
            #Fill in attributes for this group from the coordinate header
            for key in channel.channel_header.keys():
                dataset.attrs[key] = channel.channel_header[key]
            if average:
                #Average this scope trace and save it in the appropriate list            
                for i in range(len(scope_channel_names)):
                    if scope_channel_names[i] == channel_name:
                        scope_trace_averages[i].append(np.mean(channel.data))
        #Clear out this entry in headers so we don't overload memory
        coord = ""
    #If we are averaging, write these averages to the hdf file.
    if average:
        #Write datasets for x, y, scope trace averages
        write_file.create_dataset('X',data=x)
        write_file.create_dataset('Y',data=y)
        for i in range(len(scope_channel_names)):
            write_file.create_dataset(scope_channel_names[i],data=scope_trace_averages[i])
    #Close the file
    write_file.close()

def fbatch_conversion(file_nums,dg_filename_prefix='Scan_',dg_filename_suffix='.dat',
                    digits=4, path="/data/Data/SprayData/Cycle_2013_3/ISU/",
                    group_keys = ("X","Y"), convert_volts=True, average=True):
    '''Class to batch process a large number of DataGrabber files.
    Should work with strings or numbers given in file_nums list.
    '''
    #Loop through the input string numbers, converting first to int, then to
    #correct number of digits as a string.
    for i_str in file_nums:
        i = int(i_str)
        format_string = '{0:0'+str(digits)+'d}'
        fconvert_and_average_file(dg_filename_prefix+format_string.format(i)+dg_filename_suffix,
                    None,path,group_keys,convert_volts,average)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print "Code to convert DataGrabber files to HDF5 format."
    elif len(sys.argv) == 2:
        #Split input filename on period and change extension
        fconvert_and_average_file(sys.argv[1])
