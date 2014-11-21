#! /usr/bin/env python
"""Module to manipulate DataGrabber files.

This module provides functions to open and read DataGrabber files.
Data are stored as DataGrabberPoint objects, which are themselves filled
with DataGrabberChannel objects.  This will provide a useful mapping for
conversion of DataGrabber files to another format.

Alan Kastengren, XSD, Argonne National Laboratory
Started: November 12, 2012

Credit to Daniel Duke for his pyDataGrabber.py code, which provided
useful ideas for this code.

Edits:
May 5, 2014: Add check that DataGrabber file exists.

"""

import numpy as np
import os

#Dictionary for converting datatype to number of bytes
bytes_per_point = {"byte":1, "short":2, "int":4, "float":4, "long":8, "double":8}

def fread_headers(filename,record_length_key="RecordLength",
                    data_type_key="BinaryDataType",
                    channel_num_key="NumberOfChannels",
                    coord_start="FileType=DataGrabberBinary",
                    chan_start="Channel"):
    #Save a new copy of the filename, which will be useful later.
    #Instantiate a class to hold the header info
    file_data = []
    #Throw an IOError if the file doesn't exist
    if not os.path.isfile(filename):
        print "No such file exists: " + filename
        raise IOError
    """Loop through the DataGrabber file, saving headers."""
    with open(filename,'rb') as dg_file:
        #Read in a potential header.  Trap possible extra \n
        line = "a"
        while line!="":
            line = dg_file.readline()
            #If this is a good coordinate header, 
            if line.startswith(coord_start):
                #Instantiate a DataGrabberCoordinate instance for this coordinate.
                current_coord = DataGrabberCoordinate(filename)
                file_data.append(current_coord)
                #Save it as a dictionary.  Remove last two characters, which are /r/n.
                current_coord.coordinate_header = fparse_header_line(line[:-2])
                #Figure out how many channels there are.
                num_channels = int(current_coord.coordinate_header[channel_num_key])
                #For each channel:
                for i in range(num_channels):
                    #Read in potential header.
                    line = dg_file.readline()
                    while not(line.startswith(chan_start)):
                        #If we reach the end of the file here, this is a serious problem.
                        if line=="": 
                            print "Lost track of header for channel", i, ",\
                                    coordinate #", len(file_data)
                            return file_data
                        #Otherwise, read another line
                        line = dg_file.readline()
                    #If header is good, instantiate a new DataGrabberChannel object.
                    current_channel = DataGrabberChannel()
                    #Save filename for use in getting data in future.
                    current_channel.filename = filename 
                    current_coord.channels.append(current_channel)
                    #Parse the header line into a dictionary. Remove last two characters, which are /r/n.
                    current_channel.channel_header = fparse_header_line(line[:-2])
                    #Add a file pointer to the dictionary.
                    current_channel.channel_header["FPointer"] = dg_file.tell()
                    print dg_file.tell()
                    #Figure out how many bytes to skip.  
                    num_bytes_per_point = bytes_per_point[current_channel.channel_header[data_type_key]]
                    num_points = int(current_channel.channel_header[record_length_key])
                    #Skip the appropriate number of bytes.
                    dg_file.seek(num_points * num_bytes_per_point,1)
        print "Found ", len(file_data)," positions."
        return file_data

def fparse_header_line(header,pair_delimit=" ",split_delimit="="):
    """Parses a DataGrabber header line into a dictionary"""
    split_header = header.split(pair_delimit)
    output = {}
    for pair in split_header:
        key_value = pair.split(split_delimit)
        #If we have an invalid split (e.g. empty string), just skip
        if len(key_value) <2: continue
        output[key_value[0]] = key_value[1]
    return output
            
class DataGrabberChannel():
    """Holder for channel headers and actual data from DataGrabber channel"""
    def __init__(self):
        self.channel_header = {}
        self.data = None
        self.filename = None
        self.dtype_dict = {"byte":"b", "short":"h", "int":"i", "float":"f", "long":"l", "double":"d"}

    def fread_data(self,normalize=False,endian="big",data_type_key="BinaryDataType",record_length_key="RecordLength"):
        """Read in the binary data from a DataGrabber file channel."""
        #Come up with the appropriate dtype for the np.fromfile option
        data_dtype = np.dtype('byte')
        if endian=="little":
            dtype_string = "<"+self.dtype_dict[self.channel_header[data_type_key]]
            print dtype_string
            data_dtype = np.dtype(dtype_string)
        else:
            data_dtype = np.dtype(">"+self.dtype_dict[self.channel_header[data_type_key]])
        #Use the np.fromfile function to read in data            
        with open(self.filename,'rb') as dg_file:
            dg_file.seek(int(self.channel_header["FPointer"]),0)
            self.data = np.fromfile(dg_file,dtype=data_dtype,count=int(self.channel_header[record_length_key]))
    
    def fread_data_volts(self,normalize=False,endian="big",data_type_key="BinaryDataType",record_length_key="RecordLength",
                         scale_key='Scale',offset_key='Offset'):
        '''Read in the binary data from a DataGrabber file channel, but in volts.
        '''
        #Use previous method to read in the data array
        self.fread_data(normalize,endian,data_type_key,record_length_key)
        scale_value = float(self.channel_header[scale_key])
        offset_value = float(self.channel_header[offset_key])
        self.data = scale_value * self.data + offset_value

class DataGrabberCoordinate():
    """Holder for header and channels from DataGrabber coordinate (point)"""
    def __init__(self,filename):
        self.coordinate_header = {}
        self.channels = []
        self.filename = filename
    
    def fread_coord_data(self):
        """Read in the data for all channels of this position."""

        for channel in self.channels:
            channel.fread_data()
    
