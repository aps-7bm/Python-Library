'''Script to check the integrity of DataGrabber scope traces.
The script will check to find the min and max of the traces, as well as
the overall max and min across the file.

Alan Kastengren, XSD, APS

Started: January 17, 2017
'''
import readdatagrabber as rdg
import numpy as np
import os.path
import logging

#Globals
location_keys = ['X','Y','Theta']

def ftrace_extrema_yoko(dg_channel, terminal_output=True):
    '''Finds the maximum and minimum of a DataGrabber channel.
    
    Also returns maximum and minimum valid values for judging clipping.
    As of 1/2017, this functionality does not work.
    
    Inputs:
    dg_channel: a readdatagrabber Channel object
    terminal_output: print results to the console?
    
    Outputs:
    channel_max: maximum voltage seen in the trace.
    channel_min: minimum voltage seen in the trace.
    max_valid_V: maximum voltage without clipping
    min_valid_V: minimum voltage without clipping
    '''
    #Read in the scale and offset from the dg_channel header
    dg_scale = float(dg_channel.channel_header['Scale'])
    dg_offset = float(dg_channel.channel_header['Offset'])
    
    #Compute the max and min valid voltages, assuming 4.8 divisions +/-
    #from the offset voltage
    max_valid_V = dg_offset + 4.8 * dg_scale
    min_valid_V = dg_offset - 4.8 * dg_scale
    
    #Read in the trace voltages
    dg_channel.fread_data_volts()
    
    #Find the max and min of the trace
    channel_max = np.max(dg_channel.data)
    channel_min = np.min(dg_channel.data)
    
    #Delete the channel data to prevent a memory leak
    del(dg_channel.data)
    return (channel_max,channel_min,max_valid_V,min_valid_V)

def fcheck_file(filename,user_description):
    '''Checks extrema of all traces in a DataGrabber file with a given description.
    
    Outputs are printed to the terminal.
    Inputs:
    filename: name of DataGrabber file, including path
    user_description: text to be matched to a channel's UserDescription
    Outputs:
    tuple of overall maximum and minimum values seen in the file on this channel.
    '''
    #Open the DataGrabber file
    if not os.path.isfile(filename):
        print "No such file exists."
        return
    try:
        header_data = rdg.fread_headers(filename)
    except IOError:
        print "Problem reading file " + filename
        return
    #Find which location variables are actually in these scans
    actual_location_keys = []
    for k in location_keys:
        if k in header_data[0].coordinate_header:
            actual_location_keys.append(k)
    
    #Print out some info to the terminal.
    print("Analyzing channel " + user_description + " in file " + filename)
#     chan = header_data[0].fchannel_by_name(user_description)
#     print chan.channel_header
    i = 0  
    #Make a header row 
    headers = ["Trace #"] + actual_location_keys + ['Max V','Min V']#,'Max_Clip V','Min Clip V','Clipped']
    field_width = max(map(len,headers)) + 4
    format_string = ''
    for __ in headers:
        format_string = format_string + '{:' + str(field_width) + '}'
    print(format_string.format(*headers))
    print('-'*field_width*len(headers))
    
    #Develop the format string for the following rows
    format_string = '{:<' + str(field_width) + 'd}'
    for __ in actual_location_keys +['Max_V','Min_V']:
        format_string = format_string + '{:<' + str(field_width) + 'f}'
    #format_string = format_string + '{:<' + str(field_width) + 's}'
    
    #Loop through the coordinates
    max_vals = np.zeros(len(header_data))
    min_vals = np.zeros(len(header_data))
    for coord in header_data:
        #Retrieve the channel by the UserDescription name
        channel = coord.fchannel_by_name(user_description)
        #Fill in data to be printed to the terminal
        print_data = [i+1]
        for k in actual_location_keys:
            print_data.append(float(coord.coordinate_header[k]))
        #Analyze for extrema
        (max_V,min_V,clip_max_V,clip_min_V) = ftrace_extrema_yoko(channel,True)
        max_vals[i] = max_V
        min_vals[i] = min_V
        print_data.append(max_V)
        print_data.append(min_V)
#         print_data.append(clip_max_V)
#         print_data.append(clip_min_V)
#         #Check for clipping
#         print_data.append('Yes' if max_V > clip_max_V or 
#                           min_V < clip_min_V else 'No')
        print(format_string.format(*print_data))
        i += 1
    print("Overall max = " + str(np.max(max_vals)))
    print("Overall min = " + str(np.min(min_vals)))
    return(np.max(max_vals),np.min(min_vals))

if __name__ == '__main__':
    fcheck_file('/home/akastengren/data/SprayData/Cycle_2016_2/SparkRadiography/Scans/Scan_138.dat','BIM')
    