'''Underlying calculations for PyMotorTable.

Alan Kastengren, XSD
Started June 15, 2013

Change Log
November 18, 2014: Make initial points temporary, rather than confirmed, so they don't have to be erased.
'''
#imports
import numpy as np
import math
#Lists to save points.  
temp_points = [0,1]        #Provisional: all points in revised grid
confirmed_points = []   #Points in current grid
delete_points = []      #Confirmed points that will be deleted
_rounding = 0.001      
order_reversed = False
overwrite_overlaps = True  
current_points = []       
history_dict = {'Start':[],'End':[],'Spacing':[],'Num Pts':[]}
temp_dict = {'Start':None,'End':None,'Spacing':None,'Num Pts':None}
#
def fset_rounding(new_rounding):
    '''Sets the parameter than controls how close points must be to be
    considered equal.
    '''
    _rounding = new_rounding
    
def fclear_temp():
    delete_points[:] = []
    temp_points[:] = []

def fclear_all():
    '''Clears all temp, delete, and confirmed points, as well as record
    of previously confirmed points.
    '''
    delete_points[:] = []
    temp_points[:] = []
    confirmed_points[:] = []
    for key in ['Start','End','Spacing','Num Pts']:
        history_dict[key] = []
    
def fcompute_spacing(start,end,num_points):
    '''Compute the spacing between points given start, end, # of points.
    '''
    if num_points < 2:
        return 0
    else:
        return math.fabs(start - end) / (num_points - 1)

def fcompute_num_points(start,end,spacing):
    '''Compute the number of points given start, end, and spacing.
    '''
    return int(math.floor((math.fabs(start-end) + _rounding) / spacing) + 1)
    
    
def fcompute_temp_grid(start,end,spacing=None,num_points=None):
    '''Function to compute the temporary points.
    Written to accept either a fixed spacing or a fixed number of points.
    '''
    #Save the start,end,spacing,and num_pts temporarily so we can save later
    temp_dict['Start'] = start
    temp_dict['End'] = end
    #If we have a valid spacing; note that if so, we ignore num_points.
    if spacing:
        #Compute the number of points we will get.  Need to add one for endpts.
        num_strides = fcompute_num_points(start,end,spacing) - 1
        #Use numpy.linspace to compute temp_points list.  Keep track of signs
        if (end - start) > 0:
            temp_points[:] = list(np.linspace(start,start+num_strides*spacing,num_strides+1))
        else:
            temp_points[:] = list(np.linspace(end,end+num_strides*spacing,num_strides+1))
    #If no valid spacing, use number of points.
    elif num_points is not None:
        temp_points[:] = list(np.linspace(start,end,num_points))
    else:
        print "Need either a valid number of points or spacing."
        return None    
    temp_dict['Num Pts'] = len(temp_points)
    if len(temp_points) > 1:
        temp_dict['Spacing'] = temp_points[1] - temp_points[0]
    else:
        temp_dict['Spacing'] = 0

def fmerge_temp_confirmed():
    '''Function to compute full array of temporary points.
    If there are already confirmed points, figure out whether any points are in
    common between confirmed and temp.  Also figure out if any confirmed points
    will be deleted.
    If there are no temp points and the confirmed points are to be overwritten,
    the confirmed points are simply deleted.
    '''
    #First case: if there are no confirmed points, do nothing.
    if len(confirmed_points) == 0:
        return
    #Clear the delete_points list
    delete_points[:] = []
    #Sort both lists
    confirmed_points.sort()
    temp_points.sort()
    
    #If we are overwriting overlap regions, find confirmed points between
    #start and end of temp_points and add to delete_points
    if overwrite_overlaps:
        for item in confirmed_points:
            #What if there are no temp points: useful for deleting part of grid
            if not len(temp_points):    
                if temp_dict['Start'] < temp_dict['End'] and temp_dict['Start']-item < _rounding and temp_dict['End']-item > -_rounding:
                    delete_points.append(item)
                elif temp_dict['End']-item < _rounding and temp_dict['Start']-item > -_rounding:
                    delete_points.append(item)
            #Now, handle the more typical case where there actually are temp points
            elif temp_points[0]-item < _rounding and temp_points[-1]-item > -_rounding:
                delete_points.append(item)
    #If we aren't overwriting overlaps, find coincident points and remove
    #from the temp array
    else:
        for item in confirmed_points:
            for (i,temp) in enumerate(temp_points):
                if math.fabs(temp-item) < _rounding:
                    del temp_points[i] 
                #If we are larger than item, might as well break loop, since lists are sorted
                elif temp_points[i] > item:
                    break
            
def fdelete_points():
    '''Delete the points in delete_points from confirmed_points
    '''
    for delete_item in delete_points:
        for i in range(len(confirmed_points)):
            if delete_item == confirmed_points[i]:
                del confirmed_points[i]
                break
        else:
            print "Unmatched delete_point entry.  Something didn't work."
            print delete_item  
    delete_points[:] = []

def fconfirm_temp(order_reversed=False):
    '''Confirm the temp grid, making the temp points confirmed points.
    Also saves parameters of this grid in the history dictionary.
    '''
    if len(delete_points):
        fdelete_points()
    confirmed_points.extend(temp_points[:])
    confirmed_points.sort(reverse=order_reversed)
    fclear_temp()
    for key in ['Start','End','Spacing','Num Pts']:
        history_dict[key].append(temp_dict[key])
        temp_dict[key] = None
    
def getConfirmedPoints():
    return np.array(sorted(confirmed_points,reverse=order_reversed))

def getTempPoints():
    return np.array(sorted(temp_points,reverse=order_reversed))

def getDeletePoints():
    return np.array(sorted(delete_points,reverse=order_reversed))

def getCurrentPoints():
    return np.array(sorted(current_points,reverse=order_reversed))
    
