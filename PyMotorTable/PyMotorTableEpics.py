'''Module for EPICS communication for PyMotorTable.

Alan Kastengren, XSD
Started June 27, 2013

Change log
November 18, 2014: add function to get name of positioner.
November 18, 2014: fix bug that ignored user positioner number
'''
import epics
#
#PVs that will be used.  Leave as module variables.  Blank lists just to define them
_array_PV = None
_num_points_PV = None
positioner_num = '1'
variable_name = ""
scan_prefix = ""

def fopen_pvs(prefix_name):
    '''Initialize the connection to the EPICS PVs we need.
    '''
    global scan_prefix
    global _array_PV
    global _num_points_PV
    global variable_name
#    global positioner_num
#    positioner_num = str(positioner_number)
    scan_prefix = prefix_name
    temp_name = prefix_name + '.P'+positioner_num + 'PA'
    print temp_name
    _array_PV = epics.PV(prefix_name + '.P'+ positioner_num + 'PA')
    _num_points_PV = epics.PV(prefix_name + '.NPTS')
    _variable_PV = epics.PV(prefix_name + '.P'+ positioner_num + 'PV')
    print _array_PV.wait_for_connection() and _num_points_PV.wait_for_connection()
    variable_name = fget_variable_name()
    return _array_PV.wait_for_connection() and _num_points_PV.wait_for_connection()

def fcheck_connection():
    if _array_PV == None or _num_points_PV == None:
        return False
    return _array_PV.status and _num_points_PV.status 
    
def fwrite_pvs(point_array):
    _array_PV.put(point_array,wait=True,timeout=5.0,use_complete=True)
    _num_points_PV.put(len(point_array),wait=True,timeout=5.0,use_complete=True)
    return _array_PV.put_complete and _num_points_PV.put_complete

def fread_pvs():
    print _num_points_PV.get()
    return _array_PV.get()[:_num_points_PV.get()]

def fget_variable_name():
    _variable_PV = epics.PV(scan_prefix + '.P'+ positioner_num + 'PV')
    return _variable_PV.get(as_string=True)
