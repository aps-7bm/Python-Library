'''Script to transfer files and convert to HDF5.
'''
import MDA2HDF5_Fluorescence as m2h
import epics
import time
import shutil
#
#Define variables
wait_button = epics.PV('7bmb1:busy1')
next_scan_num = epics.PV('7bmb1:saveData_scanNumber')
wait_time = 1.0
digits = 4
original_path = '/home/7bmdata/vme_save/FuelSpray/Cycle_2014_2/'
final_path = '/home/SprayData/Cycle_2014_2/AFRL_Edwards/'

waiting = True
while waiting:
    if wait_button.get():
        print "I'm here."
        scan_num = int(next_scan_num.get()) - 1
        format_string = '{0:0'+str(digits)+'d}'
        filename = '7bmb1_'+format_string.format(scan_num)+'.mda'
        shutil.copy(original_path+filename,final_path+filename)
        wait_button.put(0)
        time.sleep(1.0)
        try:
            m2h.frun_main(filename,final_path,False)
        except ValueError:
            print "Problem making the HDF5 file."
            wait_button.put(0)
            time.sleep(0)
    else:
        print "Still waiting."
        time.sleep(1.0)

