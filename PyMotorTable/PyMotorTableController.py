'''Controller for PyMotorTable.  Interface between calculations, GUI, and
EPICS.

Alan Kastengren, XSD
Started: February 3, 2014

Change Log
November 18, 2014: slight change so positioner name updates on the center plot if it changes.
'''
#Imports
import PyMotorTableEpics as pmtepics
import PyMotorTableCalcs as calcs
import numpy as np

class PyMotorTableControl():
    def __init__(self,GUI_object):
        '''Initialize with the GUI object specified.
        '''
        self.GUI_object = GUI_object
    
    def fGUI_clear_all_points(self,event):  
        '''Clear all points from program.  Does not clear points in EPICS.
        '''
        pass
    
    def fGUI_display_temp(self,event):
        pass
    
    def fGUI_confirm_temp(self,event):
        calcs.fconfirm_temp()
        self.frefresh_display(event)

    def fGUI_change_positioner_num(self,event):
        pmtepics.positioner_num = event.GetEventObject().GetStringSelection()   
    
    def fGUI_recalculate_temp_grid(self,event):
        print "Recalculating temp grid"
        try:
            #Make sure that the overwriting boolean in calcs is set correctly
            calcs.overwrite_overlaps = self.GUI_object.overwrite_overlaps_cbox.GetValue()
            calcs.order_reversed = self.GUI_object.reverse_order_cbox.GetValue()
            #Read in the start and end values
            temp_start = float(self.GUI_object.start_text_entry.GetValue())
            temp_end = float(self.GUI_object.end_text_entry.GetValue())
            #Read in the spacing or number of points and fill in the text box for the one not selected.
            temp_spacing = None
            temp_num_pts = None
            if self.GUI_object.spacing_rbutton.GetValue():
                temp_spacing = float(self.GUI_object.spacing_text_entry.GetValue())
                self.GUI_object.num_points_text_entry.SetValue(str(calcs.fcompute_num_points(temp_start, temp_end, temp_spacing)))
            else:       #The number of points radio button has been selected
                temp_num_pts = int(self.GUI_object.num_points_text_entry.GetValue())
                self.GUI_object.spacing_text_entry.SetValue(str(calcs.fcompute_spacing(temp_start, temp_end, temp_num_pts)))
            #Compute the temporary grid
            calcs.fcompute_temp_grid(temp_start, temp_end, temp_spacing, temp_num_pts)
            calcs.fmerge_temp_confirmed()
        except ValueError:
            print "Error in the text.  Try again."
            GUI_object.fdisplay_dialog("Invalid Entry for Grid Specification.  Try Again.","Invalid Entry")
        #Call the plotting function
        self.frefresh_display(None)
    
    def fGUI_select_spacing(self,event):
        '''Change GUI for selecting spacing rather than # of pts.
        '''
        self.GUI_object.num_points_text_entry.SetEditable(False)
        self.GUI_object.spacing_text_entry.SetEditable(True)
    
    def fGUI_select_num_pts(self,event):
        '''Change GUI for selecting # pts rather than spacing.
        '''
        self.GUI_object.num_points_text_entry.SetEditable(True)
        self.GUI_object.spacing_text_entry.SetEditable(False)
    
    def fchange_reverse_order(self,event):
        '''Respond to changing reverse order checkbox.
        '''
        calcs.order_reversed = event.GetEventObject().GetValue()
        self.fGUI_recalculate_temp_grid(event)
    
    def fchange_erase_overlaps(self,event):
        '''Respond to changing erase overlaps checkbox.
        '''
        calcs.overwrite_overlaps = event.GetEventObject().GetValue()
        self.fGUI_recalculate_temp_grid(event)
    
    def fwrite_points(self,event):
        '''Respond to button press to write points to EPICS.
        '''
        print "Write Points"
        pmtepics.fwrite_pvs(calcs.getConfirmedPoints())
        calcs.current_points = pmtepics.fread_pvs()
        self.frefresh_display(None)
    
    def fread_points(self,event):
        '''Respond to button press to read points from EPICS.
        '''
        print "Read Points"
        calcs.current_points = pmtepics.fread_pvs()
        self.frefresh_display(None)
    
    def fclear_temps(self,event):
        '''Respond to button press to clear temporary points.
        '''
        calcs.fclear_temps()
        self.frefresh_display(None)
    
    def fclear_all_points(self,event):
        '''Respond to button press to clear all points.
        '''
        calcs.fclear_all()
        self.frefresh_display(None)
    
    def fepics_pv_entry(self,event):
        '''Respond to button press for EPICS PV to connect.
        '''
        print "Connect to PV"
        print self.GUI_object.epics_pv_label.GetValue()
        pmtepics.fopen_pvs(self.GUI_object.epics_pv_label.GetValue())
        self.frefresh_display(None)
    
    def fupdate_epics_connected(self):
        '''Changes the color next to the EPICS PV label based on connected state.
        '''
        if pmtepics.fcheck_connection():
            self.GUI_object.epics_connection_indicator.SetBackgroundColour('Green')
        else:
            self.GUI_object.epics_connection_indicator.SetBackgroundColour('Red') 
    
    def frefresh_display(self,event):
        '''Refresh the display.
        '''
        self.fupdate_epics_connected()
        #Plot the confirmed points, temporary points, and delete points
        self.GUI_object.points_plot.clear()
        if calcs.getConfirmedPoints().size:
            self.GUI_object.points_plot.plot(np.zeros_like(calcs.getConfirmedPoints()),calcs.getConfirmedPoints(),
                                           color='blue',label='Confirmed Pts',linewidth=0,marker='o')
        if calcs.getDeletePoints().size:
            self.GUI_object.points_plot.oplot(np.zeros_like(calcs.getDeletePoints()),calcs.getDeletePoints(),
                                           color='red',label='Deleted Pts',linewidth=0,marker='o')
        if calcs.getTempPoints().size:
            self.GUI_object.points_plot.oplot(np.zeros_like(calcs.getTempPoints()),calcs.getTempPoints(),
                                           color='orange',label='Temp Pts',linewidth=0,marker='o',show_legend=True)
        #Set the x axis from -1 to 1
        all_points = np.concatenate((calcs.getConfirmedPoints(),calcs.getTempPoints(),calcs.getDeletePoints()))
        self.GUI_object.points_plot.set_xylims([-1,2,-1,1])
        if all_points.size:
            pts_range = np.max(all_points) - np.min(all_points)
            self.GUI_object.points_plot.set_xylims([-1,2,np.min(all_points)-0.1*pts_range,np.max(all_points)+0.1*pts_range])
        #Set the y label to the name of the EPICS PV being controlled
        self.GUI_object.points_plot.set_ylabel(pmtepics.fget_variable_name())
        #Update the grid of points on the right-hand size
        self.GUI_object.table.fupdate_points()
        self.GUI_object.history_table.fupdate_points()
        #Update the labels for the number of current and confirmed points.
        self.GUI_object.num_confirmed_temp_text.SetLabel(str(calcs.getConfirmedPoints().size 
                                                        + calcs.getTempPoints().size - calcs.getDeletePoints().size))
        self.GUI_object.num_confirmed_text.SetLabel(str(calcs.getConfirmedPoints().size))
        
