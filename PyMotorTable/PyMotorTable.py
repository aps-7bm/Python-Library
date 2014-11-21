#!/APSshare/epd/rh6-x86_64/bin/python
'''Module for GUI for PyMotorTable.
Alan Kastengren, XSD
Started January 27, 2014
'''
#Imports
import wx
import wx.grid
#import wx.grid.GridTableMessage
from wx.lib.stattext import GenStaticText
from wx.lib.masked import NumCtrl
import PyMotorTableCalcs as model
import wxmplot
import PyMotorTableController
from copy import deepcopy

class MyApp(wx.App):
    def OnInit(self):
        frame = MainFrame(None,'PyMotorTable')
        frame.Show()
        self.SetTopWindow(frame)
        return True

class MainFrame(wx.Frame):
    '''Class for the main window
    '''
    def __init__(self,parent,title):
        self.model = PyMotorTableController.PyMotorTableControl(self)
        wx.Frame.__init__(self,parent,title=title,size=(1200,550))
        main_panel = self.fmake_main_panel(self)
        self.Layout()
        self.Show(True)
        self.model.frefresh_display(None)
        
    def fmake_main_panel(self,parent,color='Gray'):
        #Make the panel
        local_panel = wx.Panel(parent,-1)
        local_panel.SetBackgroundColour(color)
        local_sizer = wx.BoxSizer(wx.HORIZONTAL)
        local_sizer.Add(self.fmake_left_panel(local_panel),0,wx.EXPAND)
        local_sizer.Add(self.fmake_center_panel(local_panel),1,wx.EXPAND)
        local_sizer.Add(self.fmake_right_panel(local_panel),0,wx.EXPAND)
        local_panel.SetSizer(local_sizer)
        return local_panel

    def fmake_left_panel(self,parent):    
        '''Make the left-hand side of the window.
        '''
        #Make the panel
        local_panel = wx.Panel(parent,-1)
        local_panel.SetBackgroundColour('Black')
        
        #Make the subpanels
        sub_panels = [self.fmake_top_left_panel(local_panel),self.fmake_center_left_panel(local_panel),self.fmake_bottom_left_panel(local_panel)]
        proportion_sub = [0,1,0]
        border_list = [5,5,0]
        #Make the sizer: use a BoxSizer for this one
        local_sizer = wx.BoxSizer(wx.VERTICAL)
        min_size = [0,0]
        for (sub,prop,border_entry) in zip(sub_panels,proportion_sub,border_list):
            local_sizer.Add(sub,prop,flag=wx.EXPAND|wx.BOTTOM,border=border_entry)
            temp = sub.GetEffectiveMinSize()
            if temp[0] > min_size[0]:
                min_size[0] = temp[0]
                min_size[1] += temp[1]

        #Redraw panel and return it
        local_panel.SetMinSize(min_size)
        local_panel.SetSizer(local_sizer)
        local_panel.Layout()
        return local_panel
    
    def fmake_top_left_panel(self,parent):
        #Make the panel
        left_top_panel = wx.Panel(parent,-1)
        
        #Make the widgets that will populate the panel.  Make checkboxes class variables to access them later
        bold_font = wx.Font(16,wx.DEFAULT,wx.NORMAL,weight=wx.BOLD)
        left_top_label = GenStaticText(left_top_panel,-1,'Options',size=(200,40),style=wx.ALIGN_CENTER)
        left_top_label.SetFont(bold_font)
        self.reverse_order_cbox = wx.CheckBox(left_top_panel,-1,'Reverse Point Order',size=(200,30))
        self.Bind(wx.EVT_CHECKBOX,self.model.fchange_reverse_order,self.reverse_order_cbox)
        self.overwrite_overlaps_cbox = wx.CheckBox(left_top_panel,-1,'Erase Overlap Points',size=(200,30))
        self.overwrite_overlaps_cbox.SetValue(True)
        self.Bind(wx.EVT_CHECKBOX,self.model.fchange_erase_overlaps,self.overwrite_overlaps_cbox)
        left_top_widgets = [self.reverse_order_cbox,self.overwrite_overlaps_cbox]
        #Create a GridSizer to hold them and add the widgets
        left_top_sizer = wx.GridSizer(rows=len(left_top_widgets)+2,cols=1)
        left_top_sizer.Add(left_top_label,1,flag=wx.ALIGN_CENTER)
        for widget in left_top_widgets:
            left_top_sizer.Add(widget,1,flag=wx.ALIGN_CENTER_VERTICAL)
        #Redraw the panel and return it
        left_top_panel.SetSizer(left_top_sizer)
        return left_top_panel
    
    def fmake_center_left_panel(self,parent):
        #Make the panel
        local_panel = wx.Panel(parent,-1)
        
        #Make the sizer a GridBag sizer
        local_sizer = wx.GridBagSizer(vgap=10,hgap=10)
        
        #Make a list of the labels we will use.  Spacing and # Points as Radio Buttons
        entry_v_size = 20
        label_widgets_labels = ['Start','End','Spacing','# Points','# Confirmed','# Confirmed + Temp']
        label_widgets = []
        for entry in label_widgets_labels[:2]+label_widgets_labels[-2:]:
            label_widgets.append(GenStaticText(local_panel,-1,entry,style=wx.ALIGN_BOTTOM|wx.ALIGN_CENTER_HORIZONTAL))
        self.spacing_rbutton = wx.RadioButton(local_panel,-1,'Spacing',style=wx.ALIGN_BOTTOM|wx.ALIGN_CENTER_HORIZONTAL)
        self.num_points_rbutton = wx.RadioButton(local_panel,-1,'# Points',style=wx.ALIGN_BOTTOM|wx.ALIGN_CENTER_HORIZONTAL)
        label_widgets.insert(2,self.spacing_rbutton)
        label_widgets.insert(3,self.num_points_rbutton)
        
        #Make the text entry and display widgets
        text_ctrl_list = []
        self.start_text_entry = self.fmake_text_ctrl_list(text_ctrl_list,local_panel,'0')
        self.end_text_entry= self.fmake_text_ctrl_list(text_ctrl_list,local_panel,'1')
        self.spacing_text_entry = self.fmake_text_ctrl_list(text_ctrl_list,local_panel,'1')
        self.num_points_text_entry = self.fmake_text_ctrl_list(text_ctrl_list,local_panel)
        self.num_confirmed_text = self.fmake_text_ctrl_list(text_ctrl_list,local_panel,control=False)
        self.num_confirmed_temp_text = self.fmake_text_ctrl_list(text_ctrl_list,local_panel,control=False)
        self.Bind(wx.EVT_RADIOBUTTON,self.model.fGUI_select_num_pts,self.num_points_rbutton)
        self.Bind(wx.EVT_RADIOBUTTON,self.model.fGUI_select_spacing,self.spacing_rbutton)
        
        #Lay out the components on the panel.  Labels get twice the size of the text controls
        for i in range(len(label_widgets)):
            local_sizer.Add(label_widgets[i],pos=(i,0),span=(1,2),flag=wx.ALIGN_CENTER_VERTICAL)
            local_sizer.Add(text_ctrl_list[i],pos=(i,2),span=(1,1),flag=wx.ALIGN_CENTER_VERTICAL)
        
        #Set the minimum width of the panel
        temp = self.fcalculate_min_size(label_widgets)
        for widget in label_widgets:
            widget.SetMinSize(temp)
        local_panel.SetMinSize((temp[0]*3/2,temp[1]*len(label_widgets)))
        
        #Redraw panel and return it
        local_panel.SetSizer(local_sizer)
        local_panel.Fit()
        return local_panel
    
    def fmake_text_ctrl_list(self,containing_list,parent,text='0',control=True):
        '''Creates a text control bound to the text_ctrl object and puts it into list.
        '''
        event_function=self.model.fGUI_recalculate_temp_grid
        text_ctrl = ""
        if control:
            text_ctrl = wx.TextCtrl(parent,-1,text,size=(50,30),style=wx.TE_PROCESS_ENTER)
            self.Bind(wx.EVT_TEXT_ENTER,event_function)
        else:
            text_ctrl = GenStaticText(parent,-1,text,size=(50,30),style=wx.ALIGN_BOTTOM|wx.ALIGN_CENTER_HORIZONTAL)
        containing_list.append(text_ctrl)
        return text_ctrl

    def fmake_bottom_left_panel(self,parent):
        '''Make the bottom left panel: buttons to write grid, clear, read grid.
        '''
        #Make the panel
        local_panel = wx.Panel(parent,-1)
        
        #Make the buttons
        button_width = 180
        display_temp_button = wx.Button(local_panel,-1,'Display Temp Points')
        confirm_temp_button = wx.Button(local_panel,-1,'Confirm Temp Points')
        write_points_button = wx.Button(local_panel,-1,'Write Points to EPICS')
        read_points_button = wx.Button(local_panel,-1,'Read Pts from EPICS')
        clear_temp_button = wx.Button(local_panel,-1,'Clear Temp Points')
        clear_all_button = wx.Button(local_panel,-1,'Clear All Points')
        buttons = [display_temp_button,confirm_temp_button,
                   write_points_button,read_points_button,clear_temp_button,clear_all_button]
        #Make the sizer a Grid sizer, and add in the event code from the controller
        local_sizer = wx.GridSizer(rows=len(buttons),cols=1)
        event_functions = [self.model.fGUI_recalculate_temp_grid,
                            self.model.fGUI_confirm_temp,
                           self.model.fwrite_points,self.model.fread_points,
                           self.model.fclear_temps,self.model.fclear_all_points]
        for (button,func) in zip(buttons,event_functions):
            button.SetMinSize(write_points_button.GetEffectiveMinSize())
            local_sizer.Add(button,flag=wx.ALIGN_CENTER)
            self.Bind(wx.EVT_BUTTON,func,button)
                    
        #Redraw panel and return it
        local_panel.SetSizer(local_sizer)
        return local_panel
   
        
    def fmake_epics_pv_panel(self,parent,bg_color='Gray'):
        '''Makes the strip with the EPICS pv and indicator of connection status.
        '''
        #Make an overall panel
        local_panel = wx.Panel(parent,-1)
        local_panel.SetBackgroundColour(bg_color)
        
        #Make a local sizer: BoxSizer
        local_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        #Make a text label for "EPICS PV: ", a text control, spacer,
        #a label for positioner number, another spacer,and indicator of status
        self.epics_pv_label = wx.TextCtrl(local_panel,-1,'',size=(200,30),style=wx.TE_PROCESS_ENTER)
        self.epics_pv_connect_button = wx.Button(local_panel,-1,'Connect',size=(100,30))
        self.epics_connection_indicator = wx.Panel(local_panel,-1,size=(30,30))
        self.epics_connection_indicator.SetBackgroundColour('Red')
        pv_label = self.fmake_text_label(local_panel,-1,'EPICS PV:',(80,30),20,flag=wx.ALIGN_LEFT)
        positioner_label = self.fmake_text_label(local_panel,-1,'Positioner #',flag=wx.ALIGN_RIGHT)
        self.positioner_choice = wx.Choice(local_panel,-1,choices=['1','2','3','4'])
        
        #Event code for controls
        self.Bind(wx.EVT_TEXT_ENTER,self.model.fepics_pv_entry,self.epics_pv_label)
        self.Bind(wx.EVT_BUTTON,self.model.fepics_pv_entry,self.epics_pv_connect_button)
        self.Bind(wx.EVT_CHOICE,self.model.fGUI_change_positioner_num,self.positioner_choice)
        
        #Add all of the components to the panel
        panel_components = [pv_label,self.epics_pv_label,self.epics_pv_connect_button,positioner_label,
                            self.positioner_choice,wx.Panel(local_panel,-1,size=(5,30)),self.epics_connection_indicator]
        growable = [0,0,0,0,0,1,0]        #Makes the spacer panel the only one that can grow
        min_size = self.fcalculate_min_size(panel_components)
        for entry,prop in zip(panel_components,growable):
            local_sizer.Add(entry,prop)
            min_size[0] += entry.GetEffectiveMinSize()[0]
            if entry.GetEffectiveMinSize()[1] > min_size[1]:
                min_size[1] = entry.GetEffectiveMinSize()[1]
        print min_size
        #Try to set an appropriate size for the panel and return it
        local_panel.SetMinSize(min_size)
        local_panel.SetSizer(local_sizer)
        return local_panel
    
    def fmake_text_label(self,parent,id,label,total_size=(100,30),text_height=20,flag=wx.ALIGN_CENTER_HORIZONTAL,bg_color='Gray'):
        '''Uses a panel to fake a text label centered vertically.  This doesn't 
        seem to work properly with built in alignment flag of StaticText for unknown reasons.
        '''
        local_panel = wx.Panel(parent,id,size=total_size)
        local_panel.SetBackgroundColour(bg_color)
        local_sizer = wx.BoxSizer(wx.VERTICAL)
        #Compute size of the top and bottom panels.  Make it a minimum of zero.
        panel_v_size = (total_size[1]-text_height)/2 if total_size[1] > text_height else 0
        dummy1 = wx.Panel(local_panel,-1,size=(total_size[0],panel_v_size))
        local_text = GenStaticText(local_panel,-1,label,size=(total_size[0],text_height),style=flag)
        entries=[dummy1,local_text,dummy1]
        proportions = [1,0,1]
        for entry,prop in zip(entries,proportions):
            entry.SetBackgroundColour(bg_color)
            local_sizer.Add(entry,prop,wx.EXPAND)
        local_panel.SetSizer(local_sizer)
        return local_panel
    
    def fmake_right_panel(self,parent):
        '''Make the right-hand panel, which is a list of points currently confirmed.
           Make a table (Grid in wxPython parlance) for confirmed and current points.
        '''
        local_panel = wx.Panel(parent,-1)
        #Make a grid to hold the points, as well as a data model backing up the table.        
        grid = wx.grid.Grid(local_panel)
        self.table = PointsGridTable(model,grid)
        grid.SetTable(self.table,True)
        #Lay out panel, including 
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(grid, 1, wx.EXPAND)
        #Size the columns and set the sizer
        grid.AutoSize()
        local_panel.SetSizer(sizer)
        return local_panel

    def fmake_center_bottom_panel(self,parent):
        '''Make a grid for the history of confirmed points.
        '''
        local_panel = wx.Panel(parent,-1)
        grid = wx.grid.Grid(local_panel)
        self.history_table = HistoryGridTable(model,grid)
        grid.SetTable(self.history_table,True)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(grid, 1, wx.EXPAND)
        #Size the columns and set the sizer
        grid.AutoSize()
        local_panel.SetSizer(sizer)
        local_panel.SetMinSize((400,100))
        return local_panel
    
    def fmake_center_panel(self,parent,bg_color='Gray'):
        '''Make the center portion of the screen: EPICS PV bar and plot.
        '''
        local_panel = wx.Panel(parent,-1,style=wx.SIMPLE_BORDER)
        local_panel.SetBackgroundColour(bg_color)
        local_sizer = wx.BoxSizer(wx.VERTICAL)
        local_sizer.Add(self.fmake_epics_pv_panel(local_panel,bg_color),0,wx.EXPAND)
        self.points_plot = wxmplot.PlotPanel(local_panel,size=(200,200))
        local_sizer.Add(self.points_plot,1,wx.EXPAND)
        local_sizer.Add(self.fmake_center_bottom_panel(local_panel))
        local_panel.SetSizer(local_sizer)
        return local_panel

    def fdisplay_dialog(message,title="Alert"):
        '''Displays a dialong box with a given message.
        '''
        wx.MessageBox(self,message,title,style=wx.OK)

    def fcalculate_min_size(self,widgets):
        '''Takes a list of widgets and calculates the minimum size
        of the largest widget.  Used for sizing panels so everything fits.
        '''
        min_size = [0,0]
        for entry in widgets:
            if entry.GetEffectiveMinSize()[0] > min_size[0]:
                min_size[0] = entry.GetEffectiveMinSize()[0]
            if entry.GetEffectiveMinSize()[1] > min_size[1]:
                min_size[1] = entry.GetEffectiveMinSize()[1]
        return min_size

class PointsGridTable(wx.grid.PyGridTableBase):  
    '''Grid table class for table of existing and confirmed points.
    '''
    def __init__(self,model,grid):
        wx.grid.PyGridTableBase.__init__(self)
        self.grid = grid
        self.model = model
        self.col_labels = ['Confirmed','Current']
        self.finitialize_points()
        
    def finitialize_points(self):
        self.confirmed_points = self.model.getConfirmedPoints()
        self.current_points = self.model.getCurrentPoints()
    
    def fupdate_points(self):
        #Get the new number of confirmed and current points
        old_num_pts = self.GetNumberRows() 
        self.finitialize_points()          
        new_num_rows = self.GetNumberRows()
        #For added or removed rows, make a GridTableMessage and process
        if new_num_rows > old_num_pts:
            msg = wx.grid.GridTableMessage(self,wx.grid.GRIDTABLE_NOTIFY_ROWS_APPENDED,new_num_rows-old_num_pts)
            self.grid.ProcessTableMessage(msg)
        elif new_num_rows < old_num_pts:
            msg = wx.grid.GridTableMessage(self,wx.grid.GRIDTABLE_NOTIFY_ROWS_DELETED,new_num_rows,old_num_pts-new_num_rows)
            self.grid.ProcessTableMessage(msg)
        
        #Suggestion I saw online.  Slightly adjust table size to cause scrollbars to update.
        h,w = self.grid.GetSize()
        self.grid.SetSize((h+1, w))
        self.grid.SetSize((h, w))
        self.grid.ForceRefresh()
        
    def GetNumberCols(self):
        return 2
    
    def GetNumberRows(self):
        return len(self.confirmed_points) if len(
                self.confirmed_points)>len(self.current_points) else len(self.current_points)
    
    def IsEmptyCell(self,row,column):
        return len(self.GetValue(row,column))>0
    
    def SetValue(self):
        pass
    
    def GetValue(self,row,col):
        if col==0:
            if row < len(self.confirmed_points): 
                temp = str(self.confirmed_points[row])
                return temp if len(temp) < 8 else temp[:8]
            else:
                return ""
        elif col == 1:
            if row < len(self.current_points): 
                temp = str(self.current_points[row])
                return temp if len(temp) < 8 else temp[:8]
            else: 
                return ""
        else: 
            return ""
        
    def GetColLabelValue(self,col):   
        return self.col_labels[col]
    
    def AppendRows(self,numRows=1):
        return self.grid.AppendRows(numRows)
    
    def AppendCols(self,numCols=1):
        return self.grid.AppendCols(numCols)
    
    def InsertRows(self,pos=0,numRows=1):
        return self.grid.InsertRows(pos,numRows)
    
    def InsertCols(self,pos=0,numCols=1):
        return self.grid.InsertCols(pos,numCols)
    
    def DeleteRows(self,pos=0,numRows=1):
        return self.grid.DeleteRows(pos,numRows)
    
    def DeleteCols(self,pos=0,numCols=1):
        return self.grid.DeleteCols(pos,numCols)
    
    def Clear(self):
        return self.grid.ClearGrid()

class HistoryGridTable(wx.grid.PyGridTableBase):  
    '''Grid table class for table of existing and confirmed points.
    '''
    def __init__(self,model,grid):
        wx.grid.PyGridTableBase.__init__(self)
        self.grid = grid
        self.model = model
        self.col_labels = ['Start','End','Spacing','Num Pts']
        self.finitialize_points()
        
    def finitialize_points(self):
        self.history_dict = deepcopy(model.history_dict)
        #self.history_dict = model.history_dict.copy()
    
    def fupdate_points(self):
        #Get the new number of confirmed and current points
        old_num_pts = self.GetNumberRows() 
        self.finitialize_points()          
        new_num_rows = self.GetNumberRows()
        if new_num_rows > old_num_pts:
            msg = wx.grid.GridTableMessage(self,wx.grid.GRIDTABLE_NOTIFY_ROWS_APPENDED,new_num_rows-old_num_pts)
            self.grid.ProcessTableMessage(msg)
        elif new_num_rows < old_num_pts:
            msg = wx.grid.GridTableMessage(self,wx.grid.GRIDTABLE_NOTIFY_ROWS_DELETED,new_num_rows,old_num_pts-new_num_rows)
            self.grid.ProcessTableMessage(msg)
        
        #Suggestion I saw online.  Slightly adjust table size to cause scrollbars to update.
        h,w = self.grid.GetSize()
        self.grid.SetSize((h+1, w))
        self.grid.SetSize((h, w))
        self.grid.ForceRefresh()
        
    def GetNumberCols(self):
        return len(self.col_labels)
    
    def GetNumberRows(self):
        return len(self.history_dict[self.col_labels[0]])
    
    def IsEmptyCell(self,row,column):
        return column >= len(self.col_labels) or row > len(self.history_dict[self.col_labels[0]])
    
    def SetValue(self):
        pass
    
    def GetValue(self,row,col):
        return str(self.history_dict[self.col_labels[col]][row])
        
    def GetColLabelValue(self,col):   
        return self.col_labels[col]
    
    def AppendRows(self,numRows=1):
        return self.grid.AppendRows(numRows)
    
    def AppendCols(self,numCols=1):
        return self.grid.AppendCols(numCols)
    
    def InsertRows(self,pos=0,numRows=1):
        return self.grid.InsertRows(pos,numRows)
    
    def InsertCols(self,pos=0,numCols=1):
        return self.grid.InsertCols(pos,numCols)
    
    def DeleteRows(self,pos=0,numRows=1):
        return self.grid.DeleteRows(pos,numRows)
    
    def DeleteCols(self,pos=0,numCols=1):
        return self.grid.DeleteCols(pos,numCols)
    
    def Clear(self):
        return self.grid.ClearGrid()
    
app = MyApp(False)
app.MainLoop()
