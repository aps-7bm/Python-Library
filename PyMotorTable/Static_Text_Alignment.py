import wx
from wx.lib.stattext import GenStaticText

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
        wx.Frame.__init__(self,parent,title=title,size=(600,500))
        main_panel = self.fmake_main_panel(self)
        self.Layout()
        self.Show(True)
    
    def fmake_main_panel(self,parent):
        #Make an overall panel
        local_panel = wx.Panel(parent,-1)
        local_panel.SetBackgroundColour('Pink')
        local_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        #Make the subpanels
        test_text = wx.StaticText(local_panel,-1,'Test',size=(0,0),style=wx.ALIGN_LEFT|wx.ALIGN_CENTER_VERTICAL)
        sub_panels = [wx.Panel(local_panel,-1,size=(100,100)),test_text,wx.Panel(local_panel,-1,size=(100,100))]
        proportion_sub = [0,1,0]
        border_list = [5,5,0]
        #Make the sizer: use a BoxSizer for this one
        local_sizer = wx.BoxSizer(wx.VERTICAL)
        for (sub,prop,border_entry) in zip(sub_panels,proportion_sub,border_list):
            local_sizer.Add(sub,prop,flag=wx.RIGHT,border=border_entry)

        #Redraw panel and return it
        local_panel.SetSizer(local_sizer)
        local_panel.Layout()
        return local_panel

app = MyApp(False)
app.MainLoop()