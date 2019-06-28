# -*- coding: utf-8 -*-
"""
Created on Fri May 08 10:50:55 2015

@author: Andrew B. Swantek, May 2015
Email:aswantek@anl.gov or andrew.b.swantek@gmail.com

Edits: Katie Matusik
#series of abel transforms from the PyAbel Python module
"""

import numpy as np
import matplotlib.pyplot as plt
import abel
import os
import copy
from scipy.interpolate import interp1d

class Abelject:

    """This is going to be a class which will contain the data and have methods for performing the inverse abel transforms.

       Abel Transform + Object = Abelject

    Attributes
    --------
    y : numpy.array or python list
        y coordinates - needs to be given
    P : numpy.array or python list
        projection data - needs to be given  \n
    dr : float
        grid spacing in units - needs to be given  \n
    r : numpy.array
        r coordinates which are None until a transform has taken place  \n
    R : float
        maximum value from y, i.e. outer radius
    F : numpy.array
        field data from Abel inversion  \n
    method : Different methods of computing the inverse Abel transform
    MethodTypes : list
        List of method types that can currently be used
    MethodNames : Dictionary
        Dictionary of method names (clean for plotting) are indexed by elements in MethodTypes
    Ny, Nz : ints
        Number of points used for 2-D interpolation, default to **201** for both Attributes
    OV : float
        The max radius for 2-D interpolated plots
    YY, ZZ : numpy.arrays()
        Contain the Y and Z coordinates where the transform will be projected
    MM : 2D numpy array
        Contains the inverted abel data which has been turned into a 2D axisymmetric contour plot

    Methods
    --------
    D_ij_construct : function
        Builds the 2-D D_ij matrix based on the method in Abelject.method
    Onion, TwoPoint, ThreePoint, ThreePointModified: functions
        Called by D_ij_construct, loop over D_ij numpy array and build it up
    J I0 I1 : functions()
        Used by Onion, Two Point, and ThreePoint functions for caclulating indivudal components of the matricies
    reconstruct : function
        Performs the reconstruction based on Equation 1 in Dasch
    abel_inversion : function
        Wrapper function which does all steps for inversion
    """


    def __init__(self, Y, P_y,rmethod='ThreePoint',basis_dir=None,nint=20):
        self.y=np.array(Y)
        self.P=np.array(P_y)
        self.dr=Y[1]-Y[0]
        self.R=np.max(Y)
        self.newy = None
        self.newP = None
        self.r=None
        self.F=None
        self.nint = nint
        self.size=len(self.y)
        self.basis_dir = basis_dir
        self.method=rmethod
        self.MethodTypes=['Onion_Peeling','TwoPoint','ThreePoint','Hansenlaw','Basex','Direct']
        self.MethodNames={'Onion_Peeling':'Onion Peeling','TwoPoint':'Two-Point',
                          'ThreePoint':'Three-Point','Hansenlaw':'Hansen and Law','Basex':'BAsis Set EXpansion',
                          'Direct':'Direct'}
        self.Ny=201
        self.Nz=201
        self.OV=self.R
        self.YY=None
        self.ZZ=None
        self.MM = None

    def F_construct(self):

        #use this as a pass through to construct D_ij based on the reconstruction methods
        if self.method == 'Onion_Peeling':
            self.F =self.Onion_Peeling()
        if self.method == 'TwoPoint':
            self.F = self.TwoPoint()
        if self.method == 'ThreePoint':
            self.F=self.ThreePoint()
        if self.method == 'Hansenlaw':
            self.F=self.Hansenlaw()
        if self.method == 'Basex':
            self.F=self.Basex()
        if self.method == 'Direct':
            self.F=self.Direct()


    def func_prime(self):
    #prepare the profile for Abel inversion. Create a right-side image 
        y = self.P
        x = self.y
    #find center of mass of the curve
        comx = (1/np.sum(y))*np.sum(x*y)
        comy = np.interp(comx,x,y)
    #find points to the left and right of the center of mass
        leftpts = x<=comx
        rightpts = x>=comx
        xleft_temp = x[leftpts]
        xright_temp = x[rightpts]
    #add the center of mass to both of these arrays
        xleft_uf=np.zeros(len(xleft_temp)+1)
        xright = np.zeros(len(xright_temp)+1)
        xleft_uf[:len(xleft_temp)]=xleft_temp
        xleft_uf[-1]=comx
        xright[0]=comx
        xright[1:len(xright_temp)+1]=xright_temp
    #center so that center of mass is at x = 0
        xright-=comx
        yleft_uf = np.zeros(len(xleft_uf))
        yleft_uf[:len(xleft_temp)]=y[leftpts]
        yleft_uf[-1]=comy
        yright =  np.zeros(len(xright))
        yright[0]=comy
        yright[1:len(xright_temp)+1]=y[rightpts]           
    #Mirror lefthand points 
    #center of mass is now at x = 0
        xleft = np.flipud(np.abs(xleft_uf+comx))
        yleft = np.flipud(yleft_uf)
    #since these are no longer sampled on the same x, make new x and interpolate 
        xnew = np.linspace(0,min(max(xleft),max(xright)),self.nint)
        sleft = interp1d(xleft,yleft,kind='cubic',fill_value='extrapolate')
        yileft = sleft(xnew)
        sright = interp1d(xright,yright,kind='cubic',fill_value='extrapolate')
        yiright = sright(xnew[:len(yright)])
    #average the two arrays. Whichever one is shorter, fill the entries with nan
    #until it reaches the length of the longer array
        if len(yileft)<len(yiright):
            yr = copy.deepcopy(yiright)
            minpt = len(yileft)
            yl = np.zeros(len(yiright))
            yl[:minpt]=yileft
            yl[minpt:]=np.nan
        elif len(yileft)>len(yiright):
            yl = copy.deepcopy(yileft)
            minpt = len(yiright)
            yr = np.zeros(len(yileft))
            yr[:minpt]=yiright
            yr[minpt:]=np.nan
        else:
            yl = copy.deepcopy(yileft)
            yr = copy.deepcopy(yiright)   
    #take average of two arrays        
        ynew = np.nanmean(np.array([yl,yr]),axis=0)
    #add zeros on RHS of array for Abel inversion, 30% of array length
        ylen = int(len(ynew)*0)
        ypad = np.zeros(len(ynew)+ylen)
        ypad[:len(ynew)]=ynew
        ypad[len(ynew):]=0
        xdiff = xnew[1]-xnew[0]
        xpad = np.zeros(len(ypad))
        xpad[:len(xnew)]=xnew
        xpad[len(xnew):]=np.linspace(xnew[-1]+xdiff,xnew[-1]+(xdiff*ylen),ylen)
        self.newy = xpad
        self.newP = ypad
        self.dr = xpad[1]-xpad[0]
        
#        self.newy = copy.deepcopy(x)
#        self.newP = copy.deepcopy(y)
#        self.dr = x[1]-x[0]
    ################################################################################################
    ############# CALCULATION METHODS FOR THE ABEL RECONSTRUCTION ##################################
    ################################################################################################

    def Onion_Peeling(self):
        if self.basis_dir is not None and os.path.isdir(self.basis_dir)==0:
            os.makedirs(self.basis_dir)
        self.F = abel.dasch.onion_peeling_transform(self.newP,dr=self.dr,direction="inverse",basis_dir=self.basis_dir)
        return self.F
    ################################################################################################
    def TwoPoint(self):
        if self.basis_dir is not None and os.path.isdir(self.basis_dir)==0:
            os.makedirs(self.basis_dir)
        self.F = abel.dasch.two_point_transform(self.newP,dr=self.dr,direction="inverse",basis_dir=self.basis_dir)
        return self.F
    ################################################################################################
    def ThreePoint(self):
        if self.basis_dir is not None and os.path.isdir(self.basis_dir)==0:
            os.makedirs(self.basis_dir)
        self.F = abel.dasch.three_point_transform(self.newP,dr=self.dr,direction="inverse",basis_dir=self.basis_dir)
        return self.F
    ################################################################################################
    def Hansenlaw(self):
        self.F =  abel.hansenlaw.hansenlaw_transform(self.newP,dr=self.dr,direction="inverse")
        return self.F
    ################################################################################################
    def Basex(self):
        if self.basis_dir is not None and os.path.isdir(self.basis_dir)==0:
            os.makedirs(self.basis_dir)
        self.F = abel.basex.basex_transform(self.newP,dr=self.dr,direction="inverse",basis_dir=self.basis_dir)
        return self.F
    ################################################################################################
    def Direct(self):
        self.F = abel.direct.direct_transform(self.newP,dr=self.dr,direction="inverse")
        return self.F

    ################################################################################################
    ################# MAIN RUN FUNCTION ############################################################
    ################################################################################################
    def abel_inversion(self):
        """This does the inversion without having to call each individual function
        """
        self.func_prime()
        self.F_construct()
        self.r=self.newy

    ################################################################################################
    ################# GRIDDING AND 2D PLOTTING  FUNCTIONS ##########################################
    ################################################################################################

    def make_2D_grid(self):
        #define here in case we change OV before hand
        self.MM= np.zeros((self.Ny,self.Nz),dtype=np.float)
        self.YY=np.linspace(-1.0*self.OV,self.OV,self.Ny)
        self.ZZ=np.linspace(-1.0*self.OV,self.OV,self.Nz)
        if isinstance(self.F,np.ndarray):
            #loop over rows
            for i in range(self.MM.shape[0]):
            #loop over columns
                for j in range(self.MM.shape[1]):
                    # calculate radius
                    r_local=np.sqrt(self.YY[i]**2+self.ZZ[j]**2)
                    #check if r exceeds the maximum r of our data
                    if r_local>self.R:
                        #hardwire to zero
                        self.MM[i,j]=0.0
                    else:


                        self.MM[i,j]=np.interp(r_local,self.r,self.F)
        else:
            print "You need to do the reconstruction before you can plot!"


################################################################################################
################# RUNS IF FUNCTION IS MAIN, AS AN EXAMPLE ######################################
################################################################################################
if __name__=='__main__':
    """ Run two test cases so the user can see how the class is used.
        One will be an ellipse, the other a gaussian
    """
    plt.close('all')
    ################################################################################################
    ################# Ellipse Benchmark ############################################################
    ################################################################################################

    ###### Create Elliptic Function and it's Projection ######
    x1=np.linspace(-1,1,50)
    y=2*np.sqrt(1-x1[x1<1]**2)

    y1=np.zeros(len(x1)) #keeping y for later indexing
    y1[0:len(y)]=y

    #analytic inversion
    y1_I=np.zeros(len(x1))
    y1_I[0:len(y)]=1

##TROUBLE SHOOTING BELOW ###
#    #find center of mass of the curve
#    comx = (1/np.sum(y))*np.sum(x1*y)
#    comy = np.interp(comx,x1,y)
##find points to the left and right of the center of mass
#    leftpts = x1<=comx
#    rightpts = x1>=comx
#    xleft_temp = x1[leftpts]
#    xright_temp = x1[rightpts]
##add the center of mass to both of these arrays
#    xleft_uf=np.zeros(len(xleft_temp)+1)
#    xright = np.zeros(len(xright_temp)+1)
#    xleft_uf[:len(xleft_temp)]=xleft_temp
#    xleft_uf[-1]=comx
#    xright[0]=comx
#    xright[1:len(xright_temp)+1]=xright_temp
##center so that center of mass is at x = 0
#    xright-=comx
#    yleft_uf = np.zeros(len(xleft_uf))
#    yleft_uf[:len(xleft_temp)]=y[leftpts]
#    yleft_uf[-1]=comy
#    yright =  np.zeros(len(xright))
#    yright[0]=comy
#    yright[1:len(xright_temp)+1]=y[rightpts]           
##Mirror lefthand points 
##center of mass is now at x = 0
##    xleft = (xleft_uf-np.min(xleft_uf))
#    xleft = np.flipud(np.abs(xleft_uf+comx))
#    yleft = np.flipud(yleft_uf)
##since these are no longer sampled on the same x, make new x and interpolate 
#    xnew = np.linspace(0,min(max(xleft),max(xright)),100)
#    sleft = interp1d(xleft,yleft,kind='cubic',fill_value='extrapolate')
#    yileft = sleft(xnew)
#    sright = interp1d(xright,yright,kind='cubic',fill_value='extrapolate')
#    yiright = sright(xnew[:len(yright)])
##average the two arrays. Whichever one is shorter, fill the entries with nan
##until it reaches the length of the longer array
#    if len(yileft)<len(yiright):
#        yr = copy.deepcopy(yiright)
#        minpt = len(yileft)
#        yl = np.zeros(len(yiright))
#        yl[:minpt]=yileft
#        yl[minpt:]=np.nan
#    elif len(yileft)>len(yiright):
#        yl = copy.deepcopy(yileft)
#        minpt = len(yiright)
#        yr = np.zeros(len(yileft))
#        yr[:minpt]=yiright
#        yr[minpt:]=np.nan
#    else:
#        yl = copy.deepcopy(yileft)
#        yr = copy.deepcopy(yiright)   
##take average of two arrays        
#    ynew = np.nanmean(np.array([yl,yr]),axis=0)
##add zeros on RHS of array for Abel inversion, 30% of array length
#    ylen = int(len(ynew)*.3)
#    ypad = np.zeros(len(ynew)+ylen)
#    ypad[:len(ynew)]=ynew
#    ypad[len(ynew):]=0
#    xdiff = xnew[1]-xnew[0]
#    xpad = np.zeros(len(ypad))
#    xpad[:len(xnew)]=xnew
#    xpad[len(xnew):]=np.linspace(xnew[-1]+xdiff,xnew[-1]+(xdiff*ylen),ylen)
#    
#    fig1,ax1=plt.subplots(figsize=(11,10))
#    ax1.plot(x1,y1,'-b',lw=2,label='Original Ellipse Function, f(x)') #plot analytic function
#    ax1.plot(x1,y1_I,'-r',lw=2,label='Analytic Abel Inversion, f(r)' ) # plot analytic abel inversion
#    ax1.plot(xpad,ypad,'--ok',label='New function', ms=10)

##END TROUBLE SHOOTING ###
 
    
    ################################################################
    #create list to hold error for each abel method.
    Abel_err1=[]

    #numerical inversion
    AbelTest1=Abelject(x1,y1,rmethod='Onion_Peeling',basis_dir='test')
    AbelTest1.abel_inversion()
#    Abel_err1.append(np.abs(AbelTest1.F-y1_I))

    #make a figure
    fig1,ax1=plt.subplots(figsize=(11,10))
    ax1.plot(x1,y1,'-b',lw=2,label='Original Ellipse Function, f(x)') #plot analytic function
    ax1.plot(x1,y1_I,'-r',lw=2,label='Analytic Abel Inversion, f(r)' ) # plot analytic abel inversion
    ax1.plot(AbelTest1.r,AbelTest1.F,'--ok',label='Onion Peeling Method', ms=10)

    #other numerical inversions
    AbelTest1=Abelject(x1,y1,rmethod='TwoPoint',basis_dir='test')
    AbelTest1.abel_inversion()
    ax1.plot(AbelTest1.r,AbelTest1.F,'--oc',label='Two Point Method', ms=10)
#    Abel_err1.append(np.abs(AbelTest1.F-y1_I)) #append to error list

    AbelTest1=Abelject(x1,y1,rmethod='ThreePoint',basis_dir='test')
    AbelTest1.abel_inversion()
    ax1.plot(AbelTest1.r,AbelTest1.F,'--om',label='Three Point Method', ms=10)
#    Abel_err1.append(np.abs(AbelTest1.F-y1_I)) #append to error list

    AbelTest1.method='Hansenlaw'
    AbelTest1.abel_inversion()
    ax1.plot(AbelTest1.r,AbelTest1.F,'--og',label='Hansen and Law', ms=10)
#    Abel_err1.append(np.abs(AbelTest1.F-y1_I)) #append to error list

    AbelTest1=Abelject(x1,y1,rmethod='Basex',basis_dir='test')
    AbelTest1.abel_inversion()
    ax1.plot(AbelTest1.r,AbelTest1.F,'--oy',label='Basex', ms=10)
#    Abel_err1.append(np.abs(AbelTest1.F-y1_I)) #append to error list

    AbelTest1.method='Direct'
    AbelTest1.abel_inversion()
    ax1.plot(AbelTest1.r,AbelTest1.F,'--ob',label='Direct', ms=10)
#    Abel_err1.append(np.abs(AbelTest1.F-y1_I)) #append to error list

    #Make plot look nice
    ax1.set_xlabel('x, r', fontsize=32)
    ax1.set_ylabel('f(x), f(r)', fontsize=32)
    plt.xticks(size=23)
    plt.yticks(size=23)
    plt.title('Elliptic Function Test',fontsize=36,y=1.01)
    plt.grid(b='on',lw=2)
    ax1.tick_params(axis='both', pad = 10,labelsize=32)
    ax1.set_ylim(0,2.5)
    plt.legend(fontsize=22)
#%%
    ################################################################################################
    ################# Gaussian Benchmark ###########################################################
    ################################################################################################

    ###### Create Gaussian and it's Projection ######
    x2=np.linspace(-3,3,200)
    sig=2/np.sqrt(np.pi)
    y2=sig*np.sqrt(np.pi)*np.exp(-x2**2/sig**2)
    #analytic inversion
    y2_I=np.exp(-x2**2/sig**2)
    ###################################################
    #create list to hold error for each abel method.
    Abel_err2=[]

    #numerical inversion
    AbelTest2=Abelject(x2,y2,rmethod='Onion_Peeling')
    AbelTest2.abel_inversion()
#    Abel_err2.append(np.abs(AbelTest2.F-y2_I))

    #make a figure
    fig2,ax2=plt.subplots(figsize=(11,10))
    ax2.plot(x2,y2,'-b',lw=2,label='Original Gaussian Function, f(x)') #plot analytic function
    ax2.plot(AbelTest2.r,AbelTest2.newP,'-og',lw=2,label='Averaged Gaussian Function, f(x)',ms=10) #plot analytic function
    ax2.plot(x2,y2_I,'-r',lw=2,label='Analytic Abel Inversion, f(r)' ) # plot analytic abel inversion
    ax2.plot(AbelTest2.r,AbelTest2.F,'--ok',label='Onion Peeling Method', ms=10)

    #other numerical inversions
    AbelTest2.method='TwoPoint'
    AbelTest2.abel_inversion()
    ax2.plot(AbelTest2.r,AbelTest2.F,'--oc',label='Two Point Method', ms=10)
#    Abel_err2.append(np.abs(AbelTest2.F-y2_I)) #append to error list

    AbelTest2.method='ThreePoint'
    AbelTest2.abel_inversion()
    ax2.plot(AbelTest2.r,AbelTest2.F,'--om',label='Three Point Method', ms=10)
#    Abel_err2.append(np.abs(AbelTest2.F-y2_I)) #append to error list

    AbelTest2.method='Hansenlaw'
    AbelTest2.abel_inversion()
    ax2.plot(AbelTest2.r,AbelTest2.F,'--og',label='Hansen and Law', ms=10)
#    Abel_err2.append(np.abs(AbelTest2.F-y2_I)) #append to error list

    AbelTest2.method='Basex'
    AbelTest2.abel_inversion()
    ax2.plot(AbelTest2.r,AbelTest2.F,'--oy',label='Basex', ms=10)
#    Abel_err2.append(np.abs(AbelTest2.F-y2_I)) #append to error list

    AbelTest2.method='Direct'
    AbelTest2.abel_inversion()
    ax2.plot(AbelTest2.r,AbelTest2.F,'--ob',label='Direct', ms=10)
    Abel_err2.append(np.abs(AbelTest2.F-y2_I)) #append to error list

    #Make plot look nice
    ax2.set_xlabel('x, r', fontsize=32)
    ax2.set_ylabel('f(x), f(r)', fontsize=32)
    plt.xticks(size=23)
    plt.yticks(size=23)
    plt.title('Gaussian Function Test',fontsize=36,y=1.01)
    plt.grid(b='on',lw=2)
    ax2.tick_params(axis='both', pad = 10,labelsize=32)
    ax2.set_ylim(0,2.5)
    plt.legend(fontsize=15)

    plt.show()
#%%
    ################################################################################################
    ################# Gaussian w/Noise Benchmark ###################################################
    ################################################################################################

    ###### Create Gaussian and it's Projection ######
    x3=np.linspace(0,3,50)
    sig=2/np.sqrt(np.pi)
    y3n=sig*np.sqrt(np.pi)*np.exp(-x3**2/sig**2)

    ###
    ###Add in noise
    ###

    #set the random seed for repeatability
    np.random.seed(seed=1000)

    #add in 3% noise from the peak value of the Gaussian
    #need to subtract by 0.5 and mulitply by 2 to get interval from [0,1] to [-1,1], then multiply by
    #sig*np.sqrt(np.pi) to get the peak value
    #finally, multiply by 0.03 to get 3% of that
    y3=y3n+(np.random.random(size=len(y3n))-0.5)*2*sig*np.sqrt(np.pi)*0.03

    #analytic inversion(w/0 noise)
    y3_I=np.exp(-x3**2/sig**2)

    ###################################################
    Abel_err3=[]

    #numerical inversion
    AbelTest3=Abelject(x3,y3,dr=x3[1]-x3[0],rmethod='Onion_Peeling')
    AbelTest3.abel_inversion()
    Abel_err3.append(np.abs(AbelTest3.F-y3_I))

    #make a figure
    fig3,ax3=plt.subplots(figsize=(11,10))
    ax3.plot(x3,y3n,'-b',lw=2,label='Original Gaussian Function, f(x)') #plot analytic function
    ax3.plot(x3,y3,'--bo',lw=2,label='Gaussian Function w/ Noise, f(x)',mfc='w', ms=10,mew=3,mec='b')
    ax3.plot(x3,y3_I,'-r',lw=2,label='Analytic Abel Inversion, f(r)' ) # plot analtic abel inversion
    ax3.plot(AbelTest3.r,AbelTest3.F,'--ok',label='Onion Peeling Method', ms=10)

    #other numerical inversions
    AbelTest3.method='TwoPoint'
    AbelTest3.abel_inversion()
    ax3.plot(AbelTest3.r,AbelTest3.F,'--oc',label='Two Point Method', ms=10)
    Abel_err3.append(np.abs(AbelTest3.F-y3_I)) #append to error list

    AbelTest3.method='ThreePoint'
    AbelTest3.abel_inversion()
    ax3.plot(AbelTest3.r,AbelTest3.F,'--om',label='Three Point Method', ms=10)
    Abel_err3.append(np.abs(AbelTest3.F-y3_I)) #append to error list

    AbelTest3.method='Hansenlaw'
    AbelTest3.abel_inversion()
    ax3.plot(AbelTest3.r,AbelTest3.F,'--og',label='Hansen and Law', ms=10)
    Abel_err3.append(np.abs(AbelTest3.F-y3_I)) #append to error list

    AbelTest3.method='Basex'
    AbelTest3.abel_inversion()
    ax3.plot(AbelTest3.r,AbelTest3.F,'--oy',label='Basex', ms=10)
    Abel_err3.append(np.abs(AbelTest3.F-y3_I)) #append to error list

    AbelTest3.method='Direct'
    AbelTest3.abel_inversion()
    ax3.plot(AbelTest3.r,AbelTest3.F,'--ob',label='Direct', ms=10)
    Abel_err3.append(np.abs(AbelTest3.F-y3_I)) #append to error list

    #Make plot look nice
    ax3.set_xlabel('x, r', fontsize=32)
    ax3.set_ylabel('f(x), f(r)', fontsize=32)
    plt.xticks(size=23)
    plt.yticks(size=23)
    plt.title('Gaussian Function Test',fontsize=36,y=1.01)
    plt.grid(b='on',lw=2)
    ax3.tick_params(axis='both', pad = 10,labelsize=32)
    ax3.set_ylim(0,2.5)
    plt.legend(fontsize=22)

    plt.show()
 #%%
    ###### Cacluatled and print abel errors ##########################
    Fun_names=['Elliptic','Gaussian','Gaussian with noise']
    Ab_names=['Onion', 'Two-point', 'Three-point', 'Hansenlaw','Basex','Direct']
    Ab_errorList=[Abel_err1,Abel_err2,Abel_err3]

    print "Error calculations for previous plots:\n"
    for i,F_nm in enumerate(Fun_names):
        print F_nm
        for j,Ab_nm in enumerate(Ab_names):
            print '\t' +Ab_nm + ': ' + str( np.mean( Ab_errorList[i][j][ np.isnan(Ab_errorList[i][j])==False] ) )
            if i==0:
                print '\t\tNoise percentage, ' +Ab_nm + ': ' + str( np.mean( Ab_errorList[i][j][ np.isnan(Ab_errorList[i][j])==False] )/np.mean(y1_I)*100 )
            if i==1:
                print '\t\tNoise percentage, ' +Ab_nm + ': ' + str( np.mean( Ab_errorList[i][j][ np.isnan(Ab_errorList[i][j])==False] )/np.mean(y2_I)*100 )
            if i==2:
                print '\t\tNoise percentage, ' +Ab_nm + ': ' + str( np.mean( Ab_errorList[i][j][ np.isnan(Ab_errorList[i][j])==False] )/np.mean(y3_I)*100 )

        print '\n'


