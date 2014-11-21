'''Script to cover the main workflow for one data file. 
Assumption is that the converted hdf file from the DAQ is available.
Will cover radiography analysis, corrections for dead time and 
primary beam absorption, and deconvolving fluorescence.

Alan Kastengren, XSD
Started: November 15, 2013
'''
#Imports
import numpy as np
import h5py
import ALK_Utilities as ALK
import math
import os.path

#Important variables needed everywhere
path='/data/Data/SprayData/Cycle_2014_1/ISU_Point/'
base_names = ['Zn','Cu','Ni','Co','Elastic']
branching_ratios = [17.0/151.0]*5
K_alpha_energies = [8.631,8.04,7.472,6.925,10.5]
K_beta_energies = [9.572,8.905,8.265,7.649,'']
K_alpha_names = ['Raw_Zn','Raw_Cu','Raw_Ni','Raw_Co','Elastic']
K_beta_names = ['Raw_ZnKb','Raw_Zn','Raw_Cu','Raw_Ni','']
air_thickness_mm = 200
kapton_thickness_um = 25
Be_thickness_um = 25
Si_thickness_um = 0.1
Si_total_thickness_um = 400
Si_abs_coeff = 7.74e-3 #in 1/um

def fdefine_element(base_name,K_alpha_energy,K_alpha_name,K_beta_energy,K_beta_name,branching_ratio):
    '''Add a dictionary with properties of an element to the global list.
    '''
    temp_element = {}
    temp_element["Base_Name"] = base_name
    temp_element["K_alpha_energy"] = K_alpha_energy
    temp_element["K_beta_energy"] = K_beta_energy
    temp_element["K_alpha_name"] = K_alpha_name
    temp_element["K_beta_name"] = K_beta_name
    temp_element["Branching_ratio"] = branching_ratio
    return temp_element

def fcompute_ref_ext_lengths(air_pathlength_mm,kapton_pathlength_um,Be_pathlength_um,dead_layer_um):
    '''Gives the number of extinction lengths between the sample and the detector
    at an energy of 10 keV.  Used for correcting fixed absorption between sample
    and detector.
    '''
    #Compute absorption due to air
    air_abs_coeff = 5.12 #in cm^2/g, by NIST X-ray Mass Coefficients Table
    air_density = 1.173 #kg/m^3, assuming 300 K and 1 atm pressure
    air_abs_coeff = air_density/1000.*air_abs_coeff #in 1/cm
    ref_ext_lengths = air_pathlength_mm*air_abs_coeff/10.
    #Compute Kapton absorption
    kapton_attenuation_length = 2348.04 #in um, from CRXO X-Ray Interactions with Matter
    ref_ext_lengths += kapton_pathlength_um/kapton_attenuation_length
    #Add the absorption of the beryllium window
    Be_abs_coeff = 7.807e-5
    ref_ext_lengths += Be_abs_coeff * Be_pathlength_um
    #Add in the absorption of the dead layer.
    Si_abs_coeff = 7.74e-3 #in 1/um
    ref_ext_lengths += Si_abs_coeff * dead_layer_um
    return ref_ext_lengths

def fmake_dataset_energies_list(element_list):
    '''Makes a list of dictionary giving the energy associated with
    a variable (ROI of the MCA) in keV, the name of the dataset, and
    the elements for which this is K_alpha or K_beta.
    Assumes the K-alpha energy of an element for an ROI is a good
    representative for the line energy (K-beta for Z-1 is a little
    different, but probably a minor component anyway).
    '''
    #Make a list of all of the relevant K_alpha lines
    lines_list = []
    for element in element_list:
        lines_list.append({'Energy':element['K_alpha_energy'],'Root_Name':element['K_alpha_name'],
                           'K_alpha_element':element['Base_Name']})
    #Now, loop through the K-betas.  If a line isn't already in list, add it.
    for element in element_list:
        if not element['K_beta_name']:
            break
        for line in lines_list:
            if line['Root_Name'] == element['K_beta_name']:
                line['K_beta_element'] = element['Base_Name']
                break
        else:
            lines_list.append({'Energy':element['K_beta_energy'],'Root_Name':element['K_beta_name'],
                               'K_beta_element':element['Base_Name']})
    #Make a current name in each line that we will track.
    for line in lines_list:
        line['Current_Name'] = line['Root_Name']
    #Sort so we are in ascending order and return
    return sorted(lines_list,key=lambda k:k['Energy'])

def fcorrect_absorption_of_fluorescence(data_file,lines_list,elements_list,ref_ext_lengths,
                                        suffix='_Static_Abs_Corrected',spray_ext_lengths=None):
    '''Corrects fluorescence for the static absorption between the beam
    and the detector.  Give the absorption at 10 keV as the ref_ext_lengths.
    Include option to account for absorption by spray in extinction lengths.
    '''
    #Loop through fluorescence ROIs.
    for line in lines_list:
        #Handle case that this line isn't in the data file.
        if not data_file.get(line['Current_Name']):
            print "Dataset " + line['Current_Name'] + " does not exist in file.  Skipping."
            continue
        #Make an array for corrections either with or without spray ext lengths
        data_array = data_file[line['Current_Name']][...]
        correction_array = np.zeros_like(data_array)
        if spray_ext_lengths:
            correction_array += spray_ext_lengths
        correction_array += ref_ext_lengths * np.ones_like(correction_array)
        #Perform correction
        ext_lengths = (10.0/line['Energy'])**3*correction_array
        array = data_array / np.exp(-ext_lengths)
        #Update dataset name that the elements will point to.
        fupdate_element_line_name(elements_list,line['Current_Name'],line['Root_Name'] + suffix)
        #Add data to data file, including a bit of meta data
        line['Current_Name'] = line['Root_Name'] + suffix
        ALK.fwrite_HDF_dataset(data_file,line['Current_Name'],array,
                               {'Line_Energy':line['Energy'],'Ext_Lengths':(10.0/line['Energy'])**3*ref_ext_lengths,
                                'Static_Abs_Correction':'True' if not spray_ext_lengths else 'True+Spray'})

def fcorrect_detector_absorption(data_file,lines_list,elements_list,Si_thickness):
    '''Corrects fluorescence the non-unity absorption of the detector.
    Keeps the same dataset names as before.
    '''
    #Loop through fluorescence ROIs.
    for line in lines_list:
        #Handle case that this line isn't in the data file.
        if not data_file.get(line['Current_Name']):
            print "Dataset " + line['Current_Name'] + " does not exist in file.  Skipping."
            continue
        #Perform correction
        ext_lengths = (10.0/line['Energy'])**3*Si_abs_coeff*Si_thickness
        array = data_file[line['Current_Name']][...]/ (1-math.exp(-ext_lengths))
        #Add data to data file, including a bit of meta data
        ALK.fwrite_HDF_dataset(data_file,line['Current_Name'],array,
                               {'Detector_Absorption':(1-math.exp(-ext_lengths))},True)
        
def fupdate_element_line_name(elements_list,old_line_name,new_line_name):
    '''Updates the line names for the entries in the elements list.
    '''
    for element in elements_list:
        if element['K_alpha_name'] == old_line_name:
            element['K_alpha_name'] = new_line_name
        if element['K_beta_name'] == old_line_name:
            element['K_beta_name'] = new_line_name    

def fsubtract_K_betas(data_file,lines_list,elements_list,suffix='_K_Beta_Subtracted'):
    '''Subtracts K_beta of Z from K_alpha bin of Z+1 element.  Uses the
    branching ratio of each element.
    '''
    #Loop through lines when sorted in ascending order of energy
    for line in sorted(lines_list,key=lambda k:k['Energy']):
        #If there is no static corrected dataset for this line, exit.
        if not line['Current_Name'] or not data_file.get(line['Current_Name']):
            print "Problem in workflow: can't find proper line.  Skipping line."
            continue
        #If there is no K-beta element or no K-alpha for this line, no subtraction is needed.
        elif 'K_beta_element' not in line.keys() or 'K_alpha_element' not in line.keys():
            ALK.fwrite_HDF_dataset(data_file,line['Root_Name']+suffix,data_file[line['Current_Name']][...])
        #Main case: we have valid data with both K_alpha and K_beta.  We must subtract.
        else:
            data = data_file[line['Current_Name']][...]
            K_beta_contribution = np.zeros_like(data)
            #Loop through elements to find the one whose K-beta is here.  Subtract K_alpha * branching ratio
            for element in elements_list:
                if element['Base_Name'] == line['K_beta_element']:
                    K_beta_contribution = data_file[element['K_alpha_name']][...]*element['Branching_ratio']
                    break
            data -= K_beta_contribution
            #Update the dataset the element points to.  Combined with doing this correction
            #in ascending order of energy, we will be using subtracted K_alphas (which should be
            #right) for the K_beta subtraction.
            fupdate_element_line_name(elements_list,line['Current_Name'],line['Root_Name']+suffix)
            #Write new dataset to file
            line['Current_Name'] = line['Root_Name'] + suffix
            ALK.fwrite_HDF_dataset(data_file,line['Current_Name'],data,{'K_beta_Subtracted':True},True) 

def fcorrect_signal_trapping_fitted(hdf_file,lines_list,elements_list,stream_element_names,suffix='_Trapping_Corrected',
                                    ref_r0=0.0,fitting=True,degree=2,x_name='7bmb1:m26.VAL'):
    '''Perform signal trapping correction using the two elements listed in stream_element_names.
    If fitting=True (default), will perform a fit of degree degree to ratio between lines.
    '''
    #Find the actual lines that correspond to the K_alphas of the desired elements
    #in ascending order of K_alpha energy.
    lines_to_use,line_energies = ffind_lines_to_use(elements_list,stream_element_names)
    #Make sure that we actually have appropriate inputs
    if len(lines_to_use) != 2:
        print "Invalid input.  Need two lines to perform signal trapping correction.  Skipping."
        return
    #Find the exponent for the correction
    exponent = 1.0 / (1.0 - (line_energies[1]/line_energies[0])**3)
    print "Exponent = " + str(exponent)
    for line in lines_to_use:
        if not hdf_file.get(line):
            print "Line " + line + " not found in data file.  Exiting."
            return
    #Extract arrays for high and low Z materials
    high_Z_array = hdf_file.get(lines_to_use[1])[...]
    low_Z_array = hdf_file.get(lines_to_use[0])[...]
    #If either of these arrays is None, or sizes don't match, exit
    if np.size(high_Z_array) != np.size(low_Z_array) or np.any(np.isnan(high_Z_array)) or np.any(np.isnan(low_Z_array)):
        print "The arrays for the two fluorescence lines did not read in properly.  Skipping"
        return
    #Don't want to correct where the signal is nearly zero.  Set
    #threshold at 25% of way between max and min.  A bit abritrary, but better than 
    #previous method, which relied on the peak.
    threshold_ratio = 0.25
    threshold_value = np.max(high_Z_array)*threshold_ratio - np.min(high_Z_array)*(1-threshold_ratio)
    #Make a boolean array to mask the 
    above_threshold = high_Z_array>threshold_value
    #Form a ratios array
    ratio = np.ones_like(high_Z_array)
    ratio[above_threshold] = np.nan_to_num(low_Z_array[above_threshold] / high_Z_array[above_threshold])
    #If there is a reference value input, use it.  Otherwise, use max value we found before.
    r0 = ref_r0 if ref_r0 else np.max(ratio[above_threshold])
    print "Reference ratio between ROIs = " + str(r0)
    #Normalize ratio by this max value.  Only for values with a good amount of fluorescence
    ratio[above_threshold] /= r0
    #Perform correction
    if fitting:
        new_ratio = ffit_ratio(hdf_file[x_name][...],ratio,above_threshold,degree)
    else:
        new_ratio = ratio
    output_array = high_Z_array*new_ratio**exponent
    ALK.fwrite_HDF_dataset(hdf_file,lines_to_use[1]+suffix,output_array,{'Exponent':exponent,'Ref_Ratio':r0})
    ALK.fwrite_HDF_dataset(hdf_file,lines_to_use[1]+suffix+'_Raw_Ratio',ratio)
    ALK.fwrite_HDF_dataset(hdf_file,lines_to_use[1]+suffix+'_Fitted_Ratio',new_ratio)
    print "Line " + lines_to_use[1] + " processed for signal trapping successfully."
    
def ffit_ratio(x,ratio,mask=None,degree=2):
    '''Take the ratio points and perform a least-squares fit.
    Experience shows that the raw ratio can be too noisy.  Use the fit to make
    the results cleaner.
    Mask is a boolean numpy mask array to show the points to be fit.
    Default is none, which will cause all points to be fit.
    '''
    #If mask is None, make it to allow all values to be permitted
    if mask==None:
        mask = ratio != np.NAN
    #Perform fit on points allowed by mask
    fit_results = np.polyfit(x[mask],ratio[mask],degree)
    #Form a new ratio formed by the linearized results from the input.  Return this.
    new_ratio = np.ones_like(ratio)
    new_ratio[mask] = 0
    for j in range(degree+1):
        new_ratio[mask] = new_ratio[mask] * x[mask] + fit_results[j] 
    return new_ratio

def ffind_lines_to_use(elements_list,stream_element_names):
    '''Find the two datasets we will use.  Using K_alpha lines only.
    '''
    #Fist, sort the elements_list by K_alpha energy
    sorted_elements = sorted(elements_list,key=lambda k:k['K_alpha_energy'])
    lines_to_use = []
    lines_energies = []
    for entry in stream_element_names:
        for element in sorted_elements:
            if element['Base_Name'] == entry:
                lines_to_use.append(element['K_alpha_name'])
                lines_energies.append(element['K_alpha_energy'])
                break
    return lines_to_use,lines_energies

def ffile_workflow(file_name,file_path=path):
    '''Main workflow for a data file.
    '''
    print file_name
    if not os.path.isfile(file_path+file_name):
        print "File " + file_name + " does not exist.  Skipping."
        return
    with h5py.File(file_path+file_name,'r+') as hdf_file:
        #Make the elements list with the parameters at the top of the file.
        elements_list = []
        for a,b,c,d,e,f in zip(base_names,K_alpha_energies,K_alpha_names,K_beta_energies,
                               K_beta_names,branching_ratios):
            elements_list.append(fdefine_element(a,b,c,d,e,f))
        #Make the list of lines that we will use
        lines_list = fmake_dataset_energies_list(elements_list)
        #Find the number of static extinction lengths
        ref_ext_lengths = fcompute_ref_ext_lengths(air_thickness_mm,kapton_thickness_um,Be_thickness_um,Si_thickness_um)
        fcorrect_absorption_of_fluorescence(hdf_file,lines_list,elements_list,ref_ext_lengths,
                                        suffix='_Static_Abs_Corrected',spray_ext_lengths=None)
        fcorrect_detector_absorption(hdf_file,lines_list,elements_list,Si_total_thickness_um)
        fsubtract_K_betas(hdf_file,lines_list,elements_list,suffix='_K_Beta_Subtracted')
        fcorrect_signal_trapping_fitted(hdf_file,lines_list,elements_list,['Ni','Zn'],
                                    ref_r0=0.0,fitting=True,degree=2,x_name='7bmb1:m26.VAL')
        fcorrect_signal_trapping_fitted(hdf_file,lines_list,elements_list,['Co','Cu'],
                                    ref_r0=0.0,fitting=True,degree=2,x_name='7bmb1:m26.VAL')
    

if __name__ == '__main__':
    for i in range(326,983):
        ffile_workflow('7bmb1_0'+str(i)+'.hdf5')
        