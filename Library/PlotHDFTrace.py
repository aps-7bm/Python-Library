'''Script to plot the first part of an HDF file trace.
Alan Kastengren, XSD
March 28, 2014
'''
import h5py
import matplotlib.pyplot as plt



filename = 'Scan_1177.hdf5'
coord_name = 'X=-0.7000, Y=20.000205,'
filename = 'Scan_1189.hdf5'
coord_name = 'X=-0.4990, Y=20.000205,'

filepath = '/data/Data/SprayData/Cycle_2014_1/Radke/'
num_points = 60000

data_file = h5py.File(filepath+filename,'r')

variable_name = 'PINDiode'
trace = data_file.get(coord_name).get(variable_name).get(variable_name)
plt.plot(trace[:num_points])
plt.show()