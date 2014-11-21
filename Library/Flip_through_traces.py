import ALK_Utilities as ALK
file_path = ALK.fcorrect_path_start() + 'SprayData/Cycle_2014_2/Taitech/'
file_nums = range(1319,1329)
filename_list = ALK.fcreate_filename_list(file_nums, '7bmb1_', '.hdf5', 4, '', False)
for filename in filename_list:
    ALK.fplot_HDF_trace(file_path, filename, ['Radiography'], ['7bmb1:m26.VAL'])