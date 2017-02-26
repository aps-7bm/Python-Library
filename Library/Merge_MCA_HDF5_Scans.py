""" Module to merge sets of HDF5-converted MCA scans into a single set of 3-dimensional arrays.
Useful, for example, for merging a set of 1-D scans from a fluorescence detector into a single file.
The array record will end up being of size (N_outer (pyEpics loop), N_inner (scan1), N_array (scanH)).

Daniel Duke, ES
Started February 23, 2017

Changelog:  
    Feb 24, 2017: Added support for non matching array sizes to mergeDatasets & make3dMCAArrays.
                  It pads with NaN on the right side of the array.
                  This is useful when a scan gets aborted or scans of unequal length are merged.
    Feb 25, 2017: Bug fix in combineArrays

"""

import h5py, sys, os
import numpy as np

# Add new array ds to array ds_existing.
# Expand or pad with NaN where sizes don't match.
def combineArrays(ds_existing, ds):
    # Merge to new version
    if ds_existing.shape[-1] == len(ds):
        # Size matches!
        ds_new = np.vstack((ds_existing,ds[...]))
    elif ds_existing.shape[-1] >  len(ds):
        # new data smaller than expected
        padding = np.tile(np.nan,np.abs(ds_existing.shape[-1] - len(ds)))
        ds_new = np.vstack((ds_existing, np.hstack(( ds[...], padding ))))
    elif ds_existing.shape[-1] < len(ds):
        # new data larger than expected
        padding = np.tile(np.nan,(ds_existing.shape[0],np.abs(ds_existing.shape[-1] - len(ds))))
        ds_existing = np.hstack(( ds_existing, padding ))
        ds_new = np.vstack((ds_existing,ds[...]))

    return ds_new

# Merge h5py dataset "ds" into h5py container "dest".
# Preserves variable attributes by indexing them with an integer if necessary.
def mergeDatasets(ds,dest):
    
    if not ds.name in dest:
        # If nothing there, just blind copy
        dest.copy(ds,ds.name)
    
    else:
        # Get existing dataset & its attributes from destination, and delete old version.
        ds_existing = dest[ds.name][...]
        ds_existing_attrs = dict(dest[ds.name].attrs)
        del dest[ds.name]
        
        # Merge arrays
        ds_new = combineArrays(ds_existing, ds)
        
        # Create new destination dataset with compression ON
        ds_new_obj=dest.create_dataset(ds.name,data=ds_new,compression='gzip',compression_opts=4)
        
        # Preserve attributes from prior instance.
        for a in ds_existing_attrs:
            ds_new_obj.attrs[a] = ds_existing_attrs[a]
        
        # Merge in attributes from the new dataset.
        # No fancy array indexing for dataset attributes (it's probably unnecessary).
        # Just append indices to the attribute names if they aren't constant.
        for a in ds.attrs:
            # Copy new attr
            if not a in ds_new_obj.attrs: ds_new_obj.attrs[a]=ds.attrs[a]
            # Merge existing attr if not constant
            elif ds_new_obj.attrs[a] != ds.attrs[a]:
                ds_new_obj.attrs[a+"_%i" % ds_new_obj.shape[0]]=ds.attrs[a]
    return



# Merge group attributes by converting variable attributes into arrays.
def mergeGroupAttrs(obj,dest):
    # Loop attributes
    for key in obj.attrs.keys():
        
        if 'attr_'+key in dest:
            # Merge attribute into existing dataset
            assert(isinstance(dest['attr_'+key],h5py.Dataset))
            merged_attr_data=np.array(np.hstack(( dest['attr_'+key][...].ravel(), obj.attrs[key] )))
            # Delete previous dataset
            del dest['attr_'+key]
        else:
            merged_attr_data=np.array([obj.attrs[key]])
        
        # Make attribute into dataset
        new_attr_ds = dest.create_dataset('attr_'+key, data=merged_attr_data,\
                                  compression='gzip',compression_opts=4)
        # Set a flag telling us that this dataset came from a group attribute.
        # This will avoid problems in the recursive merging of other datasets.
        new_attr_ds.attrs['from_group_attribute']=1
    return




# Recursive function which duplicates the structure of HDF5 files
def mergeStructure(source, dest):
    for obj in source.values():
        if isinstance(obj,h5py.Dataset):
            # Copy/merge dataset
            mergeDatasets(obj,dest)
        elif isinstance(obj,h5py.Group):
            # Recurse into group
            grpname = obj.name
            # Check for pre-existing
            if grpname in dest: destgrp=dest[grpname]
            else: destgrp=dest.create_group(grpname)
            mergeStructure(obj, destgrp)
            # Merge group attributes - particularly useful for Extra_PVs group.
            if len(obj.attrs.keys()) > 0: mergeGroupAttrs(obj,destgrp)
    return



# Post-merge cleanup : any group attributes that were constant will be arrays
# full of the same value. We can get rid of these and convert them back to 1D
# attributes to save space.
def collapseConstantAttrArrays(name, obj):
    if 'from_group_attribute' in obj.attrs:
        if obj.attrs['from_group_attribute']==1:
            data=obj[...]
            if len(np.unique(data))<2: # If all entries in array identical...
                newname = ''.join(os.path.basename(name).split('attr_'))
                if 'mca_' in name:
                    # If the attr array comes from the MCA record, put it with
                    # one of the MCA arrays.
                    p=[ mds for mds in obj.parent.values() if 'mca_' in mds.name\
                       and not 'attr' in mds.name ]
                    if len(p)==0: p=obj.parent
                    else: p=p[0]
                    newname = ''.join(os.path.basename(newname).split('mca_'))
                else:
                    # Otherwise store it at parent group level (ie Extra PVs group)
                    p=obj.parent
            
                if len(data.shape)==1: p.attrs[newname]=data[0]
                else: p.attrs[newname]=data[0,0]
                obj.attrs['from_group_attribute']=-1 # flag for deletion
    return



# Post-merge cleanup : delete group attr arrays flagged with from_group_attribute==-1
def deleteFlaggedAttrArrays(h5obj):
    ndeleted=0
    for obj in h5obj.values():
        if isinstance(obj,h5py.Dataset):
            if 'from_group_attribute' in obj.attrs:
                if obj.attrs['from_group_attribute']==-1:
                    #print '\tdeleting',obj.name
                    del h5obj[obj.name]
                    ndeleted+=1
        elif isinstance(obj,h5py.Group):
            ndeleted+=deleteFlaggedAttrArrays(obj)
    return ndeleted


# Merge groups containing 2D MCA arrays into 3D arrays.
def make3dMCAArrays(dest, groups):
    
    # Get the motor name and positioner values
    motor = groups[0].split('.VAL=')[0]
    print "The positioner for the scan was",motor
    positionerValues=np.array([float(g.split('.VAL=')[1]) for g in groups])
    
    # Record separate dataset with positioner values from group names,
    # in case this varies from the detector value.
    dsetname=motor+'.VAL from group names'
    dest.create_dataset(dsetname,data=positionerValues,\
                        compression='gzip',compression_opts=4)
    print "Wrote",dsetname
    
    # Loop through arrays inside each group
    for dsname in dest[groups[0]].keys():
        # Find group with biggest array
        totalArrSz = [np.product(dest[groups[j]+'/'+dsname].shape) for j in range(len(groups))]
        j = np.where(totalArrSz == np.max(totalArrSz))[0][0]
        orig_shape = dest[groups[j]+'/'+dsname].shape
        dtype = dest[groups[j]+'/'+dsname].dtype
        new_shape = tuple(np.hstack((orig_shape[0],len(groups),orig_shape[1:])).astype(int))
        print "\tMerging",dsname,new_shape,dtype
        ds_new = dest.create_dataset('mca_'+dsname,shape=new_shape,dtype=dtype,\
                                     compression='gzip',compression_opts=4)
        if not 'S' in str(dtype): ds_new[...]=np.nan # fill numeric arrays with NaN in empty places
        
        # Loop through source groups. Keep indexing same as for positionerValues!
        for i in range(len(groups)):
            copy_from = dest[groups[i]+'/'+dsname][...]
            ds_new[:copy_from.shape[0],i,...]=copy_from # Allow variable number of points in scans in files, pad with NaN on RHS.

            # Copy attribute
            for a in dest[groups[i]+'/'+dsname].attrs:
                if not a in ds_new.attrs: ds_new.attrs[a] = dest[groups[i]+'/'+dsname].attrs[a]


    # Delete original groups
    for g in groups:
        del dest[g]



if __name__=='__main__':
    # Parse command line inputs
    if len(sys.argv)<4:
        print "Usage: %s hdf5-file hdf5-file [hdf5-file...] output-hdf5-file" % sys.argv[0]
        exit()
    elif sys.argv[1].strip().lower() == '-h':
        print "Usage: %s hdf5-file hdf5-file [hdf5-file...] output-hdf5-file" % sys.argv[0]
        exit()

    inputFiles = sys.argv[1:-1]
    outputFile = sys.argv[-1]
    print "\n=== Starting Merge_MCA_HDF5_Scans ==="
    print "Reading %i input files -- writing to %s" % (len(inputFiles),outputFile)

    # Open output file - overwrite existing
    hout = h5py.File(outputFile,'w')

    # Loop input files
    print "\nMerging matching arrays and groups between files..."
    for inputFilename in inputFiles:
        print "Reading",inputFilename,"..."
        hin = h5py.File(inputFilename,'r')
        mergeStructure(hin, hout)
        hin.close()
        # End reading input

    print "\nDone merging top level arrays between files"

    # Force file sync!
    hout.close()
    hout = h5py.File(outputFile,'a')

    ''' Now we can check if there is another level to merge.
        The first pass of merging will combine MCAs between the hdf5 files.
        If each file is a 2-D record with an outer positioner, then those positions
        will remain seperated into seperate groups i.e. 'MOTOR.VAL=POS'.
        
        This part of the code can combined these groups into a single 3-D array.
    '''
    groups = [ key for key in hout.keys() if '.VAL=' in key ]
    if len(groups) > 0:
        print "\nMerging inner scans: %i positions detected" % len(groups)
        make3dMCAArrays(hout, groups)

    # Now go through the file and collapse any attribute arrays that are constant.
    # We do this mainly because the heirarchy could get very messy without it.
    print "\nCleaning up..."
    hout.visititems(collapseConstantAttrArrays)

    # Force file sync!
    hout.close()
    hout = h5py.File(outputFile,'a')

    # Delete superfluous arrays 
    print "Collapsed %i attribute arrays with constant value" % deleteFlaggedAttrArrays(hout)

    hout.close()
    print "\nDone!"
