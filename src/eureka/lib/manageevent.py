import numpy as np
import h5py as h5
import pickle
import os
import glob
import astraeus.xarrayIO as xrio

from . import readECF
from . import util


def saveevent(event, filename, save=[], delete=[], protocol=3):
    """Saves an event in .dat (using cpickle) and .h5 (using h5py) files.

    Parameters
    ----------
    event : eureka.lib.readECF.MetaClass
        The meta data object to save.
    filename : str
        The string contains the name of the event file.
    save : str tuple
        The elements of this tuple contain the parameters to save.
        We usually use the values: 'data', 'uncd', 'head', 'bdmskd',
        'brmksd' or 'mask'.
    delete : str tuple
        Parameters to be deleted.

    Notes
    -----
    The input filename should not have the .dat nor the .h5 extentions.
    Side effect: This routine deletes all parameters except 'event' after
    saving it.

    History:

    - 2010-07-10  patricio
        Added documentation.
    - 2010-11-12  patricio
        reimplemented using exec()
    """
    if save != []:
        with h5.File(filename + '.h5', 'w') as handle:
            for param in save:
                exec('handle["' + param + '"] = event.' + param)
                exec('del(event.' + param + ')')
                # calibration data
                if event.havecalaor:
                    exec('handle["pre' + param + '"] = event.pre' + param)
                    exec('handle["post' + param + '"] = event.post' + param)
                    exec('del(event.pre' + param + ', event.post' +
                         param + ')')

    # delete if requested
    for param in delete:
        exec('del(event.' + param + ')')
        if event.havecalaor:
            exec('del(event.pre' + param + ', event.post' + param + ')')

    # Pickle-Save the event
    with open(filename + '.dat', 'wb') as handle:
        pickle.dump(event, handle, protocol)


def loadevent(filename, load=[], loadfilename=None):
    """Loads an event stored in .dat and .h5 files.

    Parameters
    ----------
    filename : str
        The string contains the name of the event file.
    load : str tuple; optional
        The elements of this tuple contain the parameters to read.
        We usually use the values: 'data', 'uncd', 'head', 'bdmskd',
        'brmskd' or 'mask'. Defaults to [].
    loadfilename : str; optional
        The filename of the .h5 save file (excluding the file extension).
        Defaults to None which uses filename.

    Returns
    -------
    eureka.lib.readECF.MetaClass
        The requested metadata object.

    History:

    - 2010-07-10  patricio
        Added documentation.
    - 2010-11-12  patricio
        reimplemented using exec()
    """

    if '_Meta_Save' in filename:
        # This is a standard Meta_Save.dat file.
        if filename[-4:] != '.dat':
            filename += '.dat'
        with open(filename, 'rb') as handle:
            event = pickle.load(handle, encoding='latin1')
        if not hasattr(event, 'data_format'):
            event.data_format = 'eureka'
    elif 'SpecData' in filename:
        # This is a Stage 3 SpecData.h5 file.
        if filename[-3:] != '.h5':
            filename += '.h5'
        with xrio.readXR(filename) as handle:
            meta_attrs = util.load_attrs_from_xarray(handle)
        if 'data_format' not in meta_attrs.keys():
            # All Eureka! save files should have the data_format,
            # so this must be a custom file
            meta_attrs['data_format'] = 'custom'
        # Now create the Meta class and assign attrs
        event = readECF.MetaClass(**meta_attrs)
    else:
        raise AssertionError(f'Unrecognized metadata save file {filename}'
                             'contains neither "_Meta_Save" or "SpecData".')

    # FINDME: Do we really need this following code anymore?
    if load != []:
        if loadfilename is None:
            loadfilename = filename

        with h5.File(loadfilename + '.h5', 'r') as handle:
            for param in load:
                exec('event.' + param + ' = handle["' + param + '"][:]')
                # calibration data:
                if event.havecalaor:
                    exec('event.pre' + param + ' = handle["pre' + param +
                         '"][:]')
                    exec('event.post' + param + ' = handle["post' + param +
                         '"][:]')

    return event


def updateevent(event, filename, add):
    """Adds parameters given by add from filename to event.

    Parameters
    ----------
    event : eureka.lib.readECF.MetaClass
        The metadata object to update.
    filename : str
        The string contains the name of the event file.
    add : str tuple
        The elements of this tuple contain the parameters to
        add. We usually use the values: 'data', 'uncd', 'head',
        'bdmskd', 'brmaskd' or 'mask'.

    Returns
    -------
    eureka.lib.readECF.MetaClass
        The updated metadata object.

    Notes
    -----
    The input filename should not have the .dat nor the .h5 extentions.

    History:

    - 2010-07-10  patricio
        Initial version.
    - 2010-11-12  patricio
        reimplemented using exec()
    """
    event2 = loadevent(filename, load=add)

    for param in add:
        exec('event.' + param + ' = event2.' + param)
        # calibration data
        if event.havecalaor:
            exec('event.pre' + param + ' = event2.pre' + param)
            exec('event.post' + param + ' = event2.post' + param)

    return event


def findevent(meta, stage, allowFail=False):
    """Loads in an earlier stage meta file.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The new meta object for the current processing.
    stage : str
        The previous stage (e.g. "S2" for Stage 3).
    allowFail : bool; optional
        Whether to allow the code to find no previous stage metadata files
        (for S2, S3) or throw an error if no metadata files are found.
        Default is False.

    Returns
    -------
    old_meta : eureka.lib.readECF.MetaClass
        The old metadata object.
    inputdir : str
        The new inputdir to use (based on the present location of the located
        metadata file).
    inputdir_raw : str
        The new inputdir_raw to use (based on the present location of the
        located metadata file).

    Raises
    ------
    AssertionError
        Unable to find a metadata save file and allowFail was False.

    Notes
    -----
    History:

    - April 25, 2022 Taylor Bell
        Initial version.
    """
    # Search for the output metadata in the inputdir provided
    # First just check the specific inputdir folder, then check children
    # Check for both old (Meta_Save) and new (SpecData) metadata save files
    fnames = []
    for file_suffix in ['*_Meta_Save.dat', '*SpecData.h5']:
        newfnames = glob.glob(meta.inputdir+stage+'_'+meta.eventlabel +
                              file_suffix)

        if len(newfnames) == 0:
            # There were no metadata files in that folder, so let's see if
            # there are in children folders
            newfnames = glob.glob(meta.inputdir+'**'+os.sep+stage+'_' +
                                  meta.eventlabel+file_suffix, recursive=True)

        fnames.extend(newfnames)

    if len(fnames) == 0 and allowFail:
        # We're running an early enough stage that we don't need to find a
        # previous metadata save file. Just warn the user
        print(f'WARNING: Unable to find an output metadata file from '
              f'Eureka!\'s {stage} in the folder:\n"{meta.inputdir}"\n'
              f'Assuming this {stage} data was produced by another pipeline.')
        return None, meta.inputdir, meta.inputdir_raw
    elif len(fnames) == 0:
        # There were no metafiles in the inputdir or its children - raise an
        # error and give a helpful message
        raise AssertionError(f'WARNING: Unable to find an output metadata file'
                             f' of kind {file_suffix} from Eureka!\'s {stage}'
                             f' in the folder:\n"{meta.inputdir}"')
    elif len(fnames) > 1:
        # get the folder with the latest modified time
        folders = np.unique([os.sep.join(fname.split(os.sep)[:-1])
                             for fname in fnames])
        folder = max(folders, key=os.path.getmtime) + os.sep

        if len(folders) > 1:
            # There were multiple runs - use the most recent but warn the user
            print(f'WARNING: There are {len(fnames)} metadata save files in '
                  f'the folder: {meta.inputdir}\n  '
                  f'Using the metadata file inside: {folder}\n  '
                  f'and will consider aperture ranges listed there. If this '
                  f'metadata file is not a part\n  '
                  f'of the run you intended, please provide a more precise '
                  f'folder for the metadata file.')

        # Prefer Meta_Save if present to support older runs
        fnames = glob.glob(folder+stage+'_'+meta.eventlabel+'*_Meta_Save.dat')
        if len(fnames) == 0:
            # Otherwise, use the SpecData file
            fnames = glob.glob(folder+stage+'_'+meta.eventlabel+'*SpecData.h5')
        fname = fnames[0]
    else:
        # There was only the one save file found
        fname = fnames[0]

    # Load old savefile
    old_meta = loadevent(fname)

    old_meta.folder = os.sep.join(fname.split(os.sep)[:-1])+os.sep
    old_meta.filename = fname.split(os.sep)[-1]

    return old_meta, old_meta.folder, old_meta.folder[len(meta.topdir):]


def mergeevents(new_meta, old_meta):
    """Merge the current MetaClass data into the MetaClass object from a
    previous stage.

    Parameters
    ----------
    new_meta : eureka.lib.readECF.MetaClass
        The metadata object for the current stage.
    old_meta : eureka.lib.readECF.MetaClass
        The metadata object for the previous stage.

    Returns
    -------
    new_meta : eureka.lib.readECF.MetaClass
        The current metadata object containing the details from the previous
        metadata object.

    Notes
    -----
    History:

    - April 25, 2022 Taylor Bell
        Initial version.
    """
    # Load current ECF into old MetaClass
    for key in new_meta.__dict__:
        setattr(old_meta, key, getattr(new_meta, key))

    return old_meta
