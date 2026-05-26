import numpy as np
import h5py as h5
import pickle
import os
import glob
import re
import astraeus.xarrayIO as xrio

from . import readECF
from . import util


def filter_allapers_inputdir(meta):
    """Restrict allapers aperture/background pairs using inputdir glob matches.

    When allapers=True and inputdir contains a glob pattern, findevent records
    the matched candidate folders. This helper extracts ap/bg values from
    those folders and limits the standard allapers loop to matching pairs.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object for the current processing.

    Returns
    -------
    meta : eureka.lib.readECF.MetaClass
        The metadata object, optionally updated with allapers_ap_bg_pairs,
        allapers_inputdir_pair_dirs, spec_hw_range, and bg_hw_range.

    Raises
    ------
    AssertionError
        The inputdir glob matched folders, but none of their ap/bg values
        matched the aperture/background ranges from the previous stage.
    """
    if not getattr(meta, 'allapers', False):
        return meta

    candidates = getattr(meta, 'allapers_inputdir_candidates', None)
    if candidates is None:
        return meta

    pair_dirs = {}
    for folder in candidates:
        for part in reversed(folder.rstrip(os.sep).split(os.sep)):
            match = re.search(r'(?:^|_)ap(?P<ap>[^_/]+)_bg'
                              r'(?P<bg>.+?)(?:_run\d+)?$', part)
            if match is None:
                continue

            pair = []
            for value in [match.group('ap'), match.group('bg')]:
                try:
                    value = float(value)
                except ValueError:
                    pair.append(value)
                else:
                    if value.is_integer():
                        pair.append(int(value))
                    else:
                        pair.append(value)
            pair_dirs[tuple(pair)] = folder
            break

    if len(pair_dirs) == 0:
        return meta

    all_pairs = []
    for spec_hw_val in meta.spec_hw_range:
        for bg_hw_val in meta.bg_hw_range:
            pair = util.get_unexpanded_hws(meta.expand, spec_hw_val,
                                           bg_hw_val)
            if tuple(pair) in pair_dirs:
                all_pairs.append((spec_hw_val, bg_hw_val))

    if len(all_pairs) == 0:
        inputdir_raw = getattr(meta, 'inputdir_raw',
                               getattr(meta, 'inputdir', ''))
        pattern = getattr(meta, 'allapers_inputdir_glob_pattern',
                          getattr(meta, 'inputdir', ''))
        raise AssertionError(
            'The allapers=True inputdir glob matched folders, but none of '
            'their ap/bg values matched the aperture/background ranges from '
            f'the previous stage.\ninputdir: "{inputdir_raw}"\n'
            f'Expanded search pattern: "{pattern}"')

    meta.allapers_ap_bg_pairs = all_pairs
    meta.allapers_inputdir_pair_dirs = pair_dirs
    meta.spec_hw_range = list(dict.fromkeys([pair[0] for pair in all_pairs]))
    meta.bg_hw_range = list(dict.fromkeys([pair[1] for pair in all_pairs]))

    return meta


def get_allapers_pairs(meta):
    """Return aperture/background pairs for allapers processing.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object for the current processing.

    Returns
    -------
    pairs : list of tuple
        The (spec_hw_val, bg_hw_val) pairs to process. If inputdir glob
        filtering has been applied, only the matching pairs are returned.
        Otherwise all combinations of spec_hw_range and bg_hw_range are
        returned.
    """
    if hasattr(meta, 'allapers_ap_bg_pairs'):
        return meta.allapers_ap_bg_pairs

    return [(spec_hw_val, bg_hw_val)
            for spec_hw_val in meta.spec_hw_range
            for bg_hw_val in meta.bg_hw_range]


def get_allapers_specific_inputdir(meta, spec_hw_val, bg_hw_val):
    """Return the exact matched input folder for an allapers pair, if known.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object for the current processing.
    spec_hw_val : int, float, or str
        The spectrum aperture half-width value.
    bg_hw_val : int, float, or str
        The background half-width value.

    Returns
    -------
    inputdir : str or None
        The matched input folder for this aperture/background pair if inputdir
        glob filtering has been applied. Otherwise returns None.
    """
    pair_dirs = getattr(meta, 'allapers_inputdir_pair_dirs', {})
    pair = util.get_unexpanded_hws(meta.expand, spec_hw_val, bg_hw_val)

    return pair_dirs.get(tuple(pair))


def log_allapers_inputdir_glob(meta, log):
    """Log allapers inputdir glob expansion details when relevant.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object for the current processing.
    log : eureka.lib.logedit.Logedit
        The log object for the current stage.

    Returns
    -------
    None
    """
    if not hasattr(meta, 'allapers_inputdir_glob_pattern'):
        return

    log.writelog('allapers inputdir glob search pattern: '
                 f'{meta.allapers_inputdir_glob_pattern}')
    log.writelog('allapers inputdir candidate folders found: '
                 f'{meta.allapers_inputdir_candidate_count}')


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
        # Remove the file attribute to avoid reading in the old meta file
        meta_attrs.pop('file', None)
        # Now create the Meta class and assign attrs
        event = readECF.MetaClass(**meta_attrs)
    else:
        raise AssertionError(f'Unrecognized metadata save file {filename}'
                             'contains neither "_Meta_Save" or "SpecData".')

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
    """
    # Search for the output metadata in the inputdir provided
    # First just check the specific inputdir folder, then check children
    # Check for both old (Meta_Save) and new (SpecData) metadata save files
    use_allapers_glob = (getattr(meta, 'allapers', False) and
                         glob.has_magic(getattr(meta, 'inputdir', '')))
    if use_allapers_glob:
        pattern = meta.inputdir
        matches = glob.glob(pattern, recursive=True)
        search_dirs = sorted(np.unique([
            match+os.sep if match[-1] != os.sep else match
            for match in matches
            if os.path.isdir(match)
        ]))

        meta.allapers_inputdir_glob_pattern = pattern
        meta.allapers_inputdir_candidates = search_dirs
        meta.allapers_inputdir_candidate_count = len(search_dirs)

        if len(search_dirs) == 0:
            inputdir_raw = getattr(meta, 'inputdir_raw', pattern)
            raise AssertionError(
                'No input folders matched the allapers=True inputdir glob '
                f'pattern.\ninputdir: "{inputdir_raw}"\n'
                f'Expanded search pattern: "{pattern}"')
    else:
        search_dirs = [meta.inputdir]

    fnames = []
    for file_suffix in ['*_Meta_Save.dat', '*SpecData.h5']:
        for search_dir in search_dirs:
            newfnames = glob.glob(search_dir+stage+'_'+meta.eventlabel +
                                  file_suffix)

            if len(newfnames) == 0:
                # There were no metadata files in that folder, so let's see if
                # there are in children folders
                newfnames = glob.glob(search_dir+'**'+os.sep+stage+'_' +
                                      meta.eventlabel+file_suffix,
                                      recursive=True)

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
    """
    # Load current ECF into old MetaClass
    for key in new_meta.__dict__:
        if key == 'data_format' and getattr(old_meta, key) == 'custom':
            # Don't over-ride meta.data_format='custom' if set previously
            pass
        else:
            setattr(old_meta, key, getattr(new_meta, key))

    return old_meta
