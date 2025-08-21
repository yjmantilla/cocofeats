# Inspired from:
# https://github.com/yjmantilla/sovabids/blob/main/tests/test_bids.py
# https://github.com/yjmantilla/sovabids/blob/main/sovabids/datasets.py

import os
import shutil
from cocofeats.utils import get_num_digits
from mne_bids.write import _write_raw_brainvision
import fileinput
import mne
import numpy as np

def replace_brainvision_filename(fpath,newname):
    if '.eeg' in newname:
        newname = newname.replace('.eeg','')
    if '.vmrk' in newname:
        newname = newname.replace('.vmrk','')
    for line in fileinput.input(fpath, inplace=True):
        if 'DataFile' in line:
            print(f'DataFile={newname}.eeg'.format(fileinput.filelineno(), line))
        elif 'MarkerFile' in line:
            print(f'MarkerFile={newname}.vmrk'.format(fileinput.filelineno(), line))
        else:
            print('{}'.format(line), end='')


def make_dummy_dataset(EXAMPLE,
    PATTERN='T%task%/S%session%/sub%subject%_%acquisition%_%run%',
    DATASET = 'DUMMY',
    NSUBS = 2,
    NSESSIONS = 2,
    NTASKS = 2,
    NACQS = 2,
    NRUNS = 2,
    PREFIXES = {'subject':'SU','session':'SE','task':'TA','acquisition':'AC','run':'RU'},
    ROOT=None,
):
    """Create a dummy dataset given some parameters.
    
    Parameters
    ----------
    EXAMPLE : str,PathLike|list , required
        Path of the file to replicate as each file in the dummy dataset.
        If a list, it is assumed each item is a file. All of these items are replicated.
    PATTERN : str, optional
        The pattern in placeholder notation using the following fields:
        %dataset%, %task%, %session%, %subject%, %run%, %acquisition%
    DATASET : str, optional
        Name of the dataset.
    NSUBS : int, optional
        Number of subjects.
    NSESSIONS : int, optional
        Number of sessions.
    NTASKS : int, optional
        Number of tasks.
    NACQS : int, optional
        Number of acquisitions.
    NRUNS : int, optional
        Number of runs.
    PREFIXES : dict, optional
        Dictionary with the following keys:'subject', 'session', 'task' and 'acquisition'.
        The values are the corresponding prefix. RUN is not present because it has to be a number.
    ROOT : str, optional
        Path where the files will be generated.
        If None, the _data subdir will be used.

    """

    if ROOT is None:
        this_dir = os.path.dirname(__file__)
        data_dir = os.path.abspath(os.path.join(this_dir,'..','_data'))
    else:
        data_dir = ROOT
    os.makedirs(data_dir,exist_ok=True)

    sub_zeros = get_num_digits(NSUBS)
    subs = [ PREFIXES['subject']+ str(x).zfill(sub_zeros) for x in range(NSUBS)]

    task_zeros = get_num_digits(NTASKS)
    tasks = [ PREFIXES['task']+str(x).zfill(task_zeros) for x in range(NTASKS)]

    run_zeros = get_num_digits(NRUNS)
    runs = [str(x).zfill(run_zeros) for x in range(NRUNS)]

    ses_zeros = get_num_digits(NSESSIONS)
    sessions = [ PREFIXES['session']+str(x).zfill(ses_zeros) for x in range(NSESSIONS)]

    acq_zeros = get_num_digits(NACQS)
    acquisitions = [ PREFIXES['acquisition']+str(x).zfill(acq_zeros) for x in range(NACQS)]


    for task in tasks:
        for session in sessions:
            for run in runs:
                for sub in subs:
                    for acq in acquisitions:
                        dummy = PATTERN.replace('%dataset%',DATASET)
                        dummy = dummy.replace('%task%',task)
                        dummy = dummy.replace('%session%',session)
                        dummy = dummy.replace('%subject%',sub)
                        dummy = dummy.replace('%run%',run)
                        dummy = dummy.replace('%acquisition%',acq)
                        path = [data_dir] +dummy.split('/')
                        fpath = os.path.join(*path)
                        dirpath = os.path.join(*path[:-1])
                        os.makedirs(dirpath,exist_ok=True)
                        if isinstance(EXAMPLE,list):
                            for ff in EXAMPLE:
                                fname, ext = os.path.splitext(ff)
                                shutil.copyfile(ff, fpath+ext)
                                if 'vmrk' in ext or 'vhdr' in ext:
                                    replace_brainvision_filename(fpath+ext,path[-1])
                        else:
                            fname, ext = os.path.splitext(EXAMPLE)
                            shutil.copyfile(EXAMPLE, fpath+ext)


def generate_1_over_f_noise(n_channels, n_times, exponent=1.0, random_state=None):
    rng = np.random.default_rng(random_state)
    noise = np.zeros((n_channels, n_times))

    freqs = np.fft.rfftfreq(n_times, d=1.0)  # d=1.0 assumes unit sampling rate
    freqs[0] = freqs[1]  # avoid division by zero at DC

    scale = 1.0 / np.power(freqs, exponent)

    for ch in range(n_channels):
        # Generate white noise in time domain
        white = rng.standard_normal(n_times)
        # Transform to frequency domain
        white_fft = np.fft.rfft(white)
        # Apply 1/f scaling
        pink_fft = white_fft * scale
        # Transform back to time domain
        pink = np.fft.irfft(pink_fft, n=n_times)
        # Normalize to zero mean, unit variance
        pink = (pink - pink.mean()) / pink.std()
        noise[ch, :] = pink

    return noise

def get_dummy_raw(NCHANNELS = 5,
    SFREQ = 200,
    STOP = 10,
    NUMEVENTS = 10,
):
    """
    Create a dummy MNE Raw file given some parameters.

    Parameters
    ----------
    NCHANNELS : int, optional
        Number of channels.
    SFREQ : float, optional
        Sampling frequency of the data.
    STOP : float, optional
        Time duration of the data in seconds.
    NUMEVENTS : int, optional
        Number of events along the duration.
    """
    # Create some dummy metadata
    n_channels = NCHANNELS
    sampling_freq = SFREQ  # in Hertz
    info = mne.create_info(n_channels, sfreq=sampling_freq)

    times = np.linspace(0, STOP, STOP*sampling_freq, endpoint=False)
    data = generate_1_over_f_noise(NCHANNELS, times.shape[0], exponent=1.0)
    #np.zeros((NCHANNELS,times.shape[0]))

    raw = mne.io.RawArray(data, info)
    raw.set_channel_types({x:'eeg' for x in raw.ch_names})
    new_events = mne.make_fixed_length_events(raw, duration=STOP//NUMEVENTS)

    return raw,new_events

def save_dummy_vhdr(fpath,dummy_args={}
):
    """
    Save a dummy vhdr file.

    Parameters
    ----------
    fpath : str, required
        Path where to save the file.
    kwargs : dict, optional
        Dictionary with the arguments of the get_dummy_raw function.

    Returns
    -------
    List with the Paths of the desired vhdr file, if those were succesfully created,
    None otherwise.
    """

    raw,new_events = get_dummy_raw(**dummy_args)
    _write_raw_brainvision(raw,fpath,new_events,overwrite=True)
    eegpath =fpath.replace('.vhdr','.eeg')
    vmrkpath = fpath.replace('.vhdr','.vmrk')
    if all(os.path.isfile(x) for x in [fpath,eegpath,vmrkpath]):
        return [fpath,eegpath,vmrkpath]
    else:
        return None



DEF_DATASET_PARAMS ={'PATTERN':'T%task%/S%session%/sub%subject%_%acquisition%_%run%',
'DATASET' : 'DUMMY',
'NSUBS' : 2,
'NTASKS' : 2,
'NRUNS' : 1,
'NSESSIONS' : 1,
'NACQS' : 1,
}

def generate_dummy_dataset(data_params = DEF_DATASET_PARAMS):
    """Generates a dummy dataset with the specified pattern type and format.
    Parameters
    ----------
    data_params : dict
        Parameters for dataset generation, such as number of subjects, tasks, etc.
        Follows the arguments of `sovabids.datasets.make_dummy_dataset`.
    """
    

    # Getting current file path and then going to _data directory
    this_dir = os.path.dirname(__file__)
    data_dir = os.path.join(this_dir,'..','..','_data')
    data_dir = os.path.abspath(data_dir)

    # Defining relevant conversion paths
    dataset_name = data_params.get('DATASET','DUMMY')
    test_root = os.path.join(data_dir,dataset_name)
    input_root = os.path.join(test_root,dataset_name+'_SOURCE')
    bids_path = os.path.join(test_root,dataset_name+'_BIDS')

    # Make example File
    example_fpath = save_dummy_vhdr(os.path.join(data_dir,'dummy.vhdr'))

    # PARAMS for making the dummy dataset
    DATA_PARAMS ={ 'EXAMPLE':example_fpath,
        'ROOT' : input_root
    }
    DATA_PARAMS.update(data_params)

    # Preparing directories
    dirs = [input_root,bids_path] #dont include test_root for saving multiple conversions
    for dir in dirs:
        try:
            shutil.rmtree(dir)
        except:
            pass

    [os.makedirs(dir,exist_ok=True) for dir in dirs]

    # Generating the dummy dataset
    make_dummy_dataset(**DATA_PARAMS)