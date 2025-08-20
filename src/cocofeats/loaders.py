
from mne.io import read_raw
from mne import read_epochs
from loguru import logger
import yaml
from copy import deepcopy

def load_mne_meeg(meeg_file, kwargs={}):
    """Load a MEEG file using MNE-Python.
    Parameters
    ----------
    meeg_file : str
        The path to the MEEG file.
    kwargs : dict
        Additional keyword arguments to pass to the MNE loading functions.
    Returns
    -------
    meeg : mne.io.Raw | mne.BaseEpochs
        The loaded MEEG data, either as a Raw object or Epochs object.
    Raises
    ------
    ValueError
        If the MEEG file cannot be loaded as either Raw or Epochs.
    """
    try:
        meeg = read_raw(meeg_file, **kwargs)
    except Exception as e:
        # try to load as epochs
        try:
            meeg = read_epochs(meeg_file, **kwargs)
        except Exception as e:
            logger.error(f"Failed to load MEEG file {meeg_file} as Raw or Epochs. Error: {e}")
            raise ValueError(f"Could not load MEEG file {meeg_file}. Error: {e}")
    return meeg 


def load_yaml(rules):
    """Load rules if given a path, bypass if given a dict.
    Parameters
    ----------
    
    rules : str|dict
        The path to the rules file, or the rules dictionary.
    Returns
    -------
    dict
        The rules dictionary.
    """
    if isinstance(rules,str):
        try:
            with open(rules,encoding="utf-8") as f:
                return yaml.load(f,yaml.FullLoader)
        except:
            logger.error(f"Could not read {rules} file as a rule file.")
            raise IOError(f"Couldnt read {rules} file as a rule file.")
    elif isinstance(rules,dict):
        return deepcopy(rules)
    else:
        logger.error(f"Expected str or dict as rules, got {type(rules)} instead.")
        raise ValueError(f'Expected str or dict as rules, got {type(rules)} instead.')
