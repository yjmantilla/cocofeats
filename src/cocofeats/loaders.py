from copy import deepcopy

import yaml
from loguru import logger
from mne import read_epochs
from mne.io import read_raw


def load_mne_meeg(meeg_file, kwargs=None):
    """Load a MEEG file using MNE-Python.
    Parameters
    ----------
    meeg_file : str
        The path to the MEEG file.
    kwargs : dict, optional
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
    if kwargs is None:
        kwargs = {}
    try:
        meeg = read_raw(meeg_file, **kwargs)
    except Exception:
        # try to load as epochs
        try:
            meeg = read_epochs(meeg_file, **kwargs)
        except Exception as e:
            # include traceback in logs
            logger.exception(f"Failed to load MEEG file {meeg_file} as Raw or Epochs")
            # raise a clearer error while preserving the original cause
            raise ValueError(f"Could not load MEEG file {meeg_file}") from e
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
    if isinstance(rules, str):
        try:
            with open(rules, encoding="utf-8") as f:
                return yaml.load(f, yaml.FullLoader)
        except Exception as e:
            logger.error(f"Could not read {rules} file as a rule file.")
            raise OSError(f"Couldnt read {rules} file as a rule file.") from e
    elif isinstance(rules, dict):
        return deepcopy(rules)
    else:
        logger.error(f"Expected str or dict as rules, got {type(rules)} instead.")
        raise ValueError(f"Expected str or dict as rules, got {type(rules)} instead.")
