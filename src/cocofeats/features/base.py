from cocofeats.loggers import get_logger
from cocofeats.utils import get_path,replace_bids_suffix
import glob
import os
from cocofeats.definitions import PathLike
from typing import Any


log = get_logger(__name__)

class FeatureBase:
    """
    Base class for all feature extraction classes.
    """

    @classmethod
    def compute(cls,
        input: Any,
        reference_path: PathLike | None = None,
        suffix: str = None,
        save: bool = True,
        overwrite: bool = False,
        inspection_artifact: bool = True,
        context: dict[str, Any] | None = None, # to pass additional context if needed
        args: dict[str, Any] | None = None, # to pass additional arguments to internal functions if needed
        ):
        if suffix is None:
            suffix = cls.__name__
        output_path = None
        artifact_path = None
        if reference_path is not None:
            # Replace _suffix.ext with _suffix.classname.ext
            output_path = replace_bids_suffix(reference_path, suffix, f'.{cls.__name__}.html')
        
        if os.path.exists(output_path) and not overwrite:
            log.info("FeatureBase: output file already exists and overwrite is False, skipping computation", path=output_path)
            return output_path

        # Compute the feature here
        # This is a placeholder implementation and should be overridden by subclasses
        log.debug("FeatureBase: compute method called", input=input, reference_path=reference_path)

        if save:
            # save the feature to reference_path
            if reference_path is None:
                raise ValueError("reference_path must be provided if save is True")
            log.info("FeatureBase: feature saved", path=reference_path)
            output = reference_path  # Placeholder for actual saved feature
        if inspection_artifact:
            # save plots or other inspection artifacts
            if reference_path is None:
                raise ValueError("reference_path must be provided if inspection_artifact is True")
            # Update the reference_path to indicate it's an inspection artifact
            artifact_path = replace_bids_suffix(output_path, suffix, f'.{cls.__name__}.inspection.html')
            # save the inspection artifact to reference_path
            log.info("FeatureBase: inspection artifact saved", path=artifact_path)

        return output

    @staticmethod
    def load(path: PathLike):
        # Load the feature from the given path
        log.debug("FeatureBase: load method called", path=path)
        return path  # Placeholder for actual loaded feature
    
    @classmethod
    def is_feature_artifact(cls, path: PathLike) -> bool:
        # Check if the given path is a feature artifact
        log.debug("FeatureBase: is_feature_artifact method called", path=path)
        if '.' + cls.__name__ + '.' in str(path):
            return True
        return False
