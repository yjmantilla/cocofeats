from cocofeats.definitions import Artifact, FeatureResult

from . import register_feature


@register_feature
def dummy(param1=None, param2=None) -> FeatureResult:
    """
    A dummy feature extraction function that returns a simple message.

    Parameters
    ----------
    param1 : Any, optional
        An optional parameter for demonstration purposes.
    param2 : Any, optional
        Another optional parameter for demonstration purposes.

    Returns
    -------
    FeatureResult
        A FeatureResult containing a simple message.
    """
    message = f"Dummy feature extraction completed with param1={param1} and param2={param2}"

    def write_message(path: str) -> None:
        with open(path, "w") as f:
            f.write(message)

    artifacts = {".message.txt": Artifact(item=message, writer=lambda path: write_message(path))}

    return FeatureResult(
        artifacts=artifacts,
    )
