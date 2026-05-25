from neurodags.definitions import Artifact, NodeResult

from . import register_node


@register_node
def dummy_multi(keys: list | None = None) -> NodeResult:
    """Return one artifact per key, for testing multi-artifact in-memory selection.

    Each artifact item is the key string; writer writes the key to a text file.
    Default keys: ["alpha", "beta"].
    """
    if keys is None:
        keys = ["alpha", "beta"]
    artifacts = {}
    for key in keys:
        artifacts[f".{key}.txt"] = Artifact(
            item=key,
            writer=lambda path, k=key: open(path, "w").write(k),
        )
    return NodeResult(artifacts=artifacts)


@register_node
def dummy(param1=None, param2=None) -> NodeResult:
    """
    A dummy derivative extraction function that returns a simple message.

    Parameters
    ----------
    param1 : Any, optional
        An optional parameter for demonstration purposes.
    param2 : Any, optional
        Another optional parameter for demonstration purposes.

    Returns
    -------
    NodeResult
        A NodeResult containing a simple message.
    """
    message = f"Dummy derivative extraction completed with param1={param1} and param2={param2}"

    def write_message(path: str) -> None:
        with open(path, "w") as f:
            f.write(message)

    artifacts = {".message.txt": Artifact(item=message, writer=lambda path: write_message(path))}

    return NodeResult(
        artifacts=artifacts,
    )
