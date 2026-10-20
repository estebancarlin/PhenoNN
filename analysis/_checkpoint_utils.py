"""Small helpers shared by the hand-run analysis studies.

The training checkpoint format changed over time: current runs store the model
weights under ``state_dict`` while older runs used ``model_state_dict``.  The
study scripts should not care which one they are given.
"""

from pathlib import Path


def load_state_dict(checkpoint):
    """Return the model weights from a training checkpoint.

    Parameters
    ----------
    checkpoint : dict
        Object returned by ``torch.load``.

    Returns
    -------
    dict
        The ``state_dict`` or legacy ``model_state_dict`` payload.

    Raises
    ------
    KeyError
        If neither key is present.
    """
    state = checkpoint.get("state_dict")
    if state is None:
        state = checkpoint.get("model_state_dict")
    if state is None:
        raise KeyError(
            "Checkpoint has neither 'state_dict' nor 'model_state_dict'. "
            f"Available keys: {sorted(checkpoint)}"
        )
    return state


def find_best_checkpoint(run_dir):
    """Find the best checkpoint of one experiment directory.

    Supports the current ``.pth.tar`` suffix, legacy ``.pth`` files and the
    training prefix (``run_`` by default).  ``*_full.pth`` snapshots are
    ignored because they are not regular training checkpoints.

    Parameters
    ----------
    run_dir : str or pathlib.Path
        Experiment directory, i.e. ``runs/<experiment>``.

    Returns
    -------
    pathlib.Path or None
        Matching checkpoint, or ``None`` when the run has not produced one.
    """
    checkpoints = Path(run_dir) / "checkpoints"
    if not checkpoints.is_dir():
        return None

    candidates = [
        path
        for path in checkpoints.glob("*best_model*.pth*")
        if path.is_file()
        and path.name.endswith((".pth", ".pth.tar"))
        and not path.name.endswith("_full.pth")
    ]
    if not candidates:
        return None

    def rank(path):
        name = path.name
        exact = name in ("best_model.pth.tar", "best_model.pth")
        prefixed = name.endswith(("_best_model.pth.tar", "_best_model.pth"))
        tar = name.endswith(".pth.tar")
        # Prefer current .pth.tar checkpoints over legacy .pth files, then
        # exact names over prefixed ones.
        return (not tar, not exact, not prefixed, name)

    return min(candidates, key=rank)


def has_completed_run(run_dir):
    """Return True when a run finished and wrote its final checkpoint."""
    checkpoints = Path(run_dir) / "checkpoints"
    if not checkpoints.is_dir():
        return False
    return any(checkpoints.glob("*final_model*.pth.tar"))
