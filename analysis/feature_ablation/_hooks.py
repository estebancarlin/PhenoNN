"""
Forward hooks used by the feature-ablation drivers.

The hooks are pickleable so that ``train_full_ram`` can save the whole model
with ``torch.save``; local closures would make that step fail.
"""

from phenonn.utils.config import FEATURE_CHANNELS


class ZeroChannels:
    """Forward pre-hook that zeroes the selected input channels.

    Features are z-scored, so zeroing a channel is mean-imputation. The class is
    defined at module level (not as a closure) because ``train_full_ram`` saves
    the whole model: registered hooks are pickled with it.

    ``torch`` is imported lazily inside ``__call__`` so importing this module
    (e.g. from ``gen_jobs``) does not load the whole framework up front.
    """

    def __init__(self, channels):
        self.channels = tuple(int(c) for c in channels)
        invalid = [c for c in self.channels if not 0 <= c < FEATURE_CHANNELS]
        if invalid:
            raise ValueError(
                f"channel(s) {invalid} out of range for {FEATURE_CHANNELS} features"
            )

    def __call__(self, _module, inputs):
        import torch

        x = inputs[0]
        if not torch.is_tensor(x):
            return None
        if x.shape[1] != FEATURE_CHANNELS:
            raise ValueError(
                f"expected {FEATURE_CHANNELS} feature channels, " f"got {x.shape[1]}"
            )
        x = x.clone()
        for ch in self.channels:
            x[:, ch, :] = 0.0
        return (x,) + tuple(inputs[1:])
