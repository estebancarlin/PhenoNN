# Model loader for PhenoCam GCC prediction.
#
# Same factory pattern as the original RTnn model_loader, but wraps
# every model with SingleDayWrapper so the output is (batch, 1)
# instead of (batch, 1, seq_length).

from phenocam.models.rnn import RNN_LSTM, RNN_GRU
from phenocam.models.Transformer import EncoderTorch
from phenocam.models.linear_baseline import LinearBaseline, PerDayLinearBaseline
from phenocam.models.fcn import FCN
from util.wrappers import SingleDayWrapper, permuteWrapper, LastNDaysWrapper
from phenocam.models.Transformerbis import CombinedModel, SimpleLSTM, PositionalEncoding, TransformerNTS, BiTransformer


def load_model(args):
    """
    Instantiate and wrap a model for single-day GCC prediction.

    Parameters
    ----------
    args : argparse.Namespace or EasyDict
        Must contain at minimum:
            type : str
                One of 'lstm', 'gru', 'transformer', 'fcn', 'fullyconnected'.
            feature_channel : int
                Number of input feature channels (meteo + cyclic + static + PFT).
            output_channel : int
                Number of output channels (1 for gcc_lowess).
            seq_length : int
                Window length (365).
        Plus architecture-specific parameters (see original model_loader).

    Returns
    -------
    SingleDayWrapper
        Model whose forward returns (batch, output_channel).
    """
    model_type = args.type.lower()

        # ── Linear baselines (no wrapper needed, already output (B, 1)) ──
 
    if model_type == "linear":
        return LinearBaseline(
            feature_channel=args.feature_channel,
            seq_length=args.seq_length,
        )
 
    if model_type == "linear_perday":
        return PerDayLinearBaseline(
            feature_channel=args.feature_channel,
        )


    if model_type in ["lstm", "gru"]:
        model_class = RNN_LSTM if model_type == "lstm" else RNN_GRU
        base_model = model_class(
            feature_channel=args.feature_channel,
            output_channel=args.output_channel,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
        )

    elif model_type == "1year_lstm":
        base_model = RNN_LSTM(
            feature_channel=args.feature_channel,
            output_channel=args.output_channel,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
        )
        return LastNDaysWrapper(base_model, n_days=365)

    elif model_type == "transformer":
        base_model = EncoderTorch(
            feature_channel=args.feature_channel,
            output_channel=args.output_channel,
            embed_size=args.embed_size,
            num_layers=args.num_layers,
            heads=args.nhead,
            forward_expansion=getattr(args, "forward_expansion", 4) or 4,
            seq_length=args.seq_length,
            dropout=args.dropout,
            causal=False,
        )

    elif model_type == "transformerbis":
        base_model = permuteWrapper(CombinedModel(
            input_dim = args.feature_channel,
            hidden_dim = args.hidden_size,
            hidden_dim_trans = args.hidden_size,
            output_dim = args.output_channel,
            d_model = 32,
            nr_blocks = 3,
            n_pft = 10,
        ))

    elif model_type == "bitransformer":
        base_model = permuteWrapper(BiTransformer(
            input_dim = args.feature_channel,
            hidden_dim = args.hidden_size,
            hidden_dim_trans = args.hidden_size,
            output_dim = args.output_channel,
            d_model = 32,
            nr_blocks = 3,
            n_pft = 3,
        ))

    elif model_type == "1year_bitransformer":
        base_model = permuteWrapper(BiTransformer(
            input_dim = args.feature_channel,
            d_model = args.hidden_size,
            feed_forward_trans = args.feed_forward_trans,
            feed_forward_encoder = args.feed_forward_encoder,
            output_dim = args.output_channel,
            nr_blocks = args.num_layers,
            dropout_trans = args.dropout_trans,
            dropout_encoder = args.dropout,
            n_pft = 9,
        ))
        return LastNDaysWrapper(base_model, n_days=365)

    elif model_type in ["fcn", "fullyconnected"]:
        base_model = FCN(
            feature_channel=args.feature_channel,
            output_channel=args.output_channel,
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            seq_length=args.seq_length,
            dim_expand=0,
        )

    else:
        raise ValueError(f"Model type '{args.type}' is not implemented.")

    # Choose wrapper based on training mode:
    #   n_target_days > 1 → LastNDaysWrapper (gradient loss, output last N days)
    #   otherwise         → SingleDayWrapper (standard, output last day only)
    n_target = getattr(args, "n_target_days", 1)
    if n_target > 1:
        return LastNDaysWrapper(base_model, n_days=n_target)


    return SingleDayWrapper(base_model)

