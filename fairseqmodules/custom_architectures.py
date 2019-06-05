from fairseq.models import register_model_architecture

@register_model_architecture('lstm', 'lstm_wmt2017')
def lstm_wmt2017(args):
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 1024)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', 1024)
    args.encoder_bidirectional= getattr(args, 'encoder_bidirectional', True)
    #The default=False in LSTMModel add_args breaks the getattr approach
    args.share_decoder_input_output_embed = True #getattr(args, 'share_decoder_input_output_embed', True)

