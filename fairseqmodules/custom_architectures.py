from fairseq.models import register_model_architecture

@register_model_architecture('lstm', 'lstm_wmt2017')
def lstm_wmt2017(args):
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 1024)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', 1024)
    args.encoder_bidirectional= getattr(args, 'encoder_bidirectional', True)
    args.share_input_output_embed=getattr(args, 'share_input_output_embed', True) 
    args.max_source_positions= getattr(args, 'max_source_positions', 100)
    args.max_target_positions= getattr(args, 'max_target_positions', 100)
#    args.= getattr(args, '', )
#    args.= getattr(args, '', )
