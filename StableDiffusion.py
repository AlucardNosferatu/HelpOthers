from DL.FCN import full_convolution_net_for_sd
from Model.Transformer import transformer_encoder_only
from config import TGT_VOC_SIZE, N_LAYERS, UNITS, WORD_VEC_DIM, N_HEADS, DROP, MAX_SL

if __name__ == '__main__':
    text_encoder = transformer_encoder_only(
        seq_length=MAX_SL,
        vocab_size=TGT_VOC_SIZE,
        num_layers=N_LAYERS,
        units=UNITS,
        word_vec_dim=WORD_VEC_DIM,
        num_heads=N_HEADS,
        dropout=DROP
    )

    diffusion_model = full_convolution_net_for_sd()
    print('Done')
