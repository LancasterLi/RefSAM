import argparse

def get_arguments():
    parser = argparse.ArgumentParser()

    # special data setting
    parser.add_argument('--data', type=str, default='./ref-youtube-vos/train', help='ref-youtube dataset')
    parser.add_argument('--outdir', type=str, default='./ref_ytvos_sparse_dense_embeddings', help='output data path')

    # backbone setting
    parser.add_argument('--sam', type=str, default='vit_b', help='use sam backbone')
    parser.add_argument('--text_encoder', type=str, default='t5-3b', help='use text encoder backbone')

    # mlp resizer
    parser.add_argument('--proj_mlp', action='store_true', help='use mlp replace project linear layer')
    parser.add_argument('--num_mlp_layers', type=int, help='use N layer mlp', default=0)
    parser.add_argument('--mlp_drop', type=float, help='mlp dropout', default=0)

    # track query
    parser.add_argument('--track_query', action='store_true', help='use track query to memory segment object')
    parser.add_argument('--num_frames', type=int, default=3, help='one iter use frame num for memory training')
    parser.add_argument('--lr_track_query_attn', type=float, default=1e-4)
    parser.add_argument('--track_query_attn', action='store_true', help='use temporal_aggregation_network')

    # multi-scale feature
    parser.add_argument('--multi_scale', action='store_true', help='use last three stage feature')

    # finetuning
    parser.add_argument('--train_decoder', action='store_true', help='train sam mask decoder')
    parser.add_argument('--train_image_encoder_lora', action='store_true', help='use lora to train image encoder')
    parser.add_argument('--lora_rank', type=int, default=4, help='lora low rank')

    # modality fuse
    parser.add_argument('--sparse_embeddings', action='store_true', help='use dense embeddings')
    parser.add_argument('--dense_embeddings', action='store_true', help='use dense embeddings')
    parser.add_argument('--spatial_dynamic_fusion', action='store_true', help='image&text feature fusion for dense embeddings')
    parser.add_argument('--sparse_attention', action='store_true', help='use sparse attention')

    # learning rate
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--lr_decoder', type=float, default=1e-1)
    parser.add_argument('--lr_dense_conv', type=float, default=1e-1)
    parser.add_argument('--lr_image_encoder_lora', type=float, default=1e-4)
    parser.add_argument('--lr_memory', type=float, default=1e-4)
    # adapter
    parser.add_argument('--lr_adapter', type=float, default=1e-4)

    # pretrain setting
    parser.add_argument('--pretrain', action='store_true', default=None, help="whether use pretrain model")
    parser.add_argument('--pretrain_decoder', type=str, default=None, help="sam mask decoder checkpoint")
    parser.add_argument('--pretrain_dense_conv', type=str, default=None, help="dense attention conv checkpoint")
    parser.add_argument('--pretrain_mlp', type=str, default=None, help="resizer mlp checkpoint")
    parser.add_argument('--pretrain_lora_blocks', type=str, default=None)
    parser.add_argument('--pretrain_memory_key', type=str, default=None)
    parser.add_argument('--pretrain_memory_value', type=str, default=None)
    parser.add_argument('--pretrain_mask_conv', type=str, default=None)
    parser.add_argument('--pretrain_track_query_attn', type=str, default=None)
    parser.add_argument('--pretrain_track_token_mlp', type=str, default="0")
    parser.add_argument('--pretrain_dense_conv2', type=str, default=None)
    parser.add_argument('--pretrain_fpn', type=str, default=None)
    # adapter
    parser.add_argument('--pretrain_adapter', type=str, default="0")

    # training setting
    parser.add_argument('--train_epoch', type=int)
    parser.add_argument('--log_epoch', type=int, default=1, help='N epoch logging model lr and loss')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--datasets', type=str, default="refcoco", help="refcoco;ytvos")

    # inference setting
    parser.add_argument('--multi_mask', action='store_true', help='use multi_mask setting')
    parser.add_argument('--visualize', action='store_true', help='use multi_mask setting')

    # resume setting
    parser.add_argument('--resume', action='store_true', help='use resume checkpoint')
    parser.add_argument('--resume_mlp', type=str, default=None, help="resizer mlp checkpoint")
    parser.add_argument('--resume_dense_conv', type=str, default=None, help="dense attention conv checkpoint")
    parser.add_argument('--resume_decoder', type=str, default=None, help="sam mask decoder checkpoint")
    parser.add_argument('--resume_lora_blocks', type=str)
    parser.add_argument('--resume_track_query_attn', type=str)
    parser.add_argument('--resume_track_token_mlp', type=str, default="0")
    parser.add_argument('--resume_dense_conv2', type=str, default=None)
    parser.add_argument('--resume_fpn', type=str, default=None)
    parser.add_argument('--resume_epoch', type=int)
    # adapter
    parser.add_argument('--resume_adapter', type=str, default="0")
    
    # memory setting
    parser.add_argument('--mask_word_memory', action='store_true', help='use mask+word memory')
    parser.add_argument('--word_memory', action='store_true', help='use word memory')
    parser.add_argument('--mask_memory', action='store_true', help='use mask memory')

    # track token setting
    parser.add_argument('--frame_num', type=int, default=3, help='train to use num of frames')

    # save iteration setting
    parser.add_argument('--save_iterval', action='store_true', help='save iteration model')
    parser.add_argument('--save_iterval_num', type=int, default=1000, help='N iteration save model checkpoint')

    # other setting
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    args = parser.parse_args()
    return args