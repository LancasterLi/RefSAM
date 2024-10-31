import torch.nn as nn
import torch
from torch.nn import functional as F
from transformers import T5EncoderModel, T5TokenizerFast, RobertaModel, RobertaTokenizerFast
from transformers import CLIPProcessor, CLIPModel
# import clip
from per_segment_anything import sam_model_registry, SamPredictor
from einops import rearrange
import math
# import open_clip

class _LoRALayer(nn.Module):
    def __init__(self, w:nn.Module, w_a:nn.Module, w_b:nn.Module):
        super(_LoRALayer, self).__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, x):
        x = self.w(x) + self.w_b(self.w_a(x))
        return x


class Model(nn.Module):
    def __init__(self, args, text_model_name, logger):
        super(Model, self).__init__()

        # loading text encoder
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("\n------------> Loading text encoder")

        model_name = text_model_name

        if "t5" in text_model_name:
            tokenizer = T5TokenizerFast.from_pretrained(model_name)
            text_encoder = T5EncoderModel.from_pretrained(model_name)
        elif "roberta-base" in text_model_name:
            tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
            text_encoder = RobertaModel.from_pretrained('roberta-base')
        elif "roberta-large" in text_model_name:
            tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
            text_encoder = RobertaModel.from_pretrained('roberta-large')
        elif "clip-vit-base-patch32" in text_model_name:
            pass
        # text_encoder, tokenizer = clip.load(model_name)

        if text_model_name == "t5-small":
            resizer = FeatureResizer(512, 256, args)
        elif text_model_name == "t5-11b" or text_model_name == "t5-3b":
            resizer = FeatureResizer(1024, 256, args)
        elif text_model_name == "roberta-base":
            resizer = FeatureResizer(768, 256, args)
        elif text_model_name == "roberta-large":
            resizer = FeatureResizer(1024, 256, args)
        elif text_model_name == "clip-vit-base-patch32":
            resizer = FeatureResizer(768, 256, args)
        elif text_model_name == "ViT-L/14@336px":
            resizer = FeatureResizer(768, 256, args)
        logger.info(f"resizer structure:\n{resizer}")

        # spatial dynamic fusion for dense embeddings
        if args.spatial_dynamic_fusion:
            dense_conv = nn.Conv2d(256*2, 256, kernel_size=(1,1))

        if args.sparse_attention:
            sparse_fc = nn.Linear(256*2, 256)

        # loading sam
        logger.info("======> Load SAM")
        if args.sam == "vit_h":
            args.ckpt = "./sam_vit_h_4b8939.pth"
        elif args.sam == "vit_l":
            args.ckpt = "./sam_vit_l_0b3195.pth"
        elif args.sam == "vit_b":
            args.ckpt = "./sam_vit_b_01ec64.pth"
        sam_type, sam_ckpt = args.sam, args.ckpt
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt)

        # lora
        rank = args.lora_rank
        if args.train_image_encoder_lora:
            # create for storage, then we can init them or load weights
            self.w_As = [] # These are linear layers
            self.w_Bs = []
            base_vit_dim = sam.image_encoder.blocks[0].attn.proj.in_features
            dim = base_vit_dim

            for t_layer_i, blk in enumerate(sam.image_encoder.blocks):
                # linear layer
                w_linear = blk.attn.proj
                w_a_linear = nn.Linear(dim, rank, bias=False)
                w_b_linear = nn.Linear(rank, dim, bias=False)
                self.w_As.append(w_a_linear)
                self.w_Bs.append(w_b_linear)
                blk.attn.proj = _LoRALayer(w_linear, w_a_linear, w_b_linear)

            # init lora layers
            self.reset_parameters()        

        self.text_encoder = text_encoder
        self.resizer = resizer
        # self.resizer2 = resizer2
        self.sam = sam
        self.device = device
        self.tokenizer = tokenizer
        self.args = args
        if args.spatial_dynamic_fusion:
            self.dense_conv = dense_conv
        if args.sparse_attention:
            self.sparse_fc = sparse_fc
        # self.memory = args.memory
        self.logger = logger
        # memory
        if args.mask_word_memory:
            self.memory_key = nn.Conv2d(256, 256//8, kernel_size=(3,3), padding=(1,1), stride=1)
            self.memory_value = nn.Conv2d(256, 256//2, kernel_size=(3,3), padding=(1,1), stride=1)
            self.mask_conv = nn.Conv2d(1, 256, kernel_size=(7,7), padding=(3,3), stride=1, bias=False)
        if args.word_memory:
            self.memory_key = nn.Conv2d(256, 256//8, kernel_size=(3,3), padding=(1,1), stride=1)
            self.memory_value = nn.Conv2d(256, 256//2, kernel_size=(3,3), padding=(1,1), stride=1)
        if args.mask_memory:
            self.memory_key = nn.Conv2d(256, 256//8, kernel_size=(3,3), padding=(1,1), stride=1)
            self.memory_value = nn.Conv2d(256, 256//2, kernel_size=(3,3), padding=(1,1), stride=1)
            self.mask_conv = nn.Conv2d(1, 256, kernel_size=(7,7), padding=(3,3), stride=1, bias=False)

        if args.multi_scale:
            self.multi_scale_conv = nn.Conv2d(256*2, 256, kernel_size=(1,1))

            self.multi_scale_weight = nn.Parameter(torch.full((1,), 0.5))

        if args.track_query_attn:
            self.temporal_aggregation_network = QueryInteractionModule(args)

    def forward(self, img, caption, target, origin_track_tokens, use_track_tokens=False):
        # Text feature encoding
        # captions = target['caption']
        captions = caption
        encoded_input = self.tokenizer(captions, padding=True, truncation=True, return_tensors="pt").input_ids.to('cuda')
        # for clip
        # encoded_input = clip.tokenize(captions).to('cuda')

        self.predictor = SamPredictor(self.sam)

        # Image feature encoding
        # train_mask_list = target['masks']
        interm_embeddings_list = []
        train_feat_list = []
        for one_img in img:
            self.predictor.set_image(one_img) # [480 854 3]
            train_feat = self.predictor.features.squeeze()  # [256, 64, 64]
            train_feat = train_feat / train_feat.norm(dim=0, keepdim=True)
            train_feat = train_feat.unsqueeze(0)
            train_feat_list.append(train_feat)

        if self.args.mask_word_memory or self.args.mask_memory:
            first_train_feat_list = []
            for one_img, one_mask in zip(target['first_image'], target['first_mask']):
        
                self.predictor.set_image(one_img)  # [480 854 3]
                train_feat = self.predictor.features.squeeze()  # [256, 64, 64]
                train_feat = train_feat / train_feat.norm(dim=0, keepdim=True)
        
                train_mask = F.interpolate(one_mask.unsqueeze(0).unsqueeze(0), size=train_feat.shape[-2:]).cuda()
                train_mask_feat = self.mask_conv(train_mask)
        
                train_feat = train_feat + train_mask_feat
        
                first_train_feat_list.append(train_feat)
        
            last_train_feat_list = []
            for one_img, one_mask in zip(target['last_image'], target['last_mask']):
                self.predictor.set_image(one_img)  # [480 854 3]
                train_feat = self.predictor.features.squeeze()  # [256, 64, 64]
                train_feat = train_feat / train_feat.norm(dim=0, keepdim=True)
        
                train_mask = F.interpolate(one_mask.unsqueeze(0).unsqueeze(0), size=train_feat.shape[-2:]).cuda()
                train_mask_feat = self.mask_conv(train_mask)
        
                train_feat = train_feat + train_mask_feat
        
                last_train_feat_list.append(train_feat)

        if self.args.word_memory:
            first_train_feat_list = []
            for one_img, one_mask in zip(target['first_image'], target['first_mask']):
        
                self.predictor.set_image(one_img)  # [480 854 3]
                train_feat = self.predictor.features.squeeze()  # [256, 64, 64]
                train_feat = train_feat / train_feat.norm(dim=0, keepdim=True)
        
                train_feat = train_feat.unsqueeze(0)
        
                first_train_feat_list.append(train_feat)
        
            last_train_feat_list = []
            for one_img, one_mask in zip(target['last_image'], target['last_mask']):
        
                self.predictor.set_image(one_img)  # [480 854 3]
                train_feat = self.predictor.features.squeeze()  # [256, 64, 64]
                train_feat = train_feat / train_feat.norm(dim=0, keepdim=True)
        
                train_feat = train_feat.unsqueeze(0)
        
                last_train_feat_list.append(train_feat)

        # Text feature encoding
        outputs = self.text_encoder(input_ids=encoded_input)
        encoder_output = outputs.last_hidden_state  # 获取最后一层的编码特征 [batch_size, word len + 2, 512]
        text_sentence_features = torch.mean(encoder_output, axis=1)  # [batch_size, 512]

        after_text_sentence_features = self.resizer(text_sentence_features)  # [batch_size, 256]
        after_text_sentence_features = after_text_sentence_features / after_text_sentence_features.norm(dim=-1,
                                                                                                keepdim=True)  # L2 normalize

        after_text_word_features = self.resizer(encoder_output)  # [batch_size, word len + 2, 512]
        after_text_word_features = after_text_word_features / after_text_word_features.norm(dim=-1,
                                                                                            keepdim=True)  # L2 normalize

        high_res_masks_list = []
        track_tokens_list = []
        for train_feat, after_text_word_feature, after_text_sentence_feature, track_tokens in zip(train_feat_list, after_text_word_features, after_text_sentence_features, origin_track_tokens):

            sparse_embeddings = after_text_word_feature.unsqueeze(0)  # [1, N + 2, 256]

            if self.args.sparse_attention:
                sparse_word_feature = after_text_word_feature # [N c]
                image_feature = train_feat.squeeze(0) # [c h w]
                image_feature = rearrange(image_feature, "c h w -> c (h w)") # [c hw]
                spatial_dynamic_language_attention = sparse_word_feature @ image_feature # [N hw]
                spatial_dynamic_language_feature = spatial_dynamic_language_attention @ image_feature.permute(1, 0) # [N hw] @ [hw c] = [N c]
                
                concate_language_image_feature = torch.cat((spatial_dynamic_language_feature, sparse_word_feature), dim=1).unsqueeze(0) # [1, N, 2c]
                spatial_dynamic_language_fuse_feature = self.sparse_fc(concate_language_image_feature) # [1 N c]

                sparse_embeddings = spatial_dynamic_language_fuse_feature # [1 N c]


            # use dense_embeddings
            if self.args.dense_embeddings:

                if self.args.spatial_dynamic_fusion:
                    if self.args.mask_memory:
                        ## attention
                        origin_key = self.memory_key(train_feat) # [b c h w]
                        origin_value = self.memory_value(train_feat)
                        memory_feat = torch.cat((first_train_feat, last_train_feat), dim=0)
                        memory_key = self.memory_key(memory_feat)
                        memory_value = self.memory_value(memory_feat)
                    
                        kk = rearrange(origin_key, "b c h w -> (b h w) c") @ rearrange(memory_key, "b c h w -> c (b h w)") # [HW THW]
                        norm_kk = kk / math.sqrt(memory_key.shape[1])
                        softmax_kk = F.softmax(norm_kk, dim=-1) # [hw thw]
                    
                        kkv = softmax_kk @ rearrange(memory_value, "b c h w -> (b h w) c") # [hw c]
                        kkv_2 = rearrange(kkv, "(h w) c -> c h w", h=memory_key.shape[-2], w=memory_key.shape[-1])
                    
                        fuse_train_feat = torch.cat((origin_value, kkv_2.unsqueeze(0)), dim=1)
                        # skip connection
                        fuse_train_feat = train_feat + fuse_train_feat
                        # dense_embeddings = fuse_dense_embeddings

                    # 1: get spatial-dynamic attention
                    # train_feat: [1, 256, 64, 64]
                    # after_text_word_feature: [n, 256]

                    # concate word and sentence feature
                    after_text_word_feature = torch.cat((after_text_word_feature, after_text_sentence_feature.view(1, -1)), dim=0)
                    # for clip
                    # after_text_word_feature = after_text_word_feature

                    if self.args.mask_memory:
                        image_one = rearrange(fuse_train_feat, 'b c h w -> h w (b c)')
                    else:
                        image_one = rearrange(train_feat, 'b c h w -> h w (b c)')

                    text_one = rearrange(after_text_word_feature, 'n c -> c n')

                    c = text_one.shape[0]
                    sqrt_c = torch.sqrt(torch.Tensor([c])).to(train_feat.device)
                    spatial_dynamic_attention = torch.softmax(((image_one @ text_one) / sqrt_c), dim=-1)
                    # spatial_dynamic_attention = image_one @ text_one # [h w n]

                    # 2: get spatial dynamic language feature
                    spatial_dynamic_language_feature = spatial_dynamic_attention @ text_one.permute(1, 0) # [h w bc]
                    # 3: concate image and language feature
                    concate_image_language_feature = torch.cat((image_one, spatial_dynamic_language_feature), dim=-1).unsqueeze(0) # [1 h w 2bc]
                    concate_image_language_feature = rearrange(concate_image_language_feature, 'b h w c -> b c h w')
                    # 4: conv to fuse feature
                    spatial_dynamic_fusion_feature = self.dense_conv(concate_image_language_feature) # [b c h w]
                    dense_embeddings = spatial_dynamic_fusion_feature

                    if self.args.multi_scale:
                        image_one = rearrange(interm_embeddings, 'b c h w -> h w (b c)')

                        text_one = rearrange(after_text_word_feature, 'n c -> c n')

                        c = text_one.shape[0]
                        sqrt_c = torch.sqrt(torch.Tensor([c])).to(interm_embeddings.device)
                        spatial_dynamic_attention = torch.softmax(((image_one @ text_one) / sqrt_c), dim=-1)

                        # spatial_dynamic_attention = image_one @ text_one # [h w n]

                        # 2: get spatial dynamic language feature
                        spatial_dynamic_language_feature = spatial_dynamic_attention @ text_one.permute(1,
                                                                                                        0)  # [h w bc]
                        # 3: concate image and language feature
                        concate_image_language_feature = torch.cat((image_one, spatial_dynamic_language_feature),
                                                                   dim=-1).unsqueeze(0)  # [1 h w 2bc]
                        concate_image_language_feature = rearrange(concate_image_language_feature, 'b h w c -> b c h w')
                        # 4: conv to fuse feature
                        spatial_dynamic_fusion_feature = self.dense_conv(concate_image_language_feature)  # [b c h w]

                        # method1: concate
                        # concate_multi_scale_dense_embeddings = torch.cat([dense_embeddings, spatial_dynamic_fusion_feature], dim=1)
                        # multi_scale_dense_embeddings = self.multi_scale_conv(concate_multi_scale_dense_embeddings)
                        # method2: one weight parameter
                        multi_scale_dense_embeddings = self.multi_scale_weight * dense_embeddings + (1 - self.multi_scale_weight) * spatial_dynamic_fusion_feature

                    # use text feature in memory
                    if self.args.mask_word_memory or self.args.word_memory:

                        image_one = rearrange(first_train_feat, 'b c h w -> h w (b c)')
                        text_one = rearrange(after_text_word_feature, 'n c -> c n')
                        # text_one = rearrange(after_text_sentence_feature, 'n c -> c n')
                        spatial_dynamic_attention = image_one @ text_one  # [h w n]
                        # 2: get spatial dynamic language feature
                        spatial_dynamic_language_feature = spatial_dynamic_attention @ text_one.permute(1,
                                                                                                        0)  # [h w bc]
                        # 3: concate image and language feature
                        concate_image_language_feature = torch.cat((image_one, spatial_dynamic_language_feature),
                                                                dim=-1).unsqueeze(0)  # [1 h w 2bc]
                        concate_image_language_feature = rearrange(concate_image_language_feature, 'b h w c -> b c h w')
                        # 4: conv to fuse feature
                        spatial_dynamic_fusion_feature = self.dense_conv(concate_image_language_feature)  # [b c h w]
                        first_dense_embeddings = spatial_dynamic_fusion_feature
                    
                        image_one = rearrange(last_train_feat, 'b c h w -> h w (b c)')
                        text_one = rearrange(after_text_word_feature, 'n c -> c n')
                        # text_one = rearrange(after_text_sentence_feature, 'n c -> c n')
                        spatial_dynamic_attention = image_one @ text_one  # [h w n]
                        # 2: get spatial dynamic language feature
                        spatial_dynamic_language_feature = spatial_dynamic_attention @ text_one.permute(1,
                                                                                                        0)  # [h w bc]
                        # 3: concate image and language feature
                        concate_image_language_feature = torch.cat((image_one, spatial_dynamic_language_feature),
                                                                dim=-1).unsqueeze(0)  # [1 h w 2bc]
                        concate_image_language_feature = rearrange(concate_image_language_feature, 'b h w c -> b c h w')
                        # 4: conv to fuse feature
                        spatial_dynamic_fusion_feature = self.dense_conv(concate_image_language_feature)  # [b c h w]
                        last_dense_embeddings = spatial_dynamic_fusion_feature
                    
                        memory_dense_embeddings = torch.cat((first_dense_embeddings, last_dense_embeddings), dim=0)
                    
                        # attention
                        origin_key = self.memory_key(dense_embeddings) # [b c h w]
                        origin_value = self.memory_value(dense_embeddings)
                        memory_key = self.memory_key(memory_dense_embeddings)
                        memory_value = self.memory_value(memory_dense_embeddings)
                    
                        kk = rearrange(origin_key, "b c h w -> (b h w) c") @ rearrange(memory_key, "b c h w -> c (b h w)") # [HW THW]
                        norm_kk = kk / math.sqrt(memory_key.shape[1])
                        softmax_kk = F.softmax(norm_kk, dim=-1) # [hw thw]
                    
                        kkv = softmax_kk @ rearrange(memory_value, "b c h w -> (b h w) c") # [hw c]
                        kkv_2 = rearrange(kkv, "(h w) c -> c h w", h=memory_key.shape[-2], w=memory_key.shape[-1])
                    
                        fuse_dense_embeddings = torch.cat((origin_value, kkv_2.unsqueeze(0)), dim=1)
                        # skip connection
                        dense_embeddings = dense_embeddings + fuse_dense_embeddings
                        # dense_embeddings = fuse_dense_embeddings
                else:
                    # dense origin
                    dense_embeddings = train_feat + after_text_sentence_feature.view(1, -1, 1, 1)  # [1 256 64 64]
                    # dense simple
                    # dense_embeddings = after_text_sentence_feature.view(1, -1, 1, 1)  # [1 256 64 64]
                    # dense_embeddings = dense_embeddings.expand(1, dense_embeddings.shape[1], train_feat.shape[-2], train_feat.shape[-1])
            else:
                dense_embeddings = self.predictor.model.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                    1, -1, self.predictor.model.prompt_encoder.image_embedding_size[0],
                    self.predictor.model.prompt_encoder.image_embedding_size[1]
                )

            low_res_masks, iou_predictions, object_tokens = self.predictor.model.mask_decoder(
                image_embeddings=train_feat,
                image_pe=self.predictor.model.prompt_encoder.get_dense_pe().to(train_feat.device), # maintain device consistant
                sparse_prompt_embeddings=sparse_embeddings if self.args.sparse_embeddings else torch.zeros_like(sparse_embeddings),
                # dense_prompt_embeddings=torch.relu(dense_embeddings),
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                track_tokens=track_tokens,  ## track query
                use_track_tokens=use_track_tokens,
            )

            # Upscale the masks to the original image resolution
            high_res_masks = self.predictor.model.postprocess_masks(low_res_masks, self.predictor.input_size, self.predictor.original_size)
            high_res_masks_list.append(high_res_masks)
            ## for track tokens
            if self.args.track_query_attn:

                if track_tokens is not None:
                    update_track_tokens = self.temporal_aggregation_network(object_tokens + track_tokens)  # simple QIM
                elif track_tokens is None:
                    update_track_tokens = self.temporal_aggregation_network(object_tokens)  # simple QIM

                track_tokens_list.append(update_track_tokens)

        if use_track_tokens:
            return high_res_masks_list, track_tokens_list
        return high_res_masks_list

    def reset_parameters(self):
        for w_a in self.w_As:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_Bs:
            nn.init.zeros_(w_b.weight)


def FeatureResizer(input_feat_size, output_feat_size, args):
    layers = []

    if args.proj_mlp and args.num_mlp_layers >= 0:
        hidden = input_feat_size // 2
        num_mlp_layers = args.num_mlp_layers
        layers.append(nn.Linear(input_feat_size, hidden))
        layers.append(nn.ReLU())

        if args.mlp_drop > 0.05:
            layers.append(nn.Dropout(p=args.mlp_drop))

        for _ in range(num_mlp_layers):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU())

            if args.mlp_drop > 0:
                layers.append(nn.Dropout(p=args.mlp_drop))

        layers.append(nn.Linear(hidden, output_feat_size))
    else:
        layers.append(nn.Linear(input_feat_size, output_feat_size))

    return nn.Sequential(*layers)


class QueryInteractionModule(nn.Module):
    def __init__(self, args, dropout=0.0, dim_in=256, hidden_dim=1024):
        super(QueryInteractionModule, self).__init__()

        self.self_attn = nn.MultiheadAttention(dim_in, num_heads=8, dropout=dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)

        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(True)


    def forward(self, update_query_tokens):

        tgt = update_query_tokens
        tgt2 = self.self_attn(update_query_tokens, update_query_tokens, update_query_tokens)[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = update_query_tokens + self.dropout_feat2(query_feat2)
        query_feat = self.norm_feat(query_feat)

        return query_feat