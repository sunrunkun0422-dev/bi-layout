import torch.nn
import torch
import torch.nn as nn
import models.modules as modules
import numpy as np

from models.base_model import BaseModule
from models.modules.horizon_net_feature_extractor import HorizonNetFeatureExtractor
from models.modules.patch_feature_extractor import PatchFeatureExtractor
from utils.conversion import uv2depth, get_u, lonlat2depth, get_lon, lonlat2uv, depth2xyz, uv2pixel
from utils.boundary import corners2boundaries
from models.modules.position_encoding import PositionEmbeddingSine
from utils.height import calc_ceil_ratio
from utils.misc import tensor2np

import matplotlib.pyplot as plt


class Bi_Layout(BaseModule):
    def __init__(self, ckpt_dir=None, backbone='resnet50', dropout=0.0, output_name='Bi_Layout',
                 decoder_name='Share_Feature_Guidance_Module', win_size=8, depth=6, output_number = 2,
                 feature_channel=1024, height_compression_scale=8, embedding_channel=512,
                 use_same_head=False, share_TF=True, two_conv_out=False,
                 ape=None, rpe=None, corner_heat_map=False, rpe_pos=1):
        super().__init__(ckpt_dir)

        if output_number not in (1, 2):
            raise ValueError("output_number must be either 1 or 2")
        if two_conv_out and output_number != 2:
            raise ValueError("two_conv_out=True requires output_number=2")
        if two_conv_out and backbone == 'patch':
            raise ValueError("two_conv_out=True requires a convolutional backbone")

        self.patch_num = 256
        # self.patch_dim = 1024
        self.patch_dim = feature_channel
        # for height compression
        # original setting = 8
        self.height_compression_scale = height_compression_scale
        self.decoder_name = decoder_name
        self.output_name = output_name
        self.corner_heat_map = corner_heat_map
        self.dropout_d = dropout
        # Two height compression conv output or not
        self.two_conv_out = two_conv_out
        # share transformer or not
        self.share_TF = share_TF
        # share head or not
        self.use_same_head = use_same_head
        # control output number
        self.output_number = output_number

        # Global Context Embeddings ------------------------------------------------
        # query content embeddings
        self.query_embed_origin = nn.Embedding(256, embedding_channel)
        self.query_embed_new = nn.Embedding(256, embedding_channel)
        # query position embeddings
        self.query_pos_origin = nn.Embedding(256, embedding_channel)
        self.query_pos_new = nn.Embedding(256, embedding_channel)
        # resnet feature position embedding
        self.feature_pos = PositionEmbeddingSine(256, normalize=True)

        if backbone == 'patch':
            self.feature_extractor = PatchFeatureExtractor(patch_num=self.patch_num, input_shape=[3, 512, 1024])
        else:
        # feature extractor
            self.feature_extractor = HorizonNetFeatureExtractor(
                backbone,
                second=self.two_conv_out,
                scale=self.height_compression_scale,
            )


        # transformer encoder parts -------------------------------
        transformer_dim = self.patch_dim
        transformer_layers = depth
        transformer_heads = 8
        transformer_head_dim = transformer_dim // transformer_heads
        transformer_ff_dim = 2048
        rpe = None if rpe == 'None' else rpe
        def build_transformer():
            return getattr(modules, decoder_name)(
                dim=transformer_dim,
                depth=transformer_layers,
                heads=transformer_heads,
                dim_head=transformer_head_dim,
                mlp_dim=transformer_ff_dim,
                win_size=win_size,
                dropout=self.dropout_d,
                patch_num=self.patch_num,
                ape=ape,
                rpe=rpe,
                rpe_pos=rpe_pos,
            )

        self.transformer = build_transformer()
        if self.output_number == 2 and not self.share_TF:
            self.transformer_2 = build_transformer()


        # two heads output parts ---------------------------------
        # omnidirectional-geometry aware output
        self.linear_depth_output = nn.Linear(in_features=self.patch_dim, out_features=1)
        self.linear_ratio = nn.Linear(in_features=self.patch_dim, out_features=1)
        self.linear_ratio_output = nn.Linear(in_features=self.patch_num, out_features=1)
        if output_number == 2 and not self.use_same_head:
            self.linear_depth_output_2 = nn.Linear(in_features=self.patch_dim, out_features=1)
            self.linear_ratio_2 = nn.Linear(in_features=self.patch_dim, out_features=1)
            self.linear_ratio_output_2 = nn.Linear(in_features=self.patch_num, out_features=1)

        if self.corner_heat_map:
            # corners heat map output
            self.linear_corner_heat_map_output = nn.Linear(in_features=self.patch_dim, out_features=1)

        self.name = f"{self.output_name}_Net"

    # separate transformer for two head -----------------------------------------------
    def bi_layout_outputs(self, x, new_x=None):
        """
        :param x: [ b, 256(patch_num), 1024(d)]
        :param new_x: [ b, 256(patch_num), 1024(d)]
        :return: {
            'depth': [b, 256(patch_num & d)]
            'ratio': [b, 1(d)]
        }
        """
        depth = self.linear_depth_output(x)  # [b, 256(patch_num), 1(d)]
        depth = depth.view(-1, self.patch_num)  # [b, 256(patch_num & d)]
        if self.output_number == 2:
            if new_x is None:
                raise ValueError("new_x is required when output_number=2")
            depth_head = (
                self.linear_depth_output
                if self.use_same_head
                else self.linear_depth_output_2
            )
            new_depth = depth_head(new_x)  # [b, 256(patch_num), 1(d)]
            new_depth = new_depth.view(-1, self.patch_num)  # [b, 256(patch_num & d)]
        
        # ratio represent room height
        ratio = self.linear_ratio(x)  # [b, 256(patch_num), 1(d)]
        ratio = ratio.view(-1, self.patch_num)  # [b, 256(patch_num & d)]
        ratio = self.linear_ratio_output(ratio)  # [b, 1(d)]
        if self.output_number == 2:
            ratio_head = self.linear_ratio if self.use_same_head else self.linear_ratio_2
            ratio_output_head = (
                self.linear_ratio_output
                if self.use_same_head
                else self.linear_ratio_output_2
            )
            new_ratio = ratio_head(new_x)  # [b, 256(patch_num), 1(d)]
            new_ratio = new_ratio.view(-1, self.patch_num)  # [b, 256(patch_num & d)]
            new_ratio = ratio_output_head(new_ratio)  # [b, 1(d)]
        
        if self.output_number == 1:
            output = {
                'depth': depth,
                'ratio': ratio
            }
        else:
            output = {
                'depth': depth,
                'ratio': (ratio+new_ratio)/2,
                'new_depth': new_depth,
            }
        return output    


    def forward(self, x, return_features=False):
        """
        :param x: [b, 3(d), 512(h), 1024(w)]
        :return: {
            'depth': [b, 256(patch_num & d)]
            'ratio': [b, 1(d)]
        }
        """

        # feature extractor
        if self.two_conv_out:
            x, new_x = self.feature_extractor(x, second=True)
        else:
            x = self.feature_extractor(x)  # [b 1024(d) 256(w)]
            new_x = x

        # transformer decoder
        x = x.permute(0, 2, 1)  # [b 256(patch_num) 1024(d)]
        new_x = new_x.permute(0, 2, 1)
        layout_feature = x
        new_layout_feature = new_x
        # position encoding
        pos = self.feature_pos(x) # [b 256(patch_num) 1024(d)]
        new_pos = self.feature_pos(new_x) if self.two_conv_out else pos

        # new branch transformer
        if self.output_number == 2:
            new_transformer = self.transformer if self.share_TF else self.transformer_2
            new_x = new_transformer(
                new_x,
                new_pos,
                self.query_embed_new.weight,
                self.query_pos_new.weight,
            )  # [b 256(patch_num) 1024(d)]
        else:
            new_x = None
        # original branch transformer
        x = self.transformer(x, pos, 
                            self.query_embed_origin.weight,
                            self.query_pos_origin.weight)  # [b 256(patch_num) 1024(d)]


        output = None

        output = self.bi_layout_outputs(x, new_x)

        if self.corner_heat_map:
            corner_heat_map = self.linear_corner_heat_map_output(x)  # [b, 256(patch_num), 1]
            corner_heat_map = corner_heat_map.view(-1, self.patch_num)
            corner_heat_map = torch.sigmoid(corner_heat_map)
            output['corner_heat_map'] = corner_heat_map

        if return_features:
            output['layout_feature'] = layout_feature
            output['new_layout_feature'] = new_layout_feature
            output['feature_pos'] = pos
            output['new_feature_pos'] = new_pos
            output['enc_feature'] = x
            output['ext_feature'] = new_x if new_x is not None else x

        return output


if __name__ == '__main__':
    from PIL import Image
    import numpy as np
    from models.other.init_env import init_env

    init_env(0, deterministic=True)

    net = Bi_Layout()

    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('parameter total:{:,}, trainable:{:,}'.format(total, trainable))

    img = np.array(Image.open("../src/demo.png")).transpose((2, 0, 1))
    input = torch.Tensor([img])  # 1 3 512 1024
    output = net(input)

    print(output['depth'].shape)  # 1 256
    print(output['ratio'].shape)  # 1 1
