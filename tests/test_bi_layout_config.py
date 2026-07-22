import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

import models.modules as modules
from models.bi_layout import Bi_Layout
from models.modules.horizon_net_feature_extractor import HorizonNetFeatureExtractor


class DummyFeatureExtractor(nn.Module):
    def __init__(self, backbone, second=False, scale=8):
        super().__init__()
        self.second = bool(second)

    def forward(self, image, second=False):
        batch = image.shape[0]
        primary = image.new_zeros((batch, 8, 256))
        if second:
            if not self.second:
                raise AssertionError("second feature output was not initialized")
            return primary, primary + 1.0
        return primary


class DummyPositionEncoding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, features):
        return torch.zeros_like(features)


class DummyTransformer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, features, position, query_embed, query_pos):
        return features + self.bias


class DummyEncoder(nn.Module):
    def forward(self, image):
        return [image]


class ConstantHeightStage(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)

    def forward(self, conv_list, out_w):
        batch = conv_list[0].shape[0]
        return conv_list[0].new_full((batch, 1, out_w), self.value)


def build_test_model(**overrides):
    options = {
        "backbone": "resnet50",
        "decoder_name": "DummyTransformer",
        "depth": 1,
        "output_number": 2,
        "feature_channel": 8,
        "embedding_channel": 8,
        "use_same_head": False,
        "share_TF": True,
        "two_conv_out": False,
    }
    options.update(overrides)
    with patch(
        "models.bi_layout.HorizonNetFeatureExtractor", DummyFeatureExtractor
    ), patch(
        "models.bi_layout.PositionEmbeddingSine", DummyPositionEncoding
    ), patch.object(
        modules, "DummyTransformer", DummyTransformer, create=True
    ):
        return Bi_Layout(**options)


class BiLayoutConfigTest(unittest.TestCase):
    def test_same_head_reuses_both_depth_and_ratio_heads(self):
        model = build_test_model(use_same_head=True)
        original = torch.randn(2, model.patch_num, model.patch_dim)
        extended = torch.randn(2, model.patch_num, model.patch_dim)

        output = model.bi_layout_outputs(original, extended)

        self.assertFalse(hasattr(model, "linear_depth_output_2"))
        expected_depth = model.linear_depth_output(extended).view(2, model.patch_num)
        torch.testing.assert_close(output["new_depth"], expected_depth)
        self.assertEqual(output["ratio"].shape, (2, 1))

    def test_share_tf_false_uses_an_independent_transformer(self):
        model = build_test_model(share_TF=False)
        self.assertIsNot(model.transformer, model.transformer_2)
        model.transformer.bias.data.fill_(1.0)
        model.transformer_2.bias.data.fill_(2.0)

        output = model(torch.zeros(1, 3, 8, 16), return_features=True)

        torch.testing.assert_close(
            output["enc_feature"], torch.ones_like(output["enc_feature"])
        )
        torch.testing.assert_close(
            output["ext_feature"], torch.full_like(output["ext_feature"], 2.0)
        )

    def test_two_conv_out_uses_independent_height_compression_features(self):
        model = build_test_model(two_conv_out=True)

        output = model(torch.zeros(1, 3, 8, 16), return_features=True)

        self.assertTrue(model.feature_extractor.second)
        torch.testing.assert_close(
            output["layout_feature"], torch.zeros_like(output["layout_feature"])
        )
        torch.testing.assert_close(
            output["new_layout_feature"],
            torch.ones_like(output["new_layout_feature"]),
        )

    def test_horizon_extractor_calls_second_height_stage(self):
        extractor = HorizonNetFeatureExtractor.__new__(HorizonNetFeatureExtractor)
        nn.Module.__init__(extractor)
        extractor.second = True
        extractor.step_cols = 4
        extractor.feature_extractor = DummyEncoder()
        extractor.reduce_height_module = ConstantHeightStage(1.0)
        extractor.reduce_height_module2 = ConstantHeightStage(2.0)
        extractor._prepare_x = lambda image: image

        primary, secondary = extractor(torch.zeros(1, 3, 2, 8), second=True)

        self.assertEqual(primary.unique().item(), 1.0)
        self.assertEqual(secondary.unique().item(), 2.0)

    def test_invalid_flag_combinations_fail_early(self):
        with self.assertRaisesRegex(ValueError, "output_number"):
            build_test_model(output_number=3)
        with self.assertRaisesRegex(ValueError, "output_number=2"):
            build_test_model(output_number=1, two_conv_out=True)
        with self.assertRaisesRegex(ValueError, "convolutional backbone"):
            Bi_Layout(backbone="patch", two_conv_out=True)


if __name__ == "__main__":
    unittest.main()
