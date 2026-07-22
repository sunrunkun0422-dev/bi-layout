import unittest
from unittest.mock import patch

from dataset.zind_new_dataset import ZindNewDataset


class ZindDatasetLimitTest(unittest.TestCase):
    @patch("dataset.zind_new_dataset.invalid_filter")
    @patch("dataset.zind_new_dataset.read_zind_subset")
    def test_for_test_index_limits_two_head_dataset(self, read_subset, invalid_filter):
        read_subset.return_value = [{"id": str(index)} for index in range(5)]
        invalid_filter.side_effect = lambda pano_list, **_: pano_list

        dataset = ZindNewDataset(
            "unused",
            mode="test",
            model_type="occlusion",
            data_type="both",
            simplicity="both",
            primary="both",
            shape=(512, 1024),
            for_test_index=2,
        )

        self.assertEqual(len(dataset.data), 2)
        self.assertEqual([sample["id"] for sample in dataset.data], ["0", "1"])


if __name__ == "__main__":
    unittest.main()
