import unittest
from tempfile import TemporaryDirectory

import numpy as np

from utils.joint_layout import DoorSpec, build_joint_layout, render_joint_boundary_svg


def make_layout(points):
    return {
        "cameraHeight": 1.6,
        "layoutHeight": 3.2,
        "layoutPoints": {
            "num": len(points),
            "points": [
                {"id": index, "xyz": [point[0], 1.6, point[1]]}
                for index, point in enumerate(points)
            ],
        },
        "layoutWalls": {
            "num": len(points),
            "walls": [
                {"pointsIdx": [index, (index + 1) % len(points)]}
                for index in range(len(points))
            ],
        },
    }


class JointLayoutTest(unittest.TestCase):
    def test_aligns_room_b_to_opposite_side_of_shared_door(self):
        layout_a = make_layout([[-2, -2], [2, -2], [2, 2], [-2, 2]])
        layout_b = make_layout([[-2, -2], [2, -2], [2, 2], [-2, 2]])

        result = build_joint_layout(
            layout_a,
            layout_b,
            DoorSpec(1, 0.25, 0.75),
            DoorSpec(3, 0.25, 0.75),
        )

        room_a_center = np.asarray(result["rooms"][0]["boundary"]).mean(axis=0)
        room_b_center = np.asarray(result["rooms"][1]["boundary"]).mean(axis=0)
        self.assertLess(room_a_center[0], 2)
        self.assertGreater(room_b_center[0], 2)
        np.testing.assert_allclose(result["sharedDoor"]["worldEndpoints"], [[2, -1], [2, 1]])

    def test_calibrates_scale_from_shared_door_width(self):
        layout_a = make_layout([[-2, -2], [2, -2], [2, 2], [-2, 2]])
        layout_b = make_layout([[-1, -1], [1, -1], [1, 1], [-1, 1]])

        result = build_joint_layout(
            layout_a,
            layout_b,
            DoorSpec(1, 0.25, 0.75),
            DoorSpec(3, 0.25, 0.75),
        )

        self.assertAlmostEqual(result["alignment"]["roomBScale"], 2.0)
        np.testing.assert_allclose(result["sharedDoor"]["worldEndpoints"], [[2, -1], [2, 1]])

    def test_rejects_invalid_door_spec(self):
        with self.assertRaises(ValueError):
            DoorSpec.parse("1:0.8:0.2")

    def test_renders_svg_with_wall_indices(self):
        layout = make_layout([[-2, -2], [2, -2], [2, 2], [-2, 2]])
        result = build_joint_layout(
            layout,
            layout,
            DoorSpec(1, 0.25, 0.75),
            DoorSpec(3, 0.25, 0.75),
        )
        with TemporaryDirectory() as output_dir:
            output_path = f"{output_dir}/boundaries.svg"
            render_joint_boundary_svg(result, output_path)
            with open(output_path, "r") as file:
                svg = file.read()
        self.assertIn("shared door", svg)
        self.assertIn("A:0", svg)
        self.assertIn("B:0", svg)


if __name__ == "__main__":
    unittest.main()
