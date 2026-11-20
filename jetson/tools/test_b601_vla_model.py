"""Unit and task boundaries at the model/controller interface; no hardware."""
import json
import math
from pathlib import Path
import tempfile
import unittest

from b601_vla_model import (LEGACY, load_model_spec, check_collection,
                            state_for_model, actions_for_controller)


class ModelContractTests(unittest.TestCase):
    def setUp(self):
        self.spec = {**LEGACY, 'id': 'test_screwdriver', 'units': 'radians',
                     'collection_profile': 'pick_place', 'fps': 14}

    def test_joint_order_and_gripper_unit_roundtrip(self):
        degrees = [-45, 90, 120, -75, 30, 85, 237]
        radians = state_for_model(degrees, self.spec)
        self.assertAlmostEqual(radians[1], math.pi/2)
        self.assertAlmostEqual(radians[6], 237*math.pi/180)
        restored = actions_for_controller([radians], self.spec)[0]
        for expected, actual in zip(degrees, restored):
            self.assertAlmostEqual(expected, actual)

    def test_degree_model_keeps_existing_semantics(self):
        values = [1., 2., 3., 4., 5., 6., 179.]
        self.assertEqual(state_for_model(values, LEGACY), values)
        self.assertEqual(actions_for_controller([values], LEGACY), [values])

    def test_reject_wrong_task_and_bad_unused_suffix(self):
        check_collection(self.spec, 'pick_place')
        with self.assertRaises(ValueError):
            check_collection(self.spec, 'insertion')
        with self.assertRaises(ValueError):
            actions_for_controller([[0.]*7, [0.]*6+[float('nan')]], self.spec)
        for values in ([0.]*6, [0.]*6+[True], [0.]*6+[float('inf')]):
            with self.assertRaises(ValueError):
                state_for_model(values, self.spec)

    def test_selection_preserves_explicit_contract(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'model.json'
            p.write_text(json.dumps(self.spec))
            self.assertEqual(load_model_spec(p), self.spec)
            for field, value in [('units', 'torque'), ('fps', 0), ('model_dir', 'relative')]:
                p.write_text(json.dumps({**self.spec, field: value}))
                with self.assertRaises(ValueError):
                    load_model_spec(p)


if __name__ == '__main__':
    unittest.main()
