"""Run with python -m unittest features.test_pipeline -v (no dataset required)."""
import builtins
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from features.build_tables import FEATURES, build_trip_features
from features.pipeline import prepare_trip, prepare_trip_frames


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / "student_test" / "00000007"
        self.folder.mkdir(parents=True)
        self.prefix = self.folder / "00000007"
        # First and last observations are far from all arcs, middle one is near.
        self.prefix.with_suffix(".track").write_text(
            "10.02 50.02 0\n10.0005 50 4\n10.02 50.02 100\n")
        self.prefix.with_suffix(".nodes").write_text("10 50\n10.001 50\n")
        self.prefix.with_suffix(".arcs").write_text("0 1\n1 0\n")
        self.prefix.with_suffix(".route").write_text("not valid route data")
        self.env = patch.dict(os.environ, {"PATHSNAP_DATA_ROOT": str(self.root)})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_no_reference_access_and_exact_builder_parity(self):
        real_open = builtins.open
        def guarded(file, *args, **kwargs):
            if str(file).endswith(".route"):
                self.fail("Inference opened a reference route")
            return real_open(file, *args, **kwargs)
        with patch("builtins.open", side_effect=guarded):
            result = prepare_trip("7")
        expected, counts = build_trip_features(result.track, result.nodes, result.arcs)
        expected.insert(0, "trip_id", "00000007")
        assert_frame_equal(result.candidates, expected, check_exact=True)
        np.testing.assert_array_equal(result.observations.n_cand, counts)
        self.assertEqual(list(result.model_features.columns), FEATURES)
        self.assertNotIn("label", result.candidates)
        self.assertNotIn("ref_arc", result.observations)

    def test_preserves_missing_first_and_last_observations(self):
        result = prepare_trip(7)
        self.assertEqual(result.n_obs, 3)
        self.assertEqual(result.observations.obs_idx.tolist(), [0, 1, 2])
        self.assertEqual(result.observations.n_cand.tolist(), [0, 2, 0])
        self.assertEqual(result.candidates.obs_idx.tolist(), [1, 1])
        self.assertEqual(result.track.t.tolist(), [0, 4, 100])

    def test_all_observations_without_candidates(self):
        self.prefix.with_suffix(".track").write_text("10.02 50.02 0\n10.02 50.02 10\n")
        result = prepare_trip(7)
        self.assertEqual(result.n_obs, 2)
        self.assertTrue(result.candidates.empty)
        self.assertEqual(result.model_features.shape, (0, 10))
        self.assertEqual(result.observations.n_cand.tolist(), [0, 0])

    def test_frame_entrypoint_matches_trip_entrypoint(self):
        loaded = prepare_trip(7)
        result = prepare_trip_frames(loaded.track, loaded.nodes, loaded.arcs,
                                     trip_id=loaded.trip_id, partition=loaded.partition)
        assert_frame_equal(loaded.candidates, result.candidates, check_exact=True)
        assert_frame_equal(loaded.observations, result.observations, check_exact=True)
        self.assertEqual(result.n_obs, loaded.n_obs)

    def test_single_observation(self):
        self.prefix.with_suffix(".track").write_text("10.0005 50 0\n")
        result = prepare_trip(7)
        self.assertEqual(result.n_obs, 1)
        self.assertEqual(result.candidates.dt_prev.tolist(), [1., 1.])

    def test_route_file_not_required(self):
        self.prefix.with_suffix(".route").unlink()
        self.assertEqual(prepare_trip(7).n_obs, 3)

    def test_invalid_ids(self):
        for tid in ["../7", "", "000000007", "7.0", "٧", "-7"]:
            with self.subTest(tid=tid), self.assertRaises(ValueError):
                prepare_trip(tid)

    def test_evaluator_rejected(self):
        with self.assertRaises(ValueError):
            prepare_trip(7, "evaluator")

    def test_missing_trip_rejected(self):
        with self.assertRaises(FileNotFoundError):
            prepare_trip(8)


if __name__ == "__main__":
    unittest.main()
