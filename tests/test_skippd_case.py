"""Check forecast causality, time alignment, and experiment regression coverage."""
import copy
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PGF_TPA_ROOT", ROOT))
spec = importlib.util.spec_from_file_location("skippd_case", ROOT / "scripts/run_skippd_case.py")
case = importlib.util.module_from_spec(spec)
spec.loader.exec_module(case)


class SkippdCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = case.CaseData(DATA_ROOT)

    def test_forecast_issue_and_target_alignment(self):
        data = self.data
        row = next(r for r in data.alarm_chain("PGF-TPA") if r["triggered"])
        self.assertAlmostEqual(row["issue_hour"], 12 + 34 / 60)
        self.assertAlmostEqual(row["target_hour"] - row["issue_hour"], .25)
        self.assertGreaterEqual(row["observed_kw"] - row["forecast_kw"], case.TRIG_KW)
        self.assertIsNone(data.first_alarm("Persistence"))

    def test_moving_average_cannot_read_future_measurements(self):
        data = copy.copy(self.data)
        target, issue = 13.25, 13.0
        before = data.forecast("3h-MA", target)
        data.day_power = data.day_power.copy()
        data.day_power[data.day_hours > issue] = 1e9
        self.assertEqual(before, data.forecast("3h-MA", target))

    def test_forecasts_cannot_extrapolate_outside_saved_interval(self):
        with self.assertRaises(ValueError):
            self.data.forecast("PGF-TPA", 1.0)

    def test_horizon_mismatch_is_rejected(self):
        minutes = np.datetime64("2017-09-20T12:00:00") + np.arange(40) * np.timedelta64(1, "m")
        metadata = dict(minutes=minutes, pv=np.ones(40))
        samples = dict(win=np.arange(15)[None, :], tgt=np.array([28]), anchor_time=minutes[[14]])
        with mock.patch.object(case.np, "load", side_effect=[nullcontext(metadata), nullcontext(samples)]):
            with self.assertRaisesRegex(ValueError, "15 minutes"):
                case.CaseData(ROOT)

    def test_reference_check_detects_missing_scenario(self):
        reference = json.loads((ROOT / "experiments/skippd_case/reference.json").read_text())
        expected = reference["expected"]
        actual = copy.deepcopy(expected)
        actual["runs"].pop("S3")
        with self.assertRaisesRegex(AssertionError, "coverage"):
            case.check_reference(actual, expected)

    def test_dispatch_preserves_supply_balance(self):
        data = self.data
        res = case.simulate(data, True, case.S3FLAGS, data.first_alarm("PGF-TPA"),
                            data.pre_level("PGF-TPA"), data.pv)
        case.validate_dispatch(data, res)
        np.testing.assert_allclose(res["Pv_used"] + res["Pcurt"], res["Pv"], atol=1e-10)
        self.assertAlmostEqual(res["e_shed"], 1.20, places=2)


if __name__ == "__main__":
    unittest.main()
