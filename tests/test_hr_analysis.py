"""心率分析測試——合成數列、精確值."""
import numpy as np
import pandas as pd
import pytest

from src.isa.physio.hr_analysis import (daily_load_summary,
                                        hr_recovery,
                                        hr_zone_percentages,
                                        load_fitbit_heartrate)


def _hr_series(values):
    return pd.Series(values, dtype=float)


class TestZones:
    def test_zone_percentages_exact(self):
        # max_hr=190 → 50%=95, 60%=114, 70%=133, 85%=161.5
        hr = _hr_series([76, 100, 120, 150, 175])   # rest/light/moderate/vigorous/near_max
        zones = hr_zone_percentages(hr, max_hr=190)
        assert zones["rest"] == 20.0
        assert zones["light"] == 20.0
        assert zones["moderate"] == 20.0
        assert zones["vigorous"] == 20.0
        assert zones["near_max"] == 20.0

    def test_zone_boundary_inclusive_low(self):
        hr = _hr_series([95, 95])                    # 恰 50% → light（>=50）
        zones = hr_zone_percentages(hr, max_hr=190)
        assert zones["light"] == 100.0
        assert zones["rest"] == 0.0

    def test_mean_pct(self):
        hr = _hr_series([95, 190])
        zones = hr_zone_percentages(hr, max_hr=190)
        assert zones["_mean_pct_hrmax"] == 75.0


class TestRecovery:
    def test_hr_recovery_basic(self):
        start = pd.Timestamp("2026-10-29 08:00:00")
        times = pd.Series(pd.date_range(start, periods=60, freq="10s"))
        hr = np.full(60, 130.0)
        hr[0] = 160.0                 # anchor 當下峰值，其後 1 分鐘回落 130
        hrr = hr_recovery(pd.Series(hr), times, anchor=start)
        assert hrr == 30.0

    def test_hr_recovery_insufficient_data_returns_none(self):
        start = pd.Timestamp("2026-10-29 08:00:00")
        times = pd.Series([start])
        hr = pd.Series([150.0])
        assert hr_recovery(hr, times, anchor=start) is None

    def test_low_hrr_interpretation_threshold(self):
        # HRR<12 為風險門檻——確保 10 這種值可被判別（計算層回傳數值即可）
        start = pd.Timestamp("2026-10-29 08:00:00")
        times = pd.Series(pd.date_range(start, periods=30, freq="2s"))
        hr = np.full(30, 140.0)
        hr[10:] = 132.0
        assert hr_recovery(pd.Series(hr), times, anchor=start) == 8.0


class TestDailyLoad:
    def test_daily_load_summary_values(self):
        minute_hr = pd.Series([70.0] * 100 + [120.0] * 50 + [170.0] * 10)
        s = daily_load_summary(minute_hr)
        assert s["peak_hr"] == 170
        assert s["moderate_plus_minutes"] == 60.0   # 120(63%)與170(89%)都 >=60%
        assert s["weighted_load_minutes"] == 50 * 2 + 10 * 8


class TestLoader:
    def test_load_fitbit_heartrate(self, tmp_path):
        sub = tmp_path / "fitbit"
        sub.mkdir()
        pd.DataFrame({
            "Id": [1503960366, 1503960366],
            "Time": ["4/12/2016 7:21:00 AM", "4/12/2016 7:22:00 AM"],
            "Value": [52, 55],
        }).to_csv(sub / "heartrate_seconds_merged.csv", index=False)
        df = load_fitbit_heartrate(tmp_path / "fitbit")
        assert list(df.columns) == ["Id", "Time", "hr"]
        assert len(df) == 2

    def test_load_fitbit_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_fitbit_heartrate(tmp_path / "empty")
