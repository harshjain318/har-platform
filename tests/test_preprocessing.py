import numpy as np
import pandas as pd

from preprocessing.column_mapper import ColumnMapping, detect_mapping, parse_time_seconds
from preprocessing.labels import auto_label_map, canonical_label
from preprocessing.pipeline import process_dataframe
from preprocessing.reader import read_csv_bytes
from tests.synth import acc_signal, gyro_signal

RNG = np.random.default_rng(0)


def _fmt_a(fs=50, n=600, ms=True, unit_scale=1.0):
    rows, off = [], 0
    for sub in ["u1", "u2"]:
        for act in ["walking", "sitting"]:
            t, a = acc_signal(act, n, fs, RNG, unit_scale)
            ts = ((t + off) * (1000 if ms else 1)).astype(float)
            off += n / fs + 10
            rows.append(pd.DataFrame({"timestamp": ts, "user": sub, "activity": act, "x": a[:, 0], "y": a[:, 1], "z": a[:, 2]}))
    return pd.concat(rows, ignore_index=True)


def test_labels():
    assert canonical_label("Walking_Upstairs") == "upstairs"
    assert canonical_label("dws") == "downstairs"
    assert canonical_label("stairsup") == "upstairs"
    assert canonical_label("jog") == "running"
    assert canonical_label("3", "UCI HAR") == "downstairs"
    assert canonical_label("3.0", "UCI HAR") == "downstairs"
    assert canonical_label("C", "WISDM") is None          # generic stairs cannot be split
    assert canonical_label("bike") is None
    assert auto_label_map(["Sitting", "xyz"])["xyz"] is None


def test_format_a_detection_and_pipeline():
    df = _fmt_a()
    det = detect_mapping(df)
    m = det.mapping
    assert (m.time, m.subject, m.label, m.acc) == ("timestamp", "user", "activity", ("x", "y", "z"))
    pf = process_dataframe(df, m, "A")
    s = pf.stats
    assert s["n_segments"] == 4 and s["n_subjects"] == 2
    assert s["original_fs_hz"] == 50.0 and s["acc_unit"] == "g"
    assert pf.windows.X.shape[1:] == (128, 3)
    assert pf.windows.n == 4 * ((600 - 128) // 64 + 1)    # 4 segments x 8 windows
    assert s["missing_values"] == 0


def test_no_window_crosses_boundaries():
    pf = process_dataframe(_fmt_a(), detect_mapping(_fmt_a()).mapping, "A")
    w = pf.windows
    for seg_id in np.unique(w.segment):
        seg = pf.segments[seg_id]
        st = w.start[w.segment == seg_id]
        assert st.min() >= seg.start and (st + 128).max() <= seg.start + seg.n
    # each window carries exactly one subject and one label
    for i in range(w.n):
        seg = pf.segments[w.segment[i]]
        assert seg.subject == w.subject[i] and seg.label == w.y[i]
    # gravity direction differs between walking and sitting windows -> labels were not mixed
    sit = w.X[w.y == "sitting"][:, :, 2].mean()
    walk = w.X[w.y == "walking"][:, :, 2].mean()
    assert sit > 0.5 > walk


def test_units_conversion_and_ms_timestamps():
    df = _fmt_a(unit_scale=9.80665)
    pf = process_dataframe(df, detect_mapping(df).mapping, "ms2")
    assert pf.stats["acc_unit"] == "m/s2"
    assert abs(np.linalg.norm(pf.signal[:, :3], axis=1).mean() - 1.0) < 0.2


def test_resampling_100hz_and_20hz():
    for fs in (100, 20):
        df = _fmt_a(fs=fs, n=int(fs * 12))
        pf = process_dataframe(df, detect_mapping(df).mapping, f"{fs}")
        assert abs(pf.stats["original_fs_hz"] - fs) < 1
        expected = 4 * (12 * 50)
        assert abs(pf.stats["resampled_samples"] - expected) <= 8
        assert pf.windows.n > 0


def test_duplicates_nans_unsorted():
    df = _fmt_a()
    df = pd.concat([df, df.iloc[:50]], ignore_index=True)       # duplicate timestamps
    df.loc[10:14, "x"] = np.nan
    df = df.sample(frac=1.0, random_state=1)                    # shuffle -> must be re-sorted
    pf = process_dataframe(df, detect_mapping(df).mapping, "dirty")
    assert pf.stats["duplicates_removed"] >= 45
    assert pf.stats["missing_values"] == 5
    assert np.isfinite(pf.signal).all() and np.isfinite(pf.windows.X).all()


def test_format_b_no_subject_no_label():
    t, a = acc_signal("walking", 700, 50, RNG)
    df = pd.DataFrame({"time": t, "acc_x": a[:, 0], "acc_y": a[:, 1], "acc_z": a[:, 2]})
    det = detect_mapping(df)
    assert det.mapping.subject is None and det.mapping.label is None and det.mapping.acc == ("acc_x", "acc_y", "acc_z")
    pf = process_dataframe(df, det.mapping, "B")
    assert pf.stats["n_subjects"] == 1 and not pf.has_labels and pf.windows.n == (700 - 128) // 64 + 1


def test_format_c_gravity_added_and_fs_required():
    t, a = acc_signal("standing", 500, 50, RNG)
    grav = np.tile([0.0, 1.0, 0.05], (500, 1))
    body = a - grav
    df = pd.DataFrame({"subject": "s1", "label": "std", "attitude": 0.1,
                       "gravity_x": grav[:, 0], "gravity_y": grav[:, 1], "gravity_z": grav[:, 2],
                       "body_acc_x": body[:, 0], "body_acc_y": body[:, 1], "body_acc_z": body[:, 2]})
    det = detect_mapping(df, "MotionSense")
    assert det.mapping.acc == ("body_acc_x", "body_acc_y", "body_acc_z")
    assert det.mapping.gravity == ("gravity_x", "gravity_y", "gravity_z")
    try:
        process_dataframe(df, det.mapping, "C")
        raise AssertionError("expected missing-fs error")
    except ValueError as e:
        assert "sampling frequency" in str(e)
    det.mapping.fs = 50
    pf = process_dataframe(df, det.mapping, "C")
    assert abs(np.linalg.norm(pf.signal[:, :3], axis=1).mean() - 1.0) < 0.1   # total acc ~ 1 g
    assert set(pf.windows.y) == {"standing"}


def test_format_d_long_with_gyro():
    rows = []
    for sub in ["a", "b"]:
        t, acc = acc_signal("walking", 600, 50, RNG)
        tg = np.arange(0, 600 / 50, 1 / 40)                       # gyro at 40 Hz, different timestamps
        gy = gyro_signal(len(tg), RNG)
        rows.append(pd.DataFrame({"subject_id": sub, "timestamp": t, "sensor_type": "accelerometer",
                                  "axis_x": acc[:, 0], "axis_y": acc[:, 1], "axis_z": acc[:, 2]}))
        rows.append(pd.DataFrame({"subject_id": sub, "timestamp": tg, "sensor_type": "gyroscope",
                                  "axis_x": gy[:, 0], "axis_y": gy[:, 1], "axis_z": gy[:, 2]}))
    df = pd.concat(rows, ignore_index=True)
    det = detect_mapping(df)
    m = det.mapping
    assert m.sensor_type_col == "sensor_type" and m.acc_sensor_value == "accelerometer" and m.gyro_sensor_value == "gyroscope"
    pf = process_dataframe(df, m, "D")
    assert pf.channels == ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    assert pf.windows.X.shape[2] == 6 and pf.stats["n_subjects"] == 2


def test_wisdm_headerless_with_semicolons():
    lines = []
    for i in range(300):
        lines.append(f"33,Walking,{1000000000 + i * 50000000},{0.1 * i % 3:.3f},9.5,{0.2:.3f};")
    df = read_csv_bytes("\n".join(lines).encode(), "WISDM")
    assert list(df.columns) == ["user", "activity", "timestamp", "x", "y", "z"]
    det = detect_mapping(df, "WISDM")
    pf = process_dataframe(df, det.mapping, "w", "WISDM")
    assert pf.stats["acc_unit"] == "m/s2" and abs(pf.stats["original_fs_hz"] - 20) < 1
    assert pf.windows.y[0] == "walking"


def test_time_parsing_variants():
    s, u, idx = parse_time_seconds(pd.Series(np.arange(0, 1, 0.02)))
    assert u == "s" and not idx
    s, u, idx = parse_time_seconds(pd.Series(np.arange(0, 1000, 20.0)))
    assert u == "ms"
    s, u, idx = parse_time_seconds(pd.Series(np.arange(0, 400)))
    assert idx
    s, u, _ = parse_time_seconds(pd.Series(pd.date_range("2024-01-01", periods=5, freq="20ms").astype(str)))
    assert u == "datetime" and abs(s[1] - 0.02) < 1e-6


def test_time_gap_splits_segments():
    t, a = acc_signal("walking", 1200, 50, RNG)
    t2 = t.copy(); t2[600:] += 30.0                                # 30 s hole
    df = pd.DataFrame({"time": t2, "acc_x": a[:, 0], "acc_y": a[:, 1], "acc_z": a[:, 2]})
    pf = process_dataframe(df, detect_mapping(df).mapping, "gap")
    assert pf.stats["n_segments"] == 2


def test_too_short_file_gives_warning_not_crash():
    t, a = acc_signal("walking", 60, 50, RNG)
    df = pd.DataFrame({"time": t, "acc_x": a[:, 0], "acc_y": a[:, 1], "acc_z": a[:, 2]})
    pf = process_dataframe(df, detect_mapping(df).mapping, "short")
    assert pf.windows.n == 0 and any("No windows" in w for w in pf.warnings)
