import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline import TrackObservation, temporal_consensus, overlap_ratio, severe_overlap, eta_from_position


def obs(track, cls, conf):
    return TrackObservation(track, cls, str(cls), conf, 10, 10, 20, 100, 1.0)


def test_temporal_consensus():
    winner, confidence, ratio, count = temporal_consensus([obs(1, 0, .8), obs(1, 0, .9), obs(1, 1, .6)])
    assert winner == 0
    assert count == 3
    assert ratio == 2/3
    assert confidence == .85


def test_iou_and_overlap_gate():
    assert overlap_ratio((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert severe_overlap((0, 0, 10, 10), [(2, 2, 12, 12)], .4)


def test_eta_with_latency():
    assert eta_from_position(1.0, 0.5, 0.2) == 1.8
