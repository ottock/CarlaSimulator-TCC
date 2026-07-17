"""Dataset quality report.

Checks the things that make or break a behavioral-cloning dataset: the steering
distribution (guard against "90% straight") and whether braking/recovery frames
exist. Also prints a text histogram and can dump a few steer-overlaid frames to
eyeball image/label synchronization.
"""
import csv
import glob
import os

import numpy as np

_STRAIGHT_THRESHOLD = 0.05
_BRAKE_THRESHOLD = 0.05


def _is_true(value):
    return str(value).strip().lower() in ("true", "1", "1.0", "yes")


def summarize_labels(labels):
    """Aggregate a list of label dicts into headline percentages."""
    n = len(labels)
    if n == 0:
        return {"frames": 0, "pct_straight": 0.0, "pct_brake": 0.0,
                "pct_noise": 0.0, "mean_speed": 0.0, "steer_min": 0.0, "steer_max": 0.0}
    steers = np.array([float(l["steer"]) for l in labels])
    brakes = np.array([float(l["brake"]) for l in labels])
    speeds = np.array([float(l["v"]) for l in labels])
    noise = np.array([_is_true(l.get("noise_active", False)) for l in labels])
    return {
        "frames": n,
        "pct_straight": float(np.mean(np.abs(steers) < _STRAIGHT_THRESHOLD) * 100.0),
        "pct_brake": float(np.mean(brakes > _BRAKE_THRESHOLD) * 100.0),
        "pct_noise": float(np.mean(noise) * 100.0),
        "mean_speed": float(speeds.mean()),
        "steer_min": float(steers.min()),
        "steer_max": float(steers.max()),
    }


def steer_histogram(steers, n_bins=21, lo=-1.0, hi=1.0):
    """Return ``(edges, counts)`` for the steering distribution."""
    counts, edges = np.histogram(np.asarray(steers, dtype=np.float64), bins=n_bins, range=(lo, hi))
    return edges, counts


def _read_episode_labels(episode_dir):
    path = os.path.join(episode_dir, "labels.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def dataset_report(dataset_dir):
    """Aggregate every ``ep_*`` episode's labels into one report dict."""
    episodes = sorted(glob.glob(os.path.join(dataset_dir, "ep_*")))
    labels = []
    for ep in episodes:
        labels.extend(_read_episode_labels(ep))
    report = summarize_labels(labels)
    report["num_episodes"] = len(episodes)
    if labels:
        edges, counts = steer_histogram([float(l["steer"]) for l in labels])
        report["steer_hist_edges"] = edges.tolist()
        report["steer_hist_counts"] = counts.tolist()
    return report


def print_report(report):
    """Pretty-print a report dict, including an ASCII steering histogram."""
    print("Episodes: %d   Frames: %d" % (report.get("num_episodes", 0), report["frames"]))
    print("Straight (|steer|<0.05): %.1f%%   Braking: %.1f%%   Noise-active: %.1f%%"
          % (report["pct_straight"], report["pct_brake"], report["pct_noise"]))
    print("Mean speed: %.2f m/s   Steer range: [%.2f, %.2f]"
          % (report["mean_speed"], report["steer_min"], report["steer_max"]))
    counts = report.get("steer_hist_counts")
    edges = report.get("steer_hist_edges")
    if counts:
        peak = max(counts) or 1
        print("Steering histogram:")
        for i, c in enumerate(counts):
            bar = "#" * int(40 * c / peak)
            print("  [%+.2f, %+.2f) %6d |%s" % (edges[i], edges[i + 1], c, bar))
