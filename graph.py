import argparse
import csv
import json
import os
import re
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


G = 9.81

if hasattr(np, "trapezoid"):
    integrate = np.trapezoid
else:
    integrate = np.trapz


def load_csv(path, fps=None):

    if not os.path.isfile(path):
        sys.exit(f"CSV file not found: {path}")

    frames = []
    times = []
    masses = []

    with open(path, "r", newline="", encoding="utf-8-sig") as file:

        reader = csv.DictReader(file)

        if not reader.fieldnames:
            sys.exit(f"CSV has no column headers: {path}")

        # Normalize column names
        headers = {
            name.strip().lower(): name
            for name in reader.fieldnames
            if name is not None
        }

        frame_col = headers.get("frame") or headers.get("frame_number")
        time_col = headers.get("time_s") or headers.get("timestamp_seconds")
        mass_col = headers.get("kg") or headers.get("mass_kg")

        if frame_col is None or mass_col is None:
            sys.exit(
                f"{path} must contain frame and kg columns. "
                f"Found: {reader.fieldnames}"
            )

        for line_number, row in enumerate(reader, start=2):

            try:
                frame_value = float(row[frame_col])
                mass_value = float(row[mass_col])

                if not np.isfinite(frame_value) or not np.isfinite(mass_value):
                    raise ValueError("Non-finite value")

                if mass_value < 0:
                    raise ValueError("Negative mass")

                if time_col and row.get(time_col, "").strip():
                    time_value = float(row[time_col])
                elif fps:
                    time_value = frame_value / fps
                else:
                    sys.exit(
                        f"{path}: time_s is missing at line {line_number}. "
                        "Provide --fps."
                    )

                if not np.isfinite(time_value):
                    raise ValueError("Non-finite timestamp")

                frames.append(frame_value)
                times.append(time_value)
                masses.append(mass_value)

            except (ValueError, TypeError) as error:
                sys.exit(
                    f"Invalid data in {path}, line {line_number}: {error}"
                )

    if len(masses) < 2:
        sys.exit(f"At least two valid readings are required in {path}.")

    t = np.asarray(times, dtype=float)
    kg = np.asarray(masses, dtype=float)

    # Sort readings by timestamp
    order = np.argsort(t)
    t = t[order]
    kg = kg[order]

    # Duplicate or decreasing timestamps are invalid
    if np.any(np.diff(t) <= 0):
        sys.exit(
            f"{path}: timestamps must be unique and strictly increasing."
        )

    return t, kg


def analyse(t, kg, threshold, refresh_hz, baseline_samples):

    if baseline_samples < 0:
        sys.exit("baseline-samples cannot be negative.")

    if baseline_samples >= len(kg):
        sys.exit("baseline-samples must be smaller than the number of readings.")

    # Convert indicated mass to equivalent force
    F = kg * G

    # Optional baseline correction
    if baseline_samples > 0:
        baseline = np.mean(F[:baseline_samples])
        F = F - baseline

    dt_ref = 1.0 / refresh_hz

    # Find readings above burn threshold
    above = np.where(F > threshold)[0]

    if len(above) == 0:
        sys.exit(
            "No reading exceeds the burn threshold. "
            "Check the readings or lower --threshold."
        )

    i0 = int(above[0])
    i1 = int(above[-1])

    # Extract burn interval
    tt = list(t[i0:i1 + 1])
    ff = list(F[i0:i1 + 1])

    t_start = float(t[i0])

    # Add a starting boundary
    if i0 > 0 and (t[i0] - t[i0 - 1]) <= dt_ref * 1.5:
        tt.insert(0, t[i0 - 1])
        ff.insert(0, F[i0 - 1])
    else:
        tt.insert(0, t[i0] - dt_ref)
        ff.insert(0, 0.0)

    # Add an ending boundary
    if i1 + 1 < len(t) and (t[i1 + 1] - t[i1]) <= dt_ref * 1.5:
        tt.append(t[i1 + 1])
        ff.append(F[i1 + 1])
        t_end = float(t[i1 + 1])
    else:
        tt.append(t[i1] + dt_ref)
        ff.append(0.0)
        t_end = float(t[i1] + dt_ref)

    tt = np.asarray(tt, dtype=float)
    ff = np.asarray(ff, dtype=float)

    # Total impulse using trapezoidal integration
    impulse_trap = float(integrate(ff, tt))

    # Sample-and-hold impulse estimate
    hold_t = np.append(t[i0:i1 + 1], t_end)
    impulse_hold = float(
        np.sum(F[i0:i1 + 1] * np.diff(hold_t))
    )

    # Performance parameters
    burn_time = t_end - t_start

    peak_index = i0 + int(np.argmax(F[i0:i1 + 1]))
    peak = float(F[peak_index])
    t_peak = float(t[peak_index])

    average_thrust = (
        impulse_trap / burn_time
        if burn_time > 0
        else float("nan")
    )

    results = {
        "peak_N": peak,
        "t_peak_s": t_peak,
        "burn_time_s": float(burn_time),
        "burn_time_uncert_s": float(dt_ref),
        "impulse_trapz_Ns": impulse_trap,
        "impulse_hold_Ns": impulse_hold,
        "avg_thrust_N": float(average_thrust),
        "n_burn_samples": int(i1 - i0 + 1),
        "t_start_s": t_start,
        "t_end_s": float(t_end),
    }

    return results, F, (tt, ff)


def pct(measured, published):

    if published == 0:
        return float("nan")

    return 100.0 * (measured - published) / published


def plot_single(t, F, curve, res, label, output_path):

    tt, ff = curve

    t0 = res["t_start_s"] - 0.25

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.fill_between(
        tt - t0,
        ff,
        alpha=0.25,
        label="Area = total impulse"
    )

    ax.plot(
        tt - t0,
        ff,
        "-",
        linewidth=1.5,
        label="Integrated curve"
    )

    ax.plot(
        t - t0,
        F,
        "o",
        markersize=4,
        label="Scale readings"
    )

    ax.axhline(
        res["avg_thrust_N"],
        linestyle="--",
        linewidth=1,
        color="gray",
        label=f"Average = {res['avg_thrust_N']:.2f} N"
    )

    ax.annotate(
        f"Peak = {res['peak_N']:.2f} N",
        (res["t_peak_s"] - t0, res["peak_N"]),
        textcoords="offset points",
        xytext=(10, 8)
    )

    ax.set_xlim(
        left=-0.25,
        right=res["t_end_s"] - t0 + 0.5
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Equivalent force (N)")

    ax.set_title(
        f"Thrust-Time Curve - {label}\n"
        f"Impulse = {res['impulse_trapz_Ns']:.2f} N.s, "
        f"Burn Time = {res['burn_time_s']:.2f} s"
    )

    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_overlay(runs, output_path):

    fig, ax = plt.subplots(figsize=(9, 5))

    for label, t, F, res in runs:

        t0 = res["t_start_s"] - 0.25

        ax.plot(
            t - t0,
            F,
            "o-",
            markersize=4,
            label=label
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Equivalent force (N)")
    ax.set_title("Thrust-Time Curves - Motor Comparison")

    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():

    parser = argparse.ArgumentParser(
        description="Analyse scale readings and estimate thrust performance."
    )

    parser.add_argument(
        "csv",
        nargs="+",
        help="One or more readings.csv files"
    )

    parser.add_argument(
        "--labels",
        nargs="+",
        help="Labels for each CSV"
    )

    parser.add_argument(
        "--fps",
        type=float,
        help="Video frame rate if timestamps are absent"
    )

    parser.add_argument(
        "--refresh-hz",
        type=float,
        default=4.0,
        help="Scale display update rate (default: 4 Hz)"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Burn threshold in N (default: 0.3)"
    )

    parser.add_argument(
        "--baseline-samples",
        type=int,
        default=0,
        help="Number of initial readings used for baseline correction"
    )

    parser.add_argument("--published-impulse", type=float)
    parser.add_argument("--published-burn", type=float)
    parser.add_argument("--published-peak", type=float)

    parser.add_argument(
        "--outdir",
        help="Output directory (default: next to the first CSV)"
    )

    args = parser.parse_args()

    if args.refresh_hz <= 0:
        parser.error("--refresh-hz must be greater than zero.")

    if args.threshold < 0:
        parser.error("--threshold cannot be negative.")

    for value, option in [
        (args.published_impulse, "--published-impulse"),
        (args.published_burn, "--published-burn"),
        (args.published_peak, "--published-peak")
    ]:
        if value is not None and value <= 0:
            parser.error(f"{option} must be greater than zero.")

    labels = args.labels or [
        f"Run {i + 1}" for i in range(len(args.csv))
    ]

    if len(labels) != len(args.csv):
        parser.error("Give one --labels entry per CSV.")

    output_dir = args.outdir or os.path.dirname(
        os.path.abspath(args.csv[0])
    )

    os.makedirs(output_dir, exist_ok=True)

    runs = []
    summary = []

    for path, label in zip(args.csv, labels):

        t, kg = load_csv(path, args.fps)

        res, F, curve = analyse(
            t,
            kg,
            args.threshold,
            args.refresh_hz,
            args.baseline_samples
        )

        stem = re.sub(
            r"[^a-zA-Z0-9_-]+",
            "_",
            label.lower()
        ).strip("_")

        plot_single(
            t,
            F,
            curve,
            res,
            label,
            os.path.join(output_dir, f"{stem}_thrust_curve.png")
        )

        json_path = os.path.join(
            output_dir,
            f"{stem}_results.json"
        )

        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(res, file, indent=4)

        print(f"\n===== {label} =====")
        print(f"Input file       : {path}")
        print(f"Readings         : {len(t)}")
        print(f"Peak force       : {res['peak_N']:.2f} N")
        print(f"Peak time        : {res['t_peak_s']:.3f} s")
        print(
            f"Burn time        : {res['burn_time_s']:.2f} s "
            f"(approx. ±{res['burn_time_uncert_s']:.2f} s)"
        )
        print(
            f"Total impulse    : {res['impulse_trapz_Ns']:.2f} N.s "
            "(trapezoidal)"
        )
        print(
            f"Sample-hold      : {res['impulse_hold_Ns']:.2f} N.s"
        )
        print(f"Average force    : {res['avg_thrust_N']:.2f} N")
        print(f"Burn readings    : {res['n_burn_samples']}")

        if res["n_burn_samples"] < 5:
            print(
                "WARNING: Very few readings during the burn. "
                "Performance estimates may be unreliable."
            )

        if args.published_impulse is not None:
            print(
                "Impulse difference: "
                f"{pct(res['impulse_trapz_Ns'], args.published_impulse):+.2f}%"
            )

        if args.published_burn is not None:
            print(
                "Burn-time difference: "
                f"{pct(res['burn_time_s'], args.published_burn):+.2f}%"
            )

        if args.published_peak is not None:
            print(
                "Peak difference: "
                f"{pct(res['peak_N'], args.published_peak):+.2f}%"
            )

        runs.append((label, t, F, res))
        summary.append({"label": label, **res})

    if len(runs) > 1:
        plot_overlay(
            runs,
            os.path.join(output_dir, "all_motors_overlay.png")
        )

    summary_path = os.path.join(output_dir, "summary.csv")

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(summary[0].keys())
        )

        writer.writeheader()
        writer.writerows(summary)

    print("\nAnalysis completed.")
    print(f"All output files saved in: {output_dir}")


if __name__ == "__main__":
    main()