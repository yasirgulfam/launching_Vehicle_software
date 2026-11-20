import argparse
import csv
import os
import sys

import cv2


def load_frames(folder):
    csv_path = os.path.join(folder, "frames.csv")

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        return [
            (
                int(row["Frame_Number"]),
                float(row["Timestamp_Seconds"]),
                row["Image_File"]
            )
            for row in reader
        ]


def load_readings(path):

    entries = {}

    if os.path.exists(path):

        with open(path, newline="") as f:

            for r in csv.DictReader(f):

                try:
                    entries[int(r["frame"])] = (
                        float(r["time_s"]),
                        float(r["kg"])
                    )

                except (ValueError, KeyError):
                    continue

    return entries


def save_readings(path, entries):

    with open(path, "w", newline="") as f:

        wr = csv.writer(f)

        wr.writerow(["frame", "time_s", "kg"])

        for fr in sorted(entries):

            t, kg = entries[fr]

            wr.writerow([
                fr,
                f"{t:.4f}",
                f"{kg:.4f}"
            ])


def main():

    p = argparse.ArgumentParser(
        description="Enter weighing-scale readings from cropped frames."
    )

    p.add_argument(
        "folder",
        help="Run folder created by the frame extraction script"
    )

    p.add_argument(
        "--unit",
        choices=["kg", "g"],
        default="g",
        help="Unit shown on the display (default: grams)"
    )

    args = p.parse_args()

    if not os.path.isdir(args.folder):
        sys.exit(f"Folder not found: {args.folder}")

    frames = load_frames(args.folder)

    if not frames:
        sys.exit("No frames found in frames.csv.")

    out_path = os.path.join(
        args.folder,
        "readings.csv"
    )

    entries = load_readings(out_path)

    # Resume at first frame without an entry
    i = next(
        (
            k for k, fr in enumerate(frames)
            if fr[0] not in entries
        ),
        len(frames)
    )

    win = "Scale display"

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while 0 <= i < len(frames):

        frame_no, t, fname = frames[i]

        img_path = os.path.join(
            args.folder,
            "crops",
            fname
        )

        img = cv2.imread(img_path)

        if img is None:

            print(f"Missing image: {fname}")
            i += 1
            continue

        h, w = img.shape[:2]

        if w < 500:

            s = 500 / w

            img = cv2.resize(
                img,
                None,
                fx=s,
                fy=s,
                interpolation=cv2.INTER_CUBIC
            )

        cv2.imshow(win, img)
        cv2.waitKey(1)

        prev = (
            f" [current: {entries[frame_no][1]:.4f} kg]"
            if frame_no in entries
            else ""
        )

        ans = input(
            f"({i + 1}/{len(frames)}) "
            f"Frame {frame_no}, "
            f"Time={t:.3f}s{prev} -> "
        ).strip().lower()

        if ans == "q":
            break

        if ans == "b":

            if i > 0:

                i -= 1

                entries.pop(
                    frames[i][0],
                    None
                )

                save_readings(out_path, entries)

            continue

        if ans == "":
            i += 1
            continue

        try:

            val = float(ans)

            if val < 0:
                print("Mass cannot be negative.")
                continue

        except ValueError:

            print(
                "Invalid input. Enter a number, "
                "blank, b or q."
            )

            continue

        # Convert grams to kilograms
        kg = val / 1000.0 if args.unit == "g" else val

        entries[frame_no] = (t, kg)

        save_readings(out_path, entries)

        i += 1

    cv2.destroyAllWindows()

    save_readings(out_path, entries)

    print(
        f"\nSaved {len(entries)} readings to {out_path}"
    )

    if i >= len(frames):
        print("All frames have been processed.")

    print(
        "Next: python 3_analyze_thrust.py",
        out_path
    )


if __name__ == "__main__":
    main()