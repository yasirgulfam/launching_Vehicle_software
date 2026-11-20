import argparse
import csv
import json
import os
import sys

import cv2


def parse_args():

    p = argparse.ArgumentParser(
        description="Track and crop the weighing-scale display."
    )

    p.add_argument(
        "video",
        nargs="?",
        default=r"C:\Users\Admin\OneDrive\Videos\ScreenRecorderFiles\WhatsApp Video 2026-11-15 at 18.02.41.mp4"
    )

    p.add_argument("--out", default="run2")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=None)
    p.add_argument("--roi", help="x,y,w,h")
    p.add_argument("--roi-file", help="Previously saved ROI JSON")
    p.add_argument("--use-timestamp", action="store_true")
    p.add_argument("--upscale", type=float, default=2.0)

    return p.parse_args()


def create_tracker():

    if hasattr(cv2, "legacy") and hasattr(
        cv2.legacy, "TrackerCSRT_create"
    ):
        return cv2.legacy.TrackerCSRT_create()

    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()

    sys.exit(
        "CSRT tracker unavailable. Install opencv-contrib-python."
    )


def choose_frame(cap, total, start_idx):

    win = "Select a clear reference frame - Press SPACE"

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    cv2.createTrackbar(
        "frame", win,
        start_idx,
        max(total - 1, 1),
        lambda v: None
    )

    last = -1
    img = None

    while True:

        idx = cv2.getTrackbarPos("frame", win)

        # Do not select a frame before the requested start
        idx = max(idx, start_idx)

        if idx != last:

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()

            if ok:

                img = frame.copy()

                cv2.putText(
                    img,
                    f"Frame: {idx}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )

            last = idx

        if img is not None:
            cv2.imshow(win, img)

        key = cv2.waitKey(30) & 0xFF

        if key in (32, 13):
            break

        if key == 27:
            cv2.destroyAllWindows()
            sys.exit("Cancelled.")

    cv2.destroyWindow(win)

    cap.set(cv2.CAP_PROP_POS_FRAMES, last)
    ok, frame = cap.read()

    if not ok:
        return None, None

    return frame, last


def get_roi(args, cap, total, start_idx):

    if args.roi:

        x, y, w, h = [
            int(v) for v in args.roi.split(",")
        ]

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        ok, frame = cap.read()

        if not ok:
            sys.exit("Cannot read reference frame.")

        return x, y, w, h, frame, start_idx

    if args.roi_file:

        with open(args.roi_file) as f:
            r = json.load(f)

        ref_idx = r.get("reference_frame", start_idx)

        cap.set(cv2.CAP_PROP_POS_FRAMES, ref_idx)
        ok, frame = cap.read()

        if not ok:
            sys.exit("Cannot read saved reference frame.")

        return (
            r["x"], r["y"], r["w"], r["h"],
            frame, ref_idx
        )

    frame, ref_idx = choose_frame(
        cap, total, start_idx
    )

    if frame is None:
        sys.exit("Cannot read selected frame.")

    x, y, w, h = cv2.selectROI(
        "Select LCD Display",
        frame,
        showCrosshair=False
    )

    cv2.destroyAllWindows()

    if w == 0 or h == 0:
        sys.exit("No ROI selected.")

    return (
        int(x), int(y), int(w), int(h),
        frame, ref_idx
    )


def main():

    args = parse_args()

    if args.start < 0:
        sys.exit("Start time cannot be negative.")

    if args.upscale <= 0:
        sys.exit("Upscale must be positive.")

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        sys.exit("Cannot open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total <= 0:
        cap.release()
        sys.exit("Invalid video properties.")

    start_idx = min(int(args.start * fps), total - 1)

    end_idx = (
        total - 1
        if args.end is None
        else min(int(args.end * fps), total - 1)
    )

    if end_idx < start_idx:
        cap.release()
        sys.exit("Invalid start/end time.")

    print(
        f"Video: {total} frames at {fps:.2f} FPS "
        f"({total / fps:.2f} seconds)"
    )

    # Select reference frame and display ROI
    x, y, w, h, reference_frame, ref_idx = get_roi(
        args, cap, total, start_idx
    )

    height, width = reference_frame.shape[:2]

    if (
        x < 0 or y < 0 or
        w <= 0 or h <= 0 or
        x + w > width or y + h > height
    ):
        cap.release()
        sys.exit("ROI is outside the video frame.")

    if ref_idx > end_idx:
        cap.release()
        sys.exit("Selected reference frame is after end time.")

    # Initialize CSRT tracker
    tracker = create_tracker()

    tracker.init(
        reference_frame,
        (x, y, w, h)
    )

    print("CSRT tracker initialized successfully.")

    # Output folders
    crops_dir = os.path.join(args.out, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    # Save reference ROI information
    with open(
        os.path.join(args.out, "roi.json"), "w"
    ) as f:

        json.dump(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "fps": fps,
                "video": args.video,
                "reference_frame": ref_idx
            },
            f,
            indent=4
        )

    rows = []

    # Save the reference frame crop
    reference_crop = reference_frame[y:y+h, x:x+w]

    if args.upscale != 1.0:

        reference_crop = cv2.resize(
            reference_crop,
            None,
            fx=args.upscale,
            fy=args.upscale,
            interpolation=cv2.INTER_CUBIC
        )

    reference_name = f"frame_{ref_idx:06d}.png"

    cv2.imwrite(
        os.path.join(crops_dir, reference_name),
        reference_crop
    )

    reference_time = (
        cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if args.use_timestamp
        else ref_idx / fps
    )

    rows.append(
        (ref_idx, f"{reference_time:.4f}",
         reference_name, "OK")
    )

    # Continue processing from the frame after reference
    idx = ref_idx + 1

    while idx <= end_idx:

        ok, frame = cap.read()

        if not ok:
            break

        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        success, bbox = tracker.update(frame)

        timestamp = (
            pos_ms / 1000.0
            if args.use_timestamp
            else idx / fps
        )

        if success:

            tx, ty, tw, th = bbox

            x1 = max(0, int(tx))
            y1 = max(0, int(ty))
            x2 = min(width, int(tx + tw))
            y2 = min(height, int(ty + th))

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:

                rows.append(
                    (idx, f"{timestamp:.4f}",
                     "", "INVALID_CROP")
                )

            else:

                if args.upscale != 1.0:

                    crop = cv2.resize(
                        crop,
                        None,
                        fx=args.upscale,
                        fy=args.upscale,
                        interpolation=cv2.INTER_CUBIC
                    )

                name = f"frame_{idx:06d}.png"

                saved = cv2.imwrite(
                    os.path.join(crops_dir, name),
                    crop
                )

                if saved:

                    rows.append(
                        (idx, f"{timestamp:.4f}",
                         name, "OK")
                    )

                else:

                    rows.append(
                        (idx, f"{timestamp:.4f}",
                         "", "SAVE_FAIL")
                    )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.imshow("Tracked LCD", crop)

        else:

            rows.append(
                (idx, f"{timestamp:.4f}",
                 "", "TRACK_FAIL")
            )

        cv2.imshow("Video Tracking", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # Save CSV
    with open(
        os.path.join(args.out, "frames.csv"),
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Frame_Number",
            "Timestamp_Seconds",
            "Image_File",
            "Tracking_Status"
        ])

        writer.writerows(rows)

    successful = sum(
        1 for row in rows if row[3] == "OK"
    )

    print("Processing completed.")
    print("Frames processed:", len(rows))
    print("Successfully tracked:", successful)
    print("Crops folder:", crops_dir)
    print("CSV:", os.path.join(args.out, "frames.csv"))


if __name__ == "__main__":
    main()