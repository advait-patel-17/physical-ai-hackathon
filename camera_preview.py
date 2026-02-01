#!/usr/bin/env python3
"""Open all available cameras in OpenCV to help identify which index is which."""

import cv2
import sys


def find_cameras(max_index=10):
    """Find all available camera indices."""
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available


def main():
    print("Scanning for cameras...")
    cameras = find_cameras()

    if not cameras:
        print("No cameras found!")
        sys.exit(1)

    print(f"Found {len(cameras)} camera(s): {cameras}")
    print("\nOpening all cameras. Press 'q' to quit, 'n' for next camera info.\n")

    caps = {}
    for idx in cameras:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            caps[idx] = cap
            # Print camera properties
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            backend = cap.getBackendName()
            print(f"Camera {idx}: {w}x{h} @ {fps:.1f}fps (backend: {backend})")

    if not caps:
        print("Failed to open any cameras!")
        sys.exit(1)

    print("\n[Press 'q' to quit]\n")

    while True:
        for idx, cap in caps.items():
            ret, frame = cap.read()
            if ret:
                # Add label to frame
                label = f"Camera {idx}"
                cv2.putText(
                    frame,
                    label,
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    3,
                )
                cv2.imshow(label, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Cleanup
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
