"""
Generate synthetic sample images so the pipeline can be smoke-tested without a
real camera or dataset. These are crude coloured blobs, NOT real produce — they
exist only to exercise the code paths (fresh / spotted / mouldy / bad-quality).

Usage:
    python scripts/make_sample_data.py
"""

import os
import cv2
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "sample_data")
os.makedirs(OUT, exist_ok=True)


def base_tray(bg=(180, 180, 180)):
    img = np.full((480, 640, 3), bg, dtype=np.uint8)
    # a little sensor noise so Otsu/quality metrics behave realistically
    noise = np.random.normal(0, 6, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def draw_item(img, color_bgr, center=(320, 240), radius=150):
    cv2.circle(img, center, radius, color_bgr, -1)
    # soft shading for a bit of texture
    overlay = img.copy()
    cv2.circle(overlay, (center[0] - 40, center[1] - 40), radius // 2,
               tuple(min(255, c + 40) for c in color_bgr), -1)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
    return img


def add_dark_spots(img, center, radius, n=14, spot_r=14):
    for _ in range(n):
        ang = np.random.uniform(0, 2 * np.pi)
        rr = np.random.uniform(0, radius * 0.8)
        x = int(center[0] + rr * np.cos(ang))
        y = int(center[1] + rr * np.sin(ang))
        cv2.circle(img, (x, y), np.random.randint(spot_r // 2, spot_r), (20, 20, 25), -1)
    return img


def add_mold(img, center, radius):
    # greyish-green fuzzy low-saturation patch
    for _ in range(400):
        ang = np.random.uniform(0, 2 * np.pi)
        rr = np.random.uniform(0, radius * 0.5)
        x = int(center[0] + radius * 0.4 + rr * np.cos(ang))
        y = int(center[1] + rr * np.sin(ang))
        g = np.random.randint(130, 180)
        cv2.circle(img, (x, y), np.random.randint(1, 4), (g, g, g - 20), -1)
    return img


def save(name, img):
    path = os.path.join(OUT, name)
    cv2.imwrite(path, img)
    print("wrote", os.path.relpath(path))


np.random.seed(42)

# Fresh red tomato
img = base_tray()
img = draw_item(img, (40, 40, 210))  # BGR red
save("tomato_fresh.jpg", img)

# Ripening / spotted (front view fine-ish)
img = base_tray()
img = draw_item(img, (40, 40, 205))
add_dark_spots(img, (320, 240), 150, n=8, spot_r=10)
save("tomato_front.jpg", img)

# Same item, back view with mould -> should drive the aggregate to spoiled
img = base_tray()
img = draw_item(img, (40, 40, 200))
add_mold(img, (320, 240), 150)
add_dark_spots(img, (320, 240), 150, n=20, spot_r=16)
save("tomato_back.jpg", img)

# Heavily spoiled
img = base_tray()
img = draw_item(img, (35, 45, 120))
add_dark_spots(img, (320, 240), 150, n=40, spot_r=20)
add_mold(img, (320, 240), 150)
save("tomato_spoiled.jpg", img)

# Fresh banana (yellow)
img = base_tray()
img = draw_item(img, (60, 200, 220), radius=140)
save("banana_fresh.jpg", img)

# Too dark (bad quality)
img = (base_tray() * 0.08).astype(np.uint8)
img = draw_item(img, (10, 10, 25))
save("bad_too_dark.jpg", img)

# Blurred (bad quality)
img = base_tray()
img = draw_item(img, (40, 40, 210))
img = cv2.GaussianBlur(img, (45, 45), 0)
save("bad_blurred.jpg", img)

print("Done.")
