#!/usr/bin/env python
"""
Turn the PNG sequences Fusion rendered into video.

    python cad/encode.py                 # encode every scene + the combined cut
    python cad/encode.py --fps 24 --gif  # also write a looping GIF per scene

Scenes are read from cad/out/anim/<scene>/frame_####.png and written to
cad/out/video/.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ANIM = os.path.join(HERE, "out", "anim")
VIDEO = os.path.join(HERE, "out", "video")

# Order matters: this is the cut of the final explainer.
SCENE_ORDER = ["orbit", "index", "explode"]


def ffmpeg_bin():
    exe = shutil.which("ffmpeg")
    if not exe:
        sys.exit("ffmpeg not found on PATH")
    return exe


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        tail = "\n".join(p.stderr.strip().splitlines()[-12:])
        raise RuntimeError("ffmpeg failed:\n{}".format(tail))
    return p


def scenes_present():
    if not os.path.isdir(ANIM):
        return []
    found = []
    for name in SCENE_ORDER + sorted(os.listdir(ANIM)):
        if name in [f[0] for f in found]:
            continue
        d = os.path.join(ANIM, name)
        if not os.path.isdir(d):
            continue
        n = len(glob.glob(os.path.join(d, "frame_*.png")))
        if n:
            found.append((name, d, n))
    return found


def encode_scene(ff, name, d, fps, gif):
    os.makedirs(VIDEO, exist_ok=True)
    mp4 = os.path.join(VIDEO, "{}.mp4".format(name))
    run([ff, "-y", "-framerate", str(fps),
         "-i", os.path.join(d, "frame_%04d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
         "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
         mp4])
    out = {"scene": name, "mp4": mp4, "bytes": os.path.getsize(mp4)}

    if gif:
        pal = os.path.join(VIDEO, "_{}_pal.png".format(name))
        g = os.path.join(VIDEO, "{}.gif".format(name))
        run([ff, "-y", "-framerate", str(fps),
             "-i", os.path.join(d, "frame_%04d.png"),
             "-vf", "fps={},scale=720:-1:flags=lanczos,palettegen".format(fps),
             pal])
        run([ff, "-y", "-framerate", str(fps),
             "-i", os.path.join(d, "frame_%04d.png"), "-i", pal,
             "-lavfi", "fps={},scale=720:-1:flags=lanczos[x];[x][1:v]paletteuse".format(fps),
             "-loop", "0", g])
        os.remove(pal)
        out["gif"] = g
        out["gif_bytes"] = os.path.getsize(g)
    return out


def combine(ff, parts, fps):
    listing = os.path.join(VIDEO, "_concat.txt")
    with open(listing, "w", encoding="utf-8") as fh:
        for p in parts:
            fh.write("file '{}'\n".format(p["mp4"].replace("\\", "/")))
    final = os.path.join(VIDEO, "FreshIQ_explainer.mp4")
    run([ff, "-y", "-f", "concat", "-safe", "0", "-i", listing,
         "-c", "copy", final])
    os.remove(listing)
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--gif", action="store_true")
    a = ap.parse_args()

    ff = ffmpeg_bin()
    found = scenes_present()
    if not found:
        sys.exit("no frames in {} - run the animate command first".format(ANIM))

    parts = []
    for name, d, n in found:
        print("encoding {:<10} {:>4} frames".format(name, n))
        parts.append(encode_scene(ff, name, d, a.fps, a.gif))

    for p in parts:
        line = "  {:<10} {:>8.1f} KB  {}".format(
            p["scene"], p["bytes"] / 1024.0, p["mp4"])
        if "gif" in p:
            line += "\n  {:<10} {:>8.1f} KB  {}".format(
                "", p["gif_bytes"] / 1024.0, p["gif"])
        print(line)

    if len(parts) > 1:
        final = combine(ff, parts, a.fps)
        print("\ncombined -> {} ({:.1f} KB)".format(
            final, os.path.getsize(final) / 1024.0))


if __name__ == "__main__":
    main()
