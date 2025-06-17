#!/usr/bin/env python3
"""
Optimized Precise Cut Point Video Matcher with Multiple Detection Methods
Extracts frames at cut boundaries and finds exact matches in original video
WITH ENHANCED VERBOSE OUTPUT
"""

import cv2
import numpy as np
import os
import sys
from datetime import timedelta
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

def install_requirements():
    try:
        import cv2
        import numpy as np
        print("✅ Required packages already installed")
    except ImportError:
        print("Installing required packages...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python", "numpy"])
            print("✅ Packages installed successfully")
        except Exception as e:
            print(f"❌ Failed to install packages: {e}")
            print("Please run manually: pip3 install opencv-python numpy")
            sys.exit(1)

def get_video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise Exception(f"Failed to open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration": duration,
        "width": width,
        "height": height,
    }

def read_frame(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    return frame if ret else None

def verbose_adaptive_cut_detection(video_path, initial_threshold=0.965, max_segments=20):
    print("🔍 Starting adaptive cut detection...")
    print(f"   Video: {os.path.basename(video_path)}")
    print(f"   Initial threshold: {initial_threshold}")
    print(f"   Max segments: {max_segments}")

    info = get_video_info(video_path)
    fps = info["fps"]
    total_frames = info["total_frames"]
    duration = info["duration"]

    print(f"📊 Video info:")
    print(f"   FPS: {fps:.2f}")
    print(f"   Total frames: {total_frames:,}")
    print(f"   Duration: {timedelta(seconds=int(duration))}")

    print("\n🔍 Starting quick analysis pass...")
    similarities = []
    skip_frames = max(int(fps * 2), 1)
    print(f"   Skip frames: {skip_frames} (analyzing every {skip_frames} frames)")

    cap = cv2.VideoCapture(video_path)
    last_gray = None
    frame_count = 0
    start_time = time.time()
    last_update = start_time

    while True:
        frame = read_frame(cap, frame_count)
        if frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if last_gray is not None:
            sim = np.mean(gray == last_gray)
            similarities.append(sim)
        last_gray = gray
        frame_count += skip_frames
        if frame_count % (skip_frames * 10) == 0 or time.time() - last_update > 2:
            progress = (frame_count / total_frames) * 100
            print(f"   Quick analysis: {progress:.1f}% ({frame_count:,}/{total_frames:,}) | {len(similarities)} comparisons")
            last_update = time.time()
    cap.release()
    analysis_time = time.time() - start_time
    print(f"✅ Quick analysis complete in {analysis_time:.1f}s")
    if not similarities:
        print("❌ No similarities calculated")
        return []

    print(f"\n📊 Similarity analysis (from {len(similarities)} comparisons):")
    print(f"   Min: {np.min(similarities):.4f}")
    print(f"   Max: {np.max(similarities):.4f}")
    print(f"   Mean: {np.mean(similarities):.4f}")
    print(f"   Std: {np.std(similarities):.4f}")

    percentiles = [5, 10, 15, 20]
    for p in percentiles:
        thresh = np.percentile(similarities, p)
        potential_cuts = np.sum(np.array(similarities) < thresh)
        print(f"   {p:2d}th percentile: {thresh:.4f} ({potential_cuts:3d} potential cuts)")

    chosen_threshold = None
    for p in percentiles:
        thresh = np.percentile(similarities, p)
        cuts = np.sum(np.array(similarities) < thresh)
        if cuts <= max_segments:
            chosen_threshold = thresh
            print(f"✅ Using {p}th percentile threshold: {chosen_threshold:.4f}")
            break
    if chosen_threshold is None:
        chosen_threshold = np.percentile(similarities, 10)
        print(f"⚠️  Using 10th percentile as fallback: {chosen_threshold:.4f}")

    print(f"\n🔍 Starting detailed cut detection with threshold {chosen_threshold:.4f}...")

    cap = cv2.VideoCapture(video_path)
    last_gray = None
    similarities_full = []
    cut_points = [0]
    frame_count = 0
    last_update = time.time()
    while True:
        frame = read_frame(cap, frame_count)
        if frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if last_gray is not None:
            sim = np.mean(gray == last_gray)
            similarities_full.append(sim)
            if sim < chosen_threshold:
                cut_points.append(frame_count)
                print(f"🎬 CUT DETECTED at {timedelta(seconds=int(frame_count / fps))} (sim={sim:.4f})")
        last_gray = gray
        frame_count += 1
        if frame_count % 100 == 0 or time.time() - last_update > 2:
            progress = (frame_count / total_frames) * 100
            print(f"   Progress: {progress:.1f}% ({frame_count:,}/{total_frames:,})")
            last_update = time.time()
    cap.release()
    if cut_points[-1] != total_frames:
        cut_points.append(total_frames)
    print(f"✅ Adaptive method found {len(cut_points) - 1} segments!")
    return cut_points

def verbose_similarity_drops_detection(video_path, drop_threshold=0.07, base_threshold=0.97):
    print(f"\n🔍 METHOD 2: Similarity drops detection")
    print(f"   Drop threshold: {drop_threshold}")
    print(f"   Base threshold: {base_threshold}")

    info = get_video_info(video_path)
    fps = info["fps"]
    total_frames = info["total_frames"]
    print(f"📊 Processing {total_frames:,} frames at {fps:.2f} fps...")

    cap = cv2.VideoCapture(video_path)
    last_gray = None
    last_sim = 1.0
    cut_points = [0]
    frame_count = 0
    similarities = []
    last_update = time.time()
    while True:
        frame = read_frame(cap, frame_count)
        if frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if last_gray is not None:
            sim = np.mean(gray == last_gray)
            drop = last_sim - sim
            similarities.append(sim)
            if sim < base_threshold or drop > drop_threshold:
                cut_points.append(frame_count)
                print(f"🎬 CUT DETECTED at {timedelta(seconds=int(frame_count / fps))} (sim={sim:.4f}, drop={drop:.4f})")
            last_sim = sim
        last_gray = gray
        frame_count += 1
        if frame_count % 100 == 0 or time.time() - last_update > 2:
            progress = (frame_count / total_frames) * 100
            print(f"   Progress: {progress:.1f}% ({frame_count:,}/{total_frames:,})")
            last_update = time.time()
    cap.release()
    if cut_points[-1] != total_frames:
        cut_points.append(total_frames)
    print(f"✅ Method 2 found {len(cut_points) - 1} segments!")
    return cut_points

def verbose_statistical_cut_detection(video_path, percentile_threshold=10):
    print(f"\n🔍 METHOD 3: Statistical detection")
    print(f"   Using {percentile_threshold}th percentile threshold")

    info = get_video_info(video_path)
    fps = info["fps"]
    total_frames = info["total_frames"]
    print(f"📊 Processing {total_frames:,} frames...")

    cap = cv2.VideoCapture(video_path)
    last_gray = None
    similarities = []
    frame_count = 0
    last_update = time.time()
    print("🔍 First pass: collecting similarity data...")
    while True:
        frame = read_frame(cap, frame_count)
        if frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if last_gray is not None:
            sim = np.mean(gray == last_gray)
            similarities.append(sim)
        last_gray = gray
        frame_count += 1
        if frame_count % 100 == 0 or time.time() - last_update > 2:
            progress = (frame_count / total_frames) * 100
            print(f"   First pass: {progress:.1f}% ({frame_count:,}/{total_frames:,}) | {len(similarities)} similarities")
            last_update = time.time()
    cap.release()

    if not similarities:
        print("❌ No similarities calculated")
        return []

    threshold = np.percentile(similarities, percentile_threshold)
    print(f"\n📊 Statistical analysis:")
    print(f"   Calculated threshold: {threshold:.3f} ({percentile_threshold}th percentile)")
    print(f"   Similarity range: {np.min(similarities):.3f} to {np.max(similarities):.3f}")
    print(f"   Mean similarity: {np.mean(similarities):.3f}")

    print(f"\n🔍 Finding cuts below threshold {threshold:.3f}...")
    cap = cv2.VideoCapture(video_path)
    last_gray = None
    cut_points = [0]
    frame_count = 0
    while True:
        frame = read_frame(cap, frame_count)
        if frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if last_gray is not None:
            sim = np.mean(gray == last_gray)
            if sim < threshold:
                cut_points.append(frame_count)
                print(f"🎬 Cut {len(cut_points) - 1} at {timedelta(seconds=int(frame_count / fps))} (similarity: {sim:.3f})")
        last_gray = gray
        frame_count += 1
    cap.release()
    if cut_points[-1] != total_frames:
        cut_points.append(total_frames)
    print(f"✅ Method 3 found {len(cut_points) - 1} segments!")
    return cut_points

def verbose_extract_boundary_frames(video_path, cut_points, samples_per_cut=3):
    print(f"\n🎞️  Extracting boundary frames from {os.path.basename(video_path)}...")
    print(f"   Processing {len(cut_points) - 1} segments")
    info = get_video_info(video_path)
    fps = info["fps"]
    total_frames = info["total_frames"]

    cap = cv2.VideoCapture(video_path)
    boundary_frames = []
    start_time = time.time()
    for i in range(len(cut_points) - 1):
        start_frame = cut_points[i]
        end_frame = cut_points[i+1]
        segment_duration = (end_frame - start_frame) / fps
        print(f"   Segment {i+1}: {timedelta(seconds=int(start_frame/fps))} - {timedelta(seconds=int(end_frame/fps))} ({segment_duration:.1f}s)")
        step = max(1, (end_frame - start_frame) // (samples_per_cut + 1))
        for j in range(samples_per_cut):
            sample_frame = start_frame + (j + 1) * step
            cap.set(cv2.CAP_PROP_POS_FRAMES, sample_frame)
            ret, frame = cap.read()
            if ret:
                boundary_frames.append({
                    "segment": i,
                    "frame": sample_frame,
                    "image": frame,
                    "type": f"boundary_{j+1}"
                })
                print(f"     Sample {j+1}: {timedelta(seconds=int(sample_frame/fps))} ({sample_frame})")
    cap.release()
    elapsed = time.time() - start_time
    print(f"✅ Extracted {len(boundary_frames)} boundary frames in {elapsed:.1f}s")
    return boundary_frames

def verbose_match_frames(original_path, boundary_frames, match_radius=30, verbose=True):
    print(f"\n🔍 Searching for boundary frames in {os.path.basename(original_path)}...")
    info = get_video_info(original_path)
    fps = info["fps"]
    total_frames = info["total_frames"]
    duration = info["duration"]
    print(f"📊 Original video: {duration:.1f}s ({total_frames:,} frames at {fps:.2f} fps)")
    matches = []
    cap = cv2.VideoCapture(original_path)
    for i, bf in enumerate(boundary_frames):
        segment = bf["segment"]
        target_img = cv2.cvtColor(bf["image"], cv2.COLOR_BGR2GRAY)
        best_match = None
        best_score = float("-inf")
        search_start = max(0, bf["frame"] - match_radius)
        search_end = min(total_frames, bf["frame"] + match_radius)
        print(f"   Searching boundary frame {i+1}/{len(boundary_frames)} for segment {segment+1} ({bf['type']})...")
        for frame_idx in range(search_start, search_end):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sim = np.mean(gray == target_img)
            if sim > best_score:
                best_score = sim
                best_match = frame_idx
        if best_match is not None:
            print(f"     Match found: frame {best_match} ({timedelta(seconds=int(best_match/fps))}) (similarity: {best_score:.3f})")
            matches.append({
                "segment": segment,
                "boundary_frame": bf["frame"],
                "matched_frame": best_match,
                "similarity": best_score
            })
        else:
            print(f"     No good match found for segment {segment+1}")
    cap.release()
    print(f"✅ Frame matching complete. {len(matches)} matches found.")
    return matches

def verbose_reconstruct_segments(matches, cut_points, total_frames):
    print(f"\n🔧 Reconstructing segments from {len(matches)} matches...")
    segments = []
    if not matches:
        print("❌ No matches to reconstruct from")
        return []
    matches.sort(key=lambda x: x["segment"])
    for i, match in enumerate(matches):
        if i == 0:
            start_frame = 0
        else:
            start_frame = matches[i-1]["matched_frame"]
        end_frame = match["matched_frame"]
        if end_frame > start_frame:
            segments.append((start_frame, end_frame))
            print(f"✅ Segment {i+1}: {start_frame} - {end_frame} ({end_frame-start_frame} frames)")
        else:
            print(f"⚠️  Skipping invalid segment: {start_frame}-{end_frame}")
    if segments and segments[-1][1] != total_frames:
        segments.append((segments[-1][1], total_frames))
        print(f"✅ Segment {len(segments)}: {segments[-1][0]} - {total_frames}")
    print(f"\n📊 Final segments: {len(segments)} segments")
    return segments

def verbose_create_cut_video(input_path, segments, output_path, reencode=False):
    print(f"\nCreating cut video: {os.path.basename(output_path)}")
    if not segments:
        print("❌ No segments to process")
        return False
    # Use ffmpeg to extract segments
    try:
        for i, (start, end) in enumerate(segments):
            ss = start / 30.0
            to = (end - start) / 30.0
            cmd = [
                "ffmpeg",
                "-hide_banner", "-loglevel", "error",
                "-y", "-ss", str(ss), "-t", str(to),
                "-i", input_path,
                "-c", "copy" if not reencode else "libx264",
                f"segment_{i}.mp4"
            ]
            print(f"  Running ffmpeg for segment {i+1} ({timedelta(seconds=int(ss))} - {timedelta(seconds=int(ss+to))})")
            subprocess.run(cmd, check=True)
        # Concatenate segments
        with open("segments.txt", "w") as f:
            for i in range(len(segments)):
                f.write(f"file 'segment_{i}.mp4'\n")
        concat_cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "error",
            "-y", "-f", "concat", "-safe", "0",
            "-i", "segments.txt",
            "-c", "copy",
            output_path
        ]
        print("  Concatenating segments...")
        subprocess.run(concat_cmd, check=True)
        print(f"✅ Successfully created: {output_path}")
        # Cleanup
        for i in range(len(segments)):
            os.remove(f"segment_{i}.mp4")
        os.remove("segments.txt")
        return True
    except Exception as e:
        print(f"❌ Error creating cut video: {e}")
        return False

def main():
    if len(sys.argv) < 4:
        print("Usage: verbose_video_cut.py <edited_video> <original_video> <output_path>")
        sys.exit(1)
    edited_path = sys.argv[1]
    original_path = sys.argv[2]
    output_path = sys.argv[3]
    print("🚀 OPTIMIZED Multi-Method Cut Point Matcher")
    print("=" * 50)

    # Step 1: Adaptive cut detection
    cut_points = verbose_adaptive_cut_detection(edited_path)
    if len(cut_points) < 3:
        print("❌ Adaptive method didn’t find enough cuts, trying Method 2...")
        cut_points = verbose_similarity_drops_detection(edited_path)
    if len(cut_points) < 3:
        print("❌ Method 2 didn’t find enough cuts, trying Method 3...")
        cut_points = verbose_statistical_cut_detection(edited_path)
    if len(cut_points) < 3:
        print("❌ All methods failed to detect cuts.")
        sys.exit(1)

    # Step 2: Extract boundary frames
    boundary_frames = verbose_extract_boundary_frames(edited_path, cut_points)

    # Step 3: Match boundary frames in original
    matches = verbose_match_frames(original_path, boundary_frames)

    # Step 4: Reconstruct segments
    info = get_video_info(original_path)
    segments = verbose_reconstruct_segments(matches, cut_points, info["total_frames"])

    # Step 5: Output video
    success = verbose_create_cut_video(original_path, segments, output_path)
    if success:
        print(f"\n🎉 SUCCESS! Optimized cut video saved as: {output_path}")
    else:
        print("\n❌ Failed to create cut video")

if __name__ == "__main__":
    main()