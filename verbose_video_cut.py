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
    """Install required packages if not available"""
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

def enhanced_perceptual_hash(frame, cache=None):
    """Create enhanced perceptual hash with multiple features and optional caching"""
    if cache is not None:
        frame_key = hash(frame.tobytes())
        if frame_key in cache:
            return cache[frame_key]
    
    # Resize and convert to grayscale
    resized = cv2.resize(frame, (32, 32))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Feature 1: Basic perceptual hash
    mean = np.mean(gray)
    basic_hash = gray > mean
    
    # Feature 2: Edge detection hash
    edges = cv2.Canny(gray, 50, 150)
    edge_hash = edges > 0
    
    # Feature 3: Histogram features (reduced to key bins)
    hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
    hist_features = hist.flatten() / np.sum(hist)  # Normalize
    
    result = {
        'basic': basic_hash.flatten(),
        'edges': edge_hash.flatten(),
        'hist': hist_features
    }
    
    if cache is not None:
        cache[frame_key] = result
    
    return result

def compare_enhanced_frames(hash1, hash2):
    """Compare two enhanced frame hashes with weighted similarity"""
    # Basic hash similarity
    basic_sim = np.sum(hash1['basic'] == hash2['basic']) / len(hash1['basic'])
    
    # Edge hash similarity
    edge_sim = np.sum(hash1['edges'] == hash2['edges']) / len(hash1['edges'])
    
    # Histogram similarity (using correlation)
    hist_sim = np.corrcoef(hash1['hist'], hash2['hist'])[0, 1]
    if np.isnan(hist_sim):
        hist_sim = 0
    hist_sim = (hist_sim + 1) / 2  # Normalize to 0-1
    
    # Weighted combination
    combined_similarity = (basic_sim * 0.4 + edge_sim * 0.3 + hist_sim * 0.3)
    return combined_similarity, basic_sim, edge_sim, hist_sim

def detect_cut_points_adaptive(video_path, initial_threshold=0.03, max_segments=50):
    """Adaptive cut detection that adjusts threshold based on results"""
    print(f"🔍 Starting adaptive cut detection...")
    print(f"   Video: {os.path.basename(video_path)}")
    print(f"   Initial threshold: {initial_threshold}")
    print(f"   Max segments: {max_segments}")
    
    start_time = time.time()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"📊 Video info:")
    print(f"   FPS: {fps:.2f}")
    print(f"   Total frames: {total_frames:,}")
    print(f"   Duration: {timedelta(seconds=int(duration))}")
    
    # First pass: collect similarity data with frame skipping for speed
    print("\n🔍 Starting quick analysis pass...")
    similarities = []
    frame_positions = []
    prev_hash = None
    frame_count = 0
    skip_frames = max(1, total_frames // 5000)  # Sample ~5000 frames max
    
    print(f"   Skip frames: {skip_frames} (analyzing every {skip_frames} frames)")
    
    hash_cache = {}  # Cache for frame hashes
    last_progress_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames for faster analysis
        if frame_count % skip_frames != 0:
            frame_count += 1
            continue
        
        current_hash = enhanced_perceptual_hash(frame, hash_cache)
        
        if prev_hash is not None:
            similarity, _, _, _ = compare_enhanced_frames(prev_hash, current_hash)
            similarities.append(similarity)
            frame_positions.append(frame_count)
        
        prev_hash = current_hash
        frame_count += 1
        
        # More frequent progress updates
        current_time = time.time()
        if current_time - last_progress_time > 2.0 or frame_count % 1000 == 0:
            progress = (frame_count / total_frames) * 100
            elapsed = current_time - start_time
            fps_analysis = frame_count / elapsed if elapsed > 0 else 0
            print(f"   Quick analysis: {progress:.1f}% ({frame_count:,}/{total_frames:,}) | {fps_analysis:.1f} fps | {len(similarities)} comparisons")
            last_progress_time = current_time
    
    cap.release()
    
    analysis_time = time.time() - start_time
    print(f"✅ Quick analysis complete in {analysis_time:.1f}s")
    
    if not similarities:
        print("❌ No similarities calculated")
        return [0.0, total_frames / fps], fps
    
    # Calculate adaptive threshold
    similarities = np.array(similarities)
    percentiles = [1, 3, 5, 10, 15, 20]
    
    print(f"\n📊 Similarity analysis (from {len(similarities)} comparisons):")
    print(f"   Min: {np.min(similarities):.4f}")
    print(f"   Max: {np.max(similarities):.4f}")
    print(f"   Mean: {np.mean(similarities):.4f}")
    print(f"   Std: {np.std(similarities):.4f}")
    
    for p in percentiles:
        thresh = np.percentile(similarities, p)
        potential_cuts = np.sum(similarities < thresh)
        print(f"   {p:2d}th percentile: {thresh:.4f} ({potential_cuts:3d} potential cuts)")
    
    # Choose threshold that gives reasonable number of cuts
    chosen_threshold = None
    for p in percentiles:
        thresh = np.percentile(similarities, p)
        potential_cuts = np.sum(similarities < thresh)
        if 5 <= potential_cuts <= max_segments:
            chosen_threshold = thresh
            print(f"✅ Using {p}th percentile threshold: {chosen_threshold:.4f}")
            break
    
    if chosen_threshold is None:
        # If no good threshold found, use a more aggressive one
        chosen_threshold = np.percentile(similarities, 10)
        print(f"⚠️  Using 10th percentile as fallback: {chosen_threshold:.4f}")
    
    # Second pass: detect cuts with chosen threshold
    print(f"\n🔍 Starting detailed cut detection with threshold {chosen_threshold:.4f}...")
    return detect_cut_points_method1(video_path, chosen_threshold)

def detect_cut_points_method2(video_path, drop_threshold=0.15, base_threshold=0.05):
    """Method 2: Detect cuts using similarity drops and rolling average"""
    print(f"\n🔍 METHOD 2: Similarity drops detection")
    print(f"   Drop threshold: {drop_threshold}")
    print(f"   Base threshold: {base_threshold}")
    
    start_time = time.time()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Processing {total_frames:,} frames at {fps:.2f} fps...")
    
    cut_points = [0.0]  # Always include start
    prev_hash = None
    prev_similarity = None
    frame_count = 0
    similarity_window = []  # Rolling window for adaptive threshold
    window_size = 50
    
    hash_cache = {}
    last_progress_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"   📹 Reached end of video at frame {frame_count}")
            break
        
        current_hash = enhanced_perceptual_hash(frame, hash_cache)
        
        if prev_hash is not None:
            similarity, basic_sim, edge_sim, hist_sim = compare_enhanced_frames(prev_hash, current_hash)
            similarity_window.append(similarity)
            
            # Keep window size manageable
            if len(similarity_window) > window_size:
                similarity_window.pop(0)
            
            # Calculate adaptive threshold
            if len(similarity_window) >= 10:
                window_avg = np.mean(similarity_window)
                adaptive_threshold = max(base_threshold, window_avg - 0.3)
            else:
                adaptive_threshold = base_threshold
            
            # Method 2 detection: Look for significant drops OR very low similarity
            is_cut = False
            cut_reason = ""
            
            if similarity < adaptive_threshold:
                is_cut = True
                cut_reason = f"low_sim({similarity:.3f} < {adaptive_threshold:.3f})"
            
            if prev_similarity is not None and (prev_similarity - similarity) > drop_threshold:
                is_cut = True
                cut_reason += f" drop({prev_similarity:.3f} -> {similarity:.3f})"
            
            if is_cut:
                timestamp = frame_count / fps
                # Avoid duplicate cuts too close together
                if not cut_points or (timestamp - cut_points[-1]) > 2.0:
                    cut_points.append(timestamp)
                    print(f"🎬 CUT DETECTED at {timedelta(seconds=int(timestamp))} - {cut_reason}")
                    print(f"   Frame {frame_count:,} | Basic: {basic_sim:.3f} | Edge: {edge_sim:.3f} | Hist: {hist_sim:.3f}")
            
            # Verbose output every 1000 frames
            current_time = time.time()
            if frame_count % 1000 == 0 or current_time - last_progress_time > 3.0:
                window_avg = np.mean(similarity_window) if similarity_window else 0
                progress = (frame_count / total_frames) * 100
                elapsed = current_time - start_time
                fps_analysis = frame_count / elapsed if elapsed > 0 else 0
                
                print(f"   Frame {frame_count:,}: {progress:.1f}% | sim={similarity:.3f} | window_avg={window_avg:.3f} | thresh={adaptive_threshold:.3f} | {fps_analysis:.1f} fps")
                last_progress_time = current_time
            
            prev_similarity = similarity
        
        prev_hash = current_hash
        frame_count += 1
    
    cap.release()
    
    # Add end time
    duration = total_frames / fps
    cut_points.append(duration)
    
    elapsed = time.time() - start_time
    print(f"✅ Method 2 complete in {elapsed:.1f}s")
    print(f"   Result: {len(cut_points) - 1} segments with {len(cut_points) - 2} cut points")
    return cut_points, fps

def detect_cut_points_method3(video_path, percentile_threshold=5):
    """Method 3: Statistical approach - detect cuts based on percentile of all similarities"""
    print(f"\n🔍 METHOD 3: Statistical detection")
    print(f"   Using {percentile_threshold}th percentile threshold")
    
    start_time = time.time()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Processing {total_frames:,} frames...")
    
    # First pass: collect all similarities
    print("🔍 First pass: collecting similarity data...")
    similarities = []
    timestamps = []
    prev_hash = None
    frame_count = 0
    hash_cache = {}
    last_progress_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"   📹 Reached end of video at frame {frame_count}")
            break
        
        current_hash = enhanced_perceptual_hash(frame, hash_cache)
        
        if prev_hash is not None:
            similarity, _, _, _ = compare_enhanced_frames(prev_hash, current_hash)
            similarities.append(similarity)
            timestamps.append(frame_count / fps)
        
        prev_hash = current_hash
        frame_count += 1
        
        current_time = time.time()
        if frame_count % 1000 == 0 or current_time - last_progress_time > 3.0:
            progress = (frame_count / total_frames) * 100
            elapsed = current_time - start_time
            fps_analysis = frame_count / elapsed if elapsed > 0 else 0
            print(f"   First pass: {progress:.1f}% ({frame_count:,}/{total_frames:,}) | {fps_analysis:.1f} fps | {len(similarities)} comparisons")
            last_progress_time = current_time
    
    cap.release()
    
    # Calculate statistical threshold
    threshold = np.percentile(similarities, percentile_threshold)
    print(f"\n📊 Statistical analysis:")
    print(f"   Calculated threshold: {threshold:.3f} ({percentile_threshold}th percentile)")
    print(f"   Similarity range: {np.min(similarities):.3f} to {np.max(similarities):.3f}")
    print(f"   Mean similarity: {np.mean(similarities):.3f}")
    
    # Second pass: find cuts
    print(f"\n🔍 Finding cuts below threshold {threshold:.3f}...")
    cut_points = [0.0]
    cuts_found = 0
    
    for i, (sim, timestamp) in enumerate(zip(similarities, timestamps)):
        if sim < threshold:
            # Avoid cuts too close together
            if not cut_points or (timestamp - cut_points[-1]) > 2.0:
                cut_points.append(timestamp)
                cuts_found += 1
                print(f"🎬 Cut {cuts_found} at {timedelta(seconds=int(timestamp))} (similarity: {sim:.3f})")
    
    # Add end time
    duration = total_frames / fps
    cut_points.append(duration)
    
    elapsed = time.time() - start_time
    print(f"✅ Method 3 complete in {elapsed:.1f}s")
    print(f"   Result: {len(cut_points) - 1} segments with {len(cut_points) - 2} cut points")
    return cut_points, fps

def detect_cut_points_method1(video_path, threshold=0.03):
    """Method 1: Detect cut points using absolute similarity threshold (optimized)"""
    print(f"\n🔍 METHOD 1: Absolute threshold detection")
    print(f"   Threshold: {threshold}")
    
    start_time = time.time()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Processing {total_frames:,} frames at {fps:.2f} fps...")
    
    cut_points = [0.0]  # Always include start
    prev_hash = None
    frame_count = 0
    similarity_values = []
    
    # Process frames individually for better progress tracking
    last_progress_time = time.time()
    cuts_found = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"   📹 Reached end of video at frame {frame_count}")
            break
        
        current_hash = enhanced_perceptual_hash(frame)
        
        if prev_hash is not None:
            similarity, basic_sim, edge_sim, hist_sim = compare_enhanced_frames(prev_hash, current_hash)
            similarity_values.append(similarity)
            
            if similarity < threshold:
                timestamp = frame_count / fps
                # Avoid cuts too close together
                if not cut_points or (timestamp - cut_points[-1]) > 1.0:
                    cut_points.append(timestamp)
                    cuts_found += 1
                    print(f"🎬 CUT {cuts_found} at {timedelta(seconds=int(timestamp))} (similarity: {similarity:.3f})")
                    print(f"   Frame {frame_count:,} | Basic: {basic_sim:.3f} | Edge: {edge_sim:.3f} | Hist: {hist_sim:.3f}")
        
        prev_hash = current_hash
        frame_count += 1
        
        # Progress indicator with more details
        current_time = time.time()
        if frame_count % 500 == 0 or current_time - last_progress_time > 3.0:
            progress = (frame_count / total_frames) * 100
            elapsed = current_time - start_time
            fps_analysis = frame_count / elapsed if elapsed > 0 else 0
            eta_seconds = (total_frames - frame_count) / fps_analysis if fps_analysis > 0 else 0
            
            print(f"   Progress: {progress:.1f}% ({frame_count:,}/{total_frames:,}) | {fps_analysis:.1f} fps | ETA: {int(eta_seconds)}s | Cuts: {cuts_found}")
            last_progress_time = current_time
    
    cap.release()
    
    # Add end time
    duration = total_frames / fps
    cut_points.append(duration)
    
    elapsed = time.time() - start_time
    print(f"✅ Method 1 complete in {elapsed:.1f}s")
    print(f"   Result: {len(cut_points) - 1} segments with {len(cut_points) - 2} cut points")
    
    # Show similarity statistics
    if similarity_values:
        print(f"📊 Similarity stats: min={np.min(similarity_values):.4f}, max={np.max(similarity_values):.4f}, mean={np.mean(similarity_values):.4f}")
    
    return cut_points, fps, similarity_values

def extract_boundary_frames_optimized(video_path, cut_points, fps):
    """Extract frames at segment boundaries with better sampling"""
    print(f"\n🎞️  Extracting boundary frames from {os.path.basename(video_path)}...")
    print(f"   Processing {len(cut_points) - 1} segments")
    
    start_time = time.time()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video: {video_path}")
    
    boundary_frames = []
    
    # Extract multiple frames per boundary for better matching
    for i in range(len(cut_points) - 1):
        start_time_seg = cut_points[i]
        end_time_seg = cut_points[i + 1]
        segment_duration = end_time_seg - start_time_seg
        
        print(f"   Segment {i+1}: {timedelta(seconds=int(start_time_seg))} - {timedelta(seconds=int(end_time_seg))} ({segment_duration:.1f}s)")
        
        # Sample frames strategically
        sample_times = []
        
        # Always sample start
        sample_times.append(start_time_seg)
        
        # Sample a few frames into the segment
        if segment_duration > 5:
            sample_times.extend([
                start_time_seg + 1.0,
                start_time_seg + 2.0,
                max(start_time_seg + 3.0, end_time_seg - 2.0)
            ])
        elif segment_duration > 2:
            sample_times.append(start_time_seg + 1.0)
        
        # Sample end (but not too close to next segment)
        end_sample_time = max(start_time_seg + 0.5, end_time_seg - 0.5)
        sample_times.append(end_sample_time)
        
        # Extract frames
        for j, sample_time in enumerate(sample_times):
            cap.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000)
            ret, frame = cap.read()
            if ret:
                hash_val = enhanced_perceptual_hash(frame)
                frame_type = 'start' if j == 0 else ('end' if j == len(sample_times) - 1 else 'middle')
                boundary_frames.append({
                    'time': sample_time,
                    'hash': hash_val,
                    'type': frame_type,
                    'segment': i,
                    'sample_index': j
                })
                print(f"     Sample {j+1}: {timedelta(seconds=int(sample_time))} ({frame_type})")
    
    cap.release()
    
    elapsed = time.time() - start_time
    print(f"✅ Extracted {len(boundary_frames)} boundary frames in {elapsed:.1f}s")
    return boundary_frames

def find_frames_in_original_optimized(original_path, boundary_frames, search_window=30):
    """Optimized frame search with progressive starting points and parallel processing"""
    print(f"\n🔍 Searching for boundary frames in {os.path.basename(original_path)}...")
    
    start_time = time.time()
    
    cap = cv2.VideoCapture(original_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video: {original_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()
    
    print(f"📊 Original video: {duration:.1f}s ({total_frames:,} frames at {fps:.2f} fps)")
    
    # Group boundary frames by segment for better processing
    segments = {}
    for bf in boundary_frames:
        seg_id = bf['segment']
        if seg_id not in segments:
            segments[seg_id] = []
        segments[seg_id].append(bf)
    
    print(f"   Processing {len(segments)} segments with {len(boundary_frames)} boundary frames")
    
    matches = []
    last_found_time = 0.0  # Progressive search optimization
    
    # Process segments in order
    for seg_id in sorted(segments.keys()):
        segment_boundaries = segments[seg_id]
        print(f"\n🔍 Processing segment {seg_id+1} with {len(segment_boundaries)} boundary frames...")
        print(f"   Starting search from {timedelta(seconds=int(last_found_time))}")
        
        # Find the best representative frame from this segment
        best_boundary = None
        best_match_score = 0
        best_match_time = last_found_time
        
        # Search for each boundary frame in this segment
        for i, boundary in enumerate(segment_boundaries):
            print(f"   Searching boundary frame {i+1}/{len(segment_boundaries)} ({boundary['type']})...")
            
            match_result = search_single_frame(
                original_path, boundary, last_found_time, duration, fps
            )
            
            if match_result and match_result['similarity'] > best_match_score:
                best_match_score = match_result['similarity']
                best_match_time = match_result['original_time']
                best_boundary = boundary
                print(f"     New best match: {timedelta(seconds=int(best_match_time))} (similarity: {best_match_score:.3f})")
        
        if best_boundary and best_match_score > 0.6:
            matches.append({
                'boundary': best_boundary,
                'original_time': best_match_time,
                'similarity': best_match_score,
                'segment': seg_id
            })
            
            # Update progressive search starting point
            last_found_time = best_match_time
            print(f"✅ Segment {seg_id+1} matched at {timedelta(seconds=int(best_match_time))} (similarity: {best_match_score:.3f})")
        else:
            print(f"⚠️  No good match for segment {seg_id+1} (best score: {best_match_score:.3f})")
            # Don't update last_found_time if no match
    
    elapsed = time.time() - start_time
    print(f"\n✅ Frame matching complete in {elapsed:.1f}s")
    print(f"   Found {len(matches)} segment matches out of {len(segments)} segments")
    return matches

def search_single_frame(original_path, boundary, start_time, duration, fps, search_step=2.0):
    """Search for a single frame starting from a given time"""
    cap = cv2.VideoCapture(original_path)
    if not cap.isOpened():
        return None
    
    best_match = {
        'time': -1,
        'similarity': 0
    }
    
    searches_done = 0
    
    # Search from start_time forward, with fallback to full search if needed
    search_ranges = [
        (start_time, duration),  # Forward from last found
        (0, start_time) if start_time > 0 else None  # Backward fallback
    ]
    
    for search_start, search_end in filter(None, search_ranges):
        if search_start >= search_end:
            continue
            
        for search_time in np.arange(search_start, search_end, search_step):
            cap.set(cv2.CAP_PROP_POS_MSEC, search_time * 1000)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            original_hash = enhanced_perceptual_hash(frame)
            similarity, _, _, _ = compare_enhanced_frames(
                boundary['hash'], original_hash
            )
            
            if similarity > best_match['similarity']:
                best_match = {
                    'time': search_time,
                    'similarity': similarity,
                    'original_time': search_time
                }
            
            searches_done += 1
            
            # Progress update for long searches
            if searches_done % 100 == 0:
                print(f"     Searched {searches_done} positions, best: {best_match['similarity']:.3f}")
            
            # Early termination if we find a very good match
            if similarity > 0.9:
                print(f"     Found excellent match early (similarity: {similarity:.3f})")
                break
        
        # If we found a decent match in forward search, don't do backward
        if best_match['similarity'] > 0.7:
            break
    
    cap.release()
    
    if best_match['similarity'] > 0.3:
        print(f"     Match found: {timedelta(seconds=int(best_match['time']))} (similarity: {best_match['similarity']:.3f})")
    else:
        print(f"     No good match found (best: {best_match['similarity']:.3f})")
    
    return best_match if best_match['similarity'] > 0.3 else None

def reconstruct_segments_smart(matches):
    """Smart segment reconstruction with gap filling"""
    print(f"\n🔧 Reconstructing segments from {len(matches)} matches...")
    
    if not matches:
        print("❌ No matches to reconstruct from")
        return []
    
    # Sort matches by original video time
    matches.sort(key=lambda x: x['original_time'])
    
    print("📊 Match timeline:")
    for i, match in enumerate(matches):
        print(f"   {i+1}. Segment {match['segment']+1} -> {timedelta(seconds=int(match['original_time']))} (sim: {match['similarity']:.3f})")
    
    segments = []
    
    # Simple approach: create segments from consecutive matches
    for i in range(len(matches) - 1):
        start_time = matches[i]['original_time']
        end_time = matches[i + 1]['original_time']
        
        # Only include if segment is reasonable length
        if end_time > start_time and (end_time - start_time) >= 2.0:
            segments.append((start_time, end_time))
            duration = end_time - start_time
            print(f"✅ Segment {len(segments)}: {timedelta(seconds=int(start_time))} - {timedelta(seconds=int(end_time))} ({duration:.1f}s)")
        else:
            print(f"⚠️  Skipping invalid segment: {start_time:.1f}s - {end_time:.1f}s")
    
    total_duration = sum(end - start for start, end in segments)
    print(f"\n📊 Final segments: {len(segments)} segments, {timedelta(seconds=int(total_duration))} total")
    
    return segments

def create_cut_video_optimized(input_path, segments, output_