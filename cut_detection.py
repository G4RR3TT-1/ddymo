"""
Video cut point detection module
Implements multiple detection methods with optimizations for macOS
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging
from dataclasses import dataclass
from tqdm import tqdm
import time
from datetime import timedelta

logger = logging.getLogger(__name__)

@dataclass
class CutPoint:
    """Data class for storing cut point information"""
    timestamp: float
    confidence: float
    method: str
    similarity: float

class CutDetector:
    """Handles video cut point detection using multiple methods"""
    
    def __init__(self, frame_analyzer):
        self.frame_analyzer = frame_analyzer
        self.min_segment_duration = 2.0  # Minimum segment duration in seconds
        
    def detect_cuts_adaptive(self, video_path: str, 
                           initial_threshold: float = 0.03,
                           max_segments: int = 50) -> List[CutPoint]:
        """
        Adaptive cut detection that adjusts threshold based on results
        """
        logger.info(f"Starting adaptive cut detection...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # First pass: collect similarity data
        similarities = []
        frame_positions = []
        prev_hash = None
        
        with tqdm(total=total_frames, desc="Analyzing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                current_hash = self.frame_analyzer.compute_frame_hash(frame)
                
                if prev_hash is not None:
                    similarity, _, _, _ = self.frame_analyzer.compare_hashes(
                        prev_hash, current_hash
                    )
                    similarities.append(similarity)
                    frame_positions.append(cap.get(cv2.CAP_PROP_POS_FRAMES))
                
                prev_hash = current_hash
                pbar.update(1)
                
        cap.release()
        
        # Calculate adaptive threshold
        similarities = np.array(similarities)
        percentiles = [1, 3, 5, 10, 15, 20]
        
        cut_points = []
        best_threshold = None
        best_num_cuts = 0
        
        # Find optimal threshold
        for p in percentiles:
            threshold = np.percentile(similarities, p)
            potential_cuts = np.sum(similarities < threshold)
            
            if 5 <= potential_cuts <= max_segments:
                if best_threshold is None or potential_cuts > best_num_cuts:
                    best_threshold = threshold
                    best_num_cuts = potential_cuts
                    
        if best_threshold is None:
            best_threshold = np.percentile(similarities, 10)
            
        # Detect cuts using best threshold
        for i, (sim, pos) in enumerate(zip(similarities, frame_positions)):
            if sim < best_threshold:
                timestamp = pos / fps
                if not cut_points or (timestamp - cut_points[-1].timestamp) >= self.min_segment_duration:
                    cut_points.append(CutPoint(
                        timestamp=timestamp,
                        confidence=1.0 - (sim / best_threshold),
                        method="adaptive",
                        similarity=sim
                    ))
                    
        return cut_points