"""
Frame analysis and comparison module
Optimized for macOS with hardware acceleration support
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

@dataclass
class FrameHash:
    """Data class for frame hash components"""
    basic: np.ndarray
    edges: np.ndarray
    hist: np.ndarray
    
class FrameAnalyzer:
    """Handles frame analysis and comparison operations"""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and self._check_gpu_support()
        self._cache: Dict[int, FrameHash] = {}
        self._cache_lock = threading.Lock()
        
    def _check_gpu_support(self) -> bool:
        """Check for GPU/hardware acceleration support"""
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                return True
        except:
            pass
        return False
        
    def compute_frame_hash(self, frame: np.ndarray) -> FrameHash:
        """Compute enhanced perceptual hash with GPU acceleration if available"""
        if self.use_gpu:
            # Upload to GPU
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            
            # GPU operations
            gpu_resized = cv2.cuda.resize(gpu_frame, (32, 32))
            gpu_gray = cv2.cuda.cvtColor(gpu_resized, cv2.COLOR_BGR2GRAY)
            
            # Download results
            gray = gpu_gray.download()
        else:
            # CPU operations
            resized = cv2.resize(frame, (32, 32))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Calculate hash components
        mean = np.mean(gray)
        basic_hash = gray > mean
        
        edges = cv2.Canny(gray, 50, 150)
        edge_hash = edges > 0
        
        hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
        hist_features = hist.flatten() / np.sum(hist)
        
        return FrameHash(
            basic=basic_hash.flatten(),
            edges=edge_hash.flatten(),
            hist=hist_features
        )
        
    def compare_hashes(self, hash1: FrameHash, hash2: FrameHash) -> Tuple[float, float, float, float]:
        """Compare two frame hashes with weighted similarity metrics"""
        # Basic hash similarity
        basic_sim = np.sum(hash1.basic == hash2.basic) / len(hash1.basic)
        
        # Edge hash similarity
        edge_sim = np.sum(hash1.edges == hash2.edges) / len(hash1.edges)
        
        # Histogram similarity
        hist_sim = np.corrcoef(hash1.hist, hash2.hist)[0, 1]
        if np.isnan(hist_sim):
            hist_sim = 0
        hist_sim = (hist_sim + 1) / 2
        
        # Combined similarity with weights
        combined = basic_sim * 0.4 + edge_sim * 0.3 + hist_sim * 0.3
        
        return combined, basic_sim, edge_sim, hist_sim

    def analyze_frame_batch(self, frames: list) -> list:
        """Analyze a batch of frames in parallel"""
        with ThreadPoolExecutor() as executor:
            return list(executor.map(self.compute_frame_hash, frames))