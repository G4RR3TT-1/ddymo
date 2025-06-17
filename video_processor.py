#!/usr/bin/env python3
"""
Optimized Video Processing for macOS
Core module containing the main VideoProcessor class
"""

import cv2
import numpy as np
import os
import sys
import platform
from datetime import timedelta
import subprocess
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class VideoMetadata:
    """Data class for storing video metadata"""
    fps: float
    total_frames: int
    width: int
    height: int
    duration: float
    codec: str
    container: str

class VideoProcessor:
    """Main class for video processing operations"""
    
    def __init__(self, use_hardware_acceleration: bool = True):
        self.hw_accel = use_hardware_acceleration and self._check_hardware_acceleration()
        self._frame_cache: Dict[int, dict] = {}
        self.metadata: Optional[VideoMetadata] = None
        
    def _check_hardware_acceleration(self) -> bool:
        """Check for available hardware acceleration on macOS"""
        if platform.system() == "Darwin":
            try:
                # Check for VideoToolbox support
                result = subprocess.run(
                    ['ffmpeg', '-hwaccels'],
                    capture_output=True,
                    text=True
                )
                return 'videotoolbox' in result.stdout.lower()
            except FileNotFoundError:
                logger.warning("FFmpeg not found, hardware acceleration disabled")
                return False
        return False

    def _get_video_metadata(self, video_path: str) -> VideoMetadata:
        """Get video metadata using FFprobe"""
        try:
            probe = subprocess.run([
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ], capture_output=True, text=True)
            
            if probe.returncode != 0:
                raise RuntimeError(f"FFprobe failed: {probe.stderr}")
            
            import json
            data = json.loads(probe.stdout)
            
            # Extract video stream info
            video_stream = next(
                s for s in data['streams'] 
                if s['codec_type'] == 'video'
            )
            
            return VideoMetadata(
                fps=eval(video_stream.get('r_frame_rate', '0/0')),
                total_frames=int(video_stream.get('nb_frames', 0)),
                width=int(video_stream['width']),
                height=int(video_stream['height']),
                duration=float(data['format']['duration']),
                codec=video_stream['codec_name'],
                container=data['format']['format_name']
            )
        except Exception as e:
            logger.error(f"Failed to get video metadata: {e}")
            raise

    def process_video(self, input_path: str, output_path: str) -> bool:
        """Main video processing pipeline"""
        try:
            # Validate input
            input_path = Path(input_path)
            if not input_path.exists():
                raise FileNotFoundError(f"Input video not found: {input_path}")
                
            # Get video metadata
            self.metadata = self._get_video_metadata(str(input_path))
            logger.info(f"Processing video: {input_path.name}")
            logger.info(f"Metadata: {self.metadata}")
            
            # Initialize processing with hardware acceleration if available
            if self.hw_accel:
                logger.info("Using VideoToolbox hardware acceleration")
                cv2.setNumThreads(1)  # Reduce CPU usage when using GPU
                
            return True
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            return False

    def cleanup(self):
        """Cleanup resources"""
        self._frame_cache.clear()
        cv2.destroyAllWindows()