"""
Real-time sliding-window inference engine for EEG BCI.

Continuously processes EEG streams using a sliding window approach,
extracting windows at regular intervals and running them through
preprocessing, feature extraction, and classification pipelines.
"""

import logging
import time
from collections import deque
from typing import Callable, Optional, Tuple

import numpy as np

from .buffer import CircularRingBuffer

logger = logging.getLogger(__name__)


class RealtimeInferenceEngine:
    """
    Real-time sliding-window inference engine.
    
    Maintains a ring buffer of continuous EEG data and triggers
    inference at regular intervals using a sliding window.
    
    Parameters
    ----------
    pipeline : object
        Pipeline object with predict_stream(window_data) method
    fs : float
        Sampling rate in Hz
    window_size_sec : float
        Size of sliding window in seconds (e.g., 1.0-4.0)
    update_interval_sec : float
        How often to trigger inference (e.g., 0.1 for 100ms)
    buffer_length_sec : float, optional
        Length of ring buffer in seconds (default: 10.0)
    n_channels : int, optional
        Number of channels (inferred from first data if not provided)
    """
    
    def __init__(
        self,
        pipeline,
        fs: float,
        window_size_sec: float,
        update_interval_sec: float,
        buffer_length_sec: float = 10.0,
        n_channels: Optional[int] = None,
    ) -> None:
        self.pipeline = pipeline
        self.fs = fs
        self.window_size_sec = window_size_sec
        self.update_interval_sec = update_interval_sec
        self.buffer_length_sec = buffer_length_sec
        
        self.n_channels = n_channels
        self._buffer: Optional[CircularRingBuffer] = None
        
        # Timing tracking
        self._last_update_time = None
        self._last_update_sample = 0
        
        # Latency tracking
        self._latencies_ms = deque(maxlen=1000)  # Keep last 1000 measurements
        
        # Statistics
        self._n_predictions = 0
        self._total_latency_ms = 0.0
    
    def initialize(self, n_channels: int) -> None:
        """Initialize the buffer with known channel count."""
        if self._buffer is None:
            self.n_channels = n_channels
            self._buffer = CircularRingBuffer(
                n_channels=n_channels,
                buffer_length_sec=self.buffer_length_sec,
                fs=self.fs,
            )
            logger.info(
                "RealtimeInferenceEngine initialized: "
                f"window={self.window_size_sec}s, "
                f"update_interval={self.update_interval_sec}s, "
                f"buffer={self.buffer_length_sec}s"
            )
    
    def push_samples(self, samples: np.ndarray) -> None:
        """
        Push new samples into the buffer.
        
        Parameters
        ----------
        samples : np.ndarray
            Shape (n_channels, n_samples) or (n_channels,) for single sample
        """
        if self._buffer is None:
            samples_arr = np.asarray(samples)
            if samples_arr.ndim == 1:
                n_ch = samples_arr.shape[0]
            else:
                n_ch = samples_arr.shape[0]
            self.initialize(n_ch)
        
        self._buffer.append(samples)
    
    def update(self, current_time: Optional[float] = None) -> Optional[Tuple[np.ndarray, float]]:
        """
        Check if it's time to run inference and run if needed.
        
        Parameters
        ----------
        current_time : float, optional
            Current time in seconds (uses time.perf_counter() if None)
        
        Returns
        -------
        result : tuple or None
            (prediction, latency_ms) if inference was run, else None
        """
        if self._buffer is None:
            return None
        
        if current_time is None:
            current_time = time.perf_counter()
        
        # Check if enough time has passed
        if self._last_update_time is None:
            self._last_update_time = current_time
            return None
        
        elapsed = current_time - self._last_update_time
        if elapsed < self.update_interval_sec:
            return None
        
        # Extract window
        window = self._buffer.get_window(self.window_size_sec)
        if window is None:
            logger.debug("Not enough data in buffer for window extraction")
            return None
        
        # Run inference
        t0 = time.perf_counter()
        try:
            # Convert window to trial format: (1, n_channels, n_samples)
            window_trial = window[np.newaxis, :, :]
            
            # Use predict_stream if available, otherwise fall back to predict
            if hasattr(self.pipeline, "predict_stream"):
                prediction = self.pipeline.predict_stream(window_trial)
            else:
                prediction = self.pipeline.predict(window_trial)
            
            latency_ms = (time.perf_counter() - t0) * 1000
            
            # Track latency
            self._latencies_ms.append(latency_ms)
            self._n_predictions += 1
            self._total_latency_ms += latency_ms
            
            self._last_update_time = current_time
            
            logger.debug(
                f"Inference completed: prediction={prediction[0]}, "
                f"latency={latency_ms:.2f}ms"
            )
            
            return prediction, latency_ms
            
        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            return None
    
    def get_latency_stats(self) -> dict:
        """
        Get latency statistics.
        
        Returns
        -------
        stats : dict
            Dictionary with mean, median, max, min latency in ms
        """
        if len(self._latencies_ms) == 0:
            return {
                "mean_ms": 0.0,
                "median_ms": 0.0,
                "max_ms": 0.0,
                "min_ms": 0.0,
                "n_predictions": 0,
            }
        
        latencies = np.array(self._latencies_ms)
        return {
            "mean_ms": float(np.mean(latencies)),
            "median_ms": float(np.median(latencies)),
            "max_ms": float(np.max(latencies)),
            "min_ms": float(np.min(latencies)),
            "n_predictions": self._n_predictions,
        }
    
    def reset(self) -> None:
        """Reset the engine state."""
        if self._buffer is not None:
            self._buffer.clear()
        self._last_update_time = None
        self._last_update_sample = 0
        self._latencies_ms.clear()
        self._n_predictions = 0
        self._total_latency_ms = 0.0
