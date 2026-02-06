"""
Thread-safe circular ring buffer for continuous EEG streaming.

Maintains a fixed-size buffer of continuous EEG samples with thread-safe
append and window extraction operations.
"""

import threading
from typing import Optional

import numpy as np


class CircularRingBuffer:
    """
    Thread-safe circular ring buffer for continuous EEG samples.
    
    Stores samples in a circular buffer with shape (n_channels, buffer_length_samples).
    Supports appending new samples and extracting sliding windows.
    
    Parameters
    ----------
    n_channels : int
        Number of EEG channels
    buffer_length_sec : float
        Maximum buffer duration in seconds
    fs : float
        Sampling rate in Hz
    dtype : type, optional
        Data type (default: np.float64)
    """
    
    def __init__(
        self,
        n_channels: int,
        buffer_length_sec: float,
        fs: float,
        dtype: type = np.float64,
    ) -> None:
        self.n_channels = n_channels
        self.fs = fs
        self.buffer_length_sec = buffer_length_sec
        self.buffer_length_samples = int(buffer_length_sec * fs)
        self.dtype = dtype
        
        # Circular buffer: (n_channels, buffer_length_samples)
        self._buffer = np.zeros((n_channels, self.buffer_length_samples), dtype=dtype)
        
        # Current write position (circular index)
        self._write_pos = 0
        
        # Number of valid samples (increases until buffer is full, then stays at buffer_length_samples)
        self._n_valid = 0
        
        # Thread lock for thread-safe operations
        self._lock = threading.Lock()
    
    def append(self, samples: np.ndarray) -> None:
        """
        Append new samples to the buffer.
        
        Parameters
        ----------
        samples : np.ndarray
            Shape (n_channels, n_samples) or (n_channels,) for single sample
        """
        samples = np.asarray(samples, dtype=self.dtype)
        
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        
        if samples.shape[0] != self.n_channels:
            raise ValueError(
                f"Channel mismatch: expected {self.n_channels} channels, "
                f"got {samples.shape[0]}"
            )
        
        n_new = samples.shape[1]
        
        with self._lock:
            if n_new >= self.buffer_length_samples:
                # More samples than buffer size: keep only the most recent
                self._buffer[:] = samples[:, -self.buffer_length_samples:]
                self._write_pos = 0
                self._n_valid = self.buffer_length_samples
            else:
                # Append samples circularly
                for i in range(n_new):
                    self._buffer[:, self._write_pos] = samples[:, i]
                    self._write_pos = (self._write_pos + 1) % self.buffer_length_samples
                    self._n_valid = min(self._n_valid + 1, self.buffer_length_samples)
    
    def get_window(self, window_size_sec: float) -> Optional[np.ndarray]:
        """
        Extract the most recent window of data.
        
        Parameters
        ----------
        window_size_sec : float
            Window duration in seconds
        
        Returns
        -------
        window : np.ndarray or None
            Shape (n_channels, window_samples) if enough data available, else None
        """
        window_samples = int(window_size_sec * self.fs)
        
        if window_samples > self.buffer_length_samples:
            raise ValueError(
                f"Window size {window_size_sec}s ({window_samples} samples) "
                f"exceeds buffer size {self.buffer_length_sec}s ({self.buffer_length_samples} samples)"
            )
        
        with self._lock:
            if self._n_valid < window_samples:
                return None
            
            # Extract window from circular buffer
            window = np.zeros((self.n_channels, window_samples), dtype=self.dtype)
            
            if self._n_valid == self.buffer_length_samples:
                # Buffer is full: extract from circular buffer
                start_idx = (self._write_pos - window_samples) % self.buffer_length_samples
                if start_idx + window_samples <= self.buffer_length_samples:
                    # No wrap-around
                    window[:] = self._buffer[:, start_idx : start_idx + window_samples]
                else:
                    # Wrap-around case
                    n_before_wrap = self.buffer_length_samples - start_idx
                    window[:, :n_before_wrap] = self._buffer[:, start_idx:]
                    window[:, n_before_wrap:] = self._buffer[:, :window_samples - n_before_wrap]
            else:
                # Buffer not full yet: extract from beginning
                window[:] = self._buffer[:, self._n_valid - window_samples : self._n_valid]
            
            return window
    
    def get_all(self) -> Optional[np.ndarray]:
        """
        Get all valid samples in the buffer.
        
        Returns
        -------
        data : np.ndarray or None
            Shape (n_channels, n_valid_samples) if data available, else None
        """
        with self._lock:
            if self._n_valid == 0:
                return None
            
            if self._n_valid == self.buffer_length_samples:
                # Buffer is full: extract circularly
                data = np.zeros((self.n_channels, self._n_valid), dtype=self.dtype)
                start_idx = self._write_pos
                n_before_wrap = self.buffer_length_samples - start_idx
                data[:, :n_before_wrap] = self._buffer[:, start_idx:]
                data[:, n_before_wrap:] = self._buffer[:, :start_idx]
                return data
            else:
                # Buffer not full: return from beginning
                return self._buffer[:, :self._n_valid].copy()
    
    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._write_pos = 0
            self._n_valid = 0
            self._buffer.fill(0)
    
    @property
    def n_valid_samples(self) -> int:
        """Number of valid samples currently in buffer."""
        with self._lock:
            return self._n_valid
    
    @property
    def n_valid_seconds(self) -> float:
        """Duration of valid data in seconds."""
        return self.n_valid_samples / self.fs
