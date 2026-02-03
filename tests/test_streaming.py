"""Unit tests for EEG streaming buffer and sliding window."""

import numpy as np
import pytest

from bci_framework.utils.streaming import EEGStreamBuffer, sliding_window_chunks, stream_chunk


def test_eeg_stream_buffer_append_and_get():
    buf = EEGStreamBuffer(n_channels=4, max_samples=100)
    chunk = np.random.randn(4, 30)
    buf.append(chunk)
    w = buf.get_window(30)
    assert w is not None
    assert w.shape == (4, 30)
    np.testing.assert_array_almost_equal(w, chunk)


def test_eeg_stream_buffer_overflow():
    buf = EEGStreamBuffer(n_channels=2, max_samples=50)
    for _ in range(3):
        buf.append(np.random.randn(2, 25))
    w = buf.get_window(50)
    assert w is not None
    assert w.shape == (2, 50)


def test_sliding_window_chunks():
    data = np.random.randn(2, 3, 100)
    chunks = list(sliding_window_chunks(data, window_samples=20, overlap_ratio=0.5, axis=-1))
    assert len(chunks) >= 1
    chunk, start, end = chunks[0]
    assert chunk.shape == (2, 3, 20)
    assert start == 0 and end == 20


def test_stream_chunk():
    trial = np.random.randn(4, 200)
    chunks = list(stream_chunk(trial, window_samples=50, overlap_ratio=0.5))
    assert len(chunks) >= 1
    c = chunks[0]
    assert c.shape == (4, 50)
