# Real-Time Sliding-Window Streaming

This document describes the real-time sliding-window streaming implementation for the EEG BCI framework.

## Overview

The framework supports two streaming modes:

1. **Trial mode** (`streaming_mode: "trial"`): Processes complete trials in batch, simulating real-time by sleeping between trials.
2. **Sliding mode** (`streaming_mode: "sliding"`): True real-time streaming with a sliding window buffer that continuously processes EEG samples.

## Architecture

### Components

1. **CircularRingBuffer** (`bci_framework/streaming/buffer.py`):
   - Thread-safe circular buffer for continuous EEG samples
   - Configurable buffer length (default: 10 seconds)
   - Supports appending samples and extracting sliding windows

2. **RealtimeInferenceEngine** (`bci_framework/streaming/realtime_inference.py`):
   - Manages sliding-window inference loop
   - Triggers inference at regular intervals (e.g., every 100ms)
   - Tracks latency statistics
   - Extracts windows from buffer and runs them through the pipeline

3. **Pipeline.predict_stream()**:
   - Method added to Pipeline class for streaming inference
   - Ensures causal preprocessing (no future samples)

## Configuration

### Config.yaml Settings

```yaml
dataset:
  streaming_mode: "sliding"  # "trial" | "sliding"

streaming:
  mode: "sliding"  # Alternative location for streaming mode
  window_size_sec: 1.5  # Size of sliding window (seconds)
  buffer_length_sec: 10.0  # Ring buffer length (seconds)
  update_interval_sec: 0.1  # Inference trigger interval (e.g., 100ms)
  real_time_timing: true  # Sleep to simulate real-time
  stream_full_test_set: true  # Stream all test trials

advanced_preprocessing:
  gedai:
    mode: "sliding"  # Use sliding mode for GEDAI (causal, low-latency)
    window_sec: 10.0  # GEDAI sliding window duration
    update_interval_sec: 1.0  # GEDAI eigenvector update interval
```

## Usage

### Running Sliding-Window Mode

1. Set `dataset.streaming_mode: "sliding"` in `config.yaml`
2. Ensure `mode: "online"` is set (or will be set automatically for sliding mode)
3. Run the main script:

```bash
PYTHONPATH=. python main.py
```

The framework will:
- Load and fit pipelines on training data
- Select the best pipeline
- Stream test data sample-by-sample through the ring buffer
- Trigger inference every `update_interval_sec` seconds
- Log latency statistics

### Example: Minimal Replay Script

See `examples/realtime_replay.py` for a minimal example that demonstrates:
- Creating a ring buffer
- Pushing samples
- Running inference engine
- Measuring latency

## Causal Preprocessing

In sliding mode, all preprocessing steps must be causal (no future samples):

- **Notch/Bandpass filters**: Use `lfilter` (causal) instead of `filtfilt` (non-causal)
- **ICA**: Only applies learned transforms (no refitting online)
- **GEDAI**: Uses sliding mode with causal covariance updates
- **Wavelet**: Only applies learned transforms

The `PreprocessingManager` automatically enables causal filters when `mode: "online"` is set.

## GEDAI Online Sliding Mode

GEDAI supports online sliding mode for real-time use:

- **Mode**: Set `advanced_preprocessing.gedai.mode: "sliding"`
- **Window**: Maintains a sliding covariance buffer (`window_sec` seconds)
- **Update interval**: Recomputes eigenvectors every `update_interval_sec` seconds
- **Causality**: Only uses past and current samples (strict causality)

The sliding mode uses PyTorch for efficient GPU-accelerated generalized eigenvalue decomposition.

## Latency Measurement

The `RealtimeInferenceEngine` tracks end-to-end latency for each inference:

- **Mean latency**: Average processing time per window
- **Median latency**: Median processing time
- **Max/Min latency**: Peak and minimum latencies

Latency is logged:
- Every 10 trials during streaming
- At the end of streaming (summary statistics)

## Thread Safety

The `CircularRingBuffer` uses `threading.Lock` to ensure thread-safe operations:
- `append()`: Thread-safe sample appending
- `get_window()`: Thread-safe window extraction
- `clear()`: Thread-safe buffer clearing

## Performance Considerations

1. **Window size**: Smaller windows (1-2s) reduce latency but may reduce accuracy
2. **Update interval**: More frequent updates (50-100ms) provide smoother predictions but increase CPU load
3. **Buffer length**: Longer buffers (10s) allow more context but use more memory
4. **GEDAI update interval**: Less frequent updates (1-2s) reduce computation but may miss rapid changes

## Backward Compatibility

The implementation maintains backward compatibility:
- Default mode is `"trial"` (existing behavior)
- Existing trial-based pipelines continue to work
- No changes required to feature extractors or classifiers

## Limitations

1. **No online refitting**: CSP, classifiers, and ICA are not refit online (strict train/test separation)
2. **Fixed window size**: Window size is fixed per pipeline (no adaptive sizing)
3. **Single pipeline**: Only one pipeline can be active at a time (no ensemble streaming)

## Future Enhancements

Potential improvements:
- Adaptive window sizing based on signal quality
- Online classifier adaptation (concept drift handling)
- Multi-pipeline ensemble streaming
- Real hardware integration (OpenBCI, Emotiv)
