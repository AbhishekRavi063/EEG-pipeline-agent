"""Desktop GUI: live EEG, filtered EEG, CSP/features, accuracy vs time, pipeline comparison."""

import logging
import threading
import time
from queue import Empty, Queue
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


class BCIApp:
    """
    Matplotlib-based GUI. Shows:
    - Raw EEG channels
    - Filtered EEG
    - Feature visualization (e.g. CSP)
    - Pipeline accuracy vs time
    - Pipeline comparison bar chart
    - Live prediction (e.g. left hand, right hand)
    """

    def __init__(
        self,
        fs: float,
        channel_names: list[str],
        class_names: list[str],
        refresh_rate_ms: int = 100,
        eeg_channels_display: int = 8,
        window_seconds: float = 4.0,
    ) -> None:
        self.fs = fs
        self.channel_names = channel_names
        self.class_names = class_names
        self.refresh_rate_ms = refresh_rate_ms
        self.eeg_channels_display = min(eeg_channels_display, len(channel_names))
        self.window_samples = int(window_seconds * fs)
        self._raw_buffer: np.ndarray | None = None
        self._filtered_buffer: np.ndarray | None = None
        self._feature_vectors: list[np.ndarray] = []
        self._accuracy_history: list[tuple[str, float]] = []
        self._pipeline_metrics: dict[str, float] = {}
        self._current_prediction: int | None = None
        self._current_prediction_name: str = ""
        self._best_pipeline_name: str = ""
        self._trial_index: int = 0
        self._n_trials_stream: int = 0
        self._dataset_source: str = ""
        self._phase: str = ""  # "Calibration" | "Live" for online mode
        self._rolling_accuracy: float | None = None
        self._is_labeled: bool = True
        self._calibration_metrics: dict = {}
        self._running = False
        self._fig = None
        self._queues: dict[str, Queue] = {}
        self._on_close: Callable[[], None] | None = None

    def set_raw_buffer(self, data: np.ndarray) -> None:
        """Update raw EEG buffer (n_channels, n_samples)."""
        self._raw_buffer = np.asarray(data, dtype=np.float64)

    def set_filtered_buffer(self, data: np.ndarray) -> None:
        """Update filtered EEG buffer."""
        self._filtered_buffer = np.asarray(data, dtype=np.float64)

    def update_accuracy(self, pipeline_name: str, accuracy: float) -> None:
        """Append accuracy for a pipeline over time."""
        self._accuracy_history.append((pipeline_name, accuracy))
        self._pipeline_metrics[pipeline_name] = accuracy

    def set_pipeline_metrics(self, metrics: dict[str, float]) -> None:
        """Set all pipeline accuracies for bar chart."""
        self._pipeline_metrics = dict(metrics)

    def set_prediction(self, class_idx: int) -> None:
        """Set current live prediction (class index)."""
        self._current_prediction = class_idx
        if 0 <= class_idx < len(self.class_names):
            self._current_prediction_name = self.class_names[class_idx]
        else:
            self._current_prediction_name = str(class_idx)

    def set_best_pipeline(self, name: str) -> None:
        self._best_pipeline_name = name

    def set_trial_progress(self, trial_index: int, n_trials: int) -> None:
        """Set current trial index for real-time streaming progress (e.g. Trial 5/24)."""
        self._trial_index = trial_index
        self._n_trials_stream = n_trials

    def set_dataset_source(self, source: str) -> None:
        """Set dataset name for display in GUI title (e.g. 'BCI_IV_2a')."""
        self._dataset_source = source

    def set_subject(self, subject_id: int | str | None) -> None:
        """Set current subject. For API consistency with WebApp."""
        pass  # Desktop GUI uses dataset_source in title

    def set_phase(self, phase: str) -> None:
        """Set phase for online mode: 'Calibration' or 'Live'."""
        self._phase = phase

    def set_rolling_accuracy(self, acc: float | None) -> None:
        """Set rolling accuracy over recent live trials (for GUI)."""
        self._rolling_accuracy = acc

    def set_trial_labeled(self, is_labeled: bool) -> None:
        """Legacy. Prefer set_trial_source."""
        self._is_labeled = is_labeled

    def set_trial_source(self, from_t_session: bool | None, has_label: bool) -> None:
        """Set current trial source: T or E session. For API consistency."""
        self._is_labeled = has_label

    def set_calibration_metrics(self, metrics: dict[str, dict]) -> None:
        """Set per-pipeline metrics from online calibration (for GUI bar chart)."""
        self._calibration_metrics = dict(metrics)
        for name, m in metrics.items():
            a = m.get("accuracy", 0.0)
            self._pipeline_metrics[name] = a

    def set_calibration_metrics_full(self, metrics: dict[str, dict]) -> None:
        """Alias for set_calibration_metrics (web server has detailed version)."""
        self.set_calibration_metrics(metrics)

    def record_live_trial_result(self, correct: bool) -> None:
        """Record one live trial result for accuracy-over-time (no-op for desktop GUI)."""
        pass  # Desktop GUI uses update_accuracy instead

    def run(
        self,
        data_callback: Callable[[], tuple[np.ndarray | None, np.ndarray | None]] | None = None,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        """Run GUI loop. data_callback() -> (raw, filtered) for live updates."""
        import matplotlib
        # Set interactive backend before importing pyplot (TkAgg needs _tkinter; MacOSX works on macOS)
        _interactive_backend = None
        for backend in ("MacOSX", "Qt5Agg", "TkAgg", "Qt4Agg", "GTK4Agg", "GTK3Agg", "WXAgg"):
            try:
                matplotlib.use(backend, force=True)
                _interactive_backend = backend
                break
            except Exception as e:
                logger.debug("Backend %s failed: %s", backend, e)
                continue
        if _interactive_backend is None:
            matplotlib.use("Agg", force=True)
            logger.warning(
                "No interactive GUI backend (install tkinter or PyQt5). "
                "Run with --no-gui for results only, or install python3-tk / PyQt5 to see the live stream window."
            )
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        self._on_close = on_close
        self._running = True
        n_ch = self.eeg_channels_display
        fig = plt.figure(figsize=(16, 14))
        # Vertical stacking with height ratios: Raw and Processed EEG get MORE space (3x), metrics get less (1x)
        gs = GridSpec(4, 1, figure=fig, height_ratios=[3, 3, 1, 1], hspace=0.4)
        ax_raw = fig.add_subplot(gs[0])
        ax_filt = fig.add_subplot(gs[1])
        ax_bar = fig.add_subplot(gs[2])
        ax_acc = fig.add_subplot(gs[3])

        t_axis = np.arange(self.window_samples) / self.fs
        lines_raw = []
        lines_filt = []
        for i in range(n_ch):
            l, = ax_raw.plot(t_axis, np.zeros(self.window_samples), label=self.channel_names[i] if i < len(self.channel_names) else f"Ch{i}")
            lines_raw.append(l)
        for i in range(n_ch):
            l, = ax_filt.plot(t_axis, np.zeros(self.window_samples))
            lines_filt.append(l)
        ax_raw.set_title("Raw EEG — each colored line = one EEG channel (electrode)")
        ax_raw.set_ylabel("µV (stacked)")
        ax_raw.set_xlim(0, t_axis[-1])
        ax_raw.tick_params(labelbottom=False)
        ax_raw.legend(loc="upper right", fontsize=6, ncol=4)
        ax_filt.set_title("Processed (filtered) EEG — each colored line = one channel, same order as above")
        ax_filt.set_ylabel("µV (stacked)")
        ax_filt.set_xlim(0, t_axis[-1])
        ax_filt.tick_params(labelbottom=False)

        # Initial title including live pipeline method
        _title = "BCI Motor Imagery"
        if self._dataset_source:
            _title += f" [{self._dataset_source}]"
        if self._best_pipeline_name:
            _title += f" — Live pipeline: {self._best_pipeline_name}"
        fig.suptitle(_title, fontsize=10)

        ax_bar.set_title("Pipeline comparison (calibration metrics)")
        ax_bar.set_ylabel("Accuracy")
        ax_bar.set_ylim(0, 1.05)
        ax_bar.tick_params(labelbottom=True)

        acc_line, = ax_acc.plot([], [], "b-", label="Accuracy", linewidth=2)
        ax_acc.set_title("Accuracy over time (live stream)")
        ax_acc.set_xlabel("Trial")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(0, 1.05)
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)

        bar_containers = []
        bar_labels = []

        def _buffer_2d(buf: np.ndarray) -> np.ndarray:
            """Ensure buffer is (n_channels, n_samples) for plotting."""
            if buf is None:
                return None
            buf = np.asarray(buf, dtype=np.float64)
            if buf.ndim == 1:
                buf = buf.reshape(1, -1)
            elif buf.ndim == 3:
                buf = buf[0]
            return buf

        def _stacked_ylim(buf_2d: np.ndarray, n_ch_display: int, ax_handle) -> float:
            """Return vertical spacing for stacked traces; set ax ylim so all channels fit."""
            if buf_2d is None or buf_2d.size == 0:
                return 0.0
            flat = buf_2d.ravel()
            lo, hi = np.percentile(flat, [1, 99])
            span = max(hi - lo, 1e-9)
            spacing = span * 2.0  # gap between channels so traces don't overlap
            margin = 0.2 * spacing
            y_min = lo - margin
            y_max = hi + (n_ch_display - 1) * spacing + margin
            ax_handle.set_ylim(y_min, y_max)
            return spacing

        def update_plot():
            if not self._running:
                return
            if data_callback:
                raw, filt = data_callback()
                if raw is not None:
                    self.set_raw_buffer(raw)
                if filt is not None:
                    self.set_filtered_buffer(filt)
            raw_buf = _buffer_2d(self._raw_buffer)
            if raw_buf is not None:
                n_samp = min(raw_buf.shape[1], self.window_samples)
                n_ch_plot = min(n_ch, raw_buf.shape[0])
                spacing_raw = _stacked_ylim(raw_buf, n_ch_plot, ax_raw)
                for i in range(n_ch):
                    ch_idx = min(i, raw_buf.shape[0] - 1)
                    if raw_buf.shape[0] == 0:
                        continue
                    d = raw_buf[ch_idx, -n_samp:].astype(np.float64)
                    pad = self.window_samples - len(d)
                    if pad > 0:
                        d = np.pad(d, (pad, 0), mode="edge")
                    # Stack: channel i offset by i * spacing so traces don't overlap
                    lines_raw[i].set_ydata(d + i * spacing_raw)
                ax_raw.set_ylabel("µV (stacked)")
            filt_buf = _buffer_2d(self._filtered_buffer)
            if filt_buf is not None:
                n_samp = min(filt_buf.shape[1], self.window_samples)
                n_ch_plot = min(n_ch, filt_buf.shape[0])
                spacing_filt = _stacked_ylim(filt_buf, n_ch_plot, ax_filt)
                for i in range(n_ch):
                    ch_idx = min(i, filt_buf.shape[0] - 1)
                    if filt_buf.shape[0] == 0:
                        continue
                    d = filt_buf[ch_idx, -n_samp:].astype(np.float64)
                    pad = self.window_samples - len(d)
                    if pad > 0:
                        d = np.pad(d, (pad, 0), mode="edge")
                    lines_filt[i].set_ydata(d + i * spacing_filt)
                ax_filt.set_ylabel("µV (stacked)")
            if self._accuracy_history:
                names, accs = zip(*self._accuracy_history[-100:])
                trials = list(range(len(accs)))
                acc_line.set_data(trials, accs)
                if trials:
                    ax_acc.set_xlim(0, max(trials) + 1)
            if self._pipeline_metrics:
                names = list(self._pipeline_metrics.keys())
                vals = list(self._pipeline_metrics.values())
                for c in ax_bar.containers:
                    c.remove()
                bars = ax_bar.bar(range(len(names)), vals, color="steelblue")
                ax_bar.set_xticks(range(len(names)))
                ax_bar.set_xticklabels(names, rotation=45, ha="right")
                ax_bar.set_ylabel("Accuracy")
                ax_bar.set_title("Pipeline comparison (bar)")
                ax_bar.set_ylim(0, 1.05)
            # Title: phase (online), dataset, pipeline, trial progress, prediction, rolling acc
            title_parts = ["BCI Motor Imagery"]
            if self._phase:
                title_parts.append(f" [{self._phase}]")
            if self._dataset_source:
                title_parts.append(f" [{self._dataset_source}]")
            if self._best_pipeline_name:
                title_parts.append(f" — Pipeline: {self._best_pipeline_name}")
            if self._n_trials_stream > 0:
                title_parts.append(f" — Trial {self._trial_index + 1}/{self._n_trials_stream}")
            if self._current_prediction_name:
                title_parts.append(f" — {self._current_prediction_name}")
            if self._rolling_accuracy is not None:
                title_parts.append(f" — Rolling acc: {self._rolling_accuracy:.2f}")
            fig.suptitle("".join(title_parts), fontsize=10)
            fig.canvas.draw_idle()

        timer = fig.canvas.new_timer(interval=self.refresh_rate_ms)
        timer.add_callback(update_plot)
        timer.start()
        # Paint once immediately so raw/filtered plots show seeded data (not empty)
        update_plot()

        def on_close_event(_event):
            self._running = False
            if self._on_close:
                self._on_close()
            plt.close()

        fig.canvas.mpl_connect("close_event", on_close_event)
        plt.show()

    def run_async(
        self,
        data_callback: Callable[[], tuple[np.ndarray | None, np.ndarray | None]] | None = None,
    ) -> None:
        """Start GUI in a separate thread (non-blocking)."""
        def run():
            self.run(data_callback=data_callback)
        t = threading.Thread(target=run, daemon=True)
        t.start()
