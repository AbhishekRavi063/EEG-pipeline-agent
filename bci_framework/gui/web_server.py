"""
Web interface for BCI: detailed zoomable EEG view via Plotly.js.
Serves a single-page app and streams state over WebSocket.
"""

import asyncio

# EEG frequency bands (Hz): delta, theta, alpha, beta, gamma
EEG_BANDS = [
    ("delta", 0.5, 4),
    ("theta", 4, 8),
    ("alpha", 8, 13),
    ("beta", 13, 30),
    ("gamma", 30, 45),
]

# Noise/artifact bands (Hz): drift, power line, EMG
NOISE_BANDS = [
    ("drift", 0.1, 0.5),      # baseline drift
    ("line_50", 48, 52),      # power line (EU)
    ("line_60", 58, 62),      # power line (US)
    ("emg", 45, 100),         # muscle artifacts
]

ALL_BANDS = EEG_BANDS + NOISE_BANDS


def _compute_band_powers(data: "np.ndarray", fs: float) -> dict[str, float]:
    """Compute mean power in each EEG band and noise band from data. Returns band name -> power (µV²)."""
    try:
        from scipy.signal import welch
    except ImportError:
        return {}
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.size < 256:
        return {}
    # Welch PSD: average across channels
    powers = {}
    for ch in range(data.shape[0]):
        f, psd = welch(data[ch], fs=fs, nperseg=min(256, data.shape[1] // 2))
        for name, low, high in ALL_BANDS:
            if high > fs / 2:
                continue
            mask = (f >= low) & (f <= high)
            if mask.any():
                p = float(np.trapezoid(psd[mask], f[mask]))
                powers[name] = powers.get(name, 0) + p
    n_ch = data.shape[0]
    return {k: v / max(1, n_ch) for k, v in powers.items()}


def _compute_psd_welch(data: "np.ndarray", fs: float, nperseg: int = 256):
    """Compute PSD via Welch's method. Returns (freqs_Hz, psd_mean) in µV²/Hz, or (None, None) on failure."""
    try:
        from scipy.signal import welch
    except ImportError:
        return None, None
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.size < nperseg * 2:
        return None, None
    nperseg = min(nperseg, data.shape[1] // 2)
    freqs, psd_0 = welch(data[0], fs=fs, nperseg=nperseg)
    psd_sum = np.array(psd_0, dtype=np.float64)
    for ch in range(1, data.shape[0]):
        _, p = welch(data[ch], fs=fs, nperseg=nperseg)
        psd_sum += p
    psd_mean = (psd_sum / data.shape[0]).tolist()
    return freqs.tolist(), psd_mean


import json
import logging
import threading
from pathlib import Path
from queue import Empty, Queue

import numpy as np

logger = logging.getLogger(__name__)

# Default port for web UI
DEFAULT_WEB_PORT = 8765


def _to_serializable(obj):
    """Convert numpy arrays to lists for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    return obj


class WebSocketManager:
    """Holds WebSocket connections and state; broadcasts state to all clients."""
    
    def __init__(self):
        self.connections: list = []
        self.state: dict = {}
        self._queue: Queue = Queue()
        self._loop = None
    
    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
    
    def update_state(self, **kwargs) -> None:
        """Merge kwargs into state (arrays converted to list), then queue broadcast."""
        for k, v in kwargs.items():
            if v is None:
                self.state.pop(k, None)
            else:
                self.state[k] = _to_serializable(v)
        self._queue.put_nowait(dict(self.state))
    
    def get_state(self) -> dict:
        """Return current state as JSON-serializable dict."""
        return {k: _to_serializable(v) for k, v in self.state.items()}
    
    async def register(self, websocket) -> None:
        self.connections.append(websocket)
        # Send initial state so new client sees current view
        state = self.get_state()
        if state:
            try:
                await websocket.send_text(json.dumps(state))
            except Exception as e:
                logger.debug("Send initial state: %s", e)
    
    def unregister(self, websocket) -> None:
        if websocket in self.connections:
            self.connections.remove(websocket)
    
    async def broadcast_loop(self) -> None:
        """Run in background: read from queue and send to all connections. Do not block the event loop."""
        loop = asyncio.get_event_loop()

        def get_from_queue():
            try:
                return self._queue.get(timeout=0.5)
            except Empty:
                return None

        while True:
            # Run blocking queue.get in executor so the event loop can accept HTTP/WS connections
            state = await loop.run_in_executor(None, get_from_queue)
            if state is None:
                continue
            dead = []
            for ws in self.connections:
                try:
                    await ws.send_text(json.dumps(state))
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.unregister(ws)


def _apply_filters_backend(
    raw_buffer: list,
    fs: float,
    bandpass_low: float,
    bandpass_high: float,
    notch_freq: float,
    use_ica: bool,
) -> dict:
    """Apply bandpass, notch, and optional ICA. Returns {cleaned_data: [...]} or {error, cleaned_data: None}."""
    try:
        import numpy as np
        from bci_framework.preprocessing.bandpass import BandpassFilter
        from bci_framework.preprocessing.notch import NotchFilter
    except ImportError as e:
        return {"error": str(e), "cleaned_data": None}
    if fs <= 0:
        return {"error": "Invalid sampling rate", "cleaned_data": None}
    nyq = 0.5 * fs
    if bandpass_low >= bandpass_high:
        return {"error": "Bandpass low must be < bandpass high", "cleaned_data": None}
    bandpass_low = max(0.1, min(bandpass_low, nyq - 1))
    bandpass_high = max(bandpass_low + 0.5, min(bandpass_high, nyq - 0.5))
    data = np.asarray(raw_buffer, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] < 64:
        return {"error": "Invalid raw_buffer shape (need [n_channels, n_samples], min 64 samples)", "cleaned_data": None}
    # X for filters: (n_trials, n_channels, n_samples)
    X = data[np.newaxis, :, :]
    try:
        out = X.copy()
        if notch_freq > 0 and notch_freq < nyq:
            notch = NotchFilter(fs=fs, freq=notch_freq, quality=30.0, causal=False)
            out = notch.transform(out)
        bandpass = BandpassFilter(
            fs=fs, lowcut=bandpass_low, highcut=bandpass_high, order=5, causal=False
        )
        out = bandpass.transform(out)
        if use_ica and X.shape[1] >= 2:
            try:
                from sklearn.decomposition import FastICA
                n_ch = X.shape[1]
                X_flat = out.transpose(0, 2, 1).reshape(-1, n_ch)
                n_comp = min(15, n_ch)
                ica = FastICA(n_components=n_comp, max_iter=500, random_state=42)
                S = ica.fit_transform(X_flat)
                X_recon = ica.inverse_transform(S)
                out = X_recon.reshape(X.shape[0], X.shape[2], n_ch).transpose(0, 2, 1)
            except Exception:
                pass
        cleaned = out[0].tolist()
        return {"cleaned_data": cleaned}
    except Exception as e:
        logger.exception("apply_filters_backend")
        return {"error": str(e), "cleaned_data": None}


def create_app(static_dir: Path, manager: WebSocketManager):
    """Create FastAPI app that serves static files and WebSocket."""
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, WebSocket, Request
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    executor = ThreadPoolExecutor(max_workers=1)
    
    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(manager.broadcast_loop())
        logger.info("Web server ready at http://127.0.0.1:%s — open in browser", getattr(manager, "_port", DEFAULT_WEB_PORT) or DEFAULT_WEB_PORT)
        yield
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    
    app = FastAPI(title="BCI EEG Viewer", lifespan=lifespan)

    from fastapi import APIRouter
    api_router = APIRouter(prefix="/api", tags=["api"])

    @api_router.post("/apply_filters")
    async def api_apply_filters(request: Request):
        """Apply bandpass, notch, and optional ICA to raw EEG. Returns cleaned_data."""
        from fastapi.responses import JSONResponse
        try:
            body = await request.json()
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": "Invalid JSON: " + str(e), "cleaned_data": None})
        raw_buffer = body.get("raw_buffer")
        try:
            fs_val = float(body.get("fs", 250.0))
            bandpass_low = float(body.get("bandpass_low", 1.0))
            bandpass_high = float(body.get("bandpass_high", 45.0))
            notch_freq = float(body.get("notch_freq", 50.0))
            use_ica = bool(body.get("use_ica", False))
        except (TypeError, ValueError) as e:
            return JSONResponse(status_code=400, content={"error": "Invalid filter parameters: " + str(e), "cleaned_data": None})
        if not raw_buffer or not isinstance(raw_buffer, list) or len(raw_buffer) == 0:
            return JSONResponse(status_code=400, content={"error": "raw_buffer required (list of channel arrays)", "cleaned_data": None})
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                executor,
                lambda: _apply_filters_backend(
                    raw_buffer=raw_buffer,
                    fs=fs_val,
                    bandpass_low=bandpass_low,
                    bandpass_high=bandpass_high,
                    notch_freq=notch_freq,
                    use_ica=use_ica,
                ),
            )
            if result.get("error"):
                return JSONResponse(status_code=400, content=result)
            return result
        except Exception as e:
            logger.exception("apply_filters failed")
            err_msg = str(e) if str(e) else "Filter processing failed"
            return JSONResponse(status_code=500, content={"error": err_msg, "cleaned_data": None})

    @api_router.post("/compute_psd")
    async def api_compute_psd(request: Request):
        """Compute PSD (Welch) for a given buffer. Returns { freq, psd } in Hz and µV²/Hz."""
        from fastapi.responses import JSONResponse
        try:
            body = await request.json()
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": "Invalid JSON: " + str(e)})
        buffer = body.get("buffer") or body.get("raw_buffer")
        fs_val = float(body.get("fs", 250.0))
        if not buffer or not isinstance(buffer, list) or len(buffer) == 0:
            return JSONResponse(status_code=400, content={"error": "buffer required"})
        try:
            import numpy as np
            data = np.asarray(buffer, dtype=np.float64)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.size < 256:
                return JSONResponse(status_code=400, content={"error": "buffer too short for PSD"})
            freqs, psd = _compute_psd_welch(data, fs_val)
            if freqs is None or psd is None:
                return JSONResponse(status_code=500, content={"error": "PSD computation failed"})
            return {"freq": freqs, "psd": psd}
        except Exception as e:
            logger.exception("compute_psd failed")
            return JSONResponse(status_code=500, content={"error": str(e)})

    @api_router.get("/dataset_info")
    async def api_dataset_info():
        """Return dataset metadata from live app state (same as homepage) for the compare page."""
        state = manager.get_state()
        out = {}
        for key in ("dataset_source", "n_channels", "fs", "window_seconds", "available_subjects"):
            if key in state and state[key] is not None:
                out[key] = state[key]
        return out

    app.include_router(api_router)

    @app.get("/")
    async def index():
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "BCI Web UI", "static_dir": str(static_dir)}

    @app.get("/compare")
    async def compare_page():
        """Pipeline A vs B comparison page (professor's suggestion)."""
        compare_file = static_dir / "compare.html"
        if compare_file.exists():
            return FileResponse(compare_file)
        return {"message": "compare.html not found", "path": str(compare_file)}

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.post("/api/compare_pipelines")
    async def api_compare_pipelines(request: Request):
        """Run Pipeline A and B on same subjects, return tables and statistical comparison (p-values)."""
        try:
            body = await request.json()
        except Exception:
            body = {}
        dataset = body.get("dataset", "BNCI2014_001")
        # Map internal dataset names to MOABB class names (compare uses MOABB loader)
        if dataset and str(dataset).strip().upper() in ("BCI_IV_2A", "BCI_IV_2a"):
            dataset = "BNCI2014_001"
        subjects = body.get("subjects", [1, 2, 3])
        if not isinstance(subjects, list):
            subjects = [int(x) for x in str(subjects).replace(",", " ").split()]
        subjects = [int(s) for s in subjects]
        config_path_a = body.get("config_path_a")
        config_path_b = body.get("config_path_b")
        override_b = body.get("override_b")
        pipeline_a = body.get("pipeline_a")
        pipeline_b = body.get("pipeline_b")
        name_a = body.get("name_a", "Pipeline_A")
        name_b = body.get("name_b", "Pipeline_B")
        test = body.get("test", "ttest")
        loop = asyncio.get_event_loop()
        try:
            from bci_framework.evaluation import run_ab_comparison
            result = await loop.run_in_executor(
                executor,
                lambda: run_ab_comparison(
                    dataset=dataset,
                    subjects=subjects,
                    config_path_a=config_path_a,
                    config_path_b=config_path_b,
                    override_b=override_b,
                    pipeline_a=pipeline_a,
                    pipeline_b=pipeline_b,
                    name_a=name_a,
                    name_b=name_b,
                    test=test,
                ),
            )
            return result
        except Exception as e:
            logger.exception("compare_pipelines failed")
            return {"error": str(e), "table_a": [], "table_b": [], "comparison": {}}
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        manager.set_event_loop(asyncio.get_event_loop())
        await manager.register(websocket)
        try:
            while True:
                await websocket.receive_text()
        except Exception:
            pass
        finally:
            manager.unregister(websocket)
    
    return app


class WebApp:
    """
    Web UI adapter: same API as BCIApp but pushes state to browser via WebSocket.
    Use with --web to get a detailed, zoomable Plotly view of raw and processed EEG.
    """
    
    def __init__(
        self,
        fs: float,
        channel_names: list[str],
        class_names: list[str],
        refresh_rate_ms: int = 100,
        eeg_channels_display: int = 8,
        window_seconds: float = 4.0,
        manager: WebSocketManager | None = None,
    ) -> None:
        self.fs = fs
        self.channel_names = list(channel_names)
        self.class_names = list(class_names)
        self.refresh_rate_ms = refresh_rate_ms
        self.eeg_channels_display = min(eeg_channels_display, len(channel_names))
        self.window_samples = int(window_seconds * fs)
        self._manager = manager or WebSocketManager()
        self._raw_buffer: np.ndarray | None = None
        self._filtered_buffer: np.ndarray | None = None
        self._current_prediction: int | None = None
        self._current_prediction_name: str = ""
        self._best_pipeline_name: str = ""
        self._trial_index: int = 0
        self._n_trials_stream: int = 0
        self._dataset_source: str = ""
        self._subject: str = ""  # e.g. "A01" or "Subject 1"
        self._phase: str = ""
        self._rolling_accuracy: float | None = None
        self._pipeline_metrics: dict[str, float] = {}
        self._pipeline_metrics_detail: dict[str, dict] = {}  # full metrics per pipeline (accuracy, kappa, latency_ms, stability)
        self._accuracy_history: list[tuple[str, float]] = []
        self._live_correct: list[bool] = []  # per-trial correctness during live stream
        self._is_labeled: bool = True  # current trial has label (T) or not (E)
        self._trial_source: str = ""  # "T (labeled)" or "E (unlabeled)"
        self._available_subjects: list[int | str] = []  # subject IDs available for the dataset
    
    def _broadcast(self) -> None:
        """Push current state to web clients."""
        state = {
            "fs": self.fs,
            "channel_names": self.channel_names[: self.eeg_channels_display],
            "n_channels": len(self.channel_names),
            "window_samples": self.window_samples,
            "window_seconds": self.window_samples / self.fs if self.fs else 0,
            "class_names": self.class_names,
            "phase": self._phase,
            "dataset_source": self._dataset_source,
            "subject": self._subject,
            "best_pipeline": self._best_pipeline_name,
            "trial_index": self._trial_index,
            "n_trials": self._n_trials_stream,
            "prediction": self._current_prediction,
            "prediction_name": self._current_prediction_name,
            "rolling_accuracy": self._rolling_accuracy,
            "pipeline_metrics": self._pipeline_metrics,
            "pipeline_metrics_detail": self._pipeline_metrics_detail,
            "accuracy_history": self._live_accuracy_history() if self._live_correct else self._accuracy_history[-200:],
            "is_labeled": self._is_labeled,
            "trial_source": self._trial_source,
            "available_subjects": list(self._available_subjects),
        }
        if self._raw_buffer is not None:
            buf = np.asarray(self._raw_buffer, dtype=np.float64)
            if buf.ndim == 1:
                buf = buf.reshape(1, -1)
            elif buf.ndim == 3:
                buf = buf[0]
            n_ch = min(self.eeg_channels_display, buf.shape[0])
            state["raw_buffer"] = buf[:n_ch, -self.window_samples :].tolist()
            band_powers = _compute_band_powers(buf, self.fs)
            if band_powers:
                state["band_powers"] = band_powers
            psd_freq, psd_raw = _compute_psd_welch(buf, self.fs)
            if psd_freq is not None and psd_raw is not None:
                state["psd_freq"] = psd_freq
                state["psd_raw"] = psd_raw
        if self._filtered_buffer is not None:
            buf = np.asarray(self._filtered_buffer, dtype=np.float64)
            if buf.ndim == 1:
                buf = buf.reshape(1, -1)
            elif buf.ndim == 3:
                buf = buf[0]
            n_ch = min(self.eeg_channels_display, buf.shape[0])
            state["filtered_buffer"] = buf[:n_ch, -self.window_samples :].tolist()
            band_powers_filt = _compute_band_powers(buf, self.fs)
            if band_powers_filt:
                state["band_powers_filtered"] = band_powers_filt
            psd_freq_f, psd_filt = _compute_psd_welch(buf, self.fs)
            if psd_freq_f is not None and psd_filt is not None:
                state["psd_freq_filtered"] = psd_freq_f
                state["psd_filtered"] = psd_filt
        self._manager.update_state(**state)
    
    def set_raw_buffer(self, data: np.ndarray) -> None:
        self._raw_buffer = np.asarray(data, dtype=np.float64)
        self._broadcast()
    
    def set_filtered_buffer(self, data: np.ndarray) -> None:
        self._filtered_buffer = np.asarray(data, dtype=np.float64)
        self._broadcast()
    
    def update_accuracy(self, pipeline_name: str, accuracy: float) -> None:
        self._accuracy_history.append((pipeline_name, accuracy))
        self._pipeline_metrics[pipeline_name] = accuracy
        self._broadcast()
    
    def set_pipeline_metrics(self, metrics: dict[str, float]) -> None:
        self._pipeline_metrics = dict(metrics)
        self._broadcast()
    
    def set_prediction(self, class_idx: int) -> None:
        self._current_prediction = class_idx
        if 0 <= class_idx < len(self.class_names):
            self._current_prediction_name = self.class_names[class_idx]
        else:
            self._current_prediction_name = str(class_idx)
        self._broadcast()
    
    def set_best_pipeline(self, name: str) -> None:
        self._best_pipeline_name = name
        self._broadcast()
    
    def set_trial_progress(self, trial_index: int, n_trials: int) -> None:
        self._trial_index = trial_index
        self._n_trials_stream = n_trials
        self._broadcast()
    
    def set_dataset_source(self, source: str) -> None:
        self._dataset_source = source
        self._broadcast()

    def set_available_subjects(self, subject_ids: list[int | str]) -> None:
        """Set list of subject IDs available for the dataset (for metadata display)."""
        self._available_subjects = list(subject_ids) if subject_ids else []
        self._broadcast()

    def set_subject(self, subject_id: int | str | None) -> None:
        """Set current subject (e.g. 1 -> A01). Shown in UI."""
        if subject_id is None:
            self._subject = ""
        else:
            sid = int(subject_id) if isinstance(subject_id, str) and subject_id.isdigit() else subject_id
            self._subject = f"A{sid:02d}" if isinstance(sid, int) else str(subject_id)
        self._broadcast()
    
    def set_phase(self, phase: str) -> None:
        self._phase = phase
        self._broadcast()
    
    def set_rolling_accuracy(self, acc: float | None) -> None:
        self._rolling_accuracy = acc
        self._broadcast()

    def set_trial_labeled(self, is_labeled: bool) -> None:
        """Legacy: set trial source from label only. Prefer set_trial_source(from_t_session, has_label)."""
        self.set_trial_source(None, is_labeled)

    def set_trial_source(self, from_t_session: bool | None, has_label: bool) -> None:
        """Set current trial source: T session, E session (labeled or unlabeled)."""
        self._is_labeled = has_label
        if from_t_session is True:
            self._trial_source = "T (labeled)"
        elif from_t_session is False:
            self._trial_source = "E (labeled)" if has_label else "E (unlabeled, prediction only)"
        else:
            self._trial_source = "T (labeled)" if has_label else "E (unlabeled, prediction only)"
        self._broadcast()

    def record_live_trial_result(self, correct: bool) -> None:
        """Append one live trial outcome so accuracy-over-time chart can show cumulative accuracy."""
        self._live_correct.append(correct)
        self._broadcast()

    def _live_accuracy_history(self) -> list[float]:
        """Cumulative accuracy per live trial (one value per trial) for the chart."""
        if not self._live_correct:
            return []
        n = len(self._live_correct)
        cumsum = 0
        out: list[float] = []
        for i in range(n):
            cumsum += 1 if self._live_correct[i] else 0
            out.append(cumsum / (i + 1))
        return out[-200:]
    
    def set_calibration_metrics(self, metrics: dict) -> None:
        self._pipeline_metrics = {k: (v.get("accuracy", 0) if isinstance(v, dict) else v) for k, v in metrics.items()}
        self._broadcast()

    def set_calibration_metrics_full(self, metrics: dict[str, dict]) -> None:
        """Store full calibration metrics per pipeline (accuracy, kappa, latency_ms, stability) for detailed UI."""
        self._pipeline_metrics_detail = {k: dict(v) for k, v in metrics.items()}
        self._pipeline_metrics = {k: (v.get("accuracy", 0) if isinstance(v, dict) else 0) for k, v in metrics.items()}
        self._broadcast()
    
    def run(
        self,
        data_callback=None,
        on_close=None,
    ) -> None:
        """Web UI blocks to keep process alive; stream runs in another thread. Close with Ctrl+C."""
        if on_close:
            self._on_close = on_close
        self._broadcast()
        try:
            while True:
                threading.Event().wait(timeout=1.0)
        except KeyboardInterrupt:
            pass
        if getattr(self, "_on_close", None):
            self._on_close()
    
def run_server(manager: WebSocketManager, static_dir: Path, port: int = DEFAULT_WEB_PORT) -> None:
    """Run uvicorn in current thread (blocking). Try port, then port+1..port+10 if in use."""
    import uvicorn
    app = create_app(static_dir, manager)
    for attempt in range(11):
        try_port = port + attempt
        manager._port = try_port
        try:
            uvicorn.run(app, host="127.0.0.1", port=try_port, log_level="warning")
            return
        except OSError as e:
            if ("Address already in use" in str(e) or e.errno == 48) and attempt < 10:
                logger.warning("Port %s in use, trying %s...", try_port, try_port + 1)
                continue
            raise


def start_server_thread(manager: WebSocketManager, static_dir: Path, port: int = DEFAULT_WEB_PORT) -> threading.Thread:
    """Start uvicorn in a daemon thread."""
    def run():
        try:
            run_server(manager, static_dir, port)
        except OSError as e:
            if "Address already in use" in str(e) or e.errno == 48:
                logger.error("Port %s in use. Try another port or stop the process using it: %s", port, e)
            else:
                logger.exception("Web server failed: %s", e)
        except Exception as e:
            logger.exception("Web server failed: %s", e)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t
