"""
BCI AutoML Platform v1.0 — Main entry point.

Run from project root (EEG Agent):
  PYTHONPATH=. python main.py
  or: python -m bci_framework.main

Flow: Seed & experiment ID -> Load dataset -> Trial-wise split (no leakage)
      -> Calibration -> Prune (latency budget) -> Select best -> Model persistence
      -> Drift baseline -> Live simulation (streaming) -> GUI -> Snapshots & checkpoints.
"""

import argparse
import copy
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bci_framework.utils.config_loader import load_config, get_config
from bci_framework.utils.splits import get_train_test_trials
from bci_framework.utils.experiment import set_seed, get_experiment_id, set_experiment_id, log_experiment_params, enable_mlflow
from bci_framework.datasets import get_dataset_loader
from bci_framework.pipelines import PipelineRegistry
from bci_framework.agent import PipelineSelectionAgent, OnlinePipelineSelector
from bci_framework.logging import SnapshotLogger
from bci_framework.gui import BCIApp, WebApp, WebSocketManager, start_server_thread, DEFAULT_WEB_PORT


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


def _run_online_mode(
    config: dict,
    X,
    y,
    fs: float,
    n_classes: int,
    channel_names: list,
    class_names: list[str],
    dataset_source: str,
    subject_id: int | str | None,
    no_gui: bool,
    root: Path,
    use_web: bool = False,
    n_trials_from_t: int | None = None,
) -> None:
    """Online calibration + selection + live stream. First N trials calibrate; rest use selected pipeline."""
    import time as _time
    from bci_framework.utils.experiment import get_experiment_id

    config = copy.deepcopy(config)
    config["mode"] = "online"

    registry = PipelineRegistry(config)
    pipelines = registry.build_pipelines(fs=fs, n_classes=n_classes, channel_names=channel_names)
    selector = OnlinePipelineSelector(pipelines=pipelines, config=config, n_classes=n_classes)

    log_cfg = config.get("logging", {})
    results_base = root / (log_cfg.get("results_dir", "./results").lstrip("./"))
    results_dir = results_base / get_experiment_id() if log_cfg.get("versioned_experiments", False) else results_base
    results_dir.mkdir(parents=True, exist_ok=True)
    snapshot = SnapshotLogger(results_dir=results_dir, save_all_pipelines=True)
    online_dir = snapshot.online_dir()

    gui_cfg = config.get("gui", {})
    stream_cfg = config.get("streaming", {})
    trial_dur_sec = stream_cfg.get("trial_duration_sec", config.get("agent", {}).get("trial_duration_sec", 3.0))
    n_trials = len(X)
    real_time = stream_cfg.get("real_time_timing", True)
    n_cal = min(
        n_trials - 1,
        config.get("agent", {}).get("calibration_window_trials", 5),
    )
    n_stream_after_cal = max(0, n_trials - n_cal)
    logger.info(
        "Online mode: full subject stream — first %d trials = calibration (pipeline selection), remaining %d trials = live stream (total %d)",
        n_cal, n_stream_after_cal, n_trials,
    )

    app = None
    if not no_gui:
        if use_web:
            manager = WebSocketManager()
            static_dir = ROOT / "bci_framework" / "gui" / "static"
            web_port = gui_cfg.get("web_port", DEFAULT_WEB_PORT)
            start_server_thread(manager, static_dir, web_port)
            import time
            import webbrowser
            time.sleep(3)  # give server time to bind (or try next port if 8765 in use)
            actual_port = getattr(manager, "_port", web_port)
            url = "http://127.0.0.1:%s" % actual_port
            logger.info("Web UI: opening %s in your browser (keep this terminal open)", url)
            try:
                webbrowser.open(url)
            except Exception as e:
                logger.warning("Could not open browser: %s — open %s manually", e, url)
            app = WebApp(
                fs=fs,
                channel_names=channel_names,
                class_names=class_names,
                refresh_rate_ms=gui_cfg.get("refresh_rate_ms", 100),
                eeg_channels_display=len(channel_names),
                window_seconds=gui_cfg.get("window_seconds", 4.0),
                manager=manager,
            )
        else:
            app = BCIApp(
                fs=fs,
                channel_names=channel_names,
                class_names=class_names,
                refresh_rate_ms=gui_cfg.get("refresh_rate_ms", 100),
                eeg_channels_display=len(channel_names),
                window_seconds=gui_cfg.get("window_seconds", 4.0),
            )
        app.set_dataset_source(dataset_source)
        app.set_subject(subject_id)
        app.set_available_subjects(subjects)
        app.set_phase("Calibration")
        app.set_trial_progress(0, n_trials)

    def data_callback():
        return (app._raw_buffer if app else None, app._filtered_buffer if app else None)

    def run_stream():
        was_live = False
        selected_pipeline = None  # set when entering live phase
        for i in range(n_trials):
            X_i = X[i : i + 1]
            y_i = int(y[i])
            selector.add_trial(X_i, y_i)

            if selector.is_live_phase():
                selected_pipeline = selector.selected_pipeline
                if not was_live:
                    was_live = True
                    if selected_pipeline:
                        snapshot.save_online_calibration_metrics(
                            selector.get_calibration_metrics(), get_experiment_id()
                        )
                        snapshot.save_selected_pipeline_online(selected_pipeline, get_experiment_id())
                        # Save plots for all pipelines (rejected + selected) after calibration
                        for pipe in pipelines:
                            if not pipe._fitted:
                                continue
                            # Sample raw EEG from calibration buffer (first trial)
                            X_sample = selector._calibration_buffer[0][0] if selector._calibration_buffer else X[0:1]
                            snapshot.save_raw_eeg_plot(pipe.name, X_sample, channel_names=channel_names, fs=fs)
                            X_filt = pipe.preprocess(X_sample)
                            snapshot.save_filtered_eeg_plot(pipe.name, X_filt, channel_names=channel_names, fs=fs)
                        logger.info("Saved calibration plots for %d pipelines → %s", len([p for p in pipelines if p._fitted]), results_dir)
                        if app:
                            app.set_phase("Live")
                            app.set_best_pipeline(selected_pipeline.name)
                            app.set_calibration_metrics_full(selector.get_calibration_metrics())

                if selected_pipeline is None:
                    continue
                pred, proba = selector.predict(X_i)
                pred_label = int(pred[0])
                has_label = y_i >= 0
                from_t_session = (i < n_trials_from_t) if n_trials_from_t is not None else None
                correct = (pred_label == y_i) if has_label else None
                selector.update_drift(y_i if has_label else None, pred_label)
                proba_1 = proba[0] if proba is not None and len(proba) else None
                selector.append_live_prediction(
                    i, X_i, y_i if has_label else None, pred_label, proba_1, correct
                )

                if app:
                    app.set_raw_buffer(X_i[0])
                    X_filt = selected_pipeline.preprocess(X_i)
                    app.set_filtered_buffer(X_filt[0])
                    app.set_prediction(pred_label)
                    app.set_trial_progress(i, n_trials)
                    app.set_trial_source(from_t_session, has_label)
                    acc = selector.get_rolling_accuracy()
                    app.set_rolling_accuracy(acc)
                    if has_label:
                        app.record_live_trial_result(correct)

                if real_time:
                    _time.sleep(trial_dur_sec)
            else:
                if app:
                    app.set_trial_progress(i + 1, n_trials)
                    app.set_raw_buffer(X_i[0])

        # Save live_predictions.csv
        rows = []
        for r in selector.get_live_predictions():
            rows.append({
                "trial_index": r.trial_index,
                "true_label": r.label if r.label is not None else "",
                "predicted": r.predicted if r.predicted is not None else "",
                "correct": r.correct if r.correct is not None else "",
            })
        snapshot.save_live_predictions_csv(rows)
        logger.info("Online mode done: %d trials, selected pipeline %s", n_trials, selected_pipeline.name if selected_pipeline else "None")

    if not no_gui and app:
        import threading
        stream_thread = threading.Thread(target=run_stream, daemon=True)
        stream_thread.start()
        app.run(data_callback=data_callback)
    else:
        run_stream()


def main() -> None:
    parser = argparse.ArgumentParser(description="BCI Motor Imagery AutoML")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--no-gui", action="store_true", help="Skip GUI (calibration + logs only)")
    parser.add_argument("--web", action="store_true", help="Use web interface (zoomable EEG) instead of desktop GUI")
    parser.add_argument("--online", action="store_true", help="Online mode: first N trials calibration, rest live stream")
    parser.add_argument("--subject", type=int, default=1, help="Subject ID for single-subject run")
    parser.add_argument("--experiment-id", default=None, help="Experiment ID (auto if not set)")
    parser.add_argument("--loso", type=int, default=None, help="Leave-one-subject-out: holdout subject ID")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else ROOT / "bci_framework" / "config.yaml"
    load_config(config_path)
    config = get_config()

    # Experiment tracking & reproducibility
    exp_cfg = config.get("experiment", {})
    seed = exp_cfg.get("seed", 42)
    set_seed(seed)
    if args.experiment_id:
        set_experiment_id(args.experiment_id)
    elif exp_cfg.get("experiment_id"):
        set_experiment_id(str(exp_cfg["experiment_id"]))
    if exp_cfg.get("mlflow", {}).get("enabled"):
        enable_mlflow(
            tracking_uri=exp_cfg["mlflow"].get("tracking_uri"),
            experiment_name=exp_cfg["mlflow"].get("experiment_name", "bci_automl"),
        )
    log_experiment_params({"seed": seed, "experiment_id": get_experiment_id()})

    # 1) Load dataset
    dataset_cfg = config.get("dataset", {})
    ds_name = dataset_cfg.get("name", "BCI_IV_2a")
    data_dir = dataset_cfg.get("data_dir", "./data/BCI_IV_2a")
    download = dataset_cfg.get("download_if_missing", True)
    subjects = dataset_cfg.get("subjects", [1, 2, 3, 4, 5, 6, 7, 8, 9])
    if args.subject is not None:
        subjects = [args.subject]

    loader_cls = get_dataset_loader(ds_name)
    loader = loader_cls()
    data_path = ROOT / data_dir.lstrip("./")
    trial_sec = dataset_cfg.get("trial_duration_seconds", 3.0)
    result = loader.load(
        data_dir=str(data_path),
        subjects=subjects,
        download_if_missing=download,
        trial_duration_seconds=trial_sec,
    )

    subject_id = None
    if isinstance(result, dict):
        if not result:
            dataset = None
        else:
            subject_id = list(result.keys())[0]
            dataset = result[subject_id]
            logger.info("Using subject %s", subject_id)
    else:
        dataset = result
        subject_id = subjects[0] if subjects else None

    dataset_source = "unknown"
    if dataset is None or dataset.n_trials == 0:
        # Try to download BCI IV 2a if script exists
        download_script = ROOT / "scripts" / "download_bci_iv_2a.sh"
        if download_script.exists():
            import subprocess
            logger.info("No BCI IV 2a data found. Running download script: %s", download_script)
            try:
                subprocess.run(
                    ["bash", str(download_script)],
                    cwd=str(ROOT),
                    check=True,
                    timeout=600,
                )
                result = loader.load(
                    data_dir=str(data_path),
                    subjects=subjects,
                    download_if_missing=False,
                    trial_duration_seconds=dataset_cfg.get("trial_duration_seconds", 3.0),
                )
                if isinstance(result, dict) and result:
                    subject_id = list(result.keys())[0]
                    dataset = result[subject_id]
                    dataset_source = ds_name
                    logger.info("Loaded BCI IV 2a after download; subject %s", subject_id)
                elif result is not None and getattr(result, "n_trials", 0) > 0:
                    dataset = result
                    dataset_source = ds_name
                else:
                    dataset = None
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
                logger.warning("Download script failed: %s", e)
                dataset = None
        if dataset is None or dataset.n_trials == 0:
            logger.error(
                "No BCI IV 2a data found. Place GDF/mat files in: %s\n"
                "  Download: ./scripts/download_bci_iv_2a.sh\n"
                "  Or get zip from: https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip\n"
                "  Files needed: A01T.gdf, A01E.gdf, ... (or A01T.mat, A01E.mat).",
                data_path,
            )
            return
    else:
        dataset_source = ds_name if dataset is not None else "unknown"

    logger.info("Dataset in use: %s (%d trials)", dataset_source, len(dataset))
    X = dataset.data
    y = dataset.labels
    fs = dataset.fs
    n_classes = len(dataset.class_names)
    channel_names = dataset.channel_names
    class_names = dataset.class_names

    # ---- Online mode: first N trials calibration, rest live stream ----
    if args.online:
        n_trials_from_t = getattr(dataset, "n_trials_from_t", None)
        _run_online_mode(
            config=config,
            X=X,
            y=y,
            fs=fs,
            n_classes=n_classes,
            channel_names=channel_names,
            class_names=class_names,
            dataset_source=dataset_source,
            subject_id=subject_id,
            no_gui=args.no_gui,
            root=ROOT,
            use_web=args.web,
            n_trials_from_t=n_trials_from_t,
        )
        return

    subject_ids = getattr(dataset, "subject_ids_per_trial", None)
    ds_cfg = config.get("dataset", {})
    split_level = ds_cfg.get("split_level", "trial")
    loso_subject = args.loso or ds_cfg.get("loso_holdout_subject")
    n_trials_from_t = getattr(dataset, "n_trials_from_t", None)

    # Split: train_test (80/20) or sequential (first N = calibration, rest = live stream)
    split_mode = ds_cfg.get("split_mode", "train_test")
    n_cal = ds_cfg.get("stream_calibration_trials", 20)
    use_cross_session = ds_cfg.get("use_cross_session_split", False)  # T/E split for BCI IV 2a
    evaluation_mode = ds_cfg.get("evaluation_mode", "subject_wise")
    
    # METHODOLOGICAL WARNING: For BCI IV 2a, cross-session split (T/E) is recommended
    # to avoid mixing sessions and suspiciously high accuracy
    if evaluation_mode == "subject_wise" and not use_cross_session and n_trials_from_t is not None:
        logger.warning(
            "⚠️  METHODOLOGICAL WARNING: Using subject_wise split with shuffle=True "
            "MIXES T and E sessions. This can cause data leakage and suspiciously high accuracy. "
            "For BCI IV 2a, set use_cross_session_split: true in config.yaml "
            "or evaluation_mode: 'cross_session' to use T session (train) / E session (test) split."
        )
    
    train_idx, test_idx = get_train_test_trials(
        len(X),
        subject_ids=subject_ids,
        evaluation_mode=evaluation_mode,
        train_ratio=ds_cfg.get("train_test_split", 0.8),
        loso_subject=loso_subject,
        random_state=seed,
        split_mode=split_mode,
        n_calibration_trials=n_cal,
        n_trials_from_t=n_trials_from_t,
        use_cross_session=use_cross_session,
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    if split_mode == "sequential":
        logger.info(
            "Sequential split: first %d trials = calibration (pipeline selection), rest %d = live stream (no shuffle)",
            len(train_idx), len(test_idx),
        )
    else:
        logger.info("Trial-wise split: train %d, test %d (no leakage)", len(train_idx), len(test_idx))

    # v3.1: inject spatial capabilities (from loader/dataset) so registry and preprocessing resolve strict/auto
    caps = getattr(dataset, "capabilities", None) or getattr(loader, "capabilities", None)
    if caps is not None:
        config["spatial_capabilities"] = caps

    # 2) Build pipelines
    registry = PipelineRegistry(config)
    pipelines = registry.build_pipelines(fs=fs, n_classes=n_classes, channel_names=channel_names)
    logger.info("Built %d pipelines", len(pipelines))

    # 3) Calibration + Agent (pipeline selection: fit & evaluate on training trials; only labeled trials)
    import time as _time
    agent = PipelineSelectionAgent(config)
    labeled_train = y_train >= 0
    X_train_cal = X_train[labeled_train]
    y_train_cal = y_train[labeled_train]
    n_cal = min(len(X_train_cal), agent.calibration_trials)
    t0 = _time.perf_counter()
    metrics = agent.run_calibration(pipelines, X_train_cal[:n_cal], y_train_cal[:n_cal], n_classes, max_parallel=0)
    t_cal_sec = _time.perf_counter() - t0
    logger.info(
        "Pipeline selection: used %d labeled training trials (max %d), took %.1f s → best used for live stream + GUI",
        n_cal, agent.calibration_trials, t_cal_sec,
    )
    pipelines_kept = agent.prune(pipelines)
    agent.select_top_n(pipelines_kept)
    try:
        best_pipeline = agent.select_best(pipelines)  # v3.2: argmax CV over all valid
    except RuntimeError:
        best_pipeline = None

    # 4) Snapshot logging for all pipelines (selected and rejected)
    log_cfg = config.get("logging", {})
    results_base = ROOT / (log_cfg.get("results_dir", "./results").lstrip("./"))
    results_dir = results_base / get_experiment_id() if log_cfg.get("versioned_experiments", False) else results_base
    results_dir.mkdir(parents=True, exist_ok=True)
    snapshot = SnapshotLogger(results_dir=results_dir, save_all_pipelines=log_cfg.get("save_all_pipelines", True))

    for pipe in pipelines:
        if not pipe._fitted:
            continue
        pipe_dir = snapshot.pipeline_dir(pipe.name)
        # Raw EEG sample
        snapshot.save_raw_eeg_plot(pipe.name, X_train[:1], channel_names=channel_names, fs=fs)
        X_filt = pipe.preprocess(X_train[:1])
        snapshot.save_filtered_eeg_plot(pipe.name, X_filt, channel_names=channel_names, fs=fs)
        feat = pipe.transform(X_train[:min(100, len(X_train))])
        snapshot.save_feature_visualization(pipe.name, feat, y_train[:feat.shape[0]])
        pred = pipe.predict(X_test)
        m = metrics.get(pipe.name)
        if m:
            snapshot.save_accuracy_curve(pipe.name, m.accuracies_over_time or [m.accuracy])
            snapshot.save_confusion_matrix(pipe.name, y_test, pred, class_names=dataset.class_names)
            meta = agent.get_metrics_dict().get(pipe.name, {})
            mand = getattr(pipe.preprocessing_manager, "mandatory", None)
            spatial_req = getattr(mand, "spatial_filter_requested", None)
            spatial_used = getattr(mand, "spatial_filter_used", None)
            cap_dict = None
            if config.get("spatial_capabilities") is not None:
                c = config["spatial_capabilities"]
                cap_dict = c.to_dict() if hasattr(c, "to_dict") else c
            transfer_info = None
            if config.get("transfer", {}).get("enabled"):
                tc = config["transfer"]
                transfer_info = {
                    "method": tc.get("method", "none"),
                    "regularization": tc.get("regularization"),
                    "safe_mode": tc.get("safe_mode"),
                }
            best_name = best_pipeline.name if best_pipeline is not None else None
            adaptive = agent.get_adaptive_pruning_info() if hasattr(agent, "get_adaptive_pruning_info") else {}
            snapshot.save_json_log(
                pipe.name,
                meta,
                selected=(best_pipeline is not None and pipe.name == best_pipeline.name),
                spatial_capabilities=cap_dict,
                spatial_filter_requested=spatial_req,
                spatial_filter_used=spatial_used,
                transfer=transfer_info,
                best_pipeline=best_name,
                quick_screening=adaptive.get("quick_screening"),
                early_stopped_pipelines=adaptive.get("early_stopped_pipelines"),
                progressive_halving_used=adaptive.get("progressive_halving_used"),
                adaptive_pruning=adaptive.get("adaptive_pruning"),
            )
        if log_cfg.get("save_model_checkpoints", False) and pipe._fitted:
            try:
                snapshot.save_pipeline_model(pipe.name, pipe, get_experiment_id())
            except Exception as e:
                logger.debug("Save checkpoint %s: %s", pipe.name, e)

    n_snapshots = sum(1 for p in pipelines if getattr(p, "_fitted", False))
    logger.info("Snapshot logging done: %d pipelines → %s (plots + JSON + checkpoints)", n_snapshots, results_dir)

    # 5) Live simulation with best pipeline
    if best_pipeline is None:
        logger.warning("No best pipeline selected; using first fitted pipeline")
        best_pipeline = next((p for p in pipelines if p._fitted), None)
    if best_pipeline is None:
        logger.error("No pipeline fitted. Exiting.")
        return

    # Save best pipeline checkpoint
    if log_cfg.get("save_model_checkpoints", True):
        try:
            snapshot.save_pipeline_model(best_pipeline.name, best_pipeline, get_experiment_id())
            logger.info("Saved best pipeline checkpoint: %s", best_pipeline.name)
        except Exception as e:
            logger.warning("Save best pipeline checkpoint: %s", e)

    # Drift detection baseline (for continuous adaptation)
    best_m = agent.get_metrics().get(best_pipeline.name)
    if best_m:
        agent.set_drift_baseline(best_m.accuracy)

    gui_cfg = config.get("gui", {})
    if args.web:
        manager = WebSocketManager()
        static_dir = ROOT / "bci_framework" / "gui" / "static"
        web_port = gui_cfg.get("web_port", DEFAULT_WEB_PORT)
        start_server_thread(manager, static_dir, web_port)
        import time
        time.sleep(3)  # give server time to bind (or try next port if in use)
        app = WebApp(
            fs=fs,
            channel_names=channel_names,
            class_names=dataset.class_names,
            refresh_rate_ms=gui_cfg.get("refresh_rate_ms", 100),
            eeg_channels_display=len(channel_names),
            window_seconds=gui_cfg.get("window_seconds", 4.0),
            manager=manager,
        )
    else:
        app = BCIApp(
            fs=fs,
            channel_names=channel_names,
            class_names=dataset.class_names,
            refresh_rate_ms=gui_cfg.get("refresh_rate_ms", 100),
            eeg_channels_display=len(channel_names),
            window_seconds=gui_cfg.get("window_seconds", 4.0),
        )
    app.set_pipeline_metrics({k: v["accuracy"] for k, v in agent.get_metrics_dict().items()})
    for name, m in agent.get_metrics().items():
        app.update_accuracy(name, m.accuracy)
    app.set_best_pipeline(best_pipeline.name)
    app.set_dataset_source(dataset_source)
    app.set_available_subjects(config.get("dataset", {}).get("subjects", []))

    # Real-time streaming: run full test set through best pipeline with real-time pacing
    import numpy as np
    import time

    stream_cfg = config.get("streaming", {})
    dataset_cfg = config.get("dataset", {})
    streaming_mode = dataset_cfg.get("streaming_mode", "trial") or stream_cfg.get("mode", "trial")
    stream_full = stream_cfg.get("stream_full_test_set", True)
    real_time = stream_cfg.get("real_time_timing", True)
    trial_dur_sec = stream_cfg.get("trial_duration_sec", config.get("agent", {}).get("trial_duration_sec", 3.0))
    n_stream = len(X_test) if stream_full else min(50, len(X_test))
    app.set_trial_progress(0, n_stream)

    # Check if sliding-window mode is enabled
    if streaming_mode == "sliding":
        logger.info(
            "Sliding-window real-time streaming: %d trials through best pipeline %s",
            n_stream, best_pipeline.name,
        )
        # Import streaming modules
        from bci_framework.streaming import RealtimeInferenceEngine
        
        # Configure sliding window parameters
        window_size_sec = stream_cfg.get("window_size_sec", 1.5)
        update_interval_sec = stream_cfg.get("update_interval_sec", 0.1)
        buffer_length_sec = stream_cfg.get("buffer_length_sec", 10.0)
        
        # Ensure pipeline uses online/causal preprocessing
        # Update config and preprocessing manager mode for causal filters
        config_stream = copy.deepcopy(config)
        config_stream["mode"] = "online"
        # Enable GEDAI sliding mode if GEDAI is enabled
        if "gedai" in config_stream.get("advanced_preprocessing", {}).get("enabled", []):
            gedai_cfg = config_stream.get("advanced_preprocessing", {}).get("gedai", {})
            if gedai_cfg.get("mode", "batch") != "sliding":
                logger.info("Enabling GEDAI sliding mode for real-time streaming")
                gedai_cfg["mode"] = "sliding"
        best_pipeline.preprocessing_manager.config = config_stream
        best_pipeline.preprocessing_manager.mode = "online"
        # Enable causal filters in mandatory preprocessing steps
        if hasattr(best_pipeline.preprocessing_manager.mandatory, "notch"):
            best_pipeline.preprocessing_manager.mandatory.notch.causal = True
        if hasattr(best_pipeline.preprocessing_manager.mandatory, "bandpass"):
            best_pipeline.preprocessing_manager.mandatory.bandpass.causal = True
        # Update GEDAI mode if present in advanced steps
        for name, step in best_pipeline.preprocessing_manager.advanced_steps:
            if name == "gedai" and hasattr(step, "mode"):
                if step.mode != "sliding":
                    logger.info("Switching GEDAI to sliding mode for real-time streaming")
                    step.mode = "sliding"
                    step.supports_online = True
        
        # Initialize inference engine
        inference_engine = RealtimeInferenceEngine(
            pipeline=best_pipeline,
            fs=fs,
            window_size_sec=window_size_sec,
            update_interval_sec=update_interval_sec,
            buffer_length_sec=buffer_length_sec,
            n_channels=len(channel_names),
        )
        
        def data_callback():
            return (app._raw_buffer, app._filtered_buffer)
        
        def run_sliding_streaming():
            """Stream trials sample-by-sample with sliding window inference."""
            n_samples_per_trial = int(trial_dur_sec * fs)
            sample_interval = 1.0 / fs  # Time between samples
            
            for trial_idx in range(n_stream):
                trial_data = X_test[trial_idx : trial_idx + 1]  # (1, n_channels, n_samples)
                trial_label = y_test[trial_idx] if trial_idx < len(y_test) else -1
                
                # Convert to (n_channels, n_samples) for streaming
                trial_channel_data = trial_data[0]  # (n_channels, n_samples)
                
                # Stream samples one by one
                for sample_idx in range(n_samples_per_trial):
                    if sample_idx >= trial_channel_data.shape[1]:
                        break
                    
                    # Push single sample (n_channels,)
                    sample = trial_channel_data[:, sample_idx]
                    inference_engine.push_samples(sample)
                    
                    # Check if inference should run
                    result = inference_engine.update()
                    if result is not None:
                        prediction, latency_ms = result
                        pred_label = int(prediction[0])
                        
                        # Update GUI
                        if app:
                            # Get current window for display
                            window = inference_engine._buffer.get_window(window_size_sec)
                            if window is not None:
                                window_trial = window[np.newaxis, :, :]
                                app.set_raw_buffer(window_trial[0])
                                X_filt = best_pipeline.preprocess(window_trial)
                                app.set_filtered_buffer(X_filt[0])
                                app.set_prediction(pred_label)
                                app.set_trial_progress(trial_idx, n_stream)
                    
                    # Real-time pacing: sleep for sample interval
                    if real_time:
                        time.sleep(sample_interval)
                
                # Log latency stats periodically
                if (trial_idx + 1) % 10 == 0:
                    stats = inference_engine.get_latency_stats()
                    logger.info(
                        "Trial %d/%d: latency mean=%.2fms, median=%.2fms, max=%.2fms",
                        trial_idx + 1, n_stream,
                        stats["mean_ms"], stats["median_ms"], stats["max_ms"],
                    )
            
            # Final latency stats
            stats = inference_engine.get_latency_stats()
            logger.info(
                "Sliding-window streaming finished: %d trials, %d predictions. "
                "Latency: mean=%.2fms, median=%.2fms, max=%.2fms, min=%.2fms",
                n_stream, stats["n_predictions"],
                stats["mean_ms"], stats["median_ms"], stats["max_ms"], stats["min_ms"],
            )
        
        # Seed GUI with first window
        if not args.no_gui and n_stream > 0:
            first_trial = X_test[0:1]
            app.set_raw_buffer(first_trial[0])
            X_filt_first = best_pipeline.preprocess(first_trial)
            app.set_filtered_buffer(X_filt_first[0])
            app.set_prediction(int(best_pipeline.predict(first_trial)[0]))
            app.set_trial_progress(0, n_stream)
        
        # Open browser after seed so UI receives EEG data on first load
        if args.web and not args.no_gui:
            import webbrowser
            actual_port = getattr(getattr(app, "_manager", None), "_port", None) or gui_cfg.get("web_port", DEFAULT_WEB_PORT)
            url = "http://127.0.0.1:%s" % actual_port
            logger.info("Web UI: opening %s in your browser (keep this terminal open)", url)
            try:
                webbrowser.open(url)
            except Exception as e:
                logger.warning("Could not open browser: %s — open %s manually", e, url)
        
        if not args.no_gui:
            logger.info(
                "Opening GUI for sliding-window stream with best pipeline: %s (%d test trials). Close window to exit.",
                best_pipeline.name, n_stream,
            )
            import threading
            sim_thread = threading.Thread(target=run_sliding_streaming, daemon=True)
            sim_thread.start()
            app.run(data_callback=data_callback)
        else:
            # Headless: run sliding streaming
            run_sliding_streaming()
        
        return
    
    # Original trial-based streaming mode
    logger.info(
        "Real-time streaming: %d trials through best pipeline %s (real_time=%s, %.1fs per trial)",
        n_stream, best_pipeline.name, real_time, trial_dur_sec,
    )

    def data_callback():
        return (app._raw_buffer, app._filtered_buffer)

    def run_realtime_streaming():
        for idx in range(n_stream):
            app.set_trial_progress(idx, n_stream)
            chunk = X_test[idx : idx + 1]
            if chunk.size == 0:
                break
            app.set_raw_buffer(chunk[0])
            X_filt = best_pipeline.preprocess(chunk)
            app.set_filtered_buffer(X_filt[0])
            pred = best_pipeline.predict(chunk)
            app.set_prediction(int(pred[0]))
            if real_time:
                time.sleep(trial_dur_sec)
        logger.info("Real-time streaming finished: %d trials processed", n_stream)

    # Seed GUI with first chunk so raw/filtered plots show data as soon as window opens
    if not args.no_gui and n_stream > 0:
        first_chunk = X_test[0:1]
        app.set_raw_buffer(first_chunk[0])
        X_filt_first = best_pipeline.preprocess(first_chunk)
        app.set_filtered_buffer(X_filt_first[0])
        app.set_prediction(int(best_pipeline.predict(first_chunk)[0]))
        app.set_trial_progress(0, n_stream)

    # Open browser only after seed so UI receives EEG data on first load
    if args.web and not args.no_gui:
        import webbrowser
        actual_port = getattr(getattr(app, "_manager", None), "_port", None) or gui_cfg.get("web_port", DEFAULT_WEB_PORT)
        url = "http://127.0.0.1:%s" % actual_port
        logger.info("Web UI: opening %s in your browser (keep this terminal open)", url)
        try:
            webbrowser.open(url)
        except Exception as e:
            logger.warning("Could not open browser: %s — open %s manually", e, url)

    if not args.no_gui:
        logger.info(
            "Opening GUI for live stream with best pipeline: %s (%d test trials, ~%.0f s real-time). Close window to exit.",
            best_pipeline.name, n_stream, n_stream * trial_dur_sec,
        )
        import threading
        sim_thread = threading.Thread(target=run_realtime_streaming, daemon=True)
        sim_thread.start()
        app.run(data_callback=data_callback)
    else:
        # Headless: stream full test set with real-time timing and log each prediction
        for idx in range(n_stream):
            chunk = X_test[idx : idx + 1]
            pred = best_pipeline.predict(chunk)
            logger.info("Trial %d/%d prediction: %s", idx + 1, n_stream, dataset.class_names[int(pred[0])])
            if real_time:
                time.sleep(trial_dur_sec)
        logger.info("Done. Best pipeline: %s. Streamed %d trials. Results in %s", best_pipeline.name, n_stream, results_dir)


if __name__ == "__main__":
    main()
