"""Snapshot logger: per-pipeline folders with raw/filtered EEG, features, accuracy, confusion matrix, JSON, model checkpoints."""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SnapshotLogger:
    """For every pipeline (selected or rejected): create results/pipeline_name/ and save plots + JSON."""

    def __init__(self, results_dir: str | Path = "./results", save_all_pipelines: bool = True) -> None:
        self.results_dir = Path(results_dir)
        self.save_all_pipelines = save_all_pipelines
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def pipeline_dir(self, pipeline_name: str) -> Path:
        safe = pipeline_name.replace("/", "_").replace(" ", "_")
        d = self.results_dir / safe
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_raw_eeg_plot(
        self,
        pipeline_name: str,
        data: np.ndarray,
        channel_names: list[str] | None = None,
        fs: float = 250.0,
        title: str = "Raw EEG",
    ) -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        d = self.pipeline_dir(pipeline_name)
        path = d / "raw_eeg.png"
        n_ch = min(data.shape[1], 8)
        ch_names = channel_names or [f"Ch{i}" for i in range(n_ch)]
        fig, axes = plt.subplots(n_ch, 1, figsize=(10, 1.5 * n_ch), sharex=True)
        if n_ch == 1:
            axes = [axes]
        t = np.arange(data.shape[2]) / fs
        for i in range(n_ch):
            axes[i].plot(t, data[0, i, :] if data.ndim == 3 else data[i, :])
            axes[i].set_ylabel(ch_names[i] if i < len(ch_names) else f"Ch{i}")
            axes[i].set_ylim(np.percentile(data, 1), np.percentile(data, 99))
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return path

    def save_filtered_eeg_plot(
        self,
        pipeline_name: str,
        data: np.ndarray,
        channel_names: list[str] | None = None,
        fs: float = 250.0,
        title: str = "Filtered EEG",
    ) -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        d = self.pipeline_dir(pipeline_name)
        path = d / "filtered_eeg.png"
        n_ch = min(data.shape[1], 8)
        ch_names = channel_names or [f"Ch{i}" for i in range(n_ch)]
        fig, axes = plt.subplots(n_ch, 1, figsize=(10, 1.5 * n_ch), sharex=True)
        if n_ch == 1:
            axes = [axes]
        t = np.arange(data.shape[2]) / fs
        for i in range(n_ch):
            axes[i].plot(t, data[0, i, :] if data.ndim == 3 else data[i, :])
            axes[i].set_ylabel(ch_names[i] if i < len(ch_names) else f"Ch{i}")
            axes[i].set_ylim(np.percentile(data, 1), np.percentile(data, 99))
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return path

    def save_feature_visualization(
        self,
        pipeline_name: str,
        features: np.ndarray,
        labels: np.ndarray | None = None,
        title: str = "Feature visualization",
    ) -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        d = self.pipeline_dir(pipeline_name)
        path = d / "features.png"
        fig, ax = plt.subplots(figsize=(8, 6))
        if features.shape[1] >= 2 and labels is not None:
            for c in np.unique(labels):
                mask = labels == c
                ax.scatter(features[mask, 0], features[mask, 1], label=f"Class {c}", alpha=0.6)
            ax.legend()
        else:
            ax.imshow(features[:50].T, aspect="auto")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return path

    def save_accuracy_curve(
        self,
        pipeline_name: str,
        accuracies: list[float],
        trials: list[int] | None = None,
        title: str = "Accuracy over time",
    ) -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        d = self.pipeline_dir(pipeline_name)
        path = d / "accuracy_curve.png"
        fig, ax = plt.subplots(figsize=(8, 4))
        x = trials or list(range(len(accuracies)))
        ax.plot(x, accuracies, marker="o", markersize=3)
        ax.set_xlabel("Trial / window")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return path

    def save_confusion_matrix(
        self,
        pipeline_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: list[str] | None = None,
    ) -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            from sklearn.metrics import confusion_matrix
            from sklearn.metrics import ConfusionMatrixDisplay
        except ImportError:
            return self.pipeline_dir(pipeline_name) / "confusion_matrix.png"
        d = self.pipeline_dir(pipeline_name)
        path = d / "confusion_matrix.png"
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        if len(y_true) != len(y_pred):
            y_pred = y_pred[: len(y_true)]
        cm = confusion_matrix(y_true, y_pred)
        labels = np.unique(np.concatenate([y_true, y_pred]))
        if class_names and len(class_names) >= len(labels):
            display_labels = [class_names[int(i)] for i in labels]
        else:
            display_labels = [str(int(i)) for i in labels]
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(ax=ax)
        fig.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return path

    def save_json_log(
        self,
        pipeline_name: str,
        metrics: dict[str, Any],
        selected: bool = False,
    ) -> Path:
        d = self.pipeline_dir(pipeline_name)
        path = d / "metrics.json"
        payload = {
            "pipeline_name": pipeline_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "selected": selected,
            "metrics": metrics,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path

    def save_pipeline_model(
        self,
        pipeline_name: str,
        pipeline: Any,
        experiment_id: str | None = None,
    ) -> Path:
        """
        Save pipeline (preprocessing + feature extractor + classifier) as .pkl.
        PyTorch models inside pipeline are saved via pickle; for standalone .pt use save_pipeline_torch.
        """
        d = self.pipeline_dir(pipeline_name)
        path = d / "model_checkpoint.pkl"
        meta = {"experiment_id": experiment_id, "pipeline_name": pipeline_name}
        with open(path, "wb") as f:
            pickle.dump({"pipeline": pipeline, "meta": meta}, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved pipeline model to %s", path)
        return path

    def load_pipeline_model(self, pipeline_name: str) -> tuple[Any, dict[str, Any]]:
        """Load pipeline from model_checkpoint.pkl. Returns (pipeline, meta)."""
        d = self.pipeline_dir(pipeline_name)
        path = d / "model_checkpoint.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint at {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data.get("pipeline"), data.get("meta", {})

    def save_pipeline_torch(self, pipeline_name: str, model: Any, experiment_id: str | None = None) -> Path:
        """Save PyTorch model state dict as .pt (for EEGNet etc.)."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required for .pt save")
        d = self.pipeline_dir(pipeline_name)
        path = d / "model_checkpoint.pt"
        state = {"model_state": model.state_dict() if hasattr(model, "state_dict") else model, "meta": {"experiment_id": experiment_id}}
        torch.save(state, path)
        logger.info("Saved torch model to %s", path)
        return path

    def online_dir(self) -> Path:
        """Directory for online mode: results/<experiment_id>/online/."""
        d = self.results_dir / "online"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_online_calibration_metrics(self, metrics: dict[str, dict[str, Any]], experiment_id: str | None = None) -> Path:
        """Save pipeline metrics from online calibration to online/pipeline_metrics.json."""
        d = self.online_dir()
        path = d / "pipeline_metrics.json"
        payload = {"experiment_id": experiment_id, "pipelines": metrics}
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Saved online calibration metrics to %s", path)
        return path

    def save_live_predictions_csv(
        self,
        rows: list[dict[str, Any]],
        path: Path | None = None,
    ) -> Path:
        """Save trial-by-trial predictions to online/live_predictions.csv."""
        import csv
        d = self.online_dir()
        out = path or d / "live_predictions.csv"
        if not rows:
            out.write_text("trial_index,true_label,predicted,correct\n")
            return out
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        logger.info("Saved live predictions to %s (%d rows)", out, len(rows))
        return out

    def save_selected_pipeline_online(self, pipeline: Any, experiment_id: str | None = None) -> Path:
        """Save selected pipeline for online mode to online/selected_pipeline.pkl."""
        d = self.online_dir()
        path = d / "selected_pipeline.pkl"
        with open(path, "wb") as f:
            pickle.dump({"pipeline": pipeline, "experiment_id": experiment_id}, f)
        logger.info("Saved selected pipeline to %s", path)
        return path
