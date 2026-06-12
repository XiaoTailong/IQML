from __future__ import annotations

import csv
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from iqml.data.synthetic import (
    make_henon_map,
    make_lorenz_system,
    make_mackey_glass,
    make_narma,
)
from iqml.data.windowing import DatasetSplits, make_supervised_windows, split_dataset
from iqml.models.iqml import IQMLConfig, init_iqml_params, iqml_forward
from iqml.models.lstm import LSTMConfig, init_lstm_params, lstm_forward, mse_loss as lstm_mse_loss
from iqml.models.params import count_parameters
from iqml.training.loop import (
    create_train_state,
    make_predict_fn,
    make_train_step_from_fns,
    train_epochs,
    train_epochs_with_step,
)
from iqml.training.metrics import regression_metrics


@dataclass(frozen=True)
class ExperimentResult:
    metrics: dict[str, float]
    history: list[dict[str, float]]
    output_dir: Path


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_experiment(config: dict[str, Any]) -> ExperimentResult:
    seed = int(config.get("seed", 0))
    data = _build_dataset(config["dataset"])
    model = dict(config["model"])
    model_type = model.pop("type", "iqml").lower()
    train_config = config.get("training", {})

    if model_type == "iqml":
        model_config = IQMLConfig(**_config_dataclass_kwargs(model, IQMLConfig))
        params = init_iqml_params(jax.random.PRNGKey(seed), model_config)
        state = create_train_state(params, float(train_config.get("learning_rate", 1e-3)))
        state, history = train_epochs(
            state,
            data.train.x,
            data.train.y,
            model_config,
            int(train_config.get("epochs", 5)),
            int(train_config.get("batch_size", 8)),
        )
        predict_fn = make_predict_fn(lambda p, x: iqml_forward(p, x, model_config).prediction)
        prediction = predict_fn(state.params, data.test.x)
    elif model_type == "lstm":
        model_config = LSTMConfig(**_config_dataclass_kwargs(model, LSTMConfig))
        params = init_lstm_params(jax.random.PRNGKey(seed), model_config)
        predict_fn = make_predict_fn(lambda p, x: lstm_forward(p, x, model_config).prediction)
        state = create_train_state(params, float(train_config.get("learning_rate", 1e-3)))
        compiled_step = make_train_step_from_fns(
            loss_fn=lambda p, x, y: lstm_mse_loss(p, x, y, model_config),
            predict_fn=predict_fn,
            optimizer=state.optimizer,
        )
        state, history = train_epochs_with_step(
            state,
            data.train.x,
            data.train.y,
            compiled_step,
            int(train_config.get("epochs", 5)),
            int(train_config.get("batch_size", 8)),
        )
        prediction = predict_fn(state.params, data.test.x)
    else:
        raise ValueError(f"Unsupported model type {model_type!r}. Use iqml or lstm.")

    metrics = regression_metrics(prediction, data.test.y)
    metrics["num_parameters"] = count_parameters(state.params)
    output_dir = _write_outputs(config, history, metrics, prediction, data.test.y)
    return ExperimentResult(metrics=metrics, history=history, output_dir=output_dir)


def _config_dataclass_kwargs(config: dict[str, Any], cls: type) -> dict[str, Any]:
    valid_fields = {field.name for field in fields(cls)}
    return {key: value for key, value in config.items() if key in valid_fields}


def _build_dataset(config: dict[str, Any]) -> DatasetSplits:
    name = config["name"].lower()
    length = int(config.get("length", 1000))
    if name == "mackey_glass":
        series = make_mackey_glass(
            length=length,
            tau=int(config.get("tau", 35)),
            beta=float(config.get("beta", 0.2)),
            gamma=float(config.get("gamma", 0.1)),
            n=int(config.get("n", 10)),
            seed=int(config.get("seed", 0)),
        )
    elif name == "henon":
        series = make_henon_map(
            length=length,
            a=float(config.get("a", 1.4)),
            b=float(config.get("b", 0.3)),
        )
    elif name == "lorenz":
        series = make_lorenz_system(
            length=length,
            rho=float(config.get("rho", 28.0)),
            sigma=float(config.get("sigma", 10.0)),
            beta=float(config.get("beta", 8.0 / 3.0)),
        )
    elif name == "narma":
        series = make_narma(
            length=length,
            order=int(config.get("order", 10)),
            seed=int(config.get("seed", 0)),
            input_scale=float(config.get("input_scale", 0.5)),
            alpha=float(config.get("alpha", 0.3)),
            beta=float(config.get("beta", 0.05)),
            gamma=float(config.get("gamma", 1.5)),
            delta=float(config.get("delta", 0.1)),
        )
    else:
        raise ValueError(
            f"Unsupported dataset {config['name']!r}. "
            "Use mackey_glass, henon, lorenz, or narma, or add a loader."
        )

    windows = make_supervised_windows(
        series,
        window_size=int(config.get("window_size", 8)),
        horizon=int(config.get("horizon", 1)),
        target_column=int(config.get("target_column", 0)),
        input_columns=config.get("input_columns"),
    )
    return split_dataset(
        windows,
        train_ratio=float(config.get("train_ratio", 0.6)),
        val_ratio=float(config.get("val_ratio", 0.2)),
    )


def _write_outputs(
    config: dict[str, Any],
    history: list[dict[str, float]],
    metrics: dict[str, float],
    prediction: jnp.ndarray,
    target: jnp.ndarray,
) -> Path:
    output_root = Path(config.get("output_root", "experiments"))
    experiment_name = config.get("experiment_name", "iqml_experiment")
    output_dir = output_root / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    pred = jnp.asarray(prediction)
    tgt = jnp.asarray(target)
    if pred.ndim == 1:
        pred = pred[:, None]
    if tgt.ndim == 1:
        tgt = tgt[:, None]
    if pred.shape != tgt.shape:
        raise ValueError(
            f"prediction and target must have the same shape, got {pred.shape} and {tgt.shape}"
        )

    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        columns = ["index"]
        for step in range(tgt.shape[1]):
            columns.append(f"target_t+{step + 1}")
        for step in range(pred.shape[1]):
            columns.append(f"prediction_t+{step + 1}")
        writer.writerow(columns)
        for idx in range(tgt.shape[0]):
            writer.writerow(
                [idx]
                + [float(value) for value in tgt[idx]]
                + [float(value) for value in pred[idx]]
            )
    return output_dir
