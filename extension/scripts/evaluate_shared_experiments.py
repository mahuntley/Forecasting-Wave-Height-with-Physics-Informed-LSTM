"""Run the shared LSTM comparison experiments."""

import argparse
import json
import os
import pickle
import random
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr
from sklearn import preprocessing


GRAVITY = 9.80665
STEEPNESS_LIMIT = 0.055

# Small grid so the regularizer tuning stays manageable.
PINN_WEIGHT_GRID = [
    (0.02, 0.00, 0.01),
    (0.05, 0.01, 0.01),
    (0.10, 0.02, 0.01),
    (0.20, 0.05, 0.01),
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_point(dataset_path, lat, lon):
    dataset = xr.open_dataset(dataset_path)
    point = dataset.sel(latitude=lat, longitude=lon, method="nearest")

    # Use vector components so the 0/360 degree wraparound is not a problem.
    wind_direction = np.deg2rad(point.dwi.values)
    wind = point.wind.values
    wind_x = -wind * np.sin(wind_direction)
    wind_y = -wind * np.cos(wind_direction)

    data = pd.DataFrame(
        {
            "time": pd.to_datetime(dataset.time.values),
            "Hs": point.swh.values,
            "Period": point.pp1d.values,
            "wind_x": wind_x,
            "wind_y": wind_y,
        }
    )
    data = data.set_index("time").dropna()
    print(f"Loaded {len(data)} usable rows from {dataset_path}")
    print("Using multivariate inputs: Hs, Period, wind_x, wind_y")
    return data


def make_samples(data, columns, history_hours, lead_hours):
    values = data[columns].to_numpy(dtype=np.float32)
    hs = data["Hs"].to_numpy(dtype=np.float32)
    times = data.index.to_numpy()

    X, y, target_times, latest_hs = [], [], [], []
    last_start = len(data) - history_hours - lead_hours + 1

    for start in range(last_start):
        end = start + history_hours
        target = end + lead_hours - 1

        X.append(values[start:end])
        y.append(hs[target])
        target_times.append(times[target])
        latest_hs.append(hs[end - 1])

    info = pd.DataFrame({"Data": target_times, "latest_hs": latest_hs})
    return np.asarray(X), np.asarray(y), info


def train_test_split(X, y, info, test_size):
    split_at = len(y) - test_size
    if split_at <= 0:
        raise ValueError("The test set is larger than the available samples.")

    return (
        X[:split_at],
        X[split_at:],
        y[:split_at],
        y[split_at:],
        info.iloc[:split_at].reset_index(drop=True),
        info.iloc[split_at:].reset_index(drop=True),
    )


def scale_data(X_train_raw, X_test_raw, y_train_raw):
    x_scaler = preprocessing.MinMaxScaler((0, 1))
    y_scaler = preprocessing.MinMaxScaler((0, 1))

    train_rows, history_hours, feature_count = X_train_raw.shape
    test_rows = X_test_raw.shape[0]

    X_train = x_scaler.fit_transform(X_train_raw.reshape(-1, feature_count))
    X_test = x_scaler.transform(X_test_raw.reshape(-1, feature_count))

    X_train = X_train.reshape(train_rows, history_hours, feature_count)
    X_test = X_test.reshape(test_rows, history_hours, feature_count)
    y_train = y_scaler.fit_transform(y_train_raw.reshape(-1, 1))[:, 0]

    return X_train, X_test, y_train, x_scaler, y_scaler


def make_model(history_hours, feature_count):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(history_hours, feature_count)),
            tf.keras.layers.LSTM(64, activation="relu", return_sequences=True),
            tf.keras.layers.LSTM(48, activation="relu", return_sequences=True),
            tf.keras.layers.LSTM(32, activation="relu"),
            tf.keras.layers.Dense(1, kernel_initializer=tf.initializers.zeros()),
        ]
    )


def make_pinn_targets(y_train, X_train_raw, y_scaler, lead_hours):
    latest_hs = X_train_raw[:, -1, 0]
    latest_period = np.maximum(X_train_raw[:, -1, 1], 0.1)

    # Deep-water wavelength estimate: L = gT^2 / 2pi.
    wavelength = GRAVITY * latest_period**2 / (2.0 * np.pi)
    max_hs = STEEPNESS_LIMIT * wavelength

    max_hs_norm = y_scaler.transform(max_hs.reshape(-1, 1))[:, 0]
    latest_hs_norm = y_scaler.transform(latest_hs.reshape(-1, 1))[:, 0]

    # Simple allowance for how much Hs can change over the lead time.
    jump_limit_norm = np.full_like(y_train, 0.10 * lead_hours * y_scaler.scale_[0])

    return np.column_stack((y_train, max_hs_norm, latest_hs_norm, jump_limit_norm))


def make_pinn_loss(steepness_weight, continuity_weight, bounds_weight):
    def loss(y_true, y_pred):
        target = y_true[:, 0:1]
        max_hs = y_true[:, 1:2]
        latest_hs = y_true[:, 2:3]
        jump_limit = y_true[:, 3:4]

        data_error = tf.reduce_mean(tf.abs(target - y_pred))
        steepness_error = tf.reduce_mean(tf.square(tf.nn.relu(y_pred - max_hs)))
        jump_error = tf.reduce_mean(tf.square(tf.nn.relu(tf.abs(y_pred - latest_hs) - jump_limit)))
        range_error = tf.reduce_mean(tf.square(tf.nn.relu(-y_pred)) + tf.square(tf.nn.relu(y_pred - 1.0)))

        return (
            data_error
            + steepness_weight * steepness_error
            + continuity_weight * jump_error
            + bounds_weight * range_error
        )

    return loss


def clean_name(value):
    return str(value).replace(".", "p").replace("-", "m")


def save_model_artifacts(model, model_path, x_scaler, y_scaler, metadata):
    model.save(model_path)

    x_scaler_path = model_path.with_suffix(".x_scaler.pkl")
    y_scaler_path = model_path.with_suffix(".y_scaler.pkl")
    metadata_path = model_path.with_suffix(".metadata.json")

    with x_scaler_path.open("wb") as file:
        pickle.dump(x_scaler, file)
    with y_scaler_path.open("wb") as file:
        pickle.dump(y_scaler, file)

    metadata = {
        **metadata,
        "model_file": model_path.name,
        "x_scaler_file": x_scaler_path.name,
        "y_scaler_file": y_scaler_path.name,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))


def train_and_predict(
    seed,
    X_train,
    y_train,
    X_test,
    x_scaler,
    y_scaler,
    epochs,
    loss,
    patience,
    model_path,
    metadata,
    save_models,
):
    set_seed(seed)
    tf.keras.backend.clear_session()

    model = make_model(X_train.shape[1], X_train.shape[2])
    model.compile(loss=loss, optimizer="adam")

    # Early stopping is mostly useful for longer runs.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
        restore_best_weights=True,
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0,
    )
    if save_models:
        save_model_artifacts(model, model_path, x_scaler, y_scaler, metadata)

    prediction = model.predict(X_test, verbose=0)[:, 0]
    prediction = y_scaler.inverse_transform(prediction.reshape(-1, 1))[:, 0]
    return prediction, history.history


def add_history_rows(rows, model_name, lead, seed, repeat, weights, history):
    for epoch, loss_value in enumerate(history["loss"], start=1):
        rows.append(
            {
                "model": model_name,
                "lead": lead,
                "seed": seed,
                "repeat": repeat,
                "steepness_weight": weights[0] if weights else np.nan,
                "continuity_weight": weights[1] if weights else np.nan,
                "bounds_weight": weights[2] if weights else np.nan,
                "epoch": epoch,
                "loss": loss_value,
                "val_loss": history["val_loss"][epoch - 1],
            }
        )


def summarize_histories(histories):
    groups = ["model", "lead", "seed", "repeat", "steepness_weight", "continuity_weight", "bounds_weight"]
    rows = []

    for group_values, group in histories.groupby(groups, dropna=False):
        best_row = group.loc[group["val_loss"].idxmin()]
        final_row = group.loc[group["epoch"].idxmax()]
        row = dict(zip(groups, group_values))
        row.update(
            {
                "epochs_trained": int(final_row["epoch"]),
                "best_epoch": int(best_row["epoch"]),
                "best_val_loss": best_row["val_loss"],
                "final_loss": final_row["loss"],
                "final_val_loss": final_row["val_loss"],
            }
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["lead", "model", "seed", "repeat"])


def score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    keep = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0)
    error = y_pred[keep] - y_true[keep]

    return {
        "valid": int(keep.sum()),
        "mape": np.mean(np.abs(error / y_true[keep])) * 100.0,
        "mae": np.mean(np.abs(error)),
        "rmse": np.sqrt(np.mean(error**2)),
        "bias": np.mean(error),
    }


def add_result(rows, model_name, lead, seed, repeat, weights, y_true, y_pred):
    row = {
        "model": model_name,
        "lead": lead,
        "seed": seed,
        "repeat": repeat,
        "steepness_weight": weights[0] if weights else np.nan,
        "continuity_weight": weights[1] if weights else np.nan,
        "bounds_weight": weights[2] if weights else np.nan,
    }
    row.update(score(y_true, y_pred))
    rows.append(row)


def summarize(results):
    metric_columns = ["mape", "mae", "rmse", "bias"]
    groups = ["model", "lead", "steepness_weight", "continuity_weight", "bounds_weight"]
    summary = results.groupby(groups, dropna=False)[metric_columns].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    return summary.sort_values(["lead", "mape_mean", "rmse_mean"])


def run(args):
    start_time = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    if not args.no_save_models:
        model_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        **vars(args),
        "pinn_weight_grid": PINN_WEIGHT_GRID,
        "hs_columns": ["Hs"],
        "multivariate_columns": ["Hs", "Period", "wind_x", "wind_y"],
        "model_directory": str(model_dir),
        "save_models": not args.no_save_models,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    data = load_point(args.dataset, args.lat, args.lon)
    rows = []
    history_rows = []

    for lead in args.lead_times:
        print(f"Lead time: {lead}h")

        X_multi_raw, y_raw, info = make_samples(data, ["Hs", "Period", "wind_x", "wind_y"], args.history_hours, lead)
        X_hs_raw = X_multi_raw[:, :, 0:1]

        multi_split = train_test_split(X_multi_raw, y_raw, info, args.test_size)
        hs_split = train_test_split(X_hs_raw, y_raw, info, args.test_size)

        X_multi_train_raw, X_multi_test_raw, y_train_raw, y_test_raw, _, test_info = multi_split
        X_hs_train_raw, X_hs_test_raw, _, _, _, _ = hs_split

        # Persistence predicts the latest observed Hs.
        persistence = test_info["latest_hs"].to_numpy(dtype=float)
        add_result(rows, "persistence", lead, args.base_seed, 0, None, y_test_raw, persistence)

        X_hs_train, X_hs_test, y_hs_train, hs_x_scaler, hs_y_scaler = scale_data(
            X_hs_train_raw, X_hs_test_raw, y_train_raw
        )
        X_multi_train, X_multi_test, y_multi_train, multi_x_scaler, multi_y_scaler = scale_data(
            X_multi_train_raw, X_multi_test_raw, y_train_raw
        )

        for repeat in range(args.repeats):
            seed = args.base_seed + repeat
            print(f"  repeat {repeat + 1}/{args.repeats}, seed {seed}")

            hs_pred, hs_history = train_and_predict(
                seed,
                X_hs_train,
                y_hs_train,
                X_hs_test,
                hs_x_scaler,
                hs_y_scaler,
                args.epochs,
                "mean_absolute_error",
                args.patience,
                model_dir / f"hs_only_lead{lead}_seed{seed}.keras",
                {
                    "model": "hs_only",
                    "lead": lead,
                    "seed": seed,
                    "repeat": repeat,
                    "history_hours": args.history_hours,
                    "feature_columns": ["Hs"],
                    "target": "Hs",
                },
                not args.no_save_models,
            )
            add_result(rows, "hs_only", lead, seed, repeat, None, y_test_raw, hs_pred)
            add_history_rows(history_rows, "hs_only", lead, seed, repeat, None, hs_history)

            multi_pred, multi_history = train_and_predict(
                seed,
                X_multi_train,
                y_multi_train,
                X_multi_test,
                multi_x_scaler,
                multi_y_scaler,
                args.epochs,
                "mean_absolute_error",
                args.patience,
                model_dir / f"multi_wind_lead{lead}_seed{seed}.keras",
                {
                    "model": "multi_wind",
                    "lead": lead,
                    "seed": seed,
                    "repeat": repeat,
                    "history_hours": args.history_hours,
                    "feature_columns": ["Hs", "Period", "wind_x", "wind_y"],
                    "target": "Hs",
                },
                not args.no_save_models,
            )
            add_result(rows, "multi_wind", lead, seed, repeat, None, y_test_raw, multi_pred)
            add_history_rows(history_rows, "multi_wind", lead, seed, repeat, None, multi_history)

            for weights in PINN_WEIGHT_GRID:
                pinn_y_train = make_pinn_targets(y_multi_train, X_multi_train_raw, multi_y_scaler, lead)
                weight_name = "_".join(clean_name(weight) for weight in weights)
                pinn_pred, pinn_history = train_and_predict(
                    seed,
                    X_multi_train,
                    pinn_y_train,
                    X_multi_test,
                    multi_x_scaler,
                    multi_y_scaler,
                    args.epochs,
                    make_pinn_loss(*weights),
                    args.patience,
                    model_dir / f"pinn_wind_lead{lead}_seed{seed}_weights{weight_name}.keras",
                    {
                        "model": "pinn_wind",
                        "lead": lead,
                        "seed": seed,
                        "repeat": repeat,
                        "history_hours": args.history_hours,
                        "feature_columns": ["Hs", "Period", "wind_x", "wind_y"],
                        "target": "Hs",
                        "steepness_weight": weights[0],
                        "continuity_weight": weights[1],
                        "bounds_weight": weights[2],
                    },
                    not args.no_save_models,
                )
                add_result(rows, "pinn_wind", lead, seed, repeat, weights, y_test_raw, pinn_pred)
                add_history_rows(history_rows, "pinn_wind", lead, seed, repeat, weights, pinn_history)

        pd.DataFrame(
            {
                "Data": test_info["Data"],
                "Hs Reanalysis Value": y_test_raw,
                "Persistence Predict Value": persistence,
            }
        ).to_csv(output_dir / f"shared_eval_persistence_predictions_{lead}_leadtime.csv", index=False)

    runs = pd.DataFrame(rows)
    histories = pd.DataFrame(history_rows)
    epoch_summary = summarize_histories(histories)
    summary = summarize(runs)
    best_pinn = (
        summary[summary["model"].eq("pinn_wind")]
        .sort_values(["lead", "mape_mean", "rmse_mean"])
        .groupby("lead")
        .head(1)
    )

    runs.to_csv(output_dir / "shared_eval_runs.csv", index=False)
    histories.to_csv(output_dir / "shared_eval_training_history.csv", index=False)
    epoch_summary.to_csv(output_dir / "shared_eval_epoch_summary.csv", index=False)
    summary.to_csv(output_dir / "shared_eval_summary.csv", index=False)
    best_pinn.to_csv(output_dir / "shared_eval_best_pinn_by_lead.csv", index=False)

    print(f"Wrote results to {output_dir}")
    print(f"Finished in {(time.time() - start_time) / 60.0:.2f} minutes")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate our ERA5 LSTM extension models.")
    parser.add_argument("dataset")
    parser.add_argument("--lat", type=float, default=-31.5)
    parser.add_argument("--lon", type=float, default=-50.0)
    parser.add_argument("--history-hours", type=int, default=12)
    parser.add_argument("--lead-times", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--test-size", type=int, default=744)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=228)
    parser.add_argument("--output-dir", default="extension/results/shared_eval_results")
    parser.add_argument("--no-save-models", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
