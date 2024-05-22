from src.utils import (
    load_batch_to_device,
    tensors_to_images
)
from src.metrics import MetricMonitor
from .base_trainer import BaseTrainer
from matplotlib.figure import Figure
from typing import Dict
import torch
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import numpy as np


SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
pollutant_names = {
    'no2_conc': 'NO2'.translate(SUB),
    'o3_conc': 'O3'.translate(SUB),
    'pm10_conc': 'PM10'.translate(SUB),
    'so2_conc': 'SO2'.translate(SUB),

    'chocho_conc': 'chocho_conc',
    'co_conc': 'CO',
    'dust': 'Dust',
    'pm10_ss_conc': 'pm10_ss_conc',
    'ecres_conc': 'ecres_conc',
    'ectot_conc': 'ectot_conc',
    'hcho_conc': 'hcho_conc',
    'nh3_conc': 'NH3'.translate(SUB),
    'nmvoc_conc': 'nmvoc_conc',
    'no_conc': 'NO',
    'pans_conc': 'pans_conc',
    'pm2p5_total_om_conc': 'pm2p5_total_om_conc',
    'pm2p5_conc': 'PM2.5'.translate(SUB),
    'pmwf_conc': 'pmwf_conc',
    'sia_conc': 'sia_conc'
}


class RegressionTrainer(BaseTrainer):
    def compute_metrics(self, metric_monitor: MetricMonitor, output, batch) -> dict:
        """
        Update metric_monitor with the metrics computed from output and batch.
        """
        y_true = batch["y"].detach().cpu().numpy()
        y_pred = output.detach().cpu().numpy()
        class_names = self.val_dl.dataset.class_names

        mean_mape = []
        for i, class_name in enumerate(class_names):
            mape = sklearn.metrics.mean_absolute_percentage_error(y_true[:, i], y_pred[:, i])
            metric_monitor.update(f"{class_name}_mape", mape)
            mean_mape.append(mape)
        metric_monitor.update("mean_mape", np.mean(mean_mape))

        return metric_monitor.get_metrics()

    @torch.no_grad()
    def generate_media(self) -> Dict[str, Figure]:
        """
        Generate media from output and batch.
        """
        self.model.eval()

        y_true = []
        y_pred = []
        datetimes = []
        for step, batch in enumerate(self.val_dl):
            batch = load_batch_to_device(batch, self.device)
            output = self.predict(self.model, batch)

            y_true.append(batch["y"].detach().cpu().numpy())
            y_pred.append(output.detach().cpu().numpy())
            datetimes.append(batch["datetime"])

        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)
        datetimes = np.hstack(datetimes)
        class_names = self.val_dl.dataset.class_names

        results = plot_results(y_true, y_pred, datetimes, class_names)
        return results


def smooth_values(values, window_size=3):
    """
    Smooths the values using a rolling average.

    Parameters:
    - values: 1D numpy array of values to smooth.
    - window_size: The size of the rolling window for smoothing.

    Returns:
    - smoothed_values: 1D numpy array of smoothed values.
    """
    series = pd.Series(values)
    smoothed_series = series.rolling(window=window_size, min_periods=1).mean()
    return smoothed_series.to_numpy()


def plot_results(y_true, y_pred, datetimes, class_names, window_size=10):
    # Convert datetimes to pandas datetime
    datetime_series = pd.to_datetime(datetimes, format='%Y-%m-%d_%H-%M')

    # Create a DataFrame for true and predicted values
    df_true = pd.DataFrame(y_true, columns=class_names, index=datetime_series)
    df_pred = pd.DataFrame(y_pred, columns=class_names, index=datetime_series)

    # Smooth the predicted values
    df_pred_smoothed = df_pred.apply(lambda col: smooth_values(col.values, window_size), axis=0)

    # Dictionary to hold the figures
    figs = {}

    # Plotting
    for class_name in class_names:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df_true.index, df_true[class_name], label='True', marker='o', color='blue')
        # ax.plot(df_pred.index, df_pred[class_name], label='Predicted', marker='x', color='orange', alpha=0.5, linestyle='--')
        ax.plot(df_pred_smoothed.index, df_pred_smoothed[class_name], label='Predicted', marker='x', color='orange')
        # set title with pollutant name and date range
        ax.set_title(f'{pollutant_names[class_name]} - {df_true.index[0].date()}')
        ax.set_xlabel('Datetime (MM-DD hh)')
        ax.set_ylabel('µg/m3'.translate(SUP))
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        figs[class_name] = fig

    return figs
