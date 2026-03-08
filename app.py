import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go

st.set_page_config(page_title="Traffic Flow Forecasting - STGCN", page_icon="🚦", layout="wide")

INPUT_LEN = 12
OUTPUT_LEN = 3
HIDDEN_CHANNELS = 64
DROPOUT = 0.2
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "stgcn_tuned_real_graph_model.pt"
H5_PATH = "METR-LA.h5"
PKL_PATH = "adj_METR-LA.pkl"


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(padding, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()
        self.theta = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        x = torch.einsum("ij,btjc->btic", adj, x)
        x = self.theta(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return x


class STGCNBlock(nn.Module):
    def __init__(self, channels, dropout=0.2):
        super().__init__()
        self.temp1 = TemporalConv(channels, channels, kernel_size=3, dropout=dropout)
        self.graph = GraphConv(channels, channels, dropout=dropout)
        self.temp2 = TemporalConv(channels, channels, kernel_size=3, dropout=dropout)

    def forward(self, x, adj):
        residual = x
        x = self.temp1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.graph(x, adj)
        x = x.permute(0, 3, 1, 2)
        x = self.temp2(x)
        return x + residual


class BetterSTGCN(nn.Module):
    def __init__(self, num_nodes, input_len=12, output_len=3, hidden_channels=64, dropout=0.2):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.input_proj = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels)
        )
        self.block1 = STGCNBlock(hidden_channels, dropout=dropout)
        self.block2 = STGCNBlock(hidden_channels, dropout=dropout)
        self.readout = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels),
            nn.Dropout(dropout)
        )
        self.fc = nn.Linear(input_len * hidden_channels, output_len)

    def forward(self, x, adj):
        x = x.unsqueeze(1)
        x = self.input_proj(x)
        x = self.block1(x, adj)
        x = self.block2(x, adj)
        x = self.readout(x)
        x = x.permute(0, 3, 2, 1)
        b, n, t, c = x.shape
        x = x.reshape(b, n, t * c)
        x = self.fc(x)
        x = x.permute(0, 2, 1)
        return x


def normalize_adjacency(adj):
    adj = adj.copy().astype(np.float32)
    np.fill_diagonal(adj, 1.0)
    deg = adj.sum(axis=1)
    deg[deg == 0] = 1.0
    d_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    return (d_inv_sqrt @ adj @ d_inv_sqrt).astype(np.float32)


def make_sequences(data_2d, input_len=12, output_len=3):
    X, y = [], []
    total = len(data_2d)
    for i in range(total - input_len - output_len + 1):
        X.append(data_2d[i:i + input_len])
        y.append(data_2d[i + input_len:i + input_len + output_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def inverse_transform_3d(arr_3d, scaler):
    s, h, n = arr_3d.shape
    flat = arr_3d.reshape(-1, n)
    inv = scaler.inverse_transform(flat)
    return inv.reshape(s, h, n)


def inverse_transform_2d(arr_2d, scaler):
    return scaler.inverse_transform(arr_2d)


def compute_metrics(y_true, y_pred, mape_threshold=5.0):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true > mape_threshold
    if mask.sum() == 0:
        mape = np.nan
    else:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)}


def evaluate_naive_last_value(X, y, scaler):
    pred = np.repeat(X[:, -1:, :], repeats=y.shape[1], axis=1)
    y_true_inv = inverse_transform_3d(y, scaler)
    pred_inv = inverse_transform_3d(pred, scaler)
    return pred_inv, compute_metrics(y_true_inv, pred_inv)


def load_graph_from_pickle(path):
    with open(path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")
    if isinstance(obj, (tuple, list)):
        if len(obj) == 3:
            sensor_ids, sensor_id_to_ind, adj_mx = obj
        else:
            raise ValueError(f"Unexpected tuple/list structure in pickle: length={len(obj)}")
    elif isinstance(obj, dict):
        if "sensor_ids" in obj and "sensor_id_to_ind" in obj and "adj_mx" in obj:
            sensor_ids = obj["sensor_ids"]
            sensor_id_to_ind = obj["sensor_id_to_ind"]
            adj_mx = obj["adj_mx"]
        else:
            raise ValueError("Could not infer keys in pickle dictionary.")
    else:
        raise ValueError("Unsupported pickle structure.")
    sensor_ids = list(sensor_ids)
    adj_mx = np.array(adj_mx, dtype=np.float32)
    return sensor_ids, sensor_id_to_ind, adj_mx


@st.cache_data(show_spinner=False)
def load_h5_dataframe(path):
    return pd.read_hdf(path)


@st.cache_resource(show_spinner=False)
def build_model(num_nodes, model_path):
    model = BetterSTGCN(
        num_nodes=num_nodes,
        input_len=INPUT_LEN,
        output_len=OUTPUT_LEN,
        hidden_channels=HIDDEN_CHANNELS,
        dropout=DROPOUT
    ).to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_inference_batch(model, X, adj_norm):
    adj_torch = torch.tensor(adj_norm, dtype=torch.float32, device=DEVICE)
    X_tensor = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    preds = []
    with torch.no_grad():
        batch_size = 64
        for start in range(0, len(X_tensor), batch_size):
            xb = X_tensor[start:start + batch_size]
            pred = model(xb, adj_torch)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0)


def predict_single_window(model, input_window_scaled, adj_norm):
    adj_torch = torch.tensor(adj_norm, dtype=torch.float32, device=DEVICE)
    x = torch.tensor(input_window_scaled[np.newaxis, :, :], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        pred = model(x, adj_torch).cpu().numpy()[0]
    return pred


def create_forecast_plot(past_times, past_values, future_times, predicted_values, actual_values=None, title="Forecast"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=past_times,
        y=past_values,
        mode="lines+markers",
        name="Past Input"
    ))
    fig.add_trace(go.Scatter(
        x=future_times,
        y=predicted_values,
        mode="lines+markers",
        name="Predicted"
    ))
    if actual_values is not None:
        fig.add_trace(go.Scatter(
            x=future_times,
            y=actual_values,
            mode="lines+markers",
            name="Actual"
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Timestamp",
        yaxis_title="Traffic Speed / Flow",
        template="plotly_white",
        height=500
    )
    return fig


def create_history_plot(times, actual_series, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=actual_series,
        mode="lines",
        name="Actual History"
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Timestamp",
        yaxis_title="Traffic Speed / Flow",
        template="plotly_white",
        height=450
    )
    return fig


def create_comparison_plot(times, actual_series, predicted_series, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=actual_series,
        mode="lines",
        name="Actual"
    ))
    fig.add_trace(go.Scatter(
        x=times,
        y=predicted_series,
        mode="lines",
        name="Predicted"
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Timestamp",
        yaxis_title="Traffic Speed / Flow",
        template="plotly_white",
        height=450
    )
    return fig


def check_required_files():
    missing = []
    for path in [MODEL_PATH, H5_PATH, PKL_PATH]:
        if not os.path.exists(path):
            missing.append(path)
    return missing


@st.cache_data(show_spinner=False)
def prepare_data():
    sensor_ids, sensor_id_to_ind, adj_mx = load_graph_from_pickle(PKL_PATH)
    df = load_h5_dataframe(H5_PATH).copy()
    df.columns = df.columns.map(str)
    sensor_ids_str = list(map(str, sensor_ids))
    missing_in_h5 = [sid for sid in sensor_ids_str if sid not in df.columns]
    if missing_in_h5:
        raise ValueError(f"Missing sensors in H5: {missing_in_h5[:10]}")
    df = df[sensor_ids_str]
    df.index = pd.to_datetime(df.index)

    num_timesteps, num_nodes = df.shape
    if num_nodes != adj_mx.shape[0] or num_nodes != adj_mx.shape[1]:
        raise ValueError("Mismatch between number of nodes in data and adjacency matrix.")

    train_end = int(num_timesteps * TRAIN_RATIO)
    val_end = int(num_timesteps * (TRAIN_RATIO + VAL_RATIO))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df.values)
    val_scaled = scaler.transform(val_df.values)
    test_scaled = scaler.transform(test_df.values)

    X_test, y_test = make_sequences(test_scaled, input_len=INPUT_LEN, output_len=OUTPUT_LEN)
    adj_norm = normalize_adjacency(adj_mx)

    valid_prediction_timestamps = test_df.index[INPUT_LEN: len(test_df) - OUTPUT_LEN + 1]

    return {
        "sensor_ids": sensor_ids,
        "sensor_ids_str": sensor_ids_str,
        "sensor_id_to_ind": sensor_id_to_ind,
        "adj_mx": adj_mx,
        "adj_norm": adj_norm,
        "df": df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_scaled": train_scaled,
        "val_scaled": val_scaled,
        "test_scaled": test_scaled,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "num_timesteps": num_timesteps,
        "num_nodes": num_nodes,
        "valid_prediction_timestamps": valid_prediction_timestamps
    }


st.title("🚦 Traffic Flow Forecasting with STGCN")

missing_files = check_required_files()
if missing_files:
    st.error("Missing required files in repository:")
    for f in missing_files:
        st.write(f"- {f}")
    st.stop()

try:
    with st.spinner("Loading data and model..."):
        data = prepare_data()
        model = build_model(num_nodes=data["num_nodes"], model_path=MODEL_PATH)

    with st.spinner("Running test-set evaluation..."):
        stgcn_pred = run_inference_batch(model, data["X_test"], data["adj_norm"])
        stgcn_pred_inv = inverse_transform_3d(stgcn_pred, data["scaler"])
        stgcn_true_inv = inverse_transform_3d(data["y_test"], data["scaler"])
        naive_pred_inv, naive_metrics = evaluate_naive_last_value(data["X_test"], data["y_test"], data["scaler"])
        stgcn_metrics = compute_metrics(stgcn_true_inv, stgcn_pred_inv)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sensors / Nodes", data["num_nodes"])
    c2.metric("Test Samples", len(data["X_test"]))
    c3.metric("Forecast Horizon", OUTPUT_LEN)
    c4.metric("Timesteps", data["num_timesteps"])

    st.markdown("## Model Performance")
    m1, m2, m3 = st.columns(3)
    m1.metric("STGCN MAE", f"{stgcn_metrics['MAE']:.4f}")
    m2.metric("STGCN RMSE", f"{stgcn_metrics['RMSE']:.4f}")
    m3.metric("STGCN MAPE", "nan" if np.isnan(stgcn_metrics["MAPE"]) else f"{stgcn_metrics['MAPE']:.2f}%")

    results_df = pd.DataFrame([
        {"Model": "Naive Last Value", **naive_metrics},
        {"Model": "STGCN Real Graph", **stgcn_metrics}
    ])
    st.dataframe(results_df, use_container_width=True)

    st.markdown("## Timestamp-Based Prediction")

    col_a, col_b, col_c = st.columns([1.2, 2.2, 1.2])

    sensor_idx = col_a.selectbox(
        "Select Sensor",
        options=list(range(data["num_nodes"])),
        format_func=lambda x: f"{x} (Sensor ID: {data['sensor_ids_str'][x]})"
    )

    timestamp_options = list(data["valid_prediction_timestamps"])
    selected_timestamp = col_b.selectbox(
        "Select Prediction Start Timestamp",
        options=timestamp_options,
        format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m-%d %H:%M:%S")
    )

    forecast_step = col_c.selectbox(
        "Forecast Step",
        options=[1, 2, 3],
        index=0
    )

    test_df = data["test_df"]
    test_scaled = data["test_scaled"]
    scaler = data["scaler"]

    start_pos = test_df.index.get_loc(selected_timestamp)
    input_start = start_pos - INPUT_LEN
    input_end = start_pos
    output_end = start_pos + OUTPUT_LEN

    input_window_scaled = test_scaled[input_start:input_end]
    input_window_original = test_df.iloc[input_start:input_end, :].values
    future_actual_original = test_df.iloc[start_pos:output_end, :].values
    past_times = test_df.index[input_start:input_end]
    future_times = test_df.index[start_pos:output_end]

    single_pred_scaled = predict_single_window(model, input_window_scaled, data["adj_norm"])
    single_pred_original = inverse_transform_2d(single_pred_scaled, scaler)

    selected_pred_value = float(single_pred_original[forecast_step - 1, sensor_idx])
    selected_actual_value = float(future_actual_original[forecast_step - 1, sensor_idx])
    selected_sensor_id = data["sensor_ids_str"][sensor_idx]
    target_timestamp = future_times[forecast_step - 1]

    st.markdown("### Selected Prediction")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Sensor ID", selected_sensor_id)
    p2.metric("Target Timestamp", pd.Timestamp(target_timestamp).strftime("%Y-%m-%d %H:%M:%S"))
    p3.metric("Predicted Speed", f"{selected_pred_value:.2f}")
    p4.metric("Actual Speed", f"{selected_actual_value:.2f}")

    step_labels = [f"t+{i}" for i in range(1, OUTPUT_LEN + 1)]
    step_df = pd.DataFrame({
        "Step": step_labels,
        "Timestamp": [pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S") for ts in future_times],
        "Predicted": single_pred_original[:, sensor_idx],
        "Actual": future_actual_original[:, sensor_idx]
    })
    st.dataframe(step_df, use_container_width=True)

    past_sensor_values = input_window_original[:, sensor_idx]
    pred_sensor_values = single_pred_original[:, sensor_idx]
    actual_sensor_values = future_actual_original[:, sensor_idx]

    forecast_fig = create_forecast_plot(
        past_times=past_times,
        past_values=past_sensor_values,
        future_times=future_times,
        predicted_values=pred_sensor_values,
        actual_values=actual_sensor_values,
        title=f"Past Window + Forecast | Sensor {selected_sensor_id}"
    )
    st.plotly_chart(forecast_fig, use_container_width=True)

    st.markdown("## Compare Model Over Time for Selected Sensor")

    compare_points = st.slider(
        "Number of test timestamps to compare",
        min_value=20,
        max_value=min(300, len(stgcn_true_inv)),
        value=min(120, len(stgcn_true_inv))
    )

    compare_times = test_df.index[INPUT_LEN:INPUT_LEN + compare_points]
    actual_series = stgcn_true_inv[:compare_points, 0, sensor_idx]
    predicted_series = stgcn_pred_inv[:compare_points, 0, sensor_idx]

    compare_fig = create_comparison_plot(
        times=compare_times,
        actual_series=actual_series,
        predicted_series=predicted_series,
        title=f"One-Step Prediction Comparison Over Time | Sensor {selected_sensor_id}"
    )
    st.plotly_chart(compare_fig, use_container_width=True)

    st.markdown("## Sensor History")
    history_points = st.slider(
        "Number of historical points to show",
        min_value=30,
        max_value=min(500, len(test_df)),
        value=min(150, len(test_df))
    )
    history_times = test_df.index[:history_points]
    history_values = test_df.iloc[:history_points, sensor_idx].values

    history_fig = create_history_plot(
        times=history_times,
        actual_series=history_values,
        title=f"Historical Traffic Series | Sensor {selected_sensor_id}"
    )
    st.plotly_chart(history_fig, use_container_width=True)

    st.markdown("## Download Results")
    metrics_csv = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download metrics as CSV",
        data=metrics_csv,
        file_name="stgcn_metrics.csv",
        mime="text/csv"
    )

    step_csv = step_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download selected timestamp prediction as CSV",
        data=step_csv,
        file_name="selected_timestamp_prediction.csv",
        mime="text/csv"
    )

except Exception as e:
    st.exception(e)
