import sys
import xarray as xr
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import webbrowser
import json
import numpy as np

# ---------- Load Data ----------
# filename = sys.argv[1]   # <--- uncomment to accept from command line
filename = '_outputs/psychostimulants/sub-113_ses-01_task-RESTING_run-01_eeg.vhdr@SpectrumArrayWelchEpochsAverage5seg.nc'

ds = xr.open_dataarray(filename)

dims = list(ds.dims)
coords = {dim: ds.coords[dim].values for dim in dims}

# ---------- Build App ----------
app = Dash(__name__)

def make_dropdown(name, options):
    return dcc.Dropdown(
        id=f"dropdown-{name}",
        options=[{"label": str(o), "value": str(o)} for o in options],
        value=str(options[0]),
        clearable=False,
    )

# One label + dropdown for each dimension
dropdowns = []
for dim in dims:
    dropdowns.append(html.Label(dim))
    dropdowns.append(make_dropdown(dim, coords[dim]))

app.layout = html.Div([
    html.H2(f"Explorer for {filename}"),
    html.Div(dropdowns, style={"marginBottom": "20px"}),

    html.Label("X-axis dimension"),
    dcc.Dropdown(
        id="x-dim",
        options=[{"label": d, "value": d} for d in dims],
        value=dims[-1],
        clearable=False
    ),

    html.Label("Plot type"),
    dcc.Dropdown(
        id="plot-type",
        options=[
            {"label": "Line", "value": "line"},
            {"label": "Scatter (points)", "value": "scatter"},
            {"label": "Bar", "value": "bar"}
        ],
        value="line",
        clearable=False
    ),

    dcc.Graph(id="plot"),

    html.H3("Debug info"),
    html.Pre(id="debug-output", style={"whiteSpace": "pre-wrap", "border": "1px solid #ccc", "padding": "10px"})
])

# ---------- Callbacks ----------

# Disable the dropdown that matches the selected x-dim
@app.callback(
    [Output(f"dropdown-{dim}", "disabled") for dim in dims],
    Input("x-dim", "value")
)
def disable_selected(xdim):
    return [dim == xdim for dim in dims]

def safe_sel(arr, slice_dict, xdim):
    sel_dict = {}
    for dim, val in slice_dict.items():
        coord = arr.coords[dim].values
        if np.issubdtype(coord.dtype, np.number) and np.all(np.diff(coord) >= 0):
            # numeric and sorted
            sel_dict[dim] = float(val)
            arr = arr.sel({dim: sel_dict[dim]}, method="nearest")
        else:
            # fallback to exact matching
            sel_dict[dim] = val
            arr = arr.sel({dim: sel_dict[dim]})
    return arr

@app.callback(
    [Output("plot", "figure"),
     Output("debug-output", "children")],
    [Input(f"dropdown-{dim}", "value") for dim in dims] +
    [Input("x-dim", "value"),
     Input("plot-type", "value")]
)
def update_plot(*vals):
    slice_dict = {dim: val for dim, val in zip(dims, vals[:-2]) if dim != vals[-2]}
    xdim = vals[-2]
    plot_type = vals[-1]

    # Use safe selection
    arr = safe_sel(ds, slice_dict, xdim)

    # Terminal debug
    print(f"Selected slice: {slice_dict}, xdim={xdim}, plot_type={plot_type}")
    print(f"arr shape: {arr.shape}, dims: {arr.dims}")

    fig = go.Figure()

    def add_trace(x, y, name=None):
        if plot_type == "line":
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))
        elif plot_type == "scatter":
            fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=name))
        elif plot_type == "bar":
            fig.add_trace(go.Bar(x=x, y=y, name=name))

    if arr.ndim == 0:
        fig.add_annotation(text=f"Scalar value: {arr.item()}", x=0.5, y=0.5, showarrow=False)
    elif arr.ndim == 1:
        add_trace(arr.coords[xdim], arr.values)
    elif arr.ndim == 2:
        otherdim = [d for d in arr.dims if d != xdim][0]
        for val in arr.coords[otherdim].values:
            sl = safe_sel(arr, {otherdim: val}, xdim)
            add_trace(sl.coords[xdim], sl.values, name=f"{otherdim}={val}")

    debug_info = {
        "slice_dict": {k: str(v) for k, v in slice_dict.items()},
        "xdim": str(xdim),
        "plot_type": plot_type,
        "arr_shape": tuple(int(x) for x in arr.shape),
        "arr_dims": [str(d) for d in arr.dims]
    }

    return fig, json.dumps(debug_info, indent=2)

# ---------- Launch ----------
if __name__ == "__main__":
    # webbrowser.open("http://127.0.0.1:8050")   # <--- uncomment to auto-open browser
    app.run(debug=True)
