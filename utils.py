import io
import requests
from PIL import Image

import pandas as pd
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt


def add_hours(df):
    df['hours'] = df['day'] * 24 + df['hour']
    return df


def split_trajs(data):
    tids = set(data.tid)
    trajectories = {}
    for tid in tids:
        trajectories[tid] = data[data.tid == tid]
    return trajectories


def add_traj_length(df): # in hours
    tids = set(df.tid)
    for tid in tids:
        traj = df[df.tid == tid]
        ln = len(traj)
        df.loc[df.tid == tid, 'length'] = traj.hours.iloc[ln-1] - traj.hours.iloc[0]
    return df


def dtw_distance(traj1, traj2, w_lon=1.0, w_lat=1.0, w_hours=0.01):
    """
    Compute the Dynamic Time Warping (DTW) distance between two trajectories,
    with weighting factors for longitude, latitude, and time (hours).
    
    Each trajectory is assumed to be a DataFrame with columns: 'lon', 'lat', 'hours'.
    The weights w_lon, w_lat, and w_hours determine the relative importance of each dimension.
    
    Parameters:
        traj1: pd.DataFrame with columns ['lon', 'lat', 'hours']
        traj2: pd.DataFrame with columns ['lon', 'lat', 'hours']
        w_lon: weight for the longitude difference
        w_lat: weight for the latitude difference
        w_hours: weight for the hours difference
        
    Returns:
        float: the weighted DTW distance between the two trajectories.
    """
    
    # Convert trajectories to numpy arrays with shape (N, 3) and (M, 3)
    seq1 = traj1[['lon', 'lat', 'hours']].to_numpy()
    seq2 = traj2[['lon', 'lat', 'hours']].to_numpy()
    
    n, m = len(seq1), len(seq2)
    
    # Initialize the DTW matrix with infinities
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0.0
    
    # Pre-calc squared weights for efficiency.
    w_lon2 = w_lon ** 2
    w_lat2 = w_lat ** 2
    w_hours2 = w_hours ** 2
    
    # Compute weighted Euclidean distance and fill in the DTW matrix.
    for i in range(1, n+1):
        for j in range(1, m+1):
            d_lon   = seq1[i-1, 0] - seq2[j-1, 0]
            d_lat   = seq1[i-1, 1] - seq2[j-1, 1]
            d_hours = seq1[i-1, 2] - seq2[j-1, 2]
            cost = np.sqrt(w_lon2 * d_lon**2 + w_lat2 * d_lat**2 + w_hours2 * d_hours**2)
            dtw[i, j] = cost + min(dtw[i-1, j],    # Insertion
                                   dtw[i, j-1],    # Deletion
                                   dtw[i-1, j-1])  # Match
            
    return dtw[n, m]


def fetch_map_image(min_lon, max_lon, min_lat, max_lat):

    url = f"https://static-maps.yandex.ru/1.x/?bbox={min_lon},{min_lat}~{max_lon},{max_lat}&l=map"
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content))
    return img


def plot_trajectories(*trajs, min_lat=None, max_lat=None, min_lon=None, max_lon=None, full_week=True):
    min_lat = min_lat or min(traj.lat.min() for traj in trajs)
    max_lat = max_lat or max(traj.lat.max() for traj in trajs)
    min_lon = min_lon or min(traj.lon.min() for traj in trajs)
    max_lon = max_lon or max(traj.lon.max() for traj in trajs)

    map_img = fetch_map_image(min_lon, max_lon, min_lat, max_lat)
    fp = io.BytesIO()
    map_img.save(fp=fp, format='PNG')
    fp.seek(0)
    img = plt.imread(fp)
    img_array = img[:, :, :3].reshape((img.shape[0] * img.shape[1], 3))
    colors = np.unique(img_array, axis=0)
    n_colors = colors.shape[0]
    color_to_value = {tuple(color[:3]): i / (n_colors - 1) for i, color in enumerate(colors)}
    my_cmap_ply = [(value, 'rgb({}, {}, {})'.format(*color)) for color, value in color_to_value.items()]
    fun_find_value = lambda x: color_to_value[tuple(x[:3])]
    values = np.apply_along_axis(fun_find_value, 2, np.flipud(img))

    xx = np.linspace(min_lon, max_lon, img.shape[1])
    yy = np.linspace(min_lat, max_lat, img.shape[0])
    zz = np.zeros(img.shape[:2]) if full_week else np.ones(img.shape[:2]) * orig.hours.min()

    surf = go.Surface(
        x=xx, y=yy, z=zz,
        colorscale=my_cmap_ply,
        surfacecolor=values,
        showscale=False
    )
    pts = []
    colors = ['blue', 'green'] + 10 * ['red'] #,'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']  # Define a list of colors
    for idx, traj in enumerate(trajs):
        color = colors[idx % len(colors)]  # Cycle through colors if there are more trajectories than colors
        pts.append(go.Scatter3d(x=traj.lon, y=traj.lat, z=traj.hours, mode='lines', line=dict(width=4, color=color)))
        pts.append(go.Scatter3d(x=traj.lon, y=traj.lat, z=traj.hours, mode='markers', marker=dict(size=3, color=color)))
    fig = go.Figure(data=[surf, *pts], layout=go.Layout(
        scene=dict(
            xaxis=dict(title='Longitude'),
            yaxis=dict(title='Latitude'),
            zaxis=dict(title='Time (hours)')
        )
    ))
    if full_week:
        # set hours to always be 0-167
        fig.update_layout(scene=dict(zaxis=dict(range=[0, 167])))
    else:
        pts = points_orig[:, 2]
        fig.update_layout(scene=dict(zaxis=dict(range=[pts.min(), pts.max()])))

    return fig

