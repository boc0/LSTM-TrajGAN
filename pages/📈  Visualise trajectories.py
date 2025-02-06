import io
import requests
from PIL import Image
from functools import partial

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import altair as alt
import pydeck as pdk
import numpy as np
from utils import add_hours, dtw_distance, plot_trajectories
import pydeck as pdk

data = pd.read_csv('data/train_latlon.csv')

def hist_trajs_per_user():
    # number of unique trajectories per user
    trajs_per_user = data.groupby('label').tid.nunique().reset_index()
    trajs_per_user.columns = ['label', 'num_trajectories']
    chart = alt.Chart(trajs_per_user).mark_bar().encode(
        alt.X(
            "num_trajectories:Q",
            bin=alt.Bin(extent=[10, 30], step=2),
            title="Number of trajectories per user",
            axis=alt.Axis(format='d')
        ),
        alt.Y("count()", title="Count of users")
    ).properties(
        title="Number of trajectories per user"
    )

    st.altair_chart(chart, use_container_width=True)

# hist_trajs_per_user()

original = pd.read_csv('data/test_latlon.csv')
synthetic = pd.read_csv('results/syn_traj_test.csv')
original = add_hours(original)
synthetic = add_hours(synthetic)

users = set(original.label)
user = st.selectbox('Select a user ID:', ['All'] + list(users), key='user')
if user == 'All':
    tids = set(original.tid)
else:
    tids = set(original[original.label == user].tid)
tid = st.selectbox('Select a trajectory ID:', list(tids), key='tid')

orig = original[original.tid == tid]
show_synth = st.checkbox('Show synthetic trajectory')
show_more = st.checkbox('Show more trajectories')
trajs = [orig]

data = original
# Compute ranges for each dimension
lat_range = data['lat'].max() - data['lat'].min()
lon_range = data['lon'].max() - data['lon'].min()
hours_range = data['hours'].max() - data['hours'].min()

# Compute weights as the inverse of the range
w_lat = 1.0 / lat_range
w_lon = 1.0 / lon_range
w_hours = 1.0 / hours_range


distance = partial(dtw_distance, w_lat=w_lat, w_lon=w_lon, w_hours=w_hours)
dists = []
if show_synth:
    synth = synthetic[synthetic.tid == tid]
    trajs.append(synth)
if show_more:
    count = st.slider('Number of additional trajectories to show', 1, 10, 1)
    for other in tids - {tid}:
        dist = distance(orig, synthetic[original.tid == other])
        dists.append(dist)
        # set 'dst' to the distance value in the 'original' dataframe
        original.loc[original.tid == other, 'dist'] = dist
    display_tids = original.sort_values('dist').head(count).tid
    # tids = list(tids)
    # tids = np.random.choice(tids, count, replace=False)
    trajs.extend([original[original.tid == tid] for tid in display_tids])
synth = synthetic[synthetic.tid == tid]
fig = plot_trajectories(*trajs)
st.plotly_chart(fig)

# make a histogram of the distances to all other trajectories
# mark on it the distance to the synthetic trajectory
if show_more:
    dists_df = pd.DataFrame(dists, columns=['distance'])
    chart = alt.Chart(dists_df).mark_bar().encode(
        alt.X(
            "distance:Q",
            bin=alt.Bin(maxbins=30),
            title="Distance to original trajectory"
        ),
        alt.Y("count()", title="Count")
    ).properties(
        title="Histogram of distances to other trajectories"
    )

    if show_synth:
        synth_distance = distance(orig, synth)
        vline = alt.Chart(pd.DataFrame({'distance': [synth_distance]})).mark_rule(color='red', strokeDash=[5, 5]).encode(
            x='distance:Q'
        )
        chart += vline

    st.altair_chart(chart, use_container_width=True)


# find rank of distance to synthetic trajectory among all distances
if show_synth:
    rank = (dists_df['distance'] < synth_distance).sum() + 1
    st.write(f"Rank of distance to synthetic trajectory: {rank}/{len(dists_df)}")