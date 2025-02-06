import pandas as pd
import altair as alt
import streamlit as st
import pydeck as pdk
import matplotlib.pyplot as plt

from utils import add_hours, dtw_distance, plot_trajectories

'# Foursquare New York Check-in dataset'

train = pd.read_csv('data/train_latlon.csv')
test = pd.read_csv('data/test_latlon.csv')
data = pd.concat([train, test], ignore_index=True)
data = add_hours(data)

with st.expander('Sample data points'):
    st.dataframe(data[:100], use_container_width=True)

'**Number of unique values per column**'
# add the longitude and latitude as a tuple column
data['latlon'] = list(zip(data['lat'], data['lon']))
uniq = data.nunique()
# display as single row dataframe
uniq = pd.DataFrame(uniq).T
# drop columns 'lat' and 'lon'
uniq = uniq.drop(columns=['lat', 'lon'])
# make index called 'unique values'
uniq.index = ['count']
st.dataframe(uniq, use_container_width=True)

'### label (user ID)'
f'Number of unique users:  {data.label.nunique()}'
f'Domain: integers between {data.label.min()} and {data.label.max()}'
users = data.label.unique()
# number of unique trajectories per user
trajs_per_user = data.groupby('label').tid.nunique().reset_index()
trajs_per_user.columns = ['label', 'num_trajectories']
chart = alt.Chart(trajs_per_user).mark_bar().encode(
    alt.X(
        'num_trajectories:Q',
        bin=alt.Bin(extent=[10, 45], step=2),
        title='number of trajectories',
        axis=alt.Axis(format='d')
    ),
    alt.Y('count()', title='number of users')
).properties(
    title='Number of trajectories per user with average marked'
)
# Create the rule mark for the red dashed line
mean_trajs = trajs_per_user.num_trajectories.mean()
vline = alt.Chart(pd.DataFrame({'num_trajectories': [mean_trajs]})).mark_rule(
    color='red', strokeDash=[5, 5]
).encode(
    x='num_trajectories:Q'
)
chart += vline
st.altair_chart(chart, use_container_width=True)


'### tid (trajectory ID)'
f'Number of unique trajectories: {data.tid.nunique()}'
f'Domain: integers between {data.tid.min()} and {data.tid.max()}'
# draw a histogram of trajectory lengths in hours, that is, for each trajectory, the difference between the first and last hours,
# overlaid with the histogram of trajectory lengths in number of data points
tids = data.tid.unique()
durations = []
sizes = []
for tid in tids:
    traj = data[data.tid == tid]
    duration = traj.hours.iloc[-1] - traj.hours.iloc[0]
    durations.append(duration)
    sizes.append(len(traj))

durations = pd.DataFrame({'tid': tids, 'duration': durations, 'size': sizes})
hist_duration = alt.Chart(durations).mark_bar(opacity=0.5, color='blue').encode(
    alt.X('duration', bin=alt.Bin(maxbins=30), title='total duration in hours'),
    alt.Y('count()', title='number of trajectories')
).properties(
    title='Histogram of Trajectory Durations'
)

hist_size = alt.Chart(durations).mark_bar(opacity=0.5, color='red').encode(
    alt.X('size', bin=alt.Bin(maxbins=30), title='length in number of data points'),
    alt.Y('count()', title='number of trajectories')
).properties(
    title='Histogram of Trajectory Sizes'
)

st.altair_chart(hist_size + hist_duration, use_container_width=True)




'### latitude and longitude'
f'Number of unique value pairs: {data.latlon.nunique()}'
f'Domain: pairs of floats between ({data.lat.min():.2f}, {data.lon.min():.2f}) and ({data.lat.max():.2f}, {data.lon.max():.2f})'

'**Histogram over the number of data points per location**'
hex_layer = pdk.Layer(
    "HexagonLayer",
    data,
    get_position=["lon", "lat"],
    radius=600,
    elevation_scale=15,
    elevation_range=[0, 1000],
    pickable=True,
    extruded=True,
    getTooltip=lambda d: f"Longitude: {d['position'][0]:.2f}, Latitude: {d['position'][1]:.2f}, Count: {d['points'].length}"
)

# Set the viewport location
view_state = pdk.ViewState(
    latitude=data['lat'].mean(),
    longitude=data['lon'].mean(),
    zoom=9.5,
    pitch=50,
)

# Render the deck.gl map
r = pdk.Deck(
    layers=[hex_layer],
    initial_view_state=view_state,
    tooltip={"text": "{position}\ncount: {elevationValue}"},
)

st.pydeck_chart(r)


'### hour'
f'Number of unique values: {data.hour.nunique()}'
f'Domain: integers between {data.hour.min()} and {data.hour.max()}'

hist_hour = alt.Chart(data).mark_bar().encode(
    x=alt.X('hour:O', title='Hour of the day'),  # Use ordinal type
    y=alt.Y('count()', title='Number of data points')
).properties(
    title='Number of check-ins at each hour of the day'
)
st.altair_chart(hist_hour, use_container_width=True)
'Less check ins during the night, the most check ins around the times of the three meals of the day.'

'### day'
f'Number of unique values: {data.day.nunique()}'
f'Domain: integers between {data.day.min()} and {data.day.max()}'

hist_day = alt.Chart(data).mark_bar().encode(
    x=alt.X('day:O', title='Day of the week'),  # Use ordinal type
    y=alt.Y('count()', title='Number of data points')
).properties(
    title='Number of check-ins on each day of the week'
)
st.altair_chart(hist_day, use_container_width=True)

'### now combining days and hours'
heatmap = alt.Chart(data).mark_rect().encode(
    alt.X('hour:O', title='Hour of the day'),
    alt.Y('day:O', title='Day of the week'),
    alt.Color('count()', title='Number of data points')
).properties(
    title='Number of check-ins at each hour of the day and each day of the week'
)
st.altair_chart(heatmap, use_container_width=True)
'Notice how during the week (0-4) there are many check ins at 8 in the morning, while on the weekend (5-6) the early morning hours are sparse and Friday and Saturday are the most populated evenings.'

# histogram over 'hours' column
hist_hours = alt.Chart(data).mark_bar().encode(
    alt.X('hours:O', title='Hours'),
    alt.Y('count()', title='Number of data points')
).properties(
    title='Number of check-ins at each hour of the week'
)
st.altair_chart(hist_hours, use_container_width=True)
"""We observe that on weekdays people adhere to a more regular schedule, while on weekends the distribution is more spread out. \
Let's investigate this further by looking at the distributions of numbers of check-ins across hours for each weekday separately, \
    then plot their means and standard deviations"""


pts = []
for weekday in range(7):
    points_per_hour = data[data.day == weekday].groupby('hours').size()
    pts.append(points_per_hour)

# plot means and stds as points in 2d space, the dimensions being mean and std. Each point should be labeled with its 0-6 label (weekday)
means = [pt.mean() for pt in pts]
stds = [pt.std() for pt in pts]
DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df = pd.DataFrame({'weekday': DAYS, 'mean': means, 'std': stds})
# Base points layer
points = alt.Chart(df).mark_circle(size=100).encode(
    x=alt.X('mean:Q', scale=alt.Scale(zero=False), title='Mean'),
    y=alt.Y('std:Q', scale=alt.Scale(zero=False), title='Standard Deviation'),
    tooltip=['weekday', 'mean', 'std']
)

# Text labels layer (adjust alignment and displacement as needed)
text = alt.Chart(df).mark_text(
    align='left',
    dx=10,  # horizontal offset
    dy=-10  # vertical offset
).encode(
    x='mean:Q',
    y='std:Q',
    text=alt.Text('weekday:O')  # or combine fields using a calculated field if needed
)

# Combine layers and display
chart = points + text
chart = chart.properties(
    title='Mean and standard deviation over numbers of check-ins across hours of the day for each weekday'
)

st.altair_chart(chart, use_container_width=True)

"We see that the weekend has the smalles stdev and Sunday the least average check ins."

'### category (semantic information)'
f'Number of unique values: {data.category.nunique()}'
f'Domain: integers between {data.category.min()} and {data.category.max()}'

st.altair_chart(alt.Chart(data).mark_bar().encode(
    x=alt.X('category:O', title='Category'),
    y=alt.Y('count()', title='Number of data points')
).properties(
    title='Number of check-ins in each category'
), use_container_width=True)

category = st.selectbox('Select a category:', data.category.unique())
category_data = data[data.category == category]
# histogram of numbers of check-ins per hour of the day for the selected category
hist_hour_category = alt.Chart(category_data).mark_bar().encode(
    x=alt.X('hour:O', title='Hour of the day'),
    y=alt.Y('count()', title='Number of data points')
).properties(
    title=f'Number of check-ins at each hour of the day for category {category}'
)
st.altair_chart(hist_hour_category, use_container_width=True)