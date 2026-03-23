"""
Streamlit dashboard for visualizing Spotify user-song graph analysis results.

Run with: streamlit run dashboards/app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path

# Get project root (parent of dashboards/)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

st.set_page_config(page_title="Spotify Graph Analysis", layout="wide")

st.title("Spotify User-Song Graph Analysis")

st.markdown("""
This dashboard visualizes the results of a spectral embedding and clustering analysis
on Spotify listening data. Users and songs are embedded into the same continuous vector
space based on their listening relationships, then clustered independently using k-means.

The embedding coordinates capture structural similarities in the user-song bipartite graph -
users with similar listening patterns and songs with similar listener bases end up near
each other in the embedding space.
""")


@st.cache_data
def load_data():
    """Load user and song class data with embeddings."""
    users = pd.read_csv(DATA_DIR / "user_classes.csv")
    songs = pd.read_csv(DATA_DIR / "song_classes.csv")
    listens = pd.read_csv(DATA_DIR / "filtered_listens.csv")
    return users, songs, listens


def get_embedding_columns(df):
    """Get list of embedding column names."""
    return [col for col in df.columns if col.startswith("embedding_")]


def build_match_matrix(users_df, songs_df, listens_df):
    """Build a matrix of listen counts per (user_class, song_class) pair."""
    # Create mappings from rank to class
    user_class_map = users_df.set_index("user_rank")["class"].to_dict()
    song_class_map = songs_df.set_index("song_rank")["class"].to_dict()

    # Count matches
    match_counts = {}
    for _, row in listens_df.iterrows():
        user_rank = int(row["user_rank"])
        song_rank = int(row["song_rank"])
        if user_rank in user_class_map and song_rank in song_class_map:
            uc = user_class_map[user_rank]
            sc = song_class_map[song_rank]
            match_counts[(uc, sc)] = match_counts.get((uc, sc), 0) + 1

    # Build matrix
    user_classes = sorted(users_df["class"].unique())
    song_classes = sorted(songs_df["class"].unique())

    matrix = np.zeros((len(user_classes), len(song_classes)))
    for (uc, sc), count in match_counts.items():
        ui = user_classes.index(uc)
        si = song_classes.index(sc)
        matrix[ui, si] = count

    return pd.DataFrame(matrix, index=user_classes, columns=song_classes)


# Load data
users_df, songs_df, listens_df = load_data()
embedding_cols = get_embedding_columns(users_df)

# Axis selection controls
st.sidebar.header("Embedding Axes")
x_axis = st.sidebar.selectbox("X axis", embedding_cols, index=0)
y_axis = st.sidebar.selectbox("Y axis", embedding_cols, index=1)

# Create two columns for the scatter plots
col1, col2 = st.columns(2)

with col1:
    st.subheader("User Clusters")
    fig_users = px.scatter(
        users_df,
        x=x_axis,
        y=y_axis,
        color="class",
        color_continuous_scale="viridis",
        hover_data=["user_hash"],
        title=f"Users ({len(users_df):,} points)"
    )
    fig_users.update_layout(height=500)
    fig_users.update_traces(marker=dict(size=3, opacity=0.6))
    st.plotly_chart(fig_users, use_container_width=True)

with col2:
    st.subheader("Song Clusters")
    fig_songs = px.scatter(
        songs_df,
        x=x_axis,
        y=y_axis,
        color="class",
        color_continuous_scale="plasma",
        hover_data=["artistname", "trackname"],
        title=f"Songs ({len(songs_df):,} points)"
    )
    fig_songs.update_layout(height=500)
    fig_songs.update_traces(marker=dict(size=3, opacity=0.6))
    st.plotly_chart(fig_songs, use_container_width=True)

# Heatmap
st.subheader("User-Song Cluster Affinity Heatmap")
st.markdown("""
This heatmap shows the number of listens between each user cluster and song cluster.
Brighter cells indicate stronger affinity between those cluster pairs.
""")

with st.spinner("Building match matrix..."):
    match_matrix = build_match_matrix(users_df, songs_df, listens_df)

fig_heatmap = px.imshow(
    match_matrix,
    labels=dict(x="Song Cluster", y="User Cluster", color="Listens"),
    color_continuous_scale="YlOrRd",
    aspect="auto"
)
fig_heatmap.update_layout(height=600)
st.plotly_chart(fig_heatmap, use_container_width=True)
