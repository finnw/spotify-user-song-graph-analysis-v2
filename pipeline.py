import os
import csv
import numpy as np
import pandas as pd
import sklearn.manifold
import sklearn.cluster
from pandas import DataFrame
from scipy import sparse

from os.path import join as path_join
from typing import Iterator, NamedTuple, Optional
from zipfile import ZipFile
from kagglehub import dataset_download
from EscapeFix import EscapeFixer
from LabelEncoding import LabelEncoding


embedding_dimensions = 10  # Number of dimensions for the user and song embeddings. This is a hyperparameter that can be tuned for better performance of the matrix factorization model.

class DatasetStats(NamedTuple):
    num_users: int  # of unique users in the dataset
    num_songs: int  # of unique songs (artist/title pairs) in the dataset
    num_listens: int  # of unique listens (user/song pairs) in the dataset


class ThresholdStats(NamedTuple):
    num_users: int
    num_songs: int
    num_listens: int
    min_listens: int

    def create_edge_array(self, csv_reader: Iterator[list[str]]) -> np.ndarray:
        """
        Creates an edge array of shape (num_listens, 3) where each row is a (user_id, song_id, weight) pair corresponding to a listen.
        In the current implementation, the weight is always 1 since we are treating the listens as binary interactions, but this could be modified in the future to incorporate more complex weighting schemes if desired.
        """
        n = self.num_listens
        edge_array = np.zeros((n*2, 3), dtype=np.float32)
        for i, row in enumerate(csv_reader):
            user_id = int(row[0])
            song_id = int(row[1]) + self.num_users  # Offset song ids to avoid overlap with user ids
            edge_array[i] = [user_id, song_id, 1]  # Add weight (1) for the edge
            edge_array[i+n] = [song_id, user_id, 1]  # Add reverse edge with same weight
        return edge_array


class TransformedView:
    """A view of the dataset possibly after applying transformations"""

    file_basename: str  # Basename of the file needed for this view, e.g. "listens.csv"
    heading: list[str]  # Header row for the CSV files corresponding to this view, e.g. ["user_id", "song_id"]
    in_memory_attributes: set[str]  # Set of attribute names corresponding to the data that should be loaded into memory for this view, e.g. {"song_dict"}

    def __init__(self, heading: Optional[list[str]] = None, in_memory_attributes: Optional[set[str]] = None):
        super().__init__()

    def file_path(self, pipeline: 'Pipeline') -> str:
        """Returns the file path corresponding to the data file needed for this view."""
        return os.path.join(pipeline.data_directory, self.file_basename)

    def is_present(self, pipeline: 'Pipeline') -> bool:
        """Checks whether all the file paths needed for this view are present on the filesystem."""
        return os.path.exists(self.file_path(pipeline))

    def stream(self, pipeline: 'Pipeline') -> Iterator[list[str]]:
        """Streams the data for this view as lists of strings. By default, this method assumes the data is stored in CSV files corresponding to the file_basenames, but this can be overridden for views that require more complex transformations."""
        if self.is_present(pipeline):
            with open(self.file_path(pipeline), 'r', encoding='utf-8') as f:
                csv_reader = csv.reader(f)
                next(csv_reader)  # Skip header
                yield from csv_reader
        else:
            raise FileNotFoundError(f"Data files for view {type(self).__name__} not found. Expected files: {self.file_path(pipeline)}")


class SourceView(TransformedView):
    """The raw source data view."""
    file_basename = "spotify_dataset.csv"
    heading = ["user_id", "artistname", "trackname", "playlistname"]
    in_memory_attributes = set()

    def __init__(self):
        super().__init__()

    def file_path(self, pipeline):
        # This file is in data/source/ instead of data/, so we need to override the file_path method to reflect that
        return os.path.join(pipeline.source_directory, self.file_basename)

    def is_present(self, pipeline):
        if super().is_present(pipeline):
            return True
        # We can also consider the source data to be present if the archive.zip file is present, since we can stream the data directly from the zip file without needing to extract it to a CSV file on disk
        return os.path.exists(pipeline.archive_path)

    def stream(self, pipeline) -> Iterator[list[str]]:
        try:
            # Read frome extracted CSV file if it exists...
            with open(self.file_path(pipeline), 'rb') as file:
                csv_reader = csv.reader(EscapeFixer(file))
                next(csv_reader)  # Skip header
                yield from csv_reader
        except FileNotFoundError:
            # ...otherwise read from the zip file
            with ZipFile(pipeline.archive_path, 'r') as archive:
                with archive.open(name='spotify_dataset.csv', mode='r') as file:
                    csv_reader = csv.reader(EscapeFixer(file))
                    next(csv_reader)  # Skip header
                    yield from csv_reader
        

class UserLabelsView(TransformedView):
    file_basename = "users.csv"
    heading = ["user_id", "user_hash"]
    in_memory_attributes = set()


class SongLabelsView(TransformedView):
    file_basename = "songs.csv"
    heading = ["song_id", "artistname", "trackname"]
    in_memory_attributes = {"song_dict"}


class FullListensView(TransformedView):
    file_basename = "listens.csv"
    heading = ["user_id", "song_id"]
    in_memory_attributes = set()


class StatsView(TransformedView):
    file_basename = "stats.csv"
    heading = ["num_users", "num_songs", "num_listens"]
    in_memory_attributes = {"stats"}


class FilteredListensView(TransformedView):
    file_basename = "filtered_listens.csv"
    heading = ["user_rank", "song_rank"]
    in_memory_attributes = set()


class FilteredSongsView(TransformedView):
    file_basename = "filtered_songs.csv"
    heading = ["rank", "song_id", "listen_count", "artistname", "trackname"]
    in_memory_attributes = set()


class FilteredUsersView(TransformedView):
    file_basename = "filtered_users.csv"
    heading = ["rank", "user_id"]
    in_memory_attributes = set()


class ThresholdStatsView(TransformedView):
    file_basename = "threshold_stats.csv"
    heading = ["num_users", "num_songs", "num_listens", "min_listens"]
    in_memory_attributes = {"threshold_stats"}


class UserEmbeddingsView(TransformedView):
    file_basename = "user_embeddings.csv"
    heading = ["user_rank", "user_id", "user_hash"] + [f"embedding_{i}" for i in range(embedding_dimensions)]
    in_memory_attributes = set()


class SongEmbeddingsView(TransformedView):
    file_basename = "song_embeddings.csv"
    heading = ["song_rank", "song_id", "artistname", "trackname"] + [f"embedding_{i}" for i in range(embedding_dimensions)]
    in_memory_attributes = set()


class UserClassesView(TransformedView):
    file_basename = "user_classes.csv"
    heading = ["user_rank", "user_id", "user_hash", "class"] 
    in_memory_attributes = set()


class SongClassesView(TransformedView):
    file_basename = "song_classes.csv"
    heading = ["song_rank", "song_id", "artistname", "trackname", "class"] 
    in_memory_attributes = set()


class Pipeline:
    project_directory: str  # Path to the project root directory
    support_threshold: int  # Minimum number of listens for a song to be included in the filtered data. Must be at least 2, but in practice must be larger for the matrix factorization to fit in available RAM
    stats: Optional[DatasetStats]  # Statistics about the dataset
    threshold_stats: Optional[ThresholdStats]  # Statistics about the threshold
    stream_from_zip: bool  # Whether to stream the data directly from the zip file or to read from the extracted CSV file. Streaming from the zip file can save disk space and allow processing of larger datasets, but may be slower than reading from a local CSV file.
    keep_song_names: bool  # Whether to keep the song names in memory after encoding. This is used for generating the dashboard, but we can set this to False to save memory when processing the data for matrix factorization.
    song_dict: Optional[dict[int, tuple[str, str]]]  # Mapping from song id to (artist name, song title) tuple. Used for quick lookup of song names when generating the dashboard, but not needed for matrix processing, where we want to minimize memory usage.
    keep_user_hashes: bool  # Whether to keep the user hashes in memory after encoding. This is used for generating the dashboard, but we can set this to False to save memory when processing the data for matrix factorization.
    user_dict: Optional[dict[int, str]]  # Mapping from user id to user hash. Used for quick lookup of user hashes when generating the dashboard, but not needed for matrix processing, where we want to minimize memory usage.
    song_threshold_mapping: Optional[list[int]]  # Mapping from new song ids after applying the support threshold, to their original song ids
    inverse_song_threshold_mapping: Optional[dict[int, int]]  # Inverse mapping from original song ids to new song ids after applying the support threshold
    user_threshold_mapping: Optional[list[int]]  # Mapping from new user ids after applying the support threshold, to their original user ids
    inverse_user_threshold_mapping: Optional[dict[int, int]]  # Inverse mapping from original user ids to new user ids after applying the support threshold

    def __init__(self, project_directory: str):
        super().__init__()
        self.project_directory = project_directory
        self.source_view = SourceView()
        self.user_labels_view = UserLabelsView()
        self.song_labels_view = SongLabelsView()
        self.full_listens_view = FullListensView()
        self.stats_view = StatsView()
        self.filtered_listens_view = FilteredListensView()
        self.filtered_songs_view = FilteredSongsView()
        self.filtered_users_view = FilteredUsersView()
        self.threshold_stats_view = ThresholdStatsView()
        self.user_embeddings_view = UserEmbeddingsView()
        self.song_embeddings_view = SongEmbeddingsView()
        self.user_classes_view = UserClassesView()
        self.song_classes_view = SongClassesView()
        self.views = [
            self.source_view,
            self.user_labels_view,
            self.song_labels_view,
            self.full_listens_view,
            self.stats_view,
            self.filtered_listens_view,
            self.filtered_songs_view,
            self.filtered_users_view,
            self.threshold_stats_view,
            self.user_embeddings_view,
            self.song_embeddings_view,
            self.user_classes_view,
            self.song_classes_view
        ]
        self.stats = None
        self.threshold_stats = None
        self.support_threshold = 2
        self.song_threshold_mapping = None
        self.inverse_song_threshold_mapping = None
        self.user_threshold_mapping = None
        self.inverse_user_threshold_mapping = None
        self.keep_song_names = True
        self.song_dict = None
        self.keep_user_hashes = True
        self.user_dict = None


    @property
    def data_directory(self) -> str:
        return path_join(self.project_directory, "data")

    @property
    def archive_path(self) -> str:
        return path_join(self.data_directory, "archive.zip")

    @property
    def source_directory(self) -> str:
        return path_join(self.data_directory, "source")

    def download_data(self):
        """Ensures the dataset is downloaded and available locally"""
        if not os.path.exists(self.source_path):
            print("Downloading dataset from Kaggle...")
            os.makedirs(self.source_directory, exist_ok=True)
            dataset_download("andrewmvd/spotify-playlists", output_dir=self.source_directory, force_download=True)
            print("Download complete.")

    def lookup_user_hashes(self, user_ids: list[int], force_dict_use: bool = False) -> str:
        """Given a list of user ids, returns a list of user hashes corresponding to those ids."""
        if (self.keep_user_hashes or force_dict_use) and self.user_dict is None:
            # Load the whole dictionary into memory for fast lookup
            self.user_dict = {}
            for entry in self.user_labels_view.stream(self):
                user_id = int(entry[0])
                user_hash = entry[1]
                self.user_dict[user_id] = user_hash

        # If we have the user dictionary (possibly just created), use that for fast lookup
        if self.user_dict is not None:
            return [self.user_dict[user_id] for user_id in user_ids]

        # Fall back to streaming through the users file for each lookup, which is slower but uses less memory
        partial_user_dict = {}
        for entry in self.user_labels_view.stream(self):
            user_id = int(entry[0])
            user_hash = entry[1]
            if user_id in user_ids:
                partial_user_dict[user_id] = user_hash
        return [partial_user_dict[user_id] for user_id in user_ids]

    def lookup_song_names(self, song_ids: list[int], force_dict_use: bool = False) -> list[tuple[str, str]]:
        """Given a list of song ids, returns a list of (artist name, song title) tuples corresponding to those ids."""

        if (self.keep_song_names or force_dict_use) and self.song_dict is None:
            # Load the whole dictionary into memory for fast lookup
            self.song_dict = {}
            for entry in self.song_labels_view.stream(self):
                song_id = int(entry[0])
                artist_name = entry[1]
                song_title = entry[2]
                self.song_dict[song_id] = (artist_name, song_title)

        # If we have the song dictionary (possibly just created), use that for fast lookup
        if self.song_dict is not None:
            return [self.song_dict[song_id] for song_id in song_ids]

        # Fall back to streaming through the songs file for each lookup, which is slower but uses less memory
        partial_song_dict = {}
        for entry in self.song_labels_view.stream(self):
            song_id = int(entry[0])
            artist_name = entry[1]
            song_title = entry[2]
            if song_id in song_ids:
                partial_song_dict[song_id] = (artist_name, song_title)
        return [partial_song_dict[song_id] for song_id in song_ids]

    def lookup_user_hashes(self, user_ids: list[int]) -> list[str]:
        """
        Given a list of user ids, returns a list of user hashes corresponding to those ids.
        This is less useful than looking up song names since the user hashes are anonymous and do not have any semantic meaning, but it can still be useful for generating the dashboard and for sanity checking that the user ids are being mapped correctly to their corresponding hashes.
        """
        if self.keep_user_hashes and self.user_dict is None:
            # Load the whole dictionary into memory for fast lookup
            self.user_dict = {}
            for entry in self.user_labels_view.stream(self):
                user_id = int(entry[0])
                user_hash = entry[1]
                self.user_dict[user_id] = user_hash

        # If we have the user dictionary (possibly just created), use that for fast lookup
        if self.user_dict is not None:
            return [self.user_dict[user_id] for user_id in user_ids]

        # Fall back to streaming through the users file for each lookup, which is slower but uses less memory
        partial_user_dict = {}
        for entry in self.user_labels_view.stream(self):
            user_id = int(entry[0])
            user_hash = entry[1]
            if user_id in user_ids:
                partial_user_dict[user_id] = user_hash
        return [partial_user_dict[user_id] for user_id in user_ids] 

    def encode_listens(self) -> DatasetStats:
        if all(view.is_present(self) for view in {self.user_labels_view, self.song_labels_view, self.full_listens_view, self.stats_view}):
            print("Encoded listens, users, and songs already exist. Skipping encoding step.")
            # Read the existing stats and return them
            for row in self.stats_view.stream(self):
                return DatasetStats(*map(int, row))

        """Applies a label encoding to the listens data, converting user IDs and song titles to integers."""
        user_encoding = LabelEncoding()
        song_encoding = LabelEncoding()

        unique_listens: set[tuple[int, int]] = set()  # To track unique (user_id, song_id) pairs
        for row in self.source_view.stream(self):
            user_id = user_encoding.get_or_create_id(row[0])
            song_name = (row[1], row[2])  # Use a tuple of (song title, artist name) to ensure uniqueness
            song_id = song_encoding.get_or_create_id(song_name)
            if (user_id, song_id) not in unique_listens:
                unique_listens.add((user_id, song_id))
        
        # Write the encoded listens to a new CSV file
        with open(self.full_listens_view.file_path(self), 'w', newline='', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["user_id", "song_id"])
            for user_id, song_id in unique_listens:
                csv_writer.writerow([user_id, song_id])

        # Write the user encodings to a separate CSV file
        user_encoding.export_csv(self.user_labels_view.file_path(self), heading=["user_id", "user_hash"])

        # Write the song encodings to separate CSV files
        song_encoding.export_csv(self.song_labels_view.file_path(self), heading=["song_id", "artistname", "trackname"])
            
        # Write stats about the dataset
        with open(self.stats_view.file_path(self), 'w', newline='', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["num_users", "num_songs", "num_listens"])
            num_users = len(user_encoding.mapping)
            num_songs = len(song_encoding.mapping)
            num_listens = len(unique_listens)
            csv_writer.writerow([num_users, num_songs, num_listens])
            return DatasetStats(num_users, num_songs, num_listens)

    def get_threshold_stats(self) -> Optional[ThresholdStats]:
        """
        Get statistics about the dataset after applying the support threshold
        This may be obtianed from memory or by reading from a previously saved CSV file
        """
        if self.threshold_stats is None:
            if self.threshold_stats_view.is_present(self):
                print("Threshold stats already exist. Reading from file.")
                # Read the existing stats and return them
                for row in self.threshold_stats_view.stream(self):
                    self.threshold_stats = ThresholdStats(*map(int, row))
                    return self.threshold_stats
        return self.threshold_stats

    def apply_threshold(self):
        if all(view.is_present(self) for view in {self.filtered_songs_view, self.filtered_users_view, self.filtered_listens_view, self.threshold_stats_view}):
            print("Filtered data already exists. Skipping thresholding step.")
            return self.get_threshold_stats()
    
        # Count the number of listens for each song
        song_listen_counts: dict[int, int] = {}
        total_filtered_listens = 0
        for row in self.full_listens_view.stream(self):
            song_id = int(row[1])
            song_listen_counts[song_id] = song_listen_counts.get(song_id, 0) + 1
        self.song_threshold_mapping = []
        self.inverse_song_threshold_mapping = {}
        for song_id, count in song_listen_counts.items():
            if count >= self.support_threshold:
                new_song_id = len(self.song_threshold_mapping)
                self.song_threshold_mapping.append(song_id)
                self.inverse_song_threshold_mapping[song_id] = new_song_id
                total_filtered_listens += count
    
        # Save the filtered songs to a new CSV file
        with open(self.filtered_songs_view.file_path(self), 'w', newline='', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["rank", "song_id", "listen_count", "artistname", "trackname"])
            for rank, song_id in enumerate(self.song_threshold_mapping):
                artist_name, song_title = self.lookup_song_names([song_id], True)[0]
                csv_writer.writerow([rank, song_id, song_listen_counts[song_id], artist_name, song_title])

        # Discard the song names from memory to save space
        if not self.keep_song_names:
            self.song_dict = None

        # Determine the set of users that listened to at least one song that meets the support threshold
        self.user_threshold_mapping = []
        self.inverse_user_threshold_mapping = {}
        for row in self.full_listens_view.stream(self):
            user_id = int(row[0])
            song_id = int(row[1])
            if song_id in self.inverse_song_threshold_mapping and user_id not in self.inverse_user_threshold_mapping:
                new_user_id = len(self.user_threshold_mapping)
                self.user_threshold_mapping.append(user_id)
                self.inverse_user_threshold_mapping[user_id] = new_user_id
        
        # Save the filtered users to a new CSV file
        with open(self.filtered_users_view.file_path(self), 'w', newline='', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["rank", "user_id"])
            for rank, user_id in enumerate(self.user_threshold_mapping):
                csv_writer.writerow([rank, user_id])
        
        self.threshold_stats = ThresholdStats(len(self.user_threshold_mapping), len(self.song_threshold_mapping), total_filtered_listens, self.support_threshold)

        # Save the threshold stats to a CSV file
        with open(self.threshold_stats_view.file_path(self), 'w', newline='', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["num_users", "num_songs", "num_listens", "min_listens"])
            csv_writer.writerow(self.threshold_stats)

        # Save the filtered listens to a new CSV file, replacing song ids with their new ids after thresholding, and applying user id permutation according to the user threshold mapping
        with open(self.filtered_listens_view.file_path(self), 'w', newline='', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["user_rank", "song_rank"])
            for row in self.full_listens_view.stream(self):
                user_id = int(row[0])
                song_id = int(row[1])
                if song_id in self.inverse_song_threshold_mapping and user_id in self.inverse_user_threshold_mapping:
                    new_song_id = self.inverse_song_threshold_mapping[song_id]
                    new_user_id = self.inverse_user_threshold_mapping[user_id]
                    csv_writer.writerow([new_user_id, new_song_id])

        return self.threshold_stats

    def ensure_user_threshold_mapping_loaded(self):
        """Ensures that the user threshold mapping is loaded into memory, either from a previous step or by reading from the filtered users file."""
        if self.user_threshold_mapping is None:
            if self.filtered_users_view.is_present(self):
                print("User threshold mapping already exists. Reading from file.")
                self.user_threshold_mapping = []
                self.inverse_user_threshold_mapping = {}
                for row in self.filtered_users_view.stream(self):
                    user_id = int(row[1])
                    self.inverse_user_threshold_mapping[user_id] = len(self.user_threshold_mapping)
                    self.user_threshold_mapping.append(user_id)

    def ensure_song_threshold_mapping_loaded(self):
        """Ensures that the song threshold mapping is loaded into memory, either from a previous step or by reading from the filtered songs file."""
        if self.song_threshold_mapping is None:
            if self.filtered_songs_view.is_present(self):
                print("Song threshold mapping already exists. Reading from file.")
                self.song_threshold_mapping = []
                self.inverse_song_threshold_mapping = {}
                for row in self.filtered_songs_view.stream(self):
                    song_id = int(row[1])
                    self.inverse_song_threshold_mapping[song_id] = len(self.song_threshold_mapping)
                    self.song_threshold_mapping.append(song_id)

    def build_embeddings(self):
        """Builds user and song embeddings using matrix factorisation on the filtered listens data."""
        if self.user_embeddings_view.is_present(self) and self.song_embeddings_view.is_present(self):
            print("User and song embeddings already exist. Skipping factorisation step.")
            return
        num_nodes = self.threshold_stats.num_users + self.threshold_stats.num_songs
        # Create the edge matrix of the bipartite graph
        # Users are assigned node IDs from 1...N-1 and songs from N upward
        edge_coords = self.threshold_stats.create_edge_array(self.filtered_listens_view.stream(self))
        edge_matrix = sparse.coo_matrix((edge_coords[:, 2], (edge_coords[:, 0], edge_coords[:, 1])), shape=(num_nodes, num_nodes))
        edge_matrix = edge_matrix.tocsr()
        del edge_coords  # Free up memory by deleting the edge coordinates array, which is no longer needed after constructing the sparse matrix
        # Now convert the neighbourhoods of users & songs in the graph to vectors in a continuous space
        embedding = sklearn.manifold.SpectralEmbedding(n_components=10, random_state=42, affinity='precomputed', n_jobs=-1, eigen_solver='amg')
        transform = embedding.fit_transform(edge_matrix)        
        del edge_matrix  # Free up memory by deleting the edge matrix, which is no longer needed after computing the embeddings
        self.save_user_embeddings(transform[:self.threshold_stats.num_users, :])
        self.save_song_embeddings(transform[self.threshold_stats.num_users:, :])            

    def save_user_embeddings(self, matrix):
        self.keep_user_hashes = True
        self.ensure_user_threshold_mapping_loaded()
        with open(self.user_embeddings_view.file_path(self), 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(self.user_embeddings_view.heading)
            for user_rank in range(self.threshold_stats.num_users):
                user_id = self.user_threshold_mapping[user_rank]
                user_hash = self.lookup_user_hashes([user_id])[0]
                embedding_values = matrix[user_rank].tolist()
                w.writerow([user_rank, user_id, user_hash] + embedding_values)

    def save_song_embeddings(self, matrix):
        self.keep_song_names = True
        self.ensure_song_threshold_mapping_loaded()
        with open(self.song_embeddings_view.file_path(self), 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(self.song_embeddings_view.heading)
            for song_rank in range(self.threshold_stats.num_songs):
                song_id = self.song_threshold_mapping[song_rank]
                artist_name, song_title = self.lookup_song_names([song_id], True)[0]
                embedding_values = matrix[song_rank].tolist()
                w.writerow([song_rank, song_id, artist_name, song_title] + embedding_values)

    def cluster(self, num_clusters: int = 25):
        """Performs clustering on the user and song embeddings using KMeans."""
        if not self.user_embeddings_view.is_present(self) or not self.song_embeddings_view.is_present(self):
            print("User and song embeddings not found. Please run build_embeddings() before clustering.")
            return
        kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        # Load the user and song embeddings from the CSV files
        user_embeddings = pd.read_csv(self.user_embeddings_view.file_path(self))
        song_embeddings = pd.read_csv(self.song_embeddings_view.file_path(self))
        # Ensure the embeddings are in the correct order according to user_rank and song_rank respectively
        user_embeddings = user_embeddings.set_index('user_rank').sort_index()
        song_embeddings = song_embeddings.set_index('song_rank').sort_index()
        # Extract the embedding values as numpy arrays for clustering
        user_embedding_values = user_embeddings.loc[:, 'embedding_0':]
        song_embedding_values = song_embeddings.loc[:, 'embedding_0':]
        user_kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        song_kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        user_classes = user_kmeans.fit_predict(user_embedding_values)
        song_classes = song_kmeans.fit_predict(song_embedding_values)
        user_embeddings['class'] = user_classes.tolist()
        song_embeddings['class'] = song_classes.tolist()
        return user_embeddings, song_embeddings

    def count_matches(self):
        """Counts the number of listens for each (user class, song class) pair, which can be used for generating the heatmap in the dashboard."""
        user_embeddings_frame, song_embeddings_frame = self.cluster()
        user_class_mapping = user_embeddings_frame.set_index('user_id')['class'].to_dict()
        song_class_mapping = song_embeddings_frame.set_index('song_id')['class'].to_dict()
        match_counts: dict[tuple[int, int], int] = {}
        for row in self.filtered_listens_view.stream(self):
            user_id = int(row[0])
            song_id = int(row[1])
            if user_id in user_class_mapping and song_id in song_class_mapping:
                user_class = user_class_mapping[user_id]
                song_class = song_class_mapping[song_id]
                match_counts[(user_class, song_class)] = match_counts.get((user_class, song_class), 0) + 1
        return match_counts

    def add_user_and_song_class_labels(self):
        """Adds class labels for the user and song embeddings based on the clustering results, and saves the updated data to new CSV files."""
        user_embeddings_frame, song_embeddings_frame = self.cluster()
        # Write a copy of the filtered users and songs files with an additional column for the class labels, which can be used for the dashboard to color users and songs by their assigned class
        user_embeddings_frame.to_csv(self.user_classes_view.file_path(self))
        song_embeddings_frame.to_csv(self.song_classes_view.file_path(self))

    def status(self) -> DataFrame:
        """Returns a DataFrame containing the current status of the all views
        including which files are present on disk, and the dataset statistics if available.
        """
        view_status = []
        for view in self.views:
            view_status.append({
                "view": type(view).__name__,
                "files_present": view.is_present(self),
                "file_paths": view.file_path(self)
            })
        return DataFrame(view_status)
