import csv
import numpy as np

def decoded_line_generator(file):
    # This CSV file has already been cleaned so we trust that all quotes and commas are properly escaped.
    # Also we don't have header rows in the intermediary files.
    for line in file:
        line_str = line.decode('utf-8')
        yield line_str

def extract_top_songs(listens_path, songs_path, popular_listens_path, popular_songs_path, min_listens=20):
    # Pass 1: find the maximum song id to determine the size of the count array
    max_id = -1
    with open(listens_path, 'rb') as f:
        csv_reader = csv.reader(decoded_line_generator(f))
        for row in csv_reader:
            song_id = int(row[1])
            if song_id > max_id:
                max_id = song_id
    num_songs = max_id + 1
    # Pass 2: count the number of occurrences of each song id
    song_counts = np.zeros((num_songs, 2), dtype=np.uint32)
    song_counts[:, 0] = np.arange(num_songs, dtype=np.uint32) # Keep a copy of the ids to identify songs after sorting
    with open(listens_path, 'rb') as f:
        csv_reader = csv.reader(decoded_line_generator(f))
        for row in csv_reader:
            song_id = int(row[1])
            song_counts[song_id, 1] += 1
    # Sort by count, most popular first
    song_counts = song_counts[song_counts[:, 1].argsort()[::-1]]
    # construct a dictionary mapping song id to rank
    song_rank = {row[0]: (rank, row[1]) for rank, row in enumerate(song_counts) if row[1] >= min_listens}
    # Pass 3: extract the artist and title for the most popular songs
    with open(songs_path, 'rb') as f_in, open(popular_songs_path, 'w', newline='', encoding='utf-8') as f_out:
        csv_reader = csv.reader(decoded_line_generator(f_in))
        csv_writer = csv.writer(f_out)
        for row in csv_reader:
            song_id = int(row[0])
            if song_id in song_rank:
                rank, count = song_rank[song_id]
                artist = row[1]
                title = row[2]
                csv_writer.writerow([rank, count, song_id, artist, title])
    # Pass 4: create a filtered version of the listens file
    with open(listens_path, 'rb') as f_in, open(popular_listens_path, 'w', newline='', encoding='utf-8') as f_out:
        csv_reader = csv.reader(decoded_line_generator(f_in))
        csv_writer = csv.writer(f_out)
        for row in csv_reader:
            user_id = row[0]
            song_id = int(row[1])
            if song_id in song_rank:
                rank, count = song_rank[song_id]
                csv_writer.writerow([user_id, rank, song_id])
