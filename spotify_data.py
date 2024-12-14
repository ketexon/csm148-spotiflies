import pandas as pd

data = pd.read_csv('spotify.csv')
grouped_data = pd.read_csv('grouped_cleaned_spotify.csv')

string_columns = ['track_id', 'artists', 'album_name', 'track_name']
categorical_columns = ['key', 'mode', 'time_signature', 'track_genre']