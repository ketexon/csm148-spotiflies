import pandas as pd

data = pd.read_csv('spotify.csv')
grouped_data = pd.read_csv('grouped_cleaned_spotify.csv')

string_columns = ['track_id', 'artists', 'album_name', 'track_name']
categorical_columns = ['key', 'mode', 'time_signature', 'track_genre']

# genre_duration = grouped_data.groupby('track_genre')['duration_ms'].sum()
# genre_count = grouped_data.groupby('track_genre')['track_id'].count()
# genre_duration_seconds = genre_duration / genre_count / 1000
# genre_duration_minutes_seconds = genre_duration_seconds.apply(lambda x: f"{int(x // 60)}:{int(x % 60):02d}")
# print(genre_duration_minutes_seconds)
