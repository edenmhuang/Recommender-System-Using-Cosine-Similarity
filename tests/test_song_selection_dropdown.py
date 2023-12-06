#!/usr/bin/env python
# coding: utf-8


# Import Required Libraires
import pandas as pd
import ipywidgets as widgets



# Configure Dataset
data = pd.read_csv('Best Songs on Spotify from 2000-2023.csv', sep = ";")
df = pd.DataFrame(data)
df = df.drop_duplicates(subset=["title"],keep="first")


df['title'] = df['title'].str.lower()
df['artist'] = df['artist'].str.lower()


# Enter the name of the artist
select_artist = str(input('Enter Artist: ')).lower()

# Check if the artist exists in the DataFrame
if df['artist'].str.contains(select_artist).any():
    # Filter the DataFrame for the selected artist
    selected_artist = df['artist'] == select_artist
    # New DataFrame with columns 'title', 'artist', and 'year'
    show_artist_song = df[selected_artist][['title', 'artist', 'year']]
    # Display the first 5 songs of the Artist
    print(show_artist_song.head(show_artist_song.shape[0]))
else:
    # Display message if Artist is not found in dataset
    print('Artist not in database')


select_title = widgets.Dropdown(options = list(show_artist_song['title']), description = "select song")
select_title

select_title = select_title.value
select_song = show_artist_song['title'] == select_title
selected_song = show_artist_song[select_song]
selected_song




