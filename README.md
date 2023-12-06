# Recommender-System-Using-Cosine-Similarity



<h2> Goal </h2>
<p> A Python program that reads a CSV file on music data, takes an individual's song choice, and outputs similar songs using cosine similarity. </p>

<h2> Why is this necessary? </h2>
<p> Traditional systems rely on complex algorithms to analyze user behavior and preferences to suggest songs or playlists that align with individual tastes. However, I wanted to research whether I was able to utilize a dataset of songs, each characterized by a set of acoustic features (e.g., bpm, energy, danceability, etc.) and popularity. To recommend songs without the need of user data.
</p>

<h2> What is Cosine Similarity? </h2>
<p> Cosine Similarity is a similarity metric that examines the cosine angle between two vectors in a multi-dimensional space. For example, it assesses the relationship between the music characteristics of two songs and outputs their similarity.
</p>


<h2> CSV file Data description: </h2>
<p></p>
<b>
<ul>
  <li>title - Name of Song;</li>
  <li>artist - Name of the Artist;</li>
  <li>top genre - The genre of the song</li>
  <li>year - The year the song was releasedThe year the song was released</li>
  <li>bpm - BPM (Beats Per Minute). Represents the tempo of the song</li>
  <li>energy - Shows the level of energy in a song. The higher the value, the more energetic a song is. Values range from 0-100</li>
  <li>danceability - The higher the value, the easier it is to dance to this song</li>
  <li>dB - Represents volume/loudness of the song. dB stands for Decibel</li>
  <li>liveness - The higher the value, the more likely the song is a live recording</li>
  <li>valance - Describes the musical positiveness conveyed by a track. Tracks with high valence sound more positive</li>
  <li>duration - The duration of the songs measured in seconds</li>
  <li>acousticness - Measures how acoustic the song is from 0-100. e.g) A song with a higher value for acousticness has less singing</li>
  <li>speechiness - Measures how much singing there is in a song from 0-100</li>
  <li>popularity - Measures how popular a song is on Spotify from 0-100</li>
</ul>
</b>

<h2> Python Libraries </h2>

<h4> Pandas </h4> 
[https://pandas.pydata.org/]

<h4> Matplotlib </h4>  
[https://matplotlib.org/]

<h4> Seaborn </h4>
[https://seaborn.pydata.org/]

<h4> ipywidgets </h4>
[https://ipywidgets.readthedocs.io/]

<h4> Scikit-learn (sklearn) </h4>
[https://scikit-learn.org/stable/]

{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "e7c1defc",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import Required Libraires\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import ipywidgets as widgets\n",
    "from sklearn.metrics.pairwise import cosine_similarity"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "fc99f04a",
   "metadata": {},
   "outputs": [],
   "source": [
    "pd.set_option('display.max_rows', None)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "7461363d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Configure Dataset\n",
    "data = pd.read_csv('Best Songs on Spotify from 2000-2023.csv', sep = \";\")\n",
    "df = pd.DataFrame(data)\n",
    "df = df.drop_duplicates(subset=[\"title\"],keep=\"first\")\n",
    "\n",
    "\n",
    "df['title'] = df['title'].str.lower()\n",
    "df['artist'] = df['artist'].str.lower()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "368dc6ca",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Number of Popular Songs by Top 20 Artists on Spotify from 2000-2023')"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 648x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Count the number of popular songs for each artist and select the top 20\n",
    "popular_songs_count = df['artist'].value_counts().head(20)\n",
    "\n",
    "# Create a new figure for the bar plot with specified dimensions\n",
    "plt.figure(figsize=(9, 6))\n",
    "\n",
    "# Count popular songs on the x-axis and the artist names on the y-axis\n",
    "sns.barplot(x=popular_songs_count.values, y=popular_songs_count.index, palette='deep')\n",
    "\n",
    "# Add a title to the plot\n",
    "plt.title('Number of Popular Songs by Top 20 Artists on Spotify from 2000-2023', fontsize=14)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "c8c0322f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Enter Artist: coldplay\n",
      "                              title    artist  year\n",
      "47                           yellow  coldplay  2000\n",
      "74                     viva la vida  coldplay  2008\n",
      "101             a sky full of stars  coldplay  2014\n",
      "138                   the scientist  coldplay  2002\n",
      "149                         fix you  coldplay  2005\n",
      "151            hymn for the weekend  coldplay  2015\n",
      "161                        paradise  coldplay  2011\n",
      "242                          clocks  coldplay  2002\n",
      "271                     my universe  coldplay  2021\n",
      "437         adventure of a lifetime  coldplay  2015\n",
      "819                     in my place  coldplay  2002\n",
      "923                           magic  coldplay  2014\n",
      "998   every teardrop is a waterfall  coldplay  2011\n",
      "1041                 speed of sound  coldplay  2005\n",
      "1179              princess of china  coldplay  2011\n",
      "1247                   higher power  coldplay  2021\n"
     ]
    }
   ],
   "source": [
    "# Enter the name of the artist\n",
    "select_artist = str(input('Enter Artist: ')).lower()\n",
    "\n",
    "# Check if the artist exists in the DataFrame\n",
    "if df['artist'].str.contains(select_artist).any():\n",
    "    # Filter the DataFrame for the selected artist\n",
    "    selected_artist = df['artist'] == select_artist\n",
    "    # New DataFrame with columns 'title', 'artist', and 'year'\n",
    "    show_artist_song = df[selected_artist][['title', 'artist', 'year']]\n",
    "    # Display the first 5 songs of the Artist\n",
    "    print(show_artist_song.head(show_artist_song.shape[0]))\n",
    "else:\n",
    "    # Display message if Artist is not found in dataset\n",
    "    print('Artist not in database')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "db395192",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a804fee0a3c94661823ad8ab2fda3972",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Dropdown(description='select song', options=('yellow', 'viva la vida', 'a sky full of stars', 'the scientist',…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# 5 hours\n",
    "select_title = widgets.Dropdown(options = list(show_artist_song['title']), description = \"select song\")\n",
    "select_title"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "8a3964d5",
   "metadata": {},
   "outputs": [],
   "source": [
    "select_title = select_title.value\n",
    "select_song = show_artist_song['title'] == select_title\n",
    "selected_song = show_artist_song[select_song]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "cf0dbcbc",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Recommended Songs:\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "      <th>artist</th>\n",
       "      <th>similarity</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>47</th>\n",
       "      <td>yellow</td>\n",
       "      <td>coldplay</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>777</th>\n",
       "      <td>the climb</td>\n",
       "      <td>miley cyrus</td>\n",
       "      <td>0.990563</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>blinding lights</td>\n",
       "      <td>the weeknd</td>\n",
       "      <td>0.987472</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>90</th>\n",
       "      <td>somewhere only we know</td>\n",
       "      <td>keane</td>\n",
       "      <td>0.986391</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1483</th>\n",
       "      <td>i hope you dance</td>\n",
       "      <td>lee ann womack</td>\n",
       "      <td>0.985115</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                       title          artist  similarity\n",
       "47                    yellow        coldplay    1.000000\n",
       "777                the climb     miley cyrus    0.990563\n",
       "19           blinding lights      the weeknd    0.987472\n",
       "90    somewhere only we know           keane    0.986391\n",
       "1483        i hope you dance  lee ann womack    0.985115"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Select Acoustic Features for Cosine Similarity\n",
    "acoustic_features = ['bpm', 'energy', 'danceability ', 'dB', 'liveness', 'valence', 'duration', 'acousticness', 'speechiness ', 'popularity']\n",
    "\n",
    "# Normalize Acoustic Features: Have all points between [-1,1]\n",
    "for feature in acoustic_features:\n",
    "    df[feature] = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())\n",
    "    \n",
    "# Create a vector for the selected song\n",
    "selected_song_vector = df[(df['artist'] == select_artist) & (df['title'] == select_title)][acoustic_features].values\n",
    "\n",
    "# Calculate Cosine Similarity\n",
    "df['similarity'] = cosine_similarity(df[acoustic_features], selected_song_vector)\n",
    "\n",
    "# Sort by Similarity and Recommend similar songs\n",
    "recommended_songs = df.sort_values(by='similarity', ascending=False)[['title', 'artist', 'similarity']]\n",
    "\n",
    "# Print recommended songs\n",
    "print(\"Recommended Songs:\")\n",
    "recommended_songs.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "585aab01",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
