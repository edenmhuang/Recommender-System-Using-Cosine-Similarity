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

