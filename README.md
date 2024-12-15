# Exploration, Grouping, and Prediction of Genre in Spotify Songs
## Team Spotiflies: Joanna, Aaron, Aubrey, Kennedy, Aster, Ethan
- [GitHub Link](https://github.com/ketexon/csm148-spotiflies)

### I
In this project, we predict song genre using a [massive dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) of __over 140 thousand__ songs from the music streaming platform, Spotify. Each song has information about its popularity, genre, and several other musical qualities. 

### II
The main feature we're interested in with this dataset is __song genre__, and how it relates to other features. We want to see if the musical qualities of a song are strong influences of what genre it's categorized. After cleaning, our dataset has 114 different genres, ranging from emo to classical to industrial and more. We are interested in answering questions regarding:
- what qualities of songs correlate to genre
- if certain genres are more similar to each other than others
- and much, much more.

Understanding these relationships is crucial for several reasons - first, accurate genre prediction can _enhance recommendation systems_, providing users with more personalized and enjoyable listening experiences. For artists and producers, insights from our analysis can _guide the creative process_, helping them understand trends and preferences within different genres. Finally, we contribute to the broader field of musicology by providing _data-driven insights_ into the characteristics that define and differentiate musical genres.

### III
We try numerous different methodologies, including linear regression, logistic regression, decision trees, PCA/clustering, and neural networks. However, the single most effective methodology for analysis was __d__
