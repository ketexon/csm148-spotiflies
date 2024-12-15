# Exploration, Grouping, and Prediction of Genre in Spotify Songs
## Team Spotiflies: Joanna, Aaron, Aubrey, Kennedy, Aster, Ethan
- [GitHub Link](https://github.com/ketexon/csm148-spotiflies)

### I
In this project, we predict song genre using a [massive dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) of __over 140 thousand__ songs from the music streaming platform, Spotify. Each song has information about its popularity, genre, and several other musical qualities. We tried to classify a song into one of 10 genres by using up to 12 different predictor variables.

- Relevant Predictor Variables: popularity, duration_ms, explicit, danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo.
- Genres: classical, electronic, folk, hip-hop, jazz, metal, misc, pop, rock, world

### II
The main feature we're interested in with this dataset is __song genre__, and how it relates to other features. We want to see if the musical qualities of a song are strong influences of what genre it's categorized. After cleaning, our dataset has 114 different genres, ranging from emo to classical to industrial and more. We are interested in answering questions regarding:
- what qualities of songs correlate to genre
- if certain genres are more similar to each other than others
- if we can predict a song's genre through its other qualities
- and much, much more.

Understanding these relationships is crucial for several reasons - first, accurate genre prediction can _enhance recommendation systems_, providing users with more personalized and enjoyable listening experiences. For artists and producers, insights from our analysis can _guide the creative process_, helping them understand trends and preferences within different genres. Finally, we contribute to the broader field of musicology by providing _data-driven insights_ into the characteristics that define and differentiate musical genres.

### III
We try numerous different methodologies, including __linear regression__, __logistic regression__, __decision trees__, __PCA/clustering__, and __neural networks__. Our two key methodologies were __Neural Networks__ and __Decision Trees__ - these worked most effectively to predict a song's genre through its other qualities. Our most effective methodologies were __Neural Networks__, __Decision Trees__, and __Random Forest__ - these worked most effectively to predict a song's genre through its other qualities.

Using Neural Networks, we were able to capture the complex relationships and variance between songs, making them ideal for effective predictions. However, due to insufficient computing resources for accurate training, we were unable to achieve an acceptable `F1 score`.

In contrast, with Decision Trees, we were able to obtain clear insights into the most important features correlating with genre, and obtained respectable accuracy compared to our other methods. Finally, using an ensemble of trees in a __Random Forest__ as our key methodology, we achieved much more robust and accurate predictions than ever before.

### IV
Using a train-test split of `20:80` and an ensemble of 100 trees each with a maximum tree depth of `20`, we were able to achieve a total __Accuracy__ of `0.667`. We generated a __Classification Report__, showing the precision, recall, and F1-score for each genre, which helped us understand the performance of our model across different genres. In addition, we generated a __Confusion Matrix__, which provided a visual representation of the true positive, true negative, false positive, and false negative predictions, allowing us to see where our model was making errors.

Finally, we used both __Feature & Permutation Importance__ to identify which predictor variables had the most significant impact on genre classification. In particular, we found that `popularity` and `duration_ms` were the most correlated. This makes sense, as popular songs tend to have certain characteristics that make them more appealing to a wide audience, and the duration of a song can influence its genre classification, with certain genres typically having longer or shorter average song lengths (For example, the average length of a `rock` song was 3:34, while the average length of a `hip-hop` song was 4:09).

While our Random Forest model provided robust and accurate predictions, it also had some limitations. One significant limitation was the model's tendency to overfit, especially with a large number of trees and depth. We tried to combat this by limiting the *maximum tree depth*, which resulted in lower accuracy. In addition, the computational resources required for training and tuning the model were substantial (each run took dozens of minutes) which could be a constraint for larger datasets or more extensive hyperparameter optimization. 

All in all, our maximum accuracy achieved was still well below `0.7`, which may be a limitation of our dataset itself - this is because our dataset may not capture all the nuances and variations within each genre. Additionally, the subjective nature of genre classification means that there can be overlap and ambiguity between genres, making it challenging to achieve high accuracy. Future work could involve refining the genre labels, incorporating more diverse and representative data, and experimenting with different model architectures and techniques to improve performance.

Despite these limitations, our project demonstrates the potential of machine learning techniques in predicting song genres and provides a foundation for further exploration and improvement. Future work could involve exploring additional features, using more advanced models, or leveraging larger datasets to enhance prediction accuracy and generalizability.

### V
To run the code in __project_code.ipynb__, ensure you have Jupyter Notebook or Jupyter Lab installed along with the necessary dependencies. First, clone this repository to your local machine and navigate to the project directory. Launch Jupyter Notebook by running `jupyter notebook` in your terminal, open __project_code.ipynb__, and execute the cells sequentially. Alternatively, use the `Jupyter Notebook` extension and "run all cells".