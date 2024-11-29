### Movie-Review-Clustering
Movie review grouping using K-Means Clustering 

## Objective
The goal of this assignment is to cluster movie reviews based on ratings and evaluate whether movies within the same rating clusters share the same genre. The K-means clustering algorithm is implemented using Numpy from scratch to group the reviews into meaningful clusters. Additionally, an analysis is performed to compare movie genres within these clusters and evaluate how well the clustering aligns with predefined genres.

## Dataset
IMDB-Movie-Data.csv
The dataset used in this project is IMDB Movie Reviews, which includes the following features:

- Year: Release year of the movie.
- Runtime: Duration of the movie in minutes.
- Rating: IMDb rating of the movie.
- Votes: Number of votes the movie has received.
- Revenue (Millions): Revenue generated by the movie (in millions).
- Meta Score: The meta score assigned to the movie.

The dataset is used to explore the relationship between movie ratings and genres, and how K-means clustering can be applied to group movies based on their rating scores.

# Part 1: Data Preprocessing and Feature Extraction
Rating Scale for Clustering
We first categorize the movie ratings into three clusters:

Top Rated: Movies with a rating ≥ 8
Average Rated: Movies with a rating between 6 and 7
Low Rated: Movies with a rating < 6
These clusters are used to form the basis for K-means clustering.

# Part 2: Implement K-means Clustering from Scratch
1. Cluster Reviews Based on Rating Scores
The K-means algorithm is implemented using Numpy to group movie reviews based on their ratings. The algorithm iterates through the following steps:

- Initialization: Randomly select initial centroids for each cluster.
- Assignment: Assign each movie to the nearest centroid based on the rating.
- Update: Calculate the new centroids by averaging the ratings of the movies assigned to each cluster.
- Convergence: Check for convergence by comparing the old centroids with the new centroids. The algorithm stops when the centroids no longer change.
2. Cluster Label Assignment
Each movie is assigned to one of the three clusters: Top Rated, Average Rated, or Low Rated based on the K-means clustering result.

# Part 3: Genre Analysis and Model Evaluation
1. Compare Genres Within Clusters
We analyze whether movies within the same cluster belong to the same genre. By comparing the genres of movies in each cluster, we can determine if the rating-based clusters align with the genres.

2. Evaluation Metrics
We generate a confusion matrix to compare the cluster ratings (result of K-means) with the original ratings to evaluate the performance of the clustering algorithm. The matrix provides insights into how well the clustering aligns with the actual ratings.

# Part 4: Visualize the Clusters
Using matplotlib, the clusters are visualized by plotting the movie ratings in a 2D space. Each cluster is represented by a different color, allowing us to see how the movies are grouped according to their rating scores.

## Installation and Usage
To run this project, ensure that you have the following dependencies installed:

- Python 3.x
- Numpy: pip install numpy
- Matplotlib: pip install matplotlib
- Pandas: pip install pandas

# Running the Project
- Clone this repository to your local machine.
- Place the IMDB-Movie-Data.csv dataset in the project directory.
- Run the script movie_review_clustering.py to preprocess the data, implement K-means clustering, and generate the evaluation metrics.

## Conclusion
This project demonstrates how K-means clustering can be used to group movies based on ratings and analyze the alignment of these clusters with movie genres. The results provide insights into how clustering algorithms can be used for data analysis in the context of movie reviews.