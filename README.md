# Movie Recommendation System

Collaborative filtering on the [MovieLens](https://grouplens.org/datasets/movielens/) ratings dataset with fastai.

- `movie_recommendation_system.ipynb`: dot-product model with user and movie embeddings and biases, plus analysis of
  the learned embeddings
- `movie_recommendation_system_using_ml.ipynb`: neural-network collaborative filtering (`CollabDataLoaders`,
  embedding-NN), tuned with weight decay and layer sizes

Movies with fewer than 5 ratings are filtered out before training.

## Setup

Download MovieLens (`movies.csv`, `ratings.csv`) into `data/`, then:

```bash
pip install fastai pandas
jupyter notebook
```
