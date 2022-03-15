# model definition
import torch

class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_movies, user_map, movie_map, n_factors=20):
        super().__init__()
	      # create user embeddings
        self.user_factors = torch.nn.Embedding(n_users, n_factors,
                                               sparse=True)
	      # create item embeddings
        self.movie_factors = torch.nn.Embedding(n_movies, n_factors,
                                               sparse=True)
        # save model context
        self.n_users = n_users
        self.n_movies = n_movies
        self.user_map = user_map
        self.movie_map = movie_map

    def forward(self, data):
        user, movie = data[:,0], data[:,1]
    	  # matrix multiplication
        return (self.user_factors(user)*self.movie_factors(movie)).sum(1)

    def predict(self, user, movie):
        return self.forward(user, movie)