# import requests

# req = requests.post("http://localhost:8080/predictions/movie_model", json={'user_id':99774})
# print(req.text)

import torch
from factor_model import MatrixFactorization

model = torch.load('/Users/changruimeng/Workspace/MovieRecommendation/model/model.pt')