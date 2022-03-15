import requests

req = requests.post("http://localhost:8080/predictions/movie_model", json={'user_id':99774})
print(req.text)