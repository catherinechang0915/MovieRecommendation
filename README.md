# Serve Recommendation Models with TorchServe

This is the code base for TorchServe example in Movie Recommendation Scenario. It provides ready-to-use matrix factorization recommendation model and instructions for model serving. The details of the example are available [here](https://medium.com/@catherine.chang0915/serve-recommendation-models-with-torchserve-8723b1472aed)

### Structure
```bash
.
├── README.md
├── data
│   ├── mapping.pkl
│   └── movie_data.csv
├── handler.py
├── model.py
├── model_store
│   └── movie_model.mar
├── models
│   └── model_dic.pth
├── movie_recommendation.ipynb
└── test.py
```

##### data
`movie_data.csv`: movie rating data with format (user_idx, movie_idx, rating)
`mapping.pkl`: mapping between index to metadata

##### model_store
Directory to store archived models

##### model.py
Movie Recommendation model definition.

##### handler.py
Customized handler file for creating archived `.mar` file. Instruct TorchServe how to deal with the input and output of the endpoint for model inference.

##### movie_recommendation.ipynb
Detailed data preprocessing, PyTorch model training and saving process.

### Installation

```bash
pip install torch torchvision 
pip install torchserve torch-model-archiver torch-workflow-archiver
```

### How to use

The archived model is available at `model_store/model_store.mar`.

##### Start TorchServe

```bash
torchserve --start --model-store model_store --models movie_model.mar
```

##### Run Inference

```bash
curl -X POST http://localhost:8080/predictions/movie_model \
-H 'Content-Type: application/json' \
-d '{"user_id":"99774"}'
```

##### Stop TorchServe

```bash
torchserve --stop
```
