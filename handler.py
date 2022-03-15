# custom handler file

from ts.torch_handler.base_handler import BaseHandler
import numpy as np
import torch
import os
import logging
import pickle

logger = logging.getLogger(__name__)


class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")

        from model import MatrixFactorization
        # load model dic
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        state_dict = torch.load(model_pt_path)
        
        # load extra data
        mapping_file = os.path.join(model_dir, "mapping.pkl")   
        with open(mapping_file, 'rb') as f:
            dic = pickle.load(f)
        self.user_map = dic['user_map']
        self.movie_map = dic['movie_map']
        self.users = dic['users']
        self.movies = dic['movies']
        n_users, n_movies = len(self.user_map), len(self.movie_map)
        self.n_users = n_users
        self.n_movies = n_movies

        # load model
        self.model = MatrixFactorization(n_users, n_movies)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.initialized = True

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Transform user id to index
        input = data[0].get("data")
        if input is None:
            input = data[0].get("body")
        user_id = input['user_id']
        return user_id


    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # if user not exist, return null
        if model_input not in self.user_map.keys():
            return None
        # get user index
        user_idx = self.user_map[model_input]
        # construct combination of user with all movies
        user_idxs = np.ones(self.n_movies) * user_idx
        movie_idxs = np.arange(self.n_movies)
        data = torch.tensor(np.vstack([user_idxs, movie_idxs]).T, dtype=torch.long)
        # predict rating for all movies
        with torch.no_grad():
            model_output = self.model.forward(data)
        return model_output

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        # sort the ratings, return indexes in descending order
        ratings = inference_output.numpy()
        inds = np.argsort(ratings)[::-1]
        logger.log(logging.INFO, inds)
        logger.log(logging.INFO, ratings[inds])
        # get name for 5 highest rating movie
        movie_inds = inds[:5]
        movie_names = [self.movies[i] for i in movie_inds]
        return [movie_names]

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)