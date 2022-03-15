# custom handler file

from ts.torch_handler.base_handler import BaseHandler
import numpy as np
import torch
import os
import logging

logger = logging.getLogger(__name__)
class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    # def __init__(self):
    #     self._context = None
    #     self.initialized = False
    #     self.explain = False
    #     self.target = 0

    # def initialize(self, context):
    #     """
    #     Initialize model. This will be called during model loading time
    #     :param context: Initial context contains model server system properties.
    #     :return:
    #     """
    #     self.manifest = context.manifest

    #     properties = context.system_properties
    #     model_dir = properties.get("model_dir")
    #     from model import MatrixFactorization
    #     # model = MatrixFactorization()

    #     # # Read model serialize/pt file
    #     serialized_file = self.manifest['model']['serializedFile']
    #     model_pt_path = os.path.join(model_dir, serialized_file)
    #     model = torch.load('/Users/changruimeng/Workspace/MovieRecommendation/model/model.pt')
    #     self.initialized = True

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Transform user id to index
        input = data[0].get("user_id")
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
        # if model_input not in self.model.user_map.keys():
        #     return None
        # user_idx = self.model.user_map[model_input]
        # user_idxs = np.ones(self.model.n_movies) * user_idx
        # movie_idxs = torch.tensor(np.arange(self.model.n_movies))
        # data = np.hstack([user_idxs, movie_idxs]).T
        # model_output = self.model.forward(data)
        # return model_output
        return None

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        return inference_output

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