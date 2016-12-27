import numpy as np
import keras
import maxflow

def train_unary_model(images, gt):
    return {}

def segmentation(unary_model, images):
    return [np.zeros(img.shape[:2]) for img in images]
