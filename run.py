from config import Conf as config
from train import train_the_model
from fetch_inference_data import fetch_inference_batch as fib

if not config.mode:
    train_the_model()

if config.mode:
    fib()
