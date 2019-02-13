# do NOT keep the key publicly visible
import os
comet_key = os.getenv('COMET_KEY') or input('Enter Comet API key: ')

# import comet_ml in the top of your file
from comet_ml import Experiment, Optimizer

experiment = Experiment(api_key=comet_key, project_name="diabetic-retinopathy", workspace="rozni")
optimizer = Optimizer(api_key=comet_key)


