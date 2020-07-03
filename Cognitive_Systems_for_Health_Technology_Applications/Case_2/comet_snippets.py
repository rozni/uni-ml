# do NOT keep the key publicly visible
import os
from getpass import getpass

COMET_KEY = os.getenv('COMET_KEY') or getpass('Enter Comet API key: ')

# import comet_ml in the top of your file
from comet_ml import Experiment, Optimizer

def new_experiment():
    return Experiment(
        api_key=COMET_KEY,
        project_name="diabetic-retinopathy",
        workspace="rozni"
    )

optimizer = Optimizer(api_key=COMET_KEY)