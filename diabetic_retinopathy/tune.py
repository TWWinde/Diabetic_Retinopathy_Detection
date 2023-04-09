import logging
import gin

import ray
from ray import tune

from input_pipeline.datasets import load
from models.architectures import vgg_like
from train import Trainer
from utilss import utils_params, utils_misc


def train_func(config):
    # Hyperparameters
    bindings = []
    for key, value in config.items():
        bindings.append(f'{key}={value}')

    # generate folder structures
    run_paths = utils_params.gen_run_folder(','.join(bindings))

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(
        ['/home/RUS_CIP/st180408/dl-lab-22w-team08/diabetic_retinopathy_2/configs/config.gin'],
        bindings)  # change path to absolute path of config file
    # utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = load()

    # model
    model = vgg_like(input_shape=ds_info.input_shape, n_classes=ds_info.n_classes)

    trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
    for val_accuracy in trainer.train():
        tune.report(val_accuracy=val_accuracy)


ray.init(num_cpus=12, num_gpus=1)
analysis = tune.run(
    train_func, num_samples=12, resources_per_trial={"cpu": 12, "gpu": 1},
    config={
        "Trainer.total_steps": tune.grid_search([2 * 1e4]),
        # "vgg_like.base_filters": tune.choice([8, 16]),
        # "vgg_like.n_blocks": tune.choice([6, 7, 8]),
        # "vgg_like.dense_units": tune.choice([32, 64]),
        # "vgg_like.dropout_rate": tune.uniform(0, 0.9),
        "Trainer.alpha": tune.choice([0.25, 0.5, 0.75]),
        "Trainer.gamma": tune.choice([0.5, 1, 2, 5]),

    })

print("Best config: ", analysis.get_best_config(metric="val_accuracy", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.dataframe()
