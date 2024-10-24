
import datetime
import argparse

import lightning as L
from torch import nn
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from src.modules.train_module import TrainModule
from src.modules.data_module import DataModule
from src.utils.misc import dict_to_namespace
from src.callbacks.gradient_callback import GradientLoggerCallback

import config
import yaml
import importlib
from pathlib import Path

from types import SimpleNamespace as NameSpace

def import_target(target : str) -> any:
    module_name, class_name = target.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
    
def get_loss_functions(args : NameSpace) -> tuple[list[nn.Module], list[float]]:

    weights = []
    functions = []
    for info in  args.trainmodule.loss_functions.Values():
        loss = import_target(info.target)
        weights.append(loss(**vars(info.params)))
        functions.append(info.weight)
        
    return functions, weights

def get_model(args : NameSpace) -> nn.Module:
    model = import_target(args.model.target)
    return model(**vars(args.model.params))



def train(params : NameSpace, model_params : NameSpace) -> None:
    
    # Define the dataset and the data module
    dataset = import_target(params.dataset.target)
    data_module = DataModule(dataset = dataset,
                             dataset_params = params.dataset,
                             dataloader_params = params.dataloader)
    
    # Define the loss function, model, and training module
    loss_functions, loss_weights = get_loss_functions(model_params)
    model = get_model(model_params)
    train_module = TrainModule(model = model,
                               loss_functions = loss_functions,
                               loss_weights = loss_weights,
                               optimizer_params = params.optimizer)
    
    suffix = datetime.datetime.now().strftime(params.identifier.time_format)
    identifier = f"{params.identifier.name}-{suffix}"
    
    # Logger
    logger = TensorBoardLogger(Path(params.logging.dir),
                               name=f"identifier")
    
    # Define the callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss', 
        dirpath=params.checkpointing.dir,
        filename=identifier,
        save_top_k=params.checkpointing.save_top_k,
        mode=params.checkpointing.min,
    )
    
    callbacks = [lr_monitor, checkpoint_callback]
    
    if params.logging.log_gradients.use:
        gradient_callback = GradientLoggerCallback(log_every_n=params.logging.log_gradients.every_n,
                                                   step_type=params.logging.log_gradients.step_type)
        callbacks.append(gradient_callback)
    
    # Gradient clipping
    if params.training.gradient_clipping.use:
        gradient_clip_val = params.training.gradient_clipping.value
        gradient_clip_algorithm = params.training.gradient_clipping.algorithm
    else:
        gradient_clip_val = None
        gradient_clip_algorithm = None
    
    trainer = L.Trainer(accelerator=params.accelerator,
                        callbacks=callbacks,
                        max_epochs=params.training.epochs,
                        log_every_n_steps=params.logging.log_every_n_steps,
                        accumulate_grad_batches=params.training.accumulate_grad_batches,
                        gradient_clip_val=gradient_clip_val,
                        gradient_clip_algorithm=gradient_clip_algorithm,
                        logger=logger)
    
    trainer.fit(train_module, datamodule=data_module)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the resnet model on the ear dataset.")
    parser.add_argument('--config', type=str,
                        default="config/config.yaml",
                        help='Path to the configuration file.')
    parser.add_argument('--model_config', type=str,
                        default="config/model_config.yaml",
                        help='Path to the model configuration file.')
    
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_args = yaml.safe_load(f)

    with open(args.model_config, 'r') as f:
        model_config_args = yaml.safe_load(f)
        
    assert config_args is not None, "Configuration file is empty."
    assert model_config_args is not None, "Model configuration file is empty."

    params = dict_to_namespace(config_args)
    model_params = dict_to_namespace(model_config_args)

    train(params, model_params)