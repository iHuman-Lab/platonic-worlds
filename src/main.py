import pytorch_lightning as pl
import torch
import yaml
from lightning.pytorch.loggers import TensorBoardLogger

from dataset.pre_process import ComposePreprocessor, Resize, StackWithLabels
from dataset.torch_dataset import get_torch_dataloaders
from trainers.conv_jepa import ConvJEPA
from utils import skip_run

# The configuration file
config_path = "configs/config.yaml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

LOG_DIR = config["log_dir"]


with skip_run("skip", "conv_jepa_pretraining") as check, check():
    frames, horizon = int(config["conv_jepa_frames"]), int(config["conv_jepa_horizon"])
    jepa_config = {**config, "stack_length": frames + horizon}  # context + future frames
    game = config["games"]
    games_name = "_".join(game)
    preprocessor = ComposePreprocessor([Resize(jepa_config), StackWithLabels(jepa_config)])
    data_loaders = get_torch_dataloaders(game, jepa_config, preprocessor)

    model = ConvJEPA(
        jepa_config, dim=int(config["conv_jepa_dim"]), ctx=frames, horizon=horizon, num_actions=int(config["num_actions"])
    )
    trainer = pl.Trainer(
        logger=TensorBoardLogger(LOG_DIR, name=f"{games_name}/conv_jepa"),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=int(config["epochs"]),
        log_every_n_steps=10,
    )
    trainer.fit(model, data_loaders["train"])
