import warnings

# Filter out the warning message for ShardedTensor
warnings.filterwarnings("ignore", category=UserWarning, message="Please use DTensor instead and we are deprecating "
                                                                "ShardedTensor.")

from model.snn import train_snn
import click
import yaml
from model.supervised_encoder import train_encoder


@click.command()
@click.option("--config-path", type=click.Path(exists=True), required=True, help="Path to training configuration")
@click.option("--model", type=str, required=True, help="Model to train: snn or encoder")
def main(config_path: click.Path, model: str):
    """
    Training script.
    """
    with open(str(config_path), "r") as config_file:
        config = yaml.safe_load(config_file)

    if model == "snn":
        train_snn(config)
    elif model == "encoder":
        train_encoder(config)
    else:
        raise NotImplementedError(f"There is no model {model}")


if __name__ == "__main__":
    main()
