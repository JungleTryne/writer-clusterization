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
def main(config_path: click.Path):
    """
    Training script.
    """
    with open(str(config_path), "r") as config_file:
        config = yaml.safe_load(config_file)

    if config["model"] == "snn":
        train_snn(config)
    elif config["model"] == "encoder":
        train_encoder(config)
    else:
        raise NotImplementedError(f"There is no model {config['model']}")


if __name__ == "__main__":
    main()
