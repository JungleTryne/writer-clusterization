import click

import sys
[sys.path.append(i) for i in ['.', '..']]

from dataset.cvl_resized_dataset import CVLResizedDataset
from dataset.iam_resized_dataset import IAMResizedDataset


@click.command()
@click.option("--dataset-type", type=str, help="Dataset type: [iam, cvl]")
@click.option("--root", type=click.Path(exists=True), help="Path to dataset root")
@click.option("--cut", type=int, help="Final number of samples")
def main(dataset_type: str, root: str, cut: int):
    if dataset_type == "iam":
        dataset = IAMResizedDataset(root, cut=cut)
    elif dataset_type == "cvl":
        dataset = CVLResizedDataset(root, cut=cut)
    else:
        raise Exception("Invalid dataset")
    print(dataset.number_of_authors())


if __name__ == "__main__":
    main()

