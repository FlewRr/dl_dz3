from pathlib import Path
from retrieval.trainable import RetrievalTrainable
from retrieval.config import RetrievalConfig
from retrieval.transformer_retriever import TransformerRetriever, RetrievalModel
from retrieval.trainer import Trainer
from retrieval.data import RetrievalCollator, load_data
import yaml
import click

@click.command()
@click.option('--config-path', type=Path, required=True)
def main(config_path: Path):
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    config = RetrievalConfig(**data)

    transformer = RetrievalModel(config)

    trainable = RetrievalTrainable(config.trainer)
    trainer = Trainer(config.trainer, transformer, trainable, RetrievalCollator())

    train_dataset, val_dataset = load_data(config, test=False)
    trainer.train(train_dataset, val_dataset)

if __name__ == "__main__":
    main()
    print(1)