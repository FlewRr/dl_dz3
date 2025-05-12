from retrieval.trainable import RetrievalTrainable
from retrieval.config import RetrievalConfig
from retrieval.transformer_retriever import TransformerRetriever, RetrievalModel
from retrieval.trainer import Trainer
from retrieval.data import RetrievalCollator, load_data
import yaml

if __name__ == "__main__":
    config_path = "configs/config.yaml"
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    config = RetrievalConfig(**data)
    trainable = RetrievalTrainable(config)

    transformer = RetrievalModel(config)

    trainer = Trainer(config.trainer, transformer, trainable, RetrievalCollator())

    train_dataset, val_dataset = load_data(config, test=False)
    trainer.train(train_dataset, val_dataset)
    print(1)