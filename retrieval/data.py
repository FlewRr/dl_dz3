from typing import List, Any

from datasets import load_dataset, Dataset
import random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from retrieval.config import RetrievalConfig


class RetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Dataset, config: RetrievalConfig, test: bool=False, use_contrastive_format: bool=False):
        self._dataset = dataset
        # TODO: config
        self._tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
        self._test = test
        self._config = config
        self._use_contrastive_format = use_contrastive_format

    def __len__(self):
        return len(self._dataset)

    def _tokenize(self, text: str) -> dict:
        encoding = self._tokenizer(text, max_length=self._config.max_length, truncation=True).encodings[0]

        return {
            'input_ids': torch.tensor(encoding.ids, dtype=torch.long),
            'attention_mask': torch.tensor(encoding.attention_mask, dtype=torch.long),
            'text': text
        }

    def __getitem__(self, idx: int):
        positive_pair = self._dataset[idx]

        if not self._test:
            neg_idx = idx
            while neg_idx == idx:
                neg_idx = random.randint(0, len(self._dataset) - 1)
            negative_pair = self._dataset[neg_idx]
            # self._config.query_prefix)
            negative_query = "query: " + negative_pair["query"]
            # self._config.query_prefix
        positive_query = "query: " + positive_pair["query"]
        anchor_document = "passage" + positive_pair["answer"]

        if self._test:
            return {
                "positive": self._tokenize(positive_query),
                "anchor":  self._tokenize(anchor_document)
            }
        else:
            if self._config.use_contrastive_format:
                return [
                    {"x1": self._tokenize(anchor_document), "x2": self._tokenize(positive_query), "label": 1.0},
                    {"x1": self._tokenize(anchor_document), "x2": self._tokenize(negative_query), "label": 0.0}
                ]
            else:
                return {
                    "positive":  self._tokenize(positive_query),
                    "negative":  self._tokenize(negative_query),
                    "anchor":  self._tokenize(anchor_document)
                }

class RetrievalCollator:
    def _stack_pad_tensors(self, items: List[torch.Tensor]) -> torch.Tensor:
        max_len = max(len(x) for  x in items)
        items = [F.pad(x, (0, max_len - len(x)), mode="constant", value=0) for x in items]

        return torch.stack(items)

    def _collate_single(self, items: list[dict[str, Any]]):
        return {
            'input_ids': self._stack_pad_tensors([x['input_ids'] for x in items]),
            'attention_mask': self._stack_pad_tensors([x['attention_mask'] for x in items]),
            'text': [x['text'] for x in items],
        }
    def __call__(self, items):
        if isinstance(items[0], list) and 'x1' in items[0][0]:
            flat_items = [pair for sublist in items for pair in sublist]

            return {
                'x1': self._collate_single([x['x1'] for x in flat_items]),
                'x2': self._collate_single([x['x2'] for x in flat_items]),
                'labels': torch.tensor([x['label'] for x in flat_items], dtype=torch.float)
            }

        out_dict = {
            'positive': self._collate_single([x['positive'] for x in items]),
            'anchor': self._collate_single([x['anchor'] for x in items])
        }

        if 'negative' in items[0]:
            out_dict['negative'] = self._collate_single([x['negative'] for x in items])
        return out_dict


def load_data(config: RetrievalConfig, test: bool):
    data = load_dataset("sentence-transformers/natural-questions")["train"]
    data = data.train_test_split(test_size=0.02, seed=42, shuffle=True)

    if test:
        return RetrievalDataset(data["test"], config, test=test)
    else:
        return RetrievalDataset(data["train"], config, test=test),\
               RetrievalDataset(data["test"], config, test=test)