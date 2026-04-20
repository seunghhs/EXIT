"""
Pretraining dataset: loads hypothetical MOF data (xrd, vf, mofid) from a pickle file.
Used with DataCollatorForLanguageModeling (mlm=True, mlm_probability=0.15) to
randomly mask 15% of MOFid tokens for the MLM objective.
"""
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from exit.tokenizer.mof_tokenizer import MOFTokenizer

# Module-level tokenizer instance shared across all dataset workers
tokenizer = MOFTokenizer(model_max_length=512, padding_side='right')


class BasicDataset(Dataset):
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: path to .pkl file. Each item must have keys:
                xrd   (np.ndarray, shape [seq_length])  — XRD pattern normalized to [0, 1]
                vf    (float)                            — void fraction label
                mofid (str)                              — MOFid string (SMILES && topology)
                name  (str), ref (str)                   — identifiers
        """
        super().__init__()
        self.data_dir = data_dir
        print(f"read {self.data_dir}...")

        if not os.path.isfile(self.data_dir):
            raise FileNotFoundError(
                f"{self.data_dir} doesn't exist"
            )


        with open(self.data_dir, "rb") as h:
            data_list = pickle.load(h)
                
        self.xrd, self.vf, self.mofid, self.name, self.ref =\
        zip(*[(np.expand_dims(item['xrd'], axis=0),  item['vf'], item['mofid'], item['name'], item['ref']) for item in data_list])
        self.xrd = np.array(self.xrd)
        self.vf = np.array(self.vf)
        self.xrd = torch.tensor(self.xrd, dtype=torch.float32)
        self.vf = torch.tensor(self.vf, dtype=torch.float32)
        #self.cell_params = torch.tensor(self.cell_params, dtype=torch.float32)

        self.tokens, self.attention_mask = self.get_tokens(self.mofid)

           
    def __len__(self):
        return len(self.xrd)

    def __getitem__(self, index):
        results = dict()
        results.update(
            {
                "xrd": self.xrd[index],
                
                "vf": self.vf[index],
                "mofid": self.mofid[index],
                "input_ids": self.tokens[index],
                "attention_mask": self.attention_mask[index]

                #"name": self.name[index],
                #"ref": self.ref[index],
                #"sa": self.sa[index],

            }
        )
        #results.update(self.tokens[index])
        return results
    
    def get_tokens(self, mofid):
        #token =  np.array([tokenizer.encode(i, max_length=512, truncation=True,padding='max_length') for i in mofid])
        token_dict = tokenizer(mofid,max_length=512, truncation=True,padding='max_length' )

        return token_dict['input_ids'], token_dict['attention_mask']


def custom_collate_fn(batch, data_collator):
    """
    Custom collate function that applies MLM masking per-item before batching.

    DataCollatorForLanguageModeling is called per-sample (not on the full batch) so that
    each item gets independently sampled mask positions. The collator overwrites input_ids
    with masked versions and sets labels=-100 for unmasked positions.
    """
    for item in batch:
        input_ids_batch = {"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]}
        masked_batch = data_collator([input_ids_batch])
        item.update({key: value.squeeze(0) for key, value in masked_batch.items()})

    def batch_stack_or_list(data_list):
        if isinstance(data_list[0], torch.Tensor):
            return torch.stack(data_list)
        else:
            return data_list

    merged_batch = {}
    for key in batch[0].keys():
        data_list = [bat[key] for bat in batch]
        if key in ['input_ids', 'attention_mask', 'labels']:
            merged_batch[key] = batch_stack_or_list(data_list)
        else:
            if isinstance(data_list[0], (torch.Tensor, str, float, int)):
                merged_batch[key] = batch_stack_or_list(data_list)

    return merged_batch