import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from exit.tokenizer.mof_tokenizer import MOFTokenizer

tokenizer = MOFTokenizer(model_max_length = 512, padding_side='right')

class BasicDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
    ):
        """
        Dataset for pretrained MOF.
        Args:
            data_dir (str): where data_dir(.pkl) for xrd, sa (surface area), pv (pore volume), and mofid ; 
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
        self.xrd, self.sa, self.pv, self.mofid, self.name, self.ref =\
        zip(*[(np.expand_dims(item['xrd'], axis=0), item['sa'], item['pv'], item['mofid'], item['name'], item['ref']) for item in data_list])
        self.xrd = np.array(self.xrd)
        self.sa = np.array(self.sa)
        self.pv = np.array(self.pv)
        self.xrd = torch.tensor(self.xrd, dtype=torch.float32)
        self.sa = torch.tensor(self.sa, dtype=torch.float32)
        self.pv = torch.tensor(self.pv, dtype=torch.float32)

        self.tokens, self.attention_mask = self.get_tokens(self.mofid)

           
    def __len__(self):
        return len(self.xrd)

    def __getitem__(self, index):
        results = dict()
        results.update(
            {
                "xrd": self.xrd[index],
                "sa": self.sa[index],
                "pv": self.pv[index],
                "mofid": self.mofid[index],
                "input_ids": self.tokens[index],
                "attention_mask": self.attention_mask[index]
                #"name": self.name[index],
                #"ref": self.ref[index],


            }
        )
        #results.update(self.tokens[index])
        return results
    
    def get_tokens(self, mofid):
        #token =  np.array([tokenizer.encode(i, max_length=512, truncation=True,padding='max_length') for i in mofid])
        token_dict = tokenizer(mofid,max_length=512, truncation=True,padding='max_length' )

        return token_dict['input_ids'], token_dict['attention_mask']


def custom_collate_fn(batch, data_collator):
    # Extract only input_ids for each data item and pass them individually to data_collator
    for item in batch:
        input_ids_batch = {"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]}

        # DataCollatorForLanguageModeling only processes input_ids of each data item
        masked_batch = data_collator([input_ids_batch])

        # Add masked input_ids, attention_mask, labels, etc. to the item
        item.update({key: value.squeeze(0) for key, value in masked_batch.items()})

    # Check whether it is a tensor, treat tensors as stack, others as list
    def batch_stack_or_list(data_list):
        if isinstance(data_list[0], torch.Tensor):
            return torch.stack(data_list)
        else:
            return data_list

    # Check if there are fields to merge and process them
    merged_batch = {}
    for key in batch[0].keys():
        data_list = [bat[key] for bat in batch]
        
        # 'input_ids', 'attention_mask', and 'labels' are processed unconditionally
        if key in ['input_ids', 'attention_mask', 'labels']:
            merged_batch[key] = batch_stack_or_list(data_list)
        
        else:
            if isinstance(data_list[0], (torch.Tensor, str, float, int)):  # check data type
                merged_batch[key] = batch_stack_or_list(data_list)

    return merged_batch