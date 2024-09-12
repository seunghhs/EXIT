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
        
        self.tokens = self.get_tokens(self.mofid)
           
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
                "name": self.name[index],
                "ref": self.ref[index],
                "tokens" : self.tokens[index]

            }
        )
        #results.update(self.tokens[index])
        return results
    
    def get_tokens(self, mofid):
        token =  np.array([tokenizer.encode(i, max_length=512, truncation=True,padding='max_length') for i in mofid])
        return token

class MOFpretrainDataset(Dataset):
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
        self.tokens = self.get_tokens(self.mofid)
           
    def __len__(self):
        return len(self.xrd)

    def __getitem__(self, index):
        X = torch.from_numpy(np.asarray(self.tokens[index]))
        return X.type(torch.LongTensor)
    
    def get_tokens(self, mofid):
        token =  np.array([tokenizer.encode(i, max_length=512, truncation=True,padding='max_length') for i in mofid])
        return token
