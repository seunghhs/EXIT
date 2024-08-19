import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class Dataset(Dataset):
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
        # self.task = task
        # assert task in {'pretrain', 'finetuning'}

        print(f"read {self.data_dir}...")

        if not os.path.isfile(self.data_dir):
            raise FileNotFoundError(
                f"{self.data_dir} doesn't exist"
            )

        with open(self.data_dir, "rb") as h:
            data_list = pickle.load(h)
        

        self.xrd, self.sa, self.pv, self.mofid, self.name, self.ref =\
        zip(*[(item['xrd'], item['sa'], item['pv'], item['mofid'], item['name'], item['ref']) for item in data_list])
        self.tokens = self.get_tokens(self.mofid)
           



    def __len__(self):
        return len(self.xrd)

    def __getitem__(self, index):
        ret = dict()

        ret.update(
            {
                "xrd": self.xrd[index],
                "sa": self.sa[index],
                "pv": self.pv[index],
                "mofid": self.mofid[index],
                "name": self.name[index],
                "ref": self.ref[index]

            }
        )
        #ret.update(self.tokens[index])


        return ret

    def get_tokens(self, mofid):

        ################## jw token #### 
        # self.tokens <- mofid        
        pass


