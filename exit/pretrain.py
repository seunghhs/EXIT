import os
import yaml
import torch
import argparse
from exit import __root_dir__
from tqdm.auto import tqdm
from glob import glob
from exit.dataset.dataset import BasicDataset, custom_collate_fn
from torch.utils.data import DataLoader
from exit.modules import visiontransformer
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from exit.modules.model import MultiModal
from collections import defaultdict
from transformers import DataCollatorForLanguageModeling
from exit.tokenizer.mof_tokenizer import MOFTokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import datetime
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

torch.multiprocessing.set_sharing_strategy("file_system")
num_workers =16
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
os.environ["MASTER_PORT"] = "12356"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default = f'{__root_dir__}/config/pretrain.yml' )
    parser.add_argument('--is_test', type=bool, default=False)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default = 1)
    parser.add_argument('--log_dir', type=str, default='./logs_pretrain')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt_pretrain')
    parser.add_argument('--epoch', type=int, default=50)
    args = parser.parse_args()
    
    
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    pl.seed_everything(config['seed'])
    os.makedirs(args.log_dir, exist_ok=True)
    
    train_data_dir = config['dataset']['train_data_dir']
    test_data_dir = config['dataset']['test_data_dir']    
    
    # Check if the path is an absolute path
    if not os.path.isabs(train_data_dir):
        # If it's a relative path, join it with __root_dir__
        train_data_dir = os.path.join(__root_dir__, train_data_dir)
        config['dataset']['train_data_dir'] = train_data_dir
    
    if not os.path.isabs(test_data_dir):
        test_data_dir = os.path.join(__root_dir__, test_data_dir)
        config['dataset']['test_data_dir'] = test_data_dir

    # ckpt 
    ckpt_dir = f'./ckpt_{args.ckpt_dir}/' #{datetime.datetime.now().strftime("%Y-%m-%d")}
    os.makedirs(ckpt_dir, exist_ok=True)

    
    checkpoint_callback = ModelCheckpoint(
        dirpath = ckpt_dir, 
        verbose=True,
        save_last=True,
        save_top_k=1,
        monitor="val/the_metric",
        #every_n_train_steps=10,
        #every_n_epochs=1, 
        mode='min'
    )    
    seed = config['seed']
    logger = pl.loggers.TensorBoardLogger(
        args.log_dir,
        name=f'pretrain_seed{seed}', #{datetime.datetime.now().strftime("%Y-%m-%d")}
    )        

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    early_callback = EarlyStopping(monitor="val/the_metric", mode="min",patience=10,)
    
    callbacks = [checkpoint_callback, lr_callback, early_callback]
    
    

    
    num_nodes = config['num_nodes']
    
    if args.devices==0:
        accumulate_grad_batches = config['batch_size'] // (
            config['per_gpu_batchsize'] * num_nodes
        )
    else:
        accumulate_grad_batches = config['batch_size'] // (
            config['per_gpu_batchsize'] * args.devices * num_nodes
        )    

    
    log_every_n_steps=10
    model = MultiModal(config)

    if config['resume_from'] is not None:
        model = MultiModal.load_from_checkpoint(config['resume_from'],  config=config, strict=False)

    trainer = Trainer(
                    
                      accelerator = args.accelerator,
                      devices = args.devices,
                      num_nodes = config['num_nodes'],
                      max_epochs=args.epoch, 
                      logger=logger,
                      accumulate_grad_batches=accumulate_grad_batches,
                      benchmark=True,
                      strategy=DDPStrategy(find_unused_parameters=True),
                      #resume_from_checkpoint= config.train.resume_from,
                       log_every_n_steps=log_every_n_steps,
                      callbacks=callbacks
                     )   
    
    # pretrain 15% masking
    tokenizer = MOFTokenizer(model_max_length = 512, padding_side='right')
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15  
    )   
    test_data = BasicDataset(test_data_dir)              

    test_loader =DataLoader(test_data, batch_size=config['per_gpu_batchsize'] ,collate_fn=lambda batch: custom_collate_fn(batch, data_collator),
                            num_workers =num_workers,
                             shuffle=False) 
    
    if args.is_test:
        best_ckpt_list =  glob(os.path.join(ckpt_dir, '*.ckpt'))
        best_ckpt = [ ckpt for ckpt in best_ckpt_list if os.path.basename(ckpt).startswith('epoch') ][0]
        print(f'best_ckpt: ', best_ckpt)
        model = MultiModal.load_from_checkpoint(best_ckpt, config=config, strict=False)          
        trainer.test(model, test_loader)


    else:

        train_data = BasicDataset(train_data_dir)


        train_loader =DataLoader(train_data, batch_size=config['per_gpu_batchsize'] ,collate_fn=lambda batch: custom_collate_fn(batch, data_collator),
                                num_workers =num_workers,
                                 shuffle=True)        



     
        
        trainer.fit(model, train_loader, test_loader,  ckpt_path = config['resume_from'])
    
