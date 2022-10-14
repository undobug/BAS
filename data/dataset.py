# --------------------------------------------------------
#!/usr/bin/env python
# Licensed under The MIT License [see LICENSE for details]
# Written by fyb
# --------------------------------------------------------


import pandas as pd
import numpy as np
import os
import math
import torch
import re
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from scipy import io
import multiprocessing as mp
import torch.distributed as dist
from sklearn.model_selection import StratifiedShuffleSplit
from .samplers import DistributedSpeakerSampler

def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(state='train',config=config)
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val, _ = build_dataset(state='dev', config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")
    dataset_test, _ = build_dataset(state='test', config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build test dataset")


    sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train, shuffle=True)
    sampler_val = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False)
    sampler_test = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    data_loader_test = DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    return dataset_val, data_loader_train, data_loader_val, data_loader_test

def build_dataset(state,config):
    
    if config.DATA.DATASET == 'iemocap':
        dataset = IEMOCAP_dataset(state=state,config=config)
        nb_classes = 4
    elif config.DATA.DATASET == 'meld':
        dataset = MELD_dataset(state=state,config=config)
        nb_classes = 7
    else:
        raise NotImplementedError("No support this dataset Now.")

    return dataset, nb_classes




class IEMOCAP():
    def __init__(self, state=None, csv_file=None):
        df = pd.read_csv(csv_file)
        session,names, labels, domains = [], [], [], []
        for i in range(len(df)):
            session.append(df['name'][i][4])
            names.append(df['name'][i])
            labels.append(df['label'][i])
            domains.append(df['domain'][i])

        self.names = names
        self.labels = labels
        self.domains = domains
        self.state = state

    def foldsplit(self, fold, strategy):

        name_fold, y_fold, d_fold, l_fold = [], [], [], []
        if strategy == '5fold':
            testSes = 'Ses0{}'.format(6-fold)
            for i, name in enumerate(self.names):
                if ((self.state == 'train') and (testSes not in name)) or ((self.state != 'train') and (testSes in name)):
                    name_fold.append(name)
                    y_fold.append(self.labels[i])
                    d_fold.append(self.domains[i])

                      
        else:
            gender = 'F' if fold%2 == 0 else 'M'
            fold = math.ceil(fold/2)
            testSes = 'Ses0{}'.format(6-fold)
            for i, name in enumerate(self.names):
                if ((self.state == 'train') and ((testSes not in name) and (gender not in name.split('_')[-1]))) \
                or ((self.state != 'train') and ((testSes in name) or (gender in name.split('_')[-1]))):
                    name_fold.append(name)
                    y_fold.append(self.labels[i])
                    d_fold.append(self.domains[i])


        self.names = name_fold
        self.labels = y_fold
        self.domains = d_fold



class IEMOCAP_dataset(Dataset):
    def __init__(self, config, state, strategy='5fold', **kwargs):
        self.csv_file = config.DATA.CSV_PATH
        database = IEMOCAP(state, csv_file=self.csv_file)
        database.foldsplit(fold=config.TRAIN.CURRENT_FOLD, strategy=strategy)
        self.matdir = config.DATA.DATA_PATH
        self.spkdir = config.DATA.SPK_PATH
        self.labels = database.labels
        self.domains = database.domains
        self.names= database.names
        self.config = config
    def __getitem__(self, index):
        label = torch.LongTensor([self.labels[index]]).squeeze()
        domain = torch.LongTensor([self.domains[index]]).squeeze()

        path = os.path.join(self.matdir, self.names[index])
        path_spk = os.path.join(self.spkdir, self.names[index])
        fea = np.float32(io.loadmat(path)[f'{self.config.DATA.USE_FEATURE}'])
        spk = np.float32(io.loadmat(path_spk)['xvec'])
        spk=np.squeeze(spk)
        #spk = (spk - spk.min(axis=0)) / (spk.max(axis=0) - spk.min(axis=-0))
        name = self.names[index]
        # 326 is the maxlen of 80% data in IEMOCAP
        max_length = self.config.DATA.WAV_LENGTH
        if fea.shape[0]>max_length:
            left = np.random.randint(fea.shape[0]-max_length)
        fea = fea[left:left+max_length,:] if fea.shape[0]>max_length else np.pad(fea,((0,max_length-fea.shape[0]),(0,0)),'constant',constant_values = (0,0))

        return fea, label, domain

    def __len__(self):
        return len(self.names)  




class MELD():
    def __init__(self, state=None,csv_file=None):
        df = pd.read_csv(csv_file)
        names, labels, genders, speakers = [], [], [], []
        for i in range(len(df)):
            genders.append(df['gender'][i])
            names.append(df['name'][i])
            labels.append(df['label'][i])
            speakers.append(df['speaker'][i])

        self.names = names
        self.labels = labels
        self.genders = genders
        self.speakers = speakers
        self.state = state


class MELD_dataset(Dataset):
    def __init__(self, config, state, **kwargs):
        if state == 'train':
            self.csv_file = config.DATA.CSV_PATH + 'train_7.csv'
        if state == 'dev':
            self.csv_file = config.DATA.CSV_PATH + 'dev_7.csv'
        if state == 'test':
            self.csv_file = config.DATA.CSV_PATH + 'test_7.csv'                   
        database = MELD(state, csv_file=self.csv_file)
        self.matdir = config.DATA.DATA_PATH + state+'/'
        self.names = database.names
        self.labels = database.labels
        self.genders = database.genders
        self.speakers= database.speakers
        self.config =config
    def __getitem__(self, index):
        name = self.names[index]
        label = torch.LongTensor([self.labels[index]]).squeeze()
        speaker = torch.LongTensor([self.speakers[index]]).squeeze()
        gender = torch.LongTensor([self.genders[index]]).squeeze()
        path = os.path.join(self.matdir, self.names[index])
        fea = np.float32(io.loadmat(path)[f'{self.config.DATA.USE_FEATURE}'])

        # 224 is the maxlen of 80% data in MELD
        max_length = self.config.DATA.WAV_LENGTH
        if fea.shape[0]>max_length:
            left = np.random.randint(fea.shape[0]-max_length)
        fea = fea[left:left+max_length,:] if fea.shape[0]>max_length else np.pad(fea,((0,max_length-fea.shape[0]),(0,0)),'constant',constant_values = (0,0))

        #print(fea.shape)
        return fea, label, speaker

    def __len__(self):
        return len(self.names) 








