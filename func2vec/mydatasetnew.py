from tempfile import tempdir
from pyparsing import original_text_for
from command import configs
from fairseq.models.trex import TrexModel
import torch
import numpy as np
import csv
from torch.utils.data import DataLoader, Dataset




class Func2vectrainset(Dataset):
    def __init__(self, data_dir,datasettype):

        self.trex = TrexModel.from_pretrained(f'checkpoints/similarity/',
                                 checkpoint_file='checkpoint_best.pt',
                                 data_name_or_path=f'data-src/similarity/input0/')

        self.trex = self.trex.cuda()
        self.trex.eval() 

        if data_dir[-1]!='/':
            data_dir = data_dir  +'/'

        self.samples0 = {field: [] for field in configs.fields}
        self.samples1 = {field: [] for field in configs.fields}
        self.labels = []

        original_labels=[]

        for field in configs.fields:
            with open(data_dir+datasettype+f'.{field}.input0', 'r') as f:
                for line in f:
                    self.samples0[field].append(line.strip())
        for field in configs.fields:
            with open(data_dir+datasettype+f'.{field}.input1', 'r') as f:
                for line in f:
                    self.samples1[field].append(line.strip())
        with open(data_dir+datasettype+f'.label', 'r') as f:
            for line in f:
                original_labels.append(float(line.strip()))
        top = len(original_labels)

        self.finalmean = []
        self.labels = [0]*top

        i =0
        for sample_idx in range(top):
            print(sample_idx)
            tempsample0 = {field: self.samples0[field][sample_idx] for field in configs.fields}
            tempsample1 = {field: self.samples1[field][sample_idx] for field in configs.fields}

            if original_labels[sample_idx]== -1:
                continue
            if tempsample0==tempsample1:
                continue
            self.labels[i] =original_labels[sample_idx]
   
            self.finalmean.append([tempsample0,tempsample1])


    def mycollate_fn(self,batch_list):
                
        maxlen=0
        str_len0=[]
        str_len1=[]
        for i in range(len(batch_list)):
            leni=len(batch_list[i][0][0]['static'][0])
            str_len0.append(leni)
            leni=len(batch_list[i][0][1]['static'][0])
            str_len1.append(leni)
        maxlen=max([max(str_len0),max(str_len1)])
        print(maxlen)
        for i in range(len(batch_list)):
            my0=torch.ones(maxlen-str_len0[i])
            my1=torch.ones(maxlen-str_len1[i])

            for field in configs.fields:
                temp=torch.cat((batch_list[i][0][0][field][0],my0),0)
                batch_list[i][0][0][field]=temp.unsqueeze(0)
                temp=torch.cat((batch_list[i][0][1][field][0],my1),0)
                batch_list[i][0][1][field]=temp.unsqueeze(0)
        data = torch.cat([item[0] for item in batch_list])
        return batch_list[0]




    def __getitem__(self, index):

        tempsample0 = self.finalmean[index][0]
        tempsample1 = self.finalmean[index][1]

        sample0_tokens = self.trex.encode(tempsample0)
        sample1_tokens = self.trex.encode(tempsample1)
        sample0_emb = self.trex.process_token_dict(sample0_tokens)
        sample1_emb = self.trex.process_token_dict(sample1_tokens)
        maxlen=512
        my0=torch.ones(maxlen-len(sample0_emb['static'][0])).cuda()
        my1=torch.ones(maxlen-len(sample1_emb['static'][0])).cuda()
        for field in configs.fields:
            temp0=torch.cat((sample0_emb[field][0],my0),0)
            # sample0_emb[field]=temp0.unsqueeze(0)
            sample0_emb[field]=temp0.long().cuda()

            temp0=torch.cat((sample1_emb[field][0],my1),0)
            sample1_emb[field]=temp0.long().cuda()

        target = self.labels[index]
        return [sample0_emb,sample1_emb],target

        

    def __len__(self):
        return len(self.finalmean)
