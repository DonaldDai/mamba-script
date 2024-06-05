# -*- coding: utf-8 -*-
import os
import gc
import argparse
import json
import random
import math
import random
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
import torch
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from performer_pytorch import PerformerLM
import scanpy as sc
import anndata as ad
from utils import *
import pickle as pkl

from mamba_ssm import Mamba, MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig

import datetime
import wandb

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1, help='Local process rank.')
parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
parser.add_argument("--gene_num", type=int, default=16906, help='Number of genes.')
parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
parser.add_argument("--batch_size", type=int, default=3, help='Number of batch size.')
parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate.')
parser.add_argument("--grad_acc", type=int, default=60, help='Number of gradient accumulation.')
parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')
parser.add_argument("--data_path", type=str, default='./data/Zheng68K.h5ad', help='Path of data for finetune.')
parser.add_argument("--model_path", type=str, default='./panglao_pretrained_mamba.pth', help='Path of pretrained model.')
parser.add_argument("--ckpt_dir", type=str, default='./ckpts/finetune/', help='Directory of checkpoint to save.')
parser.add_argument("--model_name", type=str, default='finetune', help='Finetuned model name.')

args = parser.parse_args()
rank = int(os.environ.get("RANK", 0))
# local_rank = args.local_rank
local_rank = int(os.environ["LOCAL_RANK"])
is_master = local_rank == 0
is_global_master = rank == 0

SEED = args.seed
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION = args.grad_acc
LEARNING_RATE = args.learning_rate
SEQ_LEN = args.gene_num + 1
VALIDATE_EVERY = args.valid_every

PATIENCE = 10
UNASSIGN_THRES = 0.0

CLASS = args.bin_num + 2
POS_EMBED_USING = args.pos_embed

model_name = args.model_name
ckpt_dir = args.ckpt_dir

dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
world_size = torch.distributed.get_world_size()

seed_all(SEED + torch.distributed.get_rank())

if is_global_master:
    wandb.login(key="fa1a9ae37df78a027fdc9c43a3892529b38fc4ce")
    # 准备上传参数
    wandb.init(
        project="scbert-mamba-finetune",
        name=get_job_mark(),
        save_code=True,
        job_type="finetune",
        tags=["finetune", "train", "mamba"],
        notes=f"finetune-{datetime.datetime.now()}",
        id=get_job_mark(),
        # track hyperparameters and run metadata
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": "MAMBA",
            "dataset": args.data_path,
            "epochs": EPOCHS,
        }
    )



class SCDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[rand_start].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        seq_label = self.label[rand_start]
        return full_seq, seq_label

    def __len__(self):
        return self.data.shape[0]


class Identity(torch.nn.Module):
    def __init__(self, dropout = 0., h_dim = 100, out_dim = 10):
        super(Identity, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 180))
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(in_features=SEQ_LEN, out_features=512, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=512, out_features=h_dim, bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

    def forward(self, x):
        x = x[:,None,:,:]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

data = sc.read_h5ad(args.data_path)
label_dict, label = np.unique(np.array(data.obs['celltype']), return_inverse=True)  # Convert strings categorical to integrate categorical, and label_dict[label] can be restored
#store the label dict and label for prediction
with open('label_dict', 'wb') as fp:
    pkl.dump(label_dict, fp)
with open('label', 'wb') as fp:
    pkl.dump(label, fp)
class_num = np.unique(label, return_counts=True)[1].tolist()
class_weight = torch.tensor([(1 - (x / sum(class_num))) ** 2 for x in class_num])
label = torch.from_numpy(label)
data = data.X

acc = []
f1 = []
f1w = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
pred_list = pd.Series(['un'] * data.shape[0])

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
for index_train, index_val in sss.split(data, label):
    data_train, label_train = data[index_train], label[index_train]
    data_val, label_val = data[index_val], label[index_val]
    train_dataset = SCDataset(data_train, label_train)
    val_dataset = SCDataset(data_val, label_val)

train_sampler = DistributedSampler(train_dataset)
val_sampler = DistributedSampler(val_dataset)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

# for index, (data_v, labels_v) in enumerate(val_loader):
#     index += 1
#     data_v, labels_v = data_v.to(device), labels_v.to(device)
#                 # print("Data:"+)
#     print("Val Label:"+str(labels_v))

# model = PerformerLM(
#     num_tokens = CLASS,
#     dim = 200,
#     depth = 6,
#     max_seq_len = SEQ_LEN,
#     heads = 10,
#     local_attn_heads = 0,
#     g2v_position_emb = POS_EMBED_USING
# )

class MambaFT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = MambaLMHeadModel(config)
        # print("config:\n"+str(config))
        self.to_out = Identity(dropout=0., h_dim=128, out_dim=label_dict.shape[0])
    
    def forward(self, data):
        output = self.model(data)[0]
        # print("logits:"+str(output.shape)+"\n"+str(output))
        # print(str(output))
        output = self.to_out(output)
        return output

config = MambaConfig()
config.d_model = 180
config.n_layer = 5
config.vocab_size = 16906
# config.vocab_size = 10
# config.fused_add_norm = False
model = MambaFT(config).to("cuda")
# model = MambaLMHeadModel(config).to("cuda")





path = args.model_path
ckpt = torch.load(path)
model.model.load_state_dict(ckpt['model_state_dict'])
for param in model.model.parameters():
    param.requires_grad = False
# for param in model.norm.parameters():
#     param.requires_grad = True
model.model.backbone.norm_f.requires_grad_ = True
for param in model.model.backbone.layers:
    param.requires_grad_ = True
# for param in model.performer.net.layers[-2].parameters():
#     param.requires_grad = True
# model.to_out = Identity(dropout=0., h_dim=128, out_dim=label_dict.shape[0])
# print("Output_shape:\n"+str(label_dict.shape))
model = model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# optimizer
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=15,
    cycle_mult=2,
    max_lr=LEARNING_RATE,
    min_lr=1e-6,
    warmup_steps=5,
    gamma=0.9
)
# loss_fn = nn.CrossEntropyLoss(weight=None).to(local_rank)
loss_fn = nn.CrossEntropyLoss().to(local_rank)

dist.barrier()
trigger_times = 0
max_acc = 0.0
for i in range(1, EPOCHS+1):
    train_loader.sampler.set_epoch(i)
    model.train()
    dist.barrier()
    running_loss = 0.0
    cum_acc = 0.0
    for index, (data, labels) in enumerate(train_loader):
        index += 1
        # print("Data Shape:"+str(data.shape))
        data, labels = data.to(device), labels.to(device)
        if index % GRADIENT_ACCUMULATION != 0:
            with model.no_sync():
                # if index % 1000 == 0:
                #     print("Model:\n"+str(model))
                # print("Data:\n"+str(data)+"\nDataShape:"+str(data.shape))
                # print("Labels:"+str(labels)+"\nShape(labels):"+str(labels.shape))
                logits = model(data)
                # logits = torch.mean(model(data)[0],dim=2)
                # logits = torch.max(model(data)[0],dim=2)[0]
                if index % 1000 == 0:
                    print("Step:"+str(index)+"\n"+str(logits))
                # logits.requires_grad = True
                # print("======================================================================\nLogits:\n"+str(logits)+"\nShape(logits):\n"+str(logits.shape))
                loss = loss_fn(logits, labels)
                loss.backward()
        if index % GRADIENT_ACCUMULATION == 0:
            logits = model(data)
            # logits = torch.mean(model(data)[0],dim=2)
            # logits = torch.max(model(data)[0],dim=2)[0]
            # logits.requires_grad = True
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
            optimizer.step()
            optimizer.zero_grad()
        running_loss += loss.item()
        softmax = nn.Softmax(dim=-1)
        final = softmax(logits)
        final = final.argmax(dim=-1)
        pred_num = labels.size(0)
        correct_num = torch.eq(final, labels).sum(dim=-1)
        cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
    epoch_loss = running_loss / index
    epoch_acc = 100 * cum_acc / index
    epoch_loss = get_reduced(epoch_loss, local_rank, 0, world_size)
    epoch_acc = get_reduced(epoch_acc, local_rank, 0, world_size)
    if is_master:
        print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}%  ==')
    if is_global_master:
        wandb.log({"accuracy": epoch_acc, "loss": epoch_loss})
    dist.barrier()
    scheduler.step()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        dist.barrier()
        running_loss = 0.0
        predictions = []
        truths = []
        with torch.no_grad():
            for index, (data_v, labels_v) in enumerate(val_loader):
                index += 1
                data_v, labels_v = data_v.to(device), labels_v.to(device)
                # print("Data:"+)
                # print("Val Label:"+str(labels_v))
                logits = model(data_v)
                # logits = torch.mean(model(data_v)[0],dim=2)
                # logits = torch.max(model(data_v)[0],dim=2)[0]
                logits.requires_grad = True
                loss = loss_fn(logits, labels_v)
                running_loss += loss.item()
                softmax = nn.Softmax(dim=-1)
                final_prob = softmax(logits)
                final = final_prob.argmax(dim=-1)
                final[np.amax(np.array(final_prob.cpu()), axis=-1) < UNASSIGN_THRES] = -1
                predictions.append(final)
                truths.append(labels_v)
            del data_v, labels_v, logits, final_prob, final
            # print("Predictions1:"+str(len(predictions))+"\n"+str(predictions))
            # print("Truths1:"+str(len(truths))+"\n"+str(truths))
            # gather
            predictions = distributed_concat(torch.cat(predictions, dim=0), len(val_sampler.dataset), world_size)
            truths = distributed_concat(torch.cat(truths, dim=0), len(val_sampler.dataset), world_size)
            # print("Predictions2:"+str(len(predictions))+"\t\t"+str(torch.max(predictions))+"\n"+str(predictions))
            # print("Truths2:"+str(len(truths))+"\t\t"+str(torch.max(truths))+"\n"+str(truths))
            no_drop = predictions != -1
            # print("no_drop:"+str(no_drop))
            predictions = np.array((predictions[no_drop]).cpu())
            truths = np.array((truths[no_drop]).cpu())
            # print("Predictions3:"+str(len(predictions))+"\n"+str(predictions))
            # print("Truths3:"+str(len(truths))+"\n"+str(truths))
            cur_acc = accuracy_score(truths, predictions)
            # print("CUR_ACC:"+str(cur_acc))
            f1 = f1_score(truths, predictions, average='macro')
            # print("f1:"+str(f1))
            val_loss = running_loss / index
            val_loss = get_reduced(val_loss, local_rank, 0, world_size)
            if is_global_master:
                print(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.6f} | F1 Score: {f1:.6f}  ==')
                print(confusion_matrix(truths, predictions))
                print(classification_report(truths, predictions, target_names=label_dict.tolist(), digits=4))
            if is_global_master and cur_acc > max_acc:
                max_acc = cur_acc
                trigger_times = 0
                save_best_ckpt(i, model, optimizer, scheduler, val_loss, model_name, ckpt_dir)
            else:
                trigger_times += 1
                if trigger_times > PATIENCE:
                    break
    del predictions, truths