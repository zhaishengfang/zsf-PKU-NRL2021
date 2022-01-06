# -*- coding: utf-8 -*-
from model.myGNN import *
import torch
from torch_geometric.loader import DataLoader
from data.load_data import load_data
from torch_geometric.data import Data
import random

BATCH_SIZE = 16

name = r"sars"
list_mol_graph, properties = load_data(name)

properties = torch.FloatTensor(properties)

# for i in range(13):
#     print('\n\n',i)
#     print('select...')
i  = 0
Datas = []
Target = i
print('Target num:', Target)

for i_mol_graph, y in zip(list_mol_graph, properties):
    # print(y)
    if not y[Target].isnan():
        e_index = torch.tensor([np.concatenate((i_mol_graph.start_indices, i_mol_graph.end_indices), axis=0),
                        np.concatenate((i_mol_graph.end_indices, i_mol_graph.start_indices), axis=0)]).long()
        nodes_x = torch.tensor(i_mol_graph.atom_features).float()
        e_attr = torch.tensor(np.concatenate((i_mol_graph.bond_features , i_mol_graph.bond_features), axis=0)).float()
        yi =   y[Target].long() # normalize_prop(torch.tensor(properties[i]).float())
        datai = Data(x = nodes_x, edge_index=e_index, edge_attr=e_attr, y=yi)
        Datas.append(datai)

print('\nlen  target:', len(Datas))

# exit()

mean = torch.mean(properties, dim=0, keepdim=True)
std = torch.std(properties, dim=0, keepdim=True)


def normalize_prop(p: torch.Tensor) -> torch.Tensor:
    return (p - mean) / std


def denormalize_prop(p: torch.Tensor) -> torch.Tensor:
    return p * std + mean


# Datas = []
# for i, i_mol_graph in enumerate(list_mol_graph):
#     # print(i_mol_graph)
#     e_index = torch.tensor([np.concatenate((i_mol_graph.start_indices, i_mol_graph.end_indices), axis=0),
#                             np.concatenate((i_mol_graph.end_indices, i_mol_graph.start_indices), axis=0)]).long()
#     nodes_x = torch.tensor(i_mol_graph.atom_features).float()
#     e_attr = torch.tensor(np.concatenate((i_mol_graph.bond_features, i_mol_graph.bond_features), axis=0)).float()
#     y = torch.tensor(properties[i]).float()  # normalize_prop(torch.tensor(properties[i]).float())
#     datai = Data(x=nodes_x, edge_index=e_index, edge_attr=e_attr, y=y)
#     Datas.append(datai)

Num_node_features = len(Datas[0].x[0])
print('Num_node_features:', Num_node_features)

print(len(Datas))
random.shuffle(Datas)

train_dataset = Datas[:int(0.7*len(Datas))]
test_dataset = Datas[int(0.7*len(Datas)):]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)



model = GCN_cls_e(hidden_channels=32, Num_node_features=Num_node_features, num_classes=4)
model_type = GCN_cls_e
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
# criterion = torch.nn.NLLLoss()
criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()
    loss_all = 0
    for data in train_loader:
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index,data.edge_attr, data.batch)
        # print(out.shape)
        loss = criterion(out, data.y)
        loss_all+=loss.cpu().detach()
        # print(out[:5])
        # print(data.y[:5], '\n\n')
        
        loss.backward()
        optimizer.step()
    print("\nloss, ", loss_all)
    
    

def test(loader):
    model.eval()
    
    correct = 0
    for data in loader:  # 批遍历测试集数据集。
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)  # 一次前向传播
        pred = out.argmax(dim=1)  # 使用概率最高的类别
        correct += int((pred == data.y).sum())  # 检查真实标签
        # print(pred[:7])
        # print(data.y[:7], '\n\n')
        
    return correct / len(loader.dataset)

import os
MODEL_DICT_DIR = 'model/pt'
if not os.path.isdir(MODEL_DICT_DIR):
    os.mkdir(MODEL_DICT_DIR)

oldacc = 0
Epochs = 120
best_epoch = 0
for epoch in range(1, Epochs):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    print()
    if epoch>Epochs-10:
        if test_acc>oldacc:
            best_epoch = epoch
            torch.save(model.state_dict(), f'{MODEL_DICT_DIR}/{name}-target_{Target}.pkl')
            print("saved model in ", f'{MODEL_DICT_DIR}/{name}-target_{Target}.pkl')
            print("best epoch changes-", best_epoch)

print("BEST epoch:", best_epoch)
print('Target num:', Target)

