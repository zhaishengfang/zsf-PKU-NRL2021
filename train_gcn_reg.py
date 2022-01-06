# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
from model.myGNN import *
import torch
from torch_geometric.loader import DataLoader
from data.load_data import load_data
from torch_geometric.data import Data
import random

BATCH_SIZE = 16

name = r"ESOL"
list_mol_graph, properties = load_data(name)
print('Name:', name)

properties = torch.FloatTensor(properties)
mean = torch.mean(properties, dim=0, keepdim=True)
std = torch.std(properties, dim=0, keepdim=True)


def normalize_prop(p: torch.Tensor) -> torch.Tensor:
    return (p - mean) / std


def denormalize_prop(p: torch.Tensor) -> torch.Tensor:
    return p * std + mean


print('load data...')

Datas = []
for i_mol_graph, y in zip(list_mol_graph, properties):
    # print(y)
    e_index = torch.tensor([np.concatenate((i_mol_graph.start_indices, i_mol_graph.end_indices), axis=0),
                    np.concatenate((i_mol_graph.end_indices, i_mol_graph.start_indices), axis=0)]).long()
    nodes_x = torch.tensor(i_mol_graph.atom_features).float()
    e_attr = torch.tensor(np.concatenate((i_mol_graph.bond_features , i_mol_graph.bond_features), axis=0)).float()
    yi =  y.float() # normalize_prop(torch.tensor(properties[i]).float())
    datai = Data(x = nodes_x, edge_index=e_index, edge_attr=e_attr, y=yi)
    Datas.append(datai)

print('\nlen graphs:', len(Datas))

Num_node_features = len(Datas[0].x[0])
print('Num_node_features:', Num_node_features)

print(len(Datas))
random.shuffle(Datas)

train_dataset = Datas[:int(0.8*len(Datas))]
test_dataset = Datas[int(0.8*len(Datas)):]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = GCN_reg_e(hidden_channels=32, Num_node_features=Num_node_features)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay=1e-4)
# criterion = torch.nn.NLLLoss()
# criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()
    
    for data in train_loader:
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index, data.edge_attr,data.batch)  # data.edge_attr,
        # out = model(data)
        out = denormalize_prop(out)
        # print('\ntwo pre:\n', out, '\n', data.y)
        loss = F.mse_loss(out, data.y)
        
        loss.backward()
        optimizer.step()


def test(loader):
    model.eval()
    
    list_pred = []
    list_y = []
    for data in loader:  # 批遍历测试集数据集。
        out = model(data.x, data.edge_index,data.edge_attr, data.batch)  # 一次前向传播   data.edge_attr,
        # out = model(data)
        # print("out,---\n", out, '\n')
        out = torch.flatten(out).tolist()
        # print("out2,---\n", out, '\n')
        list_pred.extend(out)
        # print("list_pred,---\n", list_pred, '\n')
        list_y.extend(data.y.tolist())
        # print("list_y,---\n", list_y, '\n')
    
    total_pred = torch.tensor(list_pred)
    total_y = torch.tensor(list_y)
    
    # total_pred = denormalize_prop(total_pred)
    # total_y = denormalize_prop(total_pred)
    
    total_pred = denormalize_prop(total_pred)
    rmse = torch.sqrt(F.mse_loss(total_pred, total_y))
    rmse = float(rmse)
    # print(f'\t\t\tRMSE----------: {rmse:.4f}')
    return rmse

import os
MODEL_DICT_DIR = 'model/pt'
if not os.path.isdir(MODEL_DICT_DIR):
    os.mkdir(MODEL_DICT_DIR)


oldrmse = 1e9
Epochs = 1000
best_epoch = 0
for epoch in range(1, Epochs):
    train()
    train_rmse = test(train_loader)
    test_rmse = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train rmse: {train_rmse:.4f}, Test rmse: {test_rmse:.4f}')
    
    if epoch > Epochs - 10:
        if test_rmse < oldrmse:
            best_epoch = epoch
            oldrmse = test_rmse
            torch.save(model.state_dict(), f'{MODEL_DICT_DIR}/{name}-reg.pkl')
            print("best epoch changes-", best_epoch)

print("BEST epoch:", best_epoch)
print("BSET rmse:", oldrmse)













