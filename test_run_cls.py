# -*- coding: utf-8 -*-
from model.myGNN import *
import torch
from torch_geometric.loader import DataLoader
from data.load_data import load_data
from torch_geometric.data import Data
import random

BATCH_SIZE = 16

# def get_data_model(name):

name = r"sars"

# model_paths = []
models = []


list_mol_graph, properties = load_data(data_name=name, set_name='test')

properties = torch.FloatTensor(properties)
mean = torch.mean(properties, dim=0, keepdim=True)
std = torch.std(properties, dim=0, keepdim=True)

Datas = []
for i_mol_graph in list_mol_graph:
    # print(y)
    e_index = torch.tensor([np.concatenate((i_mol_graph.start_indices, i_mol_graph.end_indices), axis=0),
                            np.concatenate((i_mol_graph.end_indices, i_mol_graph.start_indices), axis=0)]).long()
    nodes_x = torch.tensor(i_mol_graph.atom_features).float()
    e_attr = torch.tensor(np.concatenate((i_mol_graph.bond_features, i_mol_graph.bond_features), axis=0)).float()
    # yi =  y.float() # normalize_prop(torch.tensor(properties[i]).float())
    datai = Data(x=nodes_x, edge_index=e_index, edge_attr=e_attr)
    Datas.append(datai)

print('\nlen graphs:', len(Datas))

Num_node_features = len(Datas[0].x[0])
print('Num_node_features:', Num_node_features)

test_dataset = Datas

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)



for tar_i in range(0, 13):
    model_path = r"model/pt/" + name + "-target_"+str(tar_i)+".pkl"
    modeli =  GCN_cls_e(hidden_channels=32, Num_node_features=Num_node_features, num_classes=4)
    modeli.load_state_dict(torch.load(model_path))
    models.append(modeli)

for model in models:
        print('model i: ',models.index(model),'\n',  model, '\n')

print("cls models loaded.")
from data.load_data import output_answer

def test_output(loader):
    for model in models:
        model.eval()
    
    list_pred = []
    # list_y = []
    for data in loader:  # 批遍历测试集数据集。
        predi_list = []
        for model in models:
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)  # 一次前向传播   data.edge_attr,
            pred = out.argmax(dim=1)  # 使用概率最高的类别
            predi_list.append(pred.tolist())
        predi_tensor = torch.tensor(predi_list)
        # print(predi_tensor.shape)
    
        # assert len(predi_list)==13
        list_pred.append(predi_tensor)

    output_tensor = torch.cat(list_pred, dim=1)
    print(output_tensor.shape)
    output_tensor = output_tensor.T
    
    print('****total_pred:', output_tensor.shape)
    output_answer(name, np.array(output_tensor))


if __name__ == '__main__':
    test_output(test_loader)
    pass