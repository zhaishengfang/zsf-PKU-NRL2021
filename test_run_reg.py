# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
from model.myGNN import *
import torch
from torch_geometric.loader import DataLoader
from data.load_data import load_data
from torch_geometric.data import Data
import random

BATCH_SIZE = 16

# def get_data_model(name):

name = r"Lipop"
model_path = r"model/pt/"+name+"-reg.pkl"
list_mol_graph, properties = load_data(data_name=name, set_name = 'test')

properties = torch.FloatTensor(properties)
mean = torch.mean(properties, dim=0, keepdim=True)
std = torch.std(properties, dim=0, keepdim=True)

Datas = []
for i_mol_graph in list_mol_graph:
    # print(y)
    e_index = torch.tensor([np.concatenate((i_mol_graph.start_indices, i_mol_graph.end_indices), axis=0),
                    np.concatenate((i_mol_graph.end_indices, i_mol_graph.start_indices), axis=0)]).long()
    nodes_x = torch.tensor(i_mol_graph.atom_features).float()
    e_attr = torch.tensor(np.concatenate((i_mol_graph.bond_features , i_mol_graph.bond_features), axis=0)).float()
    # yi =  y.float() # normalize_prop(torch.tensor(properties[i]).float())
    datai = Data(x = nodes_x, edge_index=e_index, edge_attr=e_attr)
    Datas.append(datai)
    
print('\nlen graphs:', len(Datas))

Num_node_features = len(Datas[0].x[0])
print('Num_node_features:', Num_node_features)

test_dataset =Datas

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = GCN_reg_e(hidden_channels=32, Num_node_features=Num_node_features)

model.load_state_dict(torch.load(model_path))
print(model)

print("reg model loaded.")
from data.load_data import output_answer

def test_output(loader):
    model.eval()

    list_pred = []
    # list_y = []
    for data in loader:  # 批遍历测试集数据集。
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)  # 一次前向传播   data.edge_attr,
        # out = model(data)
        # print("out,---\n", out, '\n')
        print(out.shape)
        out = torch.flatten(out).tolist()
        # print(out.shape)
        # print("out2,---\n", out, '\n')
        list_pred.extend(out)
        # print("list_pred,---\n", list_pred, '\n')
        # print("list_y,---\n", list_y, '\n')

    total_pred = torch.tensor(list_pred).unsqueeze(1)
    
    print('****total_pred:', total_pred.shape)
    output_answer(name, np.array(total_pred))
    
    
    

    












if __name__ == '__main__':
    test_output(test_loader)
    pass