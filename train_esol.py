import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
from typing import Tuple, List
from functools import reduce

from data.load_data import load_data
from data.encode import num_atom_features, num_bond_features
from model.GNN import ThisIsNotAGNNAtAll

SEED = 0
USE_CUDA = False
LEARNING_RATE = 3e-3
LEARNING_RATE_DECAY = 0.98  # 0.98

WEIGHT_DECAY = 1e-4
EPOCH = 600
BATCH_SIZE = 32

config = {
    'HIDDEN_DIM': 128,
}

np.random.seed(SEED)
torch.manual_seed(SEED)
if USE_CUDA:
    torch.cuda.manual_seed(USE_CUDA)

print('\tLoading Data...')
list_mol_graph, properties = load_data('Lipop')
# exit()
n_mol = len(list_mol_graph)
list_torch_graph = [(
    torch.FloatTensor(mol_graph.atom_features),
    torch.FloatTensor(mol_graph.bond_features),
    mol_graph.start_indices,
    mol_graph.end_indices
) for mol_graph in list_mol_graph]
properties = torch.FloatTensor(properties)
mean = torch.mean(properties, dim=0, keepdim=True)
std = torch.std(properties, dim=0, keepdim=True)

# exit()
seq = np.random.permutation(n_mol)
train_seq = seq[:int(n_mol * 0.8)]
validate_seq = seq[int(n_mol * 0.8):-int(n_mol * 0.1)]
test_seq = seq[-int(n_mol * 0.1):]
train_ltg, validate_ltg, test_ltg = (
    [list_torch_graph[i] for i in train_seq],
    [list_torch_graph[i] for i in validate_seq],
    [list_torch_graph[i] for i in test_seq]
)
train_ppts, validate_ppts, test_ppts = properties[train_seq], properties[validate_seq], properties[test_seq]


def normalize_prop(p: torch.Tensor) -> torch.Tensor:
    return (p - mean) / std


def denormalize_prop(p: torch.Tensor) -> torch.Tensor:
    return p * std + mean


print('\tBuilding Model...')
model = ThisIsNotAGNNAtAll(
    atom_dim=num_atom_features(),
    bond_dim=num_bond_features(),
    output_dim=1,
    config=config,
    use_cuda=USE_CUDA
)
if USE_CUDA:
    model.cuda()

print('\tStructure:')
n_param = 0
for name, param in model.named_parameters():
    print(f'\t\t{name}: {param.shape}')
    n_param += reduce(lambda x, y: x * y, param.shape)
print(f'\t# Parameters: {n_param}')

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=LEARNING_RATE_DECAY)


def train(ltg: List[Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]], ppts: torch.Tensor):
    model.train()
    optimizer.zero_grad()
    list_pred = []
    list_prop = []
    for i, (af, bf, us, vs) in enumerate(ltg):
        if USE_CUDA:
            af, bf = af.cuda(), bf.cuda()
        pred = model.forward(af, bf, us, vs)
        list_pred.append(pred)
        list_prop.append(ppts[i].cuda() if USE_CUDA else ppts[i])
        if len(list_pred) >= BATCH_SIZE or i == len(ltg) - 1:
            batch_pred = torch.vstack(list_pred)
            batch_pred = denormalize_prop(batch_pred)
            batch_prop = torch.vstack(list_prop)
            loss = F.mse_loss(batch_pred, batch_prop)
            loss.backward()
            optimizer.step()
            list_pred.clear()
            list_prop.clear()


def evaluate(ltg: List[Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]], ppts: torch.Tensor) -> float:
    model.eval()
    list_pred = []
    for i, (af, bf, us, vs) in enumerate(ltg):
        if USE_CUDA:
            af, bf = af.cuda(), bf.cuda()
        pred = model.forward(af, bf, us, vs)
        list_pred.append(pred.cpu().detach())
    
    total_pred = torch.vstack(list_pred)
    total_pred = denormalize_prop(total_pred)
    rmse = torch.sqrt(F.mse_loss(total_pred, ppts))
    rmse = float(rmse)
    # print(f'\t\t\tRMSE----------: {rmse:.4f}')
    return rmse


MODEL_DICT_DIR = 'model/pt'
if not os.path.isdir(MODEL_DICT_DIR):
    os.mkdir(MODEL_DICT_DIR)

best_metric = 1e9
best_test_ans = 0
best_epoch = 0

f = open("ori_output.txt", "w", encoding='utf8')

for epoch in range(0,   EPOCH): # 20):
    if epoch % 10 == 0:
        print(f'##### IN EPOCH {epoch} #####')
        f.write(f'##### IN EPOCH {epoch} #####'+'\n')
        print('\tCurrent LR: {:.3e}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        f.write('\tCurrent LR: {:.3e}'.format(optimizer.state_dict()['param_groups'][0]['lr']) + '\n')
        print('\t\tTraining:')
    t0 = time.time()
    if epoch:
        train(train_ltg, train_ppts)
        scheduler.step()
    t1 = time.time()
    
    rmse_train = evaluate(train_ltg, train_ppts)
    
    t2 = time.time()

    rmse_val = evaluate(validate_ltg, validate_ppts)
    rmse_test = evaluate(test_ltg, test_ppts)
    
    if epoch % 10 == 0:
        # print(epoch)
        print('\t\tEvaluating Train:', f'\tRMSE----------: {rmse_train:.4f}')
        print('\t\tEvaluating Valid:', f'\tRMSE----------: {rmse_val:.4f}')
        
        
        print('\t\tEvaluating Test: ', f'\tRMSE---------------: { rmse_test:.4f}')
        
        f.write(f'train, valid, test: {rmse_train:.4f} {rmse_val:4f} {rmse_test:4f}'+'\n')
        
        
        # print('\tTraining Time: {}'.format(int(t1 - t0)))
        # print('\tEvaluating Time: {}'.format(int(t2 - t1)))
    
    # 不需要每次都存储模型
    if epoch > EPOCH-10 and rmse_val < best_metric:  # or epoch>175)
        best_metric = rmse_val
        best_test_ans = rmse_test
        best_epoch = epoch
        print(f'\tSaving Model...', 'best epoch {}, val RMSE: {:.4f}  test RMSE: {:.4f}'.format(best_epoch,  best_metric, best_test_ans))
        f.write(f'\tSaving Model...  ' +
              'best epoch {}, val RMSE: {:.4f}  test RMSE: {:.4f}'.format(best_epoch, best_metric, best_test_ans)+'\n')
        
        torch.save(model.state_dict(), f'{MODEL_DICT_DIR}/ThisIsNotAGNNAtAll-ESOL.pkl')
        

print('\nTrain finished, best epoch {} , bert_test_ans : {:.4f}'.format(best_epoch, best_test_ans) )
f.write('\nTrain finished, best epoch {} , bert_test_ans : {:.4f}'.format(best_epoch, best_test_ans) +'\n')
f.close()



