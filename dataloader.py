import torch_geometric.datasets as dt
import torch_geometric.transforms as T
import torch
import numpy as np
from dgl.data.utils import generate_mask_tensor, idx2mask
from sklearn.model_selection import train_test_split

def select_dataset(ds,spcial):
    if ds=='Cora' or ds=='Citeseer':
        ds_loader='Planetoid'
    elif ds=='Photo':
        ds_loader='Amazon'
    elif ds == 'chameleon' or ds == 'Squirrel':
        ds_loader='WikipediaNetwork'
    else:
        ds_loader=ds
    dataset=load_datas(ds_loader,ds,spcial)
    if ds == 'Actor':
        data=dataset.data
        dataset.name = ds
    else:
        data=dataset[0]

    train_mask=data.train_mask
    val_mask=data.val_mask
    test_mask=data.test_mask
    return dataset,data,train_mask,val_mask,test_mask

def load_datas(ds_loader,ds,spcial):
    if ds_loader=='Planetoid':
        dataset = dt.Planetoid(root='data/'+ds, name=ds, transform=T.NormalizeFeatures())
    else:
        dataset = getattr(dt, ds_loader)('data/'+ds,ds)

    if ds_loader == 'Actor':
        dataset.name = ds

    data = get_split(dataset, spcial)
    dataset.data = data
    return dataset

def get_split(dataset, spcial):
    data = dataset.data   
    values=np.load('hyperemb/'+dataset.name+'_values.npy')
    sorted, indices = torch.sort(torch.norm(torch.tensor(values),dim=1),descending=True)
    #train set split ratio 1:1:8
    if spcial == 1:#Top 50% in the Poincare weight
        train_idx, val_idx, test_idx = split_idx1(indices[:data.num_nodes//2],indices[data.num_nodes//2:], 0.2, 0.1, 42)
    elif spcial == 2:#Bottom 50%
        train_idx, val_idx, test_idx = split_idx1(indices[data.num_nodes//2:],indices[:data.num_nodes//2], 0.2, 0.1, 42)
    elif spcial == 3:#Top 33%
        train_idx, val_idx, test_idx = split_idx1(indices[:data.num_nodes//3],indices[data.num_nodes//3:], 0.3, 0.1, 42)
    elif spcial == 4:#Middle 33%
        remaining = torch.cat((indices[:data.num_nodes//3],indices[data.num_nodes//3+data.num_nodes//3:]))
        train_idx, val_idx, test_idx = split_idx1(indices[data.num_nodes//3:data.num_nodes//3+data.num_nodes//3],remaining, 0.3, 0.1, 42)
    elif spcial == 5:#Bottom 33%
        train_idx, val_idx, test_idx = split_idx1(indices[data.num_nodes//3+data.num_nodes//3:],indices[:data.num_nodes//3+data.num_nodes//3], 0.3, 0.1, 42)
    else:#random
        train_idx, val_idx, test_idx = split_idx(np.arange(data.num_nodes), 0.1, 0.1, 42)

    data.train_mask = generate_mask_tensor(idx2mask(train_idx, data.num_nodes))
    data.val_mask = generate_mask_tensor(idx2mask(val_idx, data.num_nodes))
    data.test_mask = generate_mask_tensor(idx2mask(test_idx, data.num_nodes))
    return data


def split_idx(samples, train_size, val_size, random_state=None):
    train, val = train_test_split(samples, train_size=train_size, random_state=random_state)
    if isinstance(val_size, float):
        val_size *= len(samples) / len(val)
    val, test = train_test_split(val, train_size=val_size, random_state=random_state)
    return train, val, test
    
def split_idx1(samples1, samples2, train_size, val_size, random_state=None):
    train, val = train_test_split(samples1, train_size=train_size, random_state=random_state)
    val = torch.cat((val,samples2))
    val, test = train_test_split(val, train_size=val_size, random_state=random_state)
    return train, val, test
