import argparse
import torch
import dataloader as dl
import torch.nn.functional as F
import numpy as np
from models import GatHyper, SageHyper, GcnHyper
import test as tt

def main(args):
    if args.dataset == 'all':
        ds_names = ['Cora','Citeseer','Photo','Actor','chameleon','Squirrel']
    else:
        ds_names = [args.dataset]
    if args.backbone in ['all','Gcn','Gat','Sage']:
        if args.backbone == 'all':
            backbones = [b+'Hyper' for b in ['Gcn','Gat','Sage']]
        else:
            backbones = [args.backbone+'Hyper'] 
    else:
        return
    
    for ds in ds_names:
        for babo in backbones:
            babotrain_acc={babo:[i for i in range(args.run_times)]}
            babovalid_acc={babo:[i for i in range(args.run_times)]}
            babotest_acc={babo:[i for i in range(args.run_times)]}
            babowf1={babo:[i for i in range(args.run_times)]}
            f2=open('results/'+ds+babo+'_scores.txt', 'w+')
            f2.write('{0:7} {1:7}\n'.format(ds,babo))
            f2.write('{0:7} {1:7} {2:7} {3:7} {4:7}\n'.format('run','train','valid','m-f1','w-f1'))
            f2.flush()
            for run in range(args.run_times):
                dataset,data,train_mask,val_mask,test_mask = dl.select_dataset(ds, args.split)
                model,data = globals()[babo].call(data,dataset.name,data.x.size(1),dataset.num_classes,args.hid_dim)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                best_val_acc = test_acc = 0.0
                best_val_loss = np.inf
                for epoch in range(1, args.epoch+1): 
                    model.train()
                    optimizer.zero_grad()
                    F.nll_loss(model(data,args.loss_hp)[train_mask], data.y[train_mask]).backward()
                    optimizer.step()

                    train_acc,val_acc,tmp_test_acc,val_loss,tmp_w_f1 = tt.test(model, data, train_mask, val_mask, test_mask, args.loss_hp)
                    #print("acc:", train_acc,val_acc,tmp_test_acc,val_loss.item(),tmp_w_f1)
                    if val_acc>=best_val_acc:
                        train_re=train_acc
                        best_val_acc=val_acc
                        test_acc=tmp_test_acc             
                        w_f1 = tmp_w_f1
                        best_val_loss=val_loss
                        wait_step=0
                    else:
                        wait_step += 1
                        if wait_step == args.stop_step:
                            #print('Early stop! Validate-- Min loss: ', best_val_loss, ', Max f1-score: ', best_val_acc)
                            break
                del model
                del data
                babotrain_acc[babo][run]=train_re
                babovalid_acc[babo][run]=best_val_acc
                babotest_acc[babo][run]=test_acc
                babowf1[babo][run]=w_f1
                log ='Epoch: 200, dataset name: '+ ds + ', Backbone: '+ babo + ', Test: {0:.4f} {1:.4f}\n'
                print((log.format(babotest_acc[babo][run],babowf1[babo][run])))
                f2.write('{0:4d} {1:4f} {2:4f} {3:4f} {4:4f}\n'.format(run,babotrain_acc[babo][run],babovalid_acc[babo][run],babotest_acc[babo][run],babowf1[babo][run]))
                f2.flush()
            f2.write('{0:4} {1:4f} {2:4f} {3:4f} {4:4f}\n'.format('std',np.std(babotrain_acc[babo]),np.std(babovalid_acc[babo]),np.std(babotest_acc[babo]),np.std(babowf1[babo])))
            f2.write('{0:4} {1:4f} {2:4f} {3:4f} {4:4f}\n'.format('mean',np.mean(babotrain_acc[babo]),np.mean(babovalid_acc[babo]),np.mean(babotest_acc[babo]),np.mean(babowf1[babo])))
            f2.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperbolic Geometric Hierarchy-IMBAlance Learning')
    parser.add_argument("--dataset", '-d', type=str, default="Cora", help="all,Cora,Citeseer,Photo,Actor,chameleon,Squirrel")
    parser.add_argument("--backbone", '-b', type=str, default="Gcn", help="all,Gcn,Gat,Sage")
    parser.add_argument("--split", '-s', type=str, default=0, 
                        help="Way of train-set split: 0~5(random,(0.5,1),(0,0.05),(0.66,1),(0.33,0.66),(0,0.33))")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU index. Default: -1, using CPU.")
    parser.add_argument("--hid_dim", type=int, default=256, help="Hidden layer dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--epoch", type=int, default=200, help="Number of epochs. Default: 200")
    parser.add_argument("--run_times", type=int, default=10, help="Run times")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate. Default: 0.01")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay. Default: 0.0005")
    parser.add_argument("--loss_hp", type=float, default=1, help="Loss hyper-parameters (alpha). Default: 1")
    #parser.add_argument('--early_stop', action='store_true', default=True, help="Indicates whether to use early stop")
    parser.add_argument('--stop_step', default=100, help="Step of early stop")

    args = parser.parse_args()
    print(args)
    main(args)
