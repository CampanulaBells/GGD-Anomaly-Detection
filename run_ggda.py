import time
from sklearn.metrics import roc_auc_score
import json
import os
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import datetime

from tqdm import tqdm
import matplotlib.pyplot as plt
from modules.model import Model
from modules.utils import load_dataset, draw_line, set_random_seeds, ensure_path

parser = argparse.ArgumentParser(description='GGD Anomaly')
parser.add_argument('--expid', type=str, default='test_implementation')
parser.add_argument('--resultdir', type=str, default='results')
parser.add_argument('--eval_freq', type=int, default=1)

parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.0)

parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--n_hidden', type=int, default=256)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--num_epoch', type=int, default=250)

parser.add_argument('--subgraph_size', type=int, default=1)
parser.add_argument('--gnn_encoder', type=str, default='gcn')
parser.add_argument('--gamma', type=float, default=0.1)

args = parser.parse_args()

# Setup device and random seed
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
seeds = [i + 1 for i in range(args.runs)]

# Load dataset
g, features, ano_label, _, _ = load_dataset(args.dataset)
features = torch.FloatTensor(features).to(device)
g = g.to(device)

all_results = []
eval_results = []
path_expr = f'./{args.resultdir}/{args.expid}'
ensure_path(path_expr)
if args.eval_freq > 0:
    path_chart = f'{path_expr}/chart'
    ensure_path(path_chart)

ensure_path(path_expr)
for run in range(args.runs):
    seed = seeds[run]
    set_random_seeds(seed)
    # Create GGD model
    model = Model(
        g,
        features.shape[1],
        args.n_hidden,
        nn.PReLU(args.n_hidden),
        args.gnn_encoder,
        args.subgraph_size
    )
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    b_xent = nn.BCEWithLogitsLoss()

    cnt_wait = 0
    best = 1e9
    best_t = 0
    best_state = None

    dur = []  # training time
    loss_list = []  # loss

    tag = str(datetime.datetime.now().strftime("%m-%d %H%M%S"))

    epoch_list = []
    auc_pos_list = []

    label_positive = torch.zeros(1, g.number_of_nodes()).cuda()
    label_negative = torch.ones(1, g.number_of_nodes()).cuda()
    with tqdm(total=args.num_epoch) as pbar:
        pbar.set_description(f'Run {run} with random seed {seed}')
        for epoch in range(args.num_epoch):
            if args.eval_freq > 0 and epoch % args.eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    epoch_list.append(epoch)
                    s_positive = model(features)[0]
                    pos_prob = s_positive.detach()[0].cpu().numpy()
                    auc_pos_list.append(roc_auc_score(ano_label, pos_prob))
            model.train()
            t0 = time.time()
            optimizer.zero_grad()

            s_positive, s_negative, ggd_score_pos, ggd_score_neg, perm = model(features)

            loss_anomaly = b_xent(s_positive, label_positive) + b_xent(s_negative, label_negative)
            loss_ggd = b_xent(ggd_score_pos, label_positive) + b_xent(ggd_score_neg, label_negative)
            loss = (1 - args.gamma) * loss_anomaly + args.gamma * loss_ggd
            loss.backward()

            optimizer.step()
            comp_time = time.time() - t0
            dur.append(comp_time)
            loss_list.append(
                (loss.detach().cpu().item(), loss_anomaly.detach().cpu().item(), loss_ggd.detach().cpu().item()))

            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                print('Early stopping!')
                break
            if args.eval_freq > 0:
                auc = 0.5
                if len(auc_pos_list) > 0:
                    auc = auc_pos_list[-1]
                pbar.set_postfix(auc=auc, loss=loss.item())
            else:
                pbar.set_postfix(loss=loss.item())
            pbar.update(1)

    if args.eval_freq > 0:
        draw_line(plt, epoch_list, auc_pos_list, 'g', label="auc positive", max_marker="^")
        plt.ylabel('auc')
        plt.legend()
        plt.ylim(0, 1)
        plt.yticks(np.arange(0, 1, step=0.1))
        plt.savefig(f"{path_chart}/{run}.png", dpi=400)
        print(f"largest auc ({epoch_list[np.argmax(np.array(auc_pos_list))]}): {max(auc_pos_list)}")
        eval_results.append((seed, epoch_list[np.argmax(np.array(auc_pos_list))], max(auc_pos_list)))

    index_min_loss = np.argmin(np.array(loss_list)[:, 0])
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        s_positive = model(features)[0]
        pos_prob = s_positive.detach()[0].cpu().numpy()
    auc_score = roc_auc_score(ano_label, pos_prob)
    all_results.append((seed, best_t, auc_score))

    path_ckpt = f'{path_expr}/checkpoints'
    ensure_path(path_ckpt)
    torch.save(model.state_dict(), f'{path_ckpt}/exp_{run}.pkl')


with open(f"{path_expr}/res_expr.csv", "w") as fp:
    fp.write(','.join(["seed", "epoch", "auc"]) + "\n")
    for result in all_results:
        fp.write(','.join([str(x) for x in result]) + "\n")

    all_auc = [x[2] for x in all_results]
    mean_auc = sum(all_auc) / len(all_auc)
    print("Mean auc score", mean_auc)
    fp.write(','.join(["mean", "", str(mean_auc)]))

# Save results
with open(f"{path_expr}/auc={mean_auc}.txt", "w") as fp:
    fp.write(json.dumps(vars(args), indent=2))

if args.eval_freq > 0:
    with open(f"{path_expr}/res_eval.csv", "w") as fp:
        fp.write(','.join(["seed", "best_epoch", "auc"]) + "\n")
        for result in eval_results:
            fp.write(','.join([str(x) for x in result]) + "\n")

        all_auc = [x[2] for x in eval_results]
        mean_auc = sum(all_auc) / len(all_auc)
        print("Mean best auc score", mean_auc)
        fp.write(','.join(["mean", "", str(mean_auc)]))

