import json
import os
from datetime import datetime
from time import time

import dgl
import torch
import torch.nn.functional as F
from dgl.data import LegacyTUDataset
from dgl.dataloading import GraphDataLoader
from torch import Tensor
from torch.utils.data import random_split

from data_preprocess import degree_as_feature, node_label_as_feature
from networks import GraphClassifier
from utils import get_stats, parse_args


def compute_loss(cls_logits:Tensor, labels:Tensor,
                 logits_s1:Tensor, logits_s2:Tensor,
                 epoch:int, total_epochs:int, device:torch.device):
    # classification loss
    classify_loss = F.nll_loss(cls_logits, labels.to(device))

    # loss for vertex infomax pooling
    scale1, scale2 = logits_s1.size(0) // 2, logits_s2.size(0) // 2
    s1_label_t, s1_label_f = torch.ones(scale1), torch.zeros(scale1)
    s2_label_t, s2_label_f = torch.ones(scale2), torch.zeros(scale2)
    s1_label = torch.cat((s1_label_t, s1_label_f), dim=0).to(device)
    s2_label = torch.cat((s2_label_t, s2_label_f), dim=0).to(device)

    pool_loss_s1 = F.binary_cross_entropy_with_logits(logits_s1, s1_label)
    pool_loss_s2 = F.binary_cross_entropy_with_logits(logits_s2, s2_label)
    pool_loss = (pool_loss_s1 + pool_loss_s2) / 2
    
    loss = classify_loss + (2 - epoch / total_epochs) * pool_loss

    return loss


def train(model:torch.nn.Module, optimizer, trainloader,
          device, curr_epoch, total_epochs):
    model.train()

    total_loss = 0.
    num_batches = len(trainloader)

    for batch in trainloader:
        optimizer.zero_grad()
        batch_graphs, batch_labels = batch
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.long().to(device)
        out, l1, l2 = model(batch_graphs, 
                            batch_graphs.ndata["feat"])
        loss = compute_loss(out, batch_labels, l1, l2,
                            curr_epoch, total_epochs, device)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss / num_batches


@torch.no_grad()
def test(model:torch.nn.Module, loader, device):
    model.eval()

    correct = 0.
    num_graphs = 0

    for batch in loader:
        batch_graphs, batch_labels = batch
        num_graphs += batch_labels.size(0)
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.long().to(device)
        out, _, _ = model(batch_graphs, batch_graphs.ndata["feat"])
        pred = out.argmax(dim=1)
        correct += pred.eq(batch_labels).sum().item()

    return correct / num_graphs


def main(args):
    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    dataset = LegacyTUDataset(args.dataset, raw_dir=args.dataset_path)

    # add self loop. We add self loop for each graph here since the function "add_self_loop" does not
    # support batch graph.
    for i in range(len(dataset)):
        dataset.graph_lists[i] = dgl.remove_self_loop(dataset.graph_lists[i])
        dataset.graph_lists[i] = dgl.add_self_loop(dataset.graph_lists[i])
    
    # preprocess: use node degree/label as node feature
    if args.degree_as_feature:
        dataset = degree_as_feature(dataset)
        mode = "concat"
    else:
        mode = "replace"
    dataset = node_label_as_feature(dataset, mode=mode)

    num_training = int(len(dataset) * 0.9)
    num_test = len(dataset) - num_training
    train_set, test_set = random_split(dataset, [num_training, num_test])

    train_loader = GraphDataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = GraphDataLoader(test_set, batch_size=args.batch_size, num_workers=1)

    device = torch.device(args.device)
    
    # Step 2: Create model =================================================================== #
    num_feature, num_classes, _ = dataset.statistics()
    args.in_dim = int(num_feature)
    args.out_dim = int(num_classes)
    args.edge_feat_dim = 0 # No edge feature in datasets that we use.
    
    model = GraphClassifier(args).to(device)

    # Step 3: Create training components ===================================================== #
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True, weight_decay=args.weight_decay)

    # Step 4: training epoches =============================================================== #
    best_test_acc = 0.0
    best_epoch = -1
    train_times = []
    for e in range(args.epochs):
        s_time = time()
        train_loss = train(model, optimizer, train_loader, device,
                           e, args.epochs)
        train_times.append(time() - s_time)
        test_acc = test(model, test_loader, device)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = e + 1

        if (e + 1) % args.print_every == 0:
            log_format = "Epoch {}: loss={:.4f}, test_acc={:.4f}, best_test_acc={:.4f}"
            print(log_format.format(e + 1, train_loss, test_acc, best_test_acc))
    print("Best Epoch {}, final test acc {:.4f}".format(best_epoch, best_test_acc))
    return best_test_acc, sum(train_times) / len(train_times)


if __name__ == "__main__":
    args = parse_args()
    res = []
    train_times = []
    for i in range(args.num_trials):
        print("Trial {}/{}".format(i + 1, args.num_trials))
        acc, train_time = main(args)
        # acc, train_time = 0, 0
        res.append(acc)
        train_times.append(train_time)

    mean, err_bd = get_stats(res, conf_interval=False)
    print("mean acc: {:.4f}, error bound: {:.4f}".format(mean, err_bd))

    out_dict = {"hyper-parameters": vars(args),
                "result_date": str(datetime.now()),
                "result": "{:.4f}(+-{:.4f})".format(mean, err_bd),
                "train_time": "{:.4f}".format(sum(train_times) / len(train_times)),
                "details": res}

    with open(os.path.join(args.output_path, "{}.log".format(args.dataset)), "w") as f:
        json.dump(out_dict, f, sort_keys=True, indent=4)
