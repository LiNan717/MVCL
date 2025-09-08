import pickle
from sklearn import metrics
import random
import torch
import time
import torch.nn.functional as F
import argparse
import yaml
from yaml import SafeLoader
from time import perf_counter as t
import numpy as np
from mvcl import MVCL
from sklearn import linear_model
import warnings
from K_mean import kmeans
from torch_geometric.utils import dropout_adj
warnings.filterwarnings("ignore")
cross_val = 10


def LogReg(train_x, train_y, test_x):
    regr = linear_model.LogisticRegression(max_iter=10000)
    regr.fit(train_x, train_y.ravel())
    pre = regr.predict_proba(test_x)
    pre = pre[:, 1]
    return pre


def create_pos_matrix(adj, num_nodes):
    adj = adj.coalesce()
    pos_matrix = torch.zeros((num_nodes, num_nodes), device=adj.device, dtype=torch.float64)
    pos_matrix.fill_diagonal_(1)
    indices = adj.indices()
    pos_matrix[indices[0], indices[1]] = 1
    pos_matrix[indices[1], indices[0]] = 1
    return pos_matrix


def train(mask):
    model.train()
    optimizer.zero_grad()
    x = data.x
    ppiAdj_index = ppiAdj.coalesce().indices()
    pathAdj_index = pathAdj.coalesce().indices()
    goAdj_index = goAdj.coalesce().indices()
    seqAdj_index = seqAdj.coalesce().indices()
    # feature mask
    x_1 = F.dropout(x, drop_feature_rate_1)
    x_2 = F.dropout(x, drop_feature_rate_2)
    x_3 = F.dropout(x, drop_feature_rate_3)
    x_4 = F.dropout(x, drop_feature_rate_4)
    # edge dropout
    ppiAdj_index = dropout_adj(ppiAdj_index, p=drop_edge_rate_1, force_undirected=True)[0]
    pathAdj_index = dropout_adj(pathAdj_index, p=drop_edge_rate_2, force_undirected=True)[0]
    goAdj_index = dropout_adj(goAdj_index, p=drop_edge_rate_3, force_undirected=True)[0]
    seqAdj_index = dropout_adj(seqAdj_index, p=drop_edge_rate_4, force_undirected=True)[0]
    pred1, pred2, pred3, pred4, _, conloss = model(ppiAdj_index, pathAdj_index, goAdj_index, seqAdj_index, x_1, x_2,
                                                   x_3, x_4)
    loss1 = F.binary_cross_entropy_with_logits(pred1[mask], Y[mask])
    loss2 = F.binary_cross_entropy_with_logits(pred2[mask], Y[mask])
    loss3 = F.binary_cross_entropy_with_logits(pred3[mask], Y[mask])
    loss4 = F.binary_cross_entropy_with_logits(pred4[mask], Y[mask])
    loss = (0.1 * loss1 + 0.1 * loss2 + 0.1 * loss3 + 0.1 * loss4) + 0.6 * conloss  # + 0.1*loss4  + 0.15 * loss4
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(mask1, mask2):
    model.eval()
    ppiAdj_index = ppiAdj.coalesce().indices()
    pathAdj_index = pathAdj.coalesce().indices()
    goAdj_index = goAdj.coalesce().indices()
    seqAdj_index = seqAdj.coalesce().indices()
    pred1, pred2, pred3, pred4, emb, conloss = model(ppiAdj_index, pathAdj_index, goAdj_index, seqAdj_index, data.x,
                                                     data.x, data.x, data.x)
    loss1 = F.binary_cross_entropy_with_logits(pred1[mask1], Y[mask1])
    loss2 = F.binary_cross_entropy_with_logits(pred2[mask1], Y[mask1])
    loss3 = F.binary_cross_entropy_with_logits(pred3[mask1], Y[mask1])
    loss4 = F.binary_cross_entropy_with_logits(pred4[mask1], Y[mask1])
    loss = (0.1 * loss1 + 0.1 * loss2 + 0.1 * loss3 + 0.1 * loss4) + 0.6 * conloss
    train_x = torch.sigmoid(emb[mask1]).cpu().detach().numpy()
    train_y = Y[mask1].cpu().numpy()
    test_x = torch.sigmoid(emb[mask2]).cpu().detach().numpy()
    Yn = Y[mask2].cpu().numpy()
    pred = LogReg(train_x, train_y, test_x)
    precision, recall, _thresholds = metrics.precision_recall_curve(Yn, pred)
    area = metrics.auc(recall, precision)
    return metrics.roc_auc_score(Yn, pred), area, precision, recall, loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CPDB')
    parser.add_argument('--cancer_type', type=str, default='pan-cancer')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]
    dataPath = "data/" + args.dataset + "/"
    cancerType = args.cancer_type
    seed = config['seed']
    LR = config['LR']
    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_edge_rate_3 = config['drop_edge_rate_3']
    drop_edge_rate_4 = config['drop_edge_rate_4']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    drop_feature_rate_3 = config['drop_feature_rate_3']
    drop_feature_rate_4 = config['drop_feature_rate_4']
    tau = config['tau']
    EPOCH = config['EPOCH']
    threshold = config['threshold']
    num_clusters = config['num_clusters']

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data = torch.load(dataPath + args.dataset + "_data.pkl")
    device = torch.device('cuda:0')
    data = data.to(device)

    Y = torch.tensor(np.logical_or(data.y, data.y_te)).type(torch.FloatTensor).to(device)

    if cancerType == 'pan-cancer':
        x = data.x[:, :48]
    else:
        cancerType_dict = {
            'kirc': [0, 16, 32],
            'brca': [1, 17, 33],
            'prad': [3, 19, 35],
            'stad': [4, 20, 36],
            'hnsc': [5, 21, 37],
            'luad': [6, 22, 38],
            'thca': [7, 23, 39],
            'blca': [8, 24, 40],

            'esca': [9, 25, 41],
            'lihc': [10, 26, 42],
            'ucec': [11, 27, 43],
            'coad': [12, 28, 44],
            'lusc': [13, 29, 45],
            'cesc': [14, 30, 46],
            'kirp': [15, 31, 47]
        }
        x = data.x[:, cancerType_dict[cancerType]]
    print(data.x)
    # node2VEC feature
    dataz = torch.load(dataPath + "Str_feature.pkl")
    dataz = dataz.to(device)
    data.x = torch.cat((x, dataz), 1)  # 64D feature
    ppiAdj = torch.load(dataPath + 'ppi_adj.pkl').to(device)
    ppiAdj_self = torch.load(dataPath + 'ppi_selfloop.pkl').to(device)
    pathAdj = torch.load(dataPath + 'pathway.pkl').to(device)
    goAdj = torch.load(dataPath + 'GO.pkl').to(device)
    seqAdj = torch.load(dataPath + 'Seq_Sim.pkl').to(device)
    seqAdj = seqAdj.to(device)
    ppi_pos = ppiAdj_self.to_dense()
    diag_mask = torch.eye(ppi_pos.size(0), device=ppi_pos.device).bool()
    mask_ppi = (ppi_pos == 0)
    path_pos = create_pos_matrix(pathAdj, 13627) * mask_ppi + diag_mask
    go_pos = create_pos_matrix(goAdj, 13627) * mask_ppi + diag_mask
    seq_pos = diag_mask
    pos = [ppi_pos, path_pos, go_pos, seq_pos]
    cluster_ids, dis, cluster_centers = kmeans(x, num_clusters=num_clusters, distance='cosine')
    high_confidence = torch.min(dis, dim=1).values
    threshold0 = torch.sort(high_confidence).values[int(len(high_confidence) * threshold)]
    high_confidence_idx = np.argwhere(high_confidence < threshold0)[0]

    if args.dataset == 'CPDB':
        with open(dataPath + "k_sets.pkl", 'rb') as handle:
            k_sets = pickle.load(handle)
    else:
        k_sets = torch.load(dataPath + "k_sets.pkl")

    AUC = np.zeros(shape=(cross_val, 5))
    AUPR = np.zeros(shape=(cross_val, 5))
    AUC2 = np.zeros(shape=(cross_val, 5))
    AUPR2 = np.zeros(shape=(cross_val, 5))
    epoch = np.zeros(shape=(cross_val, 5))
    precision_list = []
    recall_list = []
    start_time = time.time()

    # pan-cancer
    print("---------Pan-cancer Train begin--------")
    for i in range(cross_val):
        for cv_run in range(5):
            print("----------------------- i: %d, cv_run: %d -------------------------" % (i + 1, cv_run + 1))
            start = t()
            y_tr, y_te, tr_mask, te_mask = k_sets[i][cv_run]
            model = MVCL(pos=pos,
                         tau=tau,
                         high_confidence_idx=high_confidence_idx,
                         cluster_ids=cluster_ids,
                         num_clusters=num_clusters,
                         gnn_outsize=100,
                         projection_hidden_size=300,
                         projection_size=100
                         ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            best_auc = 0
            best_aupr = 0
            train_loss = []
            test_loss = []
            epochs = []
            for train_epoch in range(1, EPOCH):
                loss_train = train(tr_mask)
                AUC[i][cv_run], AUPR[i][cv_run], precision, recall, loss_test = test(tr_mask, te_mask)
                train_loss.append(loss_train)
                test_loss.append(loss_test)
                epochs.append(train_epoch)
                print(
                    f'(T) | Epoch={train_epoch:03d}, loss={loss_train:.4f},AUC={AUC[i][cv_run]:.4f}, AUPR={AUPR[i][cv_run]:.4f}')
                if AUC[i][cv_run] > best_auc:
                    best_auc = AUC[i][cv_run]
                    epoch[i][cv_run] = train_epoch
                AUC2[i][cv_run] = best_auc
                if AUPR[i][cv_run] > best_aupr:
                    best_aupr = AUPR[i][cv_run]
                AUPR2[i][cv_run] = best_aupr
            np.savetxt("./AUC.txt", AUC2, delimiter="\t")
            np.savetxt("./AUPR.txt", AUPR2, delimiter="\t")
            np.savetxt("./epoch.txt", epoch, delimiter="\t")
            now = t()
        print(AUC2[i].mean())
        print(AUPR2[i].mean())
    end_time = time.time()
    total_time = end_time - start_time
    print(f"run time: {total_time:.2f} s")

    # specific cancer
    '''
    print("---------" + cancerType + " cancer Train begin--------")
    path = dataPath + "Specific cancer/"
    for i in range(cross_val):
        label, label_pos, label_neg = load_label_single(path)
        random.shuffle(label_pos)
        random.shuffle(label_neg)
        l = len(label)
        l1 = int(len(label_pos) / 5)
        l2 = int(len(label_neg) / 5)
        Y = label
        for cv_run in range(5):
            print("----------------------- i: %d, cv_run: %d -------------------------" % (i + 1, cv_run + 1))
            start = t()
            tr_mask, te_mask = sample_division_single(label_pos, label_neg, l, l1, l2, cv_run)
            model = MVCL(pos=pos,
                         tau=tau,
                         high_confidence_idx=high_confidence_idx,
                         cluster_ids=cluster_ids,
                         num_clusters=num_clusters,
                         gnn_outsize=50,
                         projection_hidden_size=150,
                         projection_size=50
                         ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            best_auc = 0
            best_aupr = 0
            train_loss = []
            test_loss = []
            epochs = []
            for train_epoch in range(1, EPOCH):
                loss = train(tr_mask)
                AUC[i][cv_run], AUPR[i][cv_run] = test(tr_mask, te_mask)
                print(
                    f'(T) | Epoch={train_epoch:03d}, loss={loss:.4f},AUC={AUC[i][cv_run]:.4f}, AUPR={AUPR[i][cv_run]:.4f}')

            np.savetxt("./AUC.txt", AUC, delimiter="\t")
            np.savetxt("./AUPR.txt", AUPR, delimiter="\t")

            print("this cv_run spend {:.2f}  s".format(now - start))
        print(AUC[i].mean())
        print(AUPR[i].mean())
    end_time = time.time()
    total_time = end_time - start_time
    print(f"run time: {total_time:.2f} s")
    '''
