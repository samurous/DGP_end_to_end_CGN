import sys
import logging
import os.path as osp
import torch
import random
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold, train_test_split
from optuna.samplers import TPESampler
import optuna

from DiseaseNet import DiseaseNet
from GeneNet import GeneNet
from TheModel import TheModel

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.info('Start')
##### Expreiment hyperparameters

#  NEGATIVE_SAMPLES determines how the negative examples for training are created.
#  Choose from {'random', 'random_only_disease_gene'}
#    * random: Choose a random (gene, disease) pair which is not in the positive set.
#    * random_only_disease_genes: Like random but gene must be assigned to at least one disease.
NEGATIVE_SAMPLES = 'random'
EXPERIMENT_SLUG = 'ex_10_disease_links_similarity_based'
device = torch.device('cuda')

# Load gene and disease network
HERE = osp.abspath('')
ROOT = osp.join(HERE, '..', '..')
DATA_SOURCE_PATH = osp.join(ROOT, 'data_sources')
GENE_DATASET_ROOT = osp.join(DATA_SOURCE_PATH, 'gene_net_dataset_fn_and_hpo_features')
DISEASE_DATASET_ROOT = osp.join(DATA_SOURCE_PATH, 'disease_net_no_hpo_sim_based')
RESULTS_STORAGE = osp.join(HERE, 'results', EXPERIMENT_SLUG)
MODEL_TMP_STORAGE = osp.join('/', 'var', 'tmp', 'dg_tmp')
sys.path.insert(0, osp.abspath(ROOT))


gene_dataset = GeneNet(
    root=GENE_DATASET_ROOT,
    humannet_version='FN',
    features_to_use=['hpo'],
    skip_truncated_svd=True
)

disease_dataset = DiseaseNet(
    root=DISEASE_DATASET_ROOT,
    hpo_count_freq_cutoff=40,
    edge_source='feature_similarity',
    feature_source=['disease_publications'],
    skip_truncated_svd=True,
    svd_components=2048,
    svd_n_iter=12
)

gene_net_data = gene_dataset[0]
disease_net_data = disease_dataset[0]
print(gene_net_data)
print(disease_net_data)

gene_net_data = gene_net_data.to(device)
disease_net_data = disease_net_data.to(device)

# Generate training data.
disease_genes = pd.read_table(
    osp.join(DATA_SOURCE_PATH, 'genes_diseases.tsv'),
    names=['EntrezGene ID', 'OMIM ID'],
    sep='\t',
    low_memory=False,
    dtype={'EntrezGene ID': pd.Int64Dtype()}
)

disease_id_index_feature_mapping = disease_dataset.load_disease_index_feature_mapping()
gene_id_index_feature_mapping = gene_dataset.load_node_index_mapping()

all_genes = list(gene_id_index_feature_mapping.keys())
all_diseases = list(disease_id_index_feature_mapping.keys())

# 1. generate positive pairs.
# Filter the pairs to only include the ones where the corresponding nodes are available.
# i.e. gene_id should be in all_genes and disease_id should be in all_diseases.
positives = disease_genes[
    disease_genes["OMIM ID"].isin(all_diseases) & disease_genes["EntrezGene ID"].isin(all_genes)
    ]
covered_diseases = list(set(positives['OMIM ID']))
covered_genes = list(set(positives['EntrezGene ID']))

# 2. Generate negatives.
# Pick equal amount of pairs not in the positives.
negatives_list = []
while len(negatives_list) < len(positives):
    if NEGATIVE_SAMPLES == 'random_only_disease_genes':
        gene_id = covered_genes[np.random.randint(0, len(covered_genes))]
    else:
        gene_id = all_genes[np.random.randint(0, len(all_genes))]
    disease_id = covered_diseases[np.random.randint(0, len(covered_diseases))]
    if not ((positives['OMIM ID'] == disease_id) & (positives['EntrezGene ID'] == gene_id)).any():
        negatives_list.append([disease_id, gene_id])
negatives = pd.DataFrame(np.array(negatives_list), columns=['OMIM ID', 'EntrezGene ID'])


def get_training_data_from_indexes(indexes, monogenetic_disease_only=False, multigenetic_diseases_only=False):
    train_tuples = set()
    for idx in indexes:
        pos = positives[positives['OMIM ID'] == covered_diseases[idx]]
        neg = negatives[negatives['OMIM ID'] == covered_diseases[idx]]
        if monogenetic_disease_only and len(pos) != 1:
            continue
        if multigenetic_diseases_only and len(pos) == 1:
            continue
        for index, row in pos.iterrows():
            train_tuples.add((row['OMIM ID'], row['EntrezGene ID'], 1))
        for index, row in neg.iterrows():
            train_tuples.add((row['OMIM ID'], row['EntrezGene ID'], 0))
    ## 2. Concat data.
    n = len(train_tuples)
    x_out = np.ones((n, 2))  # will contain (gene_idx, disease_idx) tuples.
    y_out = torch.ones((n,), dtype=torch.long)
    for i, (omim_id, gene_id, y) in enumerate(train_tuples):
        x_out[i] = (gene_id_index_feature_mapping[int(gene_id)], disease_id_index_feature_mapping[omim_id])
        y_out[i] = y
    return x_out, y_out


def train(
        max_epochs,
        early_stopping_window=5,
        info_each_epoch=1,
        folds=5,
        lr=0.0005,
        weight_decay=5e-4,
        test_on_all_genes=True,
        fc_hidden_dim=2048,
        gene_net_hidden_dim=512,
        disease_net_hidden_dim=512,
        neg_weight=0.5,
        pos_weight=0.5
):
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([neg_weight, pos_weight]).to(device))
    metrics = []
    dis_dict = {}
    fold = 0

    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(covered_diseases):
        fold += 1
        print(f'Generate training data for fold {fold}.')
        all_train_x, all_train_y = get_training_data_from_indexes(train_index)

        # Split into train and validation set.
        id_tr, id_val = train_test_split(range(len(all_train_x)), test_size=0.1, random_state=42)
        train_x = all_train_x[id_tr]
        train_y = all_train_y[id_tr].to(device)
        val_x = all_train_x[id_val]
        val_y = all_train_y[id_val].to(device)

        # Create the model
        model = TheModel(fc_hidden_dim, gene_net_hidden_dim, disease_net_hidden_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        print(f'Stat training fold {fold}/{folds}:')

        losses = {
            'train': [],
            'val': [],
            'AUC': 0,
            'TPR': None,
            'FPR': None
        }

        best_val_loss = 1e80
        for epoch in range(max_epochs):
            # Train model.
            model.train()
            optimizer.zero_grad()
            out = model(gene_net_data, disease_net_data, train_x)
            loss = criterion(out, train_y)
            loss.backward()
            optimizer.step()
            losses['train'].append(loss.item())

            # Validation.
            with torch.no_grad():
                model.eval()
                out = model(gene_net_data, disease_net_data, val_x)
                loss = criterion(out, val_y)
                current_val_loss = loss.item()
                losses['val'].append(current_val_loss)

                if epoch % info_each_epoch == 0:
                    print(
                        'Epoch {}, train_loss: {:.4f}, val_loss: {:.4f}'.format(
                            epoch, losses['train'][epoch], losses['val'][epoch]
                        )
                    )
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    torch.save(model.state_dict(), osp.join(MODEL_TMP_STORAGE, f'best_model_fold_{fold}.ptm'))

            # Early stopping
            if epoch > early_stopping_window:
                # Stop if validation error did not decrease
                # w.r.t. the past early_stopping_window consecutive epochs.
                last_window_losses = losses['val'][epoch - early_stopping_window:epoch]
                if losses['val'][-1] > max(last_window_losses):
                    print('Early Stopping!')
                    break

        # Compute the validation AUC for the current fold to be used in hyper-parameter optimization.
        model.load_state_dict(torch.load(osp.join(MODEL_TMP_STORAGE, f'best_model_fold_{fold}.ptm')))
        with torch.no_grad():
            predicted_probs = F.log_softmax(model(gene_net_data, disease_net_data, val_x).clone().detach(), dim=1)
            true_y = val_y
            fpr, tpr, _ = roc_curve(true_y.cpu().detach().numpy(), predicted_probs[:, 1].cpu().detach().numpy(),
                                    pos_label=1)
            roc_auc = auc(fpr, tpr)
            losses['TEST_Y'] = true_y.cpu().detach().numpy()
            losses['TEST_PREDICT'] = predicted_probs.cpu().numpy()
            losses['AUC'] = roc_auc
            losses['TPR'] = tpr
            losses['FPR'] = fpr
            print(f'Auc for fold: {fold}: {roc_auc}')
        metrics.append(losses)

    print('Done!')
    return metrics, dis_dict, model


disease_idx_to_omim_mapping = dict()
for omim_id, disease_idx in disease_id_index_feature_mapping.items():
    disease_idx_to_omim_mapping[disease_idx] = omim_id

gene_idx_entrez_id_mapping = dict()
for entrez_id, gene_idx in gene_id_index_feature_mapping.items():
    gene_idx_entrez_id_mapping[gene_idx] = entrez_id


def get_genes_assoc_to_omim_disease(omim_id):
    return positives[positives["OMIM ID"].isin([omim_id])]["EntrezGene ID"].values


torch.set_num_threads = 16
def objective(trial):
    # Reproducibility
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    early_stopping_window = trial.suggest_int('early_stopping', 10, 30)
    lr = trial.suggest_loguniform('learning_rate', 1e-6, 1e-3)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-3, 1)
    fc_hidden_dim = trial.suggest_int('fc_hidden', 2 ** 6, 2 ** 12)
    gene_net_hidden_dim = trial.suggest_int('gene_hidden', 2 ** 6, 2 ** 11)
    disease_net_hidden_dim = trial.suggest_int('disease_hidden', 2 ** 6, 2 ** 11)
    # neg_weight = trial.suggest_uniform('neg_weight', 0.3, 0.7)
    neg_weight = 0.5
    pos_weight = 1 - neg_weight

    metrics, dis_dict, model = train(
        max_epochs=300,
        early_stopping_window=early_stopping_window,
        folds=5,
        lr=lr,
        weight_decay=weight_decay,
        test_on_all_genes=False,
        fc_hidden_dim=fc_hidden_dim,
        gene_net_hidden_dim=gene_net_hidden_dim,
        disease_net_hidden_dim=disease_net_hidden_dim,
        neg_weight=neg_weight,
        pos_weight=pos_weight
    )

    return float(np.mean([m['AUC'] for m in metrics]))


study = optuna.create_study(
    study_name='optimize-dgp-hyperband-2',
    direction='maximize',
    storage='sqlite:///optuna-optimize-dgp-hyperband.db',
    load_if_exists=True,
    sampler=TPESampler(),
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=30,
        interval_steps=2
    )
)

n_trails = 5000
study.optimize(objective, n_trials=n_trails)
