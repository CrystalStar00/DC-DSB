import torch
import numpy as np
import scipy
import pandas as pd
import scanpy as sc
import anndata as ad
import torch.nn.functional as F
import warnings

import matplotlib.pyplot as plt
from typing import List,Dict
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import pairwise_distances
from multiprocessing import Pool,cpu_count
from tqdm import tqdm


_shared_true = None
_shared_pred = None
_shared_control = None
_shared_adata_ctrl = None


def PearsonCorr1d(y_true, y_pred):
    y_true_c = y_true - torch.mean(y_true)
    y_pred_c = y_pred - torch.mean(y_pred)
    pearson = torch.nanmean(
        torch.sum(y_true_c * y_pred_c)
        / torch.sqrt(torch.sum(y_true_c * y_true_c))
        / torch.sqrt(torch.sum(y_pred_c * y_pred_c))
    )
    return pearson


def _eval_one_condition(args):
    cond_list, cond_idx_tuple, cond_indices, gene_names, de_idx, ndde_idx, gene_dim = args
    true_sub = _shared_true[cond_indices]
    pred_sub = _shared_pred[cond_indices]
    control = _shared_control

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata_pred = ad.AnnData(pred_sub.numpy(), obs={"condition": ["pred"] * len(pred_sub)})
        adata_true = ad.AnnData(true_sub.numpy(), obs={"condition": ["stim"] * len(true_sub)})
        adata = ad.concat([adata_true, _shared_adata_ctrl])

        if gene_names is not None:
            adata.var.index = gene_names
            adata_pred.var.index = gene_names
        sc.tl.rank_genes_groups(adata, groupby="condition", method="wilcoxon")
        diff_genes = adata.uns["rank_genes_groups"]["names"]['stim']
        diff_genes_idx = [np.where(np.array(gene_names) == x)[0].item() for x in diff_genes]
        adata = ad.concat([adata, adata_pred])
        adata.obs_names_make_unique()

        scores = reg_mean_plot(
            adata,
            condition_key='condition',
            axis_keys={"x": "pred", "y": 'stim', "x1": "ctrl"},
            gene_list=diff_genes[:10],
            top_100_genes=diff_genes[:100],
            labels={"x": "predicted", "y": "ground truth", "x1": "ctrl"},
            path_to_save=None,
            title='DSB',
            show=False,
            legend=False,
        )

    true_mean = true_sub.mean(0)
    pred_mean = pred_sub.mean(0)
    control_mean = control.mean(0)
    true_delta_mean = true_mean - control_mean
    pred_delta_mean = pred_mean - control_mean

    scores.update({
        'mse': F.mse_loss(pred_mean, true_mean).item(),
        'mse_top_100': F.mse_loss(pred_mean[diff_genes_idx[:100]], true_mean[diff_genes_idx[:100]]).item(),
        'mse_delta': F.mse_loss(pred_delta_mean, true_delta_mean).item(),
        'corr': PearsonCorr1d(pred_mean, true_mean).item(),
        'corr_top_100': PearsonCorr1d(pred_mean[diff_genes_idx[:100]], true_mean[diff_genes_idx[:100]]).item(),
        'corr_delta': PearsonCorr1d(pred_delta_mean, true_delta_mean).item(),
    })

    for num_de in (5,10,20, 100):
        if num_de > len(de_idx) or num_de > gene_dim:
            continue
        idx = de_idx[:num_de]
        scores.update(de_eval(pred_mean[idx], true_mean[idx], control_mean[idx], f"de{num_de}"))

    if ndde_idx is not None:
        scores.update(de_eval(pred_mean[ndde_idx], true_mean[ndde_idx], control_mean[ndde_idx], "ndde20"))

    return scores

@torch.inference_mode()
def perturbation_eval(
    true,
    pred,
    control,
    true_conds=None,
    gene_names=None,
    path_to_save=None,
    de_gene_idx_dict=None,
    ndde20_idx_dict=None,
    de_gene_idx=None,
    ndde20_idx=None,
    batch_size=8,
    parallel: bool = False,
):
    global _shared_true, _shared_pred, _shared_control, _shared_adata_ctrl

    if true_conds is not None:  # summarize condition wise evaluation，进行条件汇总评估
        assert de_gene_idx_dict is not None, "GEARS eval require DE gene index dict"
        assert ndde20_idx_dict is not None, "GEARS eval require top20 none dropout DE gene index dict"
        if path_to_save:
            warnings.warn(
                f"Cant save with multiple conds, got {path_to_save=}. Ignoring save option",
                UserWarning,
                stacklevel=2,
            )

        _shared_true = true.cpu()
        _shared_pred = pred.cpu()
        _shared_control = control.cpu()
        _shared_adata_ctrl = ad.AnnData(control.cpu().numpy(),
                                        obs={'condition': ["ctrl"] * len(control)})

        unique_true_conds = true_conds.unique(dim=0)    # 获取唯一的条件

        args_list = []

        for cond in unique_true_conds:
            cond_list = cond.tolist()
            cond_idx_tuple = tuple(i for i in cond_list if i != -1)
            cond_indices = (true_conds == cond).all(1).nonzero(as_tuple=True)[0].tolist()
            args_list.append((
                cond_list,
                cond_idx_tuple,
                cond_indices,
                gene_names,
                de_gene_idx_dict[cond_idx_tuple],
                ndde20_idx_dict[cond_idx_tuple],
                true.shape[1]
            ))

        score_dict_list = []
        if parallel:
            for i in range(0, len(args_list), batch_size):
                batch = args_list[i:i + batch_size]
                with Pool(processes=min(cpu_count(), len(batch))) as pool:
                    result = list(tqdm(
                        pool.imap(_eval_one_condition, batch),
                        total=len(batch),
                        desc=f"Evaluating conditions {i}~{i + len(batch) - 1}",
                    ))
                    score_dict_list.extend(result)
        else:
            for args in tqdm(args_list, desc="Evaluating conditions (serial)"):
                result = _eval_one_condition(args)
                score_dict_list.append(result)


        score_dict = dict(zip([tuple(a[0]) for a in args_list], score_dict_list))
        scores = reduce_score_dict_list(score_dict_list)
        return score_dict, scores

    return _eval_one_condition((
        [-1], (-1,), list(range(len(true))),
        gene_names, de_gene_idx, ndde20_idx, true.shape[1]
    ))



def de_eval(true, pred, ctrl, name):
    true_delta = true - ctrl
    pred_delta = pred - ctrl
    return {
        # MSE
        f'mse_{name}': F.mse_loss(pred, true).item(),
        f'mse_delta_{name}': F.mse_loss(pred_delta, true_delta).item(),
        # Correlation
        f'corr_{name}': PearsonCorr1d(pred, true).item(),
        f'corr_delta_{name}': PearsonCorr1d(pred_delta, true_delta).item(),
    }


def reg_mean_plot(adata, condition_key, axis_keys, labels, path_to_save="./reg_mean.pdf",
                  gene_list=None, top_100_genes=None, show=False, legend=True, title=None,
                  x_coeff=3, y_coeff=0, fontsize=14, **kwargs):
    """
        Adapted from https://github.com/theislab/scgen-reproducibility/blob/master/code/scgen/plotting.py
        Plots mean matching figure for a set of specific genes.

        # Parameters
            adata: `~anndata.AnnData`
                Annotated Data Matrix.
            condition_key: basestring
                Condition state to be used.
            axis_keys: dict
                dictionary of axes labels.
            path_to_save: basestring
                path to save the plot.
            gene_list: list
                list of gene names to be plotted.
            show: bool
                if `True`: will show to the plot after saving it.
    """
    import seaborn as sns
    from adjustText import adjust_text
    sns.set()
    sns.set(color_codes=True)
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (6, 6)))
    if scipy.sparse.issparse(adata.X):
        adata.X = adata.X.A
    diff_genes = top_100_genes
    stim = adata[adata.obs[condition_key] == axis_keys["y"]]
    pred = adata[adata.obs[condition_key] == axis_keys["x"]]

    if diff_genes is not None:
        if hasattr(diff_genes, "tolist"):
            diff_genes = diff_genes.tolist()
        adata_diff = adata[:, diff_genes]
        stim_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["y"]]
        pred_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["x"]]
        x_diff = np.average(pred_diff.X, axis=0)
        y_diff = np.average(stim_diff.X, axis=0)
        m, b, r_value_diff, p_value_diff, std_err_diff = scipy.stats.linregress(x_diff, y_diff)
        # print(r_value_diff ** 2)
    x = np.average(pred.X, axis=0)
    y = np.average(stim.X, axis=0)
    m, b, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    # print(r_value ** 2)
    df = pd.DataFrame({axis_keys["x"]: x, axis_keys["y"]: y})

    if path_to_save:
        ax = sns.regplot(x=axis_keys["x"], y=axis_keys["y"], data=df, ax=ax, scatter_kws={'rasterized': True})
        ax.tick_params(labelsize=fontsize)
        if "range" in kwargs:
            start, stop, step = kwargs.get("range")
            ax.set_xticks(np.arange(start, stop, step))
            ax.set_yticks(np.arange(start, stop, step))

        ax.set_xlabel(labels["x"], fontsize=fontsize)
        ax.set_ylabel(labels["y"], fontsize=fontsize)

    if "x1" in axis_keys.keys():
        ctrl = adata[adata.obs[condition_key] == axis_keys["x1"]]
        x1 = np.average(ctrl.X, axis=0)
        x_delta = x - x1
        y_delta = y - x1
        _, _, r_value_delta, _, _ = scipy.stats.linregress(x_delta, y_delta)
        if diff_genes is not None:
            ctrl_diff = ctrl[:, diff_genes]
            x1_diff = np.average(ctrl_diff.X, axis=0)
            x_delta_diff = x_diff - x1_diff
            y_delta_diff = y_diff - x1_diff
            _, _, r_value_delta_diff, _, _ = scipy.stats.linregress(x_delta_diff, y_delta_diff)
        # _p2 = pyplot.scatter(x, y1, marker="*", c="red", alpha=.5, label=f"{axis_keys['x']}-{axis_keys['y1']}")

    if path_to_save:
        if gene_list is not None:
            texts = []
            for i in gene_list:
                j = adata.var_names.tolist().index(i)
                x_bar = x[j]
                y_bar = y[j]
                texts.append(ax.text(x_bar, y_bar, i, fontsize=11, color="black"))
                ax.plot(x_bar, y_bar, 'o', color="red", markersize=5)
                # if "y1" in axis_keys.keys():
                # y1_bar = y1[j]
                # pyplot.text(x_bar, y1_bar, i, fontsize=11, color="black")
        if gene_list is not None:
            adjust_text(texts, x=x, y=y, ax=ax,  # 明确传入 ax
                        arrowprops=dict(arrowstyle="->", color='grey', lw=0.5),
                        force_points=(0.0, 0.0))
        if legend:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if title is None:
            ax.set_title("", fontsize=fontsize)
        else:
            ax.set_title(title, fontsize=fontsize)

        ax.text(x_coeff, y_coeff, r'$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= ' +
                f"{r_value ** 2:.4f}", fontsize=kwargs.get("textsize", fontsize))
        if diff_genes is not None:
            ax.text(x_coeff, y_coeff + 0.6, r'$\mathrm{R^2_{\mathrm{\mathsf{top\ 100\ DEGs}}}}$= ' +
                    f"{r_value_diff ** 2:.4f}", fontsize=kwargs.get("textsize", fontsize))
        if path_to_save is not None:
            fig.savefig(f"{path_to_save}", bbox_inches='tight', dpi=100)
        # if show:
        #     plt.show()
        plt.close()

    scores = {'R^2': r_value ** 2}
    if diff_genes is not None:
        scores['R^2_top_100'] = r_value_diff ** 2
    if "x1" in axis_keys.keys():
        scores['R^2_delta'] = r_value_delta ** 2
        if diff_genes is not None:
            scores['R^2_delta_top_100'] = r_value_delta_diff ** 2
    return scores

def reduce_score_dict_list(score_dict_list: List[Dict[str, float]]) -> Dict[str, float]:
    assert isinstance(score_dict_list, list)

    score_keys = sorted(score_dict_list[0])
    assert all(sorted(i) == score_keys for i in score_dict_list), "All score dicts must contain same score keys"

    scores = {score_key: np.mean([i[score_key] for i in score_dict_list]) for score_key in score_keys}

    return scores


def deep_analysis(dataset, scores, splittag):
    subgroup_split=splittag+"_subgroup"
    subgroup = dataset.subgroup[subgroup_split]

    seen = set()
    unique_pert_val = []
    for data in dataset.cell_graphs[splittag]:
        key = (tuple(data.pert_idx), data.pert)
        if key not in seen:
            unique_pert_val.append(key)
            seen.add(key)

    updated_scores = {}
    for key, value in scores.items():
        key_tuple = tuple(key.tolist())
        for seen_item in seen:
            if key_tuple == seen_item[0]:
                updated_scores[seen_item[1]] = value
                break

    metric_list = ["mse","mse_de20","corr","corr_delta","corr_de20"]

    subgroup_analysis = {}
    for name in subgroup.keys():
        subgroup_analysis[name] = {}
        for metric in metric_list:
            subgroup_analysis[name][metric] = []

    for pert, value in updated_scores.items():
        for name, pert_list in subgroup.items():
            if pert in pert_list:
                for metric in metric_list:
                    subgroup_analysis[name][metric].append(updated_scores[pert][metric])
                break

    for name, reuslt in subgroup_analysis.items():
        for metric in reuslt.keys():
            subgroup_analysis[name][metric] = np.mean(subgroup_analysis[name][metric])

    return subgroup_analysis


