import gc

import torch
import numpy as np
import scipy
import pandas as pd
import scanpy as sc
import anndata as ad
import torch.nn.functional as F
import warnings
from torch import nn
from adjustText import adjust_text
from matplotlib import pyplot
from typing import Union, Optional,List,Any,Literal, Dict,Type
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics.pairwise import pairwise_distances
from multiprocessing import Pool


class WassersteinLoss(nn.Module):
    def __init__(self, epsilon=0.1, n_iters=5):
        super().__init__()
        self.epsilon = epsilon  # Entropic regularization
        self.n_iters = n_iters  # Sinkhorn迭代次数

    def forward(self, P, Q):
        # P, Q: (batch, genes) 需为非负值（无需归一化）
        cost_matrix = torch.cdist(P, Q, p=2)  # 欧式距离成本矩阵
        K = torch.exp(-cost_matrix / self.epsilon)

        # Sinkhorn迭代
        u = torch.ones_like(P[:, 0]) / P.size(1)
        for _ in range(self.n_iters):
            v = 1.0 / (K @ u.unsqueeze(-1)).squeeze()
            u = 1.0 / (K @ v.unsqueeze(-1)).squeeze()

        transport_plan = u.unsqueeze(-1) * K * v.unsqueeze(1)
        wasserstein = torch.sum(transport_plan * cost_matrix)
        return wasserstein


def PearsonCorr(y_pred, y_true):
    y_true_c = y_true - torch.mean(y_true, 1)[:, None]
    y_pred_c = y_pred - torch.mean(y_pred, 1)[:, None]
    pearson = torch.nanmean(
        torch.sum(y_true_c * y_pred_c, 1)
        / torch.sqrt(torch.sum(y_true_c * y_true_c, 1))
        / torch.sqrt(torch.sum(y_pred_c * y_pred_c, 1))
    )
    return pearson


def PearsonCorr1d(y_true, y_pred):
    y_true_c = y_true - torch.mean(y_true)
    y_pred_c = y_pred - torch.mean(y_pred)
    pearson = torch.nanmean(
        torch.sum(y_true_c * y_pred_c)
        / torch.sqrt(torch.sum(y_true_c * y_true_c))
        / torch.sqrt(torch.sum(y_pred_c * y_pred_c))
    )
    return pearson


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
):
    if true_conds is not None:  # summarize condition wise evaluation，进行条件汇总评估
        assert de_gene_idx_dict is not None, "GEARS eval require DE gene index dict"
        assert ndde20_idx_dict is not None, "GEARS eval require top20 none dropout DE gene index dict"
        if path_to_save:
            warnings.warn(
                f"Cant save with multiple conds, got {path_to_save=}. Ignoring save option",
                UserWarning,
                stacklevel=2,
            )
        unique_true_conds = true_conds.unique(dim=0)    # 获取唯一的条件
        score_dict = {}
        score_dict_list = []
        for cond in unique_true_conds:  # 对每个唯一条件进行评估
            cond_ind = (true_conds == cond).all(1)  # 获取当前条件的索引
            true_sub, pred_sub = true[cond_ind], pred[cond_ind] # 获取对应条件的 true 和 pred 子集
            cond_idx_tuple = tuple(i for i in cond.tolist() if i != -1)  # XXX: specificially designed for GEARS
            cond_score=perturbation_eval(true_sub, pred_sub, control, gene_names=gene_names,
                                                     de_gene_idx=de_gene_idx_dict[cond_idx_tuple],
                                                     ndde20_idx=ndde20_idx_dict[cond_idx_tuple])
            mmd=calculate_mmd(pred_sub.numpy(),true_sub.numpy())
            cond_score['mmd']=mmd
            score_dict[cond]=cond_score
            score_dict_list.append(cond_score)
        scores = reduce_score_dict_list(score_dict_list)    # 合并所有条件的评估结果

        return score_dict,scores


    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 将预测结果、真实结果和控制组结果转换为 AnnData 对象
        adata_pred = ad.AnnData(pred.detach().cpu().numpy(),
                                obs={'condition': ["pred"] * len(pred)})
        adata_true = ad.AnnData(true.detach().cpu().numpy(),
                                obs={'condition': ["stim"] * len(true)})
        adata_ctrl = ad.AnnData(control.detach().cpu().numpy(),
                                obs={'condition': ["ctrl"] * len(control)})
        adata = ad.concat([adata_true, adata_ctrl]) # 合并真实结果和控制组结果
        if gene_names is not None:
            adata.var.index = gene_names
            adata_pred.var.index = gene_names
        sc.tl.rank_genes_groups(adata, groupby='condition', method="wilcoxon")  # 使用 Wilcoxon 方法进行基因差异表达分析
        diff_genes = adata.uns["rank_genes_groups"]["names"]['stim']    # 获取差异表达的基因
        diff_genes_idx = [np.where(np.array(gene_names) == x)[0].item() for x in diff_genes]
        adata = ad.concat([adata, adata_pred])  # 合并所有的 AnnData 对象
        adata.obs_names_make_unique()
        scores = reg_mean_plot(
            adata,
            condition_key='condition',
            axis_keys={"x": "pred", "y": 'stim', "x1": "ctrl"},
            gene_list=diff_genes[:10] if gene_names is not None else None,
            top_100_genes=diff_genes[:100],
            labels={"x": "predicted", "y": "ground truth", "x1": "ctrl"},
            path_to_save=path_to_save,
            title='DSB',
            show=False,
            legend=False,
        )
    # 计算各组的均值
    true_mean = true.mean(0)
    pred_mean = pred.mean(0)
    control_mean = control.mean(0)
    true_delta_mean = true_mean - control_mean
    pred_delta_mean = pred_mean - control_mean

    scores.update({
        # MAE
        # 'mae': (pred_mean - true_mean).abs().mean().item(),
        # 'mae_top_100': (pred_mean[diff_genes_idx[:100]] - true_mean[diff_genes_idx[:100]]).abs().mean().item(),
        # 'mae_delta': (pred_delta_mean - true_delta_mean).abs().mean().item(),
        # MSE
        'mse': F.mse_loss(pred_mean, true_mean).item(),
        'mse_top_100': F.mse_loss(pred_mean[diff_genes_idx[:100]], true_mean[diff_genes_idx[:100]]).item(),
        'mse_delta': F.mse_loss(pred_delta_mean, true_delta_mean).item(),
        # RMSE
        # 'rmse': np.sqrt(F.mse_loss(pred_mean, true_mean).item()),
        # 'rmse_top_100': np.sqrt(F.mse_loss(pred_mean[diff_genes_idx[:100]],
        #                                    true_mean[diff_genes_idx[:100]]).item()),
        # 'rmse_delta': np.sqrt(F.mse_loss(pred_delta_mean, true_delta_mean).item()),
        # Correlation
        'corr': PearsonCorr1d(pred_mean, true_mean).item(),
        'corr_top_100': PearsonCorr1d(pred_mean[diff_genes_idx[:100]],
                                      true_mean[diff_genes_idx[:100]]).item(),
        'corr_delta': PearsonCorr1d(pred_delta_mean, true_delta_mean).item(),
        # # Cosine similarity
        # 'cos': F.cosine_similarity(pred_mean.unsqueeze(0), true_mean.unsqueeze(0))[0].item(),
        # 'cos_top_100': F.cosine_similarity(pred_mean[diff_genes_idx[:100]].unsqueeze(0),
        #                                    true_mean[diff_genes_idx[:100]].unsqueeze(0))[0].item(),
        # 'cos_delta': F.cosine_similarity(pred_delta_mean.unsqueeze(0),
        #                                  true_delta_mean.unsqueeze(0))[0].item(),
    })

    if de_gene_idx is not None:
        for num_de in (20, 100):
            if num_de > len(de_gene_idx):
                warnings.warn(
                    f"Skipping {num_de} DE gene num eval since max num DE available is {len(de_gene_idx)}",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            if num_de > true.shape[1]:
                warnings.warn(
                    f"Skipping {num_de} DE gene num eval since max num genes available is {true.shape[1]}",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            idx = de_gene_idx[:num_de]
            scores.update(de_eval(pred_mean[idx], true_mean[idx], control_mean[idx], f"de{num_de}"))

    if ndde20_idx is not None:
        scores.update(de_eval(pred_mean[ndde20_idx], true_mean[ndde20_idx], control_mean[ndde20_idx], "ndde20"))

    return scores




def de_eval(true, pred, ctrl, name):
    true_delta = true - ctrl
    pred_delta = pred - ctrl
    return {
        # MAE
        # f'mae_{name}': (pred - true).abs().mean().item(),
        # f'mae_delta_{name}': (pred_delta - true_delta).abs().mean().item(),
        # MSE
        f'mse_{name}': F.mse_loss(pred, true).item(),
        f'mse_delta_{name}': F.mse_loss(pred_delta, true_delta).item(),
        # RMSE
        # f'rmse_{name}': np.sqrt(F.mse_loss(pred, true).item()),
        # f'rmse_delta_{name}': np.sqrt(F.mse_loss(pred_delta, true_delta).item()),
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
    sns.set()
    sns.set(color_codes=True)
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
        ax = sns.regplot(x=axis_keys["x"], y=axis_keys["y"], data=df, scatter_kws={'rasterized': True})
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
                texts.append(pyplot.text(x_bar, y_bar, i, fontsize=11, color="black"))
                pyplot.plot(x_bar, y_bar, 'o', color="red", markersize=5)
                # if "y1" in axis_keys.keys():
                # y1_bar = y1[j]
                # pyplot.text(x_bar, y1_bar, i, fontsize=11, color="black")
        if gene_list is not None:
            adjust_text(texts, x=x, y=y, arrowprops=dict(arrowstyle="->", color='grey', lw=0.5),
                        force_points=(0.0, 0.0))
        if legend:
            pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if title is None:
            pyplot.title("", fontsize=fontsize)
        else:
            pyplot.title(title, fontsize=fontsize)

        ax.text(x_coeff, y_coeff, r'$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= ' +
                f"{r_value ** 2:.4f}", fontsize=kwargs.get("textsize", fontsize))
        if diff_genes is not None:
            ax.text(x_coeff, y_coeff + 0.6, r'$\mathrm{R^2_{\mathrm{\mathsf{top\ 100\ DEGs}}}}$= ' +
                    f"{r_value_diff ** 2:.4f}", fontsize=kwargs.get("textsize", fontsize))
        if path_to_save is not None:
            pyplot.savefig(f"{path_to_save}", bbox_inches='tight', dpi=100)
        if show:
            pyplot.show()
        pyplot.close()

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

    # 创建一个新字典，用于存储更新后的键
    updated_scores = {}
    for key, value in scores.items():
        key_tuple = tuple(key.tolist())
        for seen_item in seen:
            if key_tuple == seen_item[0]:
                updated_scores[seen_item[1]] = value
                break

    # 获取所有指标的名称
    metric_list = ["mse","mse_de20","corr","corr_delta","corr_de20"]
    # eval_values = list(updated_scores.values())
    # for key in eval_values[0].keys():
    #     metric_list.append(key)

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

    # for name, pert_list in subgroup.items():
    #     for pert in pert_list:
    #         for metric in metric_list:
    #             subgroup_analysis[name][metric].append(updated_scores[pert][metric])

    for name, reuslt in subgroup_analysis.items():
        for metric in reuslt.keys():
            subgroup_analysis[name][metric] = np.mean(subgroup_analysis[name][metric])

    return subgroup_analysis

def _get_gene_indices(gene_names, target_genes):
    """使用生成器减少内存占用"""
    for gene in target_genes:
        yield np.where(gene_names == gene)[0][0]


def calculate_mmd(perturbed_data, unperturbed_data, sigma=None):
    """
    计算扰动前后基因表达量的最大平均差异（MMD）

    参数：
        perturbed_data (np.ndarray): 扰动后的基因表达量数据，形状为(n_samples_perturbed, n_features)
        unperturbed_data (np.ndarray): 扰动前的基因表达量数据，形状为(n_samples_unperturbed, n_features)
        sigma (float, optional): 高斯核的带宽参数，若未指定则使用中位数启发式

    返回：
        float: MMD值
    """
    X = perturbed_data
    Y = unperturbed_data

    if sigma is None:
        # 合并数据计算中位数带宽
        all_data = np.vstack([X, Y])
        # 计算所有样本对的平方欧氏距离
        distances = pairwise_distances(all_data, metric='euclidean', squared=True)
        # 取非零距离的中位数（排除对角线零值）
        non_zero_distances = distances[np.triu_indices_from(distances, k=1)]
        median_distance = np.median(non_zero_distances) if len(non_zero_distances) > 0 else 0.0
        sigma = np.sqrt(median_distance) if median_distance > 0 else 1.0  # 防止零距离情况

    # 处理可能的除零错误
    m = X.shape[0]
    n = Y.shape[0]

    # 计算核矩阵项
    def gaussian_kernel(x, y, sigma):
        x_norm = np.sum(x ** 2, axis=1)
        y_norm = np.sum(y ** 2, axis=1)
        K = np.exp(-(x_norm[:, None] + y_norm[None, :] - 2 * np.dot(x, y.T)) / (2 * sigma ** 2))
        return K

    # 计算各项的核矩阵和
    K_XX = gaussian_kernel(X, X, sigma)
    K_YY = gaussian_kernel(Y, Y, sigma)
    K_XY = gaussian_kernel(X, Y, sigma)

    # 计算各项期望估计
    term1 = (K_XX.sum() - m) / (m * (m - 1)) if m > 1 else 0.0
    term2 = (K_YY.sum() - n) / (n * (n - 1)) if n > 1 else 0.0
    term3 = 2 * K_XY.sum() / (m * n) if (m > 0 and n > 0) else 0.0

    mmd_squared = term1 + term2 - term3
    mmd = np.sqrt(max(mmd_squared, 0))  # 确保数值稳定性

    return mmd

def pearson_corr(x, y):
    # GPU加速的皮尔逊相关系数计算
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)))

def mmd_gpu(x, y, sigma=1.0):
    # GPU加速的MMD计算
    xx = torch.exp(-torch.cdist(x, x)**2 / (2*sigma**2)).mean()
    xy = torch.exp(-torch.cdist(x, y)**2 / (2*sigma**2)).mean()
    yy = torch.exp(-torch.cdist(y, y)**2 / (2*sigma**2)).mean()
    return xx - 2*xy + yy


def de_mse(pred, true, ctrl, de_gene_idx_dict):
    pred_mean = pred.mean(0)
    true_mean = true.mean(0)
    ctrl_mean = ctrl.mean(0)

    true_delta = true_mean - ctrl_mean
    pred_delta = pred_mean - ctrl_mean

    idx = list(de_gene_idx_dict.values())[0][:20]
    pred_20 = pred_mean[idx]
    true_20 = true_mean[idx]

    return {
        'mse_de20': F.mse_loss(pred_20, true_20).item(),
        f'corr_delta': PearsonCorr1d(pred_delta, true_delta).item(),
    }

# 示例测试
if __name__ == "__main__":
    # 随机生成预测值和目标值
    batch_size = 32
    pred = torch.rand(batch_size, 7000)  # 模拟预测值
    target = torch.rand(batch_size, 7000)  # 模拟真实值
    mmd=calculate_mmd(pred.numpy(),target.numpy())
    print(f"MMD Score: {mmd}")
    mmd1 = calculate_mmd(target.numpy(),pred.numpy())
    print(f"MMD Score: {mmd1}")
    # 计算 R²
    # r2 = evaluate_r_squared(pred, target)
    # print(f"R² Score: {r2:.4f}")
