import numpy as np
import torch
import argparse
from torch_geometric.data import Data
from itertools import combinations
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
import json
import yaml


def create_cell_graph_for_prediction(X, pert_idx, pert_gene):
    """
    Create a perturbation specific cell graph for inference

    Args:
        X (np.array): gene expression matrix
        pert_idx (list): list of perturbation indices
        pert_gene (list): list of perturbations

    """

    if pert_idx is None:
        pert_idx = [-1]

    return {
        'data':torch.Tensor(X).T,
        'prior':torch.Tensor(X).T,
        'batch':torch.tensor(0),
        'cell_type':torch.tensor(0),
        'pert':torch.tensor(pert_idx)
    }


def create_cell_graph_dataset_for_prediction(pert_gene, ctrl_adata, gene_names,
                                             device, num_samples = 300):
    """
    Create a perturbation specific cell graph dataset for inference

    Args:
        pert_gene (list): list of perturbations
        ctrl_adata (anndata): control anndata
        gene_names (list): list of gene names
        device (torch.device): device to use
        num_samples (int): number of samples to use for inference (default: 300)

    """

    # Get the indices (and signs) of applied perturbation
    pert_idx = [np.where(p == np.array(gene_names))[0][0] for p in pert_gene]

    Xs = ctrl_adata[np.random.randint(0, len(ctrl_adata), num_samples), :].X.toarray()
    outputs={
        'data':[],
        'prior':[],
        'batch':[],
        'cell_type': [],
        'pert': []
    }
    # Create cell graphs
    for X in Xs:
        output=create_cell_graph_for_prediction(X,pert_idx,pert_gene)
        for key in output.keys():
            outputs[key].append(output[key])
    for key in outputs.keys():
        if isinstance(outputs[key][0], torch.Tensor):
            outputs[key] = torch.stack(outputs[key], dim=0)
        else:
            outputs[key] = torch.tensor(outputs[key])
    cond={
        'batch':outputs['batch'],
        'cell_type':outputs['cell_type'],
        'pert':outputs['pert']
    }
    outputs['cond']=cond
    outputs.pop('batch')
    outputs.pop('cell_type')
    outputs.pop('pert')
    return outputs

def compute_pcc_traj(p_array):
    n_genes = p_array.shape[2]
    pcc_traj = { (i, j): [] for i, j in combinations(range(n_genes), 2) }

    for t in range(p_array.shape[0]):
        x = p_array[t]  # shape: (samples, genes)
        corr = np.corrcoef(x, rowvar=False)  # shape: (genes, genes)
        for i, j in pcc_traj.keys():
            pcc_traj[(i, j)].append(corr[i, j])
    return pcc_traj

def select_top_gene_pairs(pcc_traj, top_k=10):
    scores = []
    for pair, traj in pcc_traj.items():
        delta_max = max(traj) - min(traj)
        delta_start_end = abs(traj[-1] - traj[0])
        score = (delta_max + delta_start_end) / 2
        scores.append((pair, score))
    scores.sort(key=lambda x: -x[1])
    return [pair for pair, _ in scores[:top_k]]

def plot_pcc_trajectories(
    pcc_traj,
    top_pairs,
    gene_names,
    pert,
    real_pcc_matrix=None,
    cosine_results=None,
    ax=None
):

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    for i, j in top_pairs:
        label = f"{gene_names[i]}&{gene_names[j]}"
        traj = pcc_traj[(i, j)]
        line, = ax.plot(traj, label=label)
        color = line.get_color()

        if real_pcc_matrix is not None:
            real_pcc = real_pcc_matrix[i, j]
            ax.scatter(50, real_pcc, color=color, edgecolor='black', zorder=5)

    ax.set_xlabel("Inference timestep",fontsize=14)
    ax.set_ylabel("Pearson correlation coefficient (PCC)",fontsize=14)
    ax.set_title(f"{pert}",fontsize=16)

    legend=ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=True,
        framealpha=1,
        facecolor='white',
        edgecolor='black',
        prop={'size': 9}
    )


    return legend


def cosine_similarity_top_pairs(pcc_traj, real_pcc_matrix, top_pairs):
    pred_vec = np.array([pcc_traj[(i, j)][-1] for (i, j) in top_pairs])
    true_vec = np.array([real_pcc_matrix[i, j] for (i, j) in top_pairs])
    return 1 - cosine(pred_vec, true_vec)  # 越接近1越好

def compute_cosine_similarities(pcc_traj, real_pcc_matrix, de_idx, top_pairs):
    all_pairs = list(combinations(range(len(de_idx)), 2))
    cos_top = cosine_similarity_top_pairs(pcc_traj, real_pcc_matrix, top_pairs)
    cos_all = cosine_similarity_top_pairs(pcc_traj, real_pcc_matrix, all_pairs)

    return {
        "cosine_top_pairs": cos_top,
        "cosine_all_pairs": cos_all
    }

def compute_topk_overlap_trajectory(p_array, real_pcc, top_k=20):
    T = p_array.shape[0]
    overlaps = []

    for t in range(T):
        x=p_array[t]  # (N, len(DEGs))
        pred_pcc = np.corrcoef(x, rowvar=False)

        # Top-K overlap
        iu = np.triu_indices_from(pred_pcc, k=1)
        pred_flat = np.abs(pred_pcc[iu])
        truth_flat = np.abs(real_pcc[iu])

        top_pred = np.argsort(pred_flat)[-top_k:]
        top_truth = np.argsort(truth_flat)[-top_k:]
        overlap = len(set(top_pred) & set(top_truth)) / top_k
        overlaps.append(overlap)

    return overlaps

def plot_cosine_textbox(fig, ax, legend, cosine_results, top_pairs):

    renderer = fig.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer)
    inv = fig.transFigure.inverted()
    legend_box_fig_coords = inv.transform(legend_box)

    legend_left = legend_box_fig_coords[0, 0]
    xaxis_fig_y = inv.transform(ax.transAxes.transform((0, 0)))[1]

    fig.text(
        legend_left,
        xaxis_fig_y,
        f"Cosine Similarity:\n"
        f"Top-{len(top_pairs)} pairs: {cosine_results['cosine_top_pairs']:.3f}\n"
        f"All pairs: {cosine_results['cosine_all_pairs']:.3f}",
        fontsize=12,
        verticalalignment='bottom',
        horizontalalignment='left',
        bbox=dict(boxstyle="round", facecolor='white', alpha=0.8)
    )


def plot_pcc_with_overlap_trajectory(
    pcc_traj,
    top_pairs,
    gene_names,
    pert,
    real_pcc_matrix,
    cosine_results,
    overlap_traj,
    save_path="combined_pcc_overlap.png"
):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={'height_ratios': [3, 1]})

    legend=plot_pcc_trajectories(
        pcc_traj=pcc_traj,
        top_pairs=top_pairs,
        gene_names=gene_names,
        pert=pert,
        real_pcc_matrix=real_pcc_matrix,
        cosine_results=cosine_results,
        ax=ax1
    )

    ax2.plot(overlap_traj, marker='o')
    ax2.set_xlabel("Inference timestep",fontsize=14)
    ax2.set_ylabel("Top-K edge overlap",fontsize=14)
    # ax2.set_title("Structural recovery over time",fontsize=16)
    ax2.grid(True)

    plt.tight_layout()
    ax1.text(-0.1, 1.05, "a", transform=ax1.transAxes,
             fontsize=14, fontweight='bold', va='top', ha='left')

    ax2.text(-0.1, 1.20, "b", transform=ax2.transAxes,
             fontsize=14, fontweight='bold', va='top', ha='left')

    fig.canvas.draw()

    plot_cosine_textbox(fig, ax1, legend, cosine_results, top_pairs)

    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

def init_experiment(exp_dir, args):
    """
    Initialize experiment folder, save args/config, and optionally init wandb.

    Args:
        exp_dir (str): Experiment directory path
        args (argparse.Namespace): Argument object
        use_wandb (bool): Whether to initialize wandb
    """
    os.makedirs(exp_dir, exist_ok=True)
    txt_log = os.path.join(exp_dir, "experiment_info.txt")

    with open(txt_log, "w", encoding="utf-8") as f:
        # Basic info
        f.write(f"Experiment: {args.exp_name}\n")
        f.write(f"Checkpoint: {getattr(args, 'ckpt', 'None')}\n")

        # Save args
        f.write("==== ARGS ====\n")
        f.write(json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n")

        # Save config if provided
        f.write("==== CONFIG ====\n")
        if getattr(args, "config", None):
            try:
                with open(args.config, "r", encoding="utf-8") as cf:
                    try:
                        cfg_obj = yaml.safe_load(cf)
                        f.write(json.dumps(cfg_obj, indent=2, ensure_ascii=False) + "\n")
                    except Exception:
                        cf.seek(0)
                        f.write(cf.read() + "\n")
            except Exception as e:
                f.write(f"[config] read failed: {e}\n")
        else:
            f.write("None\n")

    print(f"[Logger] Experiment info saved to: {txt_log}")
