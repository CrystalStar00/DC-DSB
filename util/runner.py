import torch, os, wandb
from tqdm import tqdm
import pandas as pd
from util.dataset.base import create_data
from util.noiser.noiser import create_noiser
from util.model.model import create_model
from util.logger import Logger
from util.visualize import InferenceResultVisualizer
from util.evaluate import *
from util.utils import *
import time
import matplotlib.pyplot as plt


class Runner():
    def __init__(self, args):
        self.args = args
        self.rank = self.args.local_rank
        self.aug_graph=None

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        if 'gene' in self.args.prior:
            self.best_param = float('inf')
        else:
            self.best_param = 0
        self.best_epoch = -1
        self.epochs_no_improve = 0
        self.patience = 5

        self.saved_pred={}

        self.dsb = 'dsb' in self.args.method

        self.device = torch.device(f'cuda')
        self.exp_dir = os.path.join("./exp", self.args.exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        self.txt_log = os.path.join(self.exp_dir, "train_log.txt")
        init_experiment(self.exp_dir, self.args)
        if self.args.wandb:
            os.environ['WANDB_DIR'] =os.path.join("./exp/",self.args.exp_name)
            exp_dir = os.path.join("./exp/",self.args.exp_name)
            os.makedirs(exp_dir, exist_ok=True)
            wandb.init(
                project=self.args.project,
                config=self.args,
                name=self.args.exp_name,
                dir=exp_dir,
                mode="offline")
            self.wandb = wandb

        self.time_log = {
            "train_times": [],
            "eval_times": [],
            "total_time" : 0.0,
            "log_file": os.path.join("./exp/",self.args.exp_name, "time_log.txt")
        }
        self._train_start_time = time.time()

        if 'gene' in self.args.prior:
            self.eval_save_path = "./vaild_eval"
        else:
            self.eval_save_path = f"./cell_eval"
        if not os.path.exists(self.eval_save_path):
            os.makedirs(self.eval_save_path)

        base_steps_per_epoch = 2 ** 5
        self.prior_set, self.prior_sampler, self.prior_loader = create_data(
            self.args.prior, self.args.gpus, batch_size=self.args.batch_size,data=self.args.data)
        if 'gene' in self.args.prior:
            pretrained_gene_list=self.prior_set.gene_names
        else:
            pretrained_gene_list=self.prior_set.adata.var.index.to_list()
        self.prior_set.pretrained_gene_list=pretrained_gene_list

        self.prior_iterator = iter(self.prior_loader)
        self.adata = self.prior_loader.dataset.pert_data.adata
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']

        self.criterion = torch.nn.MSELoss(reduction='none')
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.args.use_amp,
            init_scale=2.0 ** 10,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000
        )


        self.val_prior_set = None
        if self.args.val_prior is not None:
            self.val_prior_set, _, self.val_prior_loader = create_data(
                self.args.val_prior, self.args.gpus, batch_size=self.args.batch_size,data=self.args.data)
            pretrained_gene_list = self.val_prior_set.adata.var.index.to_list()
            self.val_prior_set.pretrained_gene_list = pretrained_gene_list
            self.val_prior_iterator = iter(self.val_prior_loader)

        assert self.args.training_timesteps == self.args.inference_timesteps

        self.noiser = create_noiser(self.args.noiser, args, self.device)
        self.cache_size = self.cnt = base_steps_per_epoch * self.args.batch_size * 4
        self.forward_model = create_model(self.prior_set,self.args.method, self.args, self.device, self.noiser, rank=self.rank, direction='f')
        self.backward_model = create_model(self.prior_set, self.args.method, self.args, self.device, self.noiser, rank=self.rank, direction='b')

        if self.rank == 0:
            print(f'Forward Network #Params: {sum([p.numel() for p in self.forward_model.parameters()])}')
            print(f'Backward Network #Params: {sum([p.numel() for p in self.backward_model.parameters()])}')

        self.forward_model = self.forward_model.to(self.device)
        self.backward_model = self.backward_model.to(self.device)

        def _build_optimizer(model, lr, weight_decay):
            decay, no_decay = [], []
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                name = n.lower()
                if any(k in name for k in ["bias", "norm", "layernorm", "ln", "emb", "embedding"]):
                    no_decay.append(p)
                else:
                    decay.append(p)
            param_groups = [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ]
            return torch.optim.AdamW(
                param_groups, lr=lr, betas=(0.9, 0.95), eps=1e-6
            )

        self.forward_optimizer = _build_optimizer(self.forward_model, self.args.lr, self.args.weight_decay)
        self.backward_optimizer = _build_optimizer(self.backward_model, self.args.lr, self.args.weight_decay)

        self.model = {'forward': self.forward_model, 'backward': self.backward_model}
        self.optimizer = {'forward': self.forward_optimizer, 'backward': self.backward_optimizer}
        self.direction = 'backward'

        self.load_ckpt()

        self.save_path = os.path.join('exp', self.args.exp_name)
        if self.args.global_rank == 0:
            self.evaluators ={
                'Inference': InferenceResultVisualizer(self.args, self.device, save_path=self.save_path),
            }
            self.logger = Logger(os.path.join(self.save_path, 'log'), self.noiser.training_timesteps)
        if not hasattr(self, 'evaluation_records'):
            self.evaluation_records = pd.DataFrame()

    def start_timer(self):
        self._train_start = time.time()

    def end_train_timer(self, epoch):
        train_time = time.time() - self._train_start
        self.time_log["train_times"].append(train_time)
        print(f"[Timing] Epoch {epoch} - Train: {train_time:.2f}s")

    def end_eval_timer(self, epoch):
        eval_time = time.time() - self._eval_start
        self.time_log["eval_times"].append(eval_time)
        print(f"[Timing] Epoch {epoch} - Eval: {eval_time:.2f}s")

    def save_time_log(self):
        train_times = self.time_log["train_times"]
        eval_times = self.time_log["eval_times"]
        log_path = self.time_log["log_file"]

        avg_train = sum(train_times) / len(train_times) if train_times else 0
        avg_eval = sum(eval_times) / len(eval_times) if eval_times else 0

        with open(log_path, "w") as f:
            f.write("=== Training Time Log ===\n")
            for i, t in enumerate(train_times):
                f.write(f"Epoch {i}: Train Time = {t:.2f}s\n")
            f.write(f"\nAverage Train Time: {avg_train:.2f}s\n\n")

            f.write("=== Evaluation Time Log ===\n")
            for i, t in enumerate(eval_times):
                f.write(f"Epoch {i}: Eval Time = {t:.2f}s\n")
            f.write(f"\nAverage Eval Time: {avg_eval:.2f}s\n")

        print(f"[Timing] Time log saved to: {log_path}")

    def _next_batch(self):
        try:
            x_raw= next(self.prior_iterator)
        except Exception as e:
            self.prior_iterator = iter(self.prior_loader)
            x_raw = next(self.prior_iterator)
        outdict={}
        # if 'pert' in self.args.prior:
        conditions = x_raw['cond']
        for key in conditions.keys():
            conditions[key] = conditions[key].to(self.device)
        outdict['conditions'] = conditions

        if 'gene' in self.args.prior:
            if self.aug_graph is None:
                self.aug_graph = {
                    'G_go': self.prior_loader.dataset.G_go.to(self.device),
                    'G_go_weight': self.prior_loader.dataset.G_go_weight.to(self.device),
                }

        x_0, x_1 = x_raw['prior'], x_raw['data']
        outdict['x_0'] = x_0.to(self.device)
        outdict['x_1'] = x_1.to(self.device)

        return outdict

    def next_batch(self, epoch,i, dsb=False):
        if dsb:
            if i % self.args.repeat_per_epoch == 0:
                self.x_cache, self.gt_cache, self.t_cache = [], [], []
                self.x_0_cache, self.x_1_cache=[],[]
                self.cond_cache={}
                outdict = self._next_batch()
                x_0=outdict['x_0']
                x_1=outdict['x_1']
                conditions=outdict['conditions']
                with torch.no_grad():
                    if self.direction == 'backward' and epoch == 0:
                        _x_cache, _gt_cache, _t_cache = self.noiser.trajectory_dsb(x_0, x_1)
                    else:
                        model = self.model['backward' if self.direction == 'forward' else 'forward']
                        model.eval()
                        x_target = x_0 if self.direction == 'forward' else x_1
                        _x_cache, _gt_cache, _t_cache = model.inference(x_target, x_target,conditions,self.aug_graph)
                self.x_0_cache.append(x_0)
                self.x_1_cache.append(x_1)
                self.cond_cache=conditions
                current_batch_size=_x_cache.shape[0]//self.noiser.num_timesteps
                self.x_cache=[[] for _ in range(current_batch_size)]
                self.gt_cache=[[] for _ in range(current_batch_size)]
                for i in range(_x_cache.shape[0]):
                    group_index = i % current_batch_size
                    self.x_cache[group_index].append(_x_cache[i])
                    self.gt_cache[group_index].append(_gt_cache[i])

            current_batch_size=len(self.x_cache)
            t = torch.randint(0, self.noiser.num_timesteps, (current_batch_size,))
            x = [self.x_cache[i][t[i]] for i in range(current_batch_size)]
            gt = [self.gt_cache[i][t[i]] for i in range(current_batch_size)]

            x=torch.stack(x)
            gt=torch.stack(gt)
            x,gt,t=x.to(self.device),gt.to(self.device),t.to(self.device)
            x_1 = torch.stack(self.x_1_cache).squeeze(0).to(self.device)
            x_0 = torch.stack(self.x_0_cache).squeeze(0).to(self.device)
            cond = self.cond_cache

            batchout={'x_t':x,'gt':gt,'t':t,'x_0':x_0,'x_1':x_1,'cond':cond}
            return batchout


    def train(self):
        self.train_forward_losses = []
        self.train_backward_losses = []

        self.val_mse_de20 = []
        steps_per_epoch = int(len(self.prior_loader) * self.args.repeat_per_epoch)
        self.cache_size = min(self.cache_size, steps_per_epoch * self.args.batch_size)

        for epoch in range(self.args.epochs):
            self.start_timer()
            self.current_epoch = epoch
            if epoch < self.args.skip_epochs:
                print(f'Skipping ep{epoch} and evaluate.')
                self.evaluate(epoch, 0, last=True)
                continue
            self.noiser.train()
            if self.dsb:
                self.cnt = self.cache_size
                self.direction = 'backward' if epoch % 2 == 0 else 'forward'
                model, optimizer = self.model[self.direction], self.optimizer[self.direction]
            else:
                model, optimizer = self.model, self.optimizer
            model.train()
            if self.rank == 0:
                pbar = tqdm(total=steps_per_epoch)
            ema_loss, ema_loss_w = None, lambda x: min(0.99, x / 10)
            if self.prior_sampler is not None:
                self.prior_sampler.set_epoch(epoch)

            for i in range(steps_per_epoch):
                if self.dsb:
                    batchout = self.next_batch(epoch,i, dsb=True)
                    x_t=batchout['x_t']
                    t =batchout['t']
                    gt=batchout['gt']
                    x_0=batchout['x_0']
                    x_1=batchout['x_1']
                    cond=batchout['cond']
                else:
                    batchout = self.next_batch(epoch,i)
                    x_0=batchout['x_0']
                    x_1=batchout['x_1']
                    t=batchout['t']
                    cond=batchout['cond']
                    x_t = self.noiser(x_0, x_1, t)
                    gt = model.target(x_0, x_1, x_t, t)


                optimizer.zero_grad()
                with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=self.args.use_amp):

                    if self.direction=="forward":
                        pred = model(x_1, x_t, t, cond,self.aug_graph)
                    else:
                        pred=model(x_0,x_t,t,cond,self.aug_graph)
                    raw_loss = self.criterion(pred, gt).mean(dim=-1)
                    loss = raw_loss.mean()

                self.scaler.scale(loss).backward()

                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                grads_ok = True
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        grads_ok = False
                        break

                if torch.isfinite(loss) and grads_ok:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:

                    optimizer.zero_grad(set_to_none=True)
                    if self.rank == 0:
                        print("Warning: skipped step due to NaN/Inf in loss or grads.")

                ema_loss = loss.item() if ema_loss is None else (ema_loss * ema_loss_w(i) + loss.item() * (1 - ema_loss_w(i)))


                if self.rank == 0 and i == steps_per_epoch - 1:
                    desc = f'epoch {epoch}, direction {self.direction}, iteration {i}, loss {ema_loss:.14f}' if self.dsb else f'epoch {epoch}, iteration {i}, loss {ema_loss:.14f}'
                    pbar.set_description(desc, refresh=False)
                    pbar.update(i + 1 - pbar.n)
            self.end_train_timer(epoch)

            if self.direction == "forward":
                self.train_forward_losses.append(ema_loss)
            else:
                self.train_backward_losses.append(ema_loss)

            if self.direction =="backward":
                self._eval_start = time.time()
                scores=self.evaluate(epoch, steps_per_epoch, last=False)
                self.end_eval_timer(epoch)
                if 'gene' in self.args.prior:
                    self.val_mse_de20.append(scores["mse_de20"])
                    self.logger.log(f"\nEpoch {epoch} Evaluation - mse_de20: {scores['mse_de20']:.6f}")
                else:
                    self.logger.log(f"\nEpoch {epoch} Evaluation - mse_de20: {scores['R^2']:.6f}")

                for metric_name, metric_value in scores.items():
                    self.logger.log_scalar(f'Final_{metric_name}', metric_value, epoch * steps_per_epoch + steps_per_epoch)
                    self.logger.log(f"Final_eval - epoch {epoch}, {metric_name}: {metric_value}")
                    self.wandb.log({f"Val_{metric_name}": metric_value})

                epoch_record = pd.DataFrame([scores], index=[epoch])
                self.evaluation_records = pd.concat([self.evaluation_records, epoch_record])

                csv_filename = os.path.join(self.eval_save_path,f"{self.args.exp_name}.xlsx")
                sheet_name = self.args.exp_name
                file_exists = os.path.exists(csv_filename)

                if file_exists:
                    with pd.ExcelWriter(csv_filename, mode="a", if_sheet_exists="replace") as writer:
                        self.evaluation_records.to_excel(writer, sheet_name=sheet_name)
                else:
                    with pd.ExcelWriter(csv_filename, mode="w") as writer:
                        self.evaluation_records.to_excel(writer, sheet_name=sheet_name)

                param='mse_de20' if 'gene' in self.args.prior else 'R^2'
                current_param=scores[param]
                if 'gene' in self.args.prior:
                    print(f"Epoch {epoch}, mse_de20: {current_param:.6f}")
                    if current_param < self.best_param:
                        self.best_param = current_param
                        self.best_epoch = epoch
                        self.epochs_no_improve = 0

                        if self.rank == 0:
                            self.save_ckpt(epoch, iters=steps_per_epoch, best=True)
                    else:
                        self.epochs_no_improve += 1
                else:
                    print(f"Epoch {epoch}, R^2: {current_param:.6f}")
                    if current_param > self.best_param:
                        self.best_param = current_param
                        self.best_epoch = epoch
                        self.epochs_no_improve = 0

                        if self.rank == 0:
                            self.save_ckpt(epoch, iters=steps_per_epoch, best=True)
                    else:
                        self.epochs_no_improve += 1


                if self.epochs_no_improve>=self.patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    self.logger.log(f"Early stopping triggered at epoch {epoch}")
                    break

            if self.rank == 0:
                pbar.clear()
                pbar.close()

        self.time_log["total_time"] = time.time() - self._train_start_time
        self.save_time_log()

        # plt.figure()
        # plt.plot(self.train_forward_losses, label="Forward Loss")
        # plt.plot(self.train_backward_losses, label="Backward Loss")
        # plt.xlabel("Epoch")
        # plt.ylabel("Train Loss")
        # plt.title("Train Loss per Direction")
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig(f"./exp/{self.args.exp_name}/train_loss_direction.png")
        # plt.close()
        #
        # # 绘制 val mse_de20
        # plt.figure()
        # plt.plot(self.val_mse_de20, label="Val mse_de20", color='orange')
        # plt.xlabel("Epoch")
        # plt.ylabel("mse_de20")
        # plt.title(f"Validation mse_de20 - {self.args.exp_name}")
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig(f"exp/{self.args.exp_name}_val_mse_de20.png")
        # plt.close()


    def val_step(self,epoch):

        with torch.no_grad():
            for batch in tqdm(self.val_prior_loader,desc=f"Epoch {epoch}, validating",leave=False):
                x_data = batch["data"].to(self.device)
                x_prior = batch["prior"].to(self.device)
                cond = batch["cond"]
                for key in cond.keys():
                    cond[key] = cond[key].to(self.device)


                if 'gene' in self.args.prior:
                    if self.aug_graph is None:
                        self.aug_graph = {
                            'G_go': self.prior_loader.dataset.G_go.to(self.device),
                            'G_go_weight': self.prior_loader.dataset.G_go_weight.to(self.device),
                        }
                else:
                    self.aug_graph = None

                if self.dsb:
                    qs=self.backward_model.inference(x_prior,x_prior,cond=cond,aug_graph=self.aug_graph, sample=True)[0]


                out={
                    "x_prior": x_prior,
                    "x_data":x_data,
                    'true_conds':cond['pert'],
                    "qs":qs[-1],
                }
                self.vaild_step_outputs.append(out)


    def evaluate(self, epoch, iters, last=False):
        with torch.no_grad():

            if self.dsb:
                self.forward_model.eval()
                self.backward_model.eval()
            else:
                self.model.eval()

            if self.args.global_rank == 0:
                if last:
                    self.save_ckpt(epoch, iters)

                cond=None
                self.vaild_step_outputs=[]
                if 'pert' in self.args.prior:
                    with torch.no_grad():
                        self.val_step(epoch)
                outputs=self.vaild_step_outputs
                x_data=torch.cat([outdict['x_data'].cpu() for outdict in outputs])
                pred=torch.cat([outdict['qs'].cpu() for outdict in outputs])
                x_prior=torch.cat([outdict['x_prior'].cpu() for outdict in outputs])
                gene_names = self.val_prior_set.gene_names

                if 'gene' in self.args.prior:
                    extras = self.val_prior_set.extras
                    true_conds = torch.cat([outdict['true_conds'].cpu() for outdict in outputs])
                    de_gene_idx_dict = None if extras is None else extras.get("rank_genes_groups_cov_all_idx_dict")
                    ndde20_idx_dict = None if extras is None else extras.get("top_non_dropout_de_20")

                else:
                    de_gene_idx_dict = None
                    ndde20_idx_dict = None
                    true_conds = None

                if true_conds is not None:
                    if self.args.data in ['replogle_rpe1_essential','replogle_k562_essential_test']:
                        cond_batch_size=8
                    else:
                        cond_batch_size=len(true_conds.unique(dim=0))
                    _,scores = perturbation_eval(x_data,pred,x_prior,gene_names=gene_names,de_gene_idx_dict=de_gene_idx_dict,
                                             true_conds=true_conds,ndde20_idx_dict=ndde20_idx_dict,path_to_save=f'./exp/{self.args.data}_perturb.png',batch_size=cond_batch_size)
                else:
                    scores = perturbation_eval(x_data,pred,x_prior,gene_names=gene_names,de_gene_idx_dict=de_gene_idx_dict,
                                             true_conds=true_conds,ndde20_idx_dict=ndde20_idx_dict,path_to_save=f'./exp/{self.args.data}_perturb.png')


                return scores

    def save_ckpt(self, epoch, iters,best=False):

        os.makedirs(os.path.join(self.save_path, 'ckpt'), exist_ok=True)
        if self.dsb:
            ckpt = {
                'forward_model': self.forward_model.state_dict(),
                'backward_model': self.backward_model.state_dict(),
                'forward_optimizer': self.forward_optimizer.state_dict(),
                'backward_optimizer': self.backward_optimizer.state_dict(),
            }
        else:
            ckpt = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
        if best:
            torch.save(ckpt,os.path.join(self.save_path, 'ckpt', 'best.pth'))

    def load_ckpt(self):
        def match_ckpt(ckpt):
            _ckpt = {}
            for k, v in ckpt.items():
                if self.args.gpus > 1 and 'module.' not in k:
                    k = k.replace('network.', 'network.module.')
                elif self.args.gpus == 1 and 'module.' in k:
                    k = k.replace('network.module.', 'network.')
                _ckpt[k] = v
            return _ckpt
        if self.args.ckpt is not None:
            ckpt = torch.load(self.args.ckpt, map_location='cpu')
            if self.dsb:
                self.forward_model.load_state_dict(match_ckpt(ckpt['forward_model']), strict=False)
                self.backward_model.load_state_dict(match_ckpt(ckpt['backward_model']), strict=False)
                if "forward_optimizer" in ckpt:
                    self.forward_optimizer.load_state_dict(ckpt['forward_optimizer'])
                    self.backward_optimizer.load_state_dict(ckpt['backward_optimizer'])
            else:
                self.model.load_state_dict(match_ckpt(ckpt['model']), strict=False)
                if "optimizer" in ckpt:
                    self.optimizer.load_state_dict(ckpt['optimizer'])

    def predict(self, pert_list):
        """
        Predict the transcriptome given a list of genes/gene combinations being
        perturbed

        Parameters
        ----------
        pert_list: list
            list of genes/gene combiantions to be perturbed

        Returns
        -------
        results_pred: dict
            dictionary of predicted transcriptome
        results_logvar: dict
            dictionary of uncertainty score

        """
        ## given a list of single/combo genes, return the transcriptome
        ## if uncertainty mode is on, also return uncertainty score.

        self.adata=self.prior_loader.dataset.pert_data.adata
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
        for pert in pert_list:
            for i in pert:
                if i not in self.prior_set.pert_list:
                    raise ValueError(i + " is not in the perturbation graph. "
                                         "Please select from GEARS.pert_list!")


        results_pred = {}

        from torch_geometric.data import DataLoader
        for pert in pert_list:
            try:
                # If prediction is already saved, then skip inference
                results_pred['_'.join(pert)] = self.saved_pred['_'.join(pert)]
            except:
                pass

            cg = create_cell_graph_dataset_for_prediction(pert, self.ctrl_adata,
                                                          self.prior_set.pert_list, self.device)
            # loader = DataLoader(cg, 300, shuffle=False)
            batch = cg

            x_data = batch["data"].to(self.device)
            x_prior = batch["prior"].to(self.device)
            cond = batch["cond"]
            for key in cond.keys():
                cond[key] = cond[key].to(self.device)

            if 'gene' in self.args.prior:
                if self.aug_graph is None:
                    self.aug_graph = {}
                    self.aug_graph['G_go']=self.prior_loader.dataset.G_go.to(self.device)
                    self.aug_graph['G_go_weight'] = self.prior_loader.dataset.G_go_weight.to(self.device)

            aug_graph=self.aug_graph

            with torch.no_grad():
                if self.direction=='backward':
                    p = self.backward_model.inference(x_prior, x_prior, cond=cond,aug_graph=aug_graph, sample=True)[0][-1]
                else:
                    p = self.forward_model.inference(x_data, x_data, cond=cond, aug_graph=aug_graph, sample=True)[0][-1]

            results_pred['_'.join(pert)] = np.mean(p.detach().cpu().numpy(), axis=0)

        self.saved_pred.update(results_pred)


        return results_pred

    def plot_perturbation(self, query, save_file=None):
        """
        Plot the perturbation graph

        Parameters
        ----------
        query: str
            condition to be queried
        save_file: str
            path to save the plot

        Returns
        -------
        None

        """

        import seaborn as sns
        import matplotlib.pyplot as plt
        self.node_map = self.prior_loader.dataset.pert_data.node_map

        sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.5)

        adata = self.prior_loader.dataset.pert_data.adata
        gene2idx = self.node_map
        cond2name = dict(adata.obs[['condition', 'condition_name']].values)
        gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))

        de_idx = [gene2idx[gene_raw2id[i]] for i in
                  adata.uns['top_non_dropout_de_20'][cond2name[query]]]
        genes = [gene_raw2id[i] for i in
                 adata.uns['top_non_dropout_de_20'][cond2name[query]]]
        truth = adata[adata.obs.condition == query].X.toarray()[:, de_idx]

        query_ = [q for q in query.split('+') if q != 'ctrl']
        pred = self.predict([query_])['_'.join(query_)][de_idx]
        ctrl_means = adata[adata.obs['condition'] == 'ctrl'].to_df().mean()[
            de_idx].values

        pred = pred - ctrl_means
        truth = truth - ctrl_means

        return pred,truth,genes

    def plot_trajectory(self, query,top_k=10, save_file=None):
        """
        Plot the perturbation graph

        Parameters
        ----------
        query: str
            condition to be queried
        save_file: str
            path to save the plot

        Returns
        -------
        None

        """

        import seaborn as sns
        import matplotlib.pyplot as plt
        self.node_map = self.prior_loader.dataset.pert_data.node_map

        sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.5)

        adata = self.prior_loader.dataset.pert_data.adata
        gene2idx = self.node_map
        cond2name = dict(adata.obs[['condition', 'condition_name']].values)
        gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))

        de_idx = [gene2idx[gene_raw2id[i]] for i in
                  adata.uns['top_non_dropout_de_20'][cond2name[query]]]
        genes = [gene_raw2id[i] for i in
                 adata.uns['top_non_dropout_de_20'][cond2name[query]]]
        truth = adata[adata.obs.condition == query].X.toarray()

        batch=self.prior_loader.dataset.get_batch_by_perturbation(query)
        query_ = [q for q in query.split('+') if q != 'ctrl']
        # pred = self.predict([query_])['_'.join(query_)][de_idx]
        pert_list=[query_]

        for pert in pert_list:
            for i in pert:
                if i not in self.prior_set.pert_list:
                    raise ValueError(i + " is not in the perturbation graph. "
                                         "Please select from GEARS.pert_list!")


        x_data = batch["data"].to(self.device)
        x_prior = batch["prior"].to(self.device)
        cond = batch["cond"]
        for key in cond.keys():
            cond[key] = cond[key].to(self.device)

        if 'gene' in self.args.prior:
            if self.aug_graph is None:
                self.aug_graph = {}
                self.aug_graph['G_go'] = self.prior_loader.dataset.G_go.to(self.device)
                self.aug_graph['G_go_weight'] = self.prior_loader.dataset.G_go_weight.to(self.device)

        aug_graph = self.aug_graph

        with torch.no_grad():
            if self.direction == 'backward':
                p = self.backward_model.inference(x_prior, x_prior, cond=cond, aug_graph=aug_graph, sample=True)[0]
            else:
                p = self.forward_model.inference(x_data, x_data, cond=cond, aug_graph=aug_graph, sample=True)[0]

        delta_p = p[1:] - x_prior.cpu()[None, :, :]

        sub_p = delta_p[:, :, de_idx].cpu().numpy()

        pcc_traj = compute_pcc_traj(sub_p)
        real_pcc = np.corrcoef((x_data - x_prior).cpu()[:, de_idx].numpy(), rowvar=False)
        top_pairs = select_top_gene_pairs(pcc_traj, top_k=top_k)

        cosine_results = compute_cosine_similarities(pcc_traj, real_pcc, de_idx, top_pairs)

        overlap_traj = compute_topk_overlap_trajectory(sub_p, real_pcc, top_k=top_k)

        plot_pcc_with_overlap_trajectory(
            pcc_traj=pcc_traj,
            top_pairs=top_pairs,
            gene_names=genes,
            pert=query,
            real_pcc_matrix=real_pcc,
            cosine_results=cosine_results,
            overlap_traj=overlap_traj,
            save_path=save_file
        )