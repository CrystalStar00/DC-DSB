import torch
from functools import partial
from util.model.network import ResMLP, UNet1D
from omegaconf import OmegaConf
from util.model.transformer import TransformerNetwork,instantiate_module

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_size(x, coef):
    if isinstance(coef, (int, float)):
        return coef
    elif isinstance(coef, dict):
        for k, v in coef.items():
            if isinstance(v, torch.Tensor):
                while len(v.shape) < len(x.shape):
                    v = v.unsqueeze(-1)
                coef[k] = v
    elif isinstance(coef, torch.Tensor):
        while len(coef.shape) < len(x.shape):
            coef = coef.unsqueeze(-1)
    return coef


class BaseModel(torch.nn.Module):
    def __init__(self, data,args, device, noiser, rank):
        super().__init__()
        # self.data = data
        self.args = args
        self.device = device
        self.noiser = noiser
        self.rank = rank


        if self.args.network == 'mlp':
            self.network = ResMLP(dim_in=2, dim_out=2, dim_hidden=128, num_layers=5, n_cond=self.noiser.training_timesteps)
        else:
            if 'gene' in self.args.network:
                if "transformer" in self.args.network:
                    config = OmegaConf.load(self.args.config)
                    config.model['pretrained_gene_list'] = data.pretrained_gene_list
                    if config.model.cond_emb_type == 'embedding':
                        config.model['cond_num_dict'] = data.cond_num_dict
                        config.model['post_cond_num_dict']=(data.post_cond_num_dict)
                    if (
                            hasattr(data, 'G_go') and
                            hasattr(data, 'G_go_weight') and
                            hasattr(data, 'num_perts')
                    ):
                        config.model['num_perts'] = data.num_perts
                        config.model['gears_flag'] = True
                    extra_params = {}
                    self.network = instantiate_module(TransformerNetwork, config.model, extra_params)
                else:
                    self.network = UNet1D(
                        input_dim=5045,
                        hidden_dim=128,
                        time_dim=self.noiser.training_timesteps,
                    )

        self.network.to(self.device)
        self.network = torch.nn.parallel.DistributedDataParallel(self.network, device_ids=[self.device], output_device=self.device) if (hasattr(self.args, 'gpus') and self.args.gpus > 1) else self.network

    def target(self, x_0, x_1, x_t, t):
        raise NotImplementedError

    def forward(self,x_1, x_t, t,cond=None,aug_graph=None,):
        t = self.noiser.timestep_map[t]
        if isinstance(self.network, TransformerNetwork):
            x = self.network(x_orig=x_1,x=x_t, timesteps=t,conditions=cond,aug_graph=aug_graph)
        else:
            x = self.network(x_t, t)
        x.to(self.device)
        return x

    def predict_boundary(self, x_0, x_t, t):
        raise NotImplementedError
    
    def predict_next(self, x_0, x_t, t):
        x_0, x_1 = self.predict_boundary(x_0, x_t, t)
        x_t = self.noiser(x_0, x_1, t + 1)
        return x_t

    def inference(self, x_0, return_all=False):
        self.eval()
        x_t_all = [x_0.clone()]
        with torch.no_grad():
            x_t = x_0
            ones = torch.ones(size=(x_t.shape[0],), dtype=torch.int64, device=x_t.device)
            for t in range(self.noiser.num_timesteps):
                with torch.autocast(device_type="cuda", enabled=self.args.use_amp):
                    x_t = self.predict_next(x_0, x_t, ones * t)
                x_t = x_t.float()
                if return_all:
                    x_t_all.append(x_t.clone())
        if return_all:
            return x_t, torch.stack(x_t_all, dim=0)
        else:
            return x_t


class Diffusion(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prediction_type = self.args.method.split(':')[1] if ':' in self.args.method else 'x1'

    def target(self, x_0, x_1, x_t, t):
        if self.prediction_type == 'x1':
            return x_1
        elif self.prediction_type == 'x0':
            return x_0
        elif self.prediction_type == 'v':
            coef = check_size(x_t, self.noiser.coefficient(t))
            v = coef['coef1'] * x_0 - coef['coef0'] * x_1
            return v

    def predict_boundary(self, x_0, x_t, t):
        coef_t = check_size(x_t, self.noiser.coefficient(t))
        if self.prediction_type == 'x1':
            x_1 = self.forward(x_t, t)
            x_0 = (x_t - coef_t['coef1'] * x_1) / coef_t['coef0']
        elif self.prediction_type == 'x0':
            x_0 = self.forward(x_t, t)
            x_1 = (x_t - coef_t['coef0'] * x_0) / coef_t['coef1']
        elif self.prediction_type == 'v':
            v = self.forward(x_t, t)
            x_0 = coef_t['coef0'] * x_t + coef_t['coef1'] * v
            x_1 = coef_t['coef1'] * x_t - coef_t['coef0'] * v
        return x_0, x_1


class Flow(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def target(self, x_0, x_1, x_t, t):
        return x_1 - x_0
    
    def predict_next(self, x_0, x_t, t):
        coef_t_1 = check_size(x_t, self.noiser.coefficient(t))['coef1']
        coef_t_plus_one_1 = check_size(x_t, self.noiser.coefficient(t + 1))['coef1']
        v_pred = self.forward(x_t, t)
        x_t = x_t + (coef_t_plus_one_1 - coef_t_1) * v_pred
        return x_t


class DSB(BaseModel):
    def __init__(self, *args, direction=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.direction = direction
        self.num_timesteps = self.noiser.training_timesteps

        self.noiser.prepare_gamma_dsb()
        self.gammas = self.noiser.gammas
        self.is_diffusing=False

    def forward(self, x_1, x_t, t, cond=None, aug_graph=None):
        t = self.noiser.timestep_map[t]

        if isinstance(self.network, TransformerNetwork):
            if self.direction=="b":
                x = self.network(x_orig=x_1, x=x_t, timesteps=t, conditions=cond,aug_graph=aug_graph)
            else:
                x = self.network(x_orig=x_1, x=x_t, timesteps=t,conditions=None,aug_graph=None)


        else:
            x = self.network(x_t, t)

        return x


    def get_coef_ts(self, x, t, delta=1):
        coef_t = check_size(x, self.noiser.coefficient(t))
        coef_t_other = check_size(x, self.noiser.coefficient(t + delta))
        return coef_t, coef_t_other

    def _forward(self,x_orig, x, t,cond=None,aug_graph=None):
        use_cond=self.is_diffusing
        # mu_cond=0

        if isinstance(self.network, TransformerNetwork):
            if self.direction=="b":
                x_other = self.network(
                    x_orig=x_orig,
                    x=x,
                    timesteps=t,
                    conditions=cond,
                    aug_graph=aug_graph
                )
            else:
                x_other = self.network(
                    x_orig=x_orig,
                    x=x,
                    timesteps=t,
                )
        else:
            x_other = self.forward(x, t)
        if self.args.reparam == 'flow':
            v_pred = x_other
            if self.direction == 'f':
                coef_t, coef_t_next = self.get_coef_ts(x, t, 1)
                x = x + (coef_t_next['coef1'] - coef_t['coef1']) * v_pred
            elif self.direction == 'b':
                coef_t, coef_t_next = self.get_coef_ts(x, self.num_timesteps - t, -1)
                x = x + (coef_t_next['coef0'] - coef_t['coef0']) * v_pred
        elif self.args.reparam == 'term':
            if self.direction == 'f':
                coef_t, coef_t_next = self.get_coef_ts(x, t, 1)
                x_1 = x_other
                x_0 = (x - coef_t['coef1'] * x_1) / coef_t['coef0']
            elif self.direction == 'b':
                coef_t, coef_t_next = self.get_coef_ts(x, self.num_timesteps - t, -1)
                x_0 = x_other
                x_1 = (x - coef_t['coef0'] * x_0) / coef_t['coef1']
            x = coef_t_next['coef0'] * x_0 + coef_t_next['coef1'] * x_1
        else:
            x = x_other
        return x

    def inference(self,x_orig, x,cond=None, aug_graph=None, sample=False):
        ones = torch.ones(size=(x.shape[0],), dtype=torch.int64, device=self.device)
        x_cache, gt_cache, t_cache = [], [], []
        x_raw = x
        with torch.no_grad():
            for t in range(self.num_timesteps):
                tt = ones * t
                x_old = x
                with torch.autocast(device_type="cuda", enabled=self.args.use_amp):
                    t_old = self._forward(x_orig,x, tt,cond=cond,aug_graph=aug_graph)
                t_old = t_old.float()
                if sample and t == self.num_timesteps - 1:
                    x = t_old
                else:
                    x = t_old + torch.sqrt(2 * self.gammas[t]) * torch.randn_like(x)
                x_cache.append(x)
                if self.args.simplify:
                    if self.args.reparam == 'flow':
                        gt_cache.append((x_raw - x) / (t + 1) * self.num_timesteps)
                    elif self.args.reparam == 'term':
                        gt_cache.append(x_raw)
                    else:
                        gt_cache.append(x_old)
                else:
                    t_new = self._forward(x, tt)
                    gt_cache.append(x + t_old - t_new)
                t_cache.append(self.num_timesteps - 1 - tt)
        x_cache = torch.stack([x_raw] + x_cache, dim=0).cpu() if sample else torch.cat(x_cache, dim=0).cpu()
        gt_cache = torch.cat(gt_cache, dim=0).cpu()
        t_cache = torch.cat(t_cache, dim=0).cpu()
        return x_cache, gt_cache, t_cache

    def inference_dsb(self,x_orig, x,t,cond=None, aug_graph=None, sample=False):
        ones = torch.ones(size=(x.shape[0],), dtype=torch.int64, device=self.device)
        x_cache, gt_cache, t_cache = [], [], []
        x_raw = x_orig
        with torch.no_grad():

            tt = ones * t
            x_old = x
            with torch.autocast(device_type="cuda", enabled=self.args.use_amp):
                t_old = self._forward(x_orig,x, tt,cond=cond,aug_graph=aug_graph)
            t_old = t_old.float()
            if sample and t == self.num_timesteps - 1:
                x = t_old
            else:
                x = t_old + torch.sqrt(2 * self.gammas[t]) * torch.randn_like(x)
            # x_cache.append(x)
            if self.args.simplify:
                if self.args.reparam == 'flow':
                    gt=(x_raw - x) / (t + 1) * self.num_timesteps
                elif self.args.reparam == 'term':
                    gt=x_raw
                else:
                    gt=x_old
            else:
                t_new = self._forward(x, tt)
                gt_cache.append(x + t_old - t_new)

        return x, gt


def create_model(data,name, *args, **kwargs):
    name = str(name).lower()
    if 'diffusion' in name:
        model = Diffusion
    elif 'flow' in name:
        model = Flow
    elif 'dsb' in name:
        model = DSB
    else:
        raise NotImplementedError
    model = model(data,*args, **kwargs)
    return model
