import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import re

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


# 注册SRModel
@MODEL_REGISTRY.register()
# SRModel继承于BaseModel，BaseModel中提供了model共用的一些函数
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    # 初始化SRModel类，比如定义网络和load weight
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network 定义网络结构，根据配置文件，自动实例化相应的网络结构类
        self.net_g = build_network(opt['network_g']) # 根据参数，实例化网络结构
        self.net_g = self.model_to_device(self.net_g) # 将网络放到GPU上
        self.print_network(self.net_g) # 打印网络

        # load pretrained models 加载预训练模型
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        # 初始化训练相关的设置
        if self.is_train:
            self.init_training_settings()

    # 初始化与训练相关的，比如loss，设置optimizer和schedulers
    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses 根据配置文件yml中的loss的类型和参数，实例化loss
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('fft_opt'):
            self.cri_fft = build_loss(train_opt['fft_opt']).to(self.device)
        else:
            self.cri_fft = None

        if train_opt.get('wave_opt'):
            self.cri_wave = build_loss(train_opt['wave_opt']).to(self.device)
        else:
            self.cri_wave = None

        if self.cri_pix is None and self.cri_perceptual is None and self.cri_freq is None and self.cri_wave is None:
            raise ValueError('Pixel, perceptual and frequency losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    # 具体设置optimizer，可以根据实际需求，对params设置多组不同的optimizer
    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    # 提供数据，是与dataloader（dataset）的接口
    # 把训练数据送入模型，这里是从dataloader中取出数据，用于训练或测试。
    #       在SRModel中，每次取用一个batch的LR和GT图像。
    # 其他模型对batch作不同操作时，经常会改写这个函数。比如只读取GT、读取额外label
    # 对读取的数据添加 degradation 的操作，都通过修改feed__data来实现
    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    # 优化参数，即一个完整train的step，
    # 包括forward、loss计算、backward、参数优化等
    def optimize_parameters(self, current_iter):
        # 使optimizer中的梯度归零
        self.optimizer_g.zero_grad()
        # 前向传播
        self.output = self.net_g(self.lq)

        l_total = 0
        # 使用有序字典，可以在log显示的时候，保持我们添加先后顺序
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # frequency loss
        if self.cri_fft:
            l_fft = self.cri_fft(self.output, self.gt)
            l_total += l_fft
            loss_dict['l_freq'] = l_fft

        # wavelet-based frequency loss
        if self.cri_wave:
            l_wave = self.cri_wave(self.output, self.gt)
            l_total += l_wave
            loss_dict['l_wave'] = l_wave

        # perceptual loss
        if self.cri_perceptual:
            # 感知损失 和 风格损失
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        # 反向传播
        l_total.backward()
        # 更新参数
        self.optimizer_g.step()

        # 为了loss的显示
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    # 测试流程
    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)

    # 多卡验证流程
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    # 单卡验证流程
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                # 将多个衡量标准放入dict中 {psnr: 0, ssim: 0}
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')
            
        mean_psnr = {}
        count_psnr = {}

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            
            class_name = ''.join(re.findall(r'[A-Za-z]', img_name))
            
            # 喂测试数据
            self.feed_data(val_data)
            # 测试
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    # 根据配置文件yml中的metrics的配置，调用相应的函数
                    temp = calculate_metric(metric_data, opt_)
                    self.metric_results[name] += temp
                    
                    if "psnr" in name:
                        if class_name not in mean_psnr:
                            mean_psnr[class_name] = 0
                            count_psnr[class_name] = 0
                        mean_psnr[class_name] += temp
                        count_psnr[class_name] += 1
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            # 记录每个数据集中不同分类的平均PSNR值
            # for key in mean_psnr.keys():
            #     mean_psnr[key] /= count_psnr[key]
            #     print(f"{key} : {mean_psnr[key]:.4f} : {count_psnr[key]}")
                
            for metric in self.metric_results.keys():
                # 计算均值，计算 1 epoch 的均值
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    # 控制打印validation的结果
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    # 得到网络输出的结果，这个函数会在validation中用到，实际可以简化掉
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()

        # out_dict['lq'] /= 255.
        # out_dict['result'] /= 255.
        # out_dict['gt'] /= 255.

        return out_dict

    # 保存网络（.pth文件）和训练状态（.state文件）
    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
