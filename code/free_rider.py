from system import Participant
import numpy as np
import torch
import random


class FreeRider(Participant):
    def __init__(self, id, init_rep, tra_round_num, device,
                 est_round_num, 
                 atk_c_param,
                 atk_noise_dim,
                 atk_est_noise_std,
                 ):
        # 初始化父类
        super().__init__(id, "free_rider", init_rep, tra_round_num, device)
        # 初始化搭便车策略参数
        self.est_round_num = est_round_num
        self.atk_c_param = atk_c_param                        
        self.atk_noise_dim = atk_noise_dim
        self.atk_est_noise_std = atk_est_noise_std
        self.est_lambda_bar = None
        self.est_cos_beta = None 
        self.global_norm_diff_history = [] 
        self.atk_phase = "parameter_estimation"
        self.last_gt = None 
        self.sec_last_gt = None
    

    # 更新估计历史
    def _update_estimation_history(self, gt_flat_history):
        if gt_flat_history: 
            if len(gt_flat_history) > 0:
                 gt_norm = torch.linalg.norm(gt_flat_history[-1]).item() # 计算最新的 g_t 的范数
                 self.global_norm_diff_history.append(gt_norm)


    # 估计 lambda_bar
    def _estimate_lambda_bar(self, current_round_num):
        history = self.global_norm_diff_history
        if current_round_num > self.est_round_num and len(history) >= 2:
            t_eff = len(history) -1 
            ratio = history[0] / history[-1]
            lambda_val = np.log(ratio**(1.0/t_eff))
            self.est_lambda_bar = lambda_val
        else:
            self.est_lambda_bar = None 


    # 计算预期的 cos(beta)
    def _estimate_cos_beta(self, current_round_num):
        if self.est_lambda_bar is None:
            print("Warning: est_lambda_bar is None, cannot calculate expected cos(beta).")
        else:
            exp_term = np.exp(2 * self.est_lambda_bar * current_round_num)
            cos_beta = (self.atk_c_param**2) / (self.atk_c_param**2 + exp_term)
            self.est_cos_beta = max(0, min(1, cos_beta))


    # 生成伪造的更新
    def gen_fabric_update(self, current_round_num, gt_flat_history, current_global_model_state, num_total_clients):
        # 更新全局梯度的norm历史
        self._update_estimation_history(gt_flat_history)
        # 阶段判断
        if current_round_num > self.est_round_num:
            self.atk_phase = "advanced_attack"
            self._estimate_lambda_bar(current_round_num)
            self._estimate_cos_beta(current_round_num)
        else: 
            self.atk_phase = "parameter_estimation"


        # 更新 g_t 和 g_{t-1}
        if len(gt_flat_history) >= 1: self.last_gt = gt_flat_history[-1].to(self.device)
        if len(gt_flat_history) >= 2: self.sec_last_gt = gt_flat_history[-2].to(self.device)
        
        # 初始化伪造更新
        fabric_update_flat = None
        flat_global_params = self._flatten_params(current_global_model_state) # 展平全局模型参数

        # 生成梯度逻辑
        if self.atk_phase == "parameter_estimation":
            # 如果处于参数估计阶段，使用随机噪声生成伪造更新
            noise = torch.randn_like(flat_global_params, device=self.device) * self.atk_est_noise_std
            fabric_update_flat = noise
            self.current_update = self._unflatten_params(fabric_update_flat, current_global_model_state)
        else: 
            # 如果处于攻击阶段，使用高级攻击逻辑生成伪造更新
            norm_g_current = torch.linalg.norm(self.last_gt)
            norm_g_previous = torch.linalg.norm(self.sec_last_gt)
            scaled_delta_factor = norm_g_current / norm_g_previous # 计算缩放因子
            U_f_flat = scaled_delta_factor * self.last_gt # 生成缩放后的基础更新
            norm_U_f_flat = torch.linalg.norm(U_f_flat) # 计算基础更新的范数
            expected_cos_beta = self.est_cos_beta # 提取预期的 cos(beta)
            n = num_total_clients # 计算参与者数量
            # 计算添加噪声的幅度 phi
            if n > 1:
                denominator_sqrt = n + (n**2 - n) * expected_cos_beta
                phi_magnitude_factor_sqrt_term = np.sqrt((n**2 / denominator_sqrt) - 1)
                phi_magnitude = max(0, phi_magnitude_factor_sqrt_term * norm_U_f_flat)
            else:
                print("Warning: n <= 1, cannot calculate phi_magnitude.")
                phi_magnitude = 0
            # 计算添加的噪声
            noise_vector_phi = torch.randn_like(U_f_flat, device=self.device) # 生成随机噪声，服从标准正态分布
            noise_vector_phi = noise_vector_phi / (torch.linalg.norm(noise_vector_phi)) * phi_magnitude # 归一化噪声向量并缩放
            num_dims_to_add_noise = int(flat_global_params.numel() * self.atk_noise_dim) # 计算添加噪声的维度
            # 对于添加噪声的维度进行处理，剩余的维度保持不变
            if 0 < num_dims_to_add_noise < flat_global_params.numel():
                indices_to_add_noise = torch.randperm(flat_global_params.numel(), device=self.device)[:num_dims_to_add_noise]
                U_f_flat_perturbed = U_f_flat.clone()
                U_f_flat_perturbed[indices_to_add_noise] += noise_vector_phi[indices_to_add_noise]
                fabric_update_flat = U_f_flat_perturbed
            else: fabric_update_flat = U_f_flat + noise_vector_phi
            self.current_update = self._unflatten_params(fabric_update_flat, current_global_model_state)
            return self.current_update
    

    # 投标
    def submit_bid(self, highest_effect, lowest_effect, lowest_honest_promise, avg_honest_promise):
        # 蛰伏阶段
        if self.atk_phase == "parameter_estimation":
            self.bid['promise'] = lowest_honest_promise * random.uniform(0.8, 1.0) 
            self.bid['reward'] = (self.bid['promise'] / lowest_effect) * random.uniform(1.0, 1.2)
        # 攻击阶段    
        else: 
            self.bid['promise'] = avg_honest_promise
            self.bid['reward'] = (self.bid['promise'] / highest_effect) * random.uniform(0.8, 1.0)
        
        self.bid['promise'] = max(0, self.bid['promise'])
        self.bid['reward'] = max(0.01, self.bid['reward'])
        return self.bid