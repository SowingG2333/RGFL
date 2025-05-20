from system import get_mnist_data, Global_Model, Requester
from honest_client import HonestClient
from free_rider import FreeRider
import torch
import random
import numpy as np
import copy
from torch.utils.data import DataLoader
import torch.nn.functional as F
import json
import os

class Simulation:
    def __init__(self, params_X):
        self.params_X = params_X
        self.num_total_participants = params_X.get("N", 10) # 参与者总数
        self.num_free_riders = params_X.get("N_f", 3) # 搭便车者数量
        self.num_honest_clients = self.num_total_participants - self.num_free_riders # 诚实客户端数量
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        self.max_rounds = params_X.get("T_max", 20) # 最大模拟轮数
        self.target_accuracy_threshold = params_X.get("target_accuracy_threshold", None) # 目标准确率阈值
        self.initial_M_t = params_X.get("initial_M_t", self.num_total_participants // 2) # 初始每轮选择的参与者数量
        self.iid_data_distribution = params_X.get("iid_data", True) # 数据是否独立同分布
        self.non_iid_alpha = params_X.get("non_iid_alpha", 0.3) # Non-IID 数据分布的参数

        self.alpha_reward = params_X.get("alpha_reward") # 奖励计算参数 alpha
        self.beta_penalty_base = params_X.get("beta_penalty_base") # 惩罚计算参数 beta
        self.omega_m_update = params_X.get("omega_m_update") # M_t 更新参数 omega
        self.q_rounds_rep_change = params_X.get("q_rounds_rep_change") # 声誉更新周期 (对应 Participant.tra_round_num)
        self.initial_reputation = params_X.get("initial_reputation") # 初始声誉值
        self.reputation_threshold = params_X.get("reputation_threshold") # 声誉阈值
        self.min_performance_constraint = params_X.get("min_performance_constraint") # 最低性能约束 (帕累托优化用)
        
        self.num_honest_rounds_for_fr_estimation = params_X.get("num_honest_rounds_for_fr_estimation", 2) 
        self.adv_attack_c_param_fr = params_X.get("adv_attack_c_param_fr")
        self.adv_attack_noise_dim_fraction_fr = params_X.get("adv_attack_noise_dim_fraction_fr")
        self.adv_attack_scaled_delta_noise_std_fr = params_X.get("adv_attack_scaled_delta_noise_std", 0.0001) 
        
        self.adaptive_bid_adjustment_intensity_gamma_honest = params_X.get("adaptive_bid_adjustment_intensity_gamma_honest")
        self.adaptive_bid_max_adjustment_delta_honest = params_X.get("adaptive_bid_max_adjustment_delta_honest")
        self.min_commitment_scaling_factor_honest = params_X.get("min_commitment_scaling_factor_honest")

        self.bid_gamma_honest = params_X.get("bid_gamma_honest", 0.8) 
        self.local_epochs_honest = params_X.get("local_epochs_honest", 2)
        self.lr_honest = params_X.get("lr_honest", 0.005)
        self.batch_size_honest = params_X.get("batch_size_honest", 32)

        self.participants = []
        self.requester = None
        self.current_round = 0
        self.M_t = self.initial_M_t
        self.total_rewards_paid = 0.0
        self.rewards_paid_to_honest_clients = 0.0 
        self.final_global_model_performance = 0.0
        self.termination_round = self.max_rounds 
        
        self.client_l2_norm_history = {}
        self.client_cosine_similarity_history = {}
        self.client_reputation_history = {} # main.py 级别的声誉历史记录，用于绘图或整体分析
        self.client_types = {}
        self.all_fr_eliminated_logged = False

        # 新增：用于存储模拟过程中的统计数据
        self.simulation_stats = {
            "round_number": [],  # 轮次编号
            "model_accuracy": [],  # 每轮模型训练acc (全局模型准确率)
            "active_honest_clients": [],  # 每轮活跃正常客户端数目
            "active_free_riders": [],  # 每轮活跃搭便车者数目
            "M_t_value": [],  # 每轮的M_t值 (当轮选择参与者时使用的M_t)
            "cumulative_total_incentive_cost": [],  # 每轮结束时的累计总激励开销
            "cumulative_real_incentive_cost": [],  # 每轮结束时的累计真实激励开销 (支付给诚实客户端)
            "cumulative_tir_history": []  # 每轮结束时累计的真实激励率
        }


    # 初始化环境，包括全局模型、数据加载器和参与者
    def initialize_environment(self):
        print("初始化环境中...")
        initial_global_model = Global_Model()
        effective_num_honest_clients = max(0, self.num_honest_clients)

        client_datasets, test_dataset_global = get_mnist_data(
            effective_num_honest_clients, self.iid_data_distribution, self.non_iid_alpha
        )
        if effective_num_honest_clients > 0 and (not client_datasets or effective_num_honest_clients > len(client_datasets)):
            print(f"警告: 请求 {effective_num_honest_clients} 个诚实客户端数据，实际分配 {len(client_datasets)} 个。")
            if not client_datasets and effective_num_honest_clients > 0:
                 raise ValueError(f"诚实客户端数据分配失败: 请求 {effective_num_honest_clients}, 得到 {len(client_datasets)}")

        pin_memory_flag = self.device != torch.device("cpu")
        num_workers_val = os.cpu_count()
        test_loader = DataLoader(test_dataset_global, batch_size=512, shuffle=False, 
                                 pin_memory=pin_memory_flag, num_workers=num_workers_val)
        
        self.requester = Requester(initial_global_model, test_loader, self.device,
                             alpha_reward=self.alpha_reward, 
                             beta_penalty_base=self.beta_penalty_base)
        self.participants = []
        
        temp_participants = []
        for i in range(effective_num_honest_clients):
            if i >= len(client_datasets) or not client_datasets[i] or len(client_datasets[i]) == 0:
                print(f"警告：诚实客户端 {i} 数据不足或无效，跳过。")
                continue
            temp_participants.append(HonestClient(
                id=f"h_{i}", init_rep=self.initial_reputation, tra_round_num=self.q_rounds_rep_change,
                device=self.device, client_dataset=client_datasets[i], train_size=0.8,
                init_commit_scaling_factor=self.bid_gamma_honest,
                batch_size=self.batch_size_honest, local_epochs=self.local_epochs_honest, lr=random.uniform(0.0001, 0.001),
                adapt_bid_adj_intensity=self.adaptive_bid_adjustment_intensity_gamma_honest,
                adapt_bid_max_delta=self.adaptive_bid_max_adjustment_delta_honest,
                min_commit_scaling_factor=self.min_commitment_scaling_factor_honest
            ))
        for i in range(self.num_free_riders):
            temp_participants.append(FreeRider(
                id=f"f_{i}", init_rep=self.initial_reputation, tra_round_num=self.q_rounds_rep_change,
                device=self.device,
                est_round_num=self.num_honest_rounds_for_fr_estimation,
                atk_c_param=self.adv_attack_c_param_fr,
                atk_noise_dim=self.adv_attack_noise_dim_fraction_fr,
                atk_est_noise_std=self.adv_attack_scaled_delta_noise_std_fr,
            ))

        if not temp_participants and (effective_num_honest_clients > 0 or self.num_free_riders > 0) :
            raise ValueError("没有参与者被初始化，尽管请求了参与者。检查数据分配和客户端初始化。")
        
        random.shuffle(temp_participants)
        self.participants = temp_participants

        for p in self.participants:
            self.client_types[p.id] = p.type
            self.client_l2_norm_history[p.id] = []
            self.client_cosine_similarity_history[p.id] = []
            self.client_reputation_history[p.id] = [] # 初始化为空列表，用于后续存储每轮声誉

        self.current_round = 0
        self.M_t = min(self.initial_M_t, len(self.participants)) if self.participants else 0
        self.total_rewards_paid = 0.0
        self.rewards_paid_to_honest_clients = 0.0
        self.all_fr_eliminated_logged = False 

        if self.requester: # 确保请求者已初始化
            self.final_global_model_performance, _ = self.requester.evaluate_global_model()
            if self.requester.global_model: # 确保全局模型存在
                 self.requester.previous_global_model_state_flat = self.requester._flatten_params(self.requester.global_model.state_dict())
            print(f"初始全局模型性能: {self.final_global_model_performance:.4f}")
            
            # 新增：记录初始状态 (第0轮) 的统计数据
            self.simulation_stats["round_number"].append(0)
            self.simulation_stats["model_accuracy"].append(self.final_global_model_performance)
            initial_active_honest = sum(1 for p in self.participants if self.client_types.get(p.id) == "honest_client" and p.reputation >= self.reputation_threshold)
            initial_active_free_riders = sum(1 for p in self.participants if self.client_types.get(p.id) == "free_rider" and p.reputation >= self.reputation_threshold)
            self.simulation_stats["active_honest_clients"].append(initial_active_honest)
            self.simulation_stats["active_free_riders"].append(initial_active_free_riders)
            self.simulation_stats["M_t_value"].append(self.M_t) # 记录初始 M_t
            self.simulation_stats["cumulative_total_incentive_cost"].append(self.total_rewards_paid) # 初始为0
            self.simulation_stats["cumulative_real_incentive_cost"].append(self.rewards_paid_to_honest_clients) # 初始为0
            self.simulation_stats["cumulative_tir_history"].append(0.0) # 初始TIR为0

            # 修改点：记录所有客户端在第0轮的初始声誉
            for p_item in self.participants:
                if p_item.id in self.client_reputation_history:
                    self.client_reputation_history[p_item.id].append(p_item.reputation) # 此处 p_item.reputation 即为其初始声誉
                else: # 理论上不应发生，因为上面循环中已为所有 participant 初始化了 key
                    self.client_reputation_history[p_item.id] = [p_item.reputation]

        else:
            self.final_global_model_performance = 0.0
            print("警告：服务器未正确初始化，无法评估初始模型或记录初始统计。")
            self.simulation_stats["round_number"].append(0)
            self.simulation_stats["model_accuracy"].append(0.0)
            self.simulation_stats["active_honest_clients"].append(0)
            self.simulation_stats["active_free_riders"].append(0)
            mt_initial_val = self.initial_M_t
            if self.participants:
                mt_initial_val = min(self.initial_M_t, len(self.participants))
            elif self.num_total_participants is not None:
                mt_initial_val = min(self.initial_M_t, self.num_total_participants)
            self.simulation_stats["M_t_value"].append(mt_initial_val)
            self.simulation_stats["cumulative_total_incentive_cost"].append(0.0)
            self.simulation_stats["cumulative_real_incentive_cost"].append(0.0)
            self.simulation_stats["cumulative_tir_history"].append(0.0)
            # 修改点：即使初始化不完全，如果 participants 列表已形成，也尝试记录初始声誉
            if self.participants:
                for p_item in self.participants:
                    if p_item.id in self.client_reputation_history:
                        self.client_reputation_history[p_item.id].append(p_item.reputation)
                    else:
                        self.client_reputation_history[p_item.id] = [p_item.reputation]


    # 更新 M_t 值，根据当前参与者的声誉和更新情况
    def update_M_t(self):
        if self.current_round == 0 or not self.participants:
            return

        active_participants = [p for p in self.participants if p.reputation >= self.reputation_threshold]
        if not active_participants:
            if not self.participants: self.M_t = 0
            else: self.M_t = 1 
            return

        delta_reputations_map = {} 
        delta_reputations_list = []
        for p in active_participants:
            if hasattr(p, 'get_delta_reputation'):
                delta = p.get_delta_reputation()
                delta_reputations_list.append(delta)
                delta_reputations_map[p.id] = delta
        
        if not delta_reputations_list:
            return

        sorted_delta_r = sorted(delta_reputations_list, reverse=True)
        
        k_prime_t_candidate = 1
        if len(sorted_delta_r) > 1:
            max_gap = -1.0
            current_max_gap_idx = 0 
            for j in range(len(sorted_delta_r) - 1):
                gap = sorted_delta_r[j] - sorted_delta_r[j+1]
                if gap > max_gap and j > 1: 
                    max_gap = gap
                    current_max_gap_idx = j 
            k_prime_t_candidate = current_max_gap_idx + 1

        elif len(sorted_delta_r) == 1:
            k_prime_t_candidate = 1
        else: 
            return

        num_positive_delta_reps = sum(1 for dr in sorted_delta_r if dr > 1e-4) 
        final_k_prime_t = k_prime_t_candidate
        
        if k_prime_t_candidate <= 1 and num_positive_delta_reps > 1:
            final_k_prime_t = max(k_prime_t_candidate, min(num_positive_delta_reps, 3)) 
        
        final_k_prime_t = min(final_k_prime_t, len(active_participants)) 
        final_k_prime_t = max(1, final_k_prime_t) 

        new_M_t_float = self.omega_m_update * self.M_t + (1 - self.omega_m_update) * final_k_prime_t
        
        self.M_t = max(1, int(np.round(new_M_t_float)))
        if active_participants: 
             self.M_t = min(self.M_t, len(active_participants))
        self.M_t = min(self.M_t, len(self.participants)) 


    # 将梯度字典展平为一维张量
    def _flatten_gradient_dict(self, gradient_dict):
        if gradient_dict is None or not isinstance(gradient_dict, dict):
            return None
        try:
            flat_parts = []
            for p_tensor in gradient_dict.values(): 
                if isinstance(p_tensor, torch.Tensor):
                    flat_parts.append(p_tensor.detach().view(-1).cpu())
            if not flat_parts: return None
            return torch.cat(flat_parts)
        except Exception as e:
            print(f"Error flattening gradient dict: {e}")
            return None


    # 运行一轮模拟
    def run_one_round(self):
        self.current_round += 1
        m_t_for_this_round = self.M_t
        print(f"\n--- 第 {self.current_round}/{self.max_rounds} 轮 (M_t = {m_t_for_this_round}) ---")
        
        if not self.requester or not self.participants:
            print("请求者或参与者未初始化。结束本轮。")
            self.simulation_stats["round_number"].append(self.current_round)
            self.simulation_stats["model_accuracy"].append(self.final_global_model_performance) 
            self.simulation_stats["active_honest_clients"].append(0)
            self.simulation_stats["active_free_riders"].append(0)
            self.simulation_stats["M_t_value"].append(m_t_for_this_round) 
            self.simulation_stats["cumulative_total_incentive_cost"].append(self.total_rewards_paid)
            self.simulation_stats["cumulative_real_incentive_cost"].append(self.rewards_paid_to_honest_clients)
            current_tir = (self.rewards_paid_to_honest_clients / self.total_rewards_paid) if self.total_rewards_paid > 1e-9 else 0.0
            self.simulation_stats["cumulative_tir_history"].append(current_tir)
            return True 

        round_start_global_state = copy.deepcopy(self.requester.global_model.state_dict())
        round_start_global_accuracy, _ = self.requester.evaluate_global_model()

        all_client_gradient_info_for_stats = [] 
        client_updates_for_submission = {} 

        for p in self.participants:
            p.set_model_state(copy.deepcopy(round_start_global_state)) 
            flat_gradient_for_stats_calc = None
            
            if p.type == "honest_client":
                p.perf_before_local_train, _ = p.evaluate_model(on_val_set=True)
                p.local_train()
                p.gen_true_update(round_start_global_state) 
            else: 
                p.gen_fabric_update(
                    self.current_round,
                    self.requester.global_model_param_diff_history,
                    round_start_global_state, 
                    len(self.participants)
                )

            if p.current_update: 
                client_updates_for_submission[p.id] = copy.deepcopy(p.current_update)
                flat_gradient_for_stats_calc = self._flatten_gradient_dict(p.current_update)
            
            all_client_gradient_info_for_stats.append({
                'id': p.id,
                'type': self.client_types[p.id],
                'gradient_flat': flat_gradient_for_stats_calc 
            })
        
        l2_norms_this_round = {client_data['id']: np.nan for client_data in all_client_gradient_info_for_stats}
        cosine_sims_this_round = {client_data['id']: np.nan for client_data in all_client_gradient_info_for_stats}
        if self.current_round > self.num_honest_rounds_for_fr_estimation:
            avg_honest_update_direction_for_sim = None
            honest_normalized_gradients_for_avg_sim = []
            for client_data in all_client_gradient_info_for_stats:
                if client_data['type'] == "honest_client":
                    flat_grad = client_data['gradient_flat']
                    if flat_grad is not None and flat_grad.numel() > 0:
                        norm_val = torch.linalg.norm(flat_grad).item() 
                        if norm_val > 1e-9: 
                            honest_normalized_gradients_for_avg_sim.append(flat_grad / norm_val)
            if honest_normalized_gradients_for_avg_sim:
                valid_tensors_for_stacking = [t for t in honest_normalized_gradients_for_avg_sim if t is not None and t.numel() > 0]
                if valid_tensors_for_stacking:
                    sum_normalized_honest_gradients = torch.sum(torch.stack(valid_tensors_for_stacking), dim=0)
                    norm_of_sum = torch.linalg.norm(sum_normalized_honest_gradients)
                    if norm_of_sum > 1e-9:
                        avg_honest_update_direction_for_sim = sum_normalized_honest_gradients / norm_of_sum
            for client_data in all_client_gradient_info_for_stats:
                p_id = client_data['id']
                flat_grad = client_data['gradient_flat']
                if flat_grad is not None and flat_grad.numel() > 0:
                    l2_norms_this_round[p_id] = torch.linalg.norm(flat_grad).item()
                    if avg_honest_update_direction_for_sim is not None:
                        if flat_grad.shape == avg_honest_update_direction_for_sim.shape:
                            try:
                                cosine_sims_this_round[p_id] = F.cosine_similarity(
                                    flat_grad.unsqueeze(0).float(), 
                                    avg_honest_update_direction_for_sim.unsqueeze(0).float(), 
                                    dim=1
                                ).item()
                            except Exception as e:
                                print(f"计算余弦相似度时出错 (客户端 {p_id}, 轮次 {self.current_round}): {e}")
        for client_data in all_client_gradient_info_for_stats:
            p_id = client_data['id']
            self.client_l2_norm_history[p_id].append(l2_norms_this_round[p_id])
            self.client_cosine_similarity_history[p_id].append(cosine_sims_this_round[p_id])

        honest_bids_promises, honest_bids_rewards, honest_bids_ratios = [], [], []
        num_bidding_honest_clients = 0
        for p_bid in self.participants:
            if p_bid.reputation < self.reputation_threshold:
                p_bid.bid = {} 
                continue
            if p_bid.type == "honest_client":
                bid_data = p_bid.submit_bid() 
                if bid_data and 'promise' in bid_data and 'reward' in bid_data and bid_data['reward'] > 1e-6:
                    honest_bids_promises.append(bid_data['promise'])
                    honest_bids_rewards.append(bid_data['reward'])
                    honest_bids_ratios.append(bid_data['promise'] / bid_data['reward'])
                    num_bidding_honest_clients +=1
        if num_bidding_honest_clients > 0 :
            highest_honest_effectiveness = max(honest_bids_ratios)
            lowest_honest_effectiveness = min(honest_bids_ratios)
            lowest_honest_promise = min(honest_bids_promises) 
            avg_honest_promise = np.mean(honest_bids_promises)
            for p_fr_bid in self.participants: 
                if p_fr_bid.reputation >= self.reputation_threshold and p_fr_bid.type == "free_rider":
                    p_fr_bid.submit_bid(highest_honest_effectiveness, lowest_honest_effectiveness, lowest_honest_promise, avg_honest_promise)
        else: 
            for p_fr_bid_default in self.participants:
                if p_fr_bid_default.reputation >= self.reputation_threshold and p_fr_bid_default.type == "free_rider":
                    p_fr_bid_default.submit_bid(0,0,0,0) 

        selected_participants = self.requester.select_participants(self.participants, m_t_for_this_round, self.reputation_threshold) 
        
        if not selected_participants:
            print("本轮没有参与者被选中。")
        else:
            print(f"选中 {len(selected_participants)} 个参与者: {[p.id for p in selected_participants]}")
            updates_to_verify = []
            for p_sel in selected_participants:
                update_content = client_updates_for_submission.get(p_sel.id)
                if update_content: 
                    updates_to_verify.append({"participant": p_sel, "update": update_content})
                else:
                    print(f"警告: 选中参与者 {p_sel.id} 没有可提交的更新内容。")

            if updates_to_verify:
                verification_outcomes, _ = self.requester.verify_and_aggregate_updates(
                    updates_to_verify, round_start_global_accuracy 
                )
                
                rewards_paid_ref = [self.total_rewards_paid] 
                self.requester.update_reputations_and_pay(self.participants, verification_outcomes, rewards_paid_ref)
                self.total_rewards_paid = rewards_paid_ref[0]

                current_round_rewards_to_honest_clients_this_round = 0
                for outcome in verification_outcomes: 
                    if outcome["successful_verification"]:
                        participant_id = outcome["participant_id"]
                        participant = next((p_find for p_find in self.participants if p_find.id == participant_id), None)
                        if participant and self.client_types.get(participant.id) == "honest_client":
                            reward_for_this_honest_client = participant.bid.get('reward', 0)
                            current_round_rewards_to_honest_clients_this_round += reward_for_this_honest_client
                self.rewards_paid_to_honest_clients += current_round_rewards_to_honest_clients_this_round
                
                for outcome in verification_outcomes: 
                    par = next((p_find for p_find in self.participants if p_find.id == outcome["participant_id"]), None)
                    if par:
                        status_str = "成功" if outcome["successful_verification"] else "失败"
                        print(f"  - {par.id} ({self.client_types[par.id]}), 声誉: {par.reputation:.2f}, "
                              f"投标: P={par.bid.get('promise',0):.3f}/R={par.bid.get('reward',0):.2f}, "
                              f"观察提升: {outcome.get('observed_increase',0):.3f}, 状态: {status_str}")
        
        for p_every in self.participants:
            p_every.update_reputation_history() 

        for p_track in self.participants:
            # 此处 p_track.id 必然在 self.client_reputation_history 中，
            # 因为已在 initialize_environment 中为所有参与者添加了初始声誉
            self.client_reputation_history[p_track.id].append(p_track.reputation) # 追加本轮结束后的声誉

        self.requester.update_global_model_history() 
        self.final_global_model_performance, _ = self.requester.evaluate_global_model()
        
        self.update_M_t() 
        
        num_active_total = sum(1 for p in self.participants if p.reputation >= self.reputation_threshold)
        num_active_honest = sum(1 for p in self.participants if self.client_types.get(p.id) == "honest_client" and p.reputation >= self.reputation_threshold)
        num_active_free_riders = sum(1 for p in self.participants if self.client_types.get(p.id) == "free_rider" and p.reputation >= self.reputation_threshold)
        
        current_cumulative_tir = 0.0
        if self.total_rewards_paid > 1e-9:
            current_cumulative_tir = self.rewards_paid_to_honest_clients / self.total_rewards_paid
        
        print(f"第 {self.current_round} 轮结束: 活跃总数={num_active_total}, 活跃诚实={num_active_honest}, 活跃FR={num_active_free_riders}, "
              f"总奖励(累计)={self.total_rewards_paid:.2f}, 支付给诚实客户端奖励(累计)={self.rewards_paid_to_honest_clients:.2f}, "
              f"全局性能={self.final_global_model_performance:.4f}, 真实激励率(累计): {current_cumulative_tir:.4f}, "
              f"下一轮 M_t 将是: {self.M_t}") 

        self.simulation_stats["round_number"].append(self.current_round)
        self.simulation_stats["model_accuracy"].append(self.final_global_model_performance)
        self.simulation_stats["active_honest_clients"].append(num_active_honest)
        self.simulation_stats["active_free_riders"].append(num_active_free_riders)
        self.simulation_stats["M_t_value"].append(m_t_for_this_round) 
        self.simulation_stats["cumulative_total_incentive_cost"].append(self.total_rewards_paid)
        self.simulation_stats["cumulative_real_incentive_cost"].append(self.rewards_paid_to_honest_clients)
        self.simulation_stats["cumulative_tir_history"].append(current_cumulative_tir)
        
        return self.check_termination_condition()


    # 检查终止条件
    def check_termination_condition(self):
        if not self.participants: 
            print("没有参与者，终止模拟。")
            self.termination_round = self.current_round 
            return True
        if self.target_accuracy_threshold is not None:
            if self.final_global_model_performance >= self.target_accuracy_threshold:
                print(f"目标模型性能 {self.target_accuracy_threshold:.4f} 已达到 (当前性能: {self.final_global_model_performance:.4f})。模拟成功终止。")
                self.termination_round = self.current_round
                return True
        if self.current_round >= self.max_rounds:
            print(f"已达到最大模拟轮数 ({self.max_rounds})。终止模拟。")
            self.termination_round = self.max_rounds 
            return True
        if self.num_honest_clients > 0 and self.current_round > 0: 
            num_active_honest_val = sum(1 for p in self.participants if self.client_types.get(p.id) == "honest_client" and p.reputation >= self.reputation_threshold)
            if num_active_honest_val == 0:
                print("没有活跃的诚实客户端了。模拟可能卡住或所有诚实客户端被错误惩罚。终止模拟。")
                self.termination_round = self.current_round 
                return True
        if self.current_round > 0: 
            num_active_total_final_check_val = sum(1 for p in self.participants if p.reputation >= self.reputation_threshold)
            if num_active_total_final_check_val == 0: 
                print("没有活跃的参与者了。终止模拟。")
                self.termination_round = self.current_round
                return True
        if self.num_free_riders > 0 and not self.all_fr_eliminated_logged: 
            num_active_free_riders = sum(1 for p in self.participants if self.client_types.get(p.id) == "free_rider" and p.reputation >= self.reputation_threshold)
            if num_active_free_riders == 0 and self.current_round > 0:
                if self.target_accuracy_threshold is None: 
                    print("所有搭便车者已被清除 (且未设定目标准确率)。终止模拟。")
                    self.termination_round = self.current_round
                    return True
                else: 
                    print(f"信息: 所有搭便车者在第 {self.current_round} 轮被清除。模拟将继续以尝试达到目标准确率。")
                    self.all_fr_eliminated_logged = True 
        return False


    # 保存模拟统计数据到 JSON 文件
    def save_simulation_stats(self, filename="simulation_results.json"):
        """
        将模拟过程中收集的统计数据保存到 JSON 文件。
        包括模拟参数、最终总结性指标、每轮的详细数据以及各客户端声誉历史。
        """
        if not self.simulation_stats["round_number"]: 
            print("没有统计数据可供保存。")
            return

        records = []
        num_recorded_rounds = len(self.simulation_stats["round_number"]) 
        stat_keys = list(self.simulation_stats.keys())

        for i in range(num_recorded_rounds):
            record = {}
            for key in stat_keys:
                if i < len(self.simulation_stats[key]):
                    record[key] = self.simulation_stats[key][i]
                else:
                    record[key] = None 
            records.append(record)
        
        final_fpr = 0.0
        if self.num_honest_clients > 0 and self.participants: 
            num_honest_eliminated = sum(1 for p in self.participants if self.client_types.get(p.id) == "honest_client" and p.reputation < self.reputation_threshold)
            final_fpr = num_honest_eliminated / self.num_honest_clients
        elif self.num_honest_clients == 0 :
             final_fpr = 0.0 

        final_tir = (self.rewards_paid_to_honest_clients / self.total_rewards_paid) if self.total_rewards_paid > 1e-9 else 0.0
        
        # 准备最终要保存的数据结构
        final_data_to_save = {
            "simulation_parameters": self.params_X, 
            "simulation_summary": { 
                "termination_round": self.termination_round, # 终止轮数
                "total_incentive_cost_final": self.total_rewards_paid, # 总激励开销
                "real_incentive_cost_final": self.rewards_paid_to_honest_clients, # 支付给诚实客户端的激励开销
                "false_positive_rate_final": final_fpr, # 诚实客户端误判率
                "global_model_accuracy_final": self.final_global_model_performance, # 最终全局模型准确率
                "true_incentive_rate_final": final_tir, # 真实激励率 (诚实客户端奖励占比)
                "num_total_participants_config": self.num_total_participants, # 总参与者数
                "num_honest_clients_config": self.num_honest_clients, # 诚实客户端数
                "num_free_riders_config": self.num_free_riders, # 搭便车者数
            },
            "per_round_statistics": records, # 每轮统计数据
            "client_reputation_over_rounds": self.client_reputation_history # 每个客户端在每轮的声誉历史
        }

        try:
            with open(filename, 'w', encoding='utf-8') as f: 
                json.dump(final_data_to_save, f, indent=4, ensure_ascii=False) 
            print(f"模拟统计数据已成功保存到 {filename}")
        except IOError as e:
            print(f"保存统计数据到文件时发生IO错误: {e}")
        except TypeError as e:
            print(f"序列化统计数据时发生类型错误 (可能包含无法JSON序列化的对象): {e}")
            try:
                def fallback_serializer(obj):
                    return str(obj)
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(final_data_to_save, f, indent=4, ensure_ascii=False, default=fallback_serializer)
                print(f"模拟统计数据已尝试使用备用序列化器保存到 {filename}")
            except Exception as e_fallback:
                 print(f"使用备用序列化器保存统计数据仍然失败: {e_fallback}")
    

    # 运行模拟
    def run_simulation(self):
        try: 
            self.initialize_environment()
        except ValueError as e:
            print(f"模拟初始化失败: {e}")
            self.termination_round = 0 
            # 即使初始化失败，如果 client_reputation_history 有部分数据（例如空的 client id 映射），也尝试保存
            self.save_simulation_stats(f"simulation_results_init_fail_N{self.num_total_participants}_Nf{self.num_free_riders}.json")
            return self.max_rounds, float('inf'), 1.0, 0.0, 0.0 
        
        if not self.requester or (not self.participants and (self.num_honest_clients > 0 or self.num_free_riders > 0)) :
             print("模拟无法开始，因为请求者或参与者未能正确初始化。")
             self.termination_round = 0
             self.save_simulation_stats(f"simulation_results_start_fail_N{self.num_total_participants}_Nf{self.num_free_riders}.json")
             return self.max_rounds, float('inf'), 1.0, 0.0, 0.0

        terminated = False
        while not terminated:
            terminated = self.run_one_round()
            
            if not terminated and \
               self.final_global_model_performance < self.min_performance_constraint * 0.5 and \
               self.current_round > min(5, self.max_rounds / 4) and \
               self.max_rounds > 5 :
                print(f"全局模型性能 ({self.final_global_model_performance:.4f}) 过低 (低于约束 {self.min_performance_constraint*0.5:.4f}，用于帕累托优化场景)，提前终止。")
                if self.termination_round == self.max_rounds: 
                    self.termination_round = self.current_round
                terminated = True 
            
            if terminated: 
                break
        
        T_term = self.termination_round 
        C_total = self.total_rewards_paid
        
        num_honest_clients_at_start = max(1, self.num_honest_clients) 
        num_honest_eliminated = 0
        if self.participants: 
            num_honest_eliminated = sum(1 for p in self.participants if self.client_types.get(p.id) == "honest_client" and p.reputation < self.reputation_threshold)
        
        FPR = num_honest_eliminated / num_honest_clients_at_start if num_honest_clients_at_start > 0 else 0.0
        if self.num_honest_clients == 0: 
            FPR = 0.0

        PFM_final = self.final_global_model_performance

        true_incentive_rate_final = 0.0
        if self.total_rewards_paid > 1e-9:
            true_incentive_rate_final = self.rewards_paid_to_honest_clients / self.total_rewards_paid
        
        print(f"\n--- 模拟结束 ---")
        print(f"T_term (终止轮数): {T_term}")
        print(f"C_total (总奖励开销): {C_total:.2f}")
        print(f"FPR (诚实客户端误判率): {FPR:.4f} ({num_honest_eliminated}/{self.num_honest_clients if self.num_honest_clients > 0 else 'N/A'})")
        print(f"PFM_final (最终全局模型准确率): {PFM_final:.4f}")
        print(f"TIR (真实激励率 - 诚实客户端奖励占比): {true_incentive_rate_final:.4f}")

        self.save_simulation_stats() 

        return T_term, C_total, FPR, PFM_final, true_incentive_rate_final


    # 评估参数以进行优化
    def evaluate_parameters_for_optimization(self):
        T_term, C_total, FPR, PFM_final, TIR = self.run_simulation()
        
        objectives = [C_total, FPR] 
        constraint_violation = self.min_performance_constraint - PFM_final 
        
        other_metrics = {
            "T_term": T_term, 
            "PFM_final": PFM_final,
            "TIR": TIR,
            "C_total": C_total, 
            "FPR": FPR 
        }
        return objectives, [constraint_violation], other_metrics


# 设置随机种子以确保可重复性
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    set_random_seed(42)
    example_params_X = {
        "N": 10, "N_f": 3, "T_max": 50, "initial_M_t": 5, 
        "iid_data": True, "non_iid_alpha": 10,
        "alpha_reward": 2.0, "beta_penalty_base": 1.1, "omega_m_update": 0.4, 
        "q_rounds_rep_change": 5, 
        "initial_reputation": 10.0, 
        "reputation_threshold": 0.01, 
        "min_performance_constraint": 0.10, 
        "target_accuracy_threshold": 0.90, 
        
        "num_honest_rounds_for_fr_estimation": 2, 
        "adv_attack_c_param_fr": 0.5,            
        "adv_attack_noise_dim_fraction_fr": 1.0,
        "adv_attack_scaled_delta_noise_std": 0.001,

        "bid_gamma_honest": 1.0, 
        "local_epochs_honest": 1, 
        "lr_honest": 0.005,
        "batch_size_honest": 64,

        "adaptive_bid_adjustment_intensity_gamma_honest": 0.15, 
        "adaptive_bid_max_adjustment_delta_honest": 0.4,     
        "min_commitment_scaling_factor_honest": 0.2          
    }
    sim = Simulation(params_X=example_params_X)
    
    objectives, constraints, other_metrics = sim.evaluate_parameters_for_optimization()
    
    print(f"\n--- 帕累托优化及其他指标评估结果 ---")
    print(f"优化目标 (C_total, FPR): ({objectives[0]:.2f}, {objectives[1]:.4f})")
    print(f"约束违反 (min_PFM - PFM_final <= 0): {constraints[0]:.4f}")
    if constraints[0] <= 1e-5: 
        print("约束：最终模型性能达标 (相对于 min_performance_constraint)。")
    else:
        print("约束：最终模型性能未达标 (相对于 min_performance_constraint)。")
    
    print(f"\n其他重要指标 (来自 other_metrics):")
    print(f"  T_term (终止轮数): {other_metrics['T_term']}")
    print(f"  PFM_final (最终全局模型准确率): {other_metrics['PFM_final']:.4f}")
    print(f"  TIR (真实激励率 - 诚实客户端奖励占比): {other_metrics['TIR']:.4f}")
    print(f"  C_total (总奖励开销, from metrics): {other_metrics['C_total']:.2f}")
    print(f"  FPR (诚实客户端误判率, from metrics): {other_metrics['FPR']:.4f}")

    # 示例：读取并分析保存的JSON文件
    # import pandas as pd
    # try:
    #     with open("simulation_results.json", 'r', encoding='utf-8') as f: # 确保使用utf-8
    #         results_data = json.load(f)
        
    #     print("\n--- 从JSON文件加载的每轮数据预览 (前5轮) ---")
    #     if "per_round_statistics" in results_data:
    #         per_round_df = pd.DataFrame(results_data["per_round_statistics"])
    #         print(per_round_df.head())
    #     else:
    #         print("JSON文件中未找到 'per_round_statistics'。")

    #     print("\n--- 从JSON文件加载的客户端声誉历史预览 ---")
    #     if "client_reputation_over_rounds" in results_data:
    #         client_rep_history = results_data["client_reputation_over_rounds"]
    #         for client_id, rep_history in list(client_rep_history.items())[:2]: # 仅预览前2个客户端
    #             print(f"客户端 {client_id}: {rep_history}")
    #     else:
    #         print("JSON文件中未找到 'client_reputation_over_rounds'。")
            
    # except FileNotFoundError:
    #     print("\nsimulation_results.json 文件未找到。")
    # except Exception as e:
    #     print(f"\n读取或处理JSON文件时出错: {e}")