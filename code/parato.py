import torch
import random
import numpy as np
import copy
from torch.utils.data import DataLoader
import json
import time
import traceback
import os
# import concurrent.futures #不再需要并发

# --- Pymoo 导入 ---
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# --- 自定义模拟组件导入 ---
from system import get_mnist_data, Global_Model, Requester
from honest_client import HonestClient
from free_rider import FreeRider


# 用于存储所有评估的详细结果，包括声誉历史
optimization_evaluation_history = [] # 移到全局，因为 ParetoOptimizationProblem._evaluate 会填充它

class Simulation:
    def __init__(self, params_X):
        self.params_X = params_X
        self.num_total_participants = params_X.get("N")
        self.num_free_riders = params_X.get("N_f")
        self.num_honest_clients = self.num_total_participants - self.num_free_riders

        self.verbose = params_X.get("verbose")

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if self.verbose:
            print(f"--- SIM: 使用设备: {self.device} ---")

        self.max_rounds = params_X.get("T_max")
        self.target_accuracy_threshold = params_X.get("target_accuracy_threshold")
        self.initial_M_t = params_X.get("initial_M_t")
        self.iid_data_distribution = params_X.get("iid_data")
        self.non_iid_alpha = params_X.get("non_iid_alpha")

        self.alpha_reward = params_X.get("alpha_reward")
        self.beta_penalty_base = params_X.get("beta_penalty_base")
        self.omega_m_update = params_X.get("omega_m_update") # 新增：确保从参数中获取
        self.q_rounds_rep_change = params_X.get("q_rounds_rep_change")
        self.initial_reputation = params_X.get("initial_reputation")
        self.reputation_threshold = params_X.get("reputation_threshold")
        self.min_performance_constraint = params_X.get("min_performance_constraint")

        self.num_honest_rounds_for_fr_estimation = params_X.get("num_honest_rounds_for_fr_estimation")
        self.adv_attack_c_param_fr = params_X.get("adv_attack_c_param_fr")
        self.adv_attack_noise_dim_fraction_fr = params_X.get("adv_attack_noise_dim_fraction_fr")
        self.adv_attack_scaled_delta_noise_std_fr = params_X.get("adv_attack_scaled_delta_noise_std")

        self.adaptive_bid_adjustment_intensity_gamma_honest = params_X.get("adaptive_bid_adjustment_intensity_gamma_honest")
        self.adaptive_bid_max_adjustment_delta_honest = params_X.get("adaptive_bid_max_adjustment_delta_honest")
        self.min_commitment_scaling_factor_honest = params_X.get("min_commitment_scaling_factor_honest")

        self.bid_gamma_honest = params_X.get("bid_gamma_honest")
        self.local_epochs_honest = params_X.get("local_epochs_honest")
        self.lr_honest = params_X.get("lr_honest")
        self.batch_size_honest = params_X.get("batch_size_honest")

        self.participants = []
        self.requester = None
        self.current_round = 0
        self.M_t = self.initial_M_t
        self.total_rewards_paid = 0.0
        self.rewards_paid_to_honest_clients = 0.0
        self.final_global_model_performance = 0.0
        self.termination_round = self.max_rounds

        self.client_reputation_history = {}
        self.client_types = {}

        # 搭便车者剔除和新目标相关的追踪变量
        self.total_rewards_obtained_by_fr = 0.0 # 新增：所有搭便车者获得的累计奖励
        self.total_rewards_obtained_by_fr_at_elimination = float('inf') # 新增：当所有FR被剔除时，他们获得的总奖励
        self.round_at_all_fr_eliminated = -1
        self.all_fr_elimination_achieved_flag = False

        self.simulation_stats = {
            "round_number": [], "model_accuracy": [], "active_honest_clients": [],
            "active_free_riders": [], "M_t_value": [], "cumulative_total_incentive_cost": [],
            "cumulative_real_incentive_cost": [], "cumulative_tir_history": []
        }

    def log_message(self, message):
        if self.verbose:
            print(message)

    def initialize_environment(self):
        self.log_message("--- initialize_environment START ---")
        initial_global_model = Global_Model()
        client_datasets, test_dataset_global = get_mnist_data(
            self.num_honest_clients, self.iid_data_distribution, self.non_iid_alpha
        )
        if self.num_honest_clients > 0 and (not client_datasets or self.num_honest_clients > len(client_datasets)):
            self.log_message(f"警告: 请求 {self.num_honest_clients} 个诚实客户端数据，实际分配 {len(client_datasets)} 个。")
            if not client_datasets and self.num_honest_clients > 0:
                 raise ValueError(f"诚实客户端数据分配失败: 请求 {self.num_honest_clients}, 得到 {len(client_datasets)}")

        pin_memory_flag = self.device.type != "cpu"
        # 注意：之前讨论过 num_workers 的问题，这里保持 os.cpu_count()*2，但建议在实际运行中根据系统情况调整为0或小值
        num_workers_val = os.cpu_count() 
        test_loader = DataLoader(test_dataset_global, batch_size=128, shuffle=False,
                                 pin_memory=pin_memory_flag, num_workers=num_workers_val)

        self.requester = Requester(initial_global_model, test_loader, self.device,
                             alpha_reward=self.alpha_reward,
                             beta_penalty_base=self.beta_penalty_base)
        self.participants = []
        temp_participants = []
        for i in range(self.num_honest_clients):
            if i >= len(client_datasets) or not client_datasets[i] or len(client_datasets[i]) == 0:
                self.log_message(f"警告：诚实客户端 h_{i} 数据不足或无效，跳过。")
                continue
            temp_participants.append(HonestClient(
                id=f"h_{i}", init_rep=self.initial_reputation,
                tra_round_num=self.q_rounds_rep_change,
                device=self.device, client_dataset=client_datasets[i], train_size=0.8,
                init_commit_scaling_factor=self.bid_gamma_honest,
                batch_size=self.batch_size_honest, local_epochs=self.local_epochs_honest,
                lr=self.lr_honest,
                adapt_bid_adj_intensity=self.adaptive_bid_adjustment_intensity_gamma_honest,
                adapt_bid_max_delta=self.adaptive_bid_max_adjustment_delta_honest,
                min_commit_scaling_factor=self.min_commitment_scaling_factor_honest
            ))
        for i in range(self.num_free_riders):
            temp_participants.append(FreeRider(
                id=f"f_{i}", init_rep=self.initial_reputation,
                tra_round_num=self.q_rounds_rep_change,
                device=self.device,
                est_round_num=self.num_honest_rounds_for_fr_estimation,
                atk_c_param=self.adv_attack_c_param_fr,
                atk_noise_dim=self.adv_attack_noise_dim_fraction_fr,
                atk_est_noise_std=self.adv_attack_scaled_delta_noise_std_fr,
            ))

        if not temp_participants and (self.num_honest_clients > 0 or self.num_free_riders > 0) :
            raise ValueError("没有参与者被初始化。")

        random.shuffle(temp_participants)
        self.participants = temp_participants
        self.client_reputation_history = {}
        for p in self.participants:
            self.client_types[p.id] = p.type
            self.client_reputation_history[p.id] = []

        self.current_round = 0
        self.M_t = min(self.initial_M_t, len(self.participants)) if self.participants else 0
        self.total_rewards_paid = 0.0
        self.rewards_paid_to_honest_clients = 0.0
        self.total_rewards_obtained_by_fr = 0.0 # 重置
        self.total_rewards_obtained_by_fr_at_elimination = float('inf') # 重置
        self.round_at_all_fr_eliminated = -1
        self.all_fr_elimination_achieved_flag = False

        for key_stat in self.simulation_stats: self.simulation_stats[key_stat] = []

        if self.requester:
            self.final_global_model_performance, _ = self.requester.evaluate_global_model()
            if self.requester.global_model:
                 self.requester.previous_global_model_state_flat = self.requester._flatten_params(self.requester.global_model.state_dict())
            self.log_message(f"初始全局模型性能: {self.final_global_model_performance:.4f}")
            self.simulation_stats["round_number"].append(0)
            self.simulation_stats["model_accuracy"].append(self.final_global_model_performance)
            initial_active_honest = sum(1 for p in self.participants if self.client_types.get(p.id) == "honest_client" and p.reputation >= self.reputation_threshold)
            initial_active_free_riders = sum(1 for p in self.participants if self.client_types.get(p.id) == "free_rider" and p.reputation >= self.reputation_threshold)
            self.simulation_stats["active_honest_clients"].append(initial_active_honest)
            self.simulation_stats["active_free_riders"].append(initial_active_free_riders)
            self.simulation_stats["M_t_value"].append(self.M_t)
            self.simulation_stats["cumulative_total_incentive_cost"].append(self.total_rewards_paid)
            self.simulation_stats["cumulative_real_incentive_cost"].append(self.rewards_paid_to_honest_clients)
            self.simulation_stats["cumulative_tir_history"].append(0.0)
            for p_init_rep in self.participants:
                if p_init_rep.id in self.client_reputation_history:
                     self.client_reputation_history[p_init_rep.id].append(p_init_rep.reputation)
        else:
            print("警告：服务器未正确初始化。")
        self.log_message("--- initialize_environment END ---")

    def update_M_t(self):
        if self.current_round == 0 or not self.participants:
            return
        active_participants = [p for p in self.participants if p.reputation >= self.reputation_threshold]
        if not active_participants:
            self.M_t = 0 if not self.participants else 1
            return
        delta_reputations_list = [p.get_delta_reputation() for p in active_participants if hasattr(p, 'get_delta_reputation')]
        if not delta_reputations_list: return
        sorted_delta_r = sorted(delta_reputations_list, reverse=True)
        k_prime_t_candidate = 1
        if len(sorted_delta_r) > 1:
            max_gap, current_max_gap_idx = -1.0, 0
            for j in range(len(sorted_delta_r) - 1):
                gap = sorted_delta_r[j] - sorted_delta_r[j+1]
                if gap > max_gap and j > 1:
                    max_gap = gap
                    current_max_gap_idx = j
            k_prime_t_candidate = current_max_gap_idx + 1
        elif len(sorted_delta_r) == 1: k_prime_t_candidate = 1
        else: return
        num_positive_delta_reps = sum(1 for dr in sorted_delta_r if dr > 1e-4)
        final_k_prime_t = k_prime_t_candidate
        if k_prime_t_candidate <= 1 and num_positive_delta_reps > 1:
            final_k_prime_t = max(k_prime_t_candidate, min(num_positive_delta_reps, 3))
        final_k_prime_t = min(final_k_prime_t, len(active_participants))
        final_k_prime_t = max(1, final_k_prime_t)
        new_M_t_float = self.omega_m_update * self.M_t + (1 - self.omega_m_update) * final_k_prime_t
        self.M_t = max(1, int(np.round(new_M_t_float)))
        if active_participants: self.M_t = min(self.M_t, len(active_participants))
        self.M_t = min(self.M_t, len(self.participants))

    # # 修改为实例方法
    # def train_participant(self, participant, global_model_state, current_round=None, model_diff_history=None, participant_count=None):
    #     participant.set_model_state(copy.deepcopy(global_model_state))
    #     if participant.type == "honest_client":
    #         participant.perf_before_local_train, _ = participant.evaluate_model(on_val_set=True)
    #         participant.local_train() 
    #         participant.gen_true_update(global_model_state)
    #     else:  # 搭便车者
    #         participant.gen_fabric_update(current_round, model_diff_history, global_model_state, participant_count)
    #     return participant.id, participant.current_update

    def run_one_round(self):
        self.current_round += 1
        m_t_for_this_round = self.M_t
        self.log_message(f"\n--- SIM: 第 {self.current_round}/{self.max_rounds} 轮 (M_t = {m_t_for_this_round}) ---")

        if not self.requester or not self.participants:
            self.log_message("请求者或参与者未初始化。结束本轮。") # 使用 self.log_message
            return True

        round_start_global_accuracy, _ = self.requester.evaluate_global_model()
        client_updates_for_submission = {}

        # # 串行处理参与者更新
        # for p in self.participants:
        #     # 为每个参与者传递本轮开始时的全局模型状态的独立拷贝
        #     current_global_model_state_for_p = copy.deepcopy(self.requester.global_model.state_dict())
        #     participant_id, update = self.train_participant(
        #         p,
        #         current_global_model_state_for_p,
        #         self.current_round,
        #         self.requester.global_model_param_diff_history,
        #         len(self.participants)
        #     )
        #     if update:
        #         client_updates_for_submission[p.id] = copy.deepcopy(update)

        for p in self.participants:
            p.set_model_state(copy.deepcopy(self.requester.global_model.state_dict())) 
            
            if p.type == "honest_client":
                p.perf_before_local_train, _ = p.evaluate_model(on_val_set=True)
                p.local_train()
                p.gen_true_update(self.requester.global_model.state_dict()) 
            else: 
                p.gen_fabric_update(
                    self.current_round,
                    self.requester.global_model_param_diff_history,
                    self.requester.global_model.state_dict(), 
                    len(self.participants)
                )

            if p.current_update: 
                client_updates_for_submission[p.id] = copy.deepcopy(p.current_update)

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
            highest_honest_effectiveness = max(honest_bids_ratios) if honest_bids_ratios else 0
            lowest_honest_effectiveness = min(honest_bids_ratios) if honest_bids_ratios else 0
            lowest_honest_promise = min(honest_bids_promises) if honest_bids_promises else 0
            avg_honest_promise = np.mean(honest_bids_promises) if honest_bids_promises else 0
            for p_fr_bid in self.participants:
                if p_fr_bid.reputation >= self.reputation_threshold and p_fr_bid.type == "free_rider":
                    p_fr_bid.submit_bid(highest_honest_effectiveness, lowest_honest_effectiveness, lowest_honest_promise, avg_honest_promise)
        else:
            for p_fr_bid_default in self.participants:
                if p_fr_bid_default.reputation >= self.reputation_threshold and p_fr_bid_default.type == "free_rider":
                    p_fr_bid_default.submit_bid(0,0,0,0)

        selected_participants = self.requester.select_participants(self.participants, m_t_for_this_round, self.reputation_threshold)

        current_round_rewards_to_freeriders_this_round = 0 # 初始化本轮给FR的奖励

        if not selected_participants:
            self.log_message("本轮没有参与者被选中。")
        else:
            self.log_message(f"选中 {len(selected_participants)} 个参与者: {[p.id for p in selected_participants]}")
            updates_to_verify = []
            for p_sel in selected_participants:
                update_content = client_updates_for_submission.get(p_sel.id)
                if update_content:
                    updates_to_verify.append({"participant": p_sel, "update": update_content})
                else:
                    self.log_message(f"警告: 选中参与者 {p_sel.id} 没有可提交的更新内容。")

            if updates_to_verify:
                verification_outcomes, _ = self.requester.verify_and_aggregate_updates(
                    updates_to_verify, round_start_global_accuracy
                )
                rewards_paid_ref = [self.total_rewards_paid]
                self.requester.update_reputations_and_pay(self.participants, verification_outcomes, rewards_paid_ref)
                self.total_rewards_paid = rewards_paid_ref[0]

                current_round_rewards_to_honest_clients_this_round = 0
                for outcome in verification_outcomes: # 修正TIR计算的逻辑错误
                    if outcome["successful_verification"]:
                        p_find = next((p for p in self.participants if p.id == outcome["participant_id"]), None)
                        if p_find:
                            if self.client_types.get(p_find.id) == "honest_client":
                                current_round_rewards_to_honest_clients_this_round += p_find.bid.get('reward', 0)
                            elif self.client_types.get(p_find.id) == "free_rider": # 新增：累加FR获得的奖励
                                current_round_rewards_to_freeriders_this_round += p_find.bid.get('reward', 0)
                self.rewards_paid_to_honest_clients += current_round_rewards_to_honest_clients_this_round
                self.total_rewards_obtained_by_fr += current_round_rewards_to_freeriders_this_round # 累加FR总奖励

                if self.verbose:
                    for outcome in verification_outcomes:
                        par = next((p_find_log for p_find_log in self.participants if p_find_log.id == outcome["participant_id"]), None)
                        if par:
                            status_str = "成功" if outcome["successful_verification"] else "失败"
                            self.log_message(f"  - {par.id} ({self.client_types[par.id]}), 声誉: {par.reputation:.2f}, "
                                  f"投标: P={par.bid.get('promise',0):.3f}/R={par.bid.get('reward',0):.2f}, "
                                  f"观察提升: {outcome.get('observed_increase',0):.3f}, 状态: {status_str}")

        for p_every in self.participants: p_every.update_reputation_history()
        for p_track in self.participants:
            if p_track.id in self.client_reputation_history:
                 self.client_reputation_history[p_track.id].append(p_track.reputation)
            else:
                 self.client_reputation_history[p_track.id] = [p_track.reputation]

        self.requester.update_global_model_history()
        self.final_global_model_performance, _ = self.requester.evaluate_global_model()
        self.update_M_t()

        num_active_honest = sum(1 for p in self.participants if self.client_types.get(p.id) == "honest_client" and p.reputation >= self.reputation_threshold)
        num_active_free_riders = sum(1 for p in self.participants if self.client_types.get(p.id) == "free_rider" and p.reputation >= self.reputation_threshold)

        if self.num_free_riders > 0 and not self.all_fr_elimination_achieved_flag and num_active_free_riders == 0:
            self.total_rewards_obtained_by_fr_at_elimination = self.total_rewards_obtained_by_fr # 更新FR获得奖励的记录点
            self.round_at_all_fr_eliminated = self.current_round
            self.all_fr_elimination_achieved_flag = True
            self.log_message(f"*** 所有 ({self.num_free_riders}) 个搭便车者已在第 {self.current_round} 轮被剔除。 ***")
            self.log_message(f"*** 剔除时搭便车者累计获得总奖励: {self.total_rewards_obtained_by_fr_at_elimination:.2f} ***")

        current_cumulative_tir = (self.rewards_paid_to_honest_clients / self.total_rewards_paid) if self.total_rewards_paid > 1e-9 else 0.0
        self.log_message(f"第 {self.current_round} 轮结束: 活跃诚实={num_active_honest}, 活跃FR={num_active_free_riders}, "
              f"总奖励(累计)={self.total_rewards_paid:.2f}, 支付给诚实客户端奖励(累计)={self.rewards_paid_to_honest_clients:.2f}, "
              f"搭便车者获得奖励(累计)={self.total_rewards_obtained_by_fr:.2f}, " # 新增日志
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

    def check_termination_condition(self):
        if not self.participants:
            self.log_message("没有参与者，终止模拟。")
            self.termination_round = self.current_round if self.current_round > 0 else 0
            return True
        if self.target_accuracy_threshold is not None and self.final_global_model_performance >= self.target_accuracy_threshold:
            self.log_message(f"目标模型性能 {self.target_accuracy_threshold:.4f} 已达到。终止。")
            self.termination_round = self.current_round
            return True
        if self.current_round >= self.max_rounds:
            self.log_message(f"已达到最大模拟轮数 ({self.max_rounds})。终止。")
            self.termination_round = self.max_rounds
            return True
        if self.num_honest_clients > 0 and self.current_round > 0:
            num_active_honest_val = sum(1 for p in self.participants if self.client_types.get(p.id) == "honest_client" and p.reputation >= self.reputation_threshold)
            if num_active_honest_val == 0:
                self.log_message("没有活跃的诚实客户端了。终止模拟。")
                self.termination_round = self.current_round
                return True
        if self.current_round > 0:
            num_active_total_final_check_val = sum(1 for p in self.participants if p.reputation >= self.reputation_threshold)
            if num_active_total_final_check_val == 0:
                self.log_message("没有活跃的参与者了。终止模拟。")
                self.termination_round = self.current_round
                return True
        return False

    def save_simulation_stats(self, filename="simulation_results.json"):
        if not self.simulation_stats["round_number"] and self.current_round == 0 :
             self.log_message("第0轮，统计数据列表为空，但会尝试保存摘要。")
        elif not self.simulation_stats["round_number"]:
            self.log_message("没有统计数据可供保存。")
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
        elif self.num_honest_clients == 0 : final_fpr = 0.0
        final_tir = (self.rewards_paid_to_honest_clients / self.total_rewards_paid) if self.total_rewards_paid > 1e-9 else 0.0

        # 确定用于保存的 FR 剔除时 FR 获得奖励的值
        rewards_fr_at_elim_to_save = self.total_rewards_obtained_by_fr_at_elimination
        if self.num_free_riders > 0 and not self.all_fr_elimination_achieved_flag:
            rewards_fr_at_elim_to_save = "Not_Achieved_Infinity" # 或者保持 float('inf') 但json可能不支持
        elif self.num_free_riders == 0:
            rewards_fr_at_elim_to_save = 0.0


        data_to_save = {
            "simulation_parameters": self.params_X,
            "simulation_summary": {
                "termination_round": self.termination_round,
                "total_incentive_cost_final": self.total_rewards_paid,
                "real_incentive_cost_final": self.rewards_paid_to_honest_clients,
                "total_rewards_obtained_by_fr_final": self.total_rewards_obtained_by_fr, # 新增最终FR获得总奖励
                "false_positive_rate_final": final_fpr,
                "global_model_accuracy_final": self.final_global_model_performance,
                "true_incentive_rate_final": final_tir,
                "rewards_obtained_by_fr_at_elimination": rewards_fr_at_elim_to_save, # 修改变量名
                "round_at_all_fr_eliminated": self.round_at_all_fr_eliminated if self.all_fr_elimination_achieved_flag else "Not_Achieved",
                "all_fr_elimination_achieved": self.all_fr_elimination_achieved_flag,
                "is_pareto_optimal": False, # 默认为非帕累托最优
            },
            "per_round_statistics": records,
            "client_reputation_history_per_round": self.client_reputation_history,
            "client_details": {}
        }
        for client_id, reputation_history in self.client_reputation_history.items():
            client_type = self.client_types.get(client_id, "unknown")
            data_to_save["client_details"][client_id] = {
                "type": client_type,
                "reputation_history": reputation_history,
                "final_reputation": reputation_history[-1] if reputation_history else None,
                "is_active_at_end": reputation_history[-1] >= self.reputation_threshold if reputation_history else False
            }
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=4, ensure_ascii=False, default=lambda o: str(o) if isinstance(o, (np.integer, np.floating, np.bool_)) else o)
            self.log_message(f"模拟统计数据已成功保存到 {filename}")
        except Exception as e:
            self.log_message(f"保存统计数据到文件 {filename} 时发生错误: {type(e).__name__} - {e}")

    def run_simulation(self):
        self.log_message(f"--- SIM: run_simulation CALLED (verbose={self.verbose}) ---")
        original_verbose_state = self.verbose
        if not self.verbose and 'PymooOpt' in self.params_X:
            self.verbose = True
            self.log_message(f"--- SIM: Temporarily forcing verbose ON for initialize_environment call during Pymoo eval ---")
        try:
            self.initialize_environment()
        except ValueError as e_val:
            self.log_message(f"!!!!!! SIM: run_simulation - ValueError during initialize_environment: {e_val} !!!!!!")
            self.verbose = original_verbose_state
            return [float('inf'), 1.0], [1.0e9, 1.0], {"PFM_final": 0.0, "error": f"ValueError in init_env: {e_val}", "client_reputation_history": self.client_reputation_history}
        except Exception as e_generic:
            self.log_message(f"!!!!!! SIM: run_simulation - UNEXPECTED Exception during initialize_environment: {type(e_generic).__name__} - {e_generic} !!!!!!")
            traceback.print_exc()
            self.verbose = original_verbose_state
            return [float('inf'), 1.0], [1.0e9, 1.0], {"PFM_final": 0.0, "error": f"Exception in init_env: {e_generic}", "client_reputation_history": self.client_reputation_history}
        finally:
            if self.verbose != original_verbose_state:
                 self.log_message(f"--- SIM: Restoring verbose to {original_verbose_state} ---")
                 self.verbose = original_verbose_state

        if not self.requester or (not self.participants and (self.num_honest_clients > 0 or self.num_free_riders > 0)) :
             self.log_message("--- SIM: run_simulation --- ABORTING early, requester or participants not properly initialized AFTER init_env call. ---")
             self.termination_round = 0
             if self.verbose:
                filename = f"sim_results_START_FAIL_N{self.num_total_participants}_Nf{self.num_free_riders}_alphaR{self.alpha_reward:.2f}_betaP{self.beta_penalty_base:.2f}_q{self.q_rounds_rep_change}.json"
                try: self.save_simulation_stats(filename)
                except: pass
             return [float('inf'), 1.0], [1.0e9, 1.0], {"PFM_final": 0.0, "error": "Requester/Participants not initialized", "client_reputation_history": self.client_reputation_history}

        self.log_message("--- SIM: run_simulation --- Proceeding to main simulation loop. ---")
        terminated = False
        while not terminated:
            terminated = self.run_one_round()
            if not terminated and \
               self.min_performance_constraint > 0 and \
               self.final_global_model_performance < self.min_performance_constraint * 0.25 and \
               self.current_round > min(10, self.max_rounds / 3) and \
               self.max_rounds > 10 :
                self.log_message(f"全局模型性能 ({self.final_global_model_performance:.4f}) 过低 (远低于约束 {self.min_performance_constraint*0.25:.4f})，提前终止。")
                if self.termination_round == self.max_rounds:
                    self.termination_round = self.current_round
                terminated = True
            if terminated: break

        T_term = self.termination_round
        C_total_final = self.total_rewards_paid
        num_honest_clients_at_start = max(1, self.num_honest_clients)
        num_honest_eliminated = 0
        if self.participants:
            num_honest_eliminated = sum(1 for p in self.participants if self.client_types.get(p.id) == "honest_client" and p.reputation < self.reputation_threshold)
        FPR_final = num_honest_eliminated / num_honest_clients_at_start if num_honest_clients_at_start > 0 else 0.0
        if self.num_honest_clients == 0: FPR_final = 0.0
        PFM_final = self.final_global_model_performance
        true_incentive_rate_final = (self.rewards_paid_to_honest_clients / C_total_final) if C_total_final > 1e-9 else 0.0

        # 第一个优化目标：当所有搭便车者被剔除时，他们总共获得的奖励
        obj1_rewards_fr_at_elim = self.total_rewards_obtained_by_fr_at_elimination
        if self.num_free_riders > 0 and not self.all_fr_elimination_achieved_flag:
            obj1_rewards_fr_at_elim = float('inf')
        elif self.num_free_riders == 0:
            obj1_rewards_fr_at_elim = 0.0

        self.log_message(f"\n--- SIM: 模拟结束 (run_simulation) ---")
        self.log_message(f"终止轮数 (T_term): {T_term}")
        self.log_message(f"最终总奖励开销 (C_total_final): {C_total_final:.2f}")
        self.log_message(f"最终误判率 (FPR_final): {FPR_final:.4f} ({num_honest_eliminated}/{self.num_honest_clients if self.num_honest_clients > 0 else 'N/A'})")
        self.log_message(f"最终模型准确率 (PFM_final): {PFM_final:.4f}")
        self.log_message(f"最终真实激励率 (TIR_final): {true_incentive_rate_final:.4f}")
        self.log_message(f"优化目标1 (搭便车者获得总奖励@剔除): {obj1_rewards_fr_at_elim if obj1_rewards_fr_at_elim != float('inf') else 'Infinite/Not Achieved'}")
        self.log_message(f"剔除所有FR时的轮数 (Round_elim_fr): {self.round_at_all_fr_eliminated if self.all_fr_elimination_achieved_flag else 'N/A'}")

        if self.verbose and not self.params_X.get('PymooOpt', False):
            filename = f"sim_N{self.num_total_participants}_Nf{self.num_free_riders}_alphaR{self.alpha_reward:.2f}_betaP{self.beta_penalty_base:.2f}_q{self.q_rounds_rep_change}_omegaM{self.omega_m_update:.2f}.json" #文件名中加入omega_m
            try: self.save_simulation_stats(filename)
            except: pass

        objectives_for_pareto = [obj1_rewards_fr_at_elim, FPR_final]
        performance_constraint_violation = self.min_performance_constraint - PFM_final
        elimination_constraint_violation = 0.0 if self.all_fr_elimination_achieved_flag or self.num_free_riders == 0 else 1.0
        constraints_violation_list = [performance_constraint_violation, elimination_constraint_violation]

        self.log_message(f"约束1违反 (性能): {performance_constraint_violation:.4f}")
        self.log_message(f"约束2违反 (FR剔除): {elimination_constraint_violation:.4f} (目标达成: {self.all_fr_elimination_achieved_flag or self.num_free_riders == 0})\n")


        other_metrics_to_return = {
            "T_term": T_term,
            "PFM_final": PFM_final,
            "TIR_final": true_incentive_rate_final,
            "C_total_final": C_total_final,
            "total_rewards_obtained_by_fr_at_elimination": obj1_rewards_fr_at_elim,
            "round_at_all_fr_eliminated": self.round_at_all_fr_eliminated if self.all_fr_elimination_achieved_flag else -1,
            "all_fr_elimination_achieved_flag": self.all_fr_elimination_achieved_flag,
            "performance_constraint_violation": performance_constraint_violation,
            "elimination_constraint_violation": elimination_constraint_violation,
            "client_reputation_history": copy.deepcopy(self.client_reputation_history)
        }
        return objectives_for_pareto, constraints_violation_list, other_metrics_to_return

    def evaluate_parameters_for_optimization(self):
        return self.run_simulation()

class ParetoOptimizationProblem(Problem):
    def __init__(self, base_sim_params):
        print("--- PYMOO: ParetoOptimizationProblem __init__ CALLED ---")
        self.base_sim_params = base_sim_params
        self.evaluation_counter = 0
        self.variable_names = ["alpha_reward", "beta_penalty_base", "q_rounds_rep_change", "omega_m_update"]
        # 帕累托优化问题的变量范围
        xl = np.array([1.0, 1.0, 3, 0.1], dtype=np.double) 
        xu = np.array([3.0, 2.0, 6, 0.5], dtype=np.double) 

        super().__init__(n_var=len(self.variable_names),
                         n_obj=2,
                         n_constr=2, # 2个约束
                         xl=xl,
                         xu=xu,
                         elementwise=True)
        print(f"--- PYMOO: ParetoOptimizationProblem super().__init__ FINISHED ---\n")

    def _evaluate(self, x_array, out, *args, **kwargs):
        print(f"--- PYMOO: _evaluate CALLED with x_array: {x_array} ---")
        self.evaluation_counter += 1
        current_params_X = self.base_sim_params.copy()

        alpha_r_val = x_array[0]
        beta_p_val = x_array[1]
        q_rounds_val = int(round(x_array[2]))
        omega_m_val = x_array[3] # 提取 omega_m_update

        current_params_X["alpha_reward"] = alpha_r_val
        current_params_X["beta_penalty_base"] = beta_p_val
        current_params_X["q_rounds_rep_change"] = q_rounds_val
        current_params_X["omega_m_update"] = omega_m_val # 设置 omega_m_update

        print(f"--- [OPTIMIZER EVAL START] Eval #{self.evaluation_counter} ---")
        print(f"Params for this eval: alpha_R={alpha_r_val:.3f}, beta_P={beta_p_val:.3f}, q_rounds={q_rounds_val}, omega_m={omega_m_val:.3f}")

        objectives = [float('inf'), 1.0]
        constraints_violation_values = [1.0e9, 1.0] # [性能约束违反, 搭便车者剔除约束违反]
        other_metrics = {"PFM_final": 0.0, "error": "Init", "client_reputation_history": {}}

        current_eval_data = {
            "params_array": x_array.tolist(),
            "params_dict": {
                "alpha_reward": alpha_r_val, "beta_penalty_base": beta_p_val,
                "q_rounds_rep_change": q_rounds_val, "omega_m_update": omega_m_val
            },
            "objectives": objectives, "constraints_violation": constraints_violation_values,
            "other_metrics": other_metrics
        }

        sim_instance_created_successfully = False
        try:
            sim = Simulation(params_X=current_params_X)
            sim_instance_created_successfully = True
            objectives, constraints_violation_values, other_metrics = sim.evaluate_parameters_for_optimization()
            current_eval_data["objectives"] = objectives
            current_eval_data["constraints_violation"] = constraints_violation_values
            current_eval_data["other_metrics"] = other_metrics
            
            # 保存每次评估的结果
            eval_filename = f"eval_results/eval_{self.evaluation_counter}_params_aR{alpha_r_val:.2f}_bP{beta_p_val:.2f}_q{q_rounds_val}_oM{omega_m_val:.2f}.json"
            os.makedirs(os.path.dirname(eval_filename), exist_ok=True) # 确保目录存在
            try:
                sim.save_simulation_stats(eval_filename)
                current_eval_data["evaluation_filename"] = eval_filename
            except Exception as e_save:
                print(f"保存评估 #{self.evaluation_counter} 结果到 {eval_filename} 时发生错误: {e_save}")

        except Exception as e:
            print(f"评估参数组合时发生错误: {type(e).__name__} - {e}")
            traceback.print_exc()
            current_eval_data["other_metrics"]["error"] = f"Sim eval failed: {str(e)}"
            # objectives 和 constraints_violation_values 保持默认的失败值

        optimization_evaluation_history.append(current_eval_data)

        out["F"] = np.array(objectives, dtype=np.double)
        cv_values_for_pymoo = []
        for cv_val in constraints_violation_values:
            if not np.isfinite(cv_val):
                cv_values_for_pymoo.append(1.0e9)
            else:
                cv_values_for_pymoo.append(cv_val)
        out["G"] = np.array(cv_values_for_pymoo, dtype=np.double)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

BASE_SIMULATION_PARAMS = {
    "N": 10, "N_f": 3, "T_max": 100,
    "iid_data": True, "non_iid_alpha": 0.5,
    "initial_M_t": 5,
    "initial_reputation": 10.0,
    "reputation_threshold": 0.01,
    "min_performance_constraint": 0.90,
    "target_accuracy_threshold": 0.90,
    "num_honest_rounds_for_fr_estimation": 2,
    "adv_attack_c_param_fr": 0.5,
    "adv_attack_noise_dim_fraction_fr": 1.0,
    "adv_attack_scaled_delta_noise_std": 0.001,
    "bid_gamma_honest": 1.0,
    "local_epochs_honest": 1,
    "lr_honest": 0.005,
    "batch_size_honest": 256,
    "adaptive_bid_adjustment_intensity_gamma_honest": 0.15,
    "adaptive_bid_max_adjustment_delta_honest": 0.4,
    "min_commitment_scaling_factor_honest": 0.2,
    "verbose": False,
    "PymooOpt": True
}

if __name__ == "__main__":
    master_seed = 42
    set_random_seed(master_seed)
    print(f"Master seed set to {master_seed}")

    run_single_test_simulation = False # 改为False以运行优化
    if run_single_test_simulation:
        print("\n--- 运行单次模拟测试 ---")
        test_params_X = BASE_SIMULATION_PARAMS.copy()
        test_params_X["alpha_reward"] = 2.0
        test_params_X["beta_penalty_base"] = 1.1
        test_params_X["q_rounds_rep_change"] = 5
        test_params_X["omega_m_update"] = 0.4
        test_params_X["T_max"] = 50
        test_params_X["verbose"] = True
        test_params_X["PymooOpt"] = False

        sim_test = Simulation(params_X=test_params_X)
        objectives, constraints, other_metrics = sim_test.evaluate_parameters_for_optimization()

        print(f"\n--- 单次模拟评估结果 ---")
        print(f"测试参数: alpha_reward={test_params_X['alpha_reward']}, beta_penalty_base={test_params_X['beta_penalty_base']}, q_rounds_rep_change={test_params_X['q_rounds_rep_change']}, omega_m_update={test_params_X['omega_m_update']}")
        obj1_val_test = objectives[0] if objectives[0] != float('inf') else "Infinite/Not Achieved"
        print(f"优化目标 (RewardsObtainedByFR_at_Elimination, FPR_final): ({obj1_val_test}, {objectives[1]:.4f})")
        print(f"性能约束违反: {constraints[0]:.4f} ({'达标' if constraints[0] <= 1e-5 else '未达标'})")
        print(f"FR剔除约束违反: {constraints[1]:.4f} ({'已剔除所有FR' if constraints[1] <= 1e-5 else '未剔除所有FR'})")
        print(f"其他指标: {other_metrics}")
        single_run_filename = "single_simulation_test_results.json"
        sim_test.save_simulation_stats(single_run_filename)
        print(f"单次模拟测试结果已保存到: {single_run_filename}")
        print("--- 单次模拟测试结束 ---\n")
    else: # 运行帕累托优化
        print("\n--- 运行帕累托优化 ---")
        actual_base_params_for_opt = BASE_SIMULATION_PARAMS.copy()
        actual_base_params_for_opt["PymooOpt"] = True
        actual_base_params_for_opt["verbose"] = True

        problem = ParetoOptimizationProblem(base_sim_params=actual_base_params_for_opt)
        algorithm = NSGA2(
            pop_size=10, # 种群大小
            crossover = SBX(prob=0.9, eta=15),
            mutation = PM(eta=20),
            eliminate_duplicates=True
        )
        generations_count = 5 # 实际应增加
        termination = get_termination("n_gen", generations_count)

        print(f"开始优化 NSGA-II: pop_size={algorithm.pop_size}, generations_count={generations_count}")
        start_time = time.time()
        res = None
        try:
            res = minimize(problem, algorithm, termination, seed=master_seed, verbose=True, save_history=False)
        except Exception as e_minimize:
            print(f"!!!!!!!!!! Pymoo minimize() 调用时发生异常 !!!!!!!!!!")
            print(f"异常类型: {type(e_minimize)}")
            print(f"异常信息: {e_minimize}")
            traceback.print_exc()
        end_time = time.time()
        optimization_duration_minutes = (end_time - start_time) / 60
        print(f"优化完成时间: {optimization_duration_minutes:.2f} 分钟。")

        if res is not None:
            print("\n--- 帕累托优化结果 ---")
            if res.X is not None and res.F is not None and len(res.X) > 0:
                print(f"在帕累托前沿上找到 {len(res.X)} 个解。")
                pareto_solutions_output = []
                for i in range(len(res.X)):
                    solution_params_array = res.X[i]
                    solution_objectives = res.F[i]
                    matched_eval_data = None
                    for eval_data_item in optimization_evaluation_history:
                        if np.allclose(np.array(eval_data_item["params_array"]), solution_params_array, atol=1e-6):
                            matched_eval_data = eval_data_item
                            break
                    current_solution_output = {
                        "solution_index": i + 1,
                        "parameters": {
                            problem.variable_names[0]: solution_params_array[0],
                            problem.variable_names[1]: solution_params_array[1],
                            problem.variable_names[2]: int(round(solution_params_array[2])),
                            problem.variable_names[3]: solution_params_array[3] # omega_m_update
                        },
                        "objectives": {
                            "RewardsObtainedByFR_at_Elimination": solution_objectives[0] if solution_objectives[0] != float('inf') else "Infinite/NotAchieved",
                            "FPR_final": solution_objectives[1]
                        }
                    }
                    if hasattr(res.G, '__len__') and i < len(res.G) and res.G[i] is not None and len(res.G[i]) == 2:
                         current_solution_output["constraint_violations"] = res.G[i].tolist()
                    if matched_eval_data and matched_eval_data.get("other_metrics"):
                        om = matched_eval_data["other_metrics"]
                        current_solution_output["termination_round_sim"] = om.get("T_term")
                        current_solution_output["final_model_accuracy_sim"] = om.get("PFM_final")
                        current_solution_output["client_reputation_history_per_round"] = om.get("client_reputation_history", {})
                        # 标记原始评估文件为帕累托最优
                        if "evaluation_filename" in matched_eval_data:
                            try:
                                eval_filename_to_mark = matched_eval_data["evaluation_filename"]
                                with open(eval_filename_to_mark, 'r', encoding='utf-8') as f_mark:
                                    eval_data_to_mark = json.load(f_mark)
                                eval_data_to_mark["simulation_summary"]["is_pareto_optimal"] = True
                                eval_data_to_mark["simulation_summary"]["pareto_front_index"] = i + 1
                                with open(eval_filename_to_mark, 'w', encoding='utf-8') as f_mark:
                                    json.dump(eval_data_to_mark, f_mark, indent=4, ensure_ascii=False, default=str)
                                print(f"已将评估 {eval_filename_to_mark} 标记为帕累托最优解 #{i+1}")
                            except Exception as e_mark:
                                print(f"标记帕累托最优解 #{i+1} ({eval_filename_to_mark}) 时发生错误: {e_mark}")
                    else:
                        current_solution_output["client_reputation_history_per_round"] = "Not Found in History"
                        print(f"警告: 未能在历史记录中找到解 {i+1} (参数: {solution_params_array}) 的详细评估数据。")
                    pareto_solutions_output.append(current_solution_output)
                    print(f"\n解 {i+1}:")
                    print(f"  参数: alpha_R={current_solution_output['parameters']['alpha_reward']:.3f}, "
                          f"beta_P={current_solution_output['parameters']['beta_penalty_base']:.3f}, "
                          f"q_rounds={current_solution_output['parameters']['q_rounds_rep_change']}, "
                          f"omega_m={current_solution_output['parameters']['omega_m_update']:.3f}")
                    print(f"  目标: RewardsObtainedByFR_at_Elimination={current_solution_output['objectives']['RewardsObtainedByFR_at_Elimination']}, "
                          f"FPR_final={current_solution_output['objectives']['FPR_final']:.4f}")
                    if "constraint_violations" in current_solution_output:
                        print(f"  性能约束违反: {current_solution_output['constraint_violations'][0]:.4f}")
                        print(f"  FR剔除约束违反: {current_solution_output['constraint_violations'][1]:.4f} "
                              f"(已剔除所有FR: {'是' if current_solution_output['constraint_violations'][1] <= 1e-5 else '否'})")

                final_results_to_save = {
                    "optimization_summary": {
                        "total_solutions_on_pareto_front": len(res.X),
                        "optimization_time_minutes": optimization_duration_minutes,
                        "base_sim_params_used_for_opt": actual_base_params_for_opt,
                        "pymoo_algorithm_details": {
                            "name": algorithm.__class__.__name__,
                            "pop_size": algorithm.pop_size,
                            "termination_criterion": str(termination)
                        }
                    },
                    "pareto_solutions_details": pareto_solutions_output
                }
                opt_results_filename = "pareto_optimization_results_summary.json" # 改为 summary，因为单个文件在上面保存
                try:
                    with open(opt_results_filename, 'w', encoding='utf-8') as f:
                        json.dump(final_results_to_save, f, indent=4, ensure_ascii=False, default=str)
                    print(f"\n帕累托优化结果摘要已保存到 {opt_results_filename}")
                except Exception as e:
                    print(f"保存帕累托优化结果摘要时发生错误: {e}")
            else:
                print("优化过程未产生有效的帕累托前沿解。")
                if res.F is not None: print(f"目标值 (res.F): {res.F}")
                if res.X is not None: print(f"参数 (res.X): {res.X}")
        else:
            print("优化过程未能完成或未能生成结果对象 (res is None)。")
        print("\n--- 帕累托优化结束 ---")