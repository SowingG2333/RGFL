# main.py
import torch
import random
import numpy as np
import copy
from torch.utils.data import DataLoader
import torch.nn.functional as F
import json
import time

# --- Pymoo Imports ---
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
# from pymoo.core.variable import Real, Integer # Not needed when using xl, xu directly

# --- Your existing imports for simulation components ---
from system import get_mnist_data, Global_Model, Requester
from honest_client import HonestClient
from free_rider import FreeRider

class Simulation:
    def __init__(self, params_X):
        print("--- SIMULATION: __init__ CALLED ---") # 您确认这个被打印了
        print(f"Initial params_X for Simulation: {params_X}")
        self.params_X = params_X
        self.num_total_participants = params_X.get("N", 10)
        self.num_free_riders = params_X.get("N_f", 3)
        self.num_honest_clients = self.num_total_participants - self.num_free_riders
        
        self.verbose = params_X.get("verbose", True) # Control verbosity
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        self.max_rounds = params_X.get("T_max", 20)
        self.target_accuracy_threshold = params_X.get("target_accuracy_threshold", None)
        self.initial_M_t = params_X.get("initial_M_t", self.num_total_participants // 2)
        self.iid_data_distribution = params_X.get("iid_data", True)
        self.non_iid_alpha = params_X.get("non_iid_alpha", 0.3)

        self.alpha_reward = params_X.get("alpha_reward")
        self.beta_penalty_base = params_X.get("beta_penalty_base")
        self.omega_m_update = params_X.get("omega_m_update")
        self.q_rounds_rep_change = params_X.get("q_rounds_rep_change")
        self.initial_reputation = params_X.get("initial_reputation")
        self.reputation_threshold = params_X.get("reputation_threshold")
        self.min_performance_constraint = params_X.get("min_performance_constraint")

        self.num_honest_rounds_for_fr_estimation = params_X.get("num_honest_rounds_for_fr_estimation", 2)
        self.adv_attack_c_param_fr = params_X.get("adv_attack_c_param_fr")
        self.adv_attack_noise_dim_fraction_fr = params_X.get("adv_attack_noise_dim_fraction_fr")
        self.adv_attack_scaled_delta_noise_std_fr = params_X.get("adv_attack_scaled_delta_noise_std", 0.0001)

        self.adaptive_bid_adjustment_intensity_gamma_honest = params_X.get("adaptive_bid_adjustment_intensity_gamma_honest")
        self.adaptive_bid_max_adjustment_delta_honest = params_X.get("adaptive_bid_max_adjustment_delta_honest")
        self.min_commitment_scaling_factor_honest = params_X.get("min_commitment_scaling_factor_honest")

        self.bid_gamma_honest = params_X.get("bid_gamma_honest", 0.8)
        self.local_epochs_honest = params_X.get("local_epochs_honest", 2)
        self.lr_honest = params_X.get("lr_honest", 0.005) # Consider fixing for optimization consistency
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
        self.client_reputation_history = {}
        self.client_types = {}
        self.all_fr_eliminated_logged = False # Tracks if the "all FR eliminated" message was logged

        self.rewards_at_all_fr_eliminated = float('inf')
        self.round_at_all_fr_eliminated = -1
        self.all_fr_elimination_achieved_flag = False # Tracks if the state was achieved

        self.simulation_stats = {
            "round_number": [], "model_accuracy": [], "active_honest_clients": [],
            "active_free_riders": [], "M_t_value": [], "cumulative_total_incentive_cost": [],
            "cumulative_real_incentive_cost": [], "cumulative_tir_history": []
        }

    def log_message(self, message):
        if self.verbose:
            print(message)

    def initialize_environment(self):
        self.log_message("初始化环境中...")
        initial_global_model = Global_Model() # Assuming Global_Model is defined in system.py
        effective_num_honest_clients = max(0, self.num_honest_clients)

        client_datasets, test_dataset_global = get_mnist_data(
            effective_num_honest_clients, self.iid_data_distribution, self.non_iid_alpha
        )
        if effective_num_honest_clients > 0 and (not client_datasets or effective_num_honest_clients > len(client_datasets)):
            self.log_message(f"警告: 请求 {effective_num_honest_clients} 个诚实客户端数据，实际分配 {len(client_datasets)} 个。")
            if not client_datasets and effective_num_honest_clients > 0:
                 raise ValueError(f"诚实客户端数据分配失败: 请求 {effective_num_honest_clients}, 得到 {len(client_datasets)}")

        pin_memory_flag = self.device.type != "cpu" # Simpler check
        num_workers_val = 0 # Set to 0 to potentially speed up init during optimization
        test_loader = DataLoader(test_dataset_global, batch_size=128, shuffle=False,
                                 pin_memory=pin_memory_flag, num_workers=num_workers_val)

        self.requester = Requester(initial_global_model, test_loader, self.device,
                             alpha_reward=self.alpha_reward,
                             beta_penalty_base=self.beta_penalty_base)
        self.participants = []

        temp_participants = []
        for i in range(effective_num_honest_clients):
            if i >= len(client_datasets) or not client_datasets[i] or len(client_datasets[i]) == 0:
                self.log_message(f"警告：诚实客户端 {i} 数据不足或无效，跳过。")
                continue
            temp_participants.append(HonestClient(
                id=f"h_{i}", init_rep=self.initial_reputation,
                tra_round_num=self.q_rounds_rep_change,
                device=self.device, client_dataset=client_datasets[i], train_size=0.8,
                init_commit_scaling_factor=self.bid_gamma_honest,
                batch_size=self.batch_size_honest, local_epochs=self.local_epochs_honest, 
                lr=self.lr_honest, # Using the fixed lr_honest from params
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

        if not temp_participants and (effective_num_honest_clients > 0 or self.num_free_riders > 0) :
            raise ValueError("没有参与者被初始化，尽管请求了参与者。检查数据分配和客户端初始化。")

        # Consider if shuffling is needed or fixed order for strict reproducibility during optimization
        # For now, keeping the shuffle. If issues arise, one might fix participant order.
        random.shuffle(temp_participants)
        self.participants = temp_participants

        for p in self.participants:
            self.client_types[p.id] = p.type
            # Initialize history lists if needed by other parts of your code, not strictly for Sim class logic shown
            # self.client_l2_norm_history[p.id] = []
            # self.client_cosine_similarity_history[p.id] = []
            # self.client_reputation_history[p.id] = []


        # Reset simulation state variables
        self.current_round = 0
        self.M_t = min(self.initial_M_t, len(self.participants)) if self.participants else 0
        self.total_rewards_paid = 0.0
        self.rewards_paid_to_honest_clients = 0.0
        self.all_fr_eliminated_logged = False
        self.rewards_at_all_fr_eliminated = float('inf')
        self.round_at_all_fr_eliminated = -1
        self.all_fr_elimination_achieved_flag = False
        
        # Reset stats lists
        for key in self.simulation_stats: self.simulation_stats[key] = []


        if self.requester:
            self.final_global_model_performance, _ = self.requester.evaluate_global_model()
            if self.requester.global_model: # Ensure global model exists
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
        else:
            self.final_global_model_performance = 0.0
            self.log_message("警告：服务器未正确初始化，无法评估初始模型或记录初始统计。")
            # Populate stats for consistency even if init fails partway
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
            # Original logic: j > 1 (means at least 3 elements to find a gap after the first two)
            # If you want to consider gaps from the very first pair, change j > 1 to j >= 0 or similar
            for j in range(len(sorted_delta_r) - 1):
                gap = sorted_delta_r[j] - sorted_delta_r[j+1]
                if gap > max_gap and j > 1: # Original condition for finding k_prime_t
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


    def _flatten_gradient_dict(self, gradient_dict): # Not used in current optimization path logic but kept
        if gradient_dict is None or not isinstance(gradient_dict, dict): return None
        try:
            flat_parts = [p.detach().view(-1).cpu() for p in gradient_dict.values() if isinstance(p, torch.Tensor)]
            return torch.cat(flat_parts) if flat_parts else None
        except Exception as e:
            self.log_message(f"Error flattening gradient dict: {e}")
            return None

    def run_one_round(self):
        self.current_round += 1
        m_t_for_this_round = self.M_t
        self.log_message(f"\n--- 第 {self.current_round}/{self.max_rounds} 轮 (M_t = {m_t_for_this_round}) ---")

        if not self.requester or not self.participants:
            self.log_message("请求者或参与者未初始化。结束本轮。")
            # Populate stats for consistency
            self.simulation_stats["round_number"].append(self.current_round)
            self.simulation_stats["model_accuracy"].append(self.final_global_model_performance)
            self.simulation_stats["active_honest_clients"].append(0)
            self.simulation_stats["active_free_riders"].append(0)
            self.simulation_stats["M_t_value"].append(m_t_for_this_round)
            self.simulation_stats["cumulative_total_incentive_cost"].append(self.total_rewards_paid)
            self.simulation_stats["cumulative_real_incentive_cost"].append(self.rewards_paid_to_honest_clients)
            current_tir = (self.rewards_paid_to_honest_clients / self.total_rewards_paid) if self.total_rewards_paid > 1e-9 else 0.0
            self.simulation_stats["cumulative_tir_history"].append(current_tir)
            return True # Terminate

        round_start_global_state = copy.deepcopy(self.requester.global_model.state_dict())
        round_start_global_accuracy, _ = self.requester.evaluate_global_model()

        client_updates_for_submission = {}
        for p in self.participants:
            p.set_model_state(copy.deepcopy(round_start_global_state))
            if p.type == "honest_client":
                p.perf_before_local_train, _ = p.evaluate_model(on_val_set=True)
                p.local_train()
                p.gen_true_update(round_start_global_state)
            else: # FreeRider
                p.gen_fabric_update(self.current_round, self.requester.global_model_param_diff_history,
                                    round_start_global_state, len(self.participants))
            if p.current_update:
                client_updates_for_submission[p.id] = copy.deepcopy(p.current_update)
        
        # Bidding logic
        honest_bids_promises, honest_bids_rewards, honest_bids_ratios = [], [], []
        num_bidding_honest_clients = 0
        for p_bid in self.participants:
            if p_bid.reputation < self.reputation_threshold:
                p_bid.bid = {} # Clear bid if not eligible
                continue
            if p_bid.type == "honest_client":
                bid_data = p_bid.submit_bid() # HonestClient's submit_bid doesn't take args
                if bid_data and 'promise' in bid_data and 'reward' in bid_data and bid_data['reward'] > 1e-6:
                    honest_bids_promises.append(bid_data['promise'])
                    honest_bids_rewards.append(bid_data['reward'])
                    honest_bids_ratios.append(bid_data['promise'] / bid_data['reward'])
                    num_bidding_honest_clients +=1
        
        # Free rider bidding based on honest client bids
        if num_bidding_honest_clients > 0 :
            highest_honest_effectiveness = max(honest_bids_ratios) if honest_bids_ratios else 0
            lowest_honest_effectiveness = min(honest_bids_ratios) if honest_bids_ratios else 0 # Avoid error if list empty
            lowest_honest_promise = min(honest_bids_promises) if honest_bids_promises else 0
            avg_honest_promise = np.mean(honest_bids_promises) if honest_bids_promises else 0
            for p_fr_bid in self.participants:
                if p_fr_bid.reputation >= self.reputation_threshold and p_fr_bid.type == "free_rider":
                    p_fr_bid.submit_bid(highest_honest_effectiveness, lowest_honest_effectiveness, lowest_honest_promise, avg_honest_promise)
        else: # No honest bidders, FRs bid with default values
            for p_fr_bid_default in self.participants:
                if p_fr_bid_default.reputation >= self.reputation_threshold and p_fr_bid_default.type == "free_rider":
                    p_fr_bid_default.submit_bid(0,0,0,0) # FR's submit_bid with default/fallback values

        selected_participants = self.requester.select_participants(self.participants, m_t_for_this_round, self.reputation_threshold)

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
                rewards_paid_ref = [self.total_rewards_paid] # Pass as a list to be mutable
                self.requester.update_reputations_and_pay(self.participants, verification_outcomes, rewards_paid_ref)
                self.total_rewards_paid = rewards_paid_ref[0]

                current_round_rewards_to_honest_clients_this_round = 0
                for outcome in verification_outcomes:
                    if outcome["successful_verification"]:
                        p_find = next((p for p in self.participants if p.id == outcome["participant_id"]), None)
                        if p_find and self.client_types.get(p_find.id) == "honest_client":
                            current_round_rewards_to_honest_clients_this_round += p_find.bid.get('reward', 0)
                self.rewards_paid_to_honest_clients += current_round_rewards_to_honest_clients_this_round
                
                if self.verbose: # Only print details if verbose
                    for outcome in verification_outcomes:
                        par = next((p_find for p_find in self.participants if p_find.id == outcome["participant_id"]), None)
                        if par:
                            status_str = "成功" if outcome["successful_verification"] else "失败"
                            self.log_message(f"  - {par.id} ({self.client_types[par.id]}), 声誉: {par.reputation:.2f}, "
                                  f"投标: P={par.bid.get('promise',0):.3f}/R={par.bid.get('reward',0):.2f}, "
                                  f"观察提升: {outcome.get('observed_increase',0):.3f}, 状态: {status_str}")
        
        for p_every in self.participants: p_every.update_reputation_history()
        # Storing main-level reputation history (optional, if needed for global plots)
        # for p_track in self.participants:
        #     if p_track.id not in self.client_reputation_history: self.client_reputation_history[p_track.id] = []
        #     self.client_reputation_history[p_track.id].append(p_track.reputation)


        self.requester.update_global_model_history()
        self.final_global_model_performance, _ = self.requester.evaluate_global_model()
        self.update_M_t() # Update M_t for the NEXT round

        num_active_honest = sum(1 for p in self.participants if self.client_types.get(p.id) == "honest_client" and p.reputation >= self.reputation_threshold)
        num_active_free_riders = sum(1 for p in self.participants if self.client_types.get(p.id) == "free_rider" and p.reputation >= self.reputation_threshold)

        # Check for C_elim_fr
        if self.num_free_riders > 0 and not self.all_fr_elimination_achieved_flag and num_active_free_riders == 0:
            self.rewards_at_all_fr_eliminated = self.total_rewards_paid
            self.round_at_all_fr_eliminated = self.current_round
            self.all_fr_elimination_achieved_flag = True
            self.log_message(f"*** 所有 ({self.num_free_riders}) 个搭便车者已在第 {self.current_round} 轮被剔除。 ***")
            self.log_message(f"*** 剔除时累计总奖励开销: {self.rewards_at_all_fr_eliminated:.2f} ***")

        current_cumulative_tir = (self.rewards_paid_to_honest_clients / self.total_rewards_paid) if self.total_rewards_paid > 1e-9 else 0.0
        self.log_message(f"第 {self.current_round} 轮结束: 活跃诚实={num_active_honest}, 活跃FR={num_active_free_riders}, "
              f"总奖励(累计)={self.total_rewards_paid:.2f}, 支付给诚实客户端奖励(累计)={self.rewards_paid_to_honest_clients:.2f}, "
              f"全局性能={self.final_global_model_performance:.4f}, 真实激励率(累计): {current_cumulative_tir:.4f}, "
              f"下一轮 M_t 将是: {self.M_t}")

        # Record stats for this round
        self.simulation_stats["round_number"].append(self.current_round)
        self.simulation_stats["model_accuracy"].append(self.final_global_model_performance)
        self.simulation_stats["active_honest_clients"].append(num_active_honest)
        self.simulation_stats["active_free_riders"].append(num_active_free_riders)
        self.simulation_stats["M_t_value"].append(m_t_for_this_round) # M_t used in this round
        self.simulation_stats["cumulative_total_incentive_cost"].append(self.total_rewards_paid)
        self.simulation_stats["cumulative_real_incentive_cost"].append(self.rewards_paid_to_honest_clients)
        self.simulation_stats["cumulative_tir_history"].append(current_cumulative_tir)
        return self.check_termination_condition()

    def check_termination_condition(self):
        if not self.participants:
            self.log_message("没有参与者，终止模拟。")
            self.termination_round = self.current_round
            return True
        if self.target_accuracy_threshold is not None and self.final_global_model_performance >= self.target_accuracy_threshold:
            self.log_message(f"目标模型性能 {self.target_accuracy_threshold:.4f} 已达到。终止。")
            self.termination_round = self.current_round
            return True
        if self.current_round >= self.max_rounds:
            self.log_message(f"已达到最大模拟轮数 ({self.max_rounds})。终止。")
            self.termination_round = self.max_rounds # Ensure it's max_rounds if terminated here
            return True
        
        # Early termination if no honest clients are active (and there should be some)
        if self.num_honest_clients > 0 and self.current_round > 0: # Avoid check at round 0
            num_active_honest_val = sum(1 for p in self.participants if self.client_types.get(p.id) == "honest_client" and p.reputation >= self.reputation_threshold)
            if num_active_honest_val == 0:
                self.log_message("没有活跃的诚实客户端了。模拟可能卡住或所有诚实客户端被错误惩罚。终止模拟。")
                self.termination_round = self.current_round
                return True
        
        # Early termination if no participants are active at all
        if self.current_round > 0: # Avoid check at round 0
            num_active_total_final_check_val = sum(1 for p in self.participants if p.reputation >= self.reputation_threshold)
            if num_active_total_final_check_val == 0:
                self.log_message("没有活跃的参与者了。终止模拟。")
                self.termination_round = self.current_round
                return True
        return False # Continue simulation

    def save_simulation_stats(self, filename="simulation_results.json"): # Not called during optimization to save disk space
        if not self.simulation_stats["round_number"]:
            self.log_message("没有统计数据可供保存。")
            return

        records = []
        num_recorded_rounds = len(self.simulation_stats["round_number"])
        stat_keys = list(self.simulation_stats.keys())
        for i in range(num_recorded_rounds):
            record = {key: self.simulation_stats[key][i] if i < len(self.simulation_stats[key]) else None for key in stat_keys}
            records.append(record)

        final_fpr = 0.0
        if self.num_honest_clients > 0 and self.participants: # Ensure participants list exists
            num_honest_eliminated = sum(1 for p in self.participants if self.client_types.get(p.id) == "honest_client" and p.reputation < self.reputation_threshold)
            final_fpr = num_honest_eliminated / self.num_honest_clients
        elif self.num_honest_clients == 0 : final_fpr = 0.0

        final_tir = (self.rewards_paid_to_honest_clients / self.total_rewards_paid) if self.total_rewards_paid > 1e-9 else 0.0
        
        # Prepare final data structure
        data_to_save = {
            "simulation_parameters": self.params_X,
            "simulation_summary": {
                "termination_round": self.termination_round,
                "total_incentive_cost_final": self.total_rewards_paid,
                "real_incentive_cost_final": self.rewards_paid_to_honest_clients,
                "false_positive_rate_final": final_fpr,
                "global_model_accuracy_final": self.final_global_model_performance,
                "true_incentive_rate_final": final_tir,
                "cost_at_all_fr_eliminated": self.rewards_at_all_fr_eliminated if self.all_fr_elimination_achieved_flag else "Not_Achieved",
                "round_at_all_fr_eliminated": self.round_at_all_fr_eliminated if self.all_fr_elimination_achieved_flag else "Not_Achieved",
                "all_fr_elimination_achieved": self.all_fr_elimination_achieved_flag
            },
            "per_round_statistics": records # Save per-round details
        }
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                # Use a robust serializer for potentially non-standard types (like numpy floats/ints if any slip through)
                json.dump(data_to_save, f, indent=4, ensure_ascii=False, default=lambda o: str(o) if isinstance(o, (np.integer, np.floating)) else o)
            self.log_message(f"模拟统计数据已成功保存到 {filename}")
        except Exception as e: # Catch broader exceptions during file write
            self.log_message(f"保存统计数据到文件 {filename} 时发生错误: {e}")


    def run_simulation(self):
        try:
            self.initialize_environment()
        except ValueError as e:
            self.log_message(f"模拟初始化失败: {e}")
            self.termination_round = 0
            # Save failure details if verbose mode is on (not during optimization)
            if self.verbose:
                filename = f"sim_results_INIT_FAIL_N{self.num_total_participants}_Nf{self.num_free_riders}.json"
                try: self.save_simulation_stats(filename)
                except: pass # Avoid crashing if save itself fails
            return [float('inf'), 1.0], [1.0], {} # objectives, constraints, other_metrics

        if not self.requester or (not self.participants and (self.num_honest_clients > 0 or self.num_free_riders > 0)) :
             self.log_message("模拟无法开始，请求者或参与者未能正确初始化。")
             self.termination_round = 0
             if self.verbose:
                filename = f"sim_results_START_FAIL_N{self.num_total_participants}_Nf{self.num_free_riders}.json"
                try: self.save_simulation_stats(filename)
                except: pass
             return [float('inf'), 1.0], [1.0], {}

        terminated = False
        while not terminated:
            terminated = self.run_one_round()
            # Early termination for very poor performance during optimization
            if not terminated and \
               self.min_performance_constraint > 0 and \
               self.final_global_model_performance < self.min_performance_constraint * 0.25 and \
               self.current_round > min(10, self.max_rounds / 3) and \
               self.max_rounds > 10 : # Conditions for early stop due to low performance
                self.log_message(f"全局模型性能 ({self.final_global_model_performance:.4f}) 过低 (远低于约束 {self.min_performance_constraint*0.25:.4f})，提前终止。")
                if self.termination_round == self.max_rounds: # Only update if not already terminated by another condition
                    self.termination_round = self.current_round
                terminated = True
            
            if terminated: break # Exit while loop

        # --- Post-simulation calculations ---
        T_term = self.termination_round
        C_total_final = self.total_rewards_paid

        num_honest_clients_at_start = max(1, self.num_honest_clients) # Avoid division by zero
        num_honest_eliminated = 0
        if self.participants: # Ensure participants list exists
            num_honest_eliminated = sum(1 for p in self.participants if self.client_types.get(p.id) == "honest_client" and p.reputation < self.reputation_threshold)
        
        FPR_final = num_honest_eliminated / num_honest_clients_at_start if num_honest_clients_at_start > 0 else 0.0
        if self.num_honest_clients == 0: # If there were no honest clients to begin with
            FPR_final = 0.0

        PFM_final = self.final_global_model_performance
        true_incentive_rate_final = (self.rewards_paid_to_honest_clients / C_total_final) if C_total_final > 1e-9 else 0.0
        
        cost_when_all_fr_eliminated = self.rewards_at_all_fr_eliminated
        # If there were FRs but not all eliminated, penalize with infinity
        if not self.all_fr_elimination_achieved_flag and self.num_free_riders > 0:
            cost_when_all_fr_eliminated = float('inf')
            self.log_message(f"警告: 模拟结束时仍有活跃的搭便车者。剔除所有搭便车者时的开销将设为极大值。")
        elif self.num_free_riders == 0: # No FRs to begin with
            cost_when_all_fr_eliminated = 0.0 # Cost to eliminate zero FRs is zero


        self.log_message(f"\n--- 模拟结束 (run_simulation) ---")
        self.log_message(f"T_term (终止轮数): {T_term}")
        self.log_message(f"C_total_final (最终总奖励开销): {C_total_final:.2f}")
        self.log_message(f"FPR_final (最终诚实客户端误判率): {FPR_final:.4f} ({num_honest_eliminated}/{self.num_honest_clients if self.num_honest_clients > 0 else 'N/A'})")
        self.log_message(f"PFM_final (最终全局模型准确率): {PFM_final:.4f}")
        self.log_message(f"TIR_final (最终真实激励率): {true_incentive_rate_final:.4f}")
        self.log_message(f"C_elim_fr (剔除所有FR时的奖励开销): {cost_when_all_fr_eliminated if cost_when_all_fr_eliminated != float('inf') else 'Infinite/Not Achieved'}")
        self.log_message(f"Round_elim_fr (剔除所有FR时的轮数): {self.round_at_all_fr_eliminated if self.all_fr_elimination_achieved_flag else 'N/A'}")

        # Save full stats only if verbose (i.e., for single runs, not during optimization bulk runs)
        if self.verbose:
            filename = f"sim_N{self.num_total_participants}_Nf{self.num_free_riders}_alphaR{self.alpha_reward:.2f}_betaP{self.beta_penalty_base:.2f}_q{self.q_rounds_rep_change}.json"
            try: self.save_simulation_stats(filename)
            except: pass # Avoid crashing if save itself fails


        objectives_for_pareto = [cost_when_all_fr_eliminated, FPR_final]
        # Constraint: PFM_final must be >= min_performance_constraint
        # pymoo expects g(x) <= 0 for a satisfied constraint.
        # So, constraint_violation = min_performance_constraint - PFM_final
        # If PFM_final < min_performance_constraint, violation > 0 (constraint NOT met)
        # If PFM_final >= min_performance_constraint, violation <= 0 (constraint met)
        constraint_violation = self.min_performance_constraint - PFM_final
        
        other_metrics = { # For potential logging or deeper analysis if needed
            "T_term": T_term,
            "PFM_final": PFM_final,
            "TIR_final": true_incentive_rate_final,
            "C_total_final": C_total_final,
            # FPR_final is already an objective
            # cost_at_all_fr_eliminated_metric is already an objective
            "round_at_all_fr_eliminated": self.round_at_all_fr_eliminated if self.all_fr_elimination_achieved_flag else -1,
            "all_fr_elimination_achieved_flag": self.all_fr_elimination_achieved_flag,
            "constraint_PFM_violation": constraint_violation # Store how much constraint was violated/satisfied
        }
        return objectives_for_pareto, [constraint_violation], other_metrics


    def evaluate_parameters_for_optimization(self):
        # This method is specifically for the optimizer. It runs the simulation and returns objectives & constraints.
        return self.run_simulation()


# --- Global Base Configuration for Simulation ---
BASE_SIMULATION_PARAMS = {
    "N": 10, "N_f": 3, "T_max": 100, # Reduced T_max for faster optimization demo
    "iid_data": True, "non_iid_alpha": 0.5,
    # alpha_reward, beta_penalty_base, q_rounds_rep_change will be set by optimizer
    "omega_m_update": 0.4,
    "initial_reputation": 10.0,
    "reputation_threshold": 0.01, # Important threshold
    "min_performance_constraint": 0.90, # Minimum performance constraint for Pareto optimization
    "target_accuracy_threshold": 0.90, # Usually None for Pareto optimization of other metrics

    "num_honest_rounds_for_fr_estimation": 2,
    "adv_attack_c_param_fr": 0.5,
    "adv_attack_noise_dim_fraction_fr": 1.0,
    "adv_attack_scaled_delta_noise_std": 0.001,

    "bid_gamma_honest": 1.0,
    "local_epochs_honest": 1, # Kept low for speed
    "lr_honest": 0.001, # Fixed LR for honest clients
    "batch_size_honest": 64,

    "adaptive_bid_adjustment_intensity_gamma_honest": 0.15,
    "adaptive_bid_max_adjustment_delta_honest": 0.4,
    "min_commitment_scaling_factor_honest": 0.2,
    "verbose": False # IMPORTANT: Set to False to suppress detailed print output during optimization
}


# --- Pymoo Problem Definition ---
class ParetoOptimizationProblem(Problem):
    def __init__(self, base_sim_params):
        self.base_sim_params = base_sim_params
        self.evaluation_counter = 0 # To track number of evaluations
        
        # Define variable names for clarity when saving results
        self.variable_names = ["alpha_reward", "beta_penalty_base", "q_rounds_rep_change"]
        
        # Lower bounds for each variable (order must match variable_names)
        xl = np.array([
            1.0,    # alpha_reward lower bound
            1.1,   # beta_penalty_base lower bound (must be > 1)
            3       # q_rounds_rep_change lower bound (integer)
        ])
        
        # Upper bounds for each variable
        xu = np.array([
            10.0,    # alpha_reward upper bound
            10.0,    # beta_penalty_base upper bound
            10      # q_rounds_rep_change upper bound (integer)
        ])

        super().__init__(n_var=3,     # Number of variables
                         n_obj=2,     # Number of objectives: C_elim_fr, FPR_final
                         n_constr=1,  # Number of constraints: PFM_final >= min_performance_constraint
                         xl=xl,       # Lower bounds array
                         xu=xu,       # Upper bounds array
                         elementwise=True) # Evaluate one solution at a time

    def _evaluate(self, x_array, out, *args, **kwargs):
        print(f"--- PYMOO: ParetoOptimizationProblem._evaluate CALLED with x_array: {x_array} ---") # <--- 新增的顶级打印
        self.evaluation_counter += 1
        q_rounds_val = int(round(x_array[2]))
        print(f"--- [OPTIMIZER] Starting Evaluation #{self.evaluation_counter} with params: alpha_R={x_array[0]:.3f}, beta_P={x_array[1]:.3f}, q_rounds={q_rounds_val} ---")
        start_eval_time = time.time() # 记录评估开始时间

        current_params_X = self.base_sim_params.copy()
        current_params_X["alpha_reward"] = x_array[0]
        current_params_X["beta_penalty_base"] = x_array[1]
        current_params_X["q_rounds_rep_change"] = q_rounds_val
        current_params_X["verbose"] = True # 确保Simulation内部打印

        # 可以在这里也设置一下随机种子，确保每次Simulation运行的“内部”随机性（如果存在且未被全局种子覆盖）是独立的
        # set_random_seed(random.randint(1, 1000000) + self.evaluation_counter) 

        sim = Simulation(params_X=current_params_X)
        objectives, constraints_violation, other_metrics = sim.evaluate_parameters_for_optimization()

        out["F"] = np.array(objectives, dtype=float)
        out["G"] = np.array(constraints_violation, dtype=float)

        end_eval_time = time.time()
        print(f"--- [OPTIMIZER] Finished Evaluation #{self.evaluation_counter}. Time: {end_eval_time - start_eval_time:.2f}s. Objectives: {out['F']}, Constraints: {out['G']} ---")

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True # Can impact performance but good for reproducibility
        torch.backends.cudnn.benchmark = False   # Disable benchmark for reproducibility


if __name__ == "__main__":
    master_seed = 42 # Master seed for the entire process
    set_random_seed(master_seed)
    print(f"Master seed set to {master_seed}")

    # --- Option 1: Run a single simulation with specific parameters (for testing) ---
    run_single_test_simulation = False # Set to True to run this block for debugging
    if run_single_test_simulation:
        print("\n--- Running Single Test Simulation ---")
        test_params_X = BASE_SIMULATION_PARAMS.copy()
        # Override parameters for the test
        test_params_X["alpha_reward"] = 2.0
        test_params_X["beta_penalty_base"] = 1.1
        test_params_X["q_rounds_rep_change"] = 5
        test_params_X["T_max"] = 30 # Use a reasonable T_max for testing
        test_params_X["verbose"] = True # Enable verbose for single run

        sim_test = Simulation(params_X=test_params_X)
        # The evaluate_parameters_for_optimization method is what the optimizer calls
        objectives, constraints, other_metrics = sim_test.evaluate_parameters_for_optimization()

        print(f"\n--- 单次模拟评估结果 ---")
        print(f"测试参数: alpha_reward={test_params_X['alpha_reward']}, beta_penalty_base={test_params_X['beta_penalty_base']}, q_rounds_rep_change={test_params_X['q_rounds_rep_change']}")
        obj1_val_test = objectives[0] if objectives[0] != float('inf') else "Infinite/Not Achieved"
        print(f"优化目标 (C_elim_fr, FPR_final): ({obj1_val_test}, {objectives[1]:.4f})")
        constraint_val_test = constraints[0] # constraints is a list
        print(f"约束违反 (min_PFM - PFM_final <= 0): {constraint_val_test:.4f}")
        if constraint_val_test <= 1e-5: # Allow for small floating point inaccuracies
            print("约束：最终模型性能达标 (相对于 min_performance_constraint)。")
        else:
            print("约束：最终模型性能未达标 (相对于 min_performance_constraint)。")
        print(f"其他指标: {other_metrics}")
        print("--- End of Single Test Simulation ---\n")

    # --- Option 2: Run Pareto Optimization ---
    print("\n--- Running Pareto Optimization ---")
    problem = ParetoOptimizationProblem(base_sim_params=BASE_SIMULATION_PARAMS)

    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM

    algorithm = NSGA2(
        pop_size=10,  # Population size: Increase for real runs (e.g., 20, 50, 100)
        crossover = SBX(prob=0.9, eta=15),
        mutation = PM(eta=20),
        eliminate_duplicates=True # Good practice
    )

    # Termination condition:
    generations_count = 5 # Number of generations: Increase for real runs (e.g., 25, 50, 100)
    termination = get_termination("n_gen", generations_count)
    # Using time-based termination for a quicker demo run.
    # For serious runs, generation-based termination is often preferred for more control.
    # optimization_duration_str = "00:02:00" # Example: 2 minutes for a very short demo
    # termination = get_termination("time", optimization_duration_str)
    
    # print(f"Starting optimization with NSGA-II: pop_size={algorithm.pop_size}, termination criterion: {optimization_duration_str}")
    print(f"Starting optimization with NSGA-II: pop_size={algorithm.pop_size}, generations_count={generations_count}")
    start_time = time.time()

    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=master_seed, # Seed for reproducibility of the optimization algorithm's run
                   verbose=True,     # Pymoo's verbosity for generation progress
                   save_history=False) # Set to True to save history for later analysis if needed (can consume memory)

    end_time = time.time()
    optimization_duration_minutes = (end_time - start_time) / 60
    print(f"Optimization finished in {optimization_duration_minutes:.2f} minutes.")

    if res is not None:
        print("\n--- Pareto Optimization Results ---")
        if res.X is not None and res.F is not None and len(res.X) > 0:
            print(f"Found {len(res.X)} solution(s) on the Pareto front.")
            print("\nParameters (X) on the Pareto front:")
            # res.X is a NumPy array where each row is a solution [alpha, beta, q_rounds]
            for i, x_solution_array in enumerate(res.X):
                 print(f"Solution {i+1}: "
                       f"{problem.variable_names[0]}={x_solution_array[0]:.3f}, "
                       f"{problem.variable_names[1]}={x_solution_array[1]:.3f}, "
                       f"{problem.variable_names[2]}={int(round(x_solution_array[2]))}")


            print("\nObjective values (F) on the Pareto front (C_elim_fr, FPR_final):")
            for i, f_solution in enumerate(res.F):
                obj1_val = f_solution[0] if f_solution[0] != float('inf') else "Inf/NotAchieved"
                print(f"Solution {i+1}: C_elim_fr={obj1_val}, FPR_final={f_solution[1]:.4f}")

            # Save results to a JSON file
            results_to_save = {
                "pareto_solutions_parameters": {
                    "columns": problem.variable_names,
                    "data": res.X.tolist()
                },
                "pareto_objectives_F": res.F.tolist(),
                "optimization_time_minutes": optimization_duration_minutes,
                "base_sim_params_used_for_opt": BASE_SIMULATION_PARAMS,
                "pymoo_algorithm_details": {
                    "name": algorithm.__class__.__name__, # NSGA2
                    "pop_size": algorithm.pop_size,
                    "termination_criterion": str(termination) # String representation of termination
                }
            }
            if hasattr(res, 'G') and res.G is not None: # Save constraint violations if available
                results_to_save["pareto_constraints_G"] = res.G.tolist()

            opt_results_filename = "pareto_optimization_results.json"
            try:
                with open(opt_results_filename, 'w', encoding='utf-8') as f:
                    # default=str handles potential non-serializable numpy types if any remain
                    json.dump(results_to_save, f, indent=4, ensure_ascii=False, default=str)
                print(f"\nPareto optimization results saved to {opt_results_filename}")
            except Exception as e:
                print(f"Error saving Pareto optimization results to {opt_results_filename}: {e}")
        else:
            print("Optimization did not yield any valid solutions on the Pareto front.")
            if res.F is not None: print(f"Objective values (res.F): {res.F}")
            if res.X is not None: print(f"Parameters (res.X): {res.X}")


    else: # res is None
        print("Optimization process did not complete or failed to produce a result object.")

    print("\n--- End of Pareto Optimization ---")