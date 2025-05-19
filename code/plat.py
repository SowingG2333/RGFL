import torch
import random
import numpy as np
import copy
from torch.utils.data import DataLoader
import torch.nn.functional as F
import json
import time
import traceback # 用于打印完整的异常堆栈

# --- Pymoo 导入 ---
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX  # 导入SBX交叉算子
from pymoo.operators.mutation.pm import PM     # 导入PM变异算子
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# --- 自定义模拟组件导入 ---
from system import get_mnist_data, Global_Model, Requester
from honest_client import HonestClient
from free_rider import FreeRider

# 用于存储所有评估的详细结果，包括声誉历史
all_evaluation_results_store = []


class Simulation:
    def __init__(self, params_X):
        # print(f"用于模拟的初始参数 params_X: {params_X}") # 调试时使用
        self.params_X = params_X
        self.num_total_participants = params_X.get("N", 10) # 参与者总数
        self.num_free_riders = params_X.get("N_f", 3)       # 搭便车者数量
        self.num_honest_clients = self.num_total_participants - self.num_free_riders # 诚实客户端数量
        
        self.verbose = params_X.get("verbose", True) # 控制是否打印详细日志
        
        # 设备选择 (GPU或CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        if self.verbose: # 只有在verbose模式下才打印设备信息
            print(f"--- SIM: 使用设备: {self.device} ---")


        self.max_rounds = params_X.get("T_max", 20) # 最大模拟轮数
        self.target_accuracy_threshold = params_X.get("target_accuracy_threshold", None) # 目标准确率阈值
        self.initial_M_t = params_X.get("initial_M_t", self.num_total_participants // 2) # 初始每轮选择的参与者数量
        self.iid_data_distribution = params_X.get("iid_data", True) # 数据是否独立同分布
        self.non_iid_alpha = params_X.get("non_iid_alpha", 0.3) # Non-IID 数据分布的参数 (Dirichlet分布的alpha)

        # 核心机制参数
        self.alpha_reward = params_X.get("alpha_reward")          # 声誉奖励因子
        self.beta_penalty_base = params_X.get("beta_penalty_base") # 声誉惩罚底数
        self.omega_m_update = params_X.get("omega_m_update")      # M_t (每轮选择参与者数量) 更新的学习率/平滑因子
        self.q_rounds_rep_change = params_X.get("q_rounds_rep_change") # 计算声誉变化时考虑的历史轮数 (对应Participant.tra_round_num)
        self.initial_reputation = params_X.get("initial_reputation") # 参与者初始声誉
        self.reputation_threshold = params_X.get("reputation_threshold") # 参与者被选中或被剔除的声誉阈值
        self.min_performance_constraint = params_X.get("min_performance_constraint") # 帕累托优化的最低模型性能约束

        # 搭便车者特定参数
        self.num_honest_rounds_for_fr_estimation = params_X.get("num_honest_rounds_for_fr_estimation", 2) # 搭便车者估计系统参数所需的初始（诚实行为）轮数
        self.adv_attack_c_param_fr = params_X.get("adv_attack_c_param_fr") # 搭便车者高级攻击策略中的 c 参数
        self.adv_attack_noise_dim_fraction_fr = params_X.get("adv_attack_noise_dim_fraction_fr") # 搭便车者攻击时添加噪声的维度比例
        self.adv_attack_scaled_delta_noise_std_fr = params_X.get("adv_attack_scaled_delta_noise_std", 0.0001) # 搭便车者在估计阶段噪声的标准差

        # 诚实客户端特定参数 (关于自适应投标)
        self.adaptive_bid_adjustment_intensity_gamma_honest = params_X.get("adaptive_bid_adjustment_intensity_gamma_honest") # 诚实客户端自适应承诺调整强度 gamma
        self.adaptive_bid_max_adjustment_delta_honest = params_X.get("adaptive_bid_max_adjustment_delta_honest")       # 诚实客户端自适应承诺最大调整增量 delta
        self.min_commitment_scaling_factor_honest = params_X.get("min_commitment_scaling_factor_honest")             # 诚实客户端最小承诺缩放因子

        # 诚实客户端训练和投标参数
        self.bid_gamma_honest = params_X.get("bid_gamma_honest", 0.8)       # 诚实客户端初始承诺缩放因子 gamma_bid
        self.local_epochs_honest = params_X.get("local_epochs_honest", 2) # 诚实客户端本地训练轮数
        self.lr_honest = params_X.get("lr_honest", 0.005)                 # 诚实客户端本地训练学习率
        self.batch_size_honest = params_X.get("batch_size_honest", 32)    # 诚实客户端本地训练批量大小

        # 模拟状态变量初始化
        self.participants = []  # 参与者列表
        self.requester = None   # 请求者对象
        self.current_round = 0  # 当前模拟轮数
        self.M_t = self.initial_M_t # 当前轮选择的参与者数量
        self.total_rewards_paid = 0.0 # 累计支付的总奖励
        self.rewards_paid_to_honest_clients = 0.0 # 累计支付给诚实客户端的奖励
        self.final_global_model_performance = 0.0 # 最终全局模型性能
        self.termination_round = self.max_rounds  # 模拟实际终止的轮数

        # 客户端行为历史记录
        self.client_l2_norm_history = {}       # 记录客户端更新的L2范数 (可选，当前未在优化中使用)
        self.client_cosine_similarity_history = {} # 记录客户端更新与平均更新的余弦相似度 (可选)
        self.client_reputation_history = {}    # 记录每个客户端每轮的声誉值
        self.client_types = {}                 # 存储客户端类型 ('honest_client' 或 'free_rider')
        
        # 搭便车者剔除相关的追踪变量
        self.all_fr_eliminated_logged = False # 是否已记录过“所有搭便车者被剔除”的消息
        self.rewards_at_all_fr_eliminated = float('inf') # 所有搭便车者被剔除时的累计奖励开销
        self.round_at_all_fr_eliminated = -1             # 所有搭便车者被剔除时的轮数
        self.all_fr_elimination_achieved_flag = False    # 是否已达成所有搭便车者都被剔除的状态

        # 每轮模拟统计数据
        self.simulation_stats = {
            "round_number": [], "model_accuracy": [], "active_honest_clients": [],
            "active_free_riders": [], "M_t_value": [], "cumulative_total_incentive_cost": [],
            "cumulative_real_incentive_cost": [], "cumulative_tir_history": []
        }


    # 日志记录方法
    def log_message(self, message):
        # 根据 verbose 状态打印日志信息
        if self.verbose:
            print(message)


    # 初始化模拟环境
    def initialize_environment(self):
        self.log_message("--- SIM: initialize_environment START ---")
        initial_global_model = Global_Model() 
        effective_num_honest_clients = max(0, self.num_honest_clients)

        # 获取数据集
        client_datasets, test_dataset_global = get_mnist_data(
            effective_num_honest_clients, self.iid_data_distribution, self.non_iid_alpha
        )
        if effective_num_honest_clients > 0 and (not client_datasets or effective_num_honest_clients > len(client_datasets)):
            self.log_message(f"警告: 请求 {effective_num_honest_clients} 个诚实客户端数据，实际分配 {len(client_datasets)} 个。")
            if not client_datasets and effective_num_honest_clients > 0:
                 raise ValueError(f"诚实客户端数据分配失败: 请求 {effective_num_honest_clients}, 得到 {len(client_datasets)}")

        pin_memory_flag = self.device.type != "cpu"
        num_workers_val = 0 # 在优化时设为0以避免多进程问题并加速初始化
        test_loader = DataLoader(test_dataset_global, batch_size=128, shuffle=False,
                                 pin_memory=pin_memory_flag, num_workers=num_workers_val)

        # 初始化请求者
        self.requester = Requester(initial_global_model, test_loader, self.device,
                             alpha_reward=self.alpha_reward,
                             beta_penalty_base=self.beta_penalty_base)
        self.participants = [] # 清空参与者列表

        # 初始化诚实客户端和搭便车者
        temp_participants = []
        for i in range(effective_num_honest_clients):
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

        if not temp_participants and (effective_num_honest_clients > 0 or self.num_free_riders > 0) :
            raise ValueError("没有参与者被初始化，尽管请求了参与者。检查数据分配和客户端初始化逻辑。")

        random.shuffle(temp_participants) # 打乱参与者顺序
        self.participants = temp_participants

        # 初始化客户端相关的历史记录字典
        self.client_reputation_history = {} # 重置声誉历史
        for p in self.participants:
            self.client_types[p.id] = p.type
            self.client_reputation_history[p.id] = [] # 为每个客户端初始化空的声誉列表


        # 重置模拟状态变量，确保每次 run_simulation 都是干净的开始
        self.current_round = 0
        self.M_t = min(self.initial_M_t, len(self.participants)) if self.participants else 0
        self.total_rewards_paid = 0.0
        self.rewards_paid_to_honest_clients = 0.0
        self.all_fr_eliminated_logged = False
        self.rewards_at_all_fr_eliminated = float('inf')
        self.round_at_all_fr_eliminated = -1
        self.all_fr_elimination_achieved_flag = False
        
        # 重置每轮统计数据列表
        for key_stat in self.simulation_stats: self.simulation_stats[key_stat] = []


        if self.requester:
            self.final_global_model_performance, _ = self.requester.evaluate_global_model()
            if self.requester.global_model:
                 self.requester.previous_global_model_state_flat = self.requester._flatten_params(self.requester.global_model.state_dict())
            self.log_message(f"初始全局模型性能: {self.final_global_model_performance:.4f}")

            # 记录第0轮（初始状态）的统计数据
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
            # 记录初始声誉
            for p_init_rep in self.participants:
                if p_init_rep.id in self.client_reputation_history: # 应该总是在
                     self.client_reputation_history[p_init_rep.id].append(p_init_rep.reputation)

        else: # 请求者未成功初始化
            self.final_global_model_performance = 0.0
            self.log_message("警告：服务器未正确初始化，无法评估初始模型或记录初始统计。")
            self.simulation_stats["round_number"].append(0)
            self.simulation_stats["model_accuracy"].append(0.0)
            self.simulation_stats["active_honest_clients"].append(0)
            self.simulation_stats["active_free_riders"].append(0)
            mt_initial_val = self.initial_M_t
            if self.participants: mt_initial_val = min(self.initial_M_t, len(self.participants))
            elif self.num_total_participants is not None: mt_initial_val = min(self.initial_M_t, self.num_total_participants)
            self.simulation_stats["M_t_value"].append(mt_initial_val)
            self.simulation_stats["cumulative_total_incentive_cost"].append(0.0)
            self.simulation_stats["cumulative_real_incentive_cost"].append(0.0)
            self.simulation_stats["cumulative_tir_history"].append(0.0)
            # 即使初始化失败，也尝试记录初始声誉
            for p_init_rep in self.participants: # 如果 participants 列表是空的，这里不会执行
                if p_init_rep.id in self.client_reputation_history:
                     self.client_reputation_history[p_init_rep.id].append(p_init_rep.reputation)
        self.log_message("--- SIM: initialize_environment END ---")


    # 更新每轮选择的参与者数量 M_t
    def update_M_t(self):
        if self.current_round == 0 or not self.participants:
            return

        active_participants = [p for p in self.participants if p.reputation >= self.reputation_threshold]
        if not active_participants:
            self.M_t = 0 if not self.participants else 1 # 如果没有活跃的但仍有参与者，至少选1个
            return

        delta_reputations_list = [p.get_delta_reputation() for p in active_participants if hasattr(p, 'get_delta_reputation')]
        if not delta_reputations_list: return

        sorted_delta_r = sorted(delta_reputations_list, reverse=True)
        k_prime_t_candidate = 1
        if len(sorted_delta_r) > 1:
            max_gap, current_max_gap_idx = -1.0, 0
            for j in range(len(sorted_delta_r) - 1):
                gap = sorted_delta_r[j] - sorted_delta_r[j+1]
                if gap > max_gap and j > 1: # 确保至少有两个参与者
                    max_gap = gap
                    current_max_gap_idx = j
            k_prime_t_candidate = current_max_gap_idx + 1
        elif len(sorted_delta_r) == 1: k_prime_t_candidate = 1
        else: return

        num_positive_delta_reps = sum(1 for dr in sorted_delta_r if dr > 1e-4) # 声誉有显著正增长的参与者
        final_k_prime_t = k_prime_t_candidate
        if k_prime_t_candidate <= 1 and num_positive_delta_reps > 1: # 如果候选领先者少，但很多表现好
            final_k_prime_t = max(k_prime_t_candidate, min(num_positive_delta_reps, 3)) # 适当增加领先者数量
        
        final_k_prime_t = min(final_k_prime_t, len(active_participants)) # 不超过活跃参与者总数
        final_k_prime_t = max(1, final_k_prime_t) # 至少为1
        new_M_t_float = self.omega_m_update * self.M_t + (1 - self.omega_m_update) * final_k_prime_t
        self.M_t = max(1, int(np.round(new_M_t_float))) # 四舍五入（此处要求 omega_m 要小于0.5才能收敛到 k_prime_t）
        if active_participants: self.M_t = min(self.M_t, len(active_participants)) # 上限为活跃数
        self.M_t = min(self.M_t, len(self.participants)) # 最终上限为总参与者数


    # 运行一轮模拟
    def run_one_round(self):
        self.current_round += 1
        m_t_for_this_round = self.M_t # 本轮实际使用的 M_t
        self.log_message(f"\n--- SIM: 第 {self.current_round}/{self.max_rounds} 轮 (M_t = {m_t_for_this_round}) ---")

        if not self.requester or not self.participants:
            self.log_message("请求者或参与者未初始化。结束本轮。")
            # 填充统计数据以保持列表长度一致
            self.simulation_stats["round_number"].append(self.current_round)
            self.simulation_stats["model_accuracy"].append(self.final_global_model_performance)
            self.simulation_stats["active_honest_clients"].append(0)
            self.simulation_stats["active_free_riders"].append(0)
            self.simulation_stats["M_t_value"].append(m_t_for_this_round)
            self.simulation_stats["cumulative_total_incentive_cost"].append(self.total_rewards_paid)
            self.simulation_stats["cumulative_real_incentive_cost"].append(self.rewards_paid_to_honest_clients)
            current_tir = (self.rewards_paid_to_honest_clients / self.total_rewards_paid) if self.total_rewards_paid > 1e-9 else 0.0
            self.simulation_stats["cumulative_tir_history"].append(current_tir)
            return True # 终止模拟

        round_start_global_state = copy.deepcopy(self.requester.global_model.state_dict())
        round_start_global_accuracy, _ = self.requester.evaluate_global_model()

        # 参与者生成更新
        client_updates_for_submission = {}
        for p in self.participants:
            p.set_model_state(copy.deepcopy(round_start_global_state)) # 每个参与者获取当前全局模型
            if p.type == "honest_client":
                p.perf_before_local_train, _ = p.evaluate_model(on_val_set=True) # 评估本地训练前性能
                p.local_train() # 本地训练
                p.gen_true_update(round_start_global_state) # 生成真实更新
            else: # 搭便车者
                p.gen_fabric_update(self.current_round, self.requester.global_model_param_diff_history,
                                    round_start_global_state, len(self.participants)) # 生成伪造更新
            if p.current_update: # 存储生成的更新以备提交
                client_updates_for_submission[p.id] = copy.deepcopy(p.current_update)
        
        # 参与者投标
        honest_bids_promises, honest_bids_rewards, honest_bids_ratios = [], [], []
        num_bidding_honest_clients = 0
        for p_bid in self.participants:
            if p_bid.reputation < self.reputation_threshold: # 声誉过低则不能投标
                p_bid.bid = {} 
                continue
            if p_bid.type == "honest_client":
                bid_data = p_bid.submit_bid() 
                if bid_data and 'promise' in bid_data and 'reward' in bid_data and bid_data['reward'] > 1e-6: # 有效投标
                    honest_bids_promises.append(bid_data['promise'])
                    honest_bids_rewards.append(bid_data['reward'])
                    honest_bids_ratios.append(bid_data['promise'] / bid_data['reward']) # 承诺/奖励 比率
                    num_bidding_honest_clients +=1
        
        # 搭便车者根据诚实客户端的投标情况进行投标
        if num_bidding_honest_clients > 0 :
            highest_honest_effectiveness = max(honest_bids_ratios) if honest_bids_ratios else 0
            lowest_honest_effectiveness = min(honest_bids_ratios) if honest_bids_ratios else 0
            lowest_honest_promise = min(honest_bids_promises) if honest_bids_promises else 0
            avg_honest_promise = np.mean(honest_bids_promises) if honest_bids_promises else 0
            for p_fr_bid in self.participants:
                if p_fr_bid.reputation >= self.reputation_threshold and p_fr_bid.type == "free_rider":
                    p_fr_bid.submit_bid(highest_honest_effectiveness, lowest_honest_effectiveness, lowest_honest_promise, avg_honest_promise)
        else: # 没有诚实客户端投标，搭便车者使用默认投标策略
            for p_fr_bid_default in self.participants:
                if p_fr_bid_default.reputation >= self.reputation_threshold and p_fr_bid_default.type == "free_rider":
                    p_fr_bid_default.submit_bid(0,0,0,0) 

        # 请求者选择参与者
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
                # 请求者验证更新、聚合模型、更新声誉并支付奖励
                verification_outcomes, _ = self.requester.verify_and_aggregate_updates(
                    updates_to_verify, round_start_global_accuracy
                )
                rewards_paid_ref = [self.total_rewards_paid] 
                self.requester.update_reputations_and_pay(self.participants, verification_outcomes, rewards_paid_ref)
                self.total_rewards_paid = rewards_paid_ref[0] # 更新累计总奖励

                # 计算本轮支付给诚实客户端的奖励
                current_round_rewards_to_honest_clients_this_round = 0
                for outcome in verification_outcomes:
                    if outcome["successful_verification"]:
                        p_find = next((p for p in self.participants if p.id == outcome["participant_id"]), None)
                        if p_find and self.client_types.get(p_find.id) == "honest_client":
                            current_round_rewards_to_honest_clients_this_round += p_find.bid.get('reward', 0)
                self.rewards_paid_to_honest_clients += current_round_rewards_to_honest_clients_this_round # 更新累计支付给诚实客户端的奖励
                
                if self.verbose: 
                    for outcome in verification_outcomes:
                        par = next((p_find for p_find in self.participants if p_find.id == outcome["participant_id"]), None)
                        if par:
                            status_str = "成功" if outcome["successful_verification"] else "失败"
                            self.log_message(f"  - {par.id} ({self.client_types[par.id]}), 声誉: {par.reputation:.2f}, "
                                  f"投标: P={par.bid.get('promise',0):.3f}/R={par.bid.get('reward',0):.2f}, "
                                  f"观察提升: {outcome.get('observed_increase',0):.3f}, 状态: {status_str}")
        
        # 所有参与者更新自己的声誉历史（用于计算delta_reputation）
        for p_every in self.participants: p_every.update_reputation_history() 
        
        # 在 Simulation 层面记录每个客户端本轮的声誉
        for p_track in self.participants:
            if p_track.id in self.client_reputation_history: # 应该总是在，因为在 init_env 中初始化了
                 self.client_reputation_history[p_track.id].append(p_track.reputation)
            else: # 以防万一
                 self.client_reputation_history[p_track.id] = [p_track.reputation]


        self.requester.update_global_model_history() # 更新请求者的全局模型参数差异历史
        self.final_global_model_performance, _ = self.requester.evaluate_global_model() # 评估当前全局模型性能
        self.update_M_t() # 为下一轮更新 M_t

        # 统计活跃客户端数量
        num_active_honest = sum(1 for p in self.participants if self.client_types.get(p.id) == "honest_client" and p.reputation >= self.reputation_threshold)
        num_active_free_riders = sum(1 for p in self.participants if self.client_types.get(p.id) == "free_rider" and p.reputation >= self.reputation_threshold)

        # 检查是否所有搭便车者都被剔除，并记录相关信息
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

        # 记录本轮统计数据
        self.simulation_stats["round_number"].append(self.current_round)
        self.simulation_stats["model_accuracy"].append(self.final_global_model_performance)
        self.simulation_stats["active_honest_clients"].append(num_active_honest)
        self.simulation_stats["active_free_riders"].append(num_active_free_riders)
        self.simulation_stats["M_t_value"].append(m_t_for_this_round)
        self.simulation_stats["cumulative_total_incentive_cost"].append(self.total_rewards_paid)
        self.simulation_stats["cumulative_real_incentive_cost"].append(self.rewards_paid_to_honest_clients)
        self.simulation_stats["cumulative_tir_history"].append(current_cumulative_tir)
        return self.check_termination_condition() # 检查是否满足终止条件


    # 检查模拟是否满足终止条件
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
                self.log_message("没有活跃的诚实客户端了。模拟可能卡住或所有诚实客户端被错误惩罚。终止模拟。")
                self.termination_round = self.current_round
                return True
        
        if self.current_round > 0:
            num_active_total_final_check_val = sum(1 for p in self.participants if p.reputation >= self.reputation_threshold)
            if num_active_total_final_check_val == 0:
                self.log_message("没有活跃的参与者了。终止模拟。")
                self.termination_round = self.current_round
                return True
        return False # 继续模拟


    # 保存模拟统计数据到JSON文件
    def save_simulation_stats(self, filename="simulation_results.json"):
        if not self.simulation_stats["round_number"] and self.current_round == 0 : # 如果是第0轮且无数据
             self.log_message("第0轮，统计数据列表为空，但会尝试保存摘要。")
        elif not self.simulation_stats["round_number"]:
            self.log_message("没有统计数据可供保存。")
            return

        records = []
        # 以 round_number 的长度为准，因为它标志着实际记录的轮数 (包括第0轮)
        num_recorded_rounds = len(self.simulation_stats["round_number"])
        
        stat_keys = list(self.simulation_stats.keys())
        for i in range(num_recorded_rounds):
            record = {}
            for key in stat_keys:
                if i < len(self.simulation_stats[key]): # 确保所有列表都有对应索引的数据
                    record[key] = self.simulation_stats[key][i]
                else:
                    record[key] = None # 或标记为缺失值
            records.append(record)

        # 计算最终的 FPR (误判率)
        final_fpr = 0.0
        if self.num_honest_clients > 0 and self.participants:
            num_honest_eliminated = sum(1 for p in self.participants if self.client_types.get(p.id) == "honest_client" and p.reputation < self.reputation_threshold)
            final_fpr = num_honest_eliminated / self.num_honest_clients
        elif self.num_honest_clients == 0 : final_fpr = 0.0

        # 计算最终的 TIR (真实激励率)
        final_tir = (self.rewards_paid_to_honest_clients / self.total_rewards_paid) if self.total_rewards_paid > 1e-9 else 0.0
        
        # 准备最终要保存的数据结构
        data_to_save = {
            "simulation_parameters": self.params_X, # 保存模拟运行时使用的参数
            "simulation_summary": { # 保存模拟结束后的总结性指标
                "termination_round": self.termination_round, # 模拟终止轮数
                "total_incentive_cost_final": self.total_rewards_paid, # 最终总激励开销
                "real_incentive_cost_final": self.rewards_paid_to_honest_clients, # 最终真实激励开销
                "false_positive_rate_final": final_fpr, # 最终诚实客户端误判率
                "global_model_accuracy_final": self.final_global_model_performance, # 最终全局模型准确率
                "true_incentive_rate_final": final_tir, # 最终真实激励率
                "cost_at_all_fr_eliminated": self.rewards_at_all_fr_eliminated if self.all_fr_elimination_achieved_flag else "Not_Achieved",
                "round_at_all_fr_eliminated": self.round_at_all_fr_eliminated if self.all_fr_elimination_achieved_flag else "Not_Achieved",
                "all_fr_elimination_achieved": self.all_fr_elimination_achieved_flag
            },
            "per_round_statistics": records, # 保存每轮的详细统计数据
            "client_reputation_history_per_round": self.client_reputation_history # 保存每个客户端的声誉变化历史
        }
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=4, ensure_ascii=False, default=lambda o: str(o) if isinstance(o, (np.integer, np.floating, np.bool_)) else o)
            self.log_message(f"模拟统计数据已成功保存到 {filename}")
        except Exception as e: 
            self.log_message(f"保存统计数据到文件 {filename} 时发生错误: {type(e).__name__} - {e}")


    # 运行完整的模拟过程
    def run_simulation(self):
        self.log_message(f"--- SIM: run_simulation CALLED (verbose={self.verbose}) ---")
        original_verbose_state = self.verbose
        if not self.verbose and 'PymooOpt' in self.params_X: # 在Pymoo优化调用时，即使全局verbose为False，也打开初始化阶段的日志
            self.verbose = True
            self.log_message(f"--- SIM: Temporarily forcing verbose ON for initialize_environment call during Pymoo eval ---")
        
        try:
            self.initialize_environment() # 初始化环境
        except ValueError as e_val: # 捕获预期的初始化值错误
            self.log_message(f"!!!!!! SIM: run_simulation - ValueError during initialize_environment: {e_val} !!!!!!")
            self.verbose = original_verbose_state # 恢复原始 verbose 状态
            return [float('inf'), 1.0], [1.0e9], {"PFM_final": 0.0, "error": f"ValueError in init_env: {e_val}", "client_reputation_history": self.client_reputation_history}
        except Exception as e_generic: # 捕获所有其他意外的初始化错误
            self.log_message(f"!!!!!! SIM: run_simulation - UNEXPECTED Exception during initialize_environment: {type(e_generic).__name__} - {e_generic} !!!!!!")
            traceback.print_exc() 
            self.verbose = original_verbose_state 
            return [float('inf'), 1.0], [1.0e9], {"PFM_final": 0.0, "error": f"Exception in init_env: {e_generic}", "client_reputation_history": self.client_reputation_history}
        finally: # 确保 verbose 状态被恢复
            if self.verbose != original_verbose_state:
                 self.log_message(f"--- SIM: Restoring verbose to {original_verbose_state} ---")
                 self.verbose = original_verbose_state

        # 检查初始化后请求者和参与者是否有效
        if not self.requester or (not self.participants and (self.num_honest_clients > 0 or self.num_free_riders > 0)) :
             self.log_message("--- SIM: run_simulation --- ABORTING early, requester or participants not properly initialized AFTER init_env call. ---")
             self.termination_round = 0
             if self.verbose: # 只有在verbose模式下才保存失败的JSON
                filename = f"sim_results_START_FAIL_N{self.num_total_participants}_Nf{self.num_free_riders}_alphaR{self.alpha_reward:.2f}_betaP{self.beta_penalty_base:.2f}_q{self.q_rounds_rep_change}.json"
                try: self.save_simulation_stats(filename)
                except: pass 
             return [float('inf'), 1.0], [1.0e9], {"PFM_final": 0.0, "error": "Requester/Participants not initialized", "client_reputation_history": self.client_reputation_history}

        self.log_message("--- SIM: run_simulation --- Proceeding to main simulation loop. ---")
        terminated = False
        while not terminated: # 主模拟循环
            terminated = self.run_one_round()
            # 提前终止条件：性能过低
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

        # --- 模拟结束后计算最终指标 ---
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
        
        cost_when_all_fr_eliminated = self.rewards_at_all_fr_eliminated
        if not self.all_fr_elimination_achieved_flag and self.num_free_riders > 0:
            cost_when_all_fr_eliminated = float('inf') # 如果有搭便车者但未全部剔除，则成本为无穷大
            self.log_message(f"警告: 模拟结束时仍有活跃的搭便车者。剔除所有搭便车者时的开销将设为极大值。")
        elif self.num_free_riders == 0: # 如果一开始就没有搭便车者
            cost_when_all_fr_eliminated = 0.0 

        self.log_message(f"\n--- SIM: 模拟结束 (run_simulation) ---")
        self.log_message(f"终止轮数 (T_term): {T_term}")
        self.log_message(f"最终总奖励开销 (C_total_final): {C_total_final:.2f}")
        self.log_message(f"最终误判率 (FPR_final): {FPR_final:.4f} ({num_honest_eliminated}/{self.num_honest_clients if self.num_honest_clients > 0 else 'N/A'})")
        self.log_message(f"最终模型准确率 (PFM_final): {PFM_final:.4f}")
        self.log_message(f"最终真实激励率 (TIR_final): {true_incentive_rate_final:.4f}")
        self.log_message(f"剔除所有FR时的奖励开销 (C_elim_fr): {cost_when_all_fr_eliminated if cost_when_all_fr_eliminated != float('inf') else 'Infinite/Not Achieved'}")
        self.log_message(f"剔除所有FR时的轮数 (Round_elim_fr): {self.round_at_all_fr_eliminated if self.all_fr_elimination_achieved_flag else 'N/A'}")

        # 仅在 verbose 模式下（通常是单次运行）保存详细的JSON日志文件
        if self.verbose and not self.params_X.get('PymooOpt', False): # 避免优化时频繁保存
            filename = f"sim_N{self.num_total_participants}_Nf{self.num_free_riders}_alphaR{self.alpha_reward:.2f}_betaP{self.beta_penalty_base:.2f}_q{self.q_rounds_rep_change}.json"
            try: self.save_simulation_stats(filename)
            except: pass 

        # 准备返回给优化器的目标值和约束
        objectives_for_pareto = [cost_when_all_fr_eliminated, FPR_final]
        constraint_violation = self.min_performance_constraint - PFM_final
        
        other_metrics_to_return = { 
            "T_term": T_term,
            "PFM_final": PFM_final,
            "TIR_final": true_incentive_rate_final,
            "C_total_final": C_total_final,
            "round_at_all_fr_eliminated": self.round_at_all_fr_eliminated if self.all_fr_elimination_achieved_flag else -1,
            "all_fr_elimination_achieved_flag": self.all_fr_elimination_achieved_flag,
            "constraint_PFM_violation": constraint_violation,
            "client_reputation_history": copy.deepcopy(self.client_reputation_history) # 返回声誉历史的深拷贝
        }
        return objectives_for_pareto, [constraint_violation], other_metrics_to_return


    # 此方法专为优化器调用，运行模拟并返回目标和约束
    def evaluate_parameters_for_optimization(self):
        # 此方法专为优化器调用，运行模拟并返回目标和约束
        return self.run_simulation()


# --- 全局基础模拟参数配置 ---
BASE_SIMULATION_PARAMS = {
    "N": 10, "N_f": 3, "T_max": 100, # 为优化演示减少T_max，实际应更大
    "iid_data": True, "non_iid_alpha": 0.5,
    # alpha_reward, beta_penalty_base, q_rounds_rep_change 将由优化器设置
    "omega_m_update": 0.4, # M_t 更新的平滑系数
    "initial_reputation": 10.0,
    "reputation_threshold": 0.01, 
    "min_performance_constraint": 0.90, # 最低性能约束：PFM_final 必须 >= 0.90
    "target_accuracy_threshold": 0.90, # 当全局模型性能达到此值时终止模拟

    "num_honest_rounds_for_fr_estimation": 2,
    "adv_attack_c_param_fr": 0.5,
    "adv_attack_noise_dim_fraction_fr": 1.0,
    "adv_attack_scaled_delta_noise_std": 0.001,

    "bid_gamma_honest": 1.0,
    "local_epochs_honest": 1, # 为加速优化，保持较低值
    "lr_honest": 0.001, 
    "batch_size_honest": 64,

    "adaptive_bid_adjustment_intensity_gamma_honest": 0.15,
    "adaptive_bid_max_adjustment_delta_honest": 0.4,
    "min_commitment_scaling_factor_honest": 0.2,
    "verbose": False, # 重要：在优化运行时设为False以减少控制台输出
    "PymooOpt": True  # 添加一个标志，指示这是Pymoo优化调用
}

# 用于存储所有评估结果的列表（包括声誉历史）
# 注意：对于非常长的优化运行，这可能会消耗大量内存。可以考虑写入临时文件。
optimization_evaluation_history = []

# --- Pymoo 问题定义 ---
class ParetoOptimizationProblem(Problem):
    def __init__(self, base_sim_params):
        print("--- PYMOO: ParetoOptimizationProblem __init__ CALLED ---")
        self.base_sim_params = base_sim_params
        self.evaluation_counter = 0 
        
        self.variable_names = ["alpha_reward", "beta_penalty_base", "q_rounds_rep_change"]
        
        # 变量下界 (顺序必须与 variable_names 一致)
        xl = np.array([
            1.0,    # alpha_reward (奖励因子) 下界
            1.1,    # beta_penalty_base (惩罚底数) 下界 (必须 > 1)
            2       # q_rounds_rep_change (声誉更新历史轮数) 下界 (整数)
        ], dtype=np.double)
        
        # 变量上界
        xu = np.array([
            3.0,    # alpha_reward 上界
            3.0,    # beta_penalty_base 上界
            10      # q_rounds_rep_change 上界 (整数)
        ], dtype=np.double)

        # print(f"DEBUG __init__: n_var=3, n_obj=2, n_constr=1")
        # print(f"DEBUG __init__: xl={xl} (type: {type(xl)}, dtype: {xl.dtype})")
        # print(f"DEBUG __init__: xu={xu} (type: {type(xu)}, dtype: {xu.dtype})")
        if not np.all(xu >= xl):
            print("!!!!!!!!!! DEBUG __init__: XU < XL DETECTED !!!!!!!!!!")

        super().__init__(n_var=len(self.variable_names), # 变量数量
                         n_obj=2,     # 目标数量: C_elim_fr, FPR_final
                         n_constr=1,  # 约束数量: PFM_final >= min_performance_constraint
                         xl=xl,       
                         xu=xu,       
                         elementwise=True) # 一次评估一个解决方案
        print(f"--- PYMOO: ParetoOptimizationProblem super().__init__ FINISHED ---")

    def _evaluate(self, x_array, out, *args, **kwargs):
        # Pymoo 调用此方法来评估一个参数组合 x_array
        print(f"--- PYMOO: _evaluate CALLED with x_array: {x_array} ---") 
        
        self.evaluation_counter += 1
        current_params_X = self.base_sim_params.copy()
        
        # 从 x_array 中提取参数值并更新 current_params_X
        alpha_r_val = x_array[0]
        beta_p_val = x_array[1]
        q_rounds_val = int(round(x_array[2])) # q_rounds_rep_change 是整数

        current_params_X["alpha_reward"] = alpha_r_val
        current_params_X["beta_penalty_base"] = beta_p_val
        current_params_X["q_rounds_rep_change"] = q_rounds_val
        # 在优化时，即使 base_params 的 verbose 是 False，也强制打开 Simulation 内部的 verbose 以便调试初始化
        # current_params_X["verbose"] = True # 调试时可以强制打开

        print(f"--- [OPTIMIZER EVAL START] Eval #{self.evaluation_counter} ---")
        print(f"Params for this eval: alpha_R={alpha_r_val:.3f}, beta_P={beta_p_val:.3f}, q_rounds={q_rounds_val}, verbose_in_params={current_params_X['verbose']}")

        objectives = [float('inf'), 1.0]  # 初始化为默认失败的目标值
        constraints_violation = [1.0e9]   # 初始化为默认大的约束违反 (1.0e9 代替 inf)
        other_metrics = {"error": "Initialization or evaluation failed early", "PFM_final": 0.0, "client_reputation_history": {}}


        sim_instance_created_successfully = False
        try:
            # print("--- [OPTIMIZER] Attempting to create Simulation instance ---") # 调试时使用
            sim = Simulation(params_X=current_params_X) 
            sim_instance_created_successfully = True
            # print("--- [OPTIMIZER] Simulation instance CREATED successfully ---") # 调试时使用
            
            # print("--- [OPTIMIZER] Attempting sim.evaluate_parameters_for_optimization() ---") # 调试时使用
            objectives, constraints_violation, other_metrics = sim.evaluate_parameters_for_optimization()
            # print("--- [OPTIMIZER] sim.evaluate_parameters_for_optimization() COMPLETED ---") # 调试时使用

        except Exception as e: 
            print(f"!!!!!!!!!! EXCEPTION during Simulation instantiation or evaluation in _evaluate !!!!!!!!!!")
            print(f"Parameters that caused error: alpha_R={alpha_r_val:.3f}, beta_P={beta_p_val:.3f}, q_rounds={q_rounds_val}")
            print(f"Exception type: {type(e)}")
            print(f"Exception message: {e}")
            traceback.print_exc() 
            # objectives 和 constraints_violation 保持默认的失败值
            other_metrics["error"] = f"Exception: {type(e).__name__} - {e}"
            if "client_reputation_history" not in other_metrics: # 确保有这个键
                 other_metrics["client_reputation_history"] = {}


        # 存储本次评估的详细结果，包括声誉历史
        current_eval_data = {
            "params_array": x_array.tolist(), # 保存原始参数数组
            "params_dict": { # 保存参数字典形式
                "alpha_reward": alpha_r_val,
                "beta_penalty_base": beta_p_val,
                "q_rounds_rep_change": q_rounds_val
            },
            "objectives": objectives,
            "constraints_violation": constraints_violation,
            "other_metrics": other_metrics # other_metrics 已经包含了 client_reputation_history
        }
        optimization_evaluation_history.append(current_eval_data)


        pfm_final_for_debug = other_metrics.get('PFM_final', 'N/A')
        min_perf_constraint_for_debug = current_params_X.get('min_performance_constraint', 'N/A')
        
        # print(f"--- [OPTIMIZER EVAL END] Eval #{self.evaluation_counter} ---") # 调试时使用
        # if not sim_instance_created_successfully:
        #     print("  NOTE: Simulation instance was NOT created successfully.")
        # print(f"  PFM_final: {pfm_final_for_debug}")
        # print(f"  Min Perf Constraint: {min_perf_constraint_for_debug}")
        # print(f"  Returned Objectives: {objectives}")
        # print(f"  Returned Constraint Violation: {constraints_violation}")


        out["F"] = np.array(objectives, dtype=np.double)
        cv_value = constraints_violation[0] if isinstance(constraints_violation, list) and len(constraints_violation)>0 else constraints_violation
        if not np.isfinite(cv_value): 
            cv_value = 1.0e9 
        out["G"] = np.array([cv_value], dtype=np.double) 


def set_random_seed(seed):
    # 设置随机种子以保证可复现性
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False   


if __name__ == "__main__":
    master_seed = 42 
    set_random_seed(master_seed)
    print(f"Master seed set to {master_seed}")

    # --- 选项 1: 运行单次模拟进行测试 ---
    run_single_test_simulation = False # 设为 True 以运行此块进行调试
    if run_single_test_simulation:
        print("\n--- 运行单次模拟测试 ---")
        test_params_X = BASE_SIMULATION_PARAMS.copy()
        # 为测试覆盖参数
        test_params_X["alpha_reward"] = 2.0
        test_params_X["beta_penalty_base"] = 1.1
        test_params_X["q_rounds_rep_change"] = 5
        test_params_X["T_max"] = 50 # 为单次测试使用合理的 T_max
        test_params_X["verbose"] = True # 为单次运行启用详细日志
        test_params_X["PymooOpt"] = False # 表明这不是Pymoo优化调用

        sim_test = Simulation(params_X=test_params_X)
        objectives, constraints, other_metrics = sim_test.evaluate_parameters_for_optimization()

        print(f"\n--- 单次模拟评估结果 ---")
        print(f"测试参数: alpha_reward={test_params_X['alpha_reward']}, beta_penalty_base={test_params_X['beta_penalty_base']}, q_rounds_rep_change={test_params_X['q_rounds_rep_change']}")
        obj1_val_test = objectives[0] if objectives[0] != float('inf') else "Infinite/Not Achieved"
        print(f"优化目标 (C_elim_fr, FPR_final): ({obj1_val_test}, {objectives[1]:.4f})")
        constraint_val_test = constraints[0] 
        print(f"约束违反 (min_PFM - PFM_final <= 0): {constraint_val_test:.4f}")
        if constraint_val_test <= 1e-5: 
            print("约束：最终模型性能达标。")
        else:
            print("约束：最终模型性能未达标。")
        print(f"其他指标: {other_metrics}")
        
        # 单独保存这次测试的JSON结果
        single_run_filename = "single_simulation_test_results.json"
        sim_test.save_simulation_stats(single_run_filename) # 调用 save_simulation_stats 保存
        print(f"单次模拟测试结果已保存到: {single_run_filename}")
        print("--- 单次模拟测试结束 ---\n")

    # --- 选项 2: 运行帕累托优化 ---
    print("\n--- 运行帕累托优化 ---")
    # 将 BASE_SIMULATION_PARAMS 的 PymooOpt 标志设为 True
    # 这样 Simulation 内部可以根据需要调整行为（例如日志级别）
    actual_base_params_for_opt = BASE_SIMULATION_PARAMS.copy()
    actual_base_params_for_opt["PymooOpt"] = True
    actual_base_params_for_opt["verbose"] = False # 优化时通常关闭 Simulation 内部的详细日志

    problem = ParetoOptimizationProblem(base_sim_params=actual_base_params_for_opt)

    algorithm = NSGA2(
        pop_size=10,  # 种群大小: 实际运行时应增加 (例如, 20, 50, 100)
        crossover = SBX(prob=0.9, eta=15), # 使用SBX交叉
        mutation = PM(eta=20),             # 使用PM变异
        eliminate_duplicates=True 
    )

    # 终止条件
    generations_count = 5 # 迭代代数: 实际运行时应增加 (例如, 25, 50, 100)
    termination = get_termination("n_gen", generations_count)
    
    print(f"开始优化 NSGA-II: pop_size={algorithm.pop_size}, generations_count={generations_count}")
    start_time = time.time()

    res = None
    try:
        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=master_seed, 
                       verbose=True,     
                       save_history=False) # 如果需要历史数据，设为True，但会消耗内存
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
                
                # 从 optimization_evaluation_history 中找到对应的详细评估数据
                # 注意：这里假设参数数组可以作为唯一标识符。浮点数比较可能需要容差。
                # 一个更稳健的方法是在 _evaluate 中为每个评估生成一个唯一ID并存储。
                # 为简化，这里直接比较参数数组（转换为元组使其可哈希）。
                # Pymoo 的 res.X 中的解是去重和排序后的，可能不直接对应 optimization_evaluation_history 的顺序。
                # 我们需要根据参数值来匹配。
                
                matched_eval_data = None
                for eval_data in optimization_evaluation_history:
                    # 比较浮点数数组需要小心，这里用简单的列表比较，实际可能需要 np.allclose
                    if np.allclose(np.array(eval_data["params_array"]), solution_params_array, atol=1e-6):
                        matched_eval_data = eval_data
                        break
                
                current_solution_output = {
                    "solution_index": i + 1,
                    "parameters": {
                        problem.variable_names[0]: solution_params_array[0],
                        problem.variable_names[1]: solution_params_array[1],
                        problem.variable_names[2]: int(round(solution_params_array[2]))
                    },
                    "objectives": {
                        "C_elim_fr": solution_objectives[0] if solution_objectives[0] != float('inf') else "Infinite/NotAchieved",
                        "FPR_final": solution_objectives[1]
                    }
                }
                if hasattr(res.G, '__len__') and i < len(res.G) and res.G[i] is not None: # 检查 res.G 是否存在且有效
                     current_solution_output["constraint_violation"] = res.G[i][0] # G是二维的，每行一个解的约束
                
                if matched_eval_data and matched_eval_data.get("other_metrics"):
                    om = matched_eval_data["other_metrics"]
                    current_solution_output["termination_round_sim"] = om.get("T_term")
                    current_solution_output["final_model_accuracy_sim"] = om.get("PFM_final")
                    current_solution_output["client_reputation_history_per_round"] = om.get("client_reputation_history", {})
                else:
                    current_solution_output["client_reputation_history_per_round"] = "Not Found in History"
                    print(f"警告: 未能在历史记录中找到解 {i+1} (参数: {solution_params_array}) 的详细评估数据。")

                pareto_solutions_output.append(current_solution_output)

                # 打印每个解的简要信息
                print(f"\n解 {i+1}:")
                print(f"  参数: alpha_R={current_solution_output['parameters']['alpha_reward']:.3f}, "
                      f"beta_P={current_solution_output['parameters']['beta_penalty_base']:.3f}, "
                      f"q_rounds={current_solution_output['parameters']['q_rounds_rep_change']}")
                print(f"  目标: C_elim_fr={current_solution_output['objectives']['C_elim_fr']}, "
                      f"FPR_final={current_solution_output['objectives']['FPR_final']:.4f}")
                if "constraint_violation" in current_solution_output:
                    print(f"  约束违反: {current_solution_output['constraint_violation']:.4f}")


            # 保存包含声誉历史的帕累托解到JSON文件
            final_results_to_save = {
                "optimization_summary": {
                    "total_solutions_on_pareto_front": len(res.X),
                    "optimization_time_minutes": optimization_duration_minutes,
                    "base_sim_params_used_for_opt": actual_base_params_for_opt, # 使用实际传入problem的参数
                    "pymoo_algorithm_details": {
                        "name": algorithm.__class__.__name__, 
                        "pop_size": algorithm.pop_size,
                        "termination_criterion": str(termination) 
                    }
                },
                "pareto_solutions_details": pareto_solutions_output
            }
            
            opt_results_filename = "pareto_optimization_results_with_reputation.json"
            try:
                with open(opt_results_filename, 'w', encoding='utf-8') as f:
                    json.dump(final_results_to_save, f, indent=4, ensure_ascii=False, default=str)
                print(f"\n包含声誉历史的帕累托优化结果已保存到 {opt_results_filename}")
            except Exception as e:
                print(f"保存包含声誉历史的帕累托优化结果时发生错误: {e}")
        else:
            print("优化过程未产生有效的帕累托前沿解。")
            if res.F is not None: print(f"目标值 (res.F): {res.F}")
            if res.X is not None: print(f"参数 (res.X): {res.X}")

    else: 
        print("优化过程未能完成或未能生成结果对象 (res is None)。")

    # 可选：保存所有评估历史（如果内存允许且需要详细分析）
    # save_full_evaluation_history = False
    # if save_full_evaluation_history:
    #     full_history_filename = "all_optimization_evaluations_history.json"
    #     try:
    #         with open(full_history_filename, 'w', encoding='utf-8') as f:
    #             json.dump(optimization_evaluation_history, f, indent=2, ensure_ascii=False, default=str)
    #         print(f"\n所有评估历史已保存到 {full_history_filename}")
    #     except Exception as e:
    #         print(f"保存所有评估历史时发生错误: {e}")


    print("\n--- 帕累托优化结束 ---")