from system import Participant, Global_Model
from torch.utils.data import DataLoader, random_split
import torch
import torch.optim as optim
import torch.nn as nn
import random


class HonestClient(Participant):
    def __init__(self, id, init_rep, tra_round_num, device,
                 client_dataset, train_size, batch_size, local_epochs, lr,
                 init_commit_scaling_factor,
                 adapt_bid_adj_intensity,
                 adapt_bid_max_delta,
                 min_commit_scaling_factor,
                 commit_decay_rate=0.5
                ):
        # 初始化父类
        super().__init__(id, "honest_client", init_rep, tra_round_num, device)
        # 初始化数据集
        self.dataset = client_dataset
        # 如果数据集长度小于2，则训练集和验证集相同；否则，按比例划分数据集
        if len(self.dataset) < 2: self.train_subset, self.val_subset = self.dataset, self.dataset
        else:
            train_len = int(len(self.dataset) * train_size)
            val_len = len(self.dataset) - train_len
            self.train_subset, self.val_subset = random_split(self.dataset, [train_len, val_len])
        # 初始化数据加载器
        pin_memory_flag = self.device != torch.device("cpu")
        num_workers_val = 8 if pin_memory_flag else 0
        self.train_loader = DataLoader(self.train_subset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory_flag, num_workers=num_workers_val, persistent_workers=True if num_workers_val > 0 else False)
        self.val_loader = DataLoader(self.val_subset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory_flag, num_workers=num_workers_val, persistent_workers=True if num_workers_val > 0 else False)
        # 初始化训练参数
        self.local_epochs, self.lr = local_epochs, lr
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.perf_before_local_train, self.perf_after_local_train = 0.0, 0.0
        # 初始化反向拍卖参数
        self.init_commit_scaling_factor = init_commit_scaling_factor                        # 承诺缩放因子
        self.successful_commitments_count = 0                                               # 成功满足承诺的次数
        self.total_evaluated_rounds_count = 0                                               # 评估的总轮次
        self.adapt_bid_adj_intensity = float(adapt_bid_adj_intensity)                       # 适应性承诺调整强度
        self.adapt_bid_max_delta = float(adapt_bid_max_delta)                               # 适应性承诺最大调整增量
        self.min_commit_scaling_factor = float(min_commit_scaling_factor)                   # 最小承诺缩放因子
        self.commit_decay_rate = float(commit_decay_rate)                                   # 承诺衰减率
    

    # 评估模型
    def evaluate_model(self, on_val_set=True):
        if self.model is None: return 0.0, float('inf')
        self.model.eval()
        loader = self.val_loader if on_val_set else self.train_loader
        if not loader or len(loader.dataset) == 0: return 0.0, float('inf')
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        if total == 0: return 0.0, float('inf')
        return correct / total, total_loss / total


    # 模型训练
    def local_train(self):
        if self.model is None or not self.train_loader or len(self.train_loader.dataset) == 0:
            self.perf_after_local_train = self.perf_before_local_train
            return self.model.state_dict() if self.model else None
        self.model.train()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr) 
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        for epoch in range(self.local_epochs):
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        self.perf_after_local_train, _ = self.evaluate_model(on_val_set=True)
        return self.model.state_dict()
    

    # 提交真实更新
    def gen_true_update(self, current_global_model_state):
        local_model_params = self._flatten_params(self.model.state_dict())
        global_model_params = self._flatten_params(current_global_model_state)
        local_update = local_model_params - global_model_params
        self.current_update = self._unflatten_params(local_update, current_global_model_state)
        return self.current_update


    # 投标
    def submit_bid(self):
        raw_promised_increase = self.perf_after_local_train - self.perf_before_local_train
        self.bid['promise'] = raw_promised_increase * self.init_commit_scaling_factor
        self.bid['reward'] = float(self.local_epochs * random.uniform(1.00, 1.25))
        return self.bid


    # 更新承诺缩放因子
    def update_adaptive_commitment_stats(self, was_verification_successful):
        self.total_evaluated_rounds_count += 1
        if was_verification_successful:
            self.successful_commitments_count += 1
        if self.total_evaluated_rounds_count > 0:
            # 计算新的承诺缩放因子
            new_scaling_factor = self.init_commit_scaling_factor * self.commit_decay_rate
            self.init_commit_scaling_factor = max(self.min_commit_scaling_factor, new_scaling_factor)