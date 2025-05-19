import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import copy
import random

# 配置参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 45
NUM_ROUNDS = 100                                        # 总训练轮次
NUM_HONEST_CLIENTS = 20                                 # 诚实客户端数目
NUM_FREE_RIDERS = 1                                     # 搭便车者数目
CLIENT_EPOCHS = 1                                       # 本地训练轮次
CLIENT_LR = 0.005                                       # 诚实客户端学习率
CLIENT_BATCH_SIZE = 128                                 # 诚实客户端批次大小
TEST_BATCH_SIZE = 1000                                  # 验证数据集大小

# 数据分布参数              
IID_DATA_DISTRIBUTION = False                           # 是否独立同分布
NON_IID_ALPHA = 0.5                                     # 迪利克雷分布参数

# 搭便车者参数
NUM_HONEST_ROUNDS_FOR_ESTIMATION_FREERIDER = 2          # 诚实轮次数目
ADV_ATTACK_C_PARAM = 0.5                                # 计算E(cosB)的C参数
ADV_ATTACK_NOISE_DIM_FRACTION = 0.75                    # 添加噪声的维度比例


# 随机种子设置函数
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 模型定义
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 数据集划分
def get_mnist_data(num_clients, iid, non_iid_alpha, data_path='./data', client_batch_size=64, test_batch_size=1000):
    transform_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset_full = datasets.MNIST(data_path, train=True, download=True, transform=transform_mnist)
    test_dataset_full = datasets.MNIST(data_path, train=False, download=True, transform=transform_mnist)
    
    client_datasets_subsets = []
    
    if num_clients == 0:
        test_loader_local = DataLoader(test_dataset_full, batch_size=test_batch_size, shuffle=False)
        return [], test_loader_local

    if iid:
        total_size = len(train_dataset_full)
        indices = list(range(total_size))
        random.shuffle(indices)
        split_size = total_size // num_clients
        for i in range(num_clients):
            start_idx, end_idx = i * split_size, (i + 1) * split_size if i < num_clients - 1 else total_size
            if not indices[start_idx:end_idx]: continue
            client_datasets_subsets.append(Subset(train_dataset_full, indices[start_idx:end_idx]))
    else:
        labels = np.array(train_dataset_full.targets)
        num_classes = 10
        min_size_per_client = 10
        
        client_indices_map = {i: [] for i in range(num_clients)}
        
        for c in range(num_classes):
            indices_c = np.where(labels == c)[0]
            np.random.shuffle(indices_c)
            
            proportions = np.random.dirichlet([non_iid_alpha] * num_clients)

            target_samples_for_class_c = (proportions * len(indices_c)).astype(int)
            
            delta = len(indices_c) - np.sum(target_samples_for_class_c)
            if delta != 0:
                for _ in range(abs(delta)):
                    target_samples_for_class_c[random.randrange(num_clients)] += np.sign(delta)
            
            current_idx_in_class_c = 0
            for client_id in range(num_clients):
                num_samples_for_client = target_samples_for_class_c[client_id]
                client_indices_map[client_id].extend(indices_c[current_idx_in_class_c : current_idx_in_class_c + num_samples_for_client])
                current_idx_in_class_c += num_samples_for_client

        for i in range(num_clients):
            # 确保不存在空数据子集
            if not client_indices_map[i] or len(client_indices_map[i]) < min_size_per_client :
                current_client_indices = client_indices_map[i]
                needed_samples = min_size_per_client - len(current_client_indices)
                if needed_samples > 0:
                    all_train_indices = list(range(len(train_dataset_full)))
                    additional_indices = random.sample(all_train_indices, needed_samples)
                    current_client_indices.extend(additional_indices)
                client_datasets_subsets.append(Subset(train_dataset_full, list(set(current_client_indices))[:min_size_per_client*2])) # Use set to remove duplicates if any, and cap size
            else:
                client_datasets_subsets.append(Subset(train_dataset_full, client_indices_map[i]))

    # 创建数据加载器
    train_data_loaders = []
    for client_subset in client_datasets_subsets:
        if len(client_subset) > 0:
             train_data_loaders.append(DataLoader(client_subset, batch_size=client_batch_size, shuffle=True, num_workers=0, pin_memory=False)) # num_workers=0 for simplicity with small datasets
        else:
             print(f"Warning: Client dataset subset is empty. Skipping DataLoader creation for this client.")
    
    test_loader_local = DataLoader(test_dataset_full, batch_size=test_batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return train_data_loaders, test_loader_local


# 模型训练函数
def train_client_model(model, data_loader, epochs, lr, device):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
    return model.state_dict()


# 模型评估函数
def test_model(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss(reduction='sum')(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy, test_loss


# 展平模型参数
def flatten_params(model_state_dict):
    return torch.cat([p.view(-1) for p in model_state_dict.values()])


# 还原模型参数
def unflatten_params(flat_params, target_model_state_dict_template):
    new_state_dict = copy.deepcopy(target_model_state_dict_template)
    current_pos = 0
    for key in new_state_dict:
        param = new_state_dict[key]
        num_elements = param.numel()
        new_state_dict[key] = flat_params[current_pos : current_pos + num_elements].view_as(param).clone()
        current_pos += num_elements
    return new_state_dict


# 计算参数变化量
def calculate_update_delta(old_state_dict, new_state_dict, device):
    old_flat = flatten_params(old_state_dict).to(device)
    new_flat = flatten_params(new_state_dict).to(device)
    return new_flat - old_flat


# 参与者基类
class Participant:
    def __init__(self, id, type, device):
        self.id = id
        self.type = type                        # "honest" / "free_rider"
        self.device = device        
        self.current_model_state = None         # 当前模型状态
        self.current_update_flat = None         # 上一次更新的扁平化参数


    # 设置当前模型状态
    def set_model_state(self, global_model_state):
        self.current_model_state = copy.deepcopy(global_model_state)


    # 获取当前更新
    def get_update(self):
        return self.current_update_flat


# 诚实客户端
class HonestClient(Participant):
    def __init__(self, id, data_loader, epochs, lr, device):
        super().__init__(id, "honest", device)
        self.data_loader = data_loader
        self.epochs = epochs
        self.lr = lr


    # 计算更新
    def compute_update(self, global_model_state_dict):
        self.set_model_state(global_model_state_dict)
        local_model = MLP().to(self.device)
        local_model.load_state_dict(copy.deepcopy(global_model_state_dict))
        
        trained_local_state_dict = train_client_model(local_model, self.data_loader, self.epochs, self.lr, self.device)
        
        self.current_update_flat = calculate_update_delta(global_model_state_dict, trained_local_state_dict, self.device)
        return self.current_update_flat


# 搭便车者
class FreeRider(Participant):
    def __init__(self, id, data_loader, epochs, lr, device,
                 num_honest_rounds_for_estimation,
                 adv_attack_c_param,
                 adv_attack_noise_dim_fraction,
                 ):
        super().__init__(id, "free_rider", device)
        self.data_loader = data_loader 
        self.epochs = epochs         
        self.lr = lr                 

        self.num_honest_rounds_for_estimation = num_honest_rounds_for_estimation
        self.adv_attack_c_param = adv_attack_c_param
        self.adv_attack_noise_dim_fraction = adv_attack_noise_dim_fraction

        self.estimated_lambda_bar = None
        self.global_model_norm_diff_history_for_estimation = []
        self.attack_phase = "initial_hibernation"
        self.last_global_update_for_scaled_delta_flat = None
        self.second_last_global_update_for_scaled_delta_flat = None
        self.expected_cos_beta_history = []
    

    # 更新历史全局模型梯度的l2范数
    def _update_estimation_history(self, server_g_t_flat_history):
        if server_g_t_flat_history and len(server_g_t_flat_history) > 0:
            g_t_norm = torch.linalg.norm(server_g_t_flat_history[-1]).item()
            self.global_model_norm_diff_history_for_estimation.append(g_t_norm)


    # 估计lambda_bar
    def _estimate_lambda_bar(self):
        history = self.global_model_norm_diff_history_for_estimation
        if len(history) >= 2:
            t_eff = len(history) - 1                # 计算有效轮次
            ratio = history[0] / history[-1]        # 计算l(t)/l(0)
            try:
                # ！疑问
                lambda_val = (1.0 / t_eff) * np.log(ratio)
                self.estimated_lambda_bar = lambda_val
                # print(f"DEBUG: Rider {self.id} estimated lambda_bar: {self.estimated_lambda_bar} at round {current_round_num_for_context+1}")
            except Exception as e:
                print(f"DEBUG: Rider {self.id} error estimating lambda: {e}. Keep original: {self.estimated_lambda_bar}")
                pass

    # 计算预期的余弦相似度
    def _calculate_expected_cosine_beta(self, current_round_num):        
        exp_term = np.exp(2 * self.estimated_lambda_bar * current_round_num)
        cos_beta = (self.adv_attack_c_param**2) / (self.adv_attack_c_param**2 + exp_term)
        return max(0.0, min(1.0, cos_beta))


    # 计算更新
    def compute_update(self, current_round_num, global_model_state_dict, server_g_t_flat_history, num_total_clients_n):
        self.set_model_state(global_model_state_dict)
        self._update_estimation_history(server_g_t_flat_history)

        if current_round_num < self.num_honest_rounds_for_estimation:
            self.attack_phase = "parameter_estimation"
        else:
            self.attack_phase = "advanced_attack"
            self._estimate_lambda_bar()

        # 处理服务器最近两次更新梯度
        if server_g_t_flat_history:
            if len(server_g_t_flat_history) >= 1:
                self.last_global_update_for_scaled_delta_flat = server_g_t_flat_history[-1].clone().to(self.device)
            if len(server_g_t_flat_history) >= 2:
                self.second_last_global_update_for_scaled_delta_flat = server_g_t_flat_history[-2].clone().to(self.device)

        fabricated_update_flat = None
        # 扁平化全局模型参数
        flat_ref_params = flatten_params(global_model_state_dict)
        # 检查全局模型参数是否为空
        if flat_ref_params.numel() == 0:
            print(f"CRITICAL ERROR: FreeRider {self.id} received empty global model state. Returning empty update.")
            self.current_update_flat = torch.tensor([]).to(self.device)
            self.expected_cos_beta_history.append(np.nan)
            return self.current_update_flat

        # 初始化要估计的余弦相似度
        current_round_logged_cos_beta = np.nan

        if self.attack_phase == "parameter_estimation":
            # print(f"INFO: Free-rider {self.id} (Round {current_round_num + 1}) is in 'parameter_estimation' phase - performing normal training.")
            local_model = MLP().to(self.device)
            local_model.load_state_dict(copy.deepcopy(global_model_state_dict))
            trained_local_state_dict = train_client_model(local_model, self.data_loader, self.epochs, self.lr, self.device)
            fabricated_update_flat = calculate_update_delta(global_model_state_dict, trained_local_state_dict, self.device)

        elif self.attack_phase == "advanced_attack":
            # print(f"INFO: Free-rider {self.id} (Round {current_round_num + 1}) is in 'advanced_attack' phase - crafting update.")
            norm_g_current = torch.linalg.norm(self.last_global_update_for_scaled_delta_flat)
            norm_g_previous = torch.linalg.norm(self.second_last_global_update_for_scaled_delta_flat)

            scaled_delta_factor = 1.0
            if norm_g_previous > 1e-9:
                scaled_delta_factor = norm_g_current.item() / norm_g_previous.item()
            else:
                print(f"CRITICAL WARNING: FreeRider {self.id} (Round {current_round_num + 1}) - norm_g_previous is zero. Using default scaling factor of 1.0.")
                scaled_delta_factor = 1.0

            # 计算缩放后的更新
            U_f_flat = scaled_delta_factor * self.last_global_update_for_scaled_delta_flat
            norm_U_f_flat = torch.linalg.norm(U_f_flat)

            # 计算cosine_beta
            expected_cos_beta = self._calculate_expected_cosine_beta(current_round_num)
            current_round_logged_cos_beta = expected_cos_beta

            # 计算噪声幅度
            n = float(num_total_clients_n)
            phi_magnitude = 0.0

            if n > 1:
                denominator_sqrt = n + (n**2 - n) * expected_cos_beta
                phi_magnitude_factor_sqrt_term = np.sqrt((n**2 / denominator_sqrt) - 1.0)
                phi_magnitude = max(0.0, phi_magnitude_factor_sqrt_term * norm_U_f_flat.item())

                # 假设 phi_magnitude 是目标范数
                noise_temp = torch.randn_like(U_f_flat, device=self.device)
                norm_noise_temp = torch.linalg.norm(noise_temp)
                if norm_noise_temp > 1e-9:
                   scaled_noise_vector = (noise_temp / norm_noise_temp) * phi_magnitude
                else:
                   scaled_noise_vector = torch.zeros_like(U_f_flat, device=self.device)
                noise_vector_phi_flat = scaled_noise_vector

                num_dims_to_add_noise = int(flat_ref_params.numel() * self.adv_attack_noise_dim_fraction)
                if 0 < num_dims_to_add_noise < flat_ref_params.numel():
                    indices_to_add_noise = torch.randperm(flat_ref_params.numel(), device=self.device)[:num_dims_to_add_noise]
                    sparse_noise_component = torch.zeros_like(U_f_flat, device=self.device)
                    selected_noise_components = noise_vector_phi_flat[indices_to_add_noise] # Get components from already scaled vector
                    sparse_noise_component.scatter_(0, indices_to_add_noise, selected_noise_components)
                    fabricated_update_flat = U_f_flat + sparse_noise_component
                else:
                    fabricated_update_flat = U_f_flat + noise_vector_phi_flat
            else:
                print(f"CRITICAL WARNING: FreeRider {self.id} (Round {current_round_num + 1}) - n <= 1. Using only U_f_flat.")
                fabricated_update_flat = U_f_flat
        else:
            print(f"CRITICAL WARNING: FreeRider {self.id} (Round {current_round_num + 1}) - unknown attack phase: {self.attack_phase}. Submitting zero update.")
            fabricated_update_flat = torch.zeros_like(flat_ref_params, device=self.device)

        # 计算当前轮次的余弦相似度
        if len(self.expected_cos_beta_history) == current_round_num:
            self.expected_cos_beta_history.append(current_round_logged_cos_beta)
        elif len(self.expected_cos_beta_history) > current_round_num:
            self.expected_cos_beta_history[current_round_num] = current_round_logged_cos_beta
        else:
             while len(self.expected_cos_beta_history) < current_round_num:
                self.expected_cos_beta_history.append(np.nan)
             self.expected_cos_beta_history.append(current_round_logged_cos_beta)

        # 检查更新是否为None
        if fabricated_update_flat is None:
            print(f"CRITICAL WARNING: FreeRider {self.id} (Round {current_round_num + 1}) update is None at end of compute_update. Submitting zero update.")
            fabricated_update_flat = torch.zeros_like(flat_ref_params, device=self.device)
            if len(self.expected_cos_beta_history) <= current_round_num:
                self.expected_cos_beta_history.append(np.nan)
            else:
                self.expected_cos_beta_history[current_round_num] = np.nan


        self.current_update_flat = fabricated_update_flat.detach().clone()
        return self.current_update_flat


# 服务器类
class Server:
    def __init__(self, initial_model_state_dict, honest_clients, free_riders, test_loader, device):
        self.global_model_state_dict = copy.deepcopy(initial_model_state_dict)
        self.honest_clients = honest_clients
        self.free_riders = free_riders
        self.all_clients = honest_clients + free_riders
        random.shuffle(self.all_clients)
        self.test_loader = test_loader
        self.device = device
        
        self.global_model_test_accuracies = []
        self.global_model_test_losses = []
        self.server_g_t_flat_history = []
        
        self.stats_honest_l2_norms = [] 
        self.stats_freerider_l2_norms = []
        self.stats_honest_stds = []
        self.stats_freerider_stds = []
        self.stats_freerider_expected_cos_beta = [[] for _ in free_riders]

        self.stats_honest_to_avg_honest_cosine_sims = []    
        self.stats_freerider_to_avg_honest_cosine_sims = [] 
        self.stats_l2_norm_of_avg_honest_update = [] 
        self.stats_std_of_avg_honest_update = []     


    # 聚合客户端更新，采用简单的FedAvg方法
    def aggregate_updates(self, client_updates_flat, client_weights=None):
        if not client_updates_flat:
            return self.global_model_state_dict

        if client_weights is None:
            client_weights = [1.0 / len(client_updates_flat)] * len(client_updates_flat)
        
        aggregated_update_flat = torch.zeros_like(client_updates_flat[0], device=self.device)
        for weight, update_flat in zip(client_weights, client_updates_flat):
            aggregated_update_flat += weight * update_flat
        
        current_global_flat = flatten_params(self.global_model_state_dict).to(self.device)
        new_global_flat = current_global_flat + aggregated_update_flat
        
        self.server_g_t_flat_history.append(aggregated_update_flat.detach().clone())

        self.global_model_state_dict = unflatten_params(new_global_flat, self.global_model_state_dict)
        return self.global_model_state_dict

    def run_fl_round(self, current_round_num):
        print(f"\n--- Round {current_round_num + 1}/{NUM_ROUNDS} ---")
        
        client_updates_flat_this_round = []      
        honest_client_updates_this_round = []    
        freerider_client_updates_this_round = [] 

        for client in self.all_clients:
            if client.type == "honest":
                update_flat = client.compute_update(copy.deepcopy(self.global_model_state_dict))
                client_updates_flat_this_round.append(update_flat)
                honest_client_updates_this_round.append(update_flat)
            elif client.type == "free_rider":
                num_total_clients = len(self.all_clients)
                update_flat = client.compute_update(current_round_num, 
                                                    copy.deepcopy(self.global_model_state_dict),
                                                    self.server_g_t_flat_history,
                                                    num_total_clients)
                client_updates_flat_this_round.append(update_flat)
                freerider_client_updates_this_round.append(update_flat)
                
                fr_idx = self.free_riders.index(client)
                if client.expected_cos_beta_history:
                     self.stats_freerider_expected_cos_beta[fr_idx].append(client.expected_cos_beta_history[-1])
                else:
                     self.stats_freerider_expected_cos_beta[fr_idx].append(0.0)

        self.aggregate_updates(client_updates_flat_this_round)

        temp_model_for_test = MLP().to(self.device)
        temp_model_for_test.load_state_dict(self.global_model_state_dict)
        accuracy, loss = test_model(temp_model_for_test, self.test_loader, self.device)
        self.global_model_test_accuracies.append(accuracy)
        self.global_model_test_losses.append(loss)
        print(f"Global Model Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")

        round_honest_l2, round_fr_l2 = [], []
        round_honest_std, round_fr_std = [], []
        
        for update in honest_client_updates_this_round:
            round_honest_l2.append(torch.linalg.norm(update).item())
            round_honest_std.append(torch.std(update).item())
        
        for update in freerider_client_updates_this_round:
            round_fr_l2.append(torch.linalg.norm(update).item())
            round_fr_std.append(torch.std(update).item())
            
        self.stats_honest_l2_norms.append(round_honest_l2)
        self.stats_freerider_l2_norms.append(round_fr_l2)
        self.stats_honest_stds.append(round_honest_std)
        self.stats_freerider_stds.append(round_fr_std)

        avg_honest_update_flat = None
        if honest_client_updates_this_round:
            avg_honest_update_flat = torch.stack(honest_client_updates_this_round).mean(dim=0)
            self.stats_l2_norm_of_avg_honest_update.append(torch.linalg.norm(avg_honest_update_flat).item())
            self.stats_std_of_avg_honest_update.append(torch.std(avg_honest_update_flat).item())
        else:
            self.stats_l2_norm_of_avg_honest_update.append(np.nan)
            self.stats_std_of_avg_honest_update.append(np.nan)

        current_round_honest_to_avg_sims = []
        if avg_honest_update_flat is not None and torch.linalg.norm(avg_honest_update_flat) > 1e-9:
            for honest_update in honest_client_updates_this_round:
                if torch.linalg.norm(honest_update) > 1e-9: 
                    sim = torch.nn.functional.cosine_similarity(honest_update, avg_honest_update_flat, dim=0).item()
                    current_round_honest_to_avg_sims.append(sim)
                else:
                    current_round_honest_to_avg_sims.append(0.0) 
        self.stats_honest_to_avg_honest_cosine_sims.append(current_round_honest_to_avg_sims)

        current_round_freerider_to_avg_sims = []
        if avg_honest_update_flat is not None and torch.linalg.norm(avg_honest_update_flat) > 1e-9:
            for fr_update in freerider_client_updates_this_round:
                if torch.linalg.norm(fr_update) > 1e-9:
                    sim = torch.nn.functional.cosine_similarity(fr_update, avg_honest_update_flat, dim=0).item()
                    current_round_freerider_to_avg_sims.append(sim)
                else:
                    current_round_freerider_to_avg_sims.append(0.0) 
        self.stats_freerider_to_avg_honest_cosine_sims.append(current_round_freerider_to_avg_sims)


# 设置随机种子，确保实验可重复
set_seed(RANDOM_SEED)

# 数据集加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 生成客户端和测试数据加载器
client_data_loaders, test_loader = get_mnist_data(
    num_clients=NUM_HONEST_CLIENTS + NUM_FREE_RIDERS,
    iid=IID_DATA_DISTRIBUTION,
    non_iid_alpha=NON_IID_ALPHA,
    client_batch_size=CLIENT_BATCH_SIZE,
    test_batch_size=TEST_BATCH_SIZE
)
if (NUM_HONEST_CLIENTS + NUM_FREE_RIDERS) > 0 and not client_data_loaders:
    raise ValueError("Failed to create client data loaders. Check data distribution logic or client count.")
if (NUM_HONEST_CLIENTS + NUM_FREE_RIDERS) > 0 and len(client_data_loaders) != (NUM_HONEST_CLIENTS + NUM_FREE_RIDERS):
    print(f"Warning: Number of created client loaders ({len(client_data_loaders)}) does not match ({NUM_HONEST_CLIENTS + NUM_FREE_RIDERS}). This might happen if some clients got 0 samples in non-IID.")


if __name__ == "__main__":
    initial_model = MLP().to(DEVICE)
    
    honest_clients_list = []
    if NUM_HONEST_CLIENTS > 0 and client_data_loaders:
        num_actual_honest_clients_with_data = len(client_data_loaders)
        
        # 修改为动态学习率
        honest_clients_list = [
            HonestClient(id=f"h_{i}", data_loader=client_data_loaders[i], 
                         epochs=CLIENT_EPOCHS, lr=random.uniform(0.001, 0.01), device=DEVICE)   
            for i in range(num_actual_honest_clients_with_data)
        ]
        if num_actual_honest_clients_with_data < NUM_HONEST_CLIENTS:
            print(f"Warning: Only {num_actual_honest_clients_with_data} honest clients created due to data distribution.")
    
    free_riders_list = [
        FreeRider(id=f"fr_{i}", data_loader=client_data_loaders[num_actual_honest_clients_with_data + i - 1], epochs=CLIENT_EPOCHS, lr=CLIENT_LR ,device=DEVICE,
                  num_honest_rounds_for_estimation=NUM_HONEST_ROUNDS_FOR_ESTIMATION_FREERIDER,
                  adv_attack_c_param=ADV_ATTACK_C_PARAM,
                  adv_attack_noise_dim_fraction=ADV_ATTACK_NOISE_DIM_FRACTION,
                  )
        for i in range(NUM_FREE_RIDERS)
    ]

    server = Server(initial_model.state_dict(), honest_clients_list, free_riders_list, test_loader, DEVICE)

    for r in range(NUM_ROUNDS):
        server.run_fl_round(r)

    rounds_x = np.arange(1, NUM_ROUNDS + 1)

# --- 绘图逻辑 ---
# 确定攻击阶段开始的索引 (0-based)
attack_phase_start_round_index = NUM_HONEST_ROUNDS_FOR_ESTIMATION_FREERIDER

# 创建攻击阶段的轮次x轴 (1-based for plotting)
# 例如，如果 NUM_ROUNDS = 100, attack_phase_start_round_index = 2,
# 那么我们想绘制的是第3轮到第100轮的数据。
# 对应的x轴刻度是 3, 4, ..., 100
if attack_phase_start_round_index < NUM_ROUNDS:
    rounds_x_attack_phase = np.arange(attack_phase_start_round_index + 1, NUM_ROUNDS + 1)
    num_plot_rounds = len(rounds_x_attack_phase) # 实际绘制的轮数

    plt.figure(figsize=(18, 12))

    # 定义颜色
    honest_client_color = 'royalblue'
    free_rider_color = 'crimson'
    avg_honest_color = 'darkgreen'
    attacker_target_color = 'purple'
    global_metrics_color = 'black'

    # 1. Global Model Accuracy (攻击阶段)
    plt.subplot(2, 2, 1) # 调整为2x2布局
    if hasattr(server, 'global_model_test_accuracies') and len(server.global_model_test_accuracies) > attack_phase_start_round_index:
        plot_data = server.global_model_test_accuracies[attack_phase_start_round_index:NUM_ROUNDS]
        plt.plot(rounds_x_attack_phase, plot_data[:num_plot_rounds], linestyle='-', color=global_metrics_color)
    plt.title(f'Global Model Test Accuracy (Attack Phase: Rounds {attack_phase_start_round_index + 1}-{NUM_ROUNDS})')
    plt.xlabel('Communication Round')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)

    # 2. Global Model Loss (攻击阶段)
    plt.subplot(2, 2, 2) # 调整为2x2布局
    if hasattr(server, 'global_model_test_losses') and len(server.global_model_test_losses) > attack_phase_start_round_index:
        plot_data = server.global_model_test_losses[attack_phase_start_round_index:NUM_ROUNDS]
        # 检查是否有nan或inf，这可能导致绘图问题
        plot_data_cleaned = [x if np.isfinite(x) else np.nan for x in plot_data]
        plt.plot(rounds_x_attack_phase, plot_data_cleaned[:num_plot_rounds], linestyle='-', color=global_metrics_color)
    plt.title(f'Global Model Test Loss (Attack Phase: Rounds {attack_phase_start_round_index + 1}-{NUM_ROUNDS})')
    plt.xlabel('Communication Round')
    plt.ylabel('Loss')
    plt.grid(True)
    # 如果损失值范围很大，可以考虑Y轴对数尺度，但要注意nan和非正值
    # try:
    #     if any(val > 0 for val in plot_data_cleaned if np.isfinite(val)): # Check if there are positive values to plot on log scale
    #         plt.yscale('log')
    # except Exception:
    #     pass # Keep linear scale if log scale fails

    # 3. L2 Norm Comparison (攻击阶段) - 即 "步长"
    plt.subplot(2, 2, 3) # 调整为2x2布局
    # 绘制诚实客户端的L2范数
    if hasattr(server, 'stats_honest_l2_norms') and len(server.stats_honest_l2_norms) > attack_phase_start_round_index:
        first_attack_round_stats = server.stats_honest_l2_norms[attack_phase_start_round_index]
        if first_attack_round_stats is not None: # 确保该轮有数据
            num_honest_clients = len(first_attack_round_stats)
            for i in range(num_honest_clients):
                client_l2_norms = [
                    server.stats_honest_l2_norms[r][i]
                    if r < len(server.stats_honest_l2_norms) and server.stats_honest_l2_norms[r] is not None and i < len(server.stats_honest_l2_norms[r])
                    else np.nan
                    for r in range(attack_phase_start_round_index, NUM_ROUNDS)
                ]
                plt.plot(rounds_x_attack_phase, client_l2_norms[:num_plot_rounds], linestyle='-', color=honest_client_color, alpha=0.8, label='Honest Client L2 Norm' if i == 0 else None)

    # 绘制搭便车者的L2范数
    if hasattr(server, 'stats_freerider_l2_norms') and len(server.stats_freerider_l2_norms) > attack_phase_start_round_index:
        first_attack_round_stats_fr = server.stats_freerider_l2_norms[attack_phase_start_round_index]
        if first_attack_round_stats_fr is not None: # 确保该轮有数据
            num_free_riders = len(first_attack_round_stats_fr)
            for i in range(num_free_riders):
                fr_l2_norms = [
                    server.stats_freerider_l2_norms[r][i]
                    if r < len(server.stats_freerider_l2_norms) and server.stats_freerider_l2_norms[r] is not None and i < len(server.stats_freerider_l2_norms[r])
                    else np.nan
                    for r in range(attack_phase_start_round_index, NUM_ROUNDS)
                ]
                plt.plot(rounds_x_attack_phase, fr_l2_norms[:num_plot_rounds], linestyle='-', color=free_rider_color, alpha=0.8, label='Free-Rider L2 Norm' if i == 0 else None)

    plt.title(f'L2 Norm of Client Updates (Attack Phase: Rounds {attack_phase_start_round_index + 1}-{NUM_ROUNDS})')
    plt.xlabel('Communication Round')
    plt.ylabel('L2 Norm (Step Length)')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')

    # 4. Cosine Similarity with Average Honest Update (攻击阶段)
    plt.subplot(2, 2, 4) # 调整为2x2布局
    # 绘制诚实客户端的余弦相似度
    if hasattr(server, 'stats_honest_to_avg_honest_cosine_sims') and len(server.stats_honest_to_avg_honest_cosine_sims) > attack_phase_start_round_index:
        first_attack_round_stats_sim_h = server.stats_honest_to_avg_honest_cosine_sims[attack_phase_start_round_index]
        if first_attack_round_stats_sim_h is not None:
            num_honest_clients_for_sim = len(first_attack_round_stats_sim_h)
            for i in range(num_honest_clients_for_sim):
                client_sims = [
                    server.stats_honest_to_avg_honest_cosine_sims[r][i]
                    if r < len(server.stats_honest_to_avg_honest_cosine_sims) and server.stats_honest_to_avg_honest_cosine_sims[r] is not None and i < len(server.stats_honest_to_avg_honest_cosine_sims[r])
                    else np.nan
                    for r in range(attack_phase_start_round_index, NUM_ROUNDS)
                ]
                plt.plot(rounds_x_attack_phase, client_sims[:num_plot_rounds], linestyle='-', color=honest_client_color, alpha=0.8, label='Honest Client vs. Avg. Honest' if i == 0 else None)

    # 绘制搭便车者的余弦相似度
    if hasattr(server, 'stats_freerider_to_avg_honest_cosine_sims') and len(server.stats_freerider_to_avg_honest_cosine_sims) > attack_phase_start_round_index:
        first_attack_round_stats_sim_fr = server.stats_freerider_to_avg_honest_cosine_sims[attack_phase_start_round_index]
        if first_attack_round_stats_sim_fr is not None:
            num_free_riders_for_sim = len(first_attack_round_stats_sim_fr)
            for i in range(num_free_riders_for_sim):
                fr_sims = [
                    server.stats_freerider_to_avg_honest_cosine_sims[r][i]
                    if r < len(server.stats_freerider_to_avg_honest_cosine_sims) and server.stats_freerider_to_avg_honest_cosine_sims[r] is not None and i < len(server.stats_freerider_to_avg_honest_cosine_sims[r])
                    else np.nan
                    for r in range(attack_phase_start_round_index, NUM_ROUNDS)
                ]
                plt.plot(rounds_x_attack_phase, fr_sims[:num_plot_rounds], linestyle='-', color=free_rider_color, alpha=0.8, label='Free-Rider vs. Avg. Honest' if i == 0 else None)

    plt.title(f'Cosine Similarity (Attack Phase: Rounds {attack_phase_start_round_index + 1}-{NUM_ROUNDS})')
    plt.xlabel('Communication Round')
    plt.ylabel('Cosine Similarity Value')
    plt.ylim(0, 1.1) # 根据代码，Y轴下限为0
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("federated_learning_attack_phase_analysis.png")
    print(f"\nPlot saved as federated_learning_attack_phase_analysis.png (showing data from round {attack_phase_start_round_index + 1})")
    plt.show()

else:
    print(f"Not enough rounds for attack phase plotting. Estimation rounds: {NUM_HONEST_ROUNDS_FOR_ESTIMATION_FREERIDER}, Total rounds: {NUM_ROUNDS}")

print("\n--- Final Parameters for Free Rider (if any) ---")
if hasattr(server, 'free_riders'):
    for fr in server.free_riders:
        if hasattr(fr, 'id') and hasattr(fr, 'estimated_lambda_bar'):
            print(f"Free Rider ID: {fr.id}")
            print(f"  Final Estimated Lambda_bar: {fr.estimated_lambda_bar}")
