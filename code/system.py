import torch.nn as nn
import torch
import copy
import random
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset


def get_mnist_data(num_clients, iid, non_iid_alpha):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('/data/ddh/data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('/data/ddh/data', train=False, download=True, transform=transform)
    client_datasets = []
    if iid or num_clients == 0:
        if num_clients == 0: return [], test_dataset
        total_size = len(train_dataset)
        indices = list(range(total_size))
        random.shuffle(indices)
        split_size = total_size // num_clients if num_clients > 0 else total_size
        for i in range(num_clients):
            start_idx, end_idx = i * split_size, (i + 1) * split_size if i < num_clients - 1 else total_size
            if not indices[start_idx:end_idx]: continue
            client_datasets.append(Subset(train_dataset, indices[start_idx:end_idx]))
    # Non-IID
    else:   
        labels = np.array(train_dataset.targets)
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
                for _ in range(abs(delta)): target_samples_for_class_c[random.randrange(num_clients)] += np.sign(delta)
            current_idx = 0
            for client_id in range(num_clients):
                num_samples = target_samples_for_class_c[client_id]
                client_indices_map[client_id].extend(indices_c[current_idx : current_idx + num_samples])
                current_idx += num_samples
        for i in range(num_clients):
            if not client_indices_map[i] or len(client_indices_map[i]) < min_size_per_client // num_classes :
                rand_indices = random.sample(range(len(train_dataset)), min_size_per_client)
                client_datasets.append(Subset(train_dataset, rand_indices))
            else:
                client_datasets.append(Subset(train_dataset, client_indices_map[i]))
    return client_datasets, test_dataset


class Global_Model(nn.Module):
    def __init__(self):
        super(Global_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)


    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


    def deepcopy(self):
        return copy.deepcopy(self)
    

class Participant:
    def __init__(self, id, type, ini_rep, tra_round_num, device):
        self.id = id
        self.type = type
        self.reputation = ini_rep
        self.selected = False
        self.fail_num = 0
        self.reputation_history = [ini_rep] * tra_round_num
        self.bid = {}
        self.device = device
        self.tra_round_num = tra_round_num
        self.model = None


    def update_reputation_history(self):
        if len(self.reputation_history) >= self.tra_round_num: self.reputation_history.pop(0)
        self.reputation_history.append(self.reputation)


    def get_delta_reputation(self):
        if not self.reputation_history: return 0.0
        return abs(self.reputation - self.reputation_history[0])


    def set_model_state(self, state_dict):
        if self.model is None: self.model = Global_Model().to(self.device)
        try: self.model.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"错误：参与者 {self.id} 加载模型状态失败: {e}")
            self.model = Global_Model().to(self.device)
            self.model.load_state_dict(state_dict)


    # 展平参数为一维张量
    def _flatten_params(self, model_state_dict):
        return torch.cat([p.view(-1) for p in model_state_dict.values()])


    # 将一维张量还原为模型参数字典
    def _unflatten_params(self, flat_params, model_state_dict):
        new_state_dict = copy.deepcopy(model_state_dict)
        current_pos = 0
        for key in new_state_dict:
            param = new_state_dict[key]
            num_elements = param.numel()
            new_state_dict[key] = flat_params[current_pos : current_pos + num_elements].view_as(param)
            current_pos += num_elements
        return new_state_dict


class Requester:
    def __init__(self, initial_global_model, test_loader, device,
                 alpha_reward, beta_penalty_base):
        self.global_model = initial_global_model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.previous_global_model_state_flat = None
        self.global_model_param_diff_history = []
        self.alpha_reward = alpha_reward
        self.beta_penalty_base = beta_penalty_base


    def _flatten_params(self, model_state_dict):
        return torch.cat([p.view(-1) for p in model_state_dict.values()]).cpu()


    def update_global_model_history(self):
        current_global_model_state_flat = self._flatten_params(self.global_model.state_dict())
        if self.previous_global_model_state_flat is not None:
            g_t_flat = current_global_model_state_flat - self.previous_global_model_state_flat
            if torch.linalg.norm(g_t_flat) > 1e-6 :
                self.global_model_param_diff_history.append(g_t_flat)
                if len(self.global_model_param_diff_history) > 10:
                    self.global_model_param_diff_history.pop(0)
        self.previous_global_model_state_flat = current_global_model_state_flat.clone()


    def evaluate_global_model(self):
        self.global_model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.global_model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        if total == 0: return 0.0, float('inf')
        return correct / total, total_loss / total


    def select_participants(self, participants, M_t, reputation_threshold):
        active_bidders_info = []
        for p in participants:
            if p.reputation >= reputation_threshold and p.bid and 'promise' in p.bid and 'reward' in p.bid:
                promise, reward = p.bid['promise'], p.bid['reward']
                cost_effectiveness_ratio = promise / reward if reward > 1e-6 else (float('inf') if promise > 0 else float('-inf'))
                active_bidders_info.append({"participant": p, "ratio": cost_effectiveness_ratio})
        active_bidders_info.sort(key=lambda x: x["ratio"], reverse=True)
        selected_participants_obj = [info["participant"] for info in active_bidders_info[:M_t]]
        for p in participants: p.selected = (p in selected_participants_obj)
        return selected_participants_obj


    def verify_and_aggregate_updates(self, selected_participants_updates, current_global_accuracy):
        verification_outcomes = []
        valid_param_diffs_for_aggregation = []
        current_global_model_state = copy.deepcopy(self.global_model.state_dict())
        
        submitted_gradient_details = []

        for item in selected_participants_updates:
            participant = item["participant"]
            submitted_update_content = item["update"] 
            

            model_to_evaluate_this_update = Global_Model().to(self.device)
            model_to_evaluate_this_update.load_state_dict(current_global_model_state)
 
            gradient_detail_entry = {
                "participant_id": participant.id,
                "participant_type": participant.type,
                "gradient_dict": None,
                "verified": False,
                "error_processing": False 
            }
            
            if submitted_update_content is None:
                print(f"Error: Participant {participant.id} submitted a None gradient.")
                gradient_detail_entry["error_processing"] = True
            else:
                try:
                    with torch.no_grad():
                        for name, param_diff_val in submitted_update_content.items():
                            if name in model_to_evaluate_this_update.state_dict(): # Ensure key exists
                                model_to_evaluate_this_update.state_dict()[name].add_(param_diff_val.to(self.device))
                    gradient_detail_entry["gradient_dict"] = copy.deepcopy(submitted_update_content)
                except Exception as e:
                    print(f"Error applying free_rider {participant.id} gradient for evaluation: {e}")
                    gradient_detail_entry["error_processing"] = True
            
            submitted_gradient_details.append(gradient_detail_entry)

            if gradient_detail_entry["error_processing"]:
                 verification_outcomes.append({
                     "participant_id": participant.id, 
                     "successful_verification": False, 
                     "observed_increase": -float('inf')
                })
                 continue

            acc_after_update, _ = self.evaluate_model_on_temp(model_to_evaluate_this_update)
            observed_increase = acc_after_update - current_global_accuracy
            promised_increase = participant.bid.get('promise', 0)
            
            successful_verification = False
            if observed_increase >= promised_increase:
                successful_verification = True
            
            gradient_detail_entry["verified"] = successful_verification
            verification_outcomes.append({
                "participant_id": participant.id,
                "successful_verification": successful_verification,
                "observed_increase": observed_increase
            })

            if successful_verification and observed_increase > 1e-4 and gradient_detail_entry["gradient_dict"] is not None:
                valid_param_diffs_for_aggregation.append((gradient_detail_entry["gradient_dict"], observed_increase))
        
        if valid_param_diffs_for_aggregation:
            total_positive_observed_increase_sum = sum(w for _, w in valid_param_diffs_for_aggregation if w > 0)
            if total_positive_observed_increase_sum > 1e-6:
                aggregated_state_diff = {key: torch.zeros_like(param_val, device=self.device, dtype=torch.float32)
                                         for key, param_val in current_global_model_state.items()}
                for param_diff_dict_agg, weight_agg in valid_param_diffs_for_aggregation:
                    if weight_agg <=0: continue
                    for key in aggregated_state_diff:
                        if key in param_diff_dict_agg:
                             aggregated_state_diff[key] += param_diff_dict_agg[key].to(self.device) * weight_agg
                
                final_aggregated_diff = {key: val / total_positive_observed_increase_sum for key, val in aggregated_state_diff.items()}
                
                with torch.no_grad():
                    new_global_state = copy.deepcopy(self.global_model.state_dict())
                    for key in new_global_state:
                        if key in final_aggregated_diff:
                            new_global_state[key] += final_aggregated_diff[key]
                    self.global_model.load_state_dict(new_global_state)
                    
        return verification_outcomes, submitted_gradient_details


    def evaluate_model_on_temp(self, temp_model_instance):
        temp_model_instance.eval()
        correct, total = 0, 0
        if not self.test_loader or len(self.test_loader.dataset) == 0:
            return 0.0, float('inf')
            
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = temp_model_instance(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return (correct / total) if total > 0 else 0.0, 0.0


    def update_reputations_and_pay(self, participants, verification_outcomes, total_rewards_paid_ref):
        for outcome in verification_outcomes:
            participant = next((p for p in participants if p.id == outcome["participant_id"]), None)
            if not participant or not participant.selected: continue

            if participant.type == "honest_client":
                participant.update_adaptive_commitment_stats(outcome["successful_verification"])

            if outcome["successful_verification"]:
                participant.reputation += self.alpha_reward
                reward_to_pay = participant.bid.get('reward', 0)
                total_rewards_paid_ref[0] += reward_to_pay
            else:
                participant.fail_num +=1
                penalty = self.beta_penalty_base ** participant.fail_num
                participant.reputation = max(0, participant.reputation - penalty)
            participant.update_reputation_history()