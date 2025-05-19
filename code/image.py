import json
import matplotlib.pyplot as plt
import numpy as np

# 1. 将您提供的JSON数据粘贴到这里
# (为了简洁，我将使用您之前提供的数据结构，并假设它已被加载到变量 `json_data_string` 中)
json_data_string = """
{
    "optimization_summary": {
        "total_solutions_on_pareto_front": 1,
        "optimization_time_minutes": 279.8152054309845,
        "base_sim_params_used_for_opt": {
            "N": 10,
            "N_f": 3,
            "T_max": 100,
            "iid_data": true,
            "non_iid_alpha": 0.5,
            "omega_m_update": 0.4,
            "initial_reputation": 10.0,
            "reputation_threshold": 0.01,
            "min_performance_constraint": 0.9,
            "target_accuracy_threshold": 0.9,
            "num_honest_rounds_for_fr_estimation": 2,
            "adv_attack_c_param_fr": 0.5,
            "adv_attack_noise_dim_fraction_fr": 1.0,
            "adv_attack_scaled_delta_noise_std": 0.001,
            "bid_gamma_honest": 1.0,
            "local_epochs_honest": 1,
            "lr_honest": 0.001,
            "batch_size_honest": 64,
            "adaptive_bid_adjustment_intensity_gamma_honest": 0.15,
            "adaptive_bid_max_adjustment_delta_honest": 0.4,
            "min_commitment_scaling_factor_honest": 0.2,
            "verbose": false,
            "PymooOpt": true
        },
        "pymoo_algorithm_details": {
            "name": "NSGA2",
            "pop_size": 10,
            "termination_criterion": "<pymoo.termination.max_gen.MaximumGenerationTermination object at 0x7f8e2f511d30>"
        }
    },
    "pareto_solutions_details": [
        {
            "solution_index": 1,
            "parameters": {
                "alpha_reward": 1.2738997396307927,
                "beta_penalty_base": 1.3011395493790225,
                "q_rounds_rep_change": 2
            },
            "objectives": {
                "C_elim_fr": 6.313480086270032,
                "FPR_final": 0.0
            },
            "constraint_violation": -0.0013999999999999568,
            "termination_round_sim": 22,
            "final_model_accuracy_sim": 0.9014,
            "client_reputation_history_per_round": {
                "h_0": [10.0, 8.698860450620977, 9.972760190251769, 9.972760190251769, 9.972760190251769, 9.972760190251769, 9.972760190251769, 9.972760190251769, 9.972760190251769, 9.972760190251769, 11.246659929882561, 12.520559669513354, 13.794459409144146, 15.068359148774938, 15.068359148774938, 15.068359148774938, 15.068359148774938, 16.34225888840573, 17.616158628036523, 18.890058367667315, 18.890058367667315, 18.890058367667315, 18.890058367667315],
                "f_0": [10.0, 10.0, 10.0, 11.273899739630792, 9.972760190251769, 8.279796063293523, 6.077013482028221, 3.210885946860725, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "h_6": [10.0, 11.273899739630792, 11.273899739630792, 11.273899739630792, 11.273899739630792, 11.273899739630792, 11.273899739630792, 11.273899739630792, 11.273899739630792, 11.273899739630792, 12.547799479261585, 13.821699218892377, 12.520559669513354, 12.520559669513354, 12.520559669513354, 13.794459409144146, 15.068359148774938, 16.34225888840573, 16.34225888840573, 16.34225888840573, 14.649294761447484, 15.923194501078276, 15.923194501078276],
                "f_1": [10.0, 10.0, 10.0, 11.273899739630792, 9.972760190251769, 8.279796063293523, 6.077013482028221, 3.210885946860725, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "h_3": [10.0, 11.273899739630792, 12.547799479261585, 12.547799479261585, 12.547799479261585, 12.547799479261585, 12.547799479261585, 12.547799479261585, 12.547799479261585, 12.547799479261585, 12.547799479261585, 12.547799479261585, 12.547799479261585, 13.821699218892377, 15.09559895852317, 15.09559895852317, 15.09559895852317, 15.09559895852317, 15.09559895852317, 15.09559895852317, 13.794459409144146, 13.794459409144146, 15.068359148774938],
                "h_5": [10.0, 10.0, 8.698860450620977, 8.698860450620977, 8.698860450620977, 8.698860450620977, 8.698860450620977, 8.698860450620977, 8.698860450620977, 9.972760190251769, 9.972760190251769, 9.972760190251769, 9.972760190251769, 9.972760190251769, 11.246659929882561, 11.246659929882561, 11.246659929882561, 11.246659929882561, 11.246659929882561, 11.246659929882561, 11.246659929882561, 11.246659929882561, 9.553695802924315],
                "f_2": [10.0, 10.0, 10.0, 11.273899739630792, 9.972760190251769, 8.279796063293523, 6.077013482028221, 3.210885946860725, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "h_2": [10.0, 10.0, 8.698860450620977, 8.698860450620977, 8.698860450620977, 8.698860450620977, 8.698860450620977, 8.698860450620977, 8.698860450620977, 9.972760190251769, 9.972760190251769, 11.246659929882561, 12.520559669513354, 10.827595542555107, 12.1014952821859, 12.1014952821859, 13.375395021816692, 14.649294761447484, 12.446512180182182, 13.720411919812975, 13.720411919812975, 14.994311659443767, 14.994311659443767],
                "h_1": [10.0, 8.698860450620977, 8.698860450620977, 8.698860450620977, 8.698860450620977, 8.698860450620977, 8.698860450620977, 8.698860450620977, 8.698860450620977, 9.972760190251769, 11.246659929882561, 11.246659929882561, 11.246659929882561, 11.246659929882561, 11.246659929882561, 9.553695802924315, 9.553695802924315, 9.553695802924315, 9.553695802924315, 7.350913221659013, 7.350913221659013, 8.624812961289805, 8.624812961289805],
                "h_4": [10.0, 8.698860450620977, 8.698860450620977, 8.698860450620977, 8.698860450620977, 8.698860450620977, 8.698860450620977, 8.698860450620977, 8.698860450620977, 8.698860450620977, 8.698860450620977, 8.698860450620977, 8.698860450620977, 8.698860450620977, 8.698860450620977, 7.005896323662731, 8.279796063293524, 8.279796063293524, 9.553695802924317, 9.553695802924317, 10.827595542555109, 10.827595542555109, 12.101495282185901]
            }
        }
    ]
}
"""

# 2. 解析JSON数据
try:
    data = json.loads(json_data_string)
except json.JSONDecodeError as e:
    print(f"JSON解析错误: {e}")
    exit()

# 3. 提取声誉历史数据
# 假设我们只关心帕累托前沿上的第一个（也是唯一一个）解
if not data["pareto_solutions_details"]:
    print("错误：JSON数据中没有找到帕累托解的详情。")
    exit()

try:
    solution_details = data["pareto_solutions_details"][0]
    reputation_history = solution_details["client_reputation_history_per_round"]
    termination_round_sim = solution_details.get("termination_round_sim", "未知") # 获取模拟终止轮数
except (KeyError, IndexError) as e:
    print(f"提取声誉数据时出错: {e} - 请确保JSON结构正确。")
    exit()


# 4. 准备数据并绘图
plt.figure(figsize=(14, 8)) # 图表大小

# 定义颜色和线型
client_styles = {
    'honest': {'color': 'royalblue', 'linestyle': '-', 'marker': '.'},
    'free_rider': {'color': 'orangered', 'linestyle': '--', 'marker': 'x'},
    'unknown': {'color': 'grey', 'linestyle': ':', 'marker': 'o'}
}

# 绘制每个客户端的声誉变化曲线
for client_id, reputation_list in reputation_history.items():
    num_rounds_data = len(reputation_list)
    rounds = np.arange(num_rounds_data)  # 横坐标为轮数 (从0开始)

    style_key = 'unknown'
    if client_id.startswith('h_'):
        style_key = 'honest'
    elif client_id.startswith('f_'):
        style_key = 'free_rider'

    style = client_styles[style_key]
    plt.plot(rounds, reputation_list,
             marker=style['marker'],
             linestyle=style['linestyle'],
             color=style['color'],
             label=client_id, # 为每个客户端添加标签
             alpha=0.9,
             linewidth=1.5 if style_key != 'free_rider' else 2.0) # 搭便车者线加粗一点

# 5. 添加图表细节
plt.title(f"Client Reputation Changes Over Rounds (Simulation terminated at round {termination_round_sim})", fontsize=16)
plt.xlabel("Simulation Round", fontsize=14)
plt.ylabel("Reputation Score", fontsize=14)
plt.xticks(np.arange(0, num_rounds_data, step=max(1, num_rounds_data // 20))) # 调整x轴刻度密度
plt.yticks(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)

# 图例放置在图表外部右侧
plt.legend(title="Client ID", loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9, ncol=1)

plt.tight_layout(rect=[0, 0, 0.85, 1]) # 为外部图例调整布局，防止图例被截断

# 6. 显示图表
# 如果在Jupyter Notebook等环境中，可以直接显示。如果是在脚本中运行，plt.show()会打开一个窗口。
plt.show()

# 7. 可选：保存图表到文件
# plt.savefig("client_reputation_plot.png", dpi=300, bbox_inches='tight')
# print("图表已保存为 client_reputation_plot.png")