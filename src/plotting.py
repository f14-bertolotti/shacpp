import os, json, torch, logging, random, numpy
import matplotlib.pyplot as plt

def plot_rewards(data_path:str, output_path:str):
    data = {}
    for approach in os.listdir(data_path):
        approach_path = os.path.join(data_path, approach)
        if not os.path.isdir(approach_path): continue
        approach_name, env, agents = approach.split("-")
        log_path = os.path.join(approach_path, "eval.log")
        if not os.path.exists(log_path): continue
        if approach_name not in data: data[approach_name] = {}
        if env not in data[approach_name]: data[approach_name][env] = {}
        data[approach_name][env][agents] = []
        with open(log_path, "r") as log_file:
            for line in log_file:
                line = json.loads(line)
                data[approach_name][env][agents].append(line["message"])
    for approach in data:
        for env in data[approach]:
            for agents in data[approach][env]:
                rewards = torch.tensor([r["reward"] for r in data[approach][env][agents]])
                plt.plot(rewards, label=f"{approach} - {env} - {agents}")
            plt.legend()
            if not os.path.exists(output_path): os.makedirs(output_path)
            plt.savefig(f"{output_path}/{approach}_{env}.png")
            plt.clf()

base_folder = os.path.dirname(".")
data_folder = os.path.join(base_folder, "data")
charts_folder = os.path.join(base_folder, "charts")
print(data_folder, charts_folder)
plot_rewards(data_folder, charts_folder)