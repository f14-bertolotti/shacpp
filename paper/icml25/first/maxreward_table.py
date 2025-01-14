import pprint
import click
import json
import os

architectures = ["mlp", "transformer"]
scenarios = ["dispersion", "transport", "discovery", "sampling"]
algorithms = ["ppo", "shac", "shacrm", "shacwm"]
agents = [1, 3, 5]

def path2algorithm(path):
    return path.split("/")[1]

def path2scenario(path):
    return path.split("/")[2]

def path2agents(path):
    return int(path.split("/")[3])

def path2architecture(path):
    return path.split("/")[4]

def path2seed(path):
    return int(path.split("/")[5])

@click.command()
@click.option("--input-path"  , "input_path"  , default="data"   , help='Path to the data')
@click.option("--output-path" , "output_path" , default="output" , help='Path to the output')
def maxreward_table(input_path, output_path):
    # get all eval.log
    path2data = {}
    for root, dirs, files in os.walk(input_path):
        for file in files:
            path = os.path.join(root, file)
            if file == "eval.log":
                # read jsonline eval.log
                with open(path, "r") as f:
                    data = [json.loads(line) for line in f]
                    path2data[path] = data


    # get max reward
    path2max = {}
    for path, data in path2data.items():
        rewards = [d["message"]["reward"] for d in data]
        path2max[path] = max(rewards)

    # aggregate by seed
    path2max_by_seed = {}
    for path, rew in path2max.items():
        agent = path2agents(path)
        scenario = path2scenario(path)
        arch = path2architecture(path)
        algo = path2algorithm(path)
        key = (arch, agent, scenario, algo)
        if key not in path2max_by_seed:
            path2max_by_seed[key] = rew
        else:
            path2max_by_seed[key] = max(path2max_by_seed[key], rew)

    # get max in scenario-agents
    path2reference = {}
    for path, rew in path2max.items():
        scenario = path2scenario(path)
        agent = path2agents(path)
        #architecture = path2architecture(path)
        #key = (architecture, agent, scenario)
        key = (agent, scenario)
        if key not in path2reference:
            path2reference[key] = rew
        else:
            path2reference[key] = max(path2reference[(key)], rew)

    pprint.pprint(path2max_by_seed)
    print()
    pprint.pprint(path2reference)
    print()

    # normalize wrt reference
    path2max_by_seed_norm = {}
    for path, rew in path2max.items():
        scenario = path2scenario(path)
        agent = path2agents(path)
        arch = path2architecture(path)
        algo = path2algorithm(path)
        path2max_by_seed_norm[(arch, agent, scenario, algo)] = path2max_by_seed[(arch, agent, scenario, algo)] / path2reference[(agent, scenario)]

    pprint.pprint(path2max_by_seed_norm)

    # write latex table
    with open(output_path, "w") as f:
        f.write("\\begin{tabular}{ c c c c c c c c c c }\n")
        f.write("\\toprule\n")
        f.write("\\multirow{2}{*}{Environment} & \\multirow{2}{*}{agents} & PPO & SHAC & SHAC+ & SHAC++ & PPO & SHAC & SHAC+ & SHAC++ \\\\\n")
        f.write(" & & \\multicolumn{4}{c}{MLP} & \\multicolumn{4}{c}{Transformer} \\\\\n")
        f.write("\\midrule\n")
        for scenario in scenarios:
            f.write("\\multirow{3}{*}{" + scenario + "}")
            for agent in agents:
                f.write(f"& {agent}")
                for architecture in architectures:
                    for algorithm in algorithms:
                        key = (architecture, agent, scenario, algorithm)
                        if key not in path2max_by_seed_norm:
                            f.write(" & -")
                        elif architecture == "transformer" and agent == 1:
                            f.write(" & -")
                        elif key in path2max_by_seed_norm:
                            value = path2max_by_seed_norm[key]
                            f.write(" & \\textbf{"+f"{value:.2f}"+"}" if value >= .999 else f" & {value:.2f}")
                        else:
                            raise ValueError("Path not found")
                f.write(" \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


if __name__ == "__main__":
    maxreward_table()
