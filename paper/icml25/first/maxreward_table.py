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

    # aggregate rewards by seed
    path2max_by_seed = {}
    for path,reward in path2max.items():
        seed = path.split("/")[-2]
        path_no_seed = os.path.join(*path.split("/")[:-2],path.split("/")[-1])
        if path_no_seed not in path2max_by_seed:
            path2max_by_seed[path_no_seed] = reward
        else:
            path2max_by_seed[path_no_seed] = max(path2max_by_seed[path_no_seed], reward)

    # drop all transformer with 1 agent
    path2max_by_seed = {path:reward if path2agents(path) > 1 or path2architecture(path) == "mlp" else 0 for path,reward in path2max_by_seed.items() }

    # normalize by algorithm max
    algorithm2max = {}
    for path in path2max_by_seed:
        algorithm = path.split("/")[-4]
        if algorithm not in algorithm2max:
            algorithm2max[algorithm] = path2max_by_seed[path]
        else:
            algorithm2max[algorithm] = max(algorithm2max[algorithm], path2max_by_seed[path])
    path2max_by_seed_normalized = {}
    for path in path2max_by_seed:
        algorithm = path.split("/")[-4]
        path2max_by_seed_normalized[path] = path2max_by_seed[path] / algorithm2max[algorithm]

    # write latex table
    with open(output_path, "w") as f:
        f.write("\\begin{tabular}{ c c c c c c c c c c }\n")
        f.write("\\toprule\n")
        f.write("\\multirow{2}{*}{Environment} & \\multirow{2}{*}{agents} & PPO & SHAC & SHAC+ & SHAC++ & PPO & SHAC & SHAC+ & SHAC++ \\\\\n")
        f.write(" & & \\multicolumn{4}{c}{MLP} & \\multicolumn{4}{c}{Transformer} \\\\\n")
        f.write("\\midrule\n")
        for scenario in scenarios:
            f.write("\\multirow{3}{*}{" + scenario + "}")
            for agents_ in agents:
                f.write(f"& {agents_}")
                for architecture in architectures:
                    for algorithm in algorithms:
                        path = f"{input_path}/{algorithm}/{scenario}/{agents_}/{architecture}/eval.log"
                        if path not in path2max_by_seed_normalized:
                            f.write(" & -")
                        elif architecture == "transformer" and agents_ == 1:
                            f.write(" & -")
                        elif path in path2max_by_seed_normalized:
                            value = path2max_by_seed_normalized[path]
                            f.write(" & \\textbf{"+f"{value:.2f}"+"}" if value == 1 else f" & {value:.2f}")
                        else:
                            raise ValueError("Path not found")
                        

                f.write(" \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


if __name__ == "__main__":
    maxreward_table()
