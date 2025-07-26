import click
import json
import os

def walk_dict(d):
    for key, value in d.items():
        if type(value) == dict:
            for subkey, subvalue in walk_dict(value):
                yield key + "." + subkey, subvalue
        else:
            yield key, value
transformer_rename = {
    "train_envs"                         : "number of training environments"      ,
    "train_steps"                        : "training horizon"                     ,
    "eval_envs"                          : "number of evaluation environments"              ,
    "eval_steps"                         : "evaluation horizon"                   ,
    "policy_layers"                      : "policy layers"                        ,
    "policy_hidden_size"                 : "policy hidden size"                   ,
    "policy_feedforward"                 : "policy feedforward size"              ,
    "policy_heads"                       : "policy heads"                         ,
    "policy_dropout"                     : "policy dropout"                       ,
    "policy_activation"                  : "policy activation"                    ,
    "policy_var"                         : "policy variance"                      ,
    "value_layers"                       : "value layers"                         ,
    "value_hidden_size"                  : "value hidden size"                    ,
    "value_feedforward"                  : "value feedforward size"               ,
    "value_dropout"                      : "value dropout"                        ,
    "value_activation"                   : "value activation"                     ,
    "reward_layers"                      : "reward layers"                        ,
    "reward_hidden_size"                 : "reward hidden size"                   ,
    "reward_feedforward"                 : "reward feedforward size"              ,
    "reward_dropout"                     : "reward dropout"                       ,
    "reward_activation"                  : "reward activation"                    ,
    "world_layers"                       : "world layers"                         ,
    "world_hidden_size"                  : "world hidden size"                    ,
    "world_feedforward"                  : "world feedforward size"               ,
    "world_dropout"                      : "world dropout"                        ,
    "world_activation"                   : "world activation"                     ,
    "policy_learning_rate"               : "policy learning rate"                 ,
    "value_learning_rate"                : "value learning rate"                  ,
    "reward_learning_rate"               : "reward learning rate"                 ,
    "policy_clip_coefficient"            : "policy clip coefficient"              ,
    "value_clip_coefficient"             : "value clip coefficient"               ,
    "reward_clip_coefficient"            : "reward clip coefficient"              ,
    "reward_cache_size"                  : "reward cache size"                    ,
    "value_cache_size"                   : "value cache size"                     ,
    "world_cache_size"                   : "world cache size"                     ,
    "reward_batch_size"                  : "world/value/reward batch size"        ,
    "value_batch_size"                   : "world/value/reward batch size"        ,
    "world_batch_size"                   : "world/value/reward batch size"        ,
    "world_ett"                          : "world/value/reward cooldown epochs"   ,
    "value_ett"                          : "world/value/reward cooldown epochs"   ,
    "reward_ett"                         : "world/value/reward cooldown epochs"   ,
    "world_bins"                         : "world cache bins"        ,
    "value_bins"                         : "value cache bins"        ,
    "reward_bins"                        : "reward cache bins"        ,
    "out_coefficient"                    : "$\\alpha$"                            ,
    "early_stopping.max_reward_fraction" : "early stopping - max reward fraction" ,
    "early_stopping.max_envs_fraction"   : "early stopping - max envs fraction"   ,
    "seed"                               : "seed"                                 ,
    "episodes"                           : "max episodes"                         ,
    "etr"                                : "epochs between environment resets"    ,
    "etv"                                : "epochs between evaluations"           ,
    "lambda_factor"                      : "$\\lambda$"                           ,
    "gamma_factor"                       : "$\\gamma$"                            ,
}

transformer_only = {
    "policy_feedforward",
    "value_feedforward",
    "reward_feedforward",
    "policy_heads",
    "value_heads",
    "reward_heads",
}


@click.command()
@click.option("--input-path"  , "input_path"  , default=click.Path(), help="Path to the directory")
@click.option("--output-path" , "output_path" , default=click.Path(), help="Path to the output file")
def run(input_path, output_path):
    transformer_allparams = {}
    mlp_allparams = {}
    for root, dirs, files in os.walk(input_path):
        for file in filter(lambda x: x == "locals.json", files):
            currentparams = json.load(open(os.path.join(root, file)))
            allparams = transformer_allparams if "transformer" in root else mlp_allparams
            for key, value in walk_dict(currentparams):
                if type(value) == list:
                    value = tuple(value)
                elif key not in allparams:
                    allparams[key] = value
                elif key in allparams and type(allparams[key]) == set:
                    allparams[key].add(value)
                elif key in allparams and type(allparams[key]) != set and value != allparams[key]:
                    allparams[key] = set([allparams[key], value])
                elif key in allparams and type(allparams[key]) != set and value == allparams[key]:
                    pass
                else:
                    print(f"key: {key}, value: {value}, allparams[key]: {allparams[key]}")
                    raise ValueError("Unexpected value")
            
    # rename keys
    transformer_allparams = {transformer_rename[key]: value for key, value in transformer_allparams.items() if key in transformer_rename}
    mlp_allparams = {transformer_rename[key]: value for key, value in mlp_allparams.items() if key in transformer_rename and key not in transformer_only}

    # to latex table
    file = open(output_path, "w")
    file.write("\\begin{tabular}{ l l l}\n")
    file.write("\t\\toprule\n")
    file.write("\tParameter & Transformer & MLP \\\\\n")
    file.write("\t\\midrule\n")

    for key in transformer_allparams:
        file.write(f"\t{key} & {transformer_allparams[key]} & {mlp_allparams[key] if key in mlp_allparams else '-'} \\\\\n")

    file.write("\t\\bottomrule\n")
    file.write("\\end{tabular}\n")



if __name__ == "__main__":
    run()
