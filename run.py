import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import agents
from agents import Homeowner
from model import DroughtRiskModel
from pathlib import Path
import itertools


# run the model for a policy scenario and return group-level summary DataFrame
def run_model(policy_name, insurance=False, subsidy=False, nudge=False, steps=30, seeds = range(50), sensitivity_params = None):
    # set global policy flags used by agents

    results = []
    histories = []

    if sensitivity_params is None:
        sensitivity_params = {}


    # build full parameter grid (for sensitivity analysis)    
    param_keys = list(sensitivity_params.keys())    
    param_values = list(sensitivity_params.values())    
    if param_keys:        
        param_combinations = [            
            dict(zip(param_keys, values))            
            for values in itertools.product(*param_values)]    
    else:
        param_combinations = [{}]

    total_runs = len(seeds) * len(param_combinations)
    current_run = 0

    for seed in seeds:
        for param_set in param_combinations:
            current_run += 1        
            print(f"Running {current_run}/{total_runs} | seed={seed} | params={param_set}")
            agents.Policy_Insurance = insurance
            agents.Policy_Subsidy = subsidy
            agents.Policy_Nudge = nudge

    # create and run the model
            model = DroughtRiskModel(seed = seed, insurance = insurance, subsidy = subsidy, nudge=nudge, **param_set)
            for _ in range(steps):
                model.step()

            df_hist = model.get_homeowner_history()
            
            df_hist["seed"]=seed
            df_hist["policy"] = policy_name

            for k, v in param_set.items():
                df_hist[k] = v

            histories.append(df_hist)

            
            # ensure final metrics are up to date
            model.group_tracker.update_metrics()

            for gid, metrics in model.group_tracker.metrics.items():
                row = metrics.copy()
                row['group_id'] = gid
                row['policy'] = policy_name
                row["seed"]=seed

                for k,v in param_set.items():
                    row[k]=v

                results.append(row)


    return pd.DataFrame(results), pd.concat(histories, ignore_index=True)


### create the policies here, or the sensitivity to run
if __name__ == "__main__":

    sensitivity_params = {
        
        # add sensitivity parameters 
    }

    df_none, df_hist_none = run_model(
        "No policy",
        insurance=False,
        subsidy=False,
        nudge=False,
        steps=30,
        seeds=range(50),
        sensitivity_params=sensitivity_params
    )

    # df_ins, df_hist_ins = run_model(
    #     "Insurance",
    #     insurance=True,
    #     subsidy=False,
    #     nudge=False,
    #     steps=30,
    #     seeds=range(50),
    #     sensitivity_params=sensitivity_params
    # )

    # df_sub, df_hist_sub = run_model(
    #     "Subsidy",
    #     insurance=False,
    #     subsidy=True,
    #     nudge=False,
    #     steps=30,
    #     seeds=range(50),
    #     sensitivity_params=sensitivity_params
    # )

    # df_nudge, df_hist_nudge = run_model(
    #     "Nudge",
    #     insurance=False,
    #     subsidy=False,
    #     nudge=True,
    #     steps=30,
    #     seeds=range(50),
    #     sensitivity_params=sensitivity_params
    # )

    df_nudgeins, df_hist_nudgeins = run_model(
        "Nudge-ins",
        insurance=True,
        subsidy=False,
        nudge=True,
        steps=30,
        seeds=range(50),
        sensitivity_params=sensitivity_params
    )
    df_nudgesub, df_hist_nudgesub = run_model(
        "Nudge-sub",
        insurance=False,
        subsidy=True,
        nudge=True,
        steps=30,
        seeds=range(50),
        sensitivity_params=sensitivity_params
    )

    df_insesub, df_hist_inssub = run_model(
        "Ins-sub",
        insurance=True,
        subsidy=True,
        nudge=False,
        steps=30,
        seeds=range(50),
        sensitivity_params=sensitivity_params
    )




    df_cost_all = pd.concat([df_none, df_nudgeins, df_nudgesub, df_insesub], ignore_index=True)
    df_hist = pd.concat([df_hist_none, df_hist_nudgeins, df_hist_nudgesub, df_hist_inssub], ignore_index=True)

    # df_cost_all = df_none
    # df_hist = df_hist_none
    


    output_dir = Path(__file__).with_name("results")    
    output_dir.mkdir(exist_ok=True)    
    output_path = output_dir / "E9.csv"    
    df_hist.to_csv(output_path, index=False)    
    print(output_path.resolve())    
    print(len(df_hist))

