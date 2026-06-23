from os import path

from mesa import Model
from agents import Homeowner
from agents import Government
from agents import InsuranceCompany
from agents import Bank
from agents import House
import mesa
import networkx as nx
from mesa.space import NetworkGrid
import pandas as pd
import pymc as pm
import numpy as np
from agent_attributes import house_attributes
from agent_attributes import homeowner_attributes
from agent_attributes import fit_pmt_bayesian_model
from agent_attributes import fit_pmt_normal_model
from mesa.batchrunner import batch_run
from mesa.datacollection import DataCollector


# Import the data
df = pd.read_csv('Data/woningen_per_risicogroep.csv')
df = df.reset_index(drop=True)
df_woningaantal = df.drop_duplicates(subset=['gemeentena'])[['gemeentena', 'Woningen']].reset_index(drop=True)
df_PMT_attributes = pd.read_excel('Data/PMT_attr.xlsx')[
    ['PMT_attr', 'He_Mean_B', 'He_sigma', 'He_r']]


#Choose Frysian municipality
# seleted_gemeente = "Súdwest-Fryslân"
seleted_gemeente = "Leeuwarden"
df_all = pd.read_csv("Data/2_risico_paalrot_2050_laag_mild_cc__1.csv")
df = df_all[df_all['gemeentena'] == seleted_gemeente].copy()

df["risico_percentage"] = pd.to_numeric(df["risico_percentage"], errors="coerce").fillna(0.0)
df = df[df["risico_percentage"] > 0].copy()
df["risico_percentage"] = df["risico_percentage"] / df["risico_percentage"].sum()


class DroughtRiskModel(Model):
    # N is number of homeowners
    def __init__(self, seed = 42, n_households = 6000, insurance = False, subsidy = False, nudge = False, insurance_target = False, subsidy_target = False,
                nudge_ins = False, nudge_subs = False, ins_subs = False, all = False, insurance_premium_amount = 0.2, insurance_coverage_share = 0.7,
                 repair_cost_growth_rate = 0, damage_threshold_delayed = 0, damage_threshold_immediate = 0):
        super().__init__(rng=np.random.default_rng(42))
        self.n_households = 6000
        self.seed = seed

        self.n_households = n_households

        self.policy_insurance = insurance
        self.policy_subsidy = subsidy
        self.policy_nudge = nudge
        self.policy_ins_target = insurance_target
        self.policy_subs_target = subsidy_target
        self.policy_nudge_ins = nudge_ins
        self.policy_nudge_subs = nudge_subs
        self.policy_ins_subs = ins_subs
        self.policy_all = all

        if subsidy:    
            self.policy_name = "Subsidy"
        elif insurance:    
            self.policy_name = "Insurance"
        elif nudge:    
            self.policy_name = "Nudge"
        elif insurance_target:
            self.policy_name = "Insurance targeted"
        elif subsidy_target:
            self.policy_name = "Subsidy targeted"
        elif nudge_ins:
            self.policy_name = "Nudge and insurance"
        elif nudge_subs:
            self.policy_name = "Nudge and subsidy"
        elif ins_subs:
            self.policy_name = "Insurance and subsidy"
        elif all:
            self.policy_name = "All policies"
        else:    
            self.policy_name = "No policy"

        # create network grid      
        grid = nx.erdos_renyi_graph(n = self.n_households, p = 0.05, seed = self.seed)
        self.grid = NetworkGrid(grid)
        self.t = 0
        
        self.history = []
        self.group_history = []
                   
        # add PMT attributes from the function 
        self.pmt_distributions = fit_pmt_normal_model(df_PMT_attributes)
        
        # Store PMT weights 
        self.HH_PMT_weights = np.array([
            df_PMT_attributes[df_PMT_attributes['PMT_attr'] == attr]['He_Mean_B'].values[0]
            for attr in ['Fl_prob', 'Fl_damage', 'Worry', 'Resp_eff', 'Self_eff', 'Cost']
        ])

        self.repair_count = 0
        self.repair_money = 0.0
        self.repair_too_expensive_count = 0


        self.damage_threshold_delayed = 3*100 * (1/15)
        self.damage_threshold_immediate = 6*100 * (1/15)
        self.repair_cost_growth_rate = 0.06



        total_houses, node_idx = house_attributes(
        df=df,
        model=self,
        target_agents=self.n_households,
        amount_per_gemeente=False)  # Naar True for 100 per gemeente

        total_homeowners, _ = homeowner_attributes(
        df=df,
        model=self,        
        target_agents=self.n_households,
        amount_per_gemeente=False  # Naar True for 100 per gemeente
        )


        if total_houses > self.n_households:
            pass

        self.government = Government(model= self)
        self.insurance = InsuranceCompany(model= self)
        self.bank = Bank(model= self)

        self.agents.add(self.government)
        self.agents.add(self.insurance)
        self.agents.add(self.bank)
        self.house_by_id = {a.unique_id: a for a in self.agents if isinstance(a, House)}

        self.inspection_count = 0
        self.inspection_option_no_action = 0
        self.inspection_option_5y = 0
        self.inspection_option_immediate = 0
        #insurance innit
        self.insurance_premium_amount = insurance_premium_amount
        self.insurance_coverage_share = insurance_coverage_share

        self.insurance_payout_count = 0
        self.insurance_payout_amount = 0.0
        self.insurance_policyholders_count = 0
        self.insurance_policyholders_active_count = 0
        self.insurance_paid_total = 0
        self.not_fixed_cost = 0.0
        self.not_fixed_cost_total = 0.0 
        self.policy_costs = 0.0
        self.inspection_option_immediate = 0
        self.inspection_option_5y = 0
        self.inspection_option_no_action = 0

        self.homeowner_groups = self._create_homeowner_groups()
        self.group_tracker = GroupTracker(self, self.homeowner_groups)

        self.group_events = {
        'repairs': 0,
        'repair_money': 0.0,
        'repair_too_expensive': 0,
        'inspections': 0,
        'inspection_no_action': 0,
        'inspection_5y': 0,
        'inspection_immediate': 0,
        'bank_loans': 0,
        'bank_loan_amount': 0.0,
        'gov_loans': 0,
        'gov_loan_amount': 0.0,
        "insurance_premium_paid": 0.0,
        "insurance_payout_received": 0.0,
        "insurance_claims": 0,
        "insurance_has_policy": 0
        }

# This function is to get results per group (WOZ, Available_money and risk)
    def _create_homeowner_groups(self):
        homeowners = [a for a in self.agents if isinstance(a, Homeowner)]
  
        group_data = []
        for ho in homeowners:
            house = self.house_by_id.get(ho.house_id)
            if house:
                risico_num = float(getattr(house, 'risicoklasse', 0.0))
                woz = float(getattr(house, 'woz', 0.0))
                available_money = float(getattr(ho, 'available_money', 0.0))
            
                group_data.append((ho, risico_num, woz, available_money))
    
        # Sort by risk level and divide into 3 groups
        group_data_by_risk = sorted(group_data, key=lambda x: x[1])
        n = len(group_data_by_risk)
        n_per_risk = n // 3
    
        risk_groups = {
            'risk_low': group_data_by_risk[:n_per_risk],
            'risk_medium': group_data_by_risk[n_per_risk:2*n_per_risk],
            'risk_high': group_data_by_risk[2*n_per_risk:]
        }
    
        # Sort by WOZ value and divide into 3 groups
        group_data_by_woz = sorted(group_data, key=lambda x: x[2])
        woz_groups = {
            'woz_low': group_data_by_woz[:n_per_risk],
            'woz_medium': group_data_by_woz[n_per_risk:2*n_per_risk],
            'woz_high': group_data_by_woz[2*n_per_risk:]
        }
    
        # Sort by available money and divide into 3 groups
        group_data_by_money = sorted(group_data, key=lambda x: x[3])
        money_groups = {
            'money_low': group_data_by_money[:n_per_risk],
            'money_medium': group_data_by_money[n_per_risk:2*n_per_risk],
            'money_high': group_data_by_money[2*n_per_risk:]
        }
    
        # Combine all 9 groups for result summary
        groups = {}
        groups.update(risk_groups)
        groups.update(woz_groups)
        groups.update(money_groups)
    
        final_groups = {}
        for group_id, members in groups.items():
            homeowners_in_group = [ho for ho, _, _, _ in members]
            final_groups[group_id] = homeowners_in_group
        
            for ho in homeowners_in_group:
                if not hasattr(ho, 'group_id'):
                    ho.group_id = group_id
                else:
                    ho.group_id = f"{ho.group_id}|{group_id}"
    
        return final_groups
    
    
    def get_homeowner_history(self):
        return pd.DataFrame(self.history)
    
    def get_group_history(self):
        return pd.DataFrame(self.group_history)
    
    def save_homeowner_history(self, path):
        df = self.get_homeowner_history()
        df.to_csv(path, index=False)
        return path
    
    def step(self):
        for a in self.agents:
            if isinstance(a, Homeowner) or isinstance(a, House):
                a.step()
        self.t += 1
        self.government.step()
        self.insurance.step()
        self.bank.step()
        self.group_tracker.update_metrics()
        # record homeowners

        rows = []
        homeowners = [a for a in self.agents if isinstance(a, Homeowner)]
        self.homeowners = homeowners

        self.damage_stock = sum(getattr(a, "neglected_cost_stock", 0.0)    for a in self.homeowners)
        self.damage_flow = sum(getattr(a, "neglected_cost_flow", 0.0)    for a in self.homeowners)
        self.not_fixed_cost = self.damage_stock + self.damage_flow
        self.total_woz = sum(getattr(house, "woz", 0.0)    for house in self.house_by_id.values())
        self.pmt_means = {    
            "Fl_prob": np.mean([a.Fl_prob for a in self.homeowners]),    
            "Fl_damage": np.mean([a.Fl_damage for a in self.homeowners]),    
            "Worry": np.mean([a.Worry for a in self.homeowners]),    
            "Resp_eff": np.mean([a.Resp_eff for a in self.homeowners]),    
            "Self_eff": np.mean([a.Self_eff for a in self.homeowners]),    
            "Cost": np.mean([a.Cost for a in self.homeowners]),}

        row = {    "t": self.t,      
               "repair_money": self.repair_money,    
               "not_fixed_cost": self.not_fixed_cost,    
               "bank_money": self.bank.loan_amount_total,    
               "fund_money": self.government.loan_amount_funderingsherstel_total,    
               "insurance_payout": self.insurance_payout_amount, 
               "insurance_premium": self.insurance_premium_amount, 
               "insurance_premium_total": self.insurance_paid_total, 
               "subsidy": self.policy_costs,      
               "repairs": self.repair_count,    
               "inspections": self.inspection_count,
                "inspection_no_action": self.inspection_option_no_action,    
                "inspection_5y": self.inspection_option_5y,    
                "inspection_immediate": self.inspection_option_immediate,
                "total_woz": self.total_woz,
                "Fl_prob": self.pmt_means["Fl_prob"],    
                "Fl_damage": self.pmt_means["Fl_damage"],    
                "Worry": self.pmt_means["Worry"],    
                "Resp_eff": self.pmt_means["Resp_eff"],    
                "Self_eff": self.pmt_means["Self_eff"],    
                "Cost": self.pmt_means["Cost"],
               }
        
        self.history.append(row)
        
        for group_id, m in self.group_tracker.metrics.items():
            row = {
                "t": self.t,
                "policy": self.policy_name,
                "seed": self.seed,
                "group": group_id,

                "repair_count": m["repair_count"],
                "repair_money": m["repair_money"],
                "repair_too_expensive": m["repair_too_expensive"],
                "inspection_count": m["inspection_count"],
                "inspection_no_action": m["inspection_no_action"],
                "inspection_5y": m["inspection_5y"],
                "inspection_immediate": m["inspection_immediate"],
                "loan_count": m["loan_count"],
                "loan_amount": m["loan_amount"],
                "gov_loan_count": m["gov_loan_count"],
                "gov_loan_amount": m["gov_loan_amount"],
                "insurance_premium_paid": m["insurance_premium_paid"],
                "insurance_payout_received": m["insurance_payout_received"],
                "policy_adoption": m["policy_adoption"],
                "policy_costs": m["policy_costs"],
                "avg_damage": m["avg_damage"],
                "avg_vermogen": m["avg_vermogen"],}

            self.group_history.append(row)
    
        # Print summaries
        for group_id in self.homeowner_groups.keys():
            self.group_tracker.get_group_summary(group_id)

class GroupTracker:
    # find the data per group
    def __init__(self, model, groups_dict):
        self.model = model
        self.groups = groups_dict  # {group_id: [homeowners]}
        self.metrics = {}
        
        # Initialize metrics for each group
        for group_id in groups_dict.keys():
            self.metrics[group_id] = {
                'repair_count': 0,
                'repair_money': 0.0,
                'repair_too_expensive': 0,
                'inspection_count': 0,
                'inspection_no_action': 0,
                'inspection_5y': 0,
                'inspection_immediate': 0,
                'loan_count': 0,
                'loan_amount': 0.0,
                "insurance_premium_paid": 0,
                "insurance_payout_received":0,
                'gov_loan_count': 0,
                'gov_loan_amount': 0.0,
                'measures_adopted': 0,
                'avg_damage': 0.0,
                'avg_vermogen': 0.0,
                'avg_pmt_attrs': None,
                "policy_adoption": 0,
                "policy_costs": 0.0,
                "insurance_claims": 0,
                "insurance_has_policy": 0
            }
    
    def update_metrics(self):
            # update metrics every timestep
        for group_id, homeowners in self.groups.items():
            if not homeowners:
                continue
        
            # Reset counters
            self.metrics[group_id]['repair_count'] = 0
            self.metrics[group_id]['repair_money'] = 0.0
            self.metrics[group_id]['repair_too_expensive'] = 0
            self.metrics[group_id]['inspection_count'] = 0
            self.metrics[group_id]['inspection_no_action'] = 0
            self.metrics[group_id]['inspection_5y'] = 0
            self.metrics[group_id]['inspection_immediate'] = 0
            self.metrics[group_id]['loan_count'] = 0
            self.metrics[group_id]['loan_amount'] = 0.0
            self.metrics[group_id]['gov_loan_count'] = 0
            self.metrics[group_id]['gov_loan_amount'] = 0.0
            self.metrics[group_id]["policy_adoption"] = 0
            self.metrics[group_id]["policy_costs"] = 0.0
            self.metrics[group_id]["insurance_premium_paid"] = 0.0
            self.metrics[group_id]["insurance_payout_received"] = 0.0
            self.metrics[group_id]["insurance_claims"] = 0
            self.metrics[group_id]["insurance_has_policy"] = 0

        # Aggregate from homeowners
            for ho in homeowners:
                if hasattr(ho, 'group_events'):
                    self.metrics[group_id]['repair_count'] += ho.group_events['repairs']
                    self.metrics[group_id]['repair_money'] += ho.group_events['repair_money']
                    self.metrics[group_id]['repair_too_expensive'] += ho.group_events['repair_too_expensive']
                    self.metrics[group_id]['inspection_count'] += ho.group_events['inspections']
                    self.metrics[group_id]['inspection_no_action'] += ho.group_events['inspection_no_action']
                    self.metrics[group_id]['inspection_5y'] += ho.group_events['inspection_5y']
                    self.metrics[group_id]['inspection_immediate'] += ho.group_events['inspection_immediate']
                    self.metrics[group_id]['loan_count'] += ho.group_events['bank_loans']
                    self.metrics[group_id]['loan_amount'] += ho.group_events['bank_loan_amount']
                    self.metrics[group_id]['gov_loan_count'] += ho.group_events['gov_loans']
                    self.metrics[group_id]['gov_loan_amount'] += ho.group_events['gov_loan_amount']
                    self.metrics[group_id]['policy_adoption'] += ho.group_events.get('policy_adoption', 0)
                    self.metrics[group_id]['policy_costs'] += ho.group_events.get('policy_costs', 0.0)
                    self.metrics[group_id]['insurance_premium_paid'] += ho.group_events.get('insurance_premium_paid', 0.0)
                    self.metrics[group_id]['insurance_payout_received'] += ho.group_events.get('insurance_payout_received', 0.0)
                    self.metrics[group_id]['insurance_claims'] += ho.group_events.get('insurance_claims', 0)
                    self.metrics[group_id]['insurance_has_policy'] += ho.group_events.get('insurance_has_policy', 0)

             # Calculate averages
            houses = [self.model.house_by_id.get(ho.house_id) for ho in homeowners if ho.house_id in self.model.house_by_id]
            damages = [getattr(h, 'damage', 0.0) for h in houses]
            self.metrics[group_id]['avg_damage'] = np.mean(damages) if damages else 0.0
        
            vermogen_vals = [ho.vermogen for ho in homeowners if hasattr(ho, 'vermogen')]
            self.metrics[group_id]['avg_vermogen'] = np.mean(vermogen_vals) if vermogen_vals else 0.0
        
            if homeowners:
                pmt_array = np.array([ho.PMT_attrs for ho in homeowners])
                self.metrics[group_id]['avg_pmt_attrs'] = np.mean(pmt_array, axis=0)
        
            self.metrics[group_id]['measures_adopted'] = sum(1 for ho in homeowners if getattr(ho, 'measure', False))
    
    def get_group_summary(self, group_id):
        m = self.metrics[group_id]

if __name__ == "__main__":
    m = DroughtRiskModel()
    m.run_for(30)
 

    # total_subsidy = sum(g["policy_costs"] for g in m.group_tracker.metrics.values())
    # total_insurance_premium = m.insurance_paid_total
    # total_insurance_payout = m.insurance_payout_amount
    # print(f"Total insurance premium paid: €{m.insurance_paid_total:.2f}")








