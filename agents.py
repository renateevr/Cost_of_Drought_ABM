import numpy as np
import pandas as pd
import seaborn as sns
import mesa
from mesa import Agent
from scipy.stats import bernoulli
df = pd.read_csv('Data/woningen_per_risicogroep.csv')


df_PMT_attributes = pd.read_excel('Data/PMT_attr.xlsx')[
    ['PMT_attr', 'He_Mean_B', 'He_sigma', 'He_r']]

# activate policy
# Policy_Insurance = False
# Policy_Nudge = False
# Policy_Subsidy = False


class Homeowner(Agent):
    def __init__(self, model, gemeente, PMT_attrs,house_id, vermogen, available_money, mortgage_debt):
        super().__init__( model)
        self.gemeente = gemeente
        self.PMT_attrs = self._initial_PMT()
        self.house_id = house_id
        self.measure = False
        self.last_intention = None
        self.vermogen = vermogen
        self.available_money = available_money
        self.mortgage_debt = mortgage_debt
        self.inspection_outcome = None
        self.action_due_step = None
        self.neglected_steps = 0
        self.neglected_costs = 0
        self.last_inspection_step = -10**9
        self.inspection_cooldown = 1 
        self.repair_cost = np.nan
        self.estimated_cost = 0
        self.previous_repair_cost = 0
        self.neglected_cost_flow = 0
        self.neglected_cost_stock = 0
        self.base_cost = np.random.uniform(50, 150)
        

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
            "policy_adoption": 0,
            "policy_costs": 0.0,
            "insurance_premium_paid": 0.0,
            "insurance_payout_received": 0.0,
            "insurance_claims": 0,
            "insurance_has_policy": 0
        }
        # insurance
        self.has_insurance = None
        self.insurance_active = self.has_insurance
        self.model.insurance_coverage_share = 0.70          # insurer pays up to 70% of eligible cost
        self.insurance_woz_cap_share = 0.50           # payout cap: 50% of WOZ
        self.insurance_deductible_share = 0.05        
        self.model.insurance_premium_amount = 0.2   
        self.last_claim_step = -10**9
        self.claim_cooldown = 12
        self.insurance_premium_paid_total = 0.0
        self.insurance_payout_received_total = 0.0
        self.insurance_claim_count = 0
        self.neglected_costs = 0.0
        self.neglected_cost_step = 0.0
        self.neglected_cost_counted = False
        self.neglected_cost_total = 0

    # PMT
    def _initial_PMT(self):
        pmt_attributes = ['Fl_prob', 'Fl_damage', 'Worry', 'Resp_eff', 'Self_eff', 'Cost']
        sampled_values = []
        # Iterate through each attribute and sample from its distribution
        for attr_name in pmt_attributes:
            make_samples = self.model.pmt_distributions[attr_name]
            agent_value = np.random.choice(make_samples)
            sampled_values.append(agent_value)
    
        # Store sampled values for this agent
        self.PMT_attrs = np.array(sampled_values)
        return np.array(sampled_values, dtype=float)
    
    PMT_EVENT_DELTAS = {
    "inspect_no_action":            np.array([-0.05, -0.03, -0.10,  0.000,  0.005, -0.02], dtype=float),
    "inspect_action_5y":            np.array([ 0.03,  0.04,  0.08,  0.004,  0.000,  0.03], dtype=float),
    "immediate":                    np.array([ 0.08,  0.12,  0.15,  0.006, -0.004,  0.04], dtype=float),
    "success_self_fund":            np.array([-0.06, -0.05, -0.12,  0.008,  0.010, -0.04], dtype=float),
    "success_bank_loan":            np.array([-0.04, -0.03,  -0.1,  0.008,  0.006,  0.04], dtype=float),
    "success_gov_loan":             np.array([-0.05, -0.04, -0.08,  0.010,  0.004,  0.06], dtype=float),
    "missed_inspection_high_risk":  np.array([ 0.03,  0.04,  0.06,  0.001, -0.003,  0.03], dtype=float),
    "repair_too_expensive":         np.array([ 0.04,  0.04,  0.10, -0.002, -0.010,  0.12], dtype=float),
    "success_insurance":            np.array([ 0.00,  0.00, -0.10,  0.030,  0.030, -0.100], dtype=float),
    "subsidy_use":                  np.array([ 0.00,  0.00, -0.05,  0.005,  0.005, -0.30], dtype=float),
    "nudge":                        np.array([ 0.10,  0.10,  0.050,  0.000,  0.000,  0.050], dtype=float)
}

    def event_factor(self):
        return min(2.0, 1.0 + 0.15 * self.neglected_steps)

    def update_pmt_on_event(self, event_name):
        delta = self.PMT_EVENT_DELTAS[event_name].copy()
        self.PMT_attrs = np.clip(self.PMT_attrs + delta, -3, 3)
        self.compute_PMT_intention()
        
    def compute_PMT_intention(self):
        y = float(np.dot(self.model.HH_PMT_weights, self.PMT_attrs))
        self.last_intention = 1.0 / (1.0 + np.exp(-y))
        self.measure = bernoulli.rvs(self.last_intention / 4) == 1

    def learn_from_failed_repair(self):
        self.update_pmt_on_event("repair_too_expensive")


    # assessment of the inspection outcome based on damage levels
    def inspection_assessment(self, Risico: float):
        if Risico < self.model.damage_threshold_delayed:
            return "no_action"
        elif Risico < self.model.damage_threshold_immediate:
            return "action_in_5_years"
        else:
            return "immediate_action"
    
    # Actual step that the agents take in the model
    def step(self):
        house = self.model.house_by_id.get(self.house_id)
        risk = getattr(house, "risico_label", None)
        damage = float(getattr(house, "damage", 0.0))
        woz = float(getattr(house, "woz", 0.0))
        riskvalue = getattr(house, "risicoklasse", 0.0)
        base_cost = self.base_cost
        estimated_cost = base_cost * (1 + (self.model.repair_cost_growth_rate * self.neglected_steps))
        Risico = riskvalue * base_cost * (1/15)

        self.neglected_cost_step = 0
        delay_multiplier = 1.0 + (self.model.repair_cost_growth_rate * self.neglected_steps)
        cost = base_cost * delay_multiplier
        self.repair_cost = cost
        increment = cost - self.previous_repair_cost
        increment = max(increment, 0)
        
            

        # Nudge policy, that impacts the pmt attributes and intention to take measures

        self.compute_PMT_intention()

        if self.model.policy_nudge == True:
            self.update_pmt_on_event("nudge")

        else:
            pass

        should_inspect = self.measure and (self.model.t - self.last_inspection_step >= self.inspection_cooldown)
        # Alleen inspect als measure is True, no inspection if inspection has already been done
        if self.measure == True and self.model.policy_insurance == True:
            # and risk != "Zeer hoog risico" and risk != "Hoog risico":
            # for targeted policy risk!=
            self.has_insurance = True
            self.insurance_active = True
            premium = (self.model.insurance_premium_amount * woz) / 12.0
            if self.available_money >= premium:
                self.available_money -= premium
                self.insurance_premium_paid_total += premium
                self.group_events["insurance_premium_paid"] += premium
                self.model.insurance_paid_total += premium

            else:
                # premium quits if the person cannot pay for it anymore
                self.insurance_active = False
        

        if should_inspect:
            self.model.inspection_count += 1
            self.group_events['inspections'] += 1
            self.inspection_outcome = self.inspection_assessment(Risico)

            if self.inspection_outcome == "no_action":
                self.model.inspection_option_no_action += 1
                self.update_pmt_on_event("inspect_no_action")
                self.group_events['inspection_no_action'] += 1
            elif self.inspection_outcome == "action_in_5_years":
                self.model.inspection_option_5y += 1
                if self.action_due_step is None:
                    self.action_due_step = self.model.t + 5
                    self.group_events['inspection_5y'] += 1
                self.update_pmt_on_event("inspect_action_5y")
            elif self.inspection_outcome == "immediate_action":
                self.model.inspection_option_immediate += 1
                self.action_due_step = self.model.t
                self.update_pmt_on_event("immediate")
                self.group_events['inspection_immediate'] += 1
            self.last_inspection_step = self.model.t

        else:
            if Risico > self.model.damage_threshold_delayed:
                self.neglected_steps += 1
                self.update_pmt_on_event("missed_inspection_high_risk")

        action_due_now = self.action_due_step is not None and self.model.t >= self.action_due_step
        is_unresolved = (self.action_due_step is not None    and self.model.t >= self.action_due_step)
        self.neglected_cost_flow = 0
        self.neglected_cost_stock = 0
        if is_unresolved:
            self.neglected_cost_flow = increment
            self.neglected_cost_stock = cost
        if action_due_now and self.inspection_outcome == "action_in_5_years":
            self.inspection_outcome = "immediate_action"
            self.update_pmt_on_event("immediate")

        if action_due_now:
            self.repair_cost = cost
    
            subsidy_amount = 0.0
            if self.model.policy_subsidy == True:
                # and self.vermogen <= 50:
                self.repair_cost = cost * 0.7
                self.update_pmt_on_event("subsidy_use")
                subsidy_amount = cost * 0.3  
            eligible_cost = max(0.0, cost - subsidy_amount)
            
            claim_payout = 0.0
            claim_allowed = (self.model.policy_insurance and self.insurance_active and (self.model.t - self.last_claim_step >= self.claim_cooldown))
            payable_cost = eligible_cost

            if claim_allowed:
                raw_cover = min(self.model.insurance_coverage_share * eligible_cost, self.insurance_woz_cap_share * woz)
                deductible = self.insurance_deductible_share * eligible_cost
                claim_payout = max(0.0, raw_cover - deductible)
                payable_cost = max(0.0, eligible_cost - claim_payout)
                

            if self.available_money > payable_cost:
                self.available_money -= payable_cost
                Risico = Risico * 0.5
                self.neglected_steps = 0
                self.model.repair_count += 1
                self.model.repair_money += cost
                house.risico_label = "Zeer laag risico"
                house.risicoklasse = max(0.0, house.risicoklasse * 0.5)
                self.action_due_step = None
                self.update_pmt_on_event("success_self_fund")
                self.group_events['repairs'] += 1
                self.group_events['repair_money'] += cost
                self.previous_repair_cost = 0.0
                self.neglected_cost_flow = 0.0
                self.neglected_cost_stock = 0.0

                if self.model.policy_subsidy == True:
                    # and self.vermogen <= 50:
                    self.group_events['policy_adoption'] += 1
                    subsidy_value = cost * 0.3
                    self.group_events['policy_costs'] += subsidy_value
                    self.model.policy_costs += subsidy_value

                if claim_payout > 0:
                    self.model.insurance_payout_count += 1
                    self.model.insurance_payout_amount += claim_payout
                    self.last_claim_step = self.model.t
                    self.insurance_payout_received_total += claim_payout
                    self.insurance_claim_count += 1
                    self.model.insurance_payout_count += 1
                    self.model.insurance_payout_amount += claim_payout
                    self.group_events["insurance_payout_received"] += claim_payout
                    self.group_events["insurance_claims"] += 1
                    

            else:
                if self.model.bank.approve_loan(self, house, payable_cost) == True:
                    self.mortgage_debt += payable_cost
                    Risico = Risico * 0.5
                    self.neglected_steps = 0
                    self.model.repair_count += 1
                    self.model.repair_money += cost
                    house.risico_label = "Zeer laag risico"
                    house.risicoklasse = max(0.0, house.risicoklasse * 0.5)
                    self.action_due_step = None
                    self.update_pmt_on_event("success_bank_loan")
                    self.previous_repair_cost = 0.0
                    self.neglected_cost_flow = 0.0
                    self.neglected_cost_stock = 0.0
                    self.group_events['bank_loans'] += 1
                    self.group_events['bank_loan_amount'] += cost
                    self.group_events['repairs'] += 1
                    self.group_events['repair_money'] += cost

                    if self.model.policy_subsidy == True  and self.vermogen <= 200:
                        self.group_events['policy_adoption'] += 1
                        subsidy_value = cost * 0.3
                        self.group_events['policy_costs'] += subsidy_value
                        self.model.policy_costs += subsidy_value
                    
                    if claim_payout > 0:
                        self.model.insurance_payout_count += 1
                        self.model.insurance_payout_amount += claim_payout
                        self.last_claim_step = self.model.t
                        self.insurance_payout_received_total += claim_payout
                        self.insurance_claim_count += 1
                        
                        self.group_events["insurance_payout_received"] += claim_payout
                        self.group_events["insurance_claims"] += 1

                elif self.model.government.loan_funderingsherstel(self, house, payable_cost) == True:
                    self.mortgage_debt += payable_cost
                    Risico = Risico * 0.5
                    self.neglected_steps = 0
                    self.model.repair_count += 1
                    self.previous_repair_cost = 0.0
                    self.neglected_cost_flow = 0.0
                    self.neglected_cost_stock = 0.0
                    self.model.repair_money += cost
                    house.risico_label = "Zeer laag risico"
                    house.risicoklasse = max(0.0, house.risicoklasse * 0.5)
                    self.action_due_step = None
                    self.update_pmt_on_event("success_gov_loan")
                    self.group_events['gov_loans'] += 1
                    self.group_events['gov_loan_amount'] += cost
                    self.group_events['repairs'] += 1
                    self.group_events['repair_money'] += cost

                    if self.model.policy_subsidy == True:
                        # and self.vermogen <= 50:
                        self.group_events['policy_adoption'] += 1
                        subsidy_value = cost * 0.3
                        self.group_events['policy_costs'] += subsidy_value
                        self.model.policy_costs += subsidy_value

                    if claim_payout > 0:
                        self.model.insurance_payout_count += 1
                        self.model.insurance_payout_amount += claim_payout
                        self.last_claim_step = self.model.t
                        self.insurance_payout_received_total += claim_payout
                        self.insurance_claim_count += 1
                        
                        self.group_events["insurance_payout_received"] += claim_payout
                        self.group_events["insurance_claims"] += 1

                else:
                    self.model.repair_too_expensive_count += 1
                    self.group_events['repair_too_expensive'] += 1
                    self.neglected_steps += 1
                    

                    
                    self.learn_from_failed_repair()
                    self.compute_PMT_intention()
                    woz *= 0.88
                    self.update_pmt_on_event("repair_too_expensive")
                    


        self.neglected_cost_step = (self.neglected_cost_flow +self.neglected_cost_stock)
        self.neglected_cost_total += self.neglected_cost_step            
        self.previous_repair_cost = cost
        self.Fl_prob = self.PMT_attrs[0]
        self.Fl_damage = self.PMT_attrs[1]
        self.Worry = self.PMT_attrs[2]
        self.Resp_eff = self.PMT_attrs[3]
        self.Self_eff = self.PMT_attrs[4]
        self.Cost = self.PMT_attrs[5]

        house.damage = damage
        house.woz = woz

        # print(f"Homeowner {self.unique_id} in {self.gemeente} has damage: {damage:.2f}, measure: {self.measure}, intention: {self.last_intention:.4f}")
        pass


class House(Agent):
    def __init__(self, model, damage,gemeente,risico_label,woz,risicoklasse):
        super().__init__(model)
        self.gemeente = gemeente
        self.damage = damage
        self.risico_label = risico_label
        self.woz = woz
        self.risicoklasse = risicoklasse
        
    def step(self):

        pass



class Government(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.loan_count = 0
        self.loan_amount_funderingsherstel_total = 0.0
        self.model.loan_limit = 20000000 * 1/6 # dit is landelijk limiet, geen idee welk deel hieravn friesland is

    def loan_funderingsherstel(self, homeowner: Homeowner, house: House, loan_amount: float) -> bool:
        if homeowner.mortgage_debt + loan_amount > 2 * house.woz:
            return False
            
        if self.loan_amount_funderingsherstel_total >= self.model.loan_limit:
            return False
            
        else:
            self.loan_count += 1
            self.loan_amount_funderingsherstel_total += loan_amount
            return True
            

    def step(self):
        pass

class InsuranceCompany(Agent):
    def __init__(self, model):
        super().__init__(model)

    def step(self):
        pass

class Bank(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.loan_count = 0
        self.loan_amount_total = 0.0

    def approve_loan(self, homeowner: Homeowner, house: House, loan_amount: float) -> bool:
        if homeowner.mortgage_debt > 0.8 * house.woz:
                return False
            
        if loan_amount > 0.5 * house.woz:
                return False
            
        else:
            self.loan_count += 1
            self.loan_amount_total += loan_amount
            return True
            


    def step(self):
        pass
