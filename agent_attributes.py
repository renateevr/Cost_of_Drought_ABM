from importlib.metadata import distributions
from random import seed

import numpy as np
import pandas as pd
from mesa import Model
from agents import Homeowner
import networkx as nx
from agents import House
import pymc as pm
df_PMT_attributes = pd.read_excel('Data/PMT_attr.xlsx')[
    ['PMT_attr', 'He_Mean_B', 'He_sigma', 'He_r']]


df = pd.read_csv('Data/CBS_Fryslan_filled.csv')
df_factoren = pd.read_csv('Data/gdf_paalrot_fryslan.csv')
stats = (df_factoren.groupby("gemeentena", as_index=False).agg(mu_woz=("WOZ_2025", "mean"))
)
stats = stats.set_index("gemeentena")
df_spaargeld = pd.read_csv("Data/Vermogen_van_huishoudens__huishoudenskenmerken__vermogensbestanddelen_30032026_105013.csv", sep=";", skiprows=4)


def house_attributes(df: pd.DataFrame, model, target_agents: int = None,
                                    amount_per_gemeente: bool = False) -> tuple[int, int]:
    
    # Sort df by gemeente 
    df_local = df.sort_values("gemeentena", kind="stable").reset_index(drop=True)
    node_id_list = list(model.grid.G.nodes())
    woz_id_list = list(range(len(df_local)))
    woz_id = 0
    node_idx = 0
    target_agents = model.n_households


    # Amount of woningen per gemeente for proportional distribution
    woningen_per_gemeente = (
    df_local.groupby("gemeentena", sort=False)["Woningen"].first().astype(float).replace([np.inf, -np.inf], np.nan)
    .dropna()
    )
    gemeente_probs = woningen_per_gemeente / woningen_per_gemeente.sum()
    agents_per_gemeente = gemeente_probs.to_numpy() * target_agents
    int_agents_per_gemeente = np.floor(agents_per_gemeente).astype(int)
    rest_agents_per_gemeente = int(target_agents - int_agents_per_gemeente.sum())
    if rest_agents_per_gemeente > 0:
        extra_g = np.argsort(agents_per_gemeente - int_agents_per_gemeente)[-rest_agents_per_gemeente:]
        int_agents_per_gemeente[extra_g] += 1

    gemeente_quota = dict(zip(woningen_per_gemeente.index, int_agents_per_gemeente))
    

    for gemeente, groep in df_local.groupby("gemeentena", sort=False):
        quota = int(gemeente_quota[gemeente])

        # Distribute agents in this gemeente according to risico_percentage
        probs = groep["risico_percentage"].astype(float).to_numpy()
        probs = probs / probs.sum()

        raw = probs * quota
        counts = np.floor(raw).astype(int)
        rem = int(quota - counts.sum())
        if rem > 0:
            extra = np.argsort(raw - counts)[-rem:]
            counts[extra] += 1

        for i, (_, row) in enumerate(groep.iterrows()):
            n_agents = int(counts[i])
            for _ in range(n_agents):
                initial_damage = 0.0

                sigma_woz = 0.5
                rho = 0.6
                if gemeente in stats.index:
                    mu_woz = float(stats.at[gemeente, "mu_woz"])
                else:
                    mu_woz = float(stats["mu_woz"].mean())

                mu_w = np.log(max(mu_woz, 1e-9))

                z = model.rng.normal()
                eps = model.rng.normal()

                log_woz = mu_w + sigma_woz * z

                woz = float(np.exp(log_woz))

                # Sample risicoklasse from normal distribution using mean and std
                mean_risicoklasse = float(row["average"])
                std_risicoklasse = float(row["std"]) 
                risicoklasse = np.maximum(0, model.rng.normal(mean_risicoklasse, std_risicoklasse))
                

                agent = House(
                    model=model,
                    damage=initial_damage,
                    gemeente=row["gemeentena"],
                    risico_label=row["categorie"],
                    woz = woz,
                    risicoklasse = risicoklasse
                )
                
                # agent.risico_label = row["Paalrot_risk_category"]
                model.agents.add(agent)

                model.grid.place_agent(agent, node_id_list[node_idx])
                node_idx += 1
                woz_id += 1


    return node_idx, node_idx

# make as many homeowners as houses, assign them to the same nodes and link them to the woz_id

def homeowner_attributes(df: pd.DataFrame, model, target_agents: int = None,
                                    amount_per_gemeente: bool = False) -> tuple[int, int]:
    # Build gemeente-level distributions from  mean and median vermogen,
    # then assign one homeowner to existing house.
    cbs = pd.read_csv('Data/CBS_Fryslan_filled.csv').copy()
    mean_col = "Vermogen_particulieren_excl_studenten"
    median_col = "Mediaan_vermogen_particulieren_excl_studenten"
    median_woning_col = "Mediaan_vermogen_eigen_woning"
    mean_spaargeld = "Gemiddeld vermogen"
    median_spaargeld = "Mediaan vermogen"
    mean_hypotheek = "Gemiddeld vermogen"
    median_hypotheek = "Mediaan vermogen"   
    
    if target_agents is None:
        target_agents = model.n_households

    for col in [mean_col]:
        cbs[col] = (
            cbs[col].astype(str)
            .str.replace(",", ".", regex=False)   
        )
        cbs[col] = pd.to_numeric(cbs[col], errors="coerce")

    cbs["gemeente_key"] = cbs["Gemeente"].astype(str).str.strip().str.casefold()
    cbs_lookup = (cbs.groupby("gemeente_key", as_index=True)[[mean_col, median_col, median_woning_col]].mean()
    )
    for col in [mean_spaargeld, median_spaargeld, mean_hypotheek, median_hypotheek]:
        df_spaargeld[col] = pd.to_numeric(
        df_spaargeld[col].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )
    
    spaargeld_row = df_spaargeld[
    df_spaargeld["Onderwerp"].astype(str).str.strip() == "1.1.1 Bank- en spaartegoeden"
    ].iloc[0]

    df_hypotheek = df_spaargeld[
        df_spaargeld["Onderwerp"].astype(str).str.strip() == "2.1 Hypotheekschuld eigen woning"
    ].iloc[0]

    nat_mean_spaargeld = pd.to_numeric(str(spaargeld_row[mean_spaargeld]).replace(".", "").replace(",", "."), errors="coerce",)
    nat_median_spaargeld = pd.to_numeric(str(spaargeld_row[median_spaargeld]).replace(".", "").replace(",", "."), errors="coerce",)
    nat_mean_hypotheek = pd.to_numeric(str(df_hypotheek[mean_hypotheek]).replace(".", "").replace(",", "."), errors="coerce",)
    nat_median_hypotheek = pd.to_numeric(str(df_hypotheek[median_hypotheek]).replace(".", "").replace(",", "."), errors="coerce",)


    nat_mu_sp = np.log(max(nat_median_spaargeld, 1e-9))
    nat_sigma_sp = np.sqrt(2.0 * abs(np.log(max(nat_mean_spaargeld, 1e-9)) - np.log(max(nat_median_spaargeld, 1e-9))))

    nat_mu_hypotheek = np.log(max(nat_median_hypotheek, 1e-9))
    nat_sigma_hypotheek = np.sqrt(2.0 * abs(np.log(max(nat_mean_hypotheek, 1e-9)) - np.log(max(nat_median_hypotheek, 1e-9))))

    nat_mean_vermogen = float(df_spaargeld[df_spaargeld["Onderwerp"].astype(str).str.strip() == "Vermogen"][mean_spaargeld].iloc[0])
    nat_mean_hypotheek = float(df_spaargeld[df_spaargeld["Onderwerp"].astype(str).str.strip() == "2.1 Hypotheekschuld eigen woning"][mean_hypotheek].iloc[0])

    houses = [a for a in model.agents if isinstance(a, House)]
    houses_df = pd.DataFrame(
        {
            "house": houses,
            "gemeente": [h.gemeente for h in houses],
            "gemeente_key": [str(h.gemeente).strip().casefold() for h in houses],
            "woz": [float(getattr(h, "woz", np.nan)) for h in houses],
        }
    )
    created = 0

    for gemeente_key, grp in houses_df.groupby("gemeente_key", sort=False):
        n = len(grp)
        if n == 0:
            continue

        
        mean_val = float(cbs_lookup.at[gemeente_key, mean_col])
        median_val = float(cbs_lookup.at[gemeente_key, median_col])
        median_woning_val = float(cbs_lookup.at[gemeente_key, median_woning_col])

        mu = float(np.log(max(median_val, 1e-9)))
        sigma = float(np.sqrt(2.0 * np.abs(np.log(mean_val) - np.log(median_val))))

        rho = 0.6  # desired correlation
        z1 = model.rng.normal(size=n)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * model.rng.normal(size=n)
        draws = np.exp(mu + sigma * z1)
        shift = mean_val / max(nat_mean_vermogen, 1e-9)

        draws_spaargeld = np.exp(nat_mu_sp + nat_sigma_sp * z2) * shift
        
        # draws = np.exp(mu + sigma * model.rng.normal(size=n))
        mu_woning = float(np.log(max(median_woning_val, 1e-9)))
        sigma_woning = sigma
    
        
        shift_hypotheek = nat_mean_hypotheek / max(nat_mean_vermogen, 1e-9)

        rho_m = 0.6
        z_woz = (grp["woz"].rank(method="first") - n/2) / (n/6)
        z_mort = rho_m * z_woz + np.sqrt(1 - rho_m**2) * model.rng.normal(size=n)
        hypotheek = np.exp(nat_mu_hypotheek + nat_sigma_hypotheek * z_mort) * shift_hypotheek

        candidates = pd.DataFrame(
        {"vermogen": draws, "woz": grp["woz"].to_numpy(), "available_money":draws_spaargeld, "hypotheek": hypotheek})

        house_ranks = grp["woz"].rank(method="first")
        cand_ranks = candidates["vermogen"].rank(method="first")

        if grp["woz"].nunique() >= 5 and len(grp) >= 5:
            houses_bins = pd.qcut(house_ranks, q=5, labels=False, duplicates="drop")
        else:
            houses_bins = pd.Series(
                np.minimum((house_ranks - 1).astype(int), 4),
                index=grp.index
            )

        if candidates["vermogen"].nunique() >= 5 and len(candidates) >= 5:
            cand_bins = pd.qcut(cand_ranks, q=5, labels=False, duplicates="drop")
        else:
            cand_bins = pd.Series(
                np.minimum((cand_ranks - 1).astype(int), 4),
                index=candidates.index
            )


        grp_binned = grp.copy()
        grp_binned["match_bin"] = houses_bins.to_numpy()
        candidates["match_bin"] = cand_bins.to_numpy()
        # candidates["match_bin_money"] = money_bins.to_numpy()


        houses_sorted = grp.sort_values("woz", kind="stable")

        houses_sorted = grp.sort_values("woz", kind="stable").reset_index(drop=True)
        candidates_sorted = candidates.sort_values("vermogen", kind="stable").reset_index(drop=True)

        z_woz = (np.arange(n) - n/2) / (n/6)

        rho_m = 0.6
        z_mort = rho_m * z_woz + np.sqrt(1 - rho_m**2) * model.rng.normal(size=n)

        hypotheek_sorted = np.exp(nat_mu_hypotheek + nat_sigma_hypotheek * z_mort) * shift_hypotheek


        for i, house_row in enumerate(houses_sorted.itertuples(index=False)):    
            ci = i    
            house = house_row.house
            mortgage = hypotheek_sorted[i]
            mortgage_cap = 1.1 * house.woz
            

            max_tries = 5
            tries = 0
            while mortgage > mortgage_cap and tries < max_tries:
                mortgage = float(
                        np.exp(nat_mu_hypotheek + nat_sigma_hypotheek * model.rng.normal())) * shift_hypotheek
                tries += 1

            if mortgage > mortgage_cap:
                mortgage = mortgage_cap

            homeowner = Homeowner(
                    model=model,
                    gemeente=house.gemeente,
                    PMT_attrs=model.rng.random(6),
                    house_id=house.unique_id,
                    vermogen = float(candidates_sorted.at[ci, "vermogen"]),
                    available_money = float(candidates_sorted.at[ci, "available_money"]),
                    mortgage_debt = float(mortgage))

                
            model.agents.add(homeowner)
            created += 1


  
    return created, created

def fit_pmt_bayesian_model(df_PMT_attributes: pd.DataFrame):
        """Fit Bayesian models for each PMT attribute using beta, sigma, and r.
        
        Returns samples for each attribute.
        """
        pmt_attributes = ['Fl_prob', 'Fl_damage', 'Worry', 'Resp_eff', 'Self_eff', 'Cost']
        distributions = {}
        
        for attr_name in pmt_attributes:
            attr_row = df_PMT_attributes[df_PMT_attributes['PMT_attr'] == attr_name]
            
            if len(attr_row) == 0:
                continue
            
            # Extract Bayesian parameters for this attribute
            beta = float(attr_row['He_Mean_B'].iloc[0]) 
            sigma = float(attr_row['He_sigma'].iloc[0])   
            r = float(attr_row['He_r'].iloc[0]) 
            
                      
          

            with pm.Model() as model:
                theta = pm.Normal('theta', mu=beta, sigma=sigma)

                likelihood = pm.Normal('obs', mu=theta * r, sigma=sigma, observed=beta)

                idata = pm.sample(
                    draws=2000, 
                    tune=1000, 
                    target_accept=0.9,
                    return_inferencedata=True, 
                    progressbar=False,
                    random_seed=42
                )

            data_samples = idata.posterior['theta'].values.flatten()
            distributions[attr_name] = data_samples
            
        return distributions


def fit_pmt_normal_model(df_PMT_attributes: pd.DataFrame, n_samples: int = 8000, seed: int = 42):
        pmt_attributes = ["Fl_prob", "Fl_damage", "Worry", "Resp_eff", "Self_eff", "Cost"]
        rng = np.random.default_rng(seed)
        distributions = {}

        for attr_name in pmt_attributes:
            row = df_PMT_attributes[df_PMT_attributes["PMT_attr"] == attr_name]
            if row.empty:
                continue

            beta = float(row["He_Mean_B"].iloc[0])
            sigma = float(row["He_sigma"].iloc[0])
            r = float(row["He_r"].iloc[0])


            sigma_eff = max(sigma, 1e-9)

            samples = rng.normal(loc=beta, scale=sigma_eff, size=n_samples)
            distributions[attr_name] = samples

        return distributions


