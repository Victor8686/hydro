import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pulp
import time

class Reservoir(object):
    def __init__(self, name, lower_limit, upper_limit):
        self.name = name
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.downstream = ()

class HydroPlant(object):
    def __init__(self, name, list_of_units):
        self.name = name
        self.list_of_units = list_of_units
        self.downstream = ()

class HydroUnit(object):
    def __init__(self, name, pmin, pmax):
        self.name = name
        self.pmin = pmin
        self.pmax = pmax

class HydroTopology(object):
    def __init__(self):
        self.start = datetime.datetime(year=2023, month=2, day=10, hour=10)
        self.end = datetime.datetime(year=2023, month=2, day=11, hour=23)
        self.dataset = self.read_excel()
        self.output = ()

        farris = HydroPlant(name='farris', list_of_units=[HydroUnit(name='G1', pmin=5, pmax=10),
                                                            HydroUnit(name='G2', pmin=5, pmax=10)])
        upper = Reservoir(name='upper', lower_limit=600, upper_limit=700)
        lower = Reservoir(name='lower', lower_limit=500, upper_limit=570)

        self.list_of_hydro_plants = [farris]
        self.list_of_reservoirs = [upper, lower]

        lower.downstream = None
        farris.downstream = lower
        upper.downstream = farris


    def read_excel(self):
        logging.info('Reading indata excel file')
        datetime_range_pd = pd.date_range(start=self.start, end=self.end, freq='H', tz='UTC')
        df = pd.read_excel('indata.xlsx')

        df = df.set_index(datetime_range_pd)

        return df


    def create_and_run_one_big_pulp(self):
        logging.info('Create optimisation problem')

        timesteps = range(len(self.dataset))

        hydro_units = [f"{plant.name}_{unit.name}" for plant in self.list_of_hydro_plants for unit in plant.list_of_units]
        hydro_ts = ['production_mw', 'production_m3s']
        reservoirs = [reservoir.name for reservoir in self.list_of_reservoirs]
        reservoir_ts = ['level_masl', 'level_m3']

        # Create the prob
        prob = pulp.LpProblem(name="HYDRO_OPTIMISATION", sense=pulp.LpMaximize)

        # Create decision variables
        logging.info('Create decision variables')

        #  Production power for each hydro unit and hour
        hydro_unit_vars = pulp.LpVariable.dicts(name="hydro_unit", indices=(hydro_ts,hydro_units,timesteps), lowBound=0, cat=pulp.const.LpContinuous)
        #hydro_unit_production_m3s = pulp.LpVariable.dicts(name="hydro_unit_production_m3s", indices=(hydro_units,timesteps), lowBound=0, cat=pulp.const.LpContinuous)

        # Reservoir level masl and m3
        reservoir_vars = pulp.LpVariable.dicts(name="reservoir", indices=(reservoir_ts,reservoirs,timesteps), lowBound=0, cat=pulp.const.LpContinuous)
        #reservoir_level_m3 = pulp.LpVariable.dicts(name="reservoir_level_m3", indices=(reservoirs,timesteps), lowBound=0, cat=pulp.const.LpContinuous)

        # Market
        spot_market_vars = pulp.LpVariable.dicts(name="spot_market", indices=timesteps, lowBound=0, cat=pulp.const.LpContinuous)

        # Set upper bound on decision variables
        logging.info('Set upper bound on decision variables')

        for j in timesteps:
            spot_market_vars[j].upBound = self.dataset['spot_max_buy_kw'][j]
            for plant in self.list_of_hydro_plants:
                for unit in plant.list_of_units:
                    hydro_unit_vars['production_mw'][f"{plant.name}_{unit.name}"][j].upBound = unit.pmax
            for reservoir in self.list_of_reservoirs:
                reservoir_vars['level_masl'][reservoir.name][j].upBound = reservoir.upper_limit
                reservoir_vars['level_masl'][reservoir.name][j].lowBound = reservoir.lower_limit


        # The objective function is added to 'prob' first
        prob += (pulp.lpSum([spot_market_vars[j] * self.dataset['spot_price_eur_per_mwh'][j] for j in timesteps]), 'Total Cost', )

        # Constraints

        # Turbine efficiency and head mw -> m3/s conversion
        for i in hydro_units:
            for j in timesteps:
                prob += hydro_unit_vars['production_mw'][i][j] == 20 * hydro_unit_vars['production_m3s'][i][j]


        # Reservoir shape masl -> m3
        for res in reservoirs:
                for ts in timesteps:
                    prob += reservoir_vars['level_masl'][res][ts] == 0.002 * reservoir_vars['level_m3'][res][ts]

        # Water balance
        for ts in timesteps:
            if (ts == 0):
                prob += reservoir_vars['level_masl']['upper'][0] == 650
                prob += reservoir_vars['level_masl']['lower'][0] == 550
            else:
                prob += reservoir_vars['level_m3']['upper'][ts] == reservoir_vars['level_m3']['upper'][ts-1] - 3600*pulp.lpSum([hydro_unit_vars['production_m3s'][unit][ts] for unit in hydro_units])
                prob += reservoir_vars['level_m3']['lower'][ts] == reservoir_vars['level_m3']['lower'][ts-1] + 3600*pulp.lpSum([hydro_unit_vars['production_m3s'][unit][ts] for unit in hydro_units])


        # Energy produced = energy sold
        # Add energy market constraints, for each timestep total charging power = sold power from markets
        logging.info('Add energy market constraints, for timestep sum(car charing power)  = sold power from markets')
        for j in timesteps:
            prob += (pulp.lpSum([hydro_unit_vars['production_mw'][i][j] for i in hydro_units]) == spot_market_vars[j], f"ts{j}charge_power_equals_markets",)



        # The problem is solved using PuLP's choice of Solver
        logging.info('Start solve')
        tic = time.perf_counter()
        prob.solve(pulp.PULP_CBC_CMD(msg=True))
        toc = time.perf_counter()
        logging.info(f'Solve complete in {toc - tic:0.4f} seconds')

        prob.roundSolution()

        # Create result dataframes
        logging.info('Create result dataframes')

        units_df = pd.DataFrame(data=(), index=self.dataset.index)
        reservoirs_df = pd.DataFrame(data=(), index=self.dataset.index)

        for ts in hydro_ts:
            for i in hydro_units:
                units_df[f"hydro_unit_{ts}_{i}"] = [hydro_unit_vars[ts][i][j].varValue for j in timesteps]

        for ts in reservoir_ts:
            for i in reservoirs:
                reservoirs_df[f"reservoir_{ts}_{i}"] = [reservoir_vars[ts][i][j].varValue for j in timesteps]

        units_df['spot_price_eur_per_mwh'] = self.dataset['spot_price_eur_per_mwh']


        units_df.plot()
        #plt.figure()
        reservoirs_df.plot()
        plt.show()
        logging.info('Done')