import energy_models
import logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

def main():

    hydro_topology = energy_models.HydroTopology()
    hydro_topology.create_and_run_one_big_pulp()

if __name__ == '__main__':
    main()
