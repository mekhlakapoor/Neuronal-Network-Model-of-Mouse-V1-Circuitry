# this code file is an extension to the bmtk that tests perturbations for each neuron population
from configparser import ConfigParser
import json
import os
import matplotlib.pyplot as plt
from bmtk.simulator import pointnet
from bmtk.analyzer.spike_trains import plot_raster, plot_rates, plot_rates_boxplot


# creates new config file for perturbation
def create_perturbation_config(new_config_file_path, perturbation, output_path):
    data = {}
    with open("config.json", "r") as config_file:
        data = json.load(config_file) # reading the file

    data['inputs'].update(perturbation)
    data['manifest']['$OUTPUT_DIR']+=(output_path)

    with open(new_config_file_path, "w") as perturbation_file:
        json.dump(data, perturbation_file)  # Writing to the file

# functions that will write new config file for desired perturbation
def exc_perturbation(pop_name):
    new_config_file_path = "exc_perturbation" + pop_name + "_config.json"
    exc_perturbation = {
        "exc_perturbation": {
            "input_type": "current_clamp",
            "module": "IClamp",
            "node_set": {
                "population": "V1",
                "pop_name": pop_name,
            },
            "amp": 230.0,
            "delay": 1.0,
            "duration": 3000.0
        }
    }

    output_path = "/exc_perturbation" + pop_name

    create_perturbation_config(new_config_file_path, exc_perturbation, output_path)

    return new_config_file_path

def inh_perturbation(pop_name):
    new_config_file_path = "inh_perturbation" + pop_name + "_config.json"
    inh_perturbation = {
        "inh_perturbation": {
            "input_type": "current_clamp",
            "module": "IClamp",
            "node_set": {
                "population": "V1",
                "pop_name": pop_name,
            },
            "amp": -230.0,
            "delay": 1.0,
            "duration": 3000.0
        }
    }

    output_path = "/inh_perturbation" + pop_name

    create_perturbation_config(new_config_file_path, inh_perturbation, output_path)

    return new_config_file_path


def run_pointnet(config_file):
    configure = pointnet.Config.from_json(config_file)
    configure.build_env()

    network = pointnet.PointNetwork.from_config(configure)
    sim = pointnet.PointSimulator.from_config(configure, network)
    sim.run()



##################################################################

print('Exc perturbation of PV')
exc_pv = exc_perturbation("PV")

run_pointnet(exc_pv)
plt=plot_raster(config_file=exc_pv, group_by='pop_name')
plt.savefig('exc_perturbation_pv_rasterplot')
plt=plot_rates(config_file=exc_pv, group_by='pop_name')
plt.savefig('exc_perturbation_pv_firingrates')
plt=plot_rates_boxplot(config_file=exc_pv, group_by='pop_name')
plt.savefig('exc_perturbation_pv_firingrates_boxplot')


print('Inh perturbation of PV')
inh_pv = inh_perturbation("PV")

run_pointnet(inh_pv)
plt=plot_raster(config_file=inh_pv, group_by='pop_name')
plt.savefig('inh_perturbation_PV_rasterplot')
plt=plot_rates(config_file=inh_pv, group_by='pop_name')
plt.savefig('inh_perturbation_PV_firingrates')
plt=plot_rates_boxplot(config_file=inh_pv, group_by='pop_name')
plt.savefig('inh_perturbation_pv_firingrates_boxplot')

##################################################################

print('Exc perturbation of SST')
exc_sst = exc_perturbation("SST")

run_pointnet(exc_sst)
plt=plot_raster(config_file=exc_sst, group_by='pop_name')
plt.savefig('exc_perturbation_SST_rasterplot')
plt=plot_rates(config_file=exc_sst, group_by='pop_name')
plt.savefig('exc_perturbation_SST_firingrates')
plt=plot_rates_boxplot(config_file=exc_sst, group_by='pop_name')
plt.savefig('exc_perturbation_SST_firingrates_boxplot')


print('Inh perturbation of SST')
inh_sst = inh_perturbation("SST")

run_pointnet(inh_sst)
plt=plot_raster(config_file=inh_sst, group_by='pop_name')
plt.savefig('inh_perturbation_SST_rasterplot')
plt=plot_rates(config_file=inh_sst, group_by='pop_name')
plt.savefig('inh_perturbation_SST_firingrates')
plt=plot_rates_boxplot(config_file=inh_sst, group_by='pop_name')
plt.savefig('inh_perturbation_SST_firingrates_boxplot')


##################################################################

print('Exc perturbation of LIFexc')
exc_lifexc = exc_perturbation("LIF_exc")

run_pointnet(exc_lifexc)
plt=plot_raster(config_file=exc_lifexc, group_by='pop_name')
plt.savefig('exc_perturbation_LIFexc_rasterplot')
plt=plot_rates(config_file=exc_lifexc, group_by='pop_name')
plt.savefig('exc_perturbation_LIFexc_firingrates')
plt=plot_rates_boxplot(config_file=exc_lifexc, group_by='pop_name')
plt.savefig('exc_perturbation_LIFexc_firingrates_boxplot')


print('Inh perturbation of LIFexc')
inh_lifexc = inh_perturbation("LIF_exc")

run_pointnet(inh_lifexc)
plt=plot_raster(config_file=inh_lifexc, group_by='pop_name')
plt.savefig('inh_perturbation_LIFexc_rasterplot')
plt=plot_rates(config_file=inh_lifexc, group_by='pop_name')
plt.savefig('inh_perturbation_LIFexc_firingrates')
plt=plot_rates_boxplot(config_file=inh_lifexc, group_by='pop_name')
plt.savefig('inh_perturbation_LIFexc_firingrates_boxplot')

##################################################################

'''print('Exc perturbation of VIP')
exc_vip = exc_perturbation("VIP")

run_pointnet(exc_vip)
plt=plot_raster(config_file=exc_vip, group_by='pop_name')
plt.savefig('exc_perturbation_VIP_rasterplot')
plt=plot_rates(config_file=exc_vip, group_by='pop_name')
plt.savefig('exc_perturbation_VIP_firingrates')
plt=plot_rates_boxplot(config_file=exc_vip, group_by='pop_name')
plt.savefig('exc_perturbation_VIP_firingrates_boxplot')


print('Inh perturbation of VIP')
inh_vip = inh_perturbation("VIP")

run_pointnet(inh_vip)
plt=plot_raster(config_file=inh_vip, group_by='pop_name')
plt.savefig('inh_perturbation_VIP_rasterplot')
plt=plot_rates(config_file=inh_vip, group_by='pop_name')
plt.savefig('inh_perturbation_VIP_firingrates')
plt=plot_rates_boxplot(config_file=inh_vip, group_by='pop_name')
plt.savefig('inh_perturbation_VIP_firingrates_boxplot')'''

