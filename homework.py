# STEP ONE: BUILD V1 NETWORK
from bmtk.builder.networks import NetworkBuilder
from bmtk.builder.auxi.node_params import positions_columinar
import numpy as np
from bmtk.builder.auxi.edge_connectors import distance_connector
from bmtk.utils.sim_setup import build_env_pointnet

def convert_to_2d(target):
    x_coordinates = [n['positions'][0] for n in target.nodes()]
    # y_coordinates = [n['positions'][1] for n in target.nodes()]
    z_coordinates = [n['positions'][2] for n in target.nodes()]

    # convert x and z to linear degrees: tan(x) * (180/pi)
    X = np.tan(0.07 * np.array(x_coordinates) * np.pi / 180.) * 180.0 / np.pi
    Y = np.tan(0.04 * np.array(z_coordinates) * np.pi / 180.) * 180.0 / np.pi
    return np.column_stack((X, Y))

#  Distribute sources of LGN inputs retinotopically
def generate_lgn_positions(N=1, x0=0.0, x1=240.0, y0=0.0, y1=120.0):
    X = np.random.uniform(x0, x1, N)
    Y = np.random.uniform(y0, y1, N)
    return np.column_stack((X, Y))

# this function decides the number of synapses between the LGN --> V1 cells
# for every source cell, there are a limited # of presynaptic targets
def select_source_cells(sources, target, nsources_min=10, nsources_max=30, nsyns_min=3, nsyns_max=12):
    total_sources = len(sources)
    nsources = np.random.randint(nsources_min, nsources_max)
    selected_sources = np.random.choice(total_sources, nsources, replace=False)
    syns = np.zeros(total_sources)
    syns[selected_sources] = np.random.randint(nsyns_min, nsyns_max, size=nsources)
    return syns

def main():
    # 85% excitatory; 15% inhibitory (SST, PV, VIP)
    net = NetworkBuilder("V1")
    # Excitatory population -- 85%
    net.add_nodes(N=8500, pop_name='LIF_exc', ei='e',
                  positions=positions_columinar(N=8500, center=[0, 50.0, 0], min_radius=30.0, max_radius=60.0,
                                                height=100.0),
                  model_type='point_process',
                  model_template='nest:iaf_psc_alpha',
                  dynamics_params='IntFire1_exc_point.json')

    # SST population -- 3.2%
    net.add_nodes(N=320,
                  positions=positions_columinar(N=320, center=[0, 50.0, 0], max_radius=30.0, height=100.0),
                  pop_name='SST', ei='i',
                  model_type='point_process',
                  model_template='nest:iaf_psc_alpha',
                  dynamics_params='IntFire1_inh_point.json'
                  # using same dynamic params as used for PV1 in sonata 300_point neurons
                  )
    # PV population -- 4.4%
    net.add_nodes(N=440,
                  pop_name='PV', ei='i',
                  positions=positions_columinar(N=440, center=[0, 50.0, 0], max_radius=30.0, height=100.0),
                  model_type='point_process',
                  model_template='nest:iaf_psc_alpha',
                  dynamics_params='IntFire1_inh_point.json')
    # VIP Inhibitory population -- 7.4%
    net.add_nodes(N=740, pop_name='VIP', ei='i',
                  positions=positions_columinar(N=740, center=[0, 50.0, 0], max_radius=30.0, height=100.0),
                  model_type='point_process',
                  model_template='nest:iaf_psc_alpha',
                  dynamics_params='IntFire1_inh_point.json')

    # STEP TWO: DISTRIBUTE V1 NEURONS IN 2D SPACE
    retintopic_net = convert_to_2d(net)

    # STEP THREE: BUILD SYNAPTIC CONNECTIONS

    ## E-to-E connections
    net.add_edges(source={'ei': 'e'}, target={'pop_name': 'LIF_exc'},
                  connection_rule=distance_connector,
                  connection_params={'d_weight_min': 0.0, 'd_weight_max': 0.11, 'd_max': 300.0, 'nsyn_min': 3,
                                     'nsyn_max': 7},
                  syn_weight=3.0,
                  delay=2.0,
                  dynamics_params='instanteneousExc.json',
                  model_template='static_synapse')

    ### Generating I-to-I connections
    ### PV-to-I connections
    net.add_edges(source={'pop_name': 'PV'}, target={'pop_name': 'PV'},
                  connection_rule=distance_connector,
                  connection_params={'d_weight_min': 0.26, 'd_weight_max': 0.54, 'd_max': 160.0, 'nsyn_min': 3,
                                     'nsyn_max': 7},
                  syn_weight=-3.0,
                  delay=1.6,
                  dynamics_params='instanteneousInh.json',
                  model_template='static_synapse')

    net.add_edges(source={'pop_name': 'PV'}, target={'pop_name': 'SST'},
                  connection_rule=distance_connector,
                  connection_params={'d_weight_min': 0.03, 'd_weight_max': 0.18, 'd_max': 160.0, 'nsyn_min': 3,
                                     'nsyn_max': 7},
                  syn_weight=-3.0,
                  delay=1.2,
                  dynamics_params='instanteneousInh.json',
                  model_template='static_synapse')

    net.add_edges(source={'pop_name': 'PV'}, target={'pop_name': 'VIP'},
                  connection_rule=distance_connector,
                  connection_params={'d_weight_min': 0.0, 'd_weight_max': 0.06, 'd_max': 160.0, 'nsyn_min': 3,
                                     'nsyn_max': 7},
                  syn_weight=-3.0,
                  delay=1.2,
                  dynamics_params='instanteneousInh.json',
                  model_template='static_synapse')

    ### SST-to-I connections
    net.add_edges(source={'pop_name': 'SST'}, target={'pop_name': 'PV'},
                  connection_rule=distance_connector,
                  connection_params={'d_weight_min': 0.07, 'd_weight_max': 0.27, 'd_max': 160.0, 'nsyn_min': 3,
                                     'nsyn_max': 7},
                  syn_weight=-3.0,
                  delay=1.5,
                  dynamics_params='instanteneousInh.json',
                  model_template='static_synapse')

    net.add_edges(source={'pop_name': 'SST'}, target={'pop_name': 'SST'},
                  connection_rule=distance_connector,
                  connection_params={'d_weight_min': 0.0, 'd_weight_max': 0.08, 'd_max': 160.0, 'nsyn_min': 3,
                                     'nsyn_max': 7},
                  syn_weight=-3.0,
                  delay=1.5,
                  dynamics_params='instanteneousInh.json',
                  model_template='static_synapse')

    net.add_edges(source={'pop_name': 'SST'}, target={'pop_name': 'VIP'},
                  connection_rule=distance_connector,
                  connection_params={'d_weight_min': 0.21, 'd_weight_max': 0.56, 'd_max': 160.0, 'nsyn_min': 3,
                                     'nsyn_max': 7},
                  syn_weight=-3.0,
                  delay=1.5,
                  dynamics_params='instanteneousInh.json',
                  model_template='static_synapse')

    ### VIP-to-I connections
    net.add_edges(source={'pop_name': 'VIP'}, target={'pop_name': 'VIP'},
                  connection_rule=distance_connector,
                  connection_params={'d_weight_min': 0.0, 'd_weight_max': 0.045, 'd_max': 160.0, 'nsyn_min': 3,
                                     'nsyn_max': 7},
                  syn_weight=-3.0,
                  delay=1.5,
                  dynamics_params='instanteneousInh.json',
                  model_template='static_synapse')

    net.add_edges(source={'pop_name': 'VIP'}, target={'pop_name': 'PV'},
                  connection_rule=distance_connector,
                  connection_params={'d_weight_min': 0.0, 'd_weight_max': 0.09, 'd_max': 160.0, 'nsyn_min': 3,
                                     'nsyn_max': 7},
                  syn_weight=-3.0,
                  delay=1.5,
                  dynamics_params='instanteneousInh.json',
                  model_template='static_synapse')

    net.add_edges(source={'pop_name': 'VIP'}, target={'pop_name': 'SST'},
                  connection_rule=distance_connector,
                  connection_params={'d_weight_min': 0.0, 'd_weight_max': 0.31, 'd_max': 160.0, 'nsyn_min': 3,
                                     'nsyn_max': 7},
                  syn_weight=-3.0,
                  delay=1.5,
                  dynamics_params='instanteneousInh.json',
                  model_template='static_synapse')

    ### Generating I-to-E connections
    net.add_edges(source={'pop_name': 'PV'}, target={'ei': 'e'},
                  connection_rule=distance_connector,
                  connection_params={'d_weight_min': 0.2, 'd_weight_max': 0.46, 'd_max': 160.0, 'nsyn_min': 3,
                                     'nsyn_max': 7},
                  syn_weight=-3.0,
                  delay=0.9,
                  dynamics_params='instanteneousInh.json',
                  model_template='static_synapse')

    net.add_edges(source={'pop_name': 'SST'}, target={'ei': 'e'},
                  connection_rule=distance_connector,
                  connection_params={'d_weight_min': 0.12, 'd_weight_max': 0.36, 'd_max': 160.0, 'nsyn_min': 3,
                                     'nsyn_max': 7},
                  syn_weight=-3.0,
                  delay=1.5,
                  dynamics_params='instanteneousInh.json',
                  model_template='static_synapse')

    net.add_edges(source={'pop_name': 'VIP'}, target={'ei': 'e'},
                  connection_rule=distance_connector,
                  connection_params={'d_weight_min': 0.0, 'd_weight_max': 0.14, 'd_max': 160.0, 'nsyn_min': 3,
                                     'nsyn_max': 7},
                  syn_weight=-3.0,
                  delay=0.9,
                  dynamics_params='instanteneousInh.json',
                  model_template='static_synapse')

    

    ### Generating E-to-I connections
    net.add_edges(source={'ei': 'e'}, target={'pop_name': 'PV'},
                  connection_rule=distance_connector,
                  connection_params={'d_weight_min': 0.26, 'd_weight_max': 0.54, 'd_max': 300.0, 'nsyn_min': 3,
                                     'nsyn_max': 7},
                  syn_weight=3.0,
                  delay=1.2,
                  dynamics_params='instanteneousExc.json',
                  model_template='static_synapse')

    net.add_edges(source={'ei': 'e'}, target={'pop_name': 'SST'},
                  connection_rule=distance_connector,
                  connection_params={'d_weight_min': 0.21, 'd_weight_max': 0.44, 'd_max': 300.0, 'nsyn_min': 3,
                                     'nsyn_max': 7},
                  syn_weight=3.0,
                  delay=1.5,
                  dynamics_params='instanteneousExc.json',
                  model_template='static_synapse')

    net.add_edges(source={'ei': 'e'}, target={'pop_name': 'VIP'},
                  connection_rule=distance_connector,
                  connection_params={'d_weight_min': 0.09, 'd_weight_max': 0.3, 'd_max': 300.0, 'nsyn_min': 3,
                                     'nsyn_max': 7},
                  syn_weight=3.0,
                  delay=1.5,
                  dynamics_params='instanteneousExc.json',
                  model_template='static_synapse')

    net.build()
    net.save_nodes(output_dir='Homework/network')
    net.save_edges(output_dir='Homework/network')
    print('Done!', net.nodes())

    # STEP FOUR BUILD EXTERNAL LGN NETWORK
    lgn_net = NetworkBuilder('LGN')

    lgn_net.add_nodes(N=500, pop_name='tON', ei='e', model_type='virtual',
                      positions=generate_lgn_positions(N=500))

    # chose nsources_max based on lgn_conn_props
    lgn_net.add_edges(source=lgn_net.nodes(), target=net.nodes(pop_name='LIF_exc'),
                      connection_rule=select_source_cells,
                      connection_params={'nsources_min': 10, 'nsources_max': 80},
                      iterator='all_to_one',
                      syn_weight=10.0,
                      delay=2.0,
                      dynamics_params='instanteneousExc.json',
                      model_template='static_synapse')

    lgn_net.add_edges(source=lgn_net.nodes(), target=net.nodes(pop_name='VIP'),
                      connection_rule=select_source_cells,
                      connection_params={'nsources_min': 15, 'nsources_max': 20},
                      iterator='all_to_one',
                      syn_weight=10.0,
                      delay=2.0,
                      dynamics_params='instanteneousExc.json',
                      model_template='static_synapse')

    lgn_net.add_edges(source=lgn_net.nodes(), target=net.nodes(pop_name='SST'),
                      connection_rule=select_source_cells,
                      connection_params={'nsources_min': 15, 'nsources_max': 25},
                      iterator='all_to_one',
                      syn_weight=10.0,
                      delay=2.0,
                      dynamics_params='instanteneousExc.json',
                      model_template='static_synapse')

    lgn_net.add_edges(source=lgn_net.nodes(), target=net.nodes(pop_name='PV'),
                      connection_rule=select_source_cells,
                      connection_params={'nsources_min': 15, 'nsources_max': 75},
                      iterator='all_to_one',
                      syn_weight=10.0,
                      delay=2.0,
                      dynamics_params='instanteneousExc.json',
                      model_template='static_synapse')

    lgn_net.build()
    lgn_net.save_nodes(output_dir='Homework/network')
    lgn_net.save_edges(output_dir='Homework/network')

    # Setting up for Pointnet Environment
    build_env_pointnet(base_dir='Homework',
                       network_dir='Homework/network',
                       tstop=3000.0,
                       dt=0.01,
                       include_examples=True
                       )


if __name__ == '__main__':
    main()