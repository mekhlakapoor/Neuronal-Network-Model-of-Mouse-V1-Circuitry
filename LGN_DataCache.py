import os

import statistics


from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
# EcephysProjectCache is main entry pt to the Visual Coding Neuropixels dataset

data_directory = '/Users/mekhlakapoor/Desktop/Allen/bmtk/Homework'
manifest_path = os.path.join(data_directory, "manifest.json")

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

print(cache.get_all_session_types())

sessions = cache.get_session_table()
print('Total number of sessions' + str(len(sessions)))
sessions.head()


def compute_firing_rate(sessions, stimulus):
    firing_rates = []  # instantiate list of firing rates

    for count in range(len(sessions)):
        session_id = sessions.index.values[count]
        session = cache.get_session_data(session_id)

        # filter out units with high snr
        units_with_very_high_snr = session.units[session.units['snr'] > 4]
        session_rates = units_with_very_high_snr[stimulus]

        # session rates is all the firing rates of each unit in a session
        session_avg = statistics.mean(session_rates)
        firing_rates.append(session_avg)

    # firing_rates is a total list of firing rates of all units of all       sessions
    typical_firing_rate = statistics.mean(firing_rates)
    return typical_firing_rate


# FILTER OUT SST SESSIONS AND RETRIEVE ANALYTICS
sst_sessions = sessions[(sessions.full_genotype.str.find('Sst') > -1) & \
                             (sessions.session_type == 'brain_observatory_1.1') & \
                             (['LGd' in acronyms for acronyms #filtering LGN neurons
                               in sessions.ecephys_structure_acronyms])]
print(sst_sessions.head())

sst_avg_rate = compute_firing_rate(sst_sessions, 'firing_rate_fl')
print('typical sst firing rate : ', sst_avg_rate)

# FILTER OUT PV SESSIONS
pv_sessions = sessions[(sessions.full_genotype.str.find('Pvalb') > -1) & \
                             (sessions.session_type == 'brain_observatory_1.1') & \
                             (['LGd' in acronyms for acronyms #filtering LGN neurons
                               in sessions.ecephys_structure_acronyms])]
pv_sessions.head()

pv_avg_rate = compute_firing_rate(pv_sessions, 'firing_rate_fl')
print('typical pv firing rate: ', pv_avg_rate)


# FILTER OUT VIP SESSIONS
vip_sessions = sessions[(sessions.full_genotype.str.find('Vip') > -1) & \
                             (sessions.session_type == 'brain_observatory_1.1') & \
                             (['LGd' in acronyms for acronyms #filtering LGN neurons
                               in sessions.ecephys_structure_acronyms])]
vip_sessions.head()

vip_avg_rate = compute_firing_rate(vip_sessions, 'firing_rate_fl')
print('typical vip firing rate: ', vip_avg_rate)


from bmtk.utils.reports.spike_trains import PoissonSpikeGenerator

psg = PoissonSpikeGenerator(population='LGN')
psg.add(node_ids=range(500),  
        firing_rate=[sst_avg_rate, pv_avg_rate, vip_avg_rate],
        times=(0.0, 3.0))    # Firing starts at 0 s up to 3 s
psg.to_sonata('Homework/inputs/lgn_spikes.h5')

# Let's do a quick check that we have reasonable results.
print(psg.to_dataframe().head())




