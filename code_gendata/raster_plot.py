# %%
from allensdk.core.reference_space_cache import ReferenceSpaceCache

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
import scipy.signal
from pathlib import Path

from preprocess_data import ensure_allensdk_cache_layout, load_cache

mpl.rcParams['pdf.fonttype'] = 42

try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path.cwd()

RESULTS_DIR = _SCRIPT_DIR.parent / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Manually selected session. Change this ID to plot a different recording.
SESSION_ID = 1092466205 # Really good, but novel session
CHANGENUM = 6
# %%
cache_dir = ensure_allensdk_cache_layout()
cache = load_cache(cache_dir)
print("Cache directory:", cache_dir)
# %%
# CCF structure tree (separate from the Neuropixels cache; keep it writable locally).
structure_cache_dir = _SCRIPT_DIR.parent / 'data' / 'structure_cache'
structure_cache_dir.mkdir(parents=True, exist_ok=True)
manifest_path = structure_cache_dir / 'manifest.json'

reference_space_cache = ReferenceSpaceCache(
    25, 'annotation/ccf_2017', manifest=str(manifest_path)
)
structure_tree = reference_space_cache.get_structure_tree()
# %%
def get_parent(area, structure_tree):
    """Get parent structure for a given area acronym."""
    structures = structure_tree.get_structures_by_acronym([area])
    if len(structures) == 0:
        return None
    area_struct = structures[0]
    parent_id = area_struct.get('parent_structure_id')
    if parent_id is None:
        return None
    parent_structs = structure_tree.get_structures_by_id([parent_id])
    if len(parent_structs) == 0:
        return None
    return parent_structs[0]
    
def get_area(area, structure_tree):
    """Get structure information for a given area acronym."""
    structures = structure_tree.get_structures_by_acronym([area])
    if len(structures) == 0:
        return None
    return structures[0]
    
def list_parents(area, structure_tree):
    """Get list of all parent structure names for a given area."""
    if area == 'root':
        return []
    
    area_info = get_area(area, structure_tree)
    if area_info is None:
        return []
    
    starting_entry = area_info.get('name', '')
    parent_list = [starting_entry]
    current_area = area
    
    try:
        parent = get_parent(current_area, structure_tree)
        while parent is not None:
            parent_name = parent.get('name', '')
            if parent_name:
                parent_list.append(parent_name)
            parent_id = parent.get('parent_structure_id')
            if parent_id is None:
                break
            current_area = parent.get('acronym', '')
            if not current_area:
                break
            parent = get_parent(current_area, structure_tree)
    except Exception as e:
        print(f'failed to get parents for {area}: {e}')
    
    return parent_list

def get_brain_division_for_area(area, structure_tree, 
                                divisions = ['Isocortex', 'Hippocampal formation', 
                                             'Thalamus', 'Midbrain', 'Hypothalamus',
                                             'Striatum', 'Olfactory areas'],
                               cached_dict = None):
    """Determine brain division for a given area."""
    if cached_dict:
        if area in cached_dict.keys():
            return cached_dict[area]
        
    parents = list_parents(area, structure_tree)
    intersection = np.intersect1d(parents, divisions)
    if len(intersection)>0:
        return intersection[0]
    
    else:
        print(area)
        return 'not in list'

def get_area_order_group(area):
    """Get order group for a given area acronym based on specified ordering.
    Returns (group_number, group_index) for sorting.
    Lower group_number appears first (top of plot), lower group_index within group appears first.
    REVERSED ORDER: FF first, then midbrain, thalamic, ProS/SUB, hippocampal, VIS last.
    """
    # Define area groups in REVERSED order (FF appears first at top)
    area_groups = [
        ['FF'],  # Group 0 - red (was group 5)
        ['APN', 'MB', 'MRN', 'SNr'],  # Group 1 - purple (was group 4)
        ['LGd', 'LP', 'MGd', 'MGm', 'PIL', 'POL', 'PoT', 'SGN', 'SPFp', 'TH', 'VPLpc'],  # Group 2 - pink (was group 3)
        ['ProS', 'SUB'],  # Group 3 - muted dark green (was group 2)
        ['CA1', 'CA3', 'DG'],  # Group 4 - muted green (was group 1)
#        ['VISp', 'VISrl', 'VISl', 'VISal', 'VISpm', 'VISam'],
        ['VISam', 'VISpm', 'VISal', 'VISl', 'VISrl', 'VISp'],
        #['VISal', 'VISam', 'VISl', 'VISp', 'VISpm', 'VISrl'],  # Group 5 - muted blue (was group 0)
    ]
    
    # Check if area matches any group (handle subregions/layers)
    area_base = area.split('-')[0]  # Strip subregions
    
    for group_idx, group in enumerate(area_groups):
        for area_idx, group_area in enumerate(group):
            if area_base == group_area:
                return (group_idx, area_idx)

    # for group_idx, group_areas in enumerate(area_groups):
    #     for area_idx, group_area in enumerate(group_areas):
    #         if area_base == group_area or area_base.startswith(group_area):
    #             return (group_idx, area_idx)
    
    # If not found, assign to end (high number)
    return (999, 999)

# def get_area_color_custom(area, task_flag):
    
#     if task_flag:
#         print("Active")

#     """Get color for a given area acronym based on specified color scheme."""
#     # Define area groups with their colors
#     area_color_map = {
#     # VIS areas - muted blue
#     'VISal': '#6B9BD1', 'VISam': '#6B9BD1', 'VISl': '#6B9BD1', 
#     'VISp': '#6B9BD1', 'VISpm': '#6B9BD1', 'VISrl': '#6B9BD1',

#     # Hippocampal - bright green
#     'CA1': '#00CC44', 'CA3': '#00CC44', 'DG': '#00CC44',

#     # ProS, SUB - dark green
#     'ProS': '#006400', 'SUB': '#006400',

#     # Thalamic areas - muted pink
#     'LGd': '#E6A6B0', 'LP': '#E6A6B0', 'MGd': '#E6A6B0', 'MGm': '#E6A6B0',
#     'PIL': '#E6A6B0', 'POL': '#E6A6B0', 'PoT': '#E6A6B0', 'SGN': '#E6A6B0',
#     'SPFp': '#E6A6B0', 'TH': '#E6A6B0', 'VPLpc': '#E6A6B0',

#     # Midbrain - 35% darker muted orange
#     'APN': '#B87840', 'MB': '#B87840', 'MRN': '#B87840', 'SNr': '#B87840',

#     # FF - red
#     'FF': '#FF0000',
#     }

#     # Fallback color for unknown areas (neutral grey)
#     return area_color_map.get(area, '#808080')


    
#     # Check if area matches any key (handle subregions/layers)
#     area_base = area.split('-')[0]  # Strip subregions
    
#     # Try exact match first
#     if area_base in area_color_map:
#         return area_color_map[area_base]
    
#     # Try prefix match (for areas with layers/subregions)
#     for key, color in area_color_map.items():
#         if area_base.startswith(key) or key in area_base:
#             return color
    
#     # Default to gray if not found
#     return '#808080'

def get_area_color_custom(area, task_flag):
    """
    Get color for a given area acronym based on task condition.
    task_flag = True  -> Active
    task_flag = False -> Passive
    """

    if task_flag:
        print("Active")
        area_color_map = {
            # VIS areas
            'VISal': '#53BBF7', 'VISam': '#0079BC', 'VISl': '#53BBF7',
            'VISp': '#53BBF7', 'VISpm': '#0079BC', 'VISrl': '#53BBF7',

            # Hippocampal
            'CA1': '#00A87A', 'CA3': '#00A87A', 'DG': '#00A87A',

            # ProS, SUB
            'ProS': '#00A87A', 'SUB': '#00A87A',

            # Thalamic areas
            'LGd': '#D679AD', 'LP': '#D679AD', 'MGd': '#D679AD', 'MGm': '#D679AD',
            'PIL': '#D679AD', 'POL': '#D679AD', 'PoT': '#D679AD', 'SGN': '#D679AD',
            'SPFp': '#D679AD', 'TH': '#D679AD', 'VPLpc': '#D679AD',

            # Midbrain
            'APN': '#F0A600', 'MB': '#F0A600', 'MRN': '#F0A600', 'SNr': '#F0A600',

            # FF
            'FF': '#DF6300',
        }
    else:
        print("Passive")
        area_color_map = {
            # VIS areas
            'VISal': '#A7D8F4', 'VISam': '#7AB6D7', 'VISl': '#A7D8F4',
            'VISp': '#A7D8F4', 'VISpm': '#7AB6D7', 'VISrl': '#A7D8F4',

            # Hippocampal
            'CA1': '#7ACDB6', 'CA3': '#7ACDB6', 'DG': '#7ACDB6',

            # ProS, SUB
            'ProS': '#7ACDB6', 'SUB': '#7ACDB6',

            # Thalamic areas
            'LGd': '#E4B9D1', 'LP': '#E4B9D1', 'MGd': '#E4B9D1', 'MGm': '#E4B9D1',
            'PIL': '#E4B9D1', 'POL': '#E4B9D1', 'PoT': '#E4B9D1', 'SGN': '#E4B9D1',
            'SPFp': '#E4B9D1', 'TH': '#E4B9D1', 'VPLpc': '#E4B9D1',

            # Midbrain
            'APN': '#F2CD7A', 'MB': '#F2CD7A', 'MRN': '#F2CD7A', 'SNr': '#F2CD7A',

            # FF
            'FF': '#E9AB7A',
        }

    # Handle subregions / layers (e.g. VISp-2/3, CA1-d)
    area_base = area.split('-')[0]

    # Exact match
    if area_base in area_color_map:
        return area_color_map[area_base]

    # Prefix / partial match
    for key, color in area_color_map.items():
        if area_base.startswith(key) or key in area_base:
            return color

    # Fallback
    return '#808080'

    
def get_area_color(area, task_flag, structure_tree):
    """Get hex color for a given area acronym - now uses custom color scheme."""
    return get_area_color_custom(area, task_flag)
# %%
def plot_raster(task_flag, ax, units, spike_times, alignment_time, time_before = 2.5, time_after = 2.5):
        
    for iu, (uid, unit) in enumerate(units.iterrows()):
        
        # Use structure_acronym_no_layers if available, otherwise use structure_acronym
        area = unit.get('structure_acronym_no_layers', unit['structure_acronym'])
        color = get_area_color(area, task_flag, structure_tree)
        
        spikes = spike_times[uid]
        spikes = spikes[(spikes>alignment_time-time_before)&(spikes<alignment_time+time_after)]
        
        spike_collections = ax.eventplot(spikes, lineoffsets=iu, color=color, rasterized=True) # Only rasterize the dense spike layer
        
        for coll in spike_collections:
            coll.set_rasterized(True)
            coll.set_zorder(-10) # keep raster layer beneath vector overlays
        #spikes = spikes - alignment_time
        
        # ax.eventplot(spikes, lineoffsets=iu, color=color)
# %%
def strip_subregions_layers(areastr):
    """Strip subregions and layers from area acronym."""
    if not isinstance(areastr, str):
        return areastr
    
    areastr = areastr.split('-')[0]
    #remove layer stuff
    area_struct = get_area(areastr, structure_tree)
    if area_struct is None:
        return areastr
    
    areaname = area_struct.get('name', '')
    if not areaname:
        return areastr
        
    if 'layer' in areaname:
        layer = areaname.split('layer')[-1].split(' ')[-1]
        areastr = areastr.replace(layer, '')
        
    #hack for ACA and MOp which doesn't play nice
    if 'ACAd' in areastr:
        areastr = 'ACAd'
        
    if 'ACAv' in areastr:
        areastr = 'ACAv'

    if 'MOp' in areastr:
        areastr = 'MOp'
    
    if 'SC' in areastr:
        parents = list_parents(areastr, structure_tree)
        if 'Superior colliculus sensory related' in parents:
            areastr = 'SCs'
        elif 'Superior colliculus motor related' in parents:
            areastr = 'SCm'

    return areastr
# %%
save = True
task_flag = True
session = cache.get_ecephys_session(SESSION_ID)

for changenum in [CHANGENUM,]:
    spike_times = session.spike_times
    eye_tracking = session.eye_tracking
    running_speed = session.running_speed
    licks = session.licks
    eye_tracking_noblinks = eye_tracking[~eye_tracking['likely_blink']]
    units = session.get_units()
    quality_filter = ((units['isi_violations'] < 0.5) & 
                    (units['amplitude_cutoff']< 0.1) & 
                    (units['presence_ratio'] > 0.9) & 
                    (units['quality'] == 'good') &
                    #(units['amplitude'] > 500) &
                    #(units['max_drift'] < 20) &
                    ([True]*len(units)))

    good_units = units[quality_filter]
    channels = session.get_channels()
    good_units = good_units.merge(channels, left_on='peak_channel_id', right_index=True)
    good_units = good_units.merge(session.probes, left_on='probe_id', right_index=True)
    good_units = good_units[good_units['structure_acronym']!='No Area']
    good_units.value_counts('structure_acronym')

    unit_brain_divisions = good_units.apply(lambda row: get_brain_division_for_area(row['structure_acronym'], structure_tree), axis=1)
    good_units['brain_division'] = unit_brain_divisions
    good_units['brain_division'].value_counts()
    brain_division_order = {
        'Isocortex': 1,
        'Hippocampal formation': 2,
        'Thalamus': 3,
        'Striatum': 4,
        'Midbrain': 5,
        'Olfactory areas': 6,
        'Hypothalamus': 7,
        'not in list': 8
    }

    good_units['brain_division_order'] = good_units.apply(lambda row: brain_division_order[row['brain_division']], axis=1)

    structure_acronym_no_layers = good_units.apply(lambda row: strip_subregions_layers(row['structure_acronym']), axis=1)
    good_units['structure_acronym_no_layers'] = structure_acronym_no_layers
    
    # Add custom ordering based on specified area groups
    order_groups = good_units.apply(lambda row: get_area_order_group(row['structure_acronym_no_layers']), axis=1)
    good_units['area_order_group'] = order_groups.apply(lambda x: x[0])
    good_units['area_order_index'] = order_groups.apply(lambda x: x[1])
    
    stimulus_presentations = session.stimulus_presentations.merge(
        session.trials, left_on='trials_id', right_index=True, how='left',
        suffixes=(None, '_trials'),
    )
    area_labels = good_units.sort_values(by=['area_order_group', 'area_order_index', 'structure_acronym_no_layers', 'dorsal_ventral_ccf_coordinate'], ascending=[True, True, True, False])['structure_acronym'].values
    label_positions = {a:0 for a in np.unique(area_labels)}
    for label in np.unique(area_labels):
        indices = np.where(area_labels==label)[0]
        label_positions[label] = np.median(indices)

    stim_block = 0 if task_flag else 5
    change_rows = stimulus_presentations[
        (stimulus_presentations['is_change'] == True)
        & (stimulus_presentations['stimulus_block'] == stim_block)
    ]
    if change_rows.empty or changenum >= len(change_rows):
        raise ValueError(
            f"Session {SESSION_ID}: changenum={changenum} but found "
            f"{len(change_rows)} change trials in stimulus_block {stim_block}"
        )
    align_time = change_rows.iloc[changenum]['start_time']

    time_before = 5
    time_after = 5.5
    #Get running data aligned to this reward
    trial_running = running_speed.query('timestamps >= {} and timestamps <= {} '.
                                        format(align_time-time_before, align_time+time_after))

    #Get pupil data aligned to this reward
    trial_pupil_area = eye_tracking.query('timestamps >= {} and timestamps <= {} '.
                                        format(align_time-time_before, align_time+time_after))
    trial_pupil_area_smoothed = scipy.signal.medfilt(trial_pupil_area['pupil_area'].values, kernel_size=5)

    #Get stimulus presentations around this reward
    behavior_presentations = stimulus_presentations[stimulus_presentations['active']==task_flag]
    behavior_presentations = behavior_presentations[(behavior_presentations['omitted']==False)]
    trial_stimuli = behavior_presentations.query('end_time >= {} and start_time <= {}'.
                                                format(align_time-time_before, align_time+time_after))

    if task_flag:
        #Get licking aligned to this reward
        trial_licking = licks.query('timestamps >= {} and timestamps <= {} '.
                                            format(align_time-time_before, align_time+time_after))

        reward_time = session.rewards.query('timestamps >= {} and timestamps <= {} '.
                                            format(align_time-time_before, align_time+time_after))['timestamps'].values
        if len(reward_time>0):
            reward_time=reward_time[0]
        else:
            reward_time = np.nan


    fig = plt.figure(figsize=(12, 20))
    spec = fig.add_gridspec(6, 1)

    ax_spikes = fig.add_subplot(spec[1:])
    axr = fig.add_subplot(spec[0])
    axr.sharex(ax_spikes)
    plot_raster(task_flag, ax_spikes, good_units.sort_values(by=['area_order_group', 'area_order_index', 'structure_acronym_no_layers', 'dorsal_ventral_ccf_coordinate'], ascending=[True, True, True, False]), 
                session.spike_times, align_time, time_before=time_before, time_after= time_after)

    # #Plot running, pupil area and licks
    axr.plot(trial_running['timestamps'], trial_running['speed'], 'k')
    axp = axr.twinx()
    axp.plot(trial_pupil_area['timestamps'], trial_pupil_area_smoothed, 'g')
    
    if task_flag:
        rew_handle, = axr.plot(reward_time, 0, 'db', markersize=10)
        lick_handle, = axr.plot(trial_licking['timestamps'], np.zeros(len(trial_licking['timestamps'])), 'mo')
        axr.legend([rew_handle, lick_handle], ['reward', 'licks'])

    axr.set_ylabel('running speed (cm/s)')
    axp.set_ylabel('pupil area\n$(pixels^2)$')
    axr.set_xlabel('Experiment time (s)')

    axp.yaxis.label.set_color('g')
    axp.spines['right'].set_color('g')
    axp.tick_params(axis='y', colors='g')

    #Plot the image flashes as grey bars.
    flash_colors = ['0.3', '0.8']
    unique_stims = list(trial_stimuli['image_name'].unique())
    stimulus_colors = {
        stim: flash_colors[i % len(flash_colors)] for i, stim in enumerate(unique_stims)
    }
    for idx, stimulus in trial_stimuli.iterrows():
        axr.axvspan(stimulus['start_time'], stimulus['end_time'], color=stimulus_colors[stimulus['image_name']], alpha=0.5)
        
        
    area_labels = good_units.sort_values(by=['area_order_group', 'area_order_index', 'structure_acronym_no_layers', 'dorsal_ventral_ccf_coordinate'], ascending=[True, True, True, False])['structure_acronym_no_layers'].values
    for label in np.unique(area_labels):
        indices = np.where(area_labels==label)[0]
        label_positions[label] = np.median(indices)
        ax_spikes.text(align_time+time_after, np.median(indices), label, color=get_area_color(label, task_flag, structure_tree))

    if save:
        raster_path = RESULTS_DIR / f'{SESSION_ID}_raster_{changenum}.pdf'
        fig.savefig(raster_path, dpi=300)
        print(f"Raster figure saved to: {raster_path}")
# %%
# --- Extra output: raster-only SVG (transparent background, no axes) ---

sorted_units = good_units.sort_values(
    by=[
        'area_order_group',
        'area_order_index',
        'structure_acronym_no_layers',
        'dorsal_ventral_ccf_coordinate'
    ],
    ascending=[True, True, True, False]
)

fig_raster, ax_raster = plt.subplots(figsize=(12, 16))

for iu, (uid, unit) in enumerate(sorted_units.iterrows()):
    area = unit.get('structure_acronym_no_layers', unit['structure_acronym'])
    color = get_area_color(area, task_flag, structure_tree)

    spikes = session.spike_times[uid]
    spikes = spikes[
        (spikes > align_time - time_before) &
        (spikes < align_time + time_after)
    ]

    spike_collections = ax_raster.eventplot(
        spikes,
        lineoffsets=iu,
        color=color,
        rasterized=False
    )

    for coll in spike_collections:
        coll.set_zorder(-10)

ax_raster.set_xlim(align_time - time_before, align_time + time_after)
ax_raster.set_ylim(-1, len(sorted_units))

# Remove everything except the raster ticks
ax_raster.axis('off')

# Transparent background
fig_raster.patch.set_alpha(0)
ax_raster.set_facecolor('none')

# Remove extra margins around the axes
fig_raster.subplots_adjust(left=0, right=1, top=1, bottom=0)

svg_path = RESULTS_DIR / f'{SESSION_ID}_raster_only_{changenum}.svg'

fig_raster.savefig(
    svg_path,
    format='svg',
    transparent=True,
    bbox_inches='tight',
    pad_inches=0
)

plt.close(fig_raster)
print(f"Raster-only SVG saved to: {svg_path}")
# %%
