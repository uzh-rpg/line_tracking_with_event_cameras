import numpy as np
from .data_structures import Line, Cluster

config = {}

# Testing
config["start_time"] = 20
config["number_of_events_testing"] = 1e10
config["visualize"] = False
config["write_output_images"] = True
config["visualize_periodicity"] = 5  # [ms]

# General
# config["height"] = 180
# config["width"] = 240
config["height"] = 260
config["width"] = 346
config["time_scaling_factor"] = 1000
config["periodical_checks_periodicity"] = 40  # [ms]

# Filters
config["filter_events"] = False
config["filter_refactory_period_same_pol"] = 20  # [ms]
config["filter_refactory_period_opposite_pol"] = 1  # [ms]
config["filter_size_neighbourhood"] = 5
config["filter_num_events_neighbourhood"] = 3
config["filter_time_period"] = 30  # [ms]

# Lines
config["line_promotion_threshold"] = 1.2  # [pixel]
config["line_addition_threshold"] = 1.8  # [pixel]
config["line_length_deletion_threshold"] = 25  # [pixel]
config["line_length_initialization_threshold"] = 40  # [pixel]
# config["line_length_deletion_threshold"] = 18  # [pixel]
# config["line_length_initialization_threshold"] = 27  # [pixel]

config["line_parameter_update_periodicity"] = 20  # [ms]
config["line_event_remove_threshold"] = 70  # [ms]
config["line_t_deletion_threshold"] = 30  # [ms]
config["line_t_hibernate_deletion_threshold"] = 150  # [ms]
config["del_line_t_deletion_threshold"] = 40  # [ms]
config["line_volume_depth"] = 10  # [ms]
config["line_init_period"] = config["line_event_remove_threshold"]
config["line_num_events_promotion"] = 25
config["line_num_events_parameter_update"] = 8
config["line_new_events_density_threshold"] = 0.2  # [events/pixel]
config["line_angle_diff_merge_threshold"] = np.radians(7.0)
config["line_distance_merge_threshold"] = 3.5  # [pixel]
config["line_update_p_time_threshold"] = 3  # [ms]
config["line_center_distance_threshold"] = 1.0  # []
config["line_empty_bin_split_threshold"] = 2
config["line_bin_size_split"] = 2
config["line_hibernate_density_treshold"] = 0.1
config["line_hibernate_no_events_threshold"] = 10  # [ms]
config["line_out_of_frame_band"] = 10  # [pixel]
config["line_max_hibernation_time"] = 100  # [ms]

# Clusters
config["cluster_distance_threshold"] = 2
config["cluster_distance_line_threshold"] = 1.3
config["cluster_event_remove_threshold"] = 30  # [ms]
config["cluster_t_deletion_threshold"] = 50  # [ms]
config["cluster_orientation_difference_merge_threshold"] = np.radians(15.0)
config["cluster_min_num_events_merge"] = 5
config["cluster_candidate_addition_threshold"] = 8
config["cluster_visualization_delimiter"] = 20
config["cluster_num_creation"] = 10
config["cluster_num_events_parameter_update"] = 4
config["cluster_generation_aggregation_period"] = 20  # [ms]

# Chains
config["chain_search_threshold"] = 20 * config["cluster_num_creation"]
config["chain_naive_search_threshold"] = 5
config["chain_offset"] = 3
config["chain_add_distance_threshold"] = 1
config["chain_t_deletion_threshold"] = 50  # [ms]


def set_static_variables(config):
    Line.update_p_time_threshold = config["line_update_p_time_threshold"]
    Line.num_events_for_line_promotion = config["line_num_events_promotion"]
    Line.parameter_update_periodicity = config["line_parameter_update_periodicity"]
    Line.line_promotion_threshold = config["line_promotion_threshold"]
    Line.events_removal_threshold = config["line_event_remove_threshold"]
    Line.num_events_parameter_update = config["line_num_events_parameter_update"]
    Line.line_empty_bin_split_threshold = config["line_empty_bin_split_threshold"]
    Line.volume_depth = config["line_volume_depth"]
    Line.hibernate_density_treshold = config["line_hibernate_density_treshold"]
    Line.hibernate_no_events_threshold = config["line_hibernate_no_events_threshold"]

    Cluster.events_removal_threshold = config["cluster_event_remove_threshold"]
    Cluster.num_events_parameter_update = config["cluster_num_events_parameter_update"]
