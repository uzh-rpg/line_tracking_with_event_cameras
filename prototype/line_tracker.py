from utils.filters import RefactoryPeriodFilter, NeighbouringEventsFilter
import utils.data_structures as ds
import numpy as np
import pandas as pd
from utils.visualizations import *


class LineTracker:
    def __init__(self, images_df, config):
        self.config = config
        self.refactory_filter = RefactoryPeriodFilter(config)
        self.neighbouring_filter = NeighbouringEventsFilter(config)

        self.lines = {}
        self.clusters = {}
        self.unasigned_events = []
        self.unique_line_id = 0
        self.unique_cluster_id = 0
        self.last_periodical_check = 0.0
        self.mean_vel = 0.0
        self.initialization = True

        # visualization
        self.images_df = images_df
        self.curr_image_idx = 0
        self.curr_image_t = (
            self.images_df.iloc[0, 0] * self.config["time_scaling_factor"]
        )
        self.image_name = self.images_df.iloc[0, 1]
        self.image_path = self.config["path_to_data"] + self.image_name
        self.last_visualization = 0.0

        # testing
        self.all_events = []
        self.filtered_events = []

    def update(self, x, y, t, p):

        # ignore polarity
        p_org = p
        p = 0

        # Testing
        self.all_events.append([x, y, t, p_org])

        # misc
        self.get_current_image(t)

        # visualization
        self.visualize(x, y, t, p)

        # periodical checks
        self.periodical_check(t)

        # filtering
        if pow(x - self.config['width']/2, 2) + pow(y - self.config['height']/2, 2) > pow(self.config['width']/2, 2):
            return

        passed_refactory_filter = self.refactory_filter.update(x, y, t, p_org)
        passed_neighbouring_filter = self.neighbouring_filter.update(x, y, t, p_org)
        if not passed_refactory_filter or not passed_neighbouring_filter:
            return

        # testing
        self.filtered_events.append([x, y, t, p_org])

        # line growth
        if self.line_growth(x, y, t, p):
            return

        # cluster growth
        if self.cluster_growth(x, y, t, p):
            return

        # cluster generation
        if self.cluster_generation(x, y, t, p_org):
            return

        # leave event unasigned
        self.unasigned_events.append([x, y, t, p_org])

    def get_current_image(self, t):

        if t > self.curr_image_t:

            if self.config["write_output_images"]:
                print("t: " + str(t) + " , img t: " + str(self.curr_image_t))

                image_output_path_initialized = (
                    self.config["path_to_data"]
                    + "output/170121/initialized/"
                    + self.image_name
                )

                image_output_path_all = (
                    self.config["path_to_data"] + "output/170121/all/" + self.image_name
                )
                print("Written image: " + self.image_name)

                # writeLinesAndImage(self.lines, 1, self.image_path, image_output_path)
                writeInitializedLinesAndImage(
                    self.lines,
                    self.image_path,
                    image_output_path_initialized,
                    self.config["height"],
                    self.config["width"],
                )

                writeAllLinesAndImage(
                    self.lines,
                    self.image_path,
                    image_output_path_all,
                    self.config["height"],
                    self.config["width"],
                )
            # visualizeClustersAndLinesWithImage(
            #     self.clusters,
            #     self.lines,
            #     t,
            #     self.image_path,
            #     self.config["height"],
            #     self.config["width"],
            # )

            self.curr_image_idx += 1
            self.curr_image_t = (
                self.images_df.iloc[self.curr_image_idx, 0]
                * self.config["time_scaling_factor"]
            )
            self.image_name = self.images_df.iloc[self.curr_image_idx, 1]
            self.image_path = self.config["path_to_data"] + self.image_name
        # img_rows_smaller = self.images_df[
        #     self.images_df["t"] < (t / self.config["time_scaling_factor"])
        # ]

        # if img_rows_smaller.empty:
        #     curr_image_name = self.images_df.iloc[0, 1]
        # else:
        #     curr_image_name = img_rows_smaller.iloc[-1, 1]

        # curr_image_name = self.images_df[
        #     self.images_df["t"] > (t / self.config["time_scaling_factor"])
        # ].iloc[0, 1]
        # self.image_path = self.config["path_to_data"] + curr_image_name

    def line_growth(self, x, y, t, p):
        # line growth
        line_distance_dict = {}
        nearest_distance = float("inf")
        nearest_line_id = -1
        for line_id in self.lines:
            line = self.lines[line_id]
            if line.pol == p:

                # check distance from line
                center_point_distance = np.linalg.norm(
                    line.point[:2] - np.array([x, y])
                )

                if (
                    center_point_distance
                    < self.config["line_center_distance_threshold"] * line.length
                ):
                    # check for distance to inferred line
                    line_distance = line.get_distance_to_inferred_line([x, y, t])
                    if line_distance < self.config["line_addition_threshold"]:
                        line_distance_dict[line_id] = line_distance

                        if nearest_distance > line_distance_dict[line_id]:
                            nearest_distance = line_distance_dict[line_id]
                            nearest_line_id = line_id

        # found one line
        if len(line_distance_dict) == 1:
            self.lines[list(line_distance_dict)[0]].add_event([x, y, t])
            return True

        # found multiple lines
        if len(line_distance_dict) > 1:

            line_ids = list(line_distance_dict.keys())
            line_id = line_ids.pop()

            # check if lines are parallel
            lines_parallel = True
            for l_id in line_ids:
                angle_diff = np.arccos(
                    abs(np.dot(self.lines[line_id].normal, self.lines[l_id].normal))
                )
                if angle_diff > self.config["line_angle_diff_merge_threshold"]:
                    lines_parallel = False

            if lines_parallel:
                self.lines[nearest_line_id].add_event([x, y, t])

            # try to merge lines
            line_ids_to_merge = []
            for l_id in line_ids:
                if self.check_merging_lines(self.lines[line_id], self.lines[l_id]):
                    line_ids_to_merge.append(l_id)

            line_ids_to_merge.append(line_id)
            line_ids_to_merge.sort()
            line_id = line_ids_to_merge.pop(0)
            for line_id_to_merge in line_ids_to_merge:
                print(
                    "Line Growth: Merging lines "
                    + str(line_id)
                    + " and "
                    + str(line_id_to_merge)
                )
                self.lines[line_id].merge_lines(self.lines[line_id_to_merge])
                del self.lines[line_id_to_merge]

            return True

        return False

    def cluster_growth(self, x, y, t, p):
        # cluster growth

        added_event_to_cluster, cluster_id = self.try_adding_event_to_cluster(
            x, y, t, p
        )

        if added_event_to_cluster:
            self.try_promoting_cluster(cluster_id)

        return added_event_to_cluster

    def cluster_generation(self, x, y, t, p):

        if self.initialization:
            if t > self.config["cluster_generation_aggregation_period"]:
                self.initialization = False
            return False

        if len(self.unasigned_events) > self.config["chain_search_threshold"]:
            found_next_chain_event = True
            chain = [[x, y, t]]
            curr_chain_event = [x, y, t]
            unasigned_events_to_search = self.unasigned_events

            while found_next_chain_event:
                curr_x = curr_chain_event[0]
                curr_y = curr_chain_event[1]

                # ignore events too close to boarder
                if (
                    curr_x + self.config["chain_offset"] >= self.config["width"]
                    or curr_x - self.config["chain_offset"] < 0
                    or curr_y + self.config["chain_offset"] >= self.config["height"]
                    or curr_y - self.config["chain_offset"] < 0
                ):
                    break

                # neighbourhood
                x_l = curr_x - self.config["chain_offset"]
                x_r = curr_x + self.config["chain_offset"]
                y_u = curr_y - self.config["chain_offset"]
                y_d = curr_y + self.config["chain_offset"]

                un_events = np.array(unasigned_events_to_search)

                idx_neighbourhood = np.argwhere(
                    (un_events[:, 0] >= x_l)
                    & (un_events[:, 0] <= x_r)
                    & (un_events[:, 1] >= y_u)
                    & (un_events[:, 1] <= y_d)
                    & ((un_events[:, 0] != curr_x) | (un_events[:, 1] != curr_y))
                    & (t - un_events[:, 2] <= self.config["chain_t_deletion_threshold"])
                    & (un_events[:, 3] == p)
                ).flatten()

                if idx_neighbourhood.size > 0:
                    found_next_chain_event = True

                    events_neighbourhood = un_events[idx_neighbourhood]
                    distances_to_curr_event = np.sum(
                        (events_neighbourhood[:, :2] - np.array([curr_x, curr_y])) ** 2,
                        axis=-1,
                    )

                    idx_sorted = np.argsort(distances_to_curr_event)
                    next_chain_event_idx = idx_sorted[-1]
                    # chain.extend(events_neighbourhood[:, :3].tolist())
                    chain.append(events_neighbourhood[next_chain_event_idx])

                    # delete events
                    unasigned_events_to_search = np.delete(
                        unasigned_events_to_search,
                        idx_neighbourhood,
                        axis=0,
                    )

                    unasigned_events_to_search = unasigned_events_to_search.tolist()
                    curr_chain_event = events_neighbourhood[next_chain_event_idx]

                    # visualizeUnasignedEventsAndChainAndLinesWithImage(
                    #     self.unasigned_events,
                    #     chain,
                    #     self.lines,
                    #     t,
                    #     self.image_path,
                    #     self.config["height"],
                    #     self.config["width"],
                    # )

                    # cluster creation
                    if len(chain) > self.config["chain_naive_search_threshold"]:
                        (
                            chain_direction,
                            chain_normal,
                            chain_eig_val,
                            chain_eig_vec,
                            chain_mean,
                        ) = ds.calculate_line_parameters([e[:2] for e in chain])

                        chain_is_line = True
                        for ev in chain:
                            distance_to_line = ds.calculate_distance_to_line(
                                chain_mean, chain_normal, ev
                            )
                            if (
                                distance_to_line
                                > self.config["chain_add_distance_threshold"]
                            ):

                                chain_is_line = False

                        # chain is a line create cluster
                        if chain_is_line:
                            chain_events = [e[:3] for e in chain]
                            self.clusters[self.unique_cluster_id] = ds.Cluster(
                                chain_events,
                                0,
                                chain_direction,
                                chain_normal,
                                chain_mean,
                                chain_eig_val[0],
                            )
                            self.unique_cluster_id += 1

                        # remove chain elements from unasigned in both cases
                        self.unasigned_events = unasigned_events_to_search
                        return True

                else:
                    found_next_chain_event = False

            # no events found
            else:
                return False

    def periodical_check(self, t):
        # periodical checks
        if (
            t - self.last_periodical_check
            > self.config["periodical_checks_periodicity"]
        ):

            # clusters
            self.periodical_check_clusters(t)

            # lines
            self.periodical_check_lines(t)

            # unasigned events
            self.periodical_check_unasigned_events(t)

            self.last_periodical_check = t

    def visualize(self, x, y, t, p):

        if not self.config["visualize"]:
            return

        if t - self.last_visualization > self.config["visualize_periodicity"]:

            print("---------------------------------------")
            print("CURR TIME: " + str(t))

            for line_id in self.lines:
                if self.lines[line_id].hibernate:
                    print("Line " + str(line_id) + " is hibernating")

            visualizeClustersAndLinesWithImage(
                self.clusters,
                self.lines,
                t,
                self.image_path,
                self.config["height"],
                self.config["width"],
            )

            image_output_path = (
                self.config["path_to_data"] + "output/131220/" + self.image_name
            )

            self.last_visualization = t

    def try_adding_event_to_cluster(self, x, y, t, p):
        nearest_cluster_ids = []
        for cluster_id in self.clusters:
            cluster = self.clusters[cluster_id]
            if cluster.pol == p:
                distance_to_cluster = cluster.get_distance_to_nearest_neighbour(
                    [x, y, t]
                )

                distance_to_inferred_line = cluster.get_distance_to_inferred_line(
                    [x, y, t]
                )

                if (
                    distance_to_cluster <= self.config["cluster_distance_threshold"]
                    and distance_to_inferred_line
                    <= self.config["cluster_distance_line_threshold"]
                ):
                    nearest_cluster_ids.append(cluster_id)

        # add event to cluster
        if len(nearest_cluster_ids) > 0:
            curr_cluster_id = nearest_cluster_ids[0]
            curr_cluster = self.clusters[curr_cluster_id]
            curr_cluster.add_event([x, y, t])

            # merge clusters if multiple
            if len(nearest_cluster_ids) > 1:
                nearest_cluster_ids.pop(0)
                for nearest_cluster_id in nearest_cluster_ids:
                    nearest_cluster = self.clusters[nearest_cluster_id]

                    nearest_cluster_normal = nearest_cluster.normal
                    curr_cluster_normal = curr_cluster.normal
                    angle_diff = np.arccos(
                        abs(np.dot(curr_cluster_normal, nearest_cluster_normal))
                    )

                    if (
                        angle_diff
                        < self.config["cluster_orientation_difference_merge_threshold"]
                    ):

                        curr_cluster.merge_clusters(self.clusters[nearest_cluster_id])
                        del self.clusters[nearest_cluster_id]

            return True, curr_cluster_id
        else:
            return False, -1

    def try_promoting_cluster(self, cluster_id):

        # cluster promotion
        cluster = self.clusters[cluster_id]
        if len(cluster.events) > self.config["line_num_events_promotion"]:
            (
                plane_normal,
                eig_val,
                eig_vec,
                line_events_mean,
                line_length,
            ) = ds.calculate_plane_parameters(cluster.events)

            if (
                eig_val[0] < self.config["line_promotion_threshold"]
                and line_length > self.config["line_length_deletion_threshold"]
                and not self.check_if_point_out_of_frame(line_events_mean[:2])
            ):
                new_line = ds.Line(
                    plane_normal,
                    eig_val,
                    eig_vec,
                    line_events_mean,
                    line_length,
                    cluster.events,
                    cluster.pol,
                )

                print(
                    "Promoting cluster to line "
                    + str(self.unique_line_id)
                    + " widht length: "
                    + str(line_length)
                )

                # merged_with_del_line = False
                merged_with_line = False

                # check for line
                merged_with_line = self.try_to_merge_with_line(new_line)

                if not merged_with_line:
                    self.lines[self.unique_line_id] = new_line
                    self.unique_line_id += 1

                del self.clusters[cluster_id]

            else:
                print(
                    "Cluster Promotion: Deleting cluster "
                    + str(cluster_id)
                    + " too short: "
                    + str(line_length)
                )

                del self.clusters[cluster_id]

    def try_to_merge_with_line(self, new_line):

        # check for merge
        curr_line_direction = new_line.line_direction

        nearest_line_ids = []
        for l_id in self.lines:
            if self.lines[l_id].pol == new_line.pol:
                if self.check_merging_lines(new_line, self.lines[l_id]):
                    nearest_line_ids.append(l_id)

        if len(nearest_line_ids) > 0:
            nearest_line = self.lines[nearest_line_ids[0]]
            nearest_line.merge_lines(new_line)
            print("Merge promoting cluster with line " + str(nearest_line_ids[0]))
            return True

        return False

    def get_line_ids_to_merge(self, line_id):

        line = self.lines[line_id]

        line_ids_to_merge = []
        for l_id in self.lines:
            if line_id != l_id:
                if self.lines[l_id].pol == line.pol:
                    if self.check_merging_lines(line, self.lines[l_id]):
                        line_ids_to_merge.append(l_id)

        return line_ids_to_merge

    def check_merging_lines(self, line, other_line):

        line_direction = line.line_direction
        other_line_direction = other_line.line_direction
        angle_diff = np.arccos(abs(np.dot(line_direction, other_line_direction)))
        distance_to_line_1 = line.get_distance_to_inferred_line(other_line.point)
        distance_to_line_2 = other_line.get_distance_to_inferred_line(line.point)

        distance_to_curr_line = min([distance_to_line_1, distance_to_line_2])

        mid_point_distance = np.linalg.norm(other_line.point[:2] - line.point[:2])
        mid_point_distance_threshold = (line.length + other_line.length) / 2

        if (
            (angle_diff < self.config["line_angle_diff_merge_threshold"])
            and (distance_to_curr_line < self.config["line_distance_merge_threshold"])
            and (mid_point_distance < mid_point_distance_threshold)
        ):
            return True
        else:
            return False

    def periodical_check_clusters(self, t):

        if len(self.clusters) == 0:
            return

        clusters_to_delete = []
        for cluster_id in self.clusters:
            # delete old clusters
            t_newest_event = self.clusters[cluster_id].events[-1][2]

            if t - t_newest_event > self.config["cluster_t_deletion_threshold"]:
                # print(
                #     "Period Check: Deleting cluster "
                #     + str(cluster_id)
                #     + " because no new events"
                # )
                clusters_to_delete.append(cluster_id)
            else:
                self.clusters[cluster_id].remove_old_events(t)
                if (
                    len(self.clusters[cluster_id].events)
                    < self.config["cluster_num_creation"]
                ):
                    # print(
                    #     "Period Check: Deleting cluster "
                    #     + str(cluster_id)
                    #     + " too few events after old events removed"
                    # )
                    clusters_to_delete.append(cluster_id)

        for cluster_id_to_delete in clusters_to_delete:
            del self.clusters[cluster_id_to_delete]

    def periodical_check_lines(self, t):

        if len(self.lines) == 0:
            return

        line_ids = []
        line_ids_initializing = []
        line_ids_hibernating = []

        line_ids_to_delete = []

        for line_id in self.lines:

            # checks for all lines
            if self.check_if_line_out_of_frame(line_id):
                line_ids_to_delete.append(line_id)
                continue

            # seperate initializing, hibernating and actual lines
            self.lines[line_id].check_for_hibernation(t)
            if self.lines[line_id].initializing:
                line_ids_initializing.append(line_id)
            elif self.lines[line_id].hibernate:
                line_ids_hibernating.append(line_id)
            else:
                line_ids.append(line_id)

        for line_id in line_ids_initializing:
            line = self.lines[line_id]
            if t > line.initialization_period_end:
                line.initializing = False
                line_length = line.get_connected_line_length()

                print("XXXXXXXX")
                print(
                    "Period Check: Initialize line "
                    + str(line_id)
                    + " line length: "
                    + str(line_length)
                )

                if line_length < self.config["line_length_initialization_threshold"]:
                    line_ids_to_delete.append(line_id)
                    continue

        for line_id in line_ids_hibernating:

            # check for line length
            if (
                self.lines[line_id].length
                < self.config["line_length_deletion_threshold"]
            ):
                print(
                    "Period Check: Perman remove line"
                    + str(line_id)
                    + " due to length "
                    + str(line.length)
                )
                line_ids_to_delete.append(line_id)
                continue

            t_newest_event = self.lines[line_id].events[-1][2]
            if t - t_newest_event > self.config["line_max_hibernation_time"]:

                line_ids_to_delete.append(line_id)
                continue

            t_start_hibernation = self.lines[line_id].hibernate_start_time
            if t - t_start_hibernation > self.config["line_max_hibernation_time"]:
                line_ids_to_delete.append(line_id)
                continue

        for line_id in line_ids:
            line = self.lines[line_id]

            line.remove_old_events(t)
            line.update_plane_estimate()

            # check age of newest event
            t_newest_event = line.events[-1][2]
            if t - t_newest_event > self.config["line_t_deletion_threshold"]:
                print("Period Check: Remove line " + str(line_id) + " no new events")
                line_ids_to_delete.append(line_id)
                continue

            # check for line length
            if line.length < self.config["line_length_deletion_threshold"]:
                print(
                    "Period Check: Perman remove line"
                    + str(line_id)
                    + " due to length "
                    + str(line.length)
                )
                line_ids_to_delete.append(line_id)
                continue

            # check empty bin ratio
            # TODO remove since this will be checked after initialization
            # if (
            #     line.get_ratio_of_emtpy_bins()
            #     < self.config["line_empty_bin_ratio_deletion_threshold"]
            # ):
            #     print(
            #         "Period Check: Remvove line "
            #         + str(line_id)
            #         + " due to empty bin ratio:"
            #         + str(line.get_ratio_of_emtpy_bins)
            #     )
            #     line_ids_to_delete.append(line_id)
            #     continue

            # check for merge
            line_ids_to_merge = self.get_line_ids_to_merge(line_id)
            if len(line_ids_to_merge) > 0:
                line_ids_to_merge.append(line_id)
                line_ids_to_merge.sort()
                line_id = line_ids_to_merge.pop(0)
                for line_id_to_merge in line_ids_to_merge:
                    self.lines[line_id].merge_lines(self.lines[line_id_to_merge])

                    print(
                        "Period Check: Merge line "
                        + str(line_id)
                        + " and line "
                        + str(line_id_to_merge)
                    )

                    line_ids_to_delete.append(line_id_to_merge)
                continue

        # delete lines
        for line_id_to_delete in np.unique(line_ids_to_delete):
            del self.lines[line_id_to_delete]

    def periodical_check_unasigned_events(self, t):
        # remove old events
        t_deletion_chain = t - self.config["chain_t_deletion_threshold"]
        unasigned_events_np = np.array(self.unasigned_events)
        if unasigned_events_np.size > 0:
            deletion_idx = np.argmax(unasigned_events_np[:, 2] > t_deletion_chain)
            self.unasigned_events = unasigned_events_np[deletion_idx:, :].tolist()

    def check_if_line_out_of_frame(self, line_id):
        line = self.lines[line_id]
        mid_point = line.point
        line_vel = line.vel

        if mid_point[0] > self.config["width"] - self.config["line_out_of_frame_band"]:
            if line_vel[0] > 0 or line.hibernate:
                print(
                    "Line "
                    + str(line_id)
                    + " of frame. Point: "
                    + str(mid_point)
                    + " Vel: "
                    + str(line_vel)
                )
                return True

        if mid_point[0] < self.config["line_out_of_frame_band"]:
            if line_vel[0] < 0 or line.hibernate:

                print(
                    "Line "
                    + str(line_id)
                    + " of frame. Point: "
                    + str(mid_point)
                    + " Vel: "
                    + str(line_vel)
                )

                return True

        if mid_point[1] > self.config["height"] - self.config["line_out_of_frame_band"]:
            if line_vel[1] > 0 or line.hibernate:
                print(
                    "Line "
                    + str(line_id)
                    + " of frame. Point: "
                    + str(mid_point)
                    + " Vel: "
                    + str(line_vel)
                )
                return True

        if mid_point[1] < self.config["line_out_of_frame_band"]:
            if line_vel[1] < 0 or line.hibernate:

                print(
                    "Line "
                    + str(line_id)
                    + " of frame. Point: "
                    + str(mid_point)
                    + " Vel: "
                    + str(line_vel)
                )
                return True

        return False

    def check_if_point_out_of_frame(self, point):
        if point[0] > self.config["width"] - self.config["line_out_of_frame_band"]:
            return True

        if point[0] < self.config["line_out_of_frame_band"]:
            return True

        if point[1] > self.config["height"] - self.config["line_out_of_frame_band"]:
            return True

        if point[1] < self.config["line_out_of_frame_band"]:
            return True

        return False
