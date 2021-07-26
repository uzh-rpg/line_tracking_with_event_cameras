import math
from operator import itemgetter
import numpy as np


def calculate_plane_parameters(events):

    cov_mat = np.cov(np.array(events).T, bias=True)
    eig_val, eig_vec = np.linalg.eigh(cov_mat)

    idx_sort = eig_val.argsort()
    eig_val = eig_val[idx_sort]
    eig_vec = eig_vec[:, idx_sort]

    plane_normal = eig_vec[:, 0]
    line_direction = np.array([plane_normal[1], -plane_normal[0]])
    line_direction = line_direction / np.linalg.norm(line_direction)
    std_dev_line = abs(np.dot(line_direction, eig_vec[:2, 1] * eig_val[1])) + abs(
        np.dot(line_direction, eig_vec[:2, 2] * eig_val[2])
    )
    line_length = math.sqrt(12 * std_dev_line)
    line_events_mean = np.mean(events, axis=0)

    return (plane_normal, eig_val, eig_vec, line_events_mean, line_length)


def calculate_line_parameters(events):
    cov_mat = np.cov(np.array(events).T, bias=True)
    eig_val, eig_vec = np.linalg.eigh(cov_mat)

    idx_sort = eig_val.argsort()
    eig_val = eig_val[idx_sort]
    eig_vec = eig_vec[:, idx_sort]
    line_normal = eig_vec[:, 0]
    line_direction = np.array([line_normal[1], -line_normal[0]])
    line_events_mean = np.mean(events, axis=0)

    return (line_direction, line_normal, eig_val, eig_vec, line_events_mean)


def calculate_distance_to_line(point_on_line, line_normal, point):
    distance = abs(np.dot(point[:2] - point_on_line[:2], line_normal[:2]))
    return distance


class Line:

    parameter_update_periodicity = 25
    update_p_time_threshold = 1
    num_events_for_line_promotion = 20
    num_events_parameter_update = 30
    line_promotion_threshold = 1
    events_removal_threshold = 100
    line_empty_bin_split_threshold = 2
    volume_depth = 40
    hibernate_density_treshold = 0.2
    hibernate_no_events_threshold = 5

    def __init__(self, normal, eig_val, eig_vec, cog, length, events, p):

        self.normal = normal  # 3D vector normal of inferred plane
        self.eig_val = eig_val  # sorted eigenvalues
        self.eig_vec = eig_vec  # sorted eigenvectors
        self.cog = cog  # 3D vector pointing to center of gravity of events
        self.length = length  # length of line
        self.events = events  # events beloning to line
        self.num_new_events = 0
        self.pol = p
        self.line_direction = np.array(
            [
                self.normal[1],
                -self.normal[0],
            ]
        )  # 2D vector pointing in the direction of the line
        self.line_direction = self.line_direction / np.linalg.norm(self.line_direction)
        self.new_event_density = self.get_new_events_density(self.events[-1][2])
        self.last_parameter_update = self.events[-1][2]

        self.c_to_p = np.array(
            [
                self.normal[0] * self.normal[2],
                self.normal[2] * self.normal[1],
                -math.pow(self.normal[0], 2) - math.pow(self.normal[1], 2),
            ]
        )

        self.vel = self.c_to_p[:2] / self.c_to_p[2]

        self.hibernate = False
        self.initializing = True
        self.initialization_period_end = events[-1][2] + self.events_removal_threshold

        self.update_p(self.events[-1][2])

    def update_p(self, t):

        if self.hibernate:
            self.point = self.cog
            self.point[2] = t
        else:
            self.point = self.cog + (
                (t - self.cog[2])
                / (-math.pow(self.normal[0], 2) - math.pow(self.normal[1], 2))
                * self.c_to_p
            )

    def update_plane_estimate(self):
        self.num_new_events = 0

        # only update if line is not hibernating
        if not self.hibernate:
            (
                self.normal,
                self.eig_val,
                self.eig_vec,
                self.cog,
                self.length,
            ) = calculate_plane_parameters(self.events)

            self.c_to_p = np.array(
                [
                    self.normal[0] * self.normal[2],
                    self.normal[2] * self.normal[1],
                    -math.pow(self.normal[0], 2) - math.pow(self.normal[1], 2),
                ]
            )

            self.line_direction = np.array(
                [
                    self.normal[1],
                    -self.normal[0],
                ]
            )
            self.line_direction = self.line_direction / np.linalg.norm(
                self.line_direction
            )
            self.vel = self.c_to_p[:2] / self.c_to_p[2]
            self.new_event_density = self.get_new_events_density(self.events[-1][2])
            self.last_parameter_update = self.events[-1][2]

    def get_distance_to_inferred_line(self, point):

        # if go into hibernation if no events have been assigned
        self.check_for_hibernation(point[2])

        # TODO figure out if this check makes code run faster
        if abs(point[2] - self.point[2]) > self.update_p_time_threshold:
            self.update_p(point[2])

        distance = abs(
            np.dot(point[:2] - self.point[:2], self.normal[:2])
        ) / np.linalg.norm(self.normal[:2])
        return distance

    def add_event(self, event):
        self.num_new_events += 1

        if (self.num_new_events > self.num_events_parameter_update) or (
            event[2] - self.last_parameter_update > self.parameter_update_periodicity
        ):
            self.check_for_hibernation(event[2])

            if not self.hibernate:
                self.remove_old_events(event[2])
                self.update_plane_estimate()

        for i, e in reversed(list(enumerate(self.events))):
            if event[2] > e[2]:
                self.events.insert(i + 1, event)
                return

            if i == 0:
                self.events.insert(0, event)
                return

    def remove_old_events(self, t):
        t_removal = t - self.events_removal_threshold
        for i, e in enumerate(self.events):
            if e[2] > t_removal:
                del self.events[:i]
                break
        return

    def check_for_hibernation(self, t):
        if self.initializing:
            self.hibernate = False
            return

        if (
            len(self.events) == 0
            or t - self.events[-1][2] > self.hibernate_no_events_threshold
            or self.get_new_events_density(t) < self.hibernate_density_treshold
        ):
            if self.hibernate == False:
                self.hibernate_start_time = t

            self.hibernate = True
        else:
            self.hibernate = False

    def get_new_events_density(self, t):
        t_volume_lower = t - self.volume_depth
        if len(self.events) == 0 or self.length == 0:
            return 0

        for i, e in reversed(list(enumerate(self.events))):
            if e[2] < t_volume_lower:
                num_new_events = len(self.events[i:])
                return float(num_new_events) / self.length

        return float(len(self.events)) / self.length

    def get_number_new_events(self, t, age):
        t_volume_lower = t - age
        for i, e in reversed(list(enumerate(self.events))):
            if e[2] < t_volume_lower:
                num_new_events = len(self.events[i:])
                return num_new_events

        return 0

    def merge_lines(self, line):
        self.events.extend(line.events)
        self.events.sort(key=itemgetter(2))
        self.update_plane_estimate()

    def get_connected_line_length(self):

        events = np.array(self.events)
        events_projection_inferred_line = np.matmul(
            events[:, :2] - self.point[:2],
            self.line_direction,
        )

        idx_sorted = np.argsort(events_projection_inferred_line)
        events_projection_inferred_line = events_projection_inferred_line[idx_sorted]
        events = events[idx_sorted, :]

        bin_start = math.floor(events_projection_inferred_line[0])
        bin_end = math.ceil(events_projection_inferred_line[-1])

        bin_range = bin_end - bin_start
        num_bins = math.ceil(bin_range)
        bins = bin_start + np.arange(0, num_bins + 1)
        idx_bins = np.digitize(events_projection_inferred_line, bins)
        idx_bins_unique = np.unique(idx_bins)

        max_bin_length = 0
        max_bin_start = 0
        max_bin_end = 0
        curr_bin_length = 0
        curr_bin_start = idx_bins_unique[0]
        for i in range(1, len(idx_bins_unique)):
            if (
                idx_bins_unique[i] - idx_bins_unique[i - 1]
                < self.line_empty_bin_split_threshold
            ):
                curr_bin_length += 1
            else:
                if curr_bin_length > max_bin_length:
                    max_bin_length = curr_bin_length
                    max_bin_start = curr_bin_start
                    max_bin_end = idx_bins_unique[i - 1]

                curr_bin_length = 0
                curr_bin_start = idx_bins_unique[i]

        if curr_bin_length > max_bin_length:
            max_bin_length = curr_bin_length
            max_bin_start = curr_bin_start
            max_bin_end = idx_bins_unique[-1]

        # remove events outside of connected line
        start_idx = np.argmax(events_projection_inferred_line > bins[max_bin_start])
        end_idx = np.argmax(events_projection_inferred_line > bins[max_bin_end] - 1)

        events = events[start_idx:end_idx, :]
        self.events = events.tolist()

        return max_bin_length

    def check_for_split(self):

        events_split_idx = -1

        # check for split
        events = np.array(self.events)
        events_projection_inferred_line = np.matmul(
            events[:, :2] - self.point[:2],
            self.line_direction,
        )

        events_projection_inferred_line = np.sort(events_projection_inferred_line)

        # create 2px bins such that first projection in the middle of bin and last bin contains last
        # projection
        bin_start = math.floor(events_projection_inferred_line[0])
        bin_end = math.ceil(events_projection_inferred_line[-1])

        bin_range = bin_end - bin_start
        num_bins = math.ceil(bin_range / 2.0)
        bins = bin_start + np.arange(0, num_bins + 1) * 2
        idx_bins = np.digitize(events_projection_inferred_line, bins)

        # check if bins are contiuous
        split_line = False
        idx_missing_bin = -1
        for i in range(1, len(idx_bins)):
            if idx_bins[i] - idx_bins[i - 1] > self.line_empty_bin_split_threshold:
                split_line = True
                idx_missing_bin = idx_bins[i] - 1
                break

        if split_line:
            projected_split_location = bins[idx_missing_bin]
            events_split_idx = np.argmax(
                events_projection_inferred_line > projected_split_location
            )

        return events_split_idx

    def check_line_requirements(self):
        if (
            len(self.events) > self.num_events_for_line_promotion
            and self.eig_val[0] < self.line_promotion_threshold
        ):
            return True
        else:
            return False


class Cluster:

    events_removal_threshold = 100e-3
    num_events_parameter_update = 5

    def __init__(self, events, pol, direction, normal, mean, eig_val):
        self.pol = pol
        self.events = events
        self.events.sort(key=itemgetter(2))
        self.direction = direction
        self.normal = normal
        self.mean = mean
        self.eig_val = eig_val
        self.num_new_events = 0

    def get_distance_to_nearest_neighbour(self, point):
        # ignore time difference
        events_spatial = np.array(self.events)[:, :2]
        difference_vectors = events_spatial - point[:2]
        min_distance = np.min(
            np.sum(np.abs(difference_vectors) ** 2, axis=-1) ** (1.0 / 2)
        )
        return min_distance

    def get_distance_to_inferred_line(self, point):
        return calculate_distance_to_line(self.mean, self.normal, point)

    def add_event(self, event):
        self.num_new_events += 1

        if self.num_new_events > self.num_events_parameter_update:
            self.update_inferred_line()

        for i, e in reversed(list(enumerate(self.events))):
            if event[2] > e[2]:
                self.events.insert(i + 1, event)
                return

            if i == 0:
                self.events.insert(0, event)
                return

    def remove_old_events(self, t):
        t_removal = t - self.events_removal_threshold
        for i, e in enumerate(self.events):
            if e[2] > t_removal:
                del self.events[:i]
                return

    def merge_clusters(self, cluster):
        # TODO can be improved since both list of events are already sorted
        self.events.extend(cluster.events)
        self.events.sort(key=itemgetter(2))

    def update_inferred_line(self):
        (
            self.direction,
            self.normal,
            self.eig_val,
            _,
            self.mean,
        ) = calculate_line_parameters([e[:2] for e in self.events])

    def get_spatial_normal(self):
        events_spatial = np.array(self.events)[:, :2]
        cov_mat = np.cov(events_spatial.T, bias=True)
        eig_val, eig_vec = np.linalg.eig(cov_mat)
        return eig_vec[eig_val.argmin()]
