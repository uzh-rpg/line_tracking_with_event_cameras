import numpy as np


class RefactoryPeriodFilter:
    def __init__(self, config):
        self.refactory_period_same_pol = config["filter_refactory_period_same_pol"]
        self.refactory_period_opposite_pol = config[
            "filter_refactory_period_opposite_pol"
        ]
        self.last_timestep_events = np.full(
            (2, config["height"], config["width"]),
            -self.refactory_period_same_pol,
            dtype=np.float64,
        )

    def update(self, x, y, t, p):
        p_opposite = 1 - p
        last_timestep = self.last_timestep_events[p, y, x]
        last_timestep_opp = self.last_timestep_events[p_opposite, y, x]

        if (t - last_timestep > self.refactory_period_same_pol) and (
            t - last_timestep_opp > self.refactory_period_opposite_pol
        ):
            self.last_timestep_events[p, y, x] = t
            return True
        else:
            return False


class NeighbouringEventsFilter:
    def __init__(self, config):
        self.size_neighbourhood = config["filter_size_neighbourhood"]
        self.num_events_neighbourhood = config["filter_num_events_neighbourhood"]
        self.time_period = config["filter_time_period"]
        self.height = config["height"]
        self.width = config["width"]
        self.offset = int((self.size_neighbourhood - 1) / 2)

        self.last_timestep_events = np.zeros(
            (2, config["height"], config["width"]), dtype=np.float64
        )

        self.last_timestep_event_candidates = np.zeros(
            (2, config["height"], config["width"]), dtype=np.float64
        )

    def update(self, x, y, t, p):

        x_left = 0 if x - self.offset < 0 else x - self.offset
        x_right = (
            (self.width - 1) if x + self.offset > self.width - 1 else x + self.offset
        ) + 1
        y_up = 0 if y - self.offset < 0 else y - self.offset
        y_down = (
            (self.height - 1) if y + self.offset > self.height - 1 else y + self.offset
        ) + 1

        window = self.last_timestep_events[p, y_up:y_down, x_left:x_right]
        window_diff = t - window
        neighbour_idx = np.where((window > 0) & (window_diff < self.time_period))
        num_neighbours = len(neighbour_idx[0])

        if num_neighbours < self.num_events_neighbourhood:
            window_candidates = self.last_timestep_event_candidates[
                p, y_up:y_down, x_left:x_right
            ]
            window_candidates_diff = t - window_candidates
            neighbour_candidate_idx = list(
                np.where(
                    (window_candidates > 0)
                    & (window_candidates_diff < self.time_period)
                )
            )

            polarity_idx = np.full(neighbour_candidate_idx[0].shape, p)
            neighbour_candidate_idx.insert(0, polarity_idx)

            num_neighbours_candidates = len(neighbour_candidate_idx[0])

            if (
                num_neighbours + num_neighbours_candidates
                >= self.num_events_neighbourhood
            ):

                neighbour_candidate_idx[1] = neighbour_candidate_idx[1] + y_up
                neighbour_candidate_idx[2] = neighbour_candidate_idx[2] + x_left
                neighbour_candidate_idx = tuple(neighbour_candidate_idx)
                self.last_timestep_events[
                    neighbour_candidate_idx
                ] = self.last_timestep_event_candidates[neighbour_candidate_idx]
                self.last_timestep_event_candidates[neighbour_candidate_idx] = 0

                self.last_timestep_events[p, y, x] = t
                return True
            else:
                self.last_timestep_event_candidates[p, y, x] = t
                return False
        else:
            self.last_timestep_events[p, y, x] = t
            return True
