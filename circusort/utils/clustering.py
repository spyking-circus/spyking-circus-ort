import scipy.optimize
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.spatial.distance
import scipy.stats
import statsmodels.api as sm
import logging

from sklearn.decomposition import PCA

from circusort.io.template import load_template
from circusort.obj.template import Template, TemplateComponent

def nd_bhatta_dist(X1, X2):

    mu_1 = np.mean(X1, 1)
    mu_2 = np.mean(X2, 1)
    ms = mu_1 - mu_2

    cov_1 = np.cov(X1)
    cov_2 = np.cov(X2)
    cov = (cov_1 + cov_2)/2

    det_1 = np.linalg.det(cov_1)
    det_2 = np.linalg.det(cov_2)
    det = np.linalg.det(cov)

    dist = (1/8.)*np.dot(np.dot(ms.T, np.linalg.inv(cov)), ms) + 0.5*np.log(det/np.sqrt(det_1*det_2))
    return dist

class DistanceMatrix(object):

    def __init__(self, size):
        self.size = size
        self.didx = lambda i,j: i*self.size + j - i*(i+1)//2 - i - 1

    def initialize(self, data, ydata=None):
        if ydata is None:
            self.distances = scipy.spatial.distance.pdist(data, 'euclidean').astype(np.float32)
        else:
            self.distances = scipy.spatial.distance.cdist(data, ydata, 'euclidean').astype(np.float32)

    def get_value(self, i, j):
        if i < j:
            return self.distances[self.didx(i, j)]
        elif i > j:
            return self.distances[self.didx(j, i)]
        elif i == j:
            return 0

    def get_row(self, i, with_diag=True):
        start = self.distances[self.didx(np.arange(0, i), i)]
        end = self.distances[self.didx(i, np.arange(i+1, self.size))]
        if with_diag:
            result = np.concatenate((start, np.array([0], dtype=np.float32), end))
        else:
            result = np.concatenate((start, end))
    
        return result

    def get_col(self, i, with_diag=True):
        return self.get_row(i, with_diag)

    def to_dense(self):
        return scipy.spatial.distance.squareform(self.distances)

    def get_rows(self, indices, with_diag=True):
        if with_diag:
            result = np.zeros((len(indices), self.size), dtype=np.float32)
        else:
            result = np.zeros((len(indices), self.size - 1), dtype=np.float32)

        for count, i in enumerate(indices):
            result[count] = self.get_row(i, with_diag)
        return result

    def get_cols(self, indices, with_diag=True):
        if with_diag:
            result = np.zeros((self.size, len(indices)), dtype=np.float32)
        else:
            result = np.zeros((self.size - 1, len(indices)), dtype=np.float32)

        for count, i in enumerate(indices):
            result[:, count] = self.get_col(i, with_diag)
        return result

    def get_deltas_and_neighbors(self, rho):
        """Find the distance to and the index of the nearest point with a higher density.

        Argument:
            rho
        Returns:
            nearest_higher_rho_distances
                For each point, distance to the nearest point with a higher density (i.e. delta).
            nearest_higher_rho_indices
                For each point, index of the nearest point with a higher density (i.e. neighbor).
        """

        indices = np.argsort(-rho)  # sort indices by decreasing rho values
        nearest_higher_rho_indices = np.zeros(self.size, dtype=np.int)  # i.e. neighbors
        nearest_higher_rho_distances = np.zeros(self.size, dtype=np.float32)  # i.e. deltas
        for k, index in enumerate(indices):
            higher_rho_indices = indices[0:k + 1]
            higher_rho_distances = self.get_row(index)[higher_rho_indices]
            higher_rho_distances[higher_rho_distances == 0.0] = float('inf')
            nearest_index = np.argmin(higher_rho_distances)
            nearest_higher_rho_indices[index] = higher_rho_indices[nearest_index]
            nearest_higher_rho_distances[index] = higher_rho_distances[nearest_index]

        if len(indices) > 1:
            nearest_higher_rho_distances[indices[0]] = np.max(nearest_higher_rho_distances[indices[1:]])
            nearest_higher_rho_distances[np.isinf(nearest_higher_rho_distances)] = 0

        return nearest_higher_rho_distances, nearest_higher_rho_indices

    @property
    def max(self):
        return np.max(self.distances)


class MacroCluster(object):
    """Macro cluster"""

    def __init__(self, id_, pca_data, data, creation_time=0):
        self.id = id_
        self.density = len(pca_data)
        self.sum_pca = np.sum(pca_data, 0)
        self.sum_pca_sq = np.sum(pca_data ** 2, 0)
        self.sum_full = np.sum(data, 0)
        self.creation_time = creation_time
        self.last_update = creation_time
        self.label = 'sparse'
        self.cluster_id = -1

    def __add__(self, other):
        """Add two clusters together.

        Argument:
            other: MacroCluster
                cluster to add to the current one.
        Return:
            result: MacroCluster

        """

        if not isinstance(other, MacroCluster):
            string = "unsupported operand type(s) for +: '{}' and '{}'"
            message = string.format(type(self), type(other))
            raise TypeError(message)

        pass

    def set_label(self, label):
        assert label in ['sparse', 'dense']
        self.label = label
        return

    @property
    def is_sparse(self):
        return self.label == 'sparse'

    @property
    def is_dense(self):
        return self.label == 'dense'

    def add(self, pca_data, data):
        self.density += 1.
        self.sum_pca += pca_data
        self.sum_pca_sq += pca_data ** 2
        self.sum_full += data
        return

    def update(self, time, decay_factor):
        if time > self.last_update:
            factor = 2 ** (-decay_factor * (time - self.last_update))
            self.density = factor * self.density
            self.sum_pca = factor * self.sum_pca
            self.sum_pca_sq = factor * self.sum_pca_sq
            self.sum_full = factor * self.sum_full
            self.last_update = time
        return

    def remove(self, pca_data, data):
        self.density -= 1
        self.sum_pca -= pca_data
        self.sum_pca_sq -= pca_data ** 2
        self.sum_full -= data
        return

    @property
    def center(self):
        return self.sum_pca / self.density

    @property
    def center_full(self):
        return self.sum_full / self.density

    @property
    def radius(self):
        return np.sqrt((self.sum_pca_sq / self.density - self.center ** 2).max())

    @property
    def description(self):
        return [self.center, self.radius]


class OnlineManager(object):

    def __init__(self, probe, channel, sampling_rate=20e+3, decay=0.05, mu=2, epsilon='auto', theta=-np.log(0.001),
                 dispersion=(5, 5), noise_thr=0.8, pca=None, sub_dim=5, logger=None, two_components=False, name=None,
                 debug_plots=None, debug_ground_truth_templates=None, debug_file_format='pdf', local_merges=2, smart_select='ransac', hanning_filtering=False):

        if name is None:
            self.name = "OnlineManager"
        else:
            self.name = name
        self.clusters = {}
        self.decay_factor = decay
        self.mu = mu
        self.epsilon = epsilon
        self.theta = theta
        self.dispersion = dispersion
        self.noise_thr = noise_thr
        self.glob_pca = pca
        self.probe = probe
        self.channel = channel
        self.two_components = two_components
        self.debug_plots = debug_plots
        self.debug_ground_truth_templates = debug_ground_truth_templates
        self.debug_file_format = debug_file_format
        self.local_merges = local_merges
        self.sampling_rate = sampling_rate
        self.smart_select_mode = smart_select
        self.hanning_filtering = hanning_filtering

        if self.debug_plots is not None:
            self.fig_name = os.path.join(self.debug_plots, '{n}_{t}.{f}')
            self.fig_name_2 = os.path.join(self.debug_plots, '{n}_{t}_tracking.{f}')

        self.time = 0
        self.is_ready = False
        self.abs_n_min = 20
        self.nb_updates = 0
        self.sub_dim = sub_dim
        self.loc_pca = None
        self.tracking = {}
        self.beta = 0.5
        if logger is None:
            self.log = logging.getLogger(__name__)
        else:
            self.log = logger

        # Define internal variables.
        self._width = None
        self._full_width = None
        self._W = None
        self._physical_threshold = None
        self._pc_lim = None

        # Log debug message
        string = "{} is created"
        message = string.format(self.name)
        self.log.debug(message)

        self.inodes = np.zeros(self.probe.total_nb_channels, dtype=np.int32)
        self.inodes[self.probe.nodes] = np.argsort(self.probe.nodes)
        self._channel_edges = self.inodes[self.probe.edges[self.probe.nodes[self.channel]]]

    @property
    def d_threshold(self):

        return self.mu * self.beta

    @property
    def time_gap(self):

        return np.ceil((1 / self.decay_factor) * np.log(self.d_threshold / (self.d_threshold - 1)))

    def initialize(self, time, snippets):
        """Initialize the clustering.

        Arguments:
            time: integer
                Creation time.
            snippets: circusort.obj.Snippets
                Data snippets which correspond to multiple peaks.
        """

        if self.hanning_filtering:
            snippets.filter()

        full_data = snippets.to_array()
        data = full_data[:, self._channel_edges, :]

        self.time = time

        if self.glob_pca is not None:
            sub_data = np.dot(data, self.glob_pca)
        else:
            sub_data = data

        a, b, c = sub_data.shape
        sub_data = sub_data.reshape(a, b * c)

        a, b, c = data.shape
        self._width = b * c
        data = data.reshape(a, self._width)
        a, b, c = full_data.shape
        self._full_width = b * c
        full_data = full_data.reshape(a, self._full_width)

        # Log debug message.
        string = "{} computes local PCA"
        message = string.format(self.name)
        self.log.debug(message)
        # Computes local PCA.
        pca = PCA(n_components=self.sub_dim)
        pca.fit(sub_data)
        self.loc_pca = pca.components_.T

        sub_data = np.dot(sub_data, self.loc_pca)

        n_min = self.abs_n_min

        if self.debug_plots is not None:
            output = self.fig_name.format(n=self.name, t=self.time, f=self.debug_file_format)
        else:
            output = None

        labels = self.density_clustering(sub_data, n_min=n_min, output=output, local_merges=self.local_merges)

        self._W = len(sub_data) / float(self.time / self.sampling_rate)
        self.mu = self._W / 1e+3
        self.beta = 1.5 / self.mu

        mask = labels > -1

        templates = {}

        # Log debug message.
        string = "{} founds {} initial clusters from {} data points"
        message = string.format(self.name, len(np.unique(labels[mask])), len(sub_data))
        self.log.info(message)

        epsilon = np.inf

        for count, cluster_id in enumerate(np.unique(labels[mask])):

            indices = np.where(labels == cluster_id)[0]
            data_cluster = full_data[indices]
            sub_data_cluster = sub_data[indices]

            self.clusters[count] = MacroCluster(count, sub_data_cluster, data_cluster, creation_time=self.time)
            self.clusters[count].cluster_id = count

            if self.clusters[count].radius < epsilon:
                epsilon = self.clusters[count].radius

            self.tracking[count] = self.clusters[count].description
            templates[count] = self._get_template(data_cluster)

        if self.epsilon == 'auto':
            self.epsilon = epsilon / 5.

        for cluster in self.clusters.values():
            if cluster.density >= self.d_threshold:
                cluster.set_label('dense')
            else:
                cluster.set_label('sparse')

        if self.debug_plots is not None:
            # Plot templates.
            filename = "{}_{}_templates.{}".format(self.name, self.time, self.debug_file_format)
            path = os.path.join(self.debug_plots, filename)
            self.plot_templates(templates, path)
            # Plot tracking.
            path = self.fig_name_2.format(n=self.name, t=self.time, f=self.debug_file_format)
            self.plot_tracking(self.dense_clusters, path)
            # Log info message.
            string = "{} creates output {}"
            message = string.format(self.name, path)
            self.log.info(message)
            if self.debug_ground_truth_templates is not None:
                # Plot ground truth clusters.
                self.plot_ground_truth_clusters(mode='max_channel')
                self.plot_ground_truth_clusters(mode='nonzero_channel')
                # Plot ground truth templates.
                self.plot_ground_truth_templates(mode='max_channel')
                self.plot_ground_truth_templates(mode='nonzero_channel')

        self.is_ready = True

        # Log debug message.
        string = "{} is initialized with {} templates"
        message = string.format(self.name, len(self.clusters))
        self.log.debug(message)

        return templates

    @property
    def nb_sparse(self):

        return len(self.sparse_clusters)

    @property
    def nb_dense(self):

        return len(self.dense_clusters)

    @property
    def nb_clusters(self):

        return len(self.clusters)

    @property
    def dense_clusters(self):

        return [i for i in self.clusters.values() if i.is_dense]

    @property
    def sparse_clusters(self):

        return [i for i in self.clusters.values() if i.is_sparse]

    def _update_clusters(self):

        for cluster in self.clusters.values():
            cluster.update(self.time, self.decay_factor)

        return

    def _get_id(self):

        if len(self.clusters) > 0:
            return np.max(list(self.clusters.keys())) + 1
        else:
            return 0

    def _get_tracking_id(self):

        if len(self.tracking) > 0:
            return np.max(list(self.tracking.keys())) + 1
        else:
            return 0

    def _get_clusters(self, cluster_type):

        if cluster_type == 'sparse':
            return self.sparse_clusters
        elif cluster_type == 'dense':
            return self.dense_clusters

    def _get_centers(self, cluster_type='dense'):

        centers = np.zeros((0, self.sub_dim), dtype=np.float32)
        for cluster in self._get_clusters(cluster_type):
            centers = np.vstack((centers, [cluster.center]))

        return centers

    def _get_centers_full(self, cluster_type='dense'):

        centers = np.zeros((0, self._full_width), dtype=np.float32)
        for cluster in self._get_clusters(cluster_type):
            centers = np.vstack((centers, [cluster.center_full]))

        return centers

    def _get_nearest_cluster(self, data, cluster_type='dense'):

        centers = self._get_centers(cluster_type)
        if len(centers) == 0:
            cluster = None
        else:
            new_dist = scipy.spatial.distance.cdist(data, centers, 'euclidean')[0]
            index = int(np.argmin(new_dist))
            cluster = self._get_clusters(cluster_type)[index]

        return cluster

    def _get_template(self, data):

        waveforms  = np.median(data, 0)
        amplitudes, full_ = self._compute_amplitudes(data, waveforms)
        waveforms *= np.median(full_)
        amplitudes, full_ = self._compute_amplitudes(data, waveforms)

        first_component = TemplateComponent(waveforms, self.probe.nodes, self.probe.nb_channels,
                                            amplitudes)
        if self.two_components:
            waveforms = self._compute_second_component(waveforms)
            second_component = TemplateComponent(waveforms, self.probe.nodes, self.probe.nb_channels)
        else:
            second_component = None

        template = Template(first_component, self.channel, second_component, creation_time=self.time)

        return template

    def _merged_into(self, pca_data, data, cluster_type):

        merged = False

        cluster = self._get_nearest_cluster(pca_data, cluster_type)
        if cluster is not None:
            cluster.add(pca_data[0], data[0])
            merged = cluster.radius <= self.epsilon

            if merged:
                if cluster.density >= self.d_threshold:
                    cluster.set_label('dense')
            else:
                cluster.remove(pca_data[0], data[0])

        return merged

    def _prune(self):

        count = 0
        for cluster in self.dense_clusters:
            if cluster.density < self.d_threshold:
                cluster.set_label('sparse')
                count += 1

        if count > 0:
            # Log debug message.
            string = "{} turns {} dense clusters into sparse..."
            message = string.format(self.name, count)
            self.log.debug(message)

        count = 0
        for cluster in self.sparse_clusters:

            t_0 = cluster.creation_time

            if 0 < t_0 < self.time:

                numerator = (2 ** (-self.decay_factor * (self.time - t_0 + self.time_gap)) - 1)
                denominator = (2 ** (-self.decay_factor * self.time_gap) - 1)
                zeta = numerator / denominator
                delta_t = self.theta * (cluster.last_update - t_0) / cluster.density

                if cluster.density < zeta or ((self.time - cluster.last_update) > delta_t):
                    # Log debug message.
                    string = "{} removes sparse cluster {}"
                    message = string.format(self.name, cluster.id)
                    self.log.debug(message)
                    # ...
                    self.clusters.pop(cluster.id)
                    count += 1

        if count > 0:
            # Log debug message.
            string = "{} prunes {} sparse clusters..."
            message = string.format(self.name, count)
            self.log.debug(message)

        return

    def time_to_cluster(self, nb_updates):

        return self.is_ready and self.nb_updates >= nb_updates

    def reduce_data(self, data):

        assert self.glob_pca is not None
        assert self.loc_pca is not None

        # Apply global PCA (i.e. channel by channel)
        reduced_data = np.dot(data, self.glob_pca)
        # Inline channel data.
        nb_samples, nb_channels, nb_global_components = reduced_data.shape
        assert nb_channels * nb_global_components == self.loc_pca.shape[0]
        reduced_data = reduced_data.reshape(nb_samples, self.loc_pca.shape[0])
        # Apply local PCA (i.e. over all channels)
        reduced_data = np.dot(reduced_data, self.loc_pca)

        return reduced_data

    def update(self, time, snippets=None):
        """Update clustering.

        Arguments:
            time: integer
                Creation time.
            snippets: none | circusort.obj.Snippets (optional)
                Data snippet which corresponds to one peak.
        """

        self.time = time
        self._update_clusters()
        # Log debug message.
        string = "{} processes time {} with {} sparse and {} dense clusters. Time gap is {}"
        message = string.format(self.name, time, self.nb_sparse, self.nb_dense, self.time_gap)
        self.log.debug(message)

        if snippets is not None:

            if self.hanning_filtering:
                snippets.filter()

            full_data = snippets.to_array()
            data = full_data[self._channel_edges, :]

            if self.glob_pca is not None:
                red_data = np.dot(data, self.glob_pca)
                full_data = full_data.reshape(1, self._full_width)
                red_data = red_data.reshape(1, self.loc_pca.shape[0])

                if self.loc_pca is not None:
                    pca_data = np.dot(red_data, self.loc_pca)
                else:
                    pca_data = red_data
            else:
                full_data = full_data.reshape(1, self._full_width)
                if self.loc_pca is not None:
                    pca_data = np.dot(data, self.loc_pca)
                else:
                    raise NotImplementedError()

            if self._merged_into(pca_data, full_data, 'dense'):
                # Log debug message.
                string = "{} merges the data at time {} into a dense cluster"
                message = string.format(self.name, self.time)
                self.log.debug(message)
            else:
                if self._merged_into(pca_data, full_data, 'sparse'):
                    # Log debug message.
                    string = "{} merges data at time {} into a sparse cluster"
                    message = string.format(self.name, self.time)
                    self.log.debug(message)
                else:
                    # Log debug message.
                    string = "{} can not merge data at time {} and creates a new sparse cluster"
                    message = string.format(self.name, self.time)
                    self.log.debug(message)
                    # Create new sparse cluster.
                    new_id = self._get_id()
                    new_cluster = MacroCluster(new_id, pca_data, full_data, creation_time=self.time)
                    new_cluster.set_label('sparse')
                    self.clusters[new_id] = new_cluster

            self.nb_updates += 1

        if np.mod(self.time, self.time_gap) < 1:
            self._prune()

        return

    def _perform_tracking(self, new_clusters):

        new_templates = {}
        modified_templates = {}

        if len(self.tracking) > 0:
            all_centers = np.array([i[0] for i in self.tracking.values()], dtype=np.float32)
            all_sigmas = np.array([i[1] for i in self.tracking.values()], dtype=np.float32)
            all_indices = list(self.tracking.keys())

            for cluster in new_clusters:
                cluster_id = cluster.cluster_id
                center, sigma = cluster.description
                new_dist = scipy.spatial.distance.cdist(np.array([center]), all_centers, 'euclidean')
                dist_min = np.min(new_dist)
                dist_idx = int(np.argmin(new_dist))

                if dist_min <= (sigma + all_sigmas[dist_idx]):
                    self.tracking[all_indices[dist_idx]] = center, sigma
                    modified_templates[cluster_id] = all_indices[dist_idx]
                    cluster.cluster_id = all_indices[dist_idx]
                    # Log debug message.
                    string = "{} establishes a match between target {} and source {}"
                    message = string.format(self.name, cluster_id, all_indices[dist_idx])
                    self.log.debug(message)
                else:
                    idx = self._get_tracking_id()
                    self.tracking[idx] = center, sigma
                    new_templates[cluster_id] = idx
                    cluster.cluster_id = idx
                    # Log debug message.
                    string = "{} can not found a match for target {}, so creating new template {}"
                    message = string.format(self.name, cluster_id, idx)
                    self.log.debug(message)
        else:
            for cluster in new_clusters:
                cluster_id = cluster.cluster_id
                idx = self._get_tracking_id()
                self.tracking[idx] = cluster.description
                new_templates[cluster_id] = idx
                cluster.cluster_id = idx

            # Log debug message.
            string = "nothing to track: {} found {} new templates"
            message = string.format(self.name, len(new_clusters))
            self.log.debug(message)

        return new_templates, modified_templates

    def set_threshold(self, threshold):

        # TODO check if threshold is normalized.

        self._physical_threshold = self.noise_thr * threshold

        return

    def _compute_amplitudes(self, data, template):

        # # We could to this in the PCA space, to speed up the computation
        temp_flat = template.reshape(template.size, 1)
        amplitudes = np.dot(data, temp_flat)
        amplitudes /= np.sum(temp_flat ** 2)
        variation = np.median(np.abs(amplitudes - 1))

        amp_min = min(0.8, max(self._physical_threshold, np.median(amplitudes) - self.dispersion[0] * variation))
        amp_max = max(1.2, np.median(amplitudes) + self.dispersion[1] * variation)

        return np.array([amp_min, amp_max], dtype=np.float32), amplitudes

    @staticmethod
    def _compute_second_component(waveforms):

        return np.diff(waveforms).reshape(1, waveforms.size)

    def cluster(self, tracking=True):

        # Log info message.
        string = "{} launches clustering with {} sparse and {} dense clusters"
        message = string.format(self.name, self.nb_sparse, self.nb_dense)
        self.log.info(message)

        centers = self._get_centers('dense')
        centers_full = self._get_centers_full('dense')

        if self.debug_plots is not None:
            output = self.fig_name.format(n=self.name, t=self.time, f=self.debug_file_format)
        else:
            output = None

        labels = self.density_clustering(centers, n_min=0, output=output, local_merges=self.local_merges)

        self.nb_updates = 0

        mask = labels > -1
        clusters = []
        for l in np.unique(labels[mask]):
            idx = np.where(labels == l)[0]
            cluster = MacroCluster(-1, centers[idx], centers_full[idx])
            cluster.cluster_id = l
            clusters += [cluster]

        new_templates, modified_templates = self._perform_tracking(clusters)

        templates = {}

        for key, value in new_templates.items():
            data = centers_full[labels == key]
            templates[value] = self._get_template(data)
        # Log info message.
        string = "{} found {} new templates: {}"
        message = string.format(self.name, len(new_templates), new_templates.keys())
        self.log.info(message)

        if tracking:
            for key, value in modified_templates.items():
                data = centers_full[labels == value]
                templates[value] = self._get_template(data)
            # Log info message.
            string = "{} tracked {} modified templates with: {}"
            message = string.format(self.name, len(modified_templates), modified_templates.values())
            self.log.info(message)

        if self.debug_plots is not None:
            path = self.fig_name_2.format(n=self.name, t=self.time, f=self.debug_file_format)
            self.plot_tracking(clusters, path)
            # Log info message.
            string = "{} creates output {}"
            message = string.format(self.name, path)
            self.log.info(message)

        return templates

    @staticmethod
    def rho_estimation(data, neighbors_ratio=0.01):
        """Estimation of the rho values for the density clustering.

        The rho value of a sample corresponds to the mean distances between this samples and its nearest neighbors.

        Arguments:
            data: np.ndarray
                An array which contains the data sample. The size of the first dimension must be equal to the number of
                samples.
            neighbors_ratio: float (optional)
                Determine the number of neighbors to use during the estimation of rho values. The number of neighbors is
                equal to the number of samples multiplied by this ratio and is at least equal to 5.
                The default value is 0.01.
        Returns:
            rho: np.ndarray
                The rho values.
            distances: np.ndarray
                The condensed matrix of distances between pair of samples.
            nb_neighbors: integer
                The number of neighbors.
        """
        distances = DistanceMatrix(size=len(data))
        distances.initialize(data)
        dist_sorted = {}
        rho = np.zeros(len(data), dtype=np.float32)

        N = len(data)
        nb_neighbors = max(2, int(neighbors_ratio*N))

        for i in range(N):
            data = distances.get_row(i, with_diag=False)
            if len(data) > nb_neighbors:
                dist_sorted[i] = data[np.argpartition(data, nb_neighbors)[:nb_neighbors]]
            else:
                dist_sorted[i] = data
            rho[i] = np.mean(dist_sorted[i])

        rho = -rho + rho.max()
        
        return rho, distances, dist_sorted

    def fit_rho_delta(self, rho, delta, smart_select_mode='ransac'):
        """Fit relation between rho and delta values.

        Arguments:
            rho: np.ndarray
            delta: np.ndarray
            smart_select: boolean (optional)
                If true then we will try to detect automatically the clusters based on the rho and delta values.
                The default value is False.
            max_clusters: integer (optional)
                The maximal number of detected clusters (except if smart select is activated).
                The default value is 10.
            smart_select_mode: string (optional)
                Either 'curve_fit' or 'ransac'.
        """

        if smart_select_mode == 'curve_fit':

            z_score_threshold = 3.0

            rho_max = rho.max()
            off = delta.min()
            idx = np.argmin(rho)
            a_0 = (delta[idx] - off) / np.log(1 + (rho_max - rho[idx]))

            def my_func(t, a, b, c, d):
                return a * np.log(1.0 + c * ((rho_max - t) ** b)) + d  # TODO fix runtime warning...

            try:
                result, p_cov = scipy.optimize.curve_fit(my_func, rho, delta, p0=[a_0, 1., 1., off])
                prediction = my_func(rho, result[0], result[1], result[2], result[3])
                difference = rho * (delta - prediction)
                # TODO swap and clean the following lines.
                # z_score = (difference - difference.mean()) / difference.std()
                difference_median = np.median(difference)
                difference_mad = 1.4826 * np.median(np.absolute(difference - difference_median))
                z_score = (difference - difference_median) / difference_mad
                sub_indices = np.where(z_score >= z_score_threshold)[0]
                # Plot rho and delta values (if necessary).
                if self.debug_plots is not None:
                    labels = z_score >= z_score_threshold
                    self.plot_rho_delta(rho, delta - prediction, labels=labels, filename_suffix="modified")
            except Exception as exception:
                # Log debug message.
                string = "{} raises an error in 'fit_rho_delta': {}"
                message = string.format(self.name, exception)
                self.log.debug(message)
                sub_indices = np.argsort(rho * np.log(1 + delta))[::-1][:max_clusters]

        elif smart_select_mode == 'ransac':

            z_score_threshold = 3.0
            x = sm.add_constant(rho)
            model = sm.RLM(delta, x)
            results = model.fit()
            prediction = results.fittedvalues
            difference = delta - prediction
            # TODO swap and clean the following lines.
            # z_score = (difference - difference.mean()) / difference.std()
            difference_median = np.median(difference)
            difference_mad = 1.4826 * np.median(np.absolute(difference - difference_median))
            z_score = (difference - difference_median) / difference_mad
            sub_indices = np.where(z_score >= z_score_threshold)[0]
            # Plot rho and delta values (if necessary).
            if self.debug_plots is not None:
                labels = z_score >= z_score_threshold
                self.plot_rho_delta(rho, delta - prediction, labels=labels, filename_suffix="modified")

        elif smart_select_mode == 'ransac_bis':
            x = sm.add_constant(rho)
            model = sm.RLM(delta, x)
            results = model.fit()
            delta_mean = results.fittedvalues
            difference = delta - delta_mean
            # 1st solution.
            # sigma = difference.std()  # TODO this estimation is sensible to outliers
            # 2nd solution.
            # difference_median = np.median(difference)
            # difference_mad = 1.4826 * np.median(np.absolute(difference - difference_median))
            # sigma = difference_mad
            # upper = + 3.0 * sigma  # + sigma * prediction  # TODO check/optimize this last term?
            # 3rd solution (fit variance also).
            sigma = 1.4826 * np.median(np.absolute(difference - np.median(difference)))
            variance_model = sm.RLM(np.square(difference), x)
            variance_result = variance_model.fit()
            variance_delta = variance_result.fittedvalues
            variance_delta = np.maximum(sigma ** 2.0, variance_delta)  # i.e. rectify negative values
            delta_std = np.sqrt(variance_delta)
            upper = 5.0 * delta_std
            # lower = - 3.0 * sigma  # - sigma * prediction  # TODO idem?
            z_score = difference - upper  # TODO define correctly the z-score.
            sub_indices = np.where(z_score >= 0)[0]  # TODO reintroduce the z-score threshold.
            # Plot rho and delta values (if necessary).
            if self.debug_plots is not None:
                labels = z_score >= 0
                self.plot_rho_delta(rho, delta, labels=labels, mean=delta_mean, std=delta_std,
                                        threshold=delta_mean + upper, filename_suffix="ransac_bis")

        else:
            string = "unexpected smart select mode: {}"
            message = string.format(smart_select_mode)
            raise ValueError(message)

        return sub_indices

    def density_based_clustering(self, rho, distances, n_min, smart_select_mode='ransac_bis', refine=True):
        delta, neighbors = self.compute_delta_and_neighbors(distances, rho)
        nclus, labels, centers = self.find_centroids_and_cluster(distances, rho, delta, neighbors, smart_select_mode)
        halolabels = self.halo_assign(labels, rho, n_min, 3)
        halolabels -= 1
        centers = np.where(centers - 1 >= 0)[0]

        return halolabels

    @staticmethod
    def compute_delta_and_neighbors(dist, rho):
        return dist.get_deltas_and_neighbors(rho)

    def find_centroids_and_cluster(self, dist, rho, delta, neighbors, smart_select_mode, method='nearest_denser_point'):

        nb_points = len(rho)
        # Find centroids.
        centroids = np.zeros(nb_points, dtype=np.int)
        centroid_indices = self.fit_rho_delta(rho, delta, smart_select_mode)
        nb_clusters = len(centroid_indices)
        cluster_nbs = np.arange(1, nb_clusters + 1)
        centroids[centroid_indices] = cluster_nbs  # assigning cluster numbers to centroids
        # Assign each point to one cluster.
        if method == 'nearest_centroid':
            # Custom (and naive) method.
            if nb_clusters <= 1:
                labels = np.ones(nb_points, dtype=np.int)  # all points in one cluster
            else:
                distances_to_centroids = dist.get_rows(centroid_indices)
                labels = np.argmin(distances_to_centroids, axis=0) + 1
        elif method == 'nearest_denser_point':
            # Method described in [Rodriguez & Laio (2014)](https://science.sciencemag.org/content/344/6191/1492.full).
            if nb_clusters <= 1:
                labels = np.ones(nb_points, dtype=np.int)  # all points in one cluster
            else:
                labels = np.copy(centroids)
                indices = np.argsort(-rho)  # sort indices by decreasing density
                for index in indices:
                    if labels[index] == 0:
                        labels[index] = labels[neighbors[index]]
        else:
            raise ValueError("unexpected value %s" % method)

        return nb_clusters, labels, centroids
        
    @staticmethod
    def halo_assign(labels, rhos, n_min, halo_rejection=3):
        """Unassign outliers."""

        halolabels = labels.copy()
        for label_nb in np.unique(labels):
            indices = np.where(labels == label_nb)[0]
            median_rho = np.median(rhos[indices])
            # selected_indices = indices[rhos[indices] < median_rho]
            mad_rho = np.median(np.abs(rhos[indices] - median_rho))
            selected_indices = indices[rhos[indices] < (median_rho - halo_rejection*mad_rho)]  # TODO enhance?
            if len(indices) - len(selected_indices) > n_min:
                halolabels[selected_indices] = 0  # i.e. set to 0 (unassign)
        return halolabels

    @staticmethod
    def _do_merging(data, labels, clusters, local_merges):

        d_min = np.inf
        to_merge = [None, None]

        for ic1 in range(len(clusters)):
            idx1 = np.where(labels == clusters[ic1])[0]
            if len(idx1) > 0:
                sd1 = np.take(data, idx1, axis=0)
                for ic2 in range(ic1 + 1, len(clusters)):
                    idx2 = np.where(labels == clusters[ic2])[0]
                    if len(idx2) > 0:
                        sd2 = np.take(data, idx2, axis=0)
                        try:
                            dist = nd_bhatta_dist(sd1.T, sd2.T)
                        except Exception:
                            dist = np.inf

        if d_min < local_merges:
            labels[np.where(labels == clusters[to_merge[1]])[0]] = clusters[to_merge[0]]
            return True, labels

        return False, labels

    def _greedy_merges(self, data, labels, local_merges):

        has_been_merged = True
        mask = np.where(labels > -1)[0]
        clusters = np.unique(labels[mask])
        merged = [len(clusters), 0]

        while has_been_merged:
            has_been_merged, labels = self._do_merging(data, labels, clusters, local_merges)
            if has_been_merged:
                merged[1] += 1

        return labels, merged

    @staticmethod
    def _discard_small_clusters(labels, n_min):

        for cluster_index in np.unique(labels):
            if cluster_index < 0:
                continue
            samples_indices = np.where(labels == cluster_index)[0]
            if len(samples_indices) < n_min:
                labels[samples_indices] = -1

        return labels

    def density_clustering(self, data, n_min=None, output=None, local_merges=None):
        """Run a density clustering on the given data.

        Arguments:
            data: np.ndarray
            n_min: none | float (optional)
                Minimal number in any cluster.
                The default value is None.
            output: none | string (optional)
                The path used to create the debug plots.
                The default value is None.
            local_merges: none | float (optional)
                Threshold for merging clusters on this electrode (i.e. similar clusters).
                The default value is None.
        Return:
            labels: np.ndarray
                An array which contains the cluster labels of the data samples.
        """

        nb_samples = len(data)

        if nb_samples > 1:
            rhos, distances, dist_sorted = self.rho_estimation(data)
            labels = self.density_based_clustering(rhos, distances, n_min, refine=False)
        elif nb_samples == 1:
            labels = np.array([0])
        else:
            labels = np.array([])

        # Merge similar clusters (if necessary).
        if local_merges is not None:
            labels, _ = self._greedy_merges(data, labels, local_merges)

        # Discard small clusters (if necessary).
        if n_min is not None:
            labels = self._discard_small_clusters(labels, n_min)

        if output is not None:
            self.plot_cluster(data, labels, output)

        return labels

    def _compute_pc_limits(self, data):

        self._pc_lim = []
        for k in range(0, data.shape[1]):
            pc_min = np.min(data[:, k])
            pc_max = np.max(data[:, k])
            pc_range = pc_max - pc_min
            pc_min = pc_min - 0.1 * pc_range
            pc_max = pc_max + 0.1 * pc_range
            self._pc_lim.append((pc_min, pc_max))

        return

    def plot_rho_delta(self, rho, delta, labels=None, mean=None, std=None, threshold=None, marker_size=5,
                       marker_color='C0', filename_suffix=None):

        fig, ax = plt.subplots()
        if labels is not None:
            for k, label in enumerate(np.unique(labels)):
                indices = np.where(labels == label)[0]
                marker_color = "C{}".format(k % 10)
                ax.scatter(rho[indices], delta[indices], s=marker_size, c=marker_color,
                           label="data samples {}".format(label))
        else:
            ax.scatter(rho, delta, s=marker_size, c=marker_color, label="data samples")
        indices = np.argsort(rho)
        if mean is not None:
            ax.plot(rho[indices], mean[indices], color='red', linestyle='--', linewidth=1, label="mean")
        if std is not None:
            ax.plot(rho[indices], mean[indices] + std[indices], color='red', linestyle=':', linewidth=1,
                    label="mean+std")
            ax.plot(rho[indices], mean[indices] - std[indices], color='red', linestyle=':', linewidth=1,
                    label="mean-std")
        if threshold is not None:
            ax.plot(rho[indices], threshold[indices], color='red', linestyle='-', linewidth=1, label="threshold")
        ax.set_xlabel("rho")
        ax.set_ylabel("delta")

        if filename_suffix is None:
            filename = "{}_{}_rho_delta.{}".format(self.name, self.time, self.debug_file_format)
        else:
            filename = "{}_{}_rho_delta_{}.{}".format(self.name, self.time, filename_suffix, self.debug_file_format)
        path = os.path.join(self.debug_plots, filename)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

        return

    def plot_cluster(self, data, labels, output, marker_size=5, with_ticks=True):

        if self._pc_lim is None:
            self._compute_pc_limits(data)

        text_kwargs = {
            'fontsize': 'xx-small',
            'verticalalignment': 'center',
            'horizontalalignment': 'center',
        }

        fig, ax_array = plt.subplots(nrows=2, ncols=2)

        # 1st subplot.
        k_1, k_2 = 0, 1  # pair of principal components
        ax = ax_array[0, 0]
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[np.where(unique_labels >= 0)[0]]  # i.e. remove -1 label
        for k, label in enumerate(unique_labels):
            c = 'C{}'.format(k % 10)
            indices = np.where(labels == label)[0]
            x = data[indices, k_1]
            y = data[indices, k_2]
            ax.scatter(x, y, s=marker_size, c=c)
            ax.text(np.median(x), np.median(y), "{}".format(label), **text_kwargs)
        ax.set_xlim(*self._pc_lim[k_1])
        ax.set_ylim(*self._pc_lim[k_2])
        if not with_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_xlabel("PC{}".format(k_1))
        ax.set_ylabel("PC{}".format(k_2))
        # 2nd subplot.
        k_1, k_2 = 2, 1
        ax = ax_array[0, 1]
        for k, label in enumerate(unique_labels):
            c = 'C{}'.format(k % 10)
            indices = np.where(labels == label)[0]
            x = data[indices, k_1]
            y = data[indices, k_2]
            ax.scatter(x, y, s=marker_size, c=c)
            ax.text(np.median(x), np.median(y), "{}".format(label), **text_kwargs)
        ax.set_xlim(*self._pc_lim[k_1])
        ax.set_ylim(*self._pc_lim[k_2])
        if not with_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_xlabel("PC{}".format(k_1))
        ax.set_ylabel("PC{}".format(k_2))
        # 3rd subplot.
        k_1, k_2 = 0, 2
        ax = ax_array[1, 0]
        for k, label in enumerate(unique_labels):
            c = 'C{}'.format(k % 10)
            indices = np.where(labels == label)[0]
            x = data[indices, k_1]
            y = data[indices, k_2]
            ax.scatter(x, y, s=marker_size, c=c)
            ax.text(np.median(x), np.median(y), "{}".format(label), **text_kwargs)
        ax.set_xlim(*self._pc_lim[k_1])
        ax.set_ylim(*self._pc_lim[k_2])
        if not with_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_xlabel("PC{}".format(k_1))
        ax.set_ylabel("PC{}".format(k_2))
        # 4th subplot.
        ax = ax_array[1, 1]
        ax.axis('off')

        plt.savefig(output)
        plt.close()

        return

    # TODO swap and clean the 2 following methods.

    # def plot_templates(self, templates, output):
    #     # Version: one figure for all the templates.
    #
    #     nb_templates = len(templates)
    #
    #     _, ax = plt.subplots(nrows=1, ncols=nb_templates)
    #     for k in range(0, nb_templates):
    #         template = templates[k]
    #         color = 'C{}'.format(k % 10)
    #         template.plot(ax=ax[k], probe=self.probe, color=color)
    #
    #     plt.savefig(output)
    #     plt.close()
    #
    #     return

    def plot_templates(self, templates, output, mode='multiple_files'):
        """Plot templates.

        Arguments:
            templates
            output
            mode: string (optional)
                Either 'single_file' or 'multiple_files'.
                The default value is 'multiple_files'.
        """

        nb_templates = len(templates)

        if mode == 'single_file':  # i.e. one figure with all the templates

            _, ax = plt.subplots(nrows=1, ncols=nb_templates)
            for k in range(0, nb_templates):
                template = templates[k]
                color = 'C{}'.format(k % 10)
                template.plot(ax=ax[k], probe=self.probe, color=color)

            plt.savefig(output)
            plt.close()

        elif mode == 'multiple_files':  # i.e. one figure per template

            base_path, extension = os.path.splitext(output)

            for k in range(0, nb_templates):
                _, ax = plt.subplots()
                template = templates[k]
                color = 'C{}'.format(k % 10)
                template.plot(ax=ax, probe=self.probe, color=color)
                path = base_path + "_{}".format(k) + extension
                plt.savefig(path)
                plt.close()

        else:  # raise error message

            string = "unexpected mode value: {}"
            message = string.format(mode)
            raise ValueError(message)

        return

    def plot_tracking(self, dense_clusters, output, marker='+', marker_size=5):

        indices = []
        centers = []
        sigmas = []
        colors = []
        for k, item in enumerate(dense_clusters):
            indices += [k]
            colors += [item.cluster_id]
            centers += [item.description[0]]
            sigmas += [item.description[1]]
        indices = np.array(indices)
        centers = np.array(centers)
        sigmas = np.array(sigmas)
        colors = np.array(colors, dtype=np.int32)

        if len(centers) > 0:

            fig, ax_array = plt.subplots(nrows=2, ncols=2)

            # 1st subplot.
            k_1, k_2 = 0, 1  # pair of principal components
            ax = ax_array[0, 0]
            for index, center, sigma, color in zip(indices, centers, sigmas, colors):
                c = 'C{}'.format(color % 10)
                circle = plt.Circle((center[k_1], center[k_2]), sigma, color=c, fill=True, alpha=0.5,
                                    label="std {}".format(index))
                ax.add_artist(circle)
                ax.scatter(centers[:, k_1], centers[:, k_2], s=marker_size, c='k', marker=marker,
                           label="mean {}".format(index))
                if self._pc_lim is not None:
                    ax.set_xlim(*self._pc_lim[k_1])
                    ax.set_ylim(*self._pc_lim[k_2])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel("PC{}".format(k_1))
                ax.set_ylabel("PC{}".format(k_2))
            # 2nd subplot.
            k_1, k_2 = 2, 1  # pair of principal components
            ax = ax_array[0, 1]
            for index, center, sigma, color in zip(indices, centers, sigmas, colors):
                c = 'C{}'.format(color % 10)
                circle = plt.Circle((center[k_1], center[k_2]), sigma, color=c, fill=True, alpha=0.5,
                                    label="std {}".format(index))
                ax.add_artist(circle)
                ax.scatter(centers[:, k_1], centers[:, k_2], s=marker_size, c='k', marker=marker,
                           label="mean {}".format(index))
                if self._pc_lim is not None:
                    ax.set_xlim(*self._pc_lim[k_1])
                    ax.set_ylim(*self._pc_lim[k_2])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel("PC{}".format(k_1))
                ax.set_ylabel("PC{}".format(k_2))
            # 3rd subplot.
            k_1, k_2 = 0, 2  # pair of principal components
            ax = ax_array[1, 0]
            for index, center, sigma, color in zip(indices, centers, sigmas, colors):
                c = 'C{}'.format(color % 10)
                circle = plt.Circle((center[k_1], center[k_2]), sigma, color=c, fill=True, alpha=0.5,
                                    label="std {}".format(index))
                ax.add_artist(circle)
                ax.scatter(centers[:, k_1], centers[:, k_2], s=marker_size, c='k', marker=marker,
                           label="mean {}".format(index))
                if self._pc_lim is not None:
                    ax.set_xlim(*self._pc_lim[k_1])
                    ax.set_ylim(*self._pc_lim[k_2])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel("PC{}".format(k_1))
                ax.set_ylabel("PC{}".format(k_2))
            # 4th subplot.
            ax = ax_array[1, 1]
            ax.axis('off')

        plt.savefig(output)
        plt.close()

        return

    def plot_ground_truth_clusters(self, mode='max_channel', marker_size=5, with_ticks=True):
        """Plot ground truth clusters.

        Argument:
            mode: string (optional)
                The mode can be:
                    'max_channel': consider templates having their maximal deflection on the channel handle by this
                        online manager
                    'nonzero_channel': consider templates showing a deflection on the channel handle by this online
                        manager
                The default value is 'max_channel'.
            marker_size: integer (optional)
                The default value is 10.
            with_ticks: boolean (optional)
                The default value is True.
        """

        templates = [
            load_template(path)
            for path in self.debug_ground_truth_templates
        ]

        if mode == 'max_channel':
            main_indices = [
                k
                for k, template in enumerate(templates)
                if template.channel == self.channel
            ]
            main_data = np.array([
                template.first_component.to_dense()
                for template in templates
                if template.channel == self.channel
            ])
            other_data = np.array([
                template.first_component.to_dense()
                for template in templates
                if template.channel != self.channel
            ])
        elif mode == 'nonzero_channel':
            main_indices = [
                k
                for k, template in enumerate(templates)
                if self.channel in template.indices
            ]
            main_data = np.array([
                template.first_component.to_dense()
                for template in templates
                if self.channel in template.indices
            ])
            other_data = np.array([
                template.first_component.to_dense()
                for template in templates
                if self.channel not in template.indices
            ])
        else:
            string = "unexpected mode value: {}"
            message = string.format(mode)
            raise ValueError(message)

        main_reduced_data = self.reduce_data(main_data)
        other_reduced_data = self.reduce_data(other_data)

        main_marker_colors = [
            'C{}'.format(k % 10)
            for k in range(0, main_reduced_data.shape[0])
        ]

        if self._pc_lim is None:
            self._compute_pc_limits(main_reduced_data)

        text_kwargs = {
            'fontsize': 'xx-small',
            'verticalalignment': 'center',
            'horizontalalignment': 'center',
        }

        fig, ax_array = plt.subplots(2, 2)

        # 1st subplot.
        k_1, k_2 = 0, 1  # pair of principal components
        ax = ax_array[0, 0]
        ax.scatter(main_reduced_data[:, k_1], main_reduced_data[:, k_2], s=marker_size, c=main_marker_colors)
        ax.scatter(other_reduced_data[:, k_1], other_reduced_data[:, k_2], s=marker_size / 2, c='gray')
        for k, index in enumerate(main_indices):
            x = main_reduced_data[k, k_1]
            y = main_reduced_data[k, k_2]
            s = "{}".format(index)
            ax.text(x, y, s, **text_kwargs)
        ax.set_xlim(*self._pc_lim[k_1])
        ax.set_ylim(*self._pc_lim[k_2])
        if not with_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_xlabel("PC{}".format(k_1))
        ax.set_ylabel("PC{}".format(k_2))
        # 2nd subplot.
        k_1, k_2 = 2, 1
        ax = ax_array[0, 1]
        ax.scatter(main_reduced_data[:, k_1], main_reduced_data[:, k_2], s=marker_size, c=main_marker_colors)
        ax.scatter(other_reduced_data[:, k_1], other_reduced_data[:, k_2], s=marker_size / 2, c='gray')
        for k, index in enumerate(main_indices):
            x = main_reduced_data[k, k_1]
            y = main_reduced_data[k, k_2]
            s = "{}".format(index)
            ax.text(x, y, s, **text_kwargs)
        ax.set_xlim(*self._pc_lim[k_1])
        ax.set_ylim(*self._pc_lim[k_2])
        if not with_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_xlabel("PC{}".format(k_1))
        ax.set_ylabel("PC{}".format(k_2))
        # 3rd subplot.
        k_1, k_2 = 0, 2
        ax = ax_array[1, 0]
        ax.scatter(main_reduced_data[:, k_1], main_reduced_data[:, k_2], s=marker_size, c=main_marker_colors)
        ax.scatter(other_reduced_data[:, k_1], other_reduced_data[:, k_2], s=marker_size / 2, c='gray')
        for k, index in enumerate(main_indices):
            x = main_reduced_data[k, k_1]
            y = main_reduced_data[k, k_2]
            s = "{}".format(index)
            ax.text(x, y, s, **text_kwargs)
        ax.set_xlim(*self._pc_lim[k_1])
        ax.set_ylim(*self._pc_lim[k_2])
        if not with_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_xlabel("PC{}".format(k_1))
        ax.set_ylabel("PC{}".format(k_2))
        # 4th subplot.
        ax = ax_array[1, 1]
        ax.axis('off')

        filename = "{}_{}_ground_truth_clusters_{}.{}".format(self.name, self.time, mode, self.debug_file_format)
        path = os.path.join(self.debug_plots, filename)
        plt.savefig(path)
        plt.close()

        return

    def plot_ground_truth_templates(self, mode='max_channel'):

        templates = [
            load_template(path)
            for path in self.debug_ground_truth_templates
        ]

        if mode == 'max_channel':
            main_indices = [
                k
                for k, template in enumerate(templates)
                if template.channel == self.channel
            ]
        elif mode == 'nonzero_channel':
            main_indices = [
                k
                for k, template in enumerate(templates)
                if self.channel in template.indices
            ]
        else:
            string = "unexpected mode value: {}"
            message = string.format(mode)
            raise ValueError(message)

        nb_templates = len(main_indices)
        a = np.sqrt(float(nb_templates) / 6.0)
        nb_columns = int(np.ceil(3.0 * a))
        nb_rows = int(np.ceil(float(nb_templates) / float(nb_columns)))
        nb_subplots = nb_rows * nb_columns

        fig, ax = plt.subplots(nrows=nb_rows, ncols=nb_columns, figsize=(3.0 * 6.4, 2.0 * 4.8))
        for k, index in enumerate(main_indices):
            i, j = k // nb_columns, k % nb_columns
            try:
                ax_ = ax[i, j]
            except IndexError:
                try:
                    ax_ = ax[k]
                except IndexError:
                    ax_ = ax
            template = templates[index]
            title = "Template {}".format(index)
            with_xaxis = (i >= nb_rows - 1)
            with_yaxis = (j == 0)
            with_scale_bars = (with_xaxis and with_yaxis)
            color = "C{}".format(k % 10)
            template.plot(ax=ax_, probe=self.probe, title=title, with_xaxis=with_xaxis,
                          with_yaxis=with_yaxis, with_scale_bars=with_scale_bars, color=color)
        for k in range(nb_templates, nb_subplots):
            i, j = k // nb_columns, k % nb_columns
            try:
                ax_ = ax[i, j]
            except IndexError:
                try:
                    ax_ = ax[k]
                except IndexError:
                    ax_ = ax
            ax_.set_axis_off()

        filename = "{}_{}_ground_truth_templates_{}.{}".format(self.name, self.time, mode, self.debug_file_format)
        path = os.path.join(self.debug_plots, filename)
        plt.savefig(path)
        plt.close()

        return
