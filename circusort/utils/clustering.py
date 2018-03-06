import scipy.optimize
import numpy as np
import hdbscan
import os
import scipy.spatial.distance
import scipy.stats
import warnings
import logging
from circusort.obj.template import Template, TemplateComponent
from sklearn.decomposition import PCA


warnings.filterwarnings("ignore")


class MacroCluster(object):
    """Macro cluster"""
    # TODO complete docstring.

    def __init__(self, id, pca_data, data, creation_time=0):

        self.id = id
        self.density = len(pca_data)
        self.sum_pca = np.sum(pca_data, 0)
        self.sum_pca_sq = np.sum(pca_data**2, 0)
        self.sum_full = np.sum(data, 0)
        self.creation_time = creation_time
        self.last_update = creation_time
        self.label = 'sparse'

    def set_label(self, label):
        assert label in ['sparse', 'dense']
        self.label = label    

    @property
    def is_sparse(self):
        return self.label == 'sparse'

    @property
    def is_dense(self):
        return self.label == 'dense'

    def add_and_update(self, pca_data, data, time, decay_factor):
        factor = 2 ** (-decay_factor * (time - self.last_update))
        self.density = factor * self.density + 1.
        self.sum_pca = factor * self.sum_pca + pca_data
        self.sum_pca_sq = factor * self.sum_pca_sq + pca_data ** 2
        self.sum_full = factor * self.sum_full + data
        self.last_update = time

    def remove(self, pca_data, data):
        self.density -= 1
        self.sum_pca -= pca_data
        self.sum_pca_sq -= pca_data ** 2
        self.sum_full -= data

    @property
    def center(self):
        return self.sum_pca / self.density

    @property
    def center_full(self):
        return self.sum_full / self.density

    @property
    def sigma(self):
        return np.sqrt(np.linalg.norm(self.sum_pca_sq / self.density - self.center ** 2))

    @property
    def radius(self):
        return np.sqrt((self.sum_pca_sq / self.density - self.center ** 2).max())

    def get_z_score(self, pca_data, sigma):
        return np.linalg.norm(self.center - pca_data) / sigma

    @property
    def tracking_properties(self):
        return [self.center, self.sigma]


class OnlineManager(object):

    def __init__(self, probe, channel, decay=0.25, mu=10, epsilon=10, theta=-np.log(0.001), dispersion=(5, 5),
                 n_min=0.002, noise_thr=0.8, pca=None, logger=None, two_components=False, name=None, debug_plots=None,
                 local_merges=3):

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
        self.n_min = n_min
        self.glob_pca = pca
        self.probe = probe
        self.channel = channel
        self.two_components = two_components
        self.debug_plots=debug_plots
        self.local_merges = local_merges

        if self.debug_plots is not None:
            self.fig_name = os.path.join(self.debug_plots, '{n}_{t}.png')
            self.data_name = os.path.join(self.debug_plots, '{n}_{t}')

        self.time = 0
        self.is_ready = False
        self.abs_n_min = 20
        self.nb_updates = 0
        self.sub_dim = 5
        self.loc_pca = None
        self.tracking = {}
        if logger is None:
            self.log = logging.getLogger(__name__)
        else:
            self.log = logger
        self.log.debug('{n} is created'.format(n=self.name))

    @property
    def D_threshold(self):
        return self.mu * 0.2 #/ (self.nb_clusters*(1 - 2**(-self.decay_factor)))

    @property
    def time_gap(self):
        if self.D_threshold > 1:
            return np.ceil((1 / self.decay_factor) * np.log(self.D_threshold / (self.D_threshold - 1)))
        else:
            return 100

    def initialize(self, time, data):

        if self.glob_pca is not None:
            sub_data = np.dot(data, self.glob_pca)
        else:
            sub_data = data

        a, b, c = sub_data.shape
        sub_data = sub_data.reshape(a, b * c)

        a, b, c = data.shape
        self._width = b * c
        data = data.reshape(a, self._width)

        # TODO uncomment the following line.
        self.log.debug("{n} computes local PCA".format(n=self.name))
        pca = PCA(self.sub_dim)
        pca.fit(sub_data)
        self.loc_pca = pca.components_.T

        sub_data = np.dot(sub_data, self.loc_pca)

        n_min = np.maximum(self.abs_n_min, int(self.n_min * len(sub_data)))

        if self.debug_plots is not None:
            output = self.fig_name.format(n=self.name, t=self.time)
            np.save(self.data_name.format(n=self.name, t=self.time), sub_data)
        else:
            output = None
        labels = density_clustering(sub_data, n_min=n_min, output=output, local_merges=self.local_merges)

        mask = labels > -1

        templates = {}

        self.time = time

        self.log.debug("{n} founds {k} initial clusters from {m} datapoints".format(n=self.name,
                                                                                    k=len(np.unique(labels[mask])),
                                                                                    m=len(sub_data)))

        for count, i in enumerate(np.unique(labels[mask])):

            indices = np.where(labels == i)[0]
            data_cluster = data[indices]
            sub_data_cluster = sub_data[indices]

            self.clusters[count] = MacroCluster(count, sub_data_cluster, data_cluster, creation_time=self.time)
            self.tracking[count] = self.clusters[count].tracking_properties

            waveforms = np.median(data_cluster, 0)
            amplitudes, full_ = self._compute_amplitudes(data_cluster, waveforms)

            first_comp = TemplateComponent(waveforms, self.probe.edges[self.channel], self.probe.nb_channels, amplitudes)

            if self.two_components:
                waveforms = self._compute_second_component(data_cluster, waveforms, full_)
                second_comp = TemplateComponent(waveforms, self.probe.edges[self.channel], self.probe.nb_channels)
            else:
                second_comp = None

            template = Template(first_comp, self.channel, second_comp)
            templates[count] = template

        for cluster in self.clusters.values():
            if cluster.density >= self.D_threshold:
                cluster.set_label('dense')
            else:
                cluster.set_label('sparse')

        self.is_ready = True
        # TODO uncomment the following line.
        # self.log.debug('{n} is initialized with {k} templates'.format(n=self.name, k=len(self.clusters)))

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

    def time_to_cluster(self, nb_updates):
        return self.is_ready and self.nb_updates >= nb_updates

    def _get_id(self):
        if len(self.clusters) > 0:
            return np.max(self.clusters.keys()) + 1
        else:
            return 0

    def _get_tracking_id(self):
        if len(self.tracking) > 0:
            return np.max(self.tracking.keys()) + 1
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
        centers = np.zeros((0, self._width), dtype=np.float32)
        for cluster in self._get_clusters(cluster_type):
            centers = np.vstack((centers, [cluster.center_full]))
        return centers

    def _merged_into(self, pca_data, data, cluster_type):
                
        clusters = self._get_clusters(cluster_type)
        to_be_merged = False

        if len(clusters) > 0:

            centers = self._get_centers(cluster_type)
            new_dist = scipy.spatial.distance.cdist(pca_data, centers, 'euclidean')[0]
            cluster = clusters[np.argmin(new_dist)]

            cluster.add_and_update(pca_data[0], data[0], self.time, self.decay_factor)

            to_be_merged = cluster.radius <= self.epsilon

            if to_be_merged:
                if cluster.density >= self.D_threshold:
                    cluster.set_label('dense')
            else:
                cluster.remove(pca_data[0], data[0])

        return to_be_merged

    def _prune(self):

        count = 0
        for cluster in self.dense_clusters:
            if cluster.density < self.D_threshold:
                cluster.set_label('sparse')
                count += 1

        # TODO uncomment the following line.
        if count > 0:
            self.log.debug("{n} turns {m} dense clusters into sparse...".format(n=self.name, m=count))

        count = 0
        for cluster in self.sparse_clusters:

            t_0 = cluster.creation_time

            if 0 < t_0 < self.time:

                numerator = (2**(-self.decay_factor*(self.time - t_0 + self.time_gap)) - 1)
                denominator = (2**(-self.decay_factor*self.time_gap) - 1)
                zeta = numerator / denominator
                delta_t = self.theta*(cluster.last_update - t_0)/cluster.density

                if cluster.density < zeta or ((self.time - cluster.last_update) > delta_t):
                    # self.log.debug("{n} removes sparse cluster {l}".format(n=self.name, l=cluster.id))

                    self.clusters.pop(cluster.id)
                    count += 1

        # TODO uncomment the following line.
        if count > 0:
            self.log.debug("{n} prunes {m} sparse clusters...".format(n=self.name, m=count))

    def update(self, time, data=None):
        
        self.time = time
        if self.nb_sparse > 100:
            self.log.warning('{n} has too many ({s}) sparse clusters'.format(n=self.name, s=self.nb_sparse))
        
        self.log.debug("{n} processes time {t} with {s} sparse and {d} dense clusters. Time gap is {g}".format(
                                            n=self.name, t=time, s=self.nb_sparse, d=self.nb_dense, g=self.time_gap))
        
        if data is not None:
        
            if self.glob_pca is not None:
                red_data = np.dot(data, self.glob_pca)
                data = data.reshape(1, self._width)
                red_data = red_data.reshape(1, self.loc_pca.shape[0])
            
                if self.loc_pca is not None:
                    pca_data = np.dot(red_data, self.loc_pca)
                else:
                    pca_data = red_data
            else:
                data = data.reshape(1, self._width)
                if self.loc_pca is not None:
                    pca_data = np.dot(data, self.loc_pca)

            if self._merged_into(pca_data, data, 'dense'):
                self.log.debug("{n} merges the data at time {t} into a dense cluster".format(n=self.name, t=self.time))
            else:
                if self._merged_into(pca_data, data, 'sparse'):
                    self.log.debug("{n} merges data at time {t} into a sparse cluster".format(n=self.name, t=self.time))
                else:
                    self.log.debug("{n} can not merge data at time {t} and creates a new sparse cluster".format(
                        n=self.name, t=self.time))
                    new_id = self._get_id()
                    new_cluster = MacroCluster(new_id, pca_data, data, creation_time=self.time)
                    new_cluster.set_label('sparse')
                    self.clusters[new_id] = new_cluster

            self.nb_updates += 1

        if np.mod(self.time, self.time_gap) < 1:
            self._prune()

    def _perform_tracking(self, new_tracking_data):

        changes = {
            'new': {},
            'merged': {}
        }

        if len(self.tracking) > 0:
            all_centers = np.array([i[0] for i in self.tracking.values()], dtype=np.float32)
            all_sigmas = np.array([i[1] for i in self.tracking.values()], dtype=np.float32)
            all_indices = self.tracking.keys()

            for key, value in new_tracking_data.items():
                center, sigma = value
                new_dist = scipy.spatial.distance.cdist(np.array([center]), all_centers, 'euclidean')
                dist_min = np.min(new_dist)
                dist_idx = np.argmin(new_dist)

                if dist_min <= (sigma + all_sigmas[dist_idx]):
                    self.log.debug("{n} establishes a match between target {t} and source {s}".format(n=self.name,
                                                                                        t=key, s=all_indices[dist_idx]))
                    changes['merged'][key] = all_indices[dist_idx]
                    self.tracking[key] = center, sigma
                else:
                    idx = self._get_tracking_id()
                    self.tracking[idx] = center, sigma
                    self.log.debug("{n} can not found a match for target {t} assigned to {s}".format(n=self.name,
                                                                                                     t=key, s=idx))
                    changes['new'][key] = idx
        else:
            self.tracking = new_tracking_data
            for key, value in new_tracking_data.items():
                idx = self._get_tracking_id()
                changes['new'][key] = idx

        return changes

    def set_physical_threshold(self, threshold):
        self.physical_threshold = self.noise_thr * threshold

    def _compute_amplitudes(self, data, template):
        # # We could to this in the PCA space, to speed up the computation
        temp_flat = template.reshape(template.size, 1)
        amplitudes = np.dot(data, temp_flat)
        amplitudes /= np.sum(temp_flat**2)
        variation = np.median(np.abs(amplitudes - np.median(amplitudes)))

        amp_min = min(0.8, max(self.physical_threshold, np.median(amplitudes) - self.dispersion[0]*variation))
        amp_max = max(1.2, np.median(amplitudes) + self.dispersion[1]*variation)

        return np.array([amp_min, amp_max], dtype=np.float32), amplitudes

    def _compute_second_component(self, data, waveforms, amplitudes=None):

        temp_flat = waveforms.reshape(waveforms.size, 1)
        if amplitudes is None:
            amplitudes = np.dot(data, temp_flat)
            amplitudes /= np.sum(temp_flat ** 2)

        for i in range(len(data)):
            data[i, :] -= amplitudes[i] * temp_flat[:, 0]

        if len(temp_flat) > 1:
            pca = PCA(1)
            pca.fit(data)
            waveforms = pca.components_.T.astype(np.float32)
        else:
            waveforms = data / np.sum(data ** 2)

        return waveforms.reshape(1, waveforms.size)

    def cluster(self, tracking=True):

        self.log.debug('{n} launches clustering with {s} sparse and {t} dense clusters'.format(n=self.name,
                                                                                    s=self.nb_sparse, t=self.nb_dense))
        centers = self._get_centers('dense')
        centers_full = self._get_centers_full('dense')

        if self.debug_plots is not None:
            output = self.fig_name.format(n=self.name, t=self.time)
        else:
            output = None

        labels = density_clustering(centers, n_min=None, output=output, local_merges=self.local_merges)

        self.nb_updates = 0

        new_tracking_data = {}
        mask = labels > -1
        for l in np.unique(labels[mask]):
            idx = np.where(labels == l)[0]
            cluster = MacroCluster(-1, centers[idx], centers_full[idx])
            new_tracking_data[l] = cluster.tracking_properties

        changes = self._perform_tracking(new_tracking_data)

        templates = {}

        for key, value in changes['new'].items():
            data = centers_full[labels == key]
            waveforms = np.median(data, 0)
            amplitudes, full_ = self._compute_amplitudes(data, waveforms)

            first_component = TemplateComponent(waveforms, self.probe.edges[self.channel], self.probe.nb_channels,
                                                amplitudes)
            if self.two_components:
                waveforms = self._compute_second_component(data, waveforms, full_)
                second_component = TemplateComponent(waveforms, self.probe.edges[self.channel], self.probe.nb_channels)
            else:
                second_component = None

            template = Template(first_component, self.channel, second_component)
            templates[value] = template

        self.log.debug('{n} found {a} new templates: {s}'.format(n=self.name, a=len(changes['new']), s=changes['new']))

        if tracking:
            for key, value in changes['merged'].items():
                data = centers_full[labels == value]
                waveforms = np.median(data, 0)
                amplitudes, full_ = self._compute_amplitudes(data, waveforms)

                first_component = TemplateComponent(waveforms, self.probe.edges[self.channel], self.probe.nb_channels,
                                                    amplitudes)

                if self.two_components:
                    waveforms = self._compute_second_component(data, waveforms, full_)
                    second_component = TemplateComponent(waveforms, self.probe.edges[self.channel],
                                                         self.probe.nb_channels)
                else:
                    second_component = None

                template = Template(first_component, self.channel, second_component)
                templates[value] = template

            self.log.debug('{n} modified {a} templates with tracking: {s}'.format(n=self.name, a=len(changes['merged']),
                                                                                  s=changes['merged'].values()))
        return templates


def fit_rho_delta(xdata, ydata, smart_select=False, max_clusters=10):

    if smart_select:

        xmax = xdata.max()
        off = ydata.min()
        idx = np.argmin(xdata)
        a_0 = (ydata[idx] - off) / np.log(1 + (xmax - xdata[idx]))

        def myfunc(x, a, b, c, d):
            return a*np.log(1. + c * ((xmax - x) ** b)) + d

        try:
            result, pcov = scipy.optimize.curve_fit(myfunc, xdata, ydata, p0=[a_0, 1., 1., off])
            prediction = myfunc(xdata, result[0], result[1], result[2], result[3])
            difference = xdata*(ydata - prediction)
            z_score = (difference - difference.mean())/difference.std()
            subidx = np.where(z_score >= 3.)[0]
        except Exception:
            subidx = np.argsort(xdata*np.log(1 + ydata))[::-1][:max_clusters]
        
    else:
        subidx = np.argsort(xdata*np.log(1 + ydata))[::-1][:max_clusters]

    return subidx, len(subidx)


def rho_estimation(data, mratio=0.01):

    N = len(data)
    rho = np.zeros(N, dtype=np.float32)
    didx = lambda i, j: i*N + j - i*(i+1)//2 - i - 1
    dist = scipy.spatial.distance.pdist(data, 'euclidean').astype(np.float32)
    nb_selec = max(5, int(mratio*N))

    for i in range(N):
        indices = np.concatenate((didx(i, np.arange(i+1, N)), didx(np.arange(0, i-1), i)))
        tmp = np.argsort(np.take(dist, indices))[:nb_selec]
        sdist = np.take(dist, np.take(indices, tmp))
        rho[i] = np.mean(sdist)

    return rho, dist, nb_selec


def density_based_clustering(rho, dist, smart_select=True, n_min=None, max_clusters=10):

    N = len(rho)
    maxd = np.max(dist)
    didx = lambda i, j: i * N + j - i * (i + 1) // 2 - i - 1
    ordrho = np.argsort(rho)[::-1]
    delta, nneigh = np.zeros(N, dtype=np.float32), np.zeros(N, dtype=np.int32)
    delta[ordrho[0]] = -1
    for ii in range(1, N):
        delta[ordrho[ii]] = maxd
        for jj in range(ii):
            if ordrho[jj] > ordrho[ii]:
                xdist = dist[didx(ordrho[ii], ordrho[jj])]
            else:
                xdist = dist[didx(ordrho[jj], ordrho[ii])]

            if xdist < delta[ordrho[ii]]:
                delta[ordrho[ii]] = xdist
                nneigh[ordrho[ii]] = ordrho[jj]

    delta[ordrho[0]] = delta.max()
    clust_idx, max_clusters = fit_rho_delta(rho, delta, smart_select=smart_select, max_clusters=max_clusters)

    def assign_halo(idx):
        cl = np.empty(N, dtype=np.int32)
        cl[:] = -1
        NCLUST = len(idx)
        cl[idx] = np.arange(NCLUST)

        # assignation
        for i in range(N):
            if cl[ordrho[i]] == -1:
                cl[ordrho[i]] = cl[nneigh[ordrho[i]]]

        # halo (ignoring outliers ?)
        halo = cl.copy()

        if n_min is not None:

            for cluster in range(NCLUST):
                idx = np.where(halo == cluster)[0]
                if len(idx) < n_min:
                    halo[idx] = -1
                    NCLUST -= 1
        return halo, NCLUST

    halo, NCLUST = assign_halo(clust_idx[:max_clusters+1])

    return halo, clust_idx[:max_clusters]


def greedy_merges(data, labels, local_merges):

    def do_merging(data, labels, clusters, local_merges):

        dmin = np.inf
        to_merge = [None, None]

        for ic1 in xrange(len(clusters)):
            idx1 = np.where(labels == clusters[ic1])[0]
            sd1 = np.take(data, idx1, axis=0)
            m1 = np.median(sd1, 0)
            for ic2 in xrange(ic1 + 1, len(clusters)):
                idx2 = np.where(labels == clusters[ic2])[0]
                sd2 = np.take(data, idx2, axis=0)
                m2 = np.median(sd2, 0)
                v_n = m1 - m2
                pr_1 = np.dot(sd1, v_n)
                pr_2 = np.dot(sd2, v_n)

                norm = np.median(np.abs(pr_1 - np.median(pr_1))) ** 2 + np.median(
                    np.abs(pr_2 - np.median(pr_2))) ** 2
                dist = np.sum(v_n ** 2) / np.sqrt(norm)

                if dist < dmin:
                    dmin = dist
                    to_merge = [ic1, ic2]

        if dmin < local_merges:
            labels[np.where(labels == clusters[to_merge[1]])[0]] = clusters[to_merge[0]]
            return True, labels

        return False, labels

    has_been_merged = True
    mask = np.where(labels > -1)[0]
    clusters = np.unique(labels[mask])
    merged = [len(clusters), 0]

    while has_been_merged:
        has_been_merged, labels = do_merging(data, labels, clusters, local_merges)
        if has_been_merged:
            merged[1] += 1
    return labels, merged


def density_clustering(data, n_min=None, output=None, local_merges=None):

    rhos, dist, _ = rho_estimation(data)
    if len(rhos) == 1:
        labels, c = np.array([0]), np.array([0])
    elif len(rhos) > 1:
        rhos = -rhos + rhos.max()
        labels, c = density_based_clustering(rhos, dist, n_min)
    else:
        labels, c = np.array([]), np.array([])

    if output is not None:
        import pylab
        pylab.subplot(221)
        pylab.scatter(data[:, 0], data[:, 1], c=labels)
        pylab.subplot(222)
        pylab.scatter(data[:, 1], data[:, 2], c=labels)
        pylab.subplot(223)
        pylab.scatter(data[:, 0], data[:, 2], c=labels)

    if local_merges is not None:
        labels, merged = greedy_merges(data, labels, local_merges)
        if output is not None:
            pylab.subplot(224)
            pylab.title('%s' %merged)
            pylab.scatter(data[:, 0], data[:, 1], c=labels)

    if output is not None:
        pylab.savefig(output)
        pylab.close()

    return labels


def hdbscan_clustering(data, n_min=None, output=None):

    if n_min is not None:
        cluster_engine = hdbscan.HDBSCAN(min_cluster_size=int(n_min), core_dist_n_jobs=1, allow_single_cluster=True)
    else:
        cluster_engine = hdbscan.HDBSCAN(min_cluster_size=1, core_dist_n_jobs=1, allow_single_cluster=True)

    labels = cluster_engine.fit_predict(data)

    if output is not None:
        import pylab
        pylab.subplot(221)
        pylab.scatter(data[:, 0], data[:, 1], c=labels)
        pylab.subplot(222)
        pylab.scatter(data[:, 1], data[:, 2], c=labels)
        pylab.subplot(223)
        pylab.scatter(data[:, 0], data[:, 2], c=labels)
        pylab.savefig(output)
        pylab.close()

    return labels