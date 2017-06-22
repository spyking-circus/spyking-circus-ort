import scipy.optimize, numpy, pylab, scipy.spatial.distance, scipy.stats
import warnings
import logging
from circusort.utils.algorithms import PCAEstimator
warnings.filterwarnings("ignore")

class MacroCluster(object):

    def __init__(self, id, pca_data, data, creation_time=0):

        self.id            = id
        self.density       = len(pca_data)
        self.sum_pca       = numpy.sum(pca_data, 0)
        self.sum_pca_sq    = numpy.sum(pca_data**2, 0)
        self.sum_full      = numpy.sum(data, 0)
        self.creation_time = creation_time
        self.last_update   = creation_time
        self.label         = None

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
        factor           = 2**(-decay_factor*(time - self.last_update))
        self.density     = factor*self.density + 1.
        self.sum_pca     = factor*self.sum_pca + pca_data
        self.sum_pca_sq  = factor*self.sum_pca_sq + pca_data**2
        self.sum_full    = factor*self.sum_full + data
        self.last_update = time

    def remove(self, pca_data, data):
        self.density    -= 1
        self.sum_pca    -= pca_data
        self.sum_pca_sq -= pca_data**2
        self.sum_full   -= data

    @property
    def center(self):
        return self.sum_pca/self.density

    @property
    def center_full(self):
        return self.sum_full/self.density

    @property
    def sigma(self):
        return numpy.sqrt(numpy.linalg.norm(self.sum_pca_sq/self.density - self.center**2))

    def get_z_score(self, pca_data, sigma):
        return numpy.linalg.norm(self.center - pca_data)/sigma

    @property
    def tracking_properties(self):
        return [self.center, self.sigma]




class OnlineManager(object):

    def __init__(self, decay=0.35, mu=2, sigma_rad=3, epsilon=0.1, theta=-numpy.log(0.001), dispersion=(5, 5), n_min=None, noise_thr=0.8, pca=None, logger=None, name=None):

        if name is None:
            self.name = "OnlineManager"
        else:
            self.name = name
        self.clusters         = {}
        self.decay_factor     = decay
        self.mu               = mu
        self.epsilon          = epsilon
        self.theta            = theta
        self.radius           = sigma_rad
        self.dispersion       = dispersion
        self.noise_thr        = noise_thr
        self.n_min            = n_min
        self.glob_pca         = pca
        
        self.is_ready         = False
        self.abs_n_min        = 20
        self.nb_updates       = 0
        self.sub_dim          = 5
        self.loc_pca          = None
        self.tracking         = {}
        if logger is None:
            self.log = logging.getLogger(__name__)
        else:
            self.log = logger
        self.log.debug('{n} is created'.format(n=self.name))
    
    @property
    def D_threshold(self):
        return self.mu/(self.nb_clusters*(1 - 2**(-self.decay_factor)))

    @property
    def time_gap(self):
        if self.D_threshold > 1:
            return numpy.ceil((1/(self.decay_factor))*numpy.log(self.D_threshold/(self.D_threshold - 1)))
        else:
            return 100

    def initialize(self, time, data, two_components=False):

        if self.glob_pca is not None:
            sub_data = numpy.dot(data, self.glob_pca)
        else:
            sub_data = data

        a, b, c  = sub_data.shape
        sub_data = sub_data.reshape(a, b*c)

        a, b, c     = data.shape
        self._width = b*c
        data        = data.reshape(a, self._width)

        self.log.debug("{n} computes local PCA".format(n=self.name))
        pca          = PCAEstimator(self.sub_dim, copy=False)
        res_pca      = pca.fit_transform(sub_data.T).astype(numpy.float32)
        self.loc_pca = res_pca

        sub_data           = numpy.dot(sub_data, self.loc_pca)
        rhos, dist, _      = rho_estimation(sub_data)
        if len(rhos) > 0:
            rhos           = -rhos + rhos.max()
            n_min          = numpy.maximum(self.abs_n_min, int(self.n_min*len(data)))
            labels, c      = density_based_clustering(rhos, dist, n_min=n_min)
        else:
            labels, c      = numpy.array([]), numpy.array([])
        
        mask               = labels > -1
        self.nb_dimensions = sub_data.shape[1]
        amplitudes         = numpy.zeros((0, 2), dtype=numpy.float32)
        templates          = numpy.zeros((0, self._width), dtype=numpy.float32)
        indices            = numpy.zeros(0, dtype=numpy.int32)
        if two_components:
            templates2     = numpy.zeros((0, self._width), dtype=numpy.float32)
    
        for count, i in enumerate(numpy.unique(labels[mask])):

            indices = numpy.where(labels == i)[0]
            self.clusters[count] = MacroCluster(count, sub_data[indices], data[indices], creation_time=time)
            self.tracking[count] = self.clusters[count].tracking_properties
            template   = numpy.median(data[indices], 0)
            amplitudes = numpy.vstack((amplitudes, self._compute_amplitudes(data[indices], template)))
            templates  = numpy.vstack((templates, template))
            indices    = numpy.concatenate((indices, [count]))
            if two_components:
                templates2 = numpy.vstack((templates2, self._compute_template2(data[indices], template)))

        for cluster in self.clusters.values():
            if cluster.density >= self.D_threshold:
                cluster.set_label('dense')
            else:
                cluster.set_label('sparse')

        self.is_ready = True
        self.log.debug('{n} is initialized with {k} templates'.format(n=self.name, k=len(self.clusters)))

        if two_components:
            return {'dat' : templates, 'two' : templates2, 'amp' : amplitudes, 'ind' : indices}
        else:
            return {'dat' : templates, 'amp' : amplitudes, 'ind' : indices}

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
            return numpy.max(self.clusters.keys()) + 1
        else:
            return 0

    def _get_tracking_id(self):
        if len(self.tracking) > 0:
            return numpy.max(self.tracking.keys()) + 1
        else:
            return 0

    def _get_clusters(self, cluster_type):
        if cluster_type == 'sparse':
            return self.sparse_clusters
        elif cluster_type == 'dense':
            return self.dense_clusters

    def _get_centers(self, cluster_type='dense'):
        centers = numpy.zeros((0, self.nb_dimensions), dtype=numpy.float32)
        for cluster in self._get_clusters(cluster_type):
            centers = numpy.vstack((centers, [cluster.center]))
        return centers

    def _get_centers_full(self, cluster_type='dense'):
        centers = numpy.zeros((0, self._width), dtype=numpy.float32)
        for cluster in self._get_clusters(cluster_type):
            centers = numpy.vstack((centers, [cluster.center_full]))
        return centers

    def _merged_into(self, pca_data, data, cluster_type):
                
        clusters     = self._get_clusters(cluster_type)
        to_be_merged = False

        if len(clusters) > 0:

            centers  = self._get_centers(cluster_type)
            new_dist = scipy.spatial.distance.cdist(pca_data, centers, 'euclidean')[0]
            cluster  = clusters[numpy.argmin(new_dist)]

            cluster.add_and_update(pca_data[0], data[0], self.time, self.decay_factor)
            sigma = cluster.sigma
            
            if sigma == 0:
                sigma = self._estimate_sigma()
            
            to_be_merged = cluster.get_z_score(pca_data[0], sigma) <= self.epsilon

            if to_be_merged:
                if cluster.density >= self.D_threshold:
                    cluster.set_label('dense')
            else:
                cluster.remove(pca_data[0], data[0])

        return to_be_merged

    def _estimate_sigma(self):
        if len(self.clusters) > 0:
            sigma = 0
            count = 0
            for cluster in self.clusters.values():
                if cluster.sigma > 0:
                    sigma += cluster.sigma
                    count += 1
            if count >  0:
                return sigma/count
            else:
                return numpy.inf
        else:
            return numpy.inf
        

    def _prune(self):
        #self.log.debug("{n} cleans clusters...".format(n=self.name))

        for cluster in self.dense_clusters:
            if cluster.density < self.D_threshold:
                cluster.set_label('sparse')

        for cluster in self.sparse_clusters:

            T_0     = cluster.creation_time

            if T_0 < self.time and T_0 > 0:

                zeta    = (2**(-self.decay_factor*(self.time - T_0 + self.time_gap)) - 1)/(2**(-self.decay_factor*self.time_gap) - 1)
                delta_t = self.theta*(cluster.last_update - T_0)/cluster.density

                if cluster.density < zeta or ((self.time - cluster.last_update) > delta_t):
                    #self.log.debug("{n} removes sparse cluster {l}".format(n=self.name, l=cluster.id))

                    self.clusters.pop(cluster.id)

    def update(self, time, data=None):
        
        self.time = time
        if self.nb_sparse > 500:
            self.log.warning('{n} has too many ({s}) sparse clusters'.format(n=self.name, s=self.nb_sparse))
        #self.log.debug("{n} processes time {t} with {s} sparse and {d} dense clusters".format(n=self.name, t=time, s=self.nb_sparse, d=self.nb_dense)) 
        
        if data is not None:
        
            if self.glob_pca is not None:
                red_data = numpy.dot(data, self.glob_pca)
                data     = data.reshape(1, self._width)
                red_data = red_data.reshape(1, self.loc_pca.shape[0])
            
                if self.loc_pca is not None:
                    pca_data = numpy.dot(red_data, self.loc_pca)
                else:
                    pca_data = red_data
            else:
                data = data.reshape(1, self._width)
                if self.loc_pca is not None:
                    pca_data = numpy.dot(data, self.loc_pca)

            if self._merged_into(pca_data, data, 'dense'):
                #self.log.debug("{n} merges the data at time {t} into a dense cluster".format(n=self.name, t=self.time))
                pass
            else:
                if self._merged_into(pca_data, data, 'sparse'):
                    pass
                    #self.log.debug("{n} merges data at time {t} into a sparse cluster".format(n=self.name, t=self.time))
                else:
                    #self.log.debug("{n} can not merge data at time {t} and creates a new sparse cluster".format(n=self.name, t=self.time))
                    new_id      = self._get_id()
                    new_cluster = MacroCluster(new_id, pca_data, data, creation_time=self.time)
                    new_cluster.set_label('sparse')
                    self.clusters[new_id] = new_cluster

            self.nb_updates += 1

        if numpy.mod(self.time, self.time_gap) < 1:
            self._prune()

    def _perform_tracking(self, new_tracking_data):

        changes = {'new' : {}, 'merged' : {}}

        if len(self.tracking) > 0:
            all_centers = numpy.array([i[0] for i in self.tracking.values()], dtype=numpy.float32)
            all_sigmas  = [i[1] for i in self.tracking.values()]
            all_indices = self.tracking.keys()

            for key, value in new_tracking_data.items():
                center, sigma = value
                new_dist      = scipy.spatial.distance.cdist(numpy.array([center]), all_centers, 'euclidean')
                dist_min      = numpy.min(new_dist)
                dist_idx      = numpy.argmin(new_dist)

                if dist_min < self.radius*max(sigma, all_sigmas[dist_idx]):
                    #self.log.debug("{n} establishes a match between target {t} and source {s}".format(n=self.name, t=key, s=all_indices[dist_idx]))
                    changes['merged'][key] = all_indices[dist_idx]
                    self.tracking[key]     = center, sigma
                else:
                    idx = self._get_tracking_id()
                    self.tracking[idx] = center, sigma
                    #self.log.debug("{n} can not found a match for target {t} assigned to {s}".format(n=self.name, t=key, s=idx))
                    changes['new'][key] = idx
        else:
            self.tracking = new_tracking_data
            for key, value in new_tracking_data.items():
                idx = self._get_tracking_id()
                changes['new'][key] = idx

        return changes


    def set_physical_threshold(self, threshold):
        self.threshold = -self.noise_thr*threshold

    def _compute_amplitudes(self, data, template):
        ### We could to this in the PCA space, to speed up the computation
        temp_flat      = template.reshape(template.size, 1)
        amplitudes     = numpy.dot(data, temp_flat)
        amplitudes    /= numpy.sum(temp_flat**2)
        variation      = numpy.median(numpy.abs(amplitudes - numpy.median(amplitudes)))
        #physical_limit = self.threshold
        amp_min        = min(0.8, numpy.median(amplitudes) - self.dispersion[0]*variation)
        amp_max        = max(1.2, numpy.median(amplitudes) + self.dispersion[1]*variation)

        return numpy.array([amp_min, amp_max], dtype=numpy.float32)

    def _compute_template2(self, data, template):

        temp_flat   = template.reshape(template.size, 1)
        amplitudes  = numpy.dot(data, temp_flat)
        amplitudes /= numpy.sum(temp_flat**2)

        for i in xrange(len(data)):
            data[i, :] -= amplitudes[i]*temp_flat[:, 0]

        if len(temp_flat) > 1:
            pca       = PCAEstimator(1)
            res_pca   = pca.fit_transform(data).astype(numpy.float32)
            template2 = pca.components_.T.astype(numpy.float32)
        else:
            template2 = data/numpy.sum(data**2)
     
        return template2.reshape(1, template2.size)


    def cluster(self, tracking=True, two_components=False):

        self.log.debug('{n} launches clustering'.format(n=self.name))
        centers       = self._get_centers('dense')
        centers_full  = self._get_centers_full('dense')
        rhos, dist, _ = rho_estimation(centers)
        if len(rhos) > 0:
            rhos      = -rhos + rhos.max()
            labels, c = density_based_clustering(rhos, dist, n_min=None)
        else:
            labels, c = numpy.array([]), numpy.array([])

        self.nb_updates   = 0

        new_tracking_data = {}
        mask              = labels > -1
        for l in numpy.unique(labels[mask]):
            idx = numpy.where(labels == l)[0]
            cluster = MacroCluster(-1, centers[idx], centers_full[idx])
            new_tracking_data[l] = cluster.tracking_properties

        changes = self._perform_tracking(new_tracking_data)

        templates  = numpy.zeros((0, self._width), dtype=numpy.float32)
        amplitudes = numpy.zeros((0, 2), dtype=numpy.float32)
        indices    = numpy.zeros(0, dtype=numpy.int32)

        if two_components:
            templates2 = numpy.zeros((0, self._width), dtype=numpy.float32)

        for key, value in changes['new'].items():
            data       = centers_full[labels == key]
            template   = numpy.median(data, 0)
            templates  = numpy.vstack((templates, template))
            amplitudes = numpy.vstack((amplitudes, self._compute_amplitudes(data, template)))
            indices    = numpy.concatenate((indices, [value]))
            if two_components:
                template2  = self._compute_template2(data, template)
                templates2 = numpy.vstack((templates2, template2))

        self.log.debug('{n} found {a} new templates: {s}'.format(n=self.name, a=len(changes['new']), s=changes['new']))

        if tracking:
            for key, value in changes['merged'].items():
                data       = centers_full[labels == value]
                template   = numpy.median(data, 0)
                templates  = numpy.vstack((templates, template))
                amplitudes = numpy.vstack((amplitudes, self._compute_amplitudes(data, template)))
                indices    = numpy.concatenate((indices, [key]))
                if two_components:
                    template2  = self._compute_template2(data, template)
                    templates2 = numpy.vstack((templates2, template2))

            self.log.debug('{n} modified {a} templates with tracking: {s}'.format(n=self.name, a=len(changes['merged']), s=changes['merged'].values()))

        if two_components:
            return {'dat' : templates, 'two' : templates2, 'amp' : amplitudes, 'ind' : indices}
        else:
            return {'dat' : templates, 'amp' : amplitudes, 'ind' : indices}


            




def fit_rho_delta(xdata, ydata, smart_select=False, max_clusters=10):

    if smart_select:

        xmax = xdata.max()
        off  = ydata.min()
        idx  = numpy.argmin(xdata)
        a_0  = (ydata[idx] - off)/numpy.log(1 + (xmax - xdata[idx]))

        def myfunc(x, a, b, c, d):
            return a*numpy.log(1. + c*((xmax - x)**b)) + d

        try:
            result, pcov = scipy.optimize.curve_fit(myfunc, xdata, ydata, p0=[a_0, 1., 1., off])
            prediction   = myfunc(xdata, result[0], result[1], result[2], result[3])
            difference   = xdata*(ydata - prediction)
            z_score      = (difference - difference.mean())/difference.std()
            subidx       = numpy.where(z_score >= 3.)[0]
        except Exception:
            subidx = numpy.argsort(xdata*numpy.log(1 + ydata))[::-1][:max_clusters]
        
    else:
        subidx = numpy.argsort(xdata*numpy.log(1 + ydata))[::-1][:max_clusters]

    return subidx, len(subidx)


def rho_estimation(data, mratio=0.01):

    N     = len(data)
    rho   = numpy.zeros(N, dtype=numpy.float32)
        
    dist = scipy.spatial.distance.pdist(data, 'euclidean').astype(numpy.float32)
    didx = lambda i,j: i*N + j - i*(i+1)//2 - i - 1
    nb_selec = max(5, int(mratio*N))

    for i in xrange(N):
        indices  = numpy.concatenate((didx(i, numpy.arange(i+1, N)), didx(numpy.arange(0, i-1), i)))
        tmp      = numpy.argsort(numpy.take(dist, indices))[:nb_selec]
        sdist    = numpy.take(dist, numpy.take(indices, tmp))
        rho[i]   = numpy.mean(sdist)

    return rho, dist, nb_selec


def density_based_clustering(rho, dist, smart_select=True, n_min=None, max_clusters=10):

    N                 = len(rho)
    maxd              = numpy.max(dist)
    didx              = lambda i,j: i*N + j - i*(i+1)//2 - i - 1
    ordrho            = numpy.argsort(rho)[::-1]
    delta, nneigh     = numpy.zeros(N, dtype=numpy.float32), numpy.zeros(N, dtype=numpy.int32)
    delta[ordrho[0]]  = -1
    for ii in xrange(1, N):
        delta[ordrho[ii]] = maxd
        for jj in xrange(ii):
            if ordrho[jj] > ordrho[ii]:
                xdist = dist[didx(ordrho[ii], ordrho[jj])]
            else:
                xdist = dist[didx(ordrho[jj], ordrho[ii])]

            if xdist < delta[ordrho[ii]]:
                delta[ordrho[ii]]  = xdist
                nneigh[ordrho[ii]] = ordrho[jj]

    delta[ordrho[0]]        = delta.max()
    clust_idx, max_clusters = fit_rho_delta(rho, delta, smart_select=smart_select, max_clusters=max_clusters)

    def assign_halo(idx):
        cl      = numpy.empty(N, dtype=numpy.int32)
        cl[:]   = -1
        NCLUST  = len(idx)
        cl[idx] = numpy.arange(NCLUST)
        
        # assignation
        for i in xrange(N):
            if cl[ordrho[i]] == -1:
                cl[ordrho[i]] = cl[nneigh[ordrho[i]]]
        
        # halo (ignoring outliers ?)
        halo = cl.copy()
        
        if n_min is not None:

            for cluster in xrange(NCLUST):
                idx = numpy.where(halo == cluster)[0]
                if len(idx) < n_min:
                    halo[idx] = -1
                    NCLUST   -= 1
        return halo, NCLUST

    halo, NCLUST = assign_halo(clust_idx[:max_clusters+1])

    return halo, clust_idx[:max_clusters]