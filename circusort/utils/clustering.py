import scipy.optimize, numpy, pylab, scipy.spatial.distance, scipy.stats
import warnings
import logging
warnings.filterwarnings("ignore")

class MacroCluster(object):

    def __init__(self, id, data, creation_time=0):

        self.id             = id
        self.density        = len(data)
        self.sum_centers    = numpy.sum(data, 0)
        self.sum_centers_sq = numpy.sum(data**2, 0)
        self.creation_time  = creation_time
        self.last_update    = creation_time
        self.label          = None

    def set_label(self, label):
        assert label in ['sparse', 'dense']
        self.label = label    

    @property
    def is_sparse(self):
        return self.label == 'sparse'

    @property
    def is_dense(self):
        return self.label == 'dense'

    def add_and_update(self, data, time, decay_factor):
        factor              = 2**(-decay_factor*(time - self.last_update))
        self.density        = factor*self.density + 1.
        self.sum_centers    = factor*self.sum_centers + data
        self.sum_centers_sq = factor*self.sum_centers_sq + data**2
        self.last_update    = time

    def remove(self, data):
        self.density        -= 1
        self.sum_centers    -= data
        self.sum_centers_sq -= data**2

    @property
    def center(self):
        return self.sum_centers/self.density

    @property
    def sigma(self):
        return numpy.sqrt(numpy.linalg.norm(self.sum_centers_sq/self.density - self.center**2))

    def get_z_score(self, data, sigma):
        return numpy.linalg.norm(self.center - data)/sigma

    @property
    def tracking_properties(self):
        return [self.center, self.sigma]


class OnlineManager(object):

    def __init__(self, decay_factor=0.35, mu=2, epsilon=0.1, theta=-numpy.log(0.001), logger=None, name=None):

        if name is None:
            self.name = "OnlineManager"
        else:
            self.name = name
        self.clusters         = {}
        self.decay_factor     = decay_factor
        self.mu               = mu
        self.epsilon          = epsilon
        self.theta            = theta
        self.is_ready         = False
        self.nb_updates       = 0
        self.tracking         = {}
        if logger is None:
            self.log = logging.getLogger(__name__)
        else:
            self.log = logger
    
    @property
    def D_threshold(self):
        return self.mu/(self.nb_clusters*(1 - 2**(-self.decay_factor)))

    @property
    def time_gap(self):
        if self.D_threshold > 1:
            return numpy.ceil((1/(self.decay_factor))*numpy.log(self.D_threshold/(self.D_threshold - 1)))
        else:
            return 100

    def initialize(self, time, data, labels):

        mask               = labels > -1
        self.nb_dimensions = data.shape[1]

        for count, i in enumerate(numpy.unique(labels[mask])):

            indices = numpy.where(labels == i)[0]
            self.clusters[count] = MacroCluster(count, data[indices], creation_time=time)
            self.tracking[count] = self.clusters[count].tracking_properties

        for cluster in self.clusters.values():
            if cluster.density >= self.D_threshold:
                cluster.set_label('dense')
            else:
                cluster.set_label('sparse')

        self.is_ready = True

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

    def _get_id(self):
        return numpy.max(self.clusters.keys()) + 1

    def _get_tracking_id(self):
        return numpy.max(self.tracking.keys()) + 1

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

    def _merged_into(self, data, cluster_type):
                
        clusters     = self._get_clusters(cluster_type)
        to_be_merged = False

        if len(clusters) > 0:

            centers  = self._get_centers(cluster_type)
            new_dist = scipy.spatial.distance.cdist(data, centers, 'euclidean')[0]
            dist_min = numpy.min(new_dist)
            cluster  = clusters[numpy.argmin(new_dist)]

            cluster.add_and_update(data[0], self.time, self.decay_factor)
            sigma = cluster.sigma
            
            if sigma == 0:
                sigma = self._estimate_sigma()
            
            to_be_merged = cluster.get_z_score(data[0], sigma) <= self.epsilon

            if to_be_merged:
                if cluster.density >= self.D_threshold:
                    cluster.set_label('dense')
            else:
                cluster.remove(data[0])

        return to_be_merged


    def _estimate_sigma(self):
        sigma = 0
        count = 0
        for cluster in self.clusters.values():
            if cluster.sigma > 0:
                sigma += cluster.sigma
                count += 1
        return sigma/count

    def _prune(self):
        self.log.debug("Time gap reached, time to clean clusters...")

        for cluster in self.dense_clusters:
            if cluster.density < self.D_threshold:
                cluster.set_label('sparse')

        for cluster in self.sparse_clusters:

            T_0     = cluster.creation_time

            if T_0 < self.time and T_0 > 0:

                zeta    = (2**(-self.decay_factor*(self.time - T_0 + self.time_gap)) - 1)/(2**(-self.decay_factor*self.time_gap) - 1)
                delta_t = self.theta*(cluster.last_update - T_0)/cluster.density

                if cluster.density < zeta or ((time - cluster.last_update) > delta_t):
                    self.log.debug("Removing sparse cluster %d" %cluster.id)

                    self.clusters.pop(cluster.id)

    def update(self, time, new_data=None):
        
        self.time = time
        self.log.debug("Processing time {t} with {s} sparse/ {d} dense".format(t=time, s=self.nb_sparse, d=self.nb_dense)) 
        
        if new_data is not None:
            if self._merged_into(new_data, 'dense'):
                self.log.debug("We merged the time point", self.time, 'into a dense cluster')
            else:
                if self._merged_into(new_data, 'sparse'):
                    self.log.debug("We merged the time point", self.time, 'into a sparse cluster')
                else:
                    self.log.debug("We can not merged time point", self.time, "so creating a new sparse cluster")
                    new_id      = self._get_id()
                    new_cluster = MacroCluster(new_id, new_data, creation_time=self.time)
                    new_cluster.set_label('sparse')
                    self.clusters[new_id] = new_cluster

            self.nb_updates += 1

        if numpy.mod(self.time, self.time_gap) == 0:
            self._prune()

    def _perform_tracking(self, new_clusters):

        max_idx = self.real_clusters['indices'].max()

        for count in xrange(len(new_clusters['indices'])):
            new_dist = scipy.spatial.distance.cdist(numpy.array([new_clusters['mus'][count]]), self.real_clusters['mus'], 'euclidean')
            dist_min = numpy.min(new_dist)
            dist_idx = numpy.argmin(new_dist)

            if dist_min < 3*max(new_clusters['sigmas'][count], self.real_clusters['sigmas'][dist_idx]):
                self.log.debug("Match between target", new_clusters['indices'][count], "and source", dist_idx)
                new_clusters['indices'][count] = self.real_clusters['indices'][dist_idx] 
            else:
                max_idx += 1
                new_clusters['indices'][count] = max_idx
                self.log.debug("No match for target", new_clusters['indices'][count], "assigned to", max_idx)

        self.real_clusters = new_clusters

    def cluster(self):

        centers       = self._get_centers('dense')
        rhos, dist, _ = rho_estimation(centers)
        rhos          = -rhos + rhos.max() 
        labels, c     = density_based_clustering(rhos, dist, n_min=None)
        self.nb_updates   = 0

        new_tracking_data = {}
        mask              = labels > -1
        for l in numpy.unique(labels[mask]):
            idx = numpy.where(labels == l)[0]
            cluster = MacroCluster(-1, data=centers[idx])
            new_tracking_data[l] = cluster.tracking_properties

        self._perform_tracking(data, clusters, c)

        #self.tracking[count] = self.clusters[count].tracking_properties










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