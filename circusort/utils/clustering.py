import scipy.optimize, numpy, pylab, scipy.spatial.distance, scipy.stats
import warnings
warnings.filterwarnings("ignore")


class OnlineManager(object):

    def __init__(self, decay_factor=0.35, mu=4, epsilon=0.5, theta=-numpy.log(0.001)):

        self.densities        = {}
        self.last_updates     = {}   
        self.summed_centers   = {}
        self.summed_centers_2 = {}
        self.creation_times   = {}
        self.clusters         = {'sparse' : [], 'dense' : []}
    
    def initialize(self, data, labels):
        for count, i in enumerate(numpy.unique(clusters[clusters > -1])):
            self.densities[count]        = numpy.sum(clusters == i)
            self.last_updates[count]     = 0
            self.creation_times[count]   = 0
            self.summed_centers[count]   = numpy.sum(data[clusters == i], 0)
            self.summed_centers_2[count] = numpy.sum(data[clusters == i]**2, 0)

        for cluster in self.densities.keys():
            if self.densities[cluster] >= self.D_thred:
                self.clusters['dense']  += [cluster]
            else:
                self.clusters['sparse'] += [cluster]

    def check_center(self):
        pass

    def _online_update(self, time):
        self.D_thred  = mu/(len(self.densities)*(1-2**(-self.decay_factor)))
        #if D_thred > 1:
        #    time_gap = (1/(decay_factor))*numpy.log(D_thred/(D_thred - 1))
        #   #time_gap = numpy.log(mu/(mu - len(densities)*(1-2**-decay_factor)))/numpy.log(decay_factor)
        #else:
            #time_gap = 1
        self.time_gap = 100


        # Update of D_thred and time_gap to know when we need to clean clusters, and where
        # is the threshold between sparse and dense clusters

            
        #for cluster in (sparse_clusters + dense_clusters):
        #    densities[cluster]        *= 2**(-decay_factor)
        #    summed_centers[cluster]   *= 2**(-decay_factor)
        #    summed_centers_2[cluster] *= 2**(-decay_factor)

        # First, we collect all dense cluster centers
        dense_centers = numpy.zeros((0, nb_dimensions))
        dense_radius  = []
        for cluster in self.dense_clusters:
            dense_centers = numpy.vstack((dense_centers, numpy.array(self.summed_centers[cluster]/self.densities[cluster])))
            x_src         = self.summed_centers_2[cluster]/self.densities[cluster]
            x_tgt         = (self.summed_centers[cluster]/self.densities[cluster])**2
            #dense_radius += [numpy.sqrt(numpy.max([0, numpy.linalg.norm(x_src) - numpy.linalg.norm(x_tgt)]))]   #[1.8*numpy.sqrt(numpy.max(numpy.maximum(x_src - x_tgt, numpy.zeros(nb_dimensions))))] #[scipy.spatial.distance.cdist(x_src.reshape(1, nb_dimensions), x_tgt.reshape(1, nb_dimensions))[0, 0]]
            dense_radius += [1.8*numpy.sqrt(numpy.max(numpy.maximum(x_src - x_tgt, numpy.zeros(nb_dimensions))))]

        # We check if a point is next to a dense cluster, up to epsilon [TO DEFINE]
        to_be_merged = False
        if len(self.dense_centers) > 0:
            new_dist = scipy.spatial.distance.cdist(new_data, self.dense_centers, 'euclidean')[0]
            dist_min = numpy.min(new_dist)
            dist_idx = self.dense_clusters[numpy.argmin(new_dist)]

            factor     = 2**(-self.decay_factor*(time - self.last_updates[dist_idx]))
            x_src      = (self.summed_centers_2[dist_idx]*factor +  new_data[0]**2)/(self.densities[dist_idx]*factor + 1)
            x_tgt      = ((self.summed_centers[dist_idx]*factor + new_data[0])/(self.densities[dist_idx]*factor + 1))**2
            #new_radius = numpy.sqrt(numpy.max([0, numpy.linalg.norm(x_src) - numpy.linalg.norm(x_tgt)]))   #1.8*numpy.sqrt(numpy.max(numpy.maximum(x_src - x_tgt, numpy.zeros(nb_dimensions))))
            new_radius = 1.8*numpy.sqrt(numpy.max(numpy.maximum(x_src - x_tgt, numpy.zeros(nb_dimensions))))

            to_be_merged = new_radius < self.epsilon #dense_radius[numpy.argmin(new_dist)]

        if to_be_merged:
            # If yes, we assign it to the dense cluster, and update the corresponding density and center
            if verbose:
                print "Data assigned to dense cluster", dist_idx 
            self.densities[dist_idx]        = factor*self.densities[dist_idx] + 1
            self.summed_centers[dist_idx]   = factor*self.summed_centers[dist_idx] + new_data[0]
            self.summed_centers_2[dist_idx] = factor*self.summed_centers_2[dist_idx] + new_data[0]**2
            #densities[dist_idx]        += 1
            #summed_centers[dist_idx]   += new_data[0]
            #summed_centers_2[dist_idx] += new_data[0]**2
            self.last_updates[dist_idx]     = time
        else:
            # If no, we search in the sparse clusters
            sparse_centers = numpy.zeros((0, nb_dimensions))
            sparse_radius  = []
            for cluster in self.sparse_clusters:
                sparse_centers = numpy.vstack((sparse_centers, numpy.array(self.summed_centers[cluster]/self.densities[cluster])))
                x_src          = self.summed_centers_2[cluster]/self.densities[cluster]
                x_tgt          = (self.summed_centers[cluster]/self.densities[cluster])**2
                #sparse_radius += [numpy.sqrt(numpy.max([0, numpy.linalg.norm(x_src) - numpy.linalg.norm(x_tgt)]))]   #[1.8*numpy.sqrt(numpy.max(numpy.maximum(x_src - x_tgt, numpy.zeros(nb_dimensions))))] #[scipy.spatial.distance.cdist(x_src.reshape(1, nb_dimensions), x_tgt.reshape(1, nb_dimensions))[0, 0]]
                sparse_radius  += [1.8*numpy.sqrt(numpy.max(numpy.maximum(x_src - x_tgt, numpy.zeros(nb_dimensions))))]

            to_be_merged = False
            if len(self.sparse_clusters) > 0:
                new_dist = scipy.spatial.distance.cdist(new_data, sparse_centers, 'euclidean')[0]
                dist_min = numpy.min(new_dist)
                dist_idx = self.sparse_clusters[numpy.argmin(new_dist)]
                
                factor     = 2**(-self.decay_factor*(time - self.last_updates[dist_idx]))
                x_src      = (self.summed_centers_2[dist_idx]*factor +  new_data[0]**2)/(self.densities[dist_idx]*factor + 1)
                x_tgt      = ((self.summed_centers[dist_idx]*factor + new_data[0])/(self.densities[dist_idx]*factor + 1))**2
                #new_radius = numpy.sqrt(numpy.max([0, numpy.linalg.norm(x_src) - numpy.linalg.norm(x_tgt)])) #1.8*numpy.sqrt(numpy.max(numpy.maximum(x_src - x_tgt, numpy.zeros(nb_dimensions))))
                new_radius = 1.8*numpy.sqrt(numpy.max(numpy.maximum(x_src - x_tgt, numpy.zeros(nb_dimensions))))

                to_be_merged = new_radius < self.epsilon #dense_radius[numpy.argmin(new_dist)]

                #to_be_merged = dist_min < epsilon #sparse_radius[numpy.argmin(new_dist)]


            if to_be_merged:
                if verbose:
                    print "Data assigned to sparse cluster", dist_idx 
                self.densities[dist_idx]        = factor*self.densities[dist_idx] + 1
                self.summed_centers[dist_idx]   = factor*self.summed_centers[dist_idx] + new_data[0]
                self.summed_centers_2[dist_idx] = factor*self.summed_centers_2[dist_idx] + new_data[0]**2
                #densities[dist_idx]        += 1
                #summed_centers[dist_idx]   += new_data[0]
                #summed_centers_2[dist_idx] += new_data[0]**2
                self.last_updates[dist_idx]     = time
                if self.densities[dist_idx] >= self.D_thred:
                    self.sparse_clusters.remove(dist_idx)
                    self.dense_clusters += [dist_idx]
            else:
                new_idx = numpy.max(densities.keys()) + 1
                if verbose:
                   print "Creating a new cluster", new_idx
                self.densities[new_idx]        = 1.
                self.creation_times[new_idx]   = time
                self.last_updates[new_idx]     = time
                self.summed_centers[new_idx]   = new_data[0]
                self.summed_centers_2[new_idx] = new_data[0]**2
                self.sparse_clusters          += [new_idx]

        if numpy.mod(time, self.time_gap) < 1:

    #        for cluster in (sparse_clusters + dense_clusters):
    #            densities[cluster]        *= 2**(-decay_factor*(time - last_updates[cluster]))
    #            summed_centers[cluster]   *= 2**(-decay_factor*(time - last_updates[cluster]))
    #            summed_centers_2[cluster] *= 2**(-decay_factor*(time - last_updates[cluster]))

            if verbose:
                print "Time gap reached, time to clean clusters..."

            for cluster in self.dense_clusters:
                if self.densities[cluster] < self.D_thred:
                    self.dense_clusters.remove(cluster)
                    self.sparse_clusters += [cluster]

            for cluster in self.sparse_clusters:

                T_0     = self.creation_times[cluster]
                if T_0 < time and T_0 > 0:
                    zeta    = (2**(-self.decay_factor*(time - T_0 + time_gap)) - 1)/(2**(-self.decay_factor*self.time_gap) - 1)
                    delta_t = self.theta*(self.last_updates[cluster] - T_0)/self.densities[cluster] 

                    if self.densities[cluster] < zeta or ((time - self.last_updates[cluster]) > delta_t):
                        if verbose:
                            print "Removing sparse cluster", cluster
                        self.densities.pop(cluster)
                        self.last_updates.pop(cluster)
                        self.summed_centers.pop(cluster)
                        self.summed_centers_2.pop(cluster)
                        self.creation_times.pop(cluster)
                        self.sparse_clusters.remove(cluster)


def distancematrix(data):

    return scipy.spatial.distance.pdist(data, 'euclidean').astype(numpy.float32)

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
        
    dist = distancematrix(data)
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