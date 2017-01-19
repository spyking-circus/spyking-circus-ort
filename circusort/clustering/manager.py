import numpy

logger = logging.getLogger(__name__)

class OnlineClusteringManager(object):

    def __init__(self, datastream, decay_time=0.35, mu=4, nb_init=10000, sim_same_elec=3, verbose=False, radius=3, snapshot=200, display=True, t_stop=numpy.inf):
        self.time       = 0
        self.radius     = radius
        self.decay_time = decay_time
        self.datastream = datastream
        self.verbose    = verbose
        self.mu         = mu
        self.display    = display
        self.nb_init    = nb_init
        self.t_stop     = t_stop
        self.nb_snaps   = 4
        self.snapshot   = t_stop/self.nb_snaps
        self.sim_same_elec = 3
        self.real_clusters = {'indices' : [], 'mus' : [], 'sigmas' : []}
        self.history    = {}

        print "Starting the first clustering..."
        
        if self.display:
            self.fig     = pylab.figure()
            self.ax_grid = int(numpy.sqrt(self.nb_snaps))
            self.gs      = pylab.GridSpec(self.ax_grid, 2*self.ax_grid)
            self.gcount  = 0

        self.first_clustering()  
        self.history[self.time] = self.real_clusters.copy()

        for i in xrange(int(self.t_stop)):
            self.process()
            if numpy.mod(self.time, self.snapshot) == 0:
                self.offline_clustering()
                self.history[self.time] = self.real_clusters.copy()

        if self.display:
            pylab.show()

    def offline_clustering(self):
        block     = numpy.zeros((0, self.datastream.nb_dimensions))
        densities = numpy.zeros(0, dtype=numpy.float32)
        for cluster in self.dense_clusters:
            cluster.update(self.decay_time, self.time)
            block     = numpy.vstack((block, cluster.center))
            densities = numpy.concatenate((densities, [cluster.density]))

        background = numpy.zeros((0, self.datastream.nb_dimensions))
        for cluster in self.sparse_clusters:
            background = numpy.vstack((background, cluster.center))

        if len(block) > 1:
            rhos, dist        = rho_estimation(block, compute_rho=False)
            rhos              = -rhos + rhos.max() 
            clusters, r, d, c = clustering(rhos, dist, n_min=2)
            
            data = numpy.zeros((0, self.datastream.nb_dimensions), dtype=numpy.float32)
            for cluster in self.dense_clusters:
                data = numpy.vstack((data, [cluster.center]))


            clusters, merged = merging(clusters, self.sim_same_elec, data)
            indices          = numpy.unique(clusters[clusters > -1])
            mus              = []
            sigmas           = []

            for count, i in enumerate(indices):
                idx  = c[:11][i]
                mus += [self.dense_clusters[idx].center]
                tmp  = []
                for j in numpy.where(clusters == i)[0]:
                    tmp += [self.dense_clusters[j].center]

                sigmas += [numpy.mean(numpy.std(numpy.array(tmp), 0))]


            new_clusters = {'indices' : indices, 'mus' : numpy.array(mus), 'sigmas' : numpy.array(sigmas)}
            self.establish_matches(new_clusters)

        else:    
            clusters = []
            c        = []
            indices  = []
            mus      = []
            sigmas   = []

        if self.display:
            ax = pylab.subplot(self.gs[self.gcount/self.ax_grid, self.ax_grid + numpy.mod(self.gcount, self.ax_grid)])
            self.show(block, clusters, mus, background=background, ax=ax, dispersion=sigmas)
            self.gcount += 1

        
    def first_clustering(self):
        ## First we initialize the ClusterManager with a first clustering
        block             = self.datastream.get_data_block(self.nb_init, self.time)
        #self.time        += nb_init
        rhos, dist        = rho_estimation(block)
        rhos              = -rhos + rhos.max() 
        clusters, r, d, c = clustering(rhos, dist, n_min=10)
        clusters, merged  = merging(clusters, self.sim_same_elec, block)
        self.all_clusters = []
        mus               = []
        sigmas            = []

        for count, i in enumerate(numpy.unique(clusters[clusters > -1])):
            single_cluster                = Cluster(0, "sparse")
            single_cluster.density        = float(numpy.sum(clusters == i))
            single_cluster.sum_centers    = numpy.sum(block[clusters == i], 0)
            single_cluster.sum_centers_sq = numpy.sum(block[clusters == i]**2, 0)
            mus                          += [single_cluster.center]
            sigmas                       += [single_cluster.sigma]
            self.all_clusters            += [single_cluster]
            
        self.set_D_thred()

        for cluster in self.all_clusters:
            if cluster.density >= self.D_thred:
                cluster.set_label('dense')

        self.set_time_gap()
        if self.display:
            ax = pylab.subplot(self.gs[:, :self.ax_grid])
            self.show(block, clusters, mus, ax=ax, dispersion=sigmas)

        self.real_clusters = {'indices' : numpy.arange(len(mus)), 'mus' : numpy.array(mus), 'sigmas' : numpy.array(sigmas)}


    def establish_matches(self, new_clusters):

        max_idx = self.real_clusters['indices'].max()

        for count in xrange(len(new_clusters['indices'])):
            new_dist = scipy.spatial.distance.cdist(numpy.array([new_clusters['mus'][count]]), self.real_clusters['mus'], 'euclidean')
            dist_min = numpy.min(new_dist)
            dist_idx = numpy.argmin(new_dist)

            if dist_min < 3*max(new_clusters['sigmas'][count], self.real_clusters['sigmas'][dist_idx]):
                if self.verbose:
                    print "Match between target", new_clusters['indices'][count], "and source", dist_idx
                new_clusters['indices'][count] = self.real_clusters['indices'][dist_idx] 
            else:
                max_idx += 1
                new_clusters['indices'][count] = max_idx
                if self.verbose:
                    print "No match for target", new_clusters['indices'][count], "assigned to", max_idx

        self.real_clusters = new_clusters



    def show(self, data, clusters, centers, ax, dispersion=None, background=None):
        
        if background is not None:
            ax.scatter(background[:,0], background[:,1], c='k', alpha=0.1)
        ax.scatter(data[:,0], data[:,1], c=clusters)
        for idx in xrange(len(centers)):
            if dispersion is not None:
                rect = pylab.Circle((centers[idx][0], centers[idx][1]), dispersion[idx], facecolor="#FFFFFF", edgecolor="#000000", alpha=0.5)
                ax.add_patch(rect)
            ax.scatter(centers[idx][0], centers[idx][1], c='w')
            

        ax.set_title('T=%d [%d/%d]' %(self.time, self.nb_dense, self.nb_sparse))
        #pylab.setp(ax, xticks=[], yticks=[])
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        pylab.draw()

    def get_clusters(self, label):
        clusters = []
        for cluster in self.all_clusters:
            if label == 'dense' and cluster.is_dense:
                clusters += [cluster]
            if label =='sparse' and cluster.is_sparse:
                clusters += [cluster]

        return clusters

    def set_D_thred(self):
        self.D_thred = self.mu/(len(self.all_clusters)*(1-2**(-self.decay_time)))
    
    def set_time_gap(self):
        if self.D_thred > 1:
            self.time_gap = numpy.ceil((1/(self.decay_time))*numpy.log(self.D_thred/(self.D_thred - 1)))
        else:
            self.time_gap = 100

    @property
    def sparse_clusters(self):
        return self.get_clusters('sparse')

    @property
    def dense_clusters(self):
        return self.get_clusters('dense')

    @property
    def nb_dense(self):
        return len(self.dense_clusters)

    @property
    def nb_sparse(self):
        return len(self.sparse_clusters)

    def merged_into(self, data, label):
        
        clusters     = self.get_clusters(label)
        to_be_merged = False
        if len(clusters) > 0:
            centers  = numpy.zeros((0, self.datastream.nb_dimensions))
            for cluster in clusters:
                centers  = numpy.vstack((centers, cluster.center))

            new_dist = scipy.spatial.distance.cdist(data, centers, 'euclidean')[0]
            dist_min = numpy.min(new_dist)
            dist_idx = numpy.argmin(new_dist)

            clusters[dist_idx].add_and_update(data[0], self.decay_time, self.time)
            sigma    = clusters[dist_idx].sigma
            
            if sigma == 0:
                sigma = self.get_mean_nnz_sigma()
            
            to_be_merged = clusters[dist_idx].get_z_score(data[0], sigma) <= self.radius

            if to_be_merged:
                if label == 'sparse':
                    if clusters[dist_idx].density >= self.D_thred:
                        clusters[dist_idx].set_label('dense')
            else:
                clusters[dist_idx].remove(data[0])

        return to_be_merged


    def get_mean_nnz_sigma(self):
        sigma = 0
        count = 0
        for cluster in self.all_clusters:
            if cluster.sigma > 0:
                sigma += cluster.sigma
                count += 1
        return sigma/count

    def prune_clusters(self):

        if self.verbose:
            print "Time gap reached, time to clean clusters..."

        for cluster in self.all_clusters:
            if cluster.density < self.D_thred:
                cluster.set_label('sparse')

        to_remove = []
        for cluster in self.sparse_clusters:

            if cluster.creation_time < self.time and cluster.creation_time > 0:
                zeta    = (2**(-self.decay_time*(self.time - cluster.creation_time + self.time_gap)) - 1)/(2**(-self.decay_time*self.time_gap) - 1)
                delta_t = theta*(cluster.last_update - cluster.creation_time)/cluster.density

                if cluster.density < zeta or ((self.time - cluster.last_update) > delta_t):
                    if self.verbose:
                        print "Removing sparse cluster"
                to_remove += [cluster]

        for cluster in to_remove:
            self.all_clusters.remove(cluster)

    def process(self):
        new_data   = self.datastream.get_next_data_point(self.time)
        if self.verbose:
            print "Processing time", self.time, "sparse/dense=%d/%d" %(self.nb_sparse, self.nb_dense) 
        if self.merged_into(new_data, 'dense'):
            if self.verbose:
                print "We merged the time point", self.time, 'into a dense cluster'
        else:
            if self.merged_into(new_data, 'sparse'):
                if self.verbose:
                    print "We merged the time point", self.time, 'into a sparse cluster'
            else:
                if self.verbose:
                    print "We can not merged time point", self.time, "so creating a new sparse cluster"
                
                self.all_clusters += [Cluster(self.time, data=new_data[0])]

        if numpy.mod(self.time, self.time_gap) == 0:
            self.prune_clusters()

        self.set_D_thred()
        self.set_time_gap()
        self.time += 1 
        