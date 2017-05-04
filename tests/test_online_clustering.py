from circusort.utils.clustering import *
import numpy

nb_points     = 1000
nb_clusters   = 2
nb_extras     = 5
nb_dimensions = 2
decay_factor  = 0.35
sim_same_elec = 3
mu            = 4
epsilon       = 0.5
nb_stream     = 10000
offset        = 0
theta         = -numpy.log(0.001)
time_stream   = numpy.round(numpy.linspace(0, nb_stream/2., nb_extras))


### First initialization of the algorithm ####

gt_centers = []
for i in xrange(nb_clusters+nb_extras):
    gt_centers += [100*numpy.random.rand(nb_dimensions)]

data = numpy.zeros((nb_points, nb_dimensions), dtype=numpy.float32)

for i in xrange(nb_points):
    idx = numpy.random.randint(0, nb_clusters)
    data[i] = numpy.random.randn(nb_dimensions) + gt_centers[idx]

rhos, dist, _  = rho_estimation(data)
rhos           = -rhos + rhos.max() 
labels, c      = density_based_clustering(rhos, dist, n_min=None)


manager = OnlineManager()
manager.initialize(0, data, labels)

for time in xrange(nb_stream):

    idx          = numpy.random.randint(0, nb_clusters+numpy.searchsorted(time_stream, time))
    new_data     = numpy.array([numpy.random.randn(nb_dimensions) + gt_centers[idx]])

    manager.update(time, new_data)
    if manager.nb_updates == 2500:
        manager.cluster()