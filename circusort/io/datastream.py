import numpy

class DataStream(object):

    def __init__(self, nb_dimensions):
        self.nb_dimensions = nb_dimensions

    def get_next_data_point(self, time=None):
        pass

    def get_data_block(self, block_size, time=None):
        block = numpy.zeros((block_size, nb_dimensions))
        for count in xrange(block_size):
            block[count] = self.get_next_data_point(time)
        return block

class GaussianTest(DataStream):

    def __init__(self, nb_dimensions, nb_clusters, nb_extras, breakpoints, verbose=False):
        DataStream.__init__(self, nb_dimensions)
        self.centers     = []
        self.nb_clusters = nb_clusters
        self.nb_extras   = nb_extras
        self.breakpoints = numpy.sort(breakpoints)
        assert len(breakpoints) == nb_extras

        for i in xrange(nb_clusters+nb_extras):
           self.centers += [100*numpy.random.rand(self.nb_dimensions)]


    def get_next_data_point(self, time):
        idx  = numpy.random.randint(0, nb_clusters+numpy.searchsorted(self.breakpoints, time))
        data = numpy.array([numpy.random.randn(self.nb_dimensions) + self.centers[idx]])
        return data