import numpy

class Cluster(object):

    def __init__(self, creation_time, label="sparse", data=None):
        self.density       = 1
        self.label         = label
        self.last_update   = creation_time
        if data is None:
            self.sum_centers    = 0
            self.sum_centers_sq = 0
        else:
            self.sum_centers    = data
            self.sum_centers_sq = data**2
        self.creation_time = creation_time

    def add_and_update(self, data, decay_time, time):
        factor              = 2**(-decay_time*(time - self.last_update))
        self.density        = factor*self.density + 1
        self.sum_centers    = factor*self.sum_centers + data
        self.sum_centers_sq = factor*self.sum_centers_sq + data**2
        self.last_update    = time

    def update(self, decay_time, time):
        factor               = 2**-decay_time
        self.density        *= factor
        self.sum_centers    *= factor
        self.sum_centers_sq *= factor
        self.last_update     = time

    def add(self, data):
        self.density        += 1
        self.sum_centers    += data
        self.sum_centers_sq += data**2

    def remove(self, data):
        self.density        -= 1
        self.sum_centers    -= data
        self.sum_centers_sq -= data**2

    @property
    def center(self):
        return self.sum_centers/self.density

    @property
    def is_dense(self):
        return self.label == 'dense'

    @property
    def is_sparse(self):
        return not self.is_dense

    def set_label(self, value):
        assert value in ['sparse', 'dense']
        self.label = value

    def get_z_score(self, data, sigma):
        return numpy.linalg.norm(self.center - data)/sigma

    @property
    def sigma(self):
        return numpy.sqrt(numpy.linalg.norm(self.sum_centers_sq/self.density - self.center**2))