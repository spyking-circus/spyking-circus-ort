from .block import Block
import numpy
import scipy.interpolate
from circusort.config.probe import Probe
import scipy.optimize, numpy, pylab, scipy.spatial.distance, scipy.stats


class Density_clustering(Block):
    '''TODO add docstring'''

    name = "Density Clustering"

    params = {'alignment'     : True,
              'time_constant' : 1.,
              'sampling_rate' : 20000.,
              'spike_width'   : 5,
              'nb_waveforms'  : 10000, 
              'probe'         : None,
              'radius'        : None,
              'm_ratio'       : 0.01,
              'extraction'    : 'median-raw'}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        if self.probe == None:
            self.log.error('{n}: the probe file must be specified!'.format(n=self.name))
        else:
            self.probe = Probe(self.probe, radius=self.radius, logger=self.log)
            self.log.info('{n} reads the probe layout'.format(n=self.name))
        self.add_input('data')
        self.add_input('pcs')
        self.add_input('peaks')
        self.add_input('mads')
        self.add_output('templates', 'dict')

    def _initialize(self):

        self._spike_width_ = int(self.sampling_rate*self.spike_width*1e-3)
        self.sign_peaks    = []
        self.receive_pcs   = True
        if numpy.mod(self._spike_width_, 2) == 0:
            self._spike_width_ += 1
        self._width = (self._spike_width_-1)//2

        if self.alignment:
            self.cdata = numpy.linspace(-self._width, self._width, 5*self._spike_width_)
            self.xdata = numpy.arange(-2*self._width, 2*self._width + 1)
        return

    @property
    def nb_channels(self):
        return self.inputs['data'].shape[0]

    @property
    def nb_samples(self):
        return self.inputs['data'].shape[1]

    def _is_valid(self, peak):
        if self.alignment:
            return (peak >= 2*self._width) and (peak + 2*self._width < self.nb_samples)
        else:
            return (peak >= self._width) and (peak + self._width < self.nb_samples)

    def _get_best_channel(self, batch, peak, key):
        if key == 'negative':
            channel = numpy.argmin(batch[:, peak])
            is_neg  = True
        elif key == 'positive':
            channel = numpy.argmax(batch[:, peak])
            is_neg  = False
        elif key == 'both':
            if numpy.abs(numpy.max(batch[:, peak])) > numpy.abs(numpy.min(batch[:, peak])):
                channel = numpy.argmax(batch[:, peak])
                is_neg  = False
            else:
                channel = numpy.argmin(batch[:, peak])
                is_neg = True
        return channel, is_neg

    def _get_snippet(self, batch, channel, peak, is_neg):
        if self.alignment:
            indices = self.probe.edges[channel]
            idx     = self.chan_positions[channel]
            zdata   = batch[indices, peak - 2*self._width:peak + 2*self._width + 1]
            ydata   = numpy.arange(len(indices))

            if len(ydata) == 1:
                f        = scipy.interpolate.UnivariateSpline(self.xdata, zdata, s=0)
                if is_neg:
                    rmin = (numpy.argmin(f(self.cdata)) - len(self.cdata)/2.)/5.
                else:
                    rmin = (numpy.argmax(f(self.cdata)) - len(self.cdata)/2.)/5.
                ddata    = numpy.linspace(rmin - self._width, rmin + self._width, self._spike_width_)
                sub_mat  = f(ddata).astype(numpy.float32).reshape(self._spike_width_, 1)
            else:
                f        = scipy.interpolate.RectBivariateSpline(ydata, self.xdata, zdata, s=0, ky=min(len(ydata)-1, 3))
                if is_neg:
                    rmin = (numpy.argmin(f(self.cdata, idx)[:, 0]) - len(self.cdata)/2.)/5.
                else:
                    rmin = (numpy.argmax(f(self.cdata, idx)[:, 0]) - len(self.cdata)/2.)/5.
                ddata    = numpy.linspace(rmin-self._width, rmin+self._width, self._spike_width_)
                sub_mat  = f(ddata, ydata).astype(numpy.float32)
        else:
            sub_mat = batch[indices, peak - self._width:peak + self._width + 1]

        return sub_mat

    def _guess_output_endpoints(self):
        if self.inputs['data'].dtype is not None:
            self.decay_time = numpy.exp(-self.nb_samples/float(self.time_constant))
            #self.output.configure(dtype=self.input.dtype, shape=self.input.shape)        
            self.chan_positions = numpy.zeros(self.nb_channels, dtype=numpy.int32)
            for channel in xrange(self.nb_channels):
                #indices = numpy.take(inv_nodes, edges[nodes[i]])
                self.chan_positions[channel] = numpy.where(self.probe.edges[channel] == channel)[0]
                
    def _init_data_structures(self):
        self.pca_data  = {}
        self.raw_data  = {}
        self.clusters  = {}
        self.templates = {}
        if not numpy.all(self.pcs[0] == 0):
            self.sign_peaks += ['negative']
        if not numpy.all(self.pcs[1] == 0):
            self.sign_peaks += ['positive']
        self.log.debug("{n} will detect peaks {s}".format(n=self.name, s=self.sign_peaks))

        for key in self.sign_peaks:
            self.pca_data[key]  = {}
            self.raw_data[key]  = {}
            self.clusters[key]  = {}
            self.templates[key] = {}

        for key in self.sign_peaks:
            for channel in xrange(self.nb_channels):
                self._reset_data_structures(key, channel)


    def _perform_clustering(self, key, channel):
        a, b, c = self.pca_data[key][channel].shape
        self.log.info("{n} clusters {m} waveforms on channel {d}".format(n=self.name_and_counter, m=a, d=channel))
        data    = self.pca_data[key][channel].reshape(a, b*c)
        rho, dist, sdist, nb_selec = rho_estimation(data, compute_rho=True, mratio=self.m_ratio)
        self.clusters[key][channel], r, d, c = clustering(rho, dist, smart_select=True)
        ### SHould we add the merging step
        self._update_templates(key, channel)


    def _update_templates(self, key, channel):

        labels = numpy.unique(self.clusters[key][channel])
        labels = labels[labels > -1]
        self.log.debug("{n} found {m} templates on channel {d}".format(n=self.name_and_counter, m=len(labels), d=channel))
        for l in labels:
            indices = numpy.where(labels == l)[0]
            data = self.raw_data[key][channel][indices]
            if self.extraction == 'mean-raw':
                template = numpy.mean(data, 0)
            elif self.extraction == 'median-raw':
                template = numpy.median(data, 0)
            template = template.reshape(template.shape[0], template.shape[1]).T
            template = self._center_template(template, key)

            ## Here we should compress the templates for large-scale
            #template = self._sparsify_template(template, channel)
            
            self.templates[key][channel] = numpy.vstack((self.templates[key][channel], template.reshape(1, template.shape[0], template.shape[1])))
            self.to_reset += [(key, channel)]

    def _center_template(self, template, key):
        if key == 'negative':
            tmpidx = divmod(template.argmin(), template.shape[1])
        elif key == 'positive':
            tmpidx = divmod(template.argmax(), template.shape[1])

        shift            = self._width - tmpidx[1]

        aligned_template = numpy.zeros(template.shape, dtype=numpy.float32)
        if shift > 0:
            aligned_template[:, shift:] = template[:, :-shift]
        elif shift < 0:
            aligned_template[:, :shift] = template[:, -shift:]
        else:
            aligned_template = template
        return aligned_template


    def _sparsify_template(self, template, channel):
        for i in xrange(template.shape[0]):
            if (numpy.abs(templates[i]).max() < 0.5*self.thresholds[i]):
                template[i] = 0
        return template


    def _reset_data_structures(self, key, channel):
        self.pca_data[key][channel]  = numpy.zeros((0, self.pcs.shape[1], len(self.probe.edges[channel])), dtype=numpy.float32)
        self.raw_data[key][channel]  = numpy.zeros((0, self._spike_width_, len(self.probe.edges[channel])), dtype=numpy.float32)
        self.clusters[key][channel]  = numpy.zeros(0, dtype=numpy.int32)
        self.templates[key][channel] = numpy.zeros((0, len(self.probe.edges[channel]), self._spike_width_), dtype=numpy.float32)


    def _process(self):
        if self.receive_pcs:
            self.pcs = self.inputs['pcs'].receive()
            self.log.info("{n} receives the PCA matrices".format(n=self.name_and_counter))
            self.receive_pcs = False
            self._init_data_structures()
        
        self.to_reset = []

        batch = self.inputs['data'].receive()
        peaks = self.inputs['peaks'].receive()
        self.thresholds = self.inputs['mads'].receive()

        for key in self.sign_peaks:
            for channel, signed_peaks in peaks[key].items():
                for peak in signed_peaks:
                    if self._is_valid(peak):
                        channel, is_neg = self._get_best_channel(batch, peak, key)
                        waveforms       = self._get_snippet(batch, channel, peak, is_neg)
                        projection      = numpy.dot(self.pcs[0], waveforms)
                        projection      = projection.reshape(1, projection.shape[0], projection.shape[1])
                        waveforms       = waveforms.reshape(1, waveforms.shape[0], waveforms.shape[1])
                        if is_neg:
                            key = 'negative'
                        else:
                            key = 'positive'

                        self.pca_data[key][channel] = numpy.vstack((self.pca_data[key][channel], projection))
                        self.raw_data[key][channel] = numpy.vstack((self.raw_data[key][channel], waveforms))
                
                if len(self.pca_data[key][channel]) >= self.nb_waveforms:
                    self._perform_clustering(key, channel)

        self.outputs['templates'].send(self.templates)
        for key, channel in self.to_reset:
            self._reset_data_structures(key, channel)

        return


def distancematrix(data, ydata=None):
    
    if ydata is None:
        distances = scipy.spatial.distance.pdist(data, 'euclidean')
    else:
        distances = scipy.spatial.distance.cdist(data, ydata, 'euclidean')
    return distances.astype(numpy.float32)



def fit_rho_delta(xdata, ydata, smart_select=False, display=False, max_clusters=10, save=False):

    if smart_select:

        xmax = xdata.max()
        off  = ydata.min()
        idx  = numpy.argmin(xdata)
        a_0  = (ydata[idx] - off)/numpy.log(1 + (xmax - xdata[idx]))

        def myfunc(x, a, b, c, d):
            return a*numpy.log(1. + c*((xmax - x)**b)) + d

        def imyfunc(x, a, b, c, d):
            return numpy.exp(d)*((1. + c*(xmax - x)**b)**a)

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

    if display:
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.plot(xdata, ydata, 'ko')
        ax.plot(xdata[subidx[:len(subidx)]], ydata[subidx[:len(subidx)]], 'ro')
        if smart_select:
            idx = numpy.argsort(xdata)
            ax.plot(xdata[idx], prediction[idx], 'r')
            ax.set_yscale('log')
        if save:
            pylab.savefig(os.path.join(save[0], 'rho_delta_%s.png' %(save[1])))
            pylab.close()
        else:
            pylab.show()
    return subidx, len(subidx)


def rho_estimation(data, update=None, compute_rho=True, mratio=0.01):

    N     = len(data)
    rho   = numpy.zeros(N, dtype=numpy.float32)
        
    if update is None:
        dist = distancematrix(data)
        didx = lambda i,j: i*N + j - i*(i+1)//2 - i - 1
        nb_selec = max(5, int(mratio*N))
        sdist    = {}

        if compute_rho:
            for i in xrange(N):
                indices  = numpy.concatenate((didx(i, numpy.arange(i+1, N)), didx(numpy.arange(0, i-1), i)))
                tmp      = numpy.argsort(numpy.take(dist, indices))[:nb_selec]
                sdist[i] = numpy.take(dist, numpy.take(indices, tmp))
                rho[i]   = numpy.mean(sdist[i])

    else:
        M        = len(update[0])
        nb_selec = max(5, int(mratio*M))
        sdist    = {}  

        for i in xrange(N):
            dist     = distancematrix(data[i].reshape(1, len(data[i])), update[0]).ravel()
            all_dist = numpy.concatenate((dist, update[1][i]))
            idx      = numpy.argsort(all_dist)[:nb_selec]
            sdist[i] = numpy.take(all_dist, idx)
            rho[i]   = numpy.mean(sdist[i])
    return rho, dist, sdist, nb_selec


def clustering(rho, dist, smart_select=True, n_min=None, max_clusters=10):

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

    return halo, rho, delta, clust_idx[:max_clusters]