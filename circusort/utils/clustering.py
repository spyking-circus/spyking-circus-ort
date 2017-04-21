import scipy.optimize, numpy, pylab, scipy.spatial.distance, scipy.stats
import warnings
warnings.filterwarnings("ignore")

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