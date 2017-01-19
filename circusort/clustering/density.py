import numpy, pylab

def fit_rho_delta(xdata, ydata, display=False, threshold=0, max_clusters=10, save=False):

    #threshold = xdata[numpy.argsort(xdata)][int(len(xdata)*threshold/100.)]
    gidx   = numpy.where(xdata >= threshold)[0]
    ymdata = ydata[gidx]  
    xmdata = xdata[gidx]
    subidx = numpy.take(gidx, numpy.argsort(xmdata*numpy.log(1 + ymdata))[::-1])

    if display:
        ax.plot(xdata[subidx[:max_clusters]], ydata[subidx[:max_clusters]], 'ro')
        if save:
            pylab.savefig(os.path.join(save[0], 'rho_delta_%s.png' %(save[1])))
            pylab.close()
        else:
            pylab.show()
    return subidx


def rho_estimation(data, compute_rho=True, mratio=0.1):

    N    = len(data)
    rho  = numpy.zeros(N, dtype=numpy.float32)
        
    dist = scipy.spatial.distance.pdist(data, 'euclidean')
    didx = lambda i,j: i*N + j - i*(i+1)//2 - i - 1

    if compute_rho:
        for i in xrange(N):
            indices = numpy.concatenate((didx(i, numpy.arange(i+1, N)), didx(numpy.arange(0, i-1), i)))
            tmp     = numpy.argsort(numpy.take(dist, indices))[:max(1, int(mratio*N))]
            rho[i]  = numpy.sum(numpy.take(dist, numpy.take(indices, tmp)))  

    return rho, dist


def clustering(rho, dist, display=None, n_min=None, max_clusters=10, save=False):

    N                 = len(rho)
    maxd              = numpy.max(dist)
    didx              = lambda i,j: i*N + j - i*(i+1)//2 - i - 1
    ordrho            = numpy.argsort(rho)[::-1]
    delta, nneigh     = numpy.zeros(N, dtype=numpy.float64), numpy.zeros(N, dtype=numpy.int32)
    delta[ordrho[0]]  = -1
    for ii in xrange(N):
        delta[ordrho[ii]] = maxd
        for jj in xrange(ii):
            if ordrho[jj] > ordrho[ii]:
                xdist = dist[didx(ordrho[ii], ordrho[jj])]
            else:
                xdist = dist[didx(ordrho[jj], ordrho[ii])]

            if xdist < delta[ordrho[ii]]:
                delta[ordrho[ii]]  = xdist
                nneigh[ordrho[ii]] = ordrho[jj]

    delta[ordrho[0]] = delta.max()
    clust_idx        = fit_rho_delta(rho, delta, max_clusters=max_clusters)

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

    return halo, rho, delta, clust_idx


def merging(groups, sim_same_elec, data):

    def perform_merging(groups, sim_same_elec, data):
        mask      = numpy.where(groups > -1)[0]
        clusters  = numpy.unique(groups[mask])
        dmin      = numpy.inf
        to_merge  = [None, None]
        
        for ic1 in xrange(len(clusters)):
            idx1 = numpy.where(groups == clusters[ic1])[0]
            sd1  = numpy.take(data, idx1, axis=0)
            m1   = numpy.median(sd1, 0)
            for ic2 in xrange(ic1+1, len(clusters)):
                idx2 = numpy.where(groups == clusters[ic2])[0]
                sd2  = numpy.take(data, idx2, axis=0)
                m2   = numpy.median(sd2, 0)
                v_n  = m1 - m2      
                pr_1 = numpy.dot(sd1, v_n)
                pr_2 = numpy.dot(sd2, v_n)
                norm = numpy.median(numpy.abs(pr_1 - numpy.median(pr_1)))**2 + numpy.median(numpy.abs(pr_2 - numpy.median(pr_2)))**2
                dist = numpy.abs(numpy.median(pr_1) - numpy.median(pr_2))/numpy.sqrt(norm)
                    
                if dist < dmin:
                    dmin     = dist
                    to_merge = [ic1, ic2]

        if dmin < sim_same_elec:
            groups[numpy.where(groups == clusters[to_merge[1]])[0]] = clusters[to_merge[0]]
            return True, groups, to_merge
        
        return False, groups, []

    has_been_merged = True
    mask            = numpy.where(groups > -1)[0]
    clusters        = numpy.unique(groups[mask])
    merged          = []

    while has_been_merged:
        has_been_merged, groups, to_merge = perform_merging(groups, sim_same_elec, data)
        if has_been_merged:
            merged += [to_merge]
    return groups, merged