from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering

def tofloat(A):
    return np.interp(A, (np.amin(A), np.amax(A)), (0,1)).astype(np.float32)


def toint(A):
    return np.interp(A, (np.amin(A), np.amax(A)), (0,255)).astype(np.uint8)


def image_coords(shape, centered=False, aslist=False):
    s0, s1 = shape

    if centered:
        X, Y = np.meshgrid(np.arange(-s1//2+s1%2, s1//2+s1%2), np.arange(-s0//2+s0%2, s0//2+s0%2))
    else:
        X, Y = np.meshgrid(np.arange(B0.shape[1]), np.arange(B0.shape[0]))

    if aslist:
        return [(x,y) for x,y in zip(X.ravel(), Y.ravel())]
    else:
        return (X, Y)


def image_plane(A):
    coords = image_coords(A.shape, aslist=True)
    lr = LinearRegression()
    lr.fit(coords, A.reshape(-1,1))

    return lr.predict(coords).reshape(A.shape)


def image_clusters(A, n=3):
    km = KMeans(n_clusters=n)
    km.fit(A.reshape(-1,1))
    return km.labels_.reshape(A.shape)


def image_inrange(A, minp, maxp):
    minp = np.percentile(A, minp)
    maxp = np.percentile(A, maxp)
    return rescale_intensity(A, in_range=(minp, maxp))


def locations(shape, r, step=1):
    sy, sx = shape
    X, Y = np.meshgrid(np.arange(r, sx-r, step), np.arange(r, sy-r, step))
    return [i for i in zip(Y.ravel(), X.ravel())]


def divide_image(img, r, step=1, max_n=None, start_n=0):
    shape = (img.shape[0], img.shape[1])
    locs = locations(shape, r, step)[start_n:]

    if img.ndim == 2:
        A = np.zeros((len(locs), (2*r+1)**2))
        for n, loc in enumerate(locs):
            j, i = loc
            A[n,:] = img[(j-r):(j+r+1), (i-r):(i+r+1)].ravel()

            if max_n:
                if n > max_n:
                    break
    else:
        A = np.zeros((len(locs), 3*(2*r+1)**2))
        for n, loc in enumerate(locs):
            j, i = loc
            A[n,:] = img[(j-r):(j+r+1), (i-r):(i+r+1), :].ravel()

            if max_n:
                if n > max_n:
                    break

    return A


def cluster_neighborhoods(A, r, n_c=None, kmeans=None, pca=None):
    if pca is None:
        pca = PCA()
    kmeans = KMeans(n_clusters=n_c)
    new_size = tuple(i-2*r for i in A.shape)

    X = divide_image(A, r, 1)

    pca.fit(X)
    X = pca.transform(X)

    kmeans.fit(X)
    S = kmeans.predict(X)

    S = S.reshape(new_size)

    A = A[r:(A.shape[0]-r),r:(A.shape[1]-r)]
    A = np.interp(A, (np.amin(A), np.amax(A)), (0,1))

    return (S, A)
