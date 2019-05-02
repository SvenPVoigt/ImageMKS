def local_avg(img, rad, mask=None):
    ogS = img.shape
    ex_pad = 1
    img = np.pad(img, pad_width=((0,rad+ex_pad),(0,rad+ex_pad)),
                 mode='constant', constant_values=0)

    ySize = img.shape[0]
    xSize = img.shape[1]

    y, x = np.ogrid[0:ySize, 0:xSize]
    y = y - ySize//2
    y = np.roll(y, ySize//2, axis = 0)
    x = x - xSize//2
    x = np.roll(x, xSize//2, axis = 1)

    c_kernel = (x**2 + y**2 <= rad**2).astype(np.float64)

    if mask is None:
        norm = np.ones(ogS)
    else:
        norm = mask.copy()

    norm = np.pad(norm, pad_width=((0,rad+ex_pad),(0,rad+ex_pad)),
                  mode='constant', constant_values=0)

    H1 = fftn(img)
    H2 = fftn(c_kernel)
    H3 = fftn(norm)

    loc_sum = ifftn(H1*H2.conj()).real
    # loc_sum = loc_sum[rad+ex_pad:ogS[0]+rad+ex_pad, rad+ex_pad:ogS[1]+rad+ex_pad]
    loc_sum = loc_sum[:ogS[0],:ogS[1]]
    loc_norm = ifftn(H3*H2.conj()).real
    # loc_norm = loc_norm[rad+ex_pad:ogS[0]+rad+ex_pad, rad+ex_pad:ogS[1]+rad+ex_pad]
    loc_norm = loc_norm[:ogS[0], :ogS[1]]

    if mask is None:
        loc_avg = np.divide(loc_sum, loc_norm)
    else:
        loc_avg = np.divide(loc_sum, loc_norm, where=loc_norm!=0) * (loc_norm>0)

    return (loc_avg, loc_sum, loc_norm)
