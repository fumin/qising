import argparse
import collections
import csv
import logging
import os

import numpy as np
import scipy


def wc(fpath):
    cnt = 0
    has_imag = False
    with open(fpath, "r") as f:
        r = csv.reader(f)
        for row in r:
            cnt += 1
            if "j" in row[0]:
                has_imag = True

    dtype = np.float32
    if has_imag:
        dtype = np.complex64
    return cnt, dtype


vRowCol = collections.namedtuple("vRowCol", ["v", "row", "col"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-coo", dest="coo_dir", default="")
    parser.add_argument("-eig", dest="eig", default="")
    args = parser.parse_args()

    logging.basicConfig()
    lg = logging.getLogger()
    lg.setLevel(logging.INFO)
    lg.handlers[0].setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(pathname)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    # Read matrix information.
    shapePath = os.path.join(args.coo_dir, "shape.csv")
    shape = np.genfromtxt(shapePath, dtype=int, delimiter=",")
    cooPath = os.path.join(args.coo_dir, "coo.csv")
    nnz, dtype = wc(cooPath)
    m = scipy.sparse.coo_matrix((shape[0], shape[1]), dtype=dtype)

    # Read matrix data.
    m.data = np.zeros(nnz, dtype=m.dtype)
    coord_type = np.int64
    if nnz < 1<<32:
        coord_type = np.int32
    logging.info("number of non-zeros %d, dtype %s %s", nnz, dtype, coord_type)
    m.coords = (np.zeros(nnz, dtype=coord_type), np.zeros(nnz, dtype=coord_type))
    prev = vRowCol(v=np.nan, row=-1, col=-1)
    with open(cooPath, "r") as f:
        r = csv.reader(f)
        for coorow in r:
            i = r.line_num - 1

            if coorow[0] == "":
                v = prev.v
            else:
                v = dtype(coorow[0])

            if coorow[1] == "":
                y = prev.row
            else:
                y = coord_type(coorow[1])

            x = coord_type(coorow[2])

            m.data[i] = v
            m.coords[0][i] = y
            m.coords[1][i] = x

            prev = vRowCol(v=v, row=y, col=x)

            if i % 1e7 == 0:
                logging.info("%.0f%% %d/%d", 100*i/len(m.data), i, len(m.data))

    # Compute eigenvalue.
    k = 3
    # vals, vecs = scipy.sparse.linalg.eigs(m, which="SR", k=k)
    vals, vecs = scipy.sparse.linalg.eigsh(m, which="SA", k=k)

    # Write eigenvalue.
    vecs = np.insert(vecs, 0, vals, axis=0)
    np.savetxt(args.eig, vecs, delimiter=",")


if __name__ == "__main__":
    main()
