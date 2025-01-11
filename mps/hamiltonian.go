package mps

import (
	"qising/mat"
	"qising/tensor"
)

var (
	zero = [][]complex64{
		{0, 0},
		{0, 0},
	}
	identity = [][]complex64{
		{1, 0},
		{0, 1},
	}
)

func MagnetizationZ(n [2]int) []*tensor.Dense {
	w := tensor.T4([][][][]complex64{
		{identity, zero},
		{mat.PauliZ, identity},
	})
	return newMPO(w, n)
}

func Ising(n [2]int, h complex64) []*tensor.Dense {
	mul := func(c complex64, x [][]complex64) [][]complex64 {
		return tensor.T2(x).Mul(c).ToSlice2()
	}
	w := tensor.T4([][][][]complex64{
		{identity, zero, zero},
		{mat.PauliZ, zero, zero},
		{mul(-h, mat.PauliX), mul(-1, mat.PauliZ), identity},
	})
	return newMPO(w, n)
}

func newMPO(w *tensor.Dense, n [2]int) []*tensor.Dense {
	d0, d1, d2, d3 := w.Shape()[0], w.Shape()[1], w.Shape()[2], w.Shape()[3]
	mpo := make([]*tensor.Dense, 0, n[0])

	// First MPO is w[-1].
	mpo = append(mpo, w.Slice([][2]int{{d0 - 1, d0}, {0, d1}, {0, d2}, {0, d3}}))

	for _ = range n[0] - 2 {
		mpo = append(mpo, w)
	}

	// Last MPO is w[:, 0].
	mpo = append(mpo, w.Slice([][2]int{{0, d0}, {0, 1}, {0, d2}, {0, d3}}))

	return mpo
}
