package mps

import (
	"qising/mat"
	"qising/tensor"
)

func Ising(n [2]int, h complex64) []*tensor.Dense {
	zero := [][]complex64{
		{0, 0},
		{0, 0},
	}
	identity := [][]complex64{
		{1, 0},
		{0, 1},
	}
	nhPauliX := tensor.Mul(tensor.Zeros(2, 2), -h, tensor.T2(mat.PauliX)).ToSlice2()
	w := tensor.T4([][][][]complex64{
		{identity, zero, zero},
		{mat.PauliZ, zero, zero},
		{nhPauliX, mat.PauliZ, identity},
	})
	r0, r1, r2, r3 := w.Shape()[0], w.Shape()[1], w.Shape()[2], w.Shape()[3]

	mpo := make([]*tensor.Dense, 0, n[0])
	mpo = append(mpo, w.Slice([][2]int{{r0 - 1, r0}, {0, r1}, {0, r2}, {0, r3}}))
	for _ = range n[0] - 2 {
		mpo = append(mpo, w)
	}
	mpo = append(mpo, w.Slice([][2]int{{0, r0}, {0, 1}, {0, r2}, {0, r3}}))
	return mpo
}
