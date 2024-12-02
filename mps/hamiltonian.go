package mps

import (
	"qising/mat"
	"qising/mps/tensor"
)

func Ising(n [2]int, h complex64) []*tensor.Tensor {
	zero := [][]complex64{
		{0, 0},
		{0, 0},
	}
	identity := [][]complex64{
		{1, 0},
		{0, 1},
	}
	w := [][][][]complex64{
		{identity, zero, zero},
		{mat.PauliZ, zero, zero},
		{mul(-h, mat.PauliX), mat.PauliZ, identity},
	}

	mpo := make([]*tensor.Tensor, 0, n[0])
	mpo = append(mpo, tensor.T().T4([][][][]complex64{{w[2][0], w[2][1], w[2][2]}}))
	for _ = range n[0] - 2 {
		mpo = append(mpo, tensor.T().T4(w))
	}
	mpo = append(mpo, tensor.T().T4([][][][]complex64{{w[0][0]}, {w[1][0]}, {w[2][0]}}))
	return mpo
}

func mul(c complex64, m [][]complex64) [][]complex64 {
	for i := range len(m) {
		for j := range len(m[0]) {
			m[i][j] *= c
		}
	}
	return m
}
