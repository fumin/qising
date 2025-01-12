package mps_test

import (
	"fmt"
	"log"
	"math/cmplx"

	"github.com/fumin/qising/mps"
	"github.com/fumin/tensor"
)

func Example() {
	// Create an Ising chain of length n and transverse field strength h.
	const n = 4
	const h = 0.031623
	mpo := mps.Ising([2]int{n, 1}, h)

	// Buffers.
	fs := make([]*tensor.Dense, 0, len(mpo))
	for _ = range mpo {
		fs = append(fs, tensor.Zeros(1))
	}
	var bufs [10]*tensor.Dense
	for i := range len(bufs) {
		bufs[i] = tensor.Zeros(1)
	}

	// Search for the ground state.
	const bondDim = 2
	state := mps.RandMPS(mpo, bondDim)
	if err := mps.SearchGroundState(fs, mpo, state, bufs); err != nil {
		log.Fatalf("%+v", err)
	}
	// Compute expectation values of the ground state.
	bufs2 := [2]*tensor.Dense(bufs[:2])
	norm2 := mps.InnerProduct(state, state, bufs2)        // <state|state>
	e0 := mps.LExpressions(fs, mpo, state, bufs2) / norm2 // ground energy
	fmt.Printf("Ground energy %.4f\n", real(e0))

	// Output:
	// Ground energy -3.0015
}

func abs(x complex64) float64 {
	return cmplx.Abs(complex128(x))
}
