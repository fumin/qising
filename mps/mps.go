// Why we need to normalize: Eq. 211, page 66
package mps

import (
	"math/rand/v2"

	"qising/tensor"
)

const (
	// mpsLeftAxis is the axis of a_{l-1} in Figure 6.
	mpsLeftAxis  = 0
	mpsUpAxis    = 1
	mpsRightAxis = 2
	// mpoLeftAxis is the axis of b_{l-1} in Figure 35.
	mpoLeftAxis  = 0
	mpoRightAxis = 1
	mpoUpAxis    = 2
	mpoDownAxis  = 3
)

type MPS struct {
	sites []*tensor.Dense
}

// NewMPS creates a matrix product state.
// d is D below equation 71 in section 4.1.4.
// Consult Figure 37 for details.
func NewMPS(mpo []*tensor.Dense, d int) *MPS {
	sites := make([]*tensor.Dense, 0, len(mpo))
	sites = append(sites, randTensor([]int{1, mpo[0].Shape()[mpoDownAxis], d}))
	for i := 1; i <= len(mpo)-2; i++ {
		sites = append(sites, randTensor([]int{d, mpo[i].Shape()[mpoDownAxis], d}))
	}
	sites = append(sites, randTensor([]int{d, mpo[len(mpo)-1].Shape()[mpoDownAxis], 1}))
	return mps
}

// RightNormalize normalizes a MPS site from the right.
// Consult Section 4.4.2 for details.
func RightNormalize(ms []*tensor.Dense) {
	for i := len(ms) - 1; i >= 1; i-- {
		s := ms[i].Shape()
		ms[i].Reshape(s[mpsLeftAxis], s[mpsUpAxis]*s[mpsRightAxis])
	}

	//m := make([][]complex64, d, b.Shape[mpsUpAxis]*d)

	// Cut! Equation 58, Section 4.1.3.
}

func randTensor(shape []int) *tensor.Dense {
	t := tensor.Zeros(shape...)
	for ijk := range t.All() {
		v := complex(rand.Float32()*2-1, rand.Float32()*2-1)
		t.SetAt(ijk, v)
	}
	return t
}
