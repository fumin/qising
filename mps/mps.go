// Why we need to normalize: Eq. 211, page 66
package mps

import "qising/mps/tensor"

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

// NewMPS creates a matrix product state.
// d is D below equation 71 in section 4.1.4.
// Consult Figure 37 for details.
func NewMPS(mpo []*tensor.Tensor, d int) []*tensor.Tensor {
	mps := make([]*tensor.Tensor, 0, len(mpo))
	mps = append(mps, tensor.T().Rand([]int{1, d, mpo[0].Shape[mpoDownAxis]}))
	for i := 1; i <= len(mpo)-2; i++ {
		mps = append(mps, tensor.T().Rand([]int{d, d, mpo[i].Shape[mpoDownAxis]}))
	}
	mps = append(mps, tensor.T().Rand([]int{d, 1, mpo[len(mpo)-1].Shape[mpoDownAxis]}))
	return mps
}

// RightNormalize normalizes a MPS site from the right.
// Consult Section 4.4.2 for details.
func RightNormalize(b1, b []*tensor.Tensor) {
	//d := b.Shape[mpsLeftAxis]

	//m := make([][]complex64, d, b.Shape[mpsUpAxis]*d)

	// Cut! Equation 58, Section 4.1.3.
}
