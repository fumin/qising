package tensor

import (
	"fmt"
	"slices"
)

type householder struct {
	tau complex64
	v   *Dense
}

// newHouseholder creates a Householder matrix.
// This implementation is based on the clarfg LAPACK routine.
func newHouseholder(v, x *Dense, k int) householder {
	if !slices.Equal(v.Shape(), []int{x.Shape()[0], 1}) {
		panic(fmt.Sprintf("wrong shape %#v %#v", v.Shape(), []int{x.Shape()[0], 1}))
	}
	h := householder{v: v}
	for i := range k {
		h.v.SetAt([]int{i, 0}, 0)
	}
	for i := k; i < h.v.Shape()[0]; i++ {
		h.v.SetAt([]int{i, 0}, x.At(i, 0))
	}

	xnorm := h.v.Slice([][2]int{{k + 1, h.v.Shape()[0]}, {0, 1}}).FrobeniusNorm()
	alpha := h.v.At(k, 0)
	if xnorm == 0 && imag(alpha) == 0 {
		h.tau = 0
		return h
	}
	beta := -sign(lapy(alpha, complex(xnorm, 0)), real(alpha))

	// Scale and recalculate norms.
	var se float32 = safmin / epsilon
	knt := 0
	for absf(beta) < se {
		knt++
		for i := k; i < h.v.Shape()[0]; i++ {
			h.v.SetAt([]int{i, 0}, h.v.At(i, 0)/complex(se, 0))
		}
		beta /= se
	}
	xnorm = h.v.Slice([][2]int{{k + 1, h.v.Shape()[0]}, {0, 1}}).FrobeniusNorm()
	alpha = h.v.At(k, 0)
	beta = -sign(lapy(alpha, complex(xnorm, 0)), real(alpha))

	h.tau = complex((beta-real(alpha))/beta, -imag(alpha)/beta)
	alpha = 1 / (alpha - complex(beta, 0))
	for i := k + 1; i < h.v.Shape()[0]; i++ {
		h.v.SetAt([]int{i, 0}, h.v.At(i, 0)*alpha)
	}
	h.v.SetAt([]int{k, 0}, 1)

	return h
}

// applyLeft applies the Householder matrix to the left of matrix A.
// For more details, see Section 5.1.4, Matrix Computations, G. H. Golub and C. F. Van Loan.
func (h householder) applyLeft(a, buf *Dense) {
	w := Mul(buf, h.tau, MatMul(buf, a.H(), h.v))
	for i := range a.Shape()[0] {
		for j := range a.Shape()[1] {
			cross := h.v.At(i, 0) * conj(w.At(j, 0))
			a.SetAt([]int{i, j}, a.At(i, j)-cross)
		}
	}
}

func (h householder) applyRight(a, buf *Dense) {
	w := Mul(buf, h.tau, MatMul(buf, a, h.v))
	for i := range a.Shape()[0] {
		for j := range a.Shape()[1] {
			cross := w.At(i, 0) * conj(h.v.At(j, 0))
			a.SetAt([]int{i, j}, a.At(i, j)-cross)
		}
	}
}
