package tensor

import "math/cmplx"

type Diagonal struct {
	diag *Dense

	shape  [2]int
	digits [2]int
}

func (t *Diagonal) Shape() []int {
	return t.shape[:]
}

func (t *Diagonal) SetAt(digits []int, c complex64) {
	panic("not supported")
}

func (t *Diagonal) At(digits ...int) complex64 {
	if digits[0] != digits[1] {
		return 0
	}
	return t.diag.At(digits[0])
}

func (t *Diagonal) Digits() []int {
	return t.digits[:]
}

func (t *Diagonal) Data() []complex64 {
	return t.diag.Data()
}

func eig22(t Tensor) (complex64, complex64) {
	a, b := t.At(0, 0), t.At(0, 1)
	c, d := t.At(1, 0), t.At(1, 1)
	iSqrt := complex64(cmplx.Sqrt(complex128(a*a - 2*a*d + 4*b*c + d*d)))
	return 0.5 * (-iSqrt + a + d), 0.5 * (iSqrt + a + d)
}

func upperTriangle(a Tensor) float32 {
	var norm float32
	for i := range a.Shape()[0] - 1 {
		for j := i + 1; j < a.Shape()[0]; j++ {
			v := a.At(i, j)
			norm += real(v)*real(v) + imag(v)*imag(v)
		}
	}
	norm = sqrt(norm)
	return norm
}

func lowerTriangle(a Tensor) float32 {
	var norm float32
	for i := 1; i < a.Shape()[0]; i++ {
		for j := range i {
			v := a.At(i, j)
			norm += real(v)*real(v) + imag(v)*imag(v)
		}
	}
	norm = sqrt(norm)
	return norm
}
