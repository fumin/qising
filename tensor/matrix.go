package tensor

import (
	"math/cmplx"
)

func (t *Dense) Eye(n, k int) *Dense {
	t.Zeros(n, n)
	for i := range n {
		j := i + k
		if !(j >= 0 && j < n) {
			continue
		}

		ptr := i*t.shape[1] + j
		t.data[ptr] = 1
	}
	return t
}

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

func (t *Dense) Triu(k int) *Dense {
	t.initDigits()
	for t.incrDigits() {
		d := t.Digits()
		i, j := d[len(d)-2], d[len(d)-1]
		if j < i+k {
			t.SetAt(d, 0)
		}
	}
	return t
}

func (t *Dense) Tril(k int) *Dense {
	t.initDigits()
	for t.incrDigits() {
		d := t.Digits()
		i, j := d[len(d)-2], d[len(d)-1]
		if j > i+k {
			t.SetAt(d, 0)
		}
	}
	return t
}
