package math

import (
	"math/cmplx"

	"github.com/pkg/errors"
)

func Make128(m, n int) [][]complex128 {
	x := make([][]complex128, m)
	for i := range m {
		x[i] = make([]complex128, n)
	}
	return x
}

func New128(src [][]complex128) [][]complex128 {
	dst := Make128(len(src), len(src[0]))
	return Copy128(dst, src)
}

func Copy128(dst, src [][]complex128) [][]complex128 {
	for i := range src {
		copy(dst[i], src[i])
	}
	return dst
}

func Equal128(a, b [][]complex128, tol float64) error {
	if len(a) != len(b) {
		return errors.Errorf("%d %d", len(a), len(b))
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return errors.Errorf("%d", i)
		}
		for j := range a[i] {
			if diff := cmplx.Abs(a[i][j] - b[i][j]); diff > tol {
				return errors.Errorf("%f %d %d", diff, i, j)
			}
		}
	}
	return nil
}

func Transpose128(a [][]complex128) [][]complex128 {
	m, n := len(a), len(a[0])
	switch {
	case m > n:
		for i := range len(a) {
			a[i] = append(a[i], make([]complex128, m-n)...)
		}
	case m < n:
		for _ = range n - m {
			a = append(a, make([]complex128, n))
		}
	}

	for i := range len(a) {
		for j := i + 1; j < len(a[i]); j++ {
			a[i][j], a[j][i] = a[j][i], a[i][j]
		}
	}

	switch {
	case m > n:
		a = a[:n]
	case m < n:
		for i := range len(a) {
			a[i] = a[i][:m]
		}
	}
	return a
}

func Conjugate128(m [][]complex128) [][]complex128 {
	for i := range len(m) {
		for j := range len(m[i]) {
			m[i][j] = complex(real(m[i][j]), -imag(m[i][j]))
		}
	}
	return m
}

func Adjoint128(m [][]complex128) [][]complex128 {
	m = Transpose128(m)
	m = Conjugate128(m)
	return m
}

func MatMul128(a, b [][]complex128) [][]complex128 {
	c := make([][]complex128, len(a))
	for i := range len(c) {
		c[i] = make([]complex128, len(b[0]))
		for j := range len(c[i]) {
			for k := range len(a[0]) {
				c[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return c
}
