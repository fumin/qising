package tensor

import (
	"fmt"
	"testing"
)

func TestGivens(t *testing.T) {
	t.Parallel()
	tests := []struct {
		a     *Dense
		i     int
		k     int
		left  [][]complex64
		right [][]complex64
	}{
		{
			a: T2([][]complex64{
				{3, 2, 1},
				{7, 8, 9},
				{4, 5, 6},
			}),
			i:     0,
			k:     2,
			left:  [][]complex64{{5, 5.2, 5.4}, {7, 8, 9}, {0, 1.4, 2.8}},
			right: [][]complex64{{2.6, 2, -1.8}, {11.4, 8, -0.2}, {7.2, 5, 0.4}},
		},
		{
			a: T2([][]complex64{{0, 1}, {0, 1}}),
			i: 0, k: 1,
			left:  [][]complex64{{0, 1}, {0, 1}},
			right: [][]complex64{{0, 1}, {0, 1}},
		},
		{
			a: T2([][]complex64{{0, 1}, {1e-17 - 1e-17i, 1}}),
			i: 0, k: 1,
			left:  [][]complex64{{0, 0.70710678 + 0.70710678i}, {0, -0.70710678 + 0.70710678i}},
			right: [][]complex64{{0.70710678 - 0.70710678i, 0}, {0.70710678 - 0.70710678i, 0}},
		},
		{
			a: T2([][]complex64{{1e-17 + 1e-17i, 1}, {1e-17 - 1e-17i, 1}}),
			i: 0, k: 1,
			left:  [][]complex64{{0, 0.70710678 + 0.70710678i}, {0, 0.70710678 + 0.70710678i}},
			right: [][]complex64{{-0.70710678i, 0.70710678}, {-0.70710678i, 0.70710678}},
		},
		{
			a: T2([][]complex64{{1 + 1e-17i, 1}, {1e-17 - 1e-17i, 1}}),
			i: 0, k: 1,
			left:  [][]complex64{{1, 1}, {0, 1}},
			right: [][]complex64{{1, 1}, {0, 1}},
		},
		{
			a: T2([][]complex64{{1 + 1i, 1}, {1e38 + 1e38i, 1}}),
			i: 0, k: 1,
			left:  [][]complex64{{1e38 + 1e38i, 1}, {0, -1}},
			right: [][]complex64{{1, -1 - 1i}, {2 + 1i, -1e38 - 1e38i}},
		},
		{
			a: T2([][]complex64{{1e-20 + 1e-20i, 1}, {1e20 + 1e20i, 1}}),
			i: 0, k: 1,
			left:  [][]complex64{{1e20 + 1e20i, 1}, {0, -1}},
			right: [][]complex64{{1, 0}, {1, -1e20 - 1e20i}},
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			g := newGivens(test.a.At(test.i, test.i), test.a.At(test.k, test.i), test.i, test.k)
			q := Zeros(1).Eye(test.a.Shape()[0], 0)
			g.applyRight(q)

			// Check q is unitary.
			qq := MatMul(Zeros(1), q, q.H())
			if err := equal2(qq, Zeros(1).Eye(qq.Shape()[0], 0).ToSlice2(), epsilon); err != nil {
				t.Fatalf("%+v", err)
			}

			// Check applyLeft.
			left := Zeros(test.a.Shape()...).Set([]int{0, 0}, test.a)
			g.applyLeft(left)
			if err := equal2(left, MatMul(Zeros(1), q.H(), test.a).ToSlice2(), epsilon); err != nil {
				t.Fatalf("%+v", err)
			}
			if err := equal2(left, test.left, epsilon); err != nil {
				t.Fatalf("%+v", err)
			}

			// Check applyRight.
			right := Zeros(test.a.Shape()...).Set([]int{0, 0}, test.a)
			g.applyRight(right)
			if err := equal2(right, MatMul(Zeros(1), test.a, q).ToSlice2(), epsilon); err != nil {
				t.Fatalf("%+v", err)
			}
			if err := equal2(right, test.right, 2*epsilon); err != nil {
				t.Fatalf("%+v", err)
			}
		})
	}
}

func TestChaseBulgeHessenberg(t *testing.T) {
	t.Parallel()
	tests := []struct {
		a *Dense
	}{
		{
			a: T2([][]complex64{
				{1i, 2, 3, 4},
				{5, 6i, 7, 8},
				{0, 9, 1i, 2},
				{0, 0, 3, 4i},
			}),
		},
		{
			a: T2([][]complex64{{1e-10 + 1e-10i, 1}, {1e10 - 1e10i, -1}}),
		},
		{
			a: T2([][]complex64{{1 + 1i, 1}, {1e20 - 1e20i, -1i}}),
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			m := test.a.Shape()[0]
			eye := Zeros(1).Eye(m, 0)

			a := Zeros(test.a.Shape()...).Set([]int{0, 0}, test.a)
			q, r := Zeros(1), Zeros(1)
			chaseBulgeHessenberg(a, q, r)

			// a remains hessenberg.
			alow := Zeros(a.Shape()...).Set([]int{0, 0}, a)
			alow.Tril(-2)
			if aln := alow.FrobeniusNorm(); aln > epsilon {
				t.Fatalf("%f %#v", aln, alow.ToSlice2())
			}

			// q is unitary.
			qq := Zeros(1)
			MatMul(qq, q, q.H())
			if err := equal2(qq, eye.ToSlice2(), 10*epsilon); err != nil {
				t.Fatalf("%+v", err)
			}

			// a @ q.H is triangular.
			aq := Zeros(1)
			MatMul(aq, a, q.H())
			aqlow := Zeros(aq.Shape()...).Set([]int{0, 0}, aq)
			aqlow.Tril(-1)
			if err := equal2(aqlow, Zeros(aqlow.Shape()...).ToSlice2(), 2*epsilon); err != nil {
				t.Fatalf("%+v %#v", err, aqlow.ToSlice2())
			}
			// r is triangular.
			rlow := Zeros(r.Shape()...).Set([]int{0, 0}, r)
			rlow.Tril(-1)
			if rlow.FrobeniusNorm() > epsilon {
				t.Fatalf("%#v", rlow.ToSlice2())
			}

			// a = qr.
			qr := MatMul(Zeros(1), q, r)
			if err := equal2(qr, test.a.ToSlice2(), epsilon*test.a.FrobeniusNorm()); err != nil {
				t.Fatalf("%+v", err)
			}
		})
	}
}
