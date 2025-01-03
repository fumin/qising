package tensor

import (
	"fmt"
	"testing"
)

func TestHouseholder(t *testing.T) {
	t.Parallel()
	tests := []struct {
		a *Dense
		i int
		j int
		b [][]complex64
	}{
		{
			a: T2([][]complex64{
				{4, 1, -2, 2},
				{1, 2, 0, 1},
				{-2, 0, 3, -2},
				{2, 1, -2, -1},
			}),
			i: 1, j: 0,
			b: [][]complex64{
				{4, -3, 0, 0},
				{-3, 10. / 3, 1, 4. / 3},
				{0, 1, 5. / 3, -4. / 3},
				{0, 4. / 3, -4. / 3, -1},
			},
		},
		{
			a: T2([][]complex64{
				{1, 0, 0, 0},
				{0, 1 + 2i, 0, 0},
				{0, -2 - 3i, 0, 0},
				{0, 2 + 5i, 0, 0},
			}),
			i: 1, j: 1,
			b: [][]complex64{
				{1, 0, 0, 0},
				{0, 1 + 2i, -3.19108653 + 1.67838219i, 4.14747041 - 3.4348929i},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
			},
		},
		{
			a: T2([][]complex64{
				{1e-33, 1, 0, 0},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
			}),
			i: 0, j: 0,
			b: [][]complex64{
				{0, 1, 0, 0},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
			},
		},
		{
			a: T2([][]complex64{
				{1e-33i, 1, 0, 0},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
			}),
			i: 0, j: 0,
			b: [][]complex64{
				{0, 1i, 0, 0},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
			},
		},
		{
			a: T2([][]complex64{
				{1 + 1e-33i, 1, 0, 0},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
			}),
			i: 0, j: 0,
			b: [][]complex64{
				{1, -1, 0, 0},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
			},
		},
		{
			a: T2([][]complex64{
				{0, 1, 0, 0},
				{1 + 1i, 0, 0, 0},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
			}),
			i: 0, j: 0,
			b: [][]complex64{
				{0, 1 - 1i, 0, 0},
				{1i, 0, 0, 0},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
			},
		},
		{
			a: T2([][]complex64{
				{1e-8 - 1e-8i, 1, 0, 0},
				{1 + 1i, 0, 0, 0},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
			}),
			i: 0, j: 0,
			b: [][]complex64{
				{0, 1 - 1i, 0, 0},
				{1i, 0, 0, 0},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
			},
		},
		{
			a: T2([][]complex64{
				{1e-33i, 1, 0, 0},
				{1e-34 + 1e-34i, 0, 0, 0},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
			}),
			i: 0, j: 0,
			b: [][]complex64{
				{0.09803921 - 0.0980392012i, 0.00970685 + 0.980344050i, 0, 0},
				{0.01960688 + 1.94136630e-04i, -0.09803921 + 0.0980392099i, 0, 0},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
			},
		},
		{
			a: T2([][]complex64{
				{1 + 1e-33i, 1, 0, 0},
				{1e-8 - 1e-8i, 0, 0, 0},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
			}),
			i: 0, j: 0,
			b: [][]complex64{
				{1, -1, 0, 0},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
			},
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			a := Zeros(test.a.Shape()...).Set([]int{0, 0}, test.a)
			x := a.Slice([][2]int{{0, a.Shape()[0]}, {test.j, test.j + 1}})
			v, buf := Zeros(x.Shape()[0], 1), Zeros(1)
			h := newHouseholder(v, x, test.i)
			h.applyLeft(a, buf)
			q := Zeros(1).Eye(a.Shape()[0], 0)
			h.applyRight(q, buf)

			// Check multiplication.
			qa := MatMul(Zeros(1), q.H(), test.a)
			if err := equal2(a, qa.ToSlice2(), 5*epsilon); err != nil {
				t.Fatalf("%+v", err)
			}

			// Check q is unitary.
			qq := MatMul(Zeros(1), q, q.H())
			eye := Zeros(1).Eye(qq.Shape()[0], 0).ToSlice2()
			if err := equal2(qq, eye, 2*epsilon); err != nil {
				t.Fatalf("%+v", err)
			}

			// Check a[i+1:, j] is zero.
			aj := a.Slice([][2]int{{test.i + 1, a.Shape()[0]}, {test.j, test.j + 1}})
			if err := equal2(aj, Zeros(aj.Shape()...).ToSlice2(), 5*epsilon); err != nil {
				t.Fatalf("%+v", err)
			}
			// Check a[:i] remains the same.
			ax := [][2]int{{0, test.i}, {0, a.Shape()[1]}}
			if test.i > 0 {
				if err := equal2(a.Slice(ax), test.a.Slice(ax).ToSlice2(), 0); err != nil {
					t.Fatalf("%+v", err)
				}
			}

			// Perform a @ h.
			h.applyRight(a, buf)

			// Check multiplication.
			qaq := MatMul(Zeros(1), MatMul(Zeros(1), q.H(), test.a), q)
			if err := equal2(a, qaq.ToSlice2(), 5*epsilon); err != nil {
				t.Fatalf("%+v", err)
			}

			// Check q.H @ a @ q is as expected.
			if err := equal2(a, test.b, 10*epsilon); err != nil {
				t.Fatalf("%+v", err)
			}
		})
	}
}
