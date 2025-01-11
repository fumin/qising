package mat

import (
	"fmt"
	"testing"
)

func TestSlice(t *testing.T) {
	t.Parallel()
	tests := []struct {
		m *COO
		y [2]int
		x [2]int
		s *COO
	}{
		{
			m: M([][]complex64{
				{0, 1, 2, 3, 4},
				{5, 6, 7, 8, 9},
				{10, 11, 12, 13, 14},
				{15, 16, 17, 18, 19},
				{20, 21, 22, 23, 24},
				{25, 26, 27, 28, 29},
			}),
			y: [2]int{-5, -2},
			x: [2]int{1, 3},
			s: M([][]complex64{
				{6, 7},
				{11, 12},
				{16, 17},
			}),
		},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("%s", test.m), func(t *testing.T) {
			t.Parallel()
			s := test.m.Slice(test.y, test.x)
			if !s.Equal(test.s) {
				t.Fatalf("%s, expected %s", s, test.s)
			}
		})
	}
}

func TestAdd(t *testing.T) {
	t.Parallel()
	tests := []struct {
		a          *COO
		c          complex64
		b          *COO
		z          *COO
		numNonZero int
	}{
		{
			a: M([][]complex64{
				{1, 0},
				{0, 2i},
			}),
			c: 1i,
			b: M([][]complex64{
				{1i, 0},
				{2, -5},
			}),
			z: M([][]complex64{
				{0, 0},
				{2i, -3i},
			}),
			numNonZero: 2,
		},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("%s", test.a), func(t *testing.T) {
			t.Parallel()
			test.a.Add(test.c, test.b)
			if !test.a.Equal(test.z) {
				t.Fatalf("%s, expected %s", test.a, test.z)
			}
			if len(test.a.Data) != test.numNonZero {
				t.Fatalf("%d, expected %d", len(test.a.Data), test.numNonZero)
			}
		})
	}
}

func TestMul(t *testing.T) {
	t.Parallel()
	tests := []struct {
		a *COO
		b *COO
		c *COO
	}{
		{
			a: M([][]complex64{
				{0, 0},
				{-1, 2},
			}),
			b: M([][]complex64{
				{0, 1},
				{0, 2},
			}),
			c: M([][]complex64{
				{0, 0},
				{0, 4},
			}),
		},
		// Multiply scalar using broadcast.
		{
			a: M([][]complex64{
				{0, 3},
				{-1, 2},
			}),
			b: M([][]complex64{{-2}}),
			c: M([][]complex64{
				{0, -6},
				{2, -4},
			}),
		},
		// Multiply vector using broadcast.
		{
			a: M([][]complex64{
				{0, 3},
				{-1, 2},
			}),
			b: M([][]complex64{{3}, {-2}}),
			c: M([][]complex64{
				{0, 9},
				{2, -4},
			}),
		},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("%s", test.a), func(t *testing.T) {
			t.Parallel()
			test.a.Mul(test.b)
			if !test.a.Equal(test.c) {
				t.Fatalf("%s, expected %s", test.a, test.c)
			}
		})
	}
}

func TestKron(t *testing.T) {
	t.Parallel()
	tests := []struct {
		a *COO
		b *COO
		c *COO
	}{
		{
			a: M([][]complex64{
				{1, -4, 7},
				{-2, 0, 3},
			}),
			b: M([][]complex64{
				{8, -9, -6, 5},
				{1, -3, 0, 7},
				{2, 8, -8, -3},
				{1, 2, -5, -1},
			}),
			c: M([][]complex64{
				{8, -9, -6, 5, -32, 36, 24, -20, 56, -63, -42, 35},
				{1, -3, 0, 7, -4, 12, 0, -28, 7, -21, 0, 49},
				{2, 8, -8, -3, -8, -32, 32, 12, 14, 56, -56, -21},
				{1, 2, -5, -1, -4, -8, 20, 4, 7, 14, -35, -7},
				{-16, 18, 12, -10, 0, 0, 0, 0, 24, -27, -18, 15},
				{-2, 6, 0, -14, 0, 0, 0, 0, 3, -9, 0, 21},
				{-4, -16, 16, 6, 0, 0, 0, 0, 6, 24, -24, -9},
				{-2, -4, 10, 2, 0, 0, 0, 0, 3, 6, -15, -3},
			}),
		},
		// Scalar kronecker.
		{
			a: M([][]complex64{{1}}),
			b: M([][]complex64{
				{1, 2},
				{3, 4},
			}),
			c: M([][]complex64{
				{1, 2},
				{3, 4},
			}),
		},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("%s", test.a), func(t *testing.T) {
			t.Parallel()
			test.a.Kron(test.b)
			if !test.a.Equal(test.c) {
				t.Fatalf("%s, expected %s", test.a, test.c)
			}
		})
	}
}
