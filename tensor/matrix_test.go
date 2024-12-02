package tensor

import (
	"fmt"
	"testing"
)

func TestEye(t *testing.T) {
	t.Parallel()
	tests := []struct {
		n int
		k int
		b *Dense
	}{
		{n: 4, k: 0, b: T2([][]complex64{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}})},
		{n: 4, k: 2, b: T2([][]complex64{{0, 0, 1, 0}, {0, 0, 0, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}})},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			b := Zeros(1).Eye(test.n, test.k)
			if !b.Equal(test.b) {
				t.Fatalf("%#v %#v", b, test.b)
			}
		})
	}
}

func TestInfNorm(t *testing.T) {
	t.Parallel()
	tests := []struct {
		a    *Dense
		norm float32
	}{
		{
			a:    T2([][]complex64{{-3, 3 + 4i, 7}, {2, 6, 4}, {0, 2, 8}}),
			norm: 15,
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			if test.a.InfNorm() != test.norm {
				t.Fatalf("%f %f", test.a.InfNorm(), test.norm)
			}
		})
	}
}

func TestTriu(t *testing.T) {
	t.Parallel()
	tests := []struct {
		a *Dense
		k int
		b [][]complex64
	}{
		{
			a: T2([][]complex64{
				{999, 1 + 2i, 5i, 3},
				{-2, 999, 3i, 4},
				{-1i, 1 - 3i, 999, 7},
				{1, 0, -3, 999},
				{99, 99, 99, 99},
			}),
			k: -3,
			b: [][]complex64{
				{999, 1 + 2i, 5i, 3},
				{-2, 999, 3i, 4},
				{-1i, 1 - 3i, 999, 7},
				{1, 0, -3, 999},
				{0, 99, 99, 99},
			},
		},
		{
			a: T2([][]complex64{
				{999, 1 + 2i, 5i, 3},
				{-2, 999, 3i, 4},
				{-1i, 1 - 3i, 999, 7},
				{1, 0, -3, 999},
				{99, 99, 99, 99},
			}),
			k: -2,
			b: [][]complex64{
				{999, 1 + 2i, 5i, 3},
				{-2, 999, 3i, 4},
				{-1i, 1 - 3i, 999, 7},
				{0, 0, -3, 999},
				{0, 0, 99, 99},
			},
		},
		{
			a: T2([][]complex64{
				{999, 1 + 2i, 5i, 3},
				{-2, 999, 3i, 4},
				{-1i, 1 - 3i, 999, 7},
				{1, 0, -3, 999},
				{99, 99, 99, 99},
			}),
			k: -1,
			b: [][]complex64{
				{999, 1 + 2i, 5i, 3},
				{-2, 999, 3i, 4},
				{0, 1 - 3i, 999, 7},
				{0, 0, -3, 999},
				{0, 0, 0, 99},
			},
		},
		{
			a: T2([][]complex64{
				{999, 1 + 2i, 5i, 3},
				{-2, 999, 3i, 4},
				{-1i, 1 - 3i, 999, 7},
				{1, 0, -3, 999},
				{99, 99, 99, 99},
			}),
			k: 0,
			b: [][]complex64{
				{999, 1 + 2i, 5i, 3},
				{0, 999, 3i, 4},
				{0, 0, 999, 7},
				{0, 0, 0, 999},
				{0, 0, 0, 0},
			},
		},
		{
			a: T2([][]complex64{
				{999, 1 + 2i, 5i, 3},
				{-2, 999, 3i, 4},
				{-1i, 1 - 3i, 999, 7},
				{1, 0, -3, 999},
				{99, 99, 99, 99},
			}),
			k: 1,
			b: [][]complex64{
				{0, 1 + 2i, 5i, 3},
				{0, 0, 3i, 4},
				{0, 0, 0, 7},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
			},
		},
		{
			a: T2([][]complex64{
				{999, 1 + 2i, 5i, 3},
				{-2, 999, 3i, 4},
				{-1i, 1 - 3i, 999, 7},
				{1, 0, -3, 999},
				{99, 99, 99, 99},
			}),
			k: 2,
			b: [][]complex64{
				{0, 0, 5i, 3},
				{0, 0, 0, 4},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
			},
		},
		{
			a: T2([][]complex64{
				{999, 1 + 2i, 5i, 3},
				{-2, 999, 3i, 4},
				{-1i, 1 - 3i, 999, 7},
				{1, 0, -3, 999},
				{99, 99, 99, 99},
			}),
			k: 3,
			b: [][]complex64{
				{0, 0, 0, 3},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
			},
		},
		{
			a: T2([][]complex64{
				{999, 1 + 2i, 5i, 3, 99},
				{-2, 999, 3i, 4, 99},
				{-1i, 1 - 3i, 999, 7, 99},
				{1, 0, -3, 999, 99},
			}),
			k: 3,
			b: [][]complex64{
				{0, 0, 0, 3, 99},
				{0, 0, 0, 0, 99},
				{0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0},
			},
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			if err := equal2(test.a.Triu(test.k), test.b, 0); err != nil {
				t.Fatalf("%+v", err)
			}
		})
	}
}

func TestTril(t *testing.T) {
	t.Parallel()
	tests := []struct {
		a *Dense
		k int
		b [][]complex64
	}{
		{
			a: T2([][]complex64{
				{999, 1 + 2i, 5i, 3},
				{-2, 999, 3i, 4},
				{-1i, 1 - 3i, 999, 7},
				{1, 0, -3, 999},
				{99, 99, 99, 99},
			}),
			k: 2,
			b: [][]complex64{
				{999, 1 + 2i, 5i, 0},
				{-2, 999, 3i, 4},
				{-1i, 1 - 3i, 999, 7},
				{1, 0, -3, 999},
				{99, 99, 99, 99},
			},
		},
		{
			a: T2([][]complex64{
				{999, 1 + 2i, 5i, 3},
				{-2, 999, 3i, 4},
				{-1i, 1 - 3i, 999, 7},
				{1, 0, -3, 999},
				{99, 99, 99, 99},
			}),
			k: 1,
			b: [][]complex64{
				{999, 1 + 2i, 0, 0},
				{-2, 999, 3i, 0},
				{-1i, 1 - 3i, 999, 7},
				{1, 0, -3, 999},
				{99, 99, 99, 99},
			},
		},
		{
			a: T2([][]complex64{
				{999, 1 + 2i, 5i, 3},
				{-2, 999, 3i, 4},
				{-1i, 1 - 3i, 999, 7},
				{1, 0, -3, 999},
				{99, 99, 99, 99},
			}),
			k: 0,
			b: [][]complex64{
				{999, 0, 0, 0},
				{-2, 999, 0, 0},
				{-1i, 1 - 3i, 999, 0},
				{1, 0, -3, 999},
				{99, 99, 99, 99},
			},
		},
		{
			a: T2([][]complex64{
				{999, 1 + 2i, 5i, 3},
				{-2, 999, 3i, 4},
				{-1i, 1 - 3i, 999, 7},
				{1, 0, -3, 999},
				{99, 99, 99, 99},
			}),
			k: -3,
			b: [][]complex64{
				{0, 0, 0, 0},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
				{1, 0, 0, 0},
				{99, 99, 0, 0},
			},
		},
		{
			a: T2([][]complex64{
				{999, 1 + 2i, 5i, 3, 99},
				{-2, 999, 3i, 4, 99},
				{-1i, 1 - 3i, 999, 7, 99},
				{1, 0, -3, 999, 99},
			}),
			k: 2,
			b: [][]complex64{
				{999, 1 + 2i, 5i, 0, 0},
				{-2, 999, 3i, 4, 0},
				{-1i, 1 - 3i, 999, 7, 99},
				{1, 0, -3, 999, 99},
			},
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			if err := equal2(test.a.Tril(test.k), test.b, 0); err != nil {
				t.Fatalf("%+v", err)
			}
		})
	}
}

func TestEig22(t *testing.T) {
	t.Parallel()
	tests := []struct {
		a       *Dense
		lambda0 complex64
		lambda1 complex64
	}{
		{
			a:       T2([][]complex64{{2 + 1i, -3 + 2i}, {1 - 3i, -1i}}),
			lambda0: -1.85847 - 2.27395i,
			lambda1: 3.85847 + 2.27395i,
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			lambda0, lambda1 := eig22(test.a)
			if abs(lambda0-test.lambda0) > 1e-5 {
				t.Fatalf("%v %v", lambda0, test.lambda0)
			}
			if abs(lambda1-test.lambda1) > 1e-5 {
				t.Fatalf("%v %v", lambda1, test.lambda1)
			}
		})
	}
}
