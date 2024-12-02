package tensor

import (
	"fmt"
	"testing"
)

func TestUpperTriangle(t *testing.T) {
	t.Parallel()
	tests := []struct {
		a     *Dense
		upper float32
		lower float32
	}{
		{
			a: T2([][]complex64{
				{999, 1 + 2i, 5i, 3},
				{-2, 999, 3i, 4},
				{-1i, 1 - 3i, 999, 7},
				{1, 0, -3, 999},
			}),
			upper: 10.63014581273465,
			lower: 5,
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			if up := upperTriangle(test.a); up != test.upper {
				t.Fatalf("%f", up)
			}
			if low := lowerTriangle(test.a); low != test.lower {
				t.Fatalf("%f", low)
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
