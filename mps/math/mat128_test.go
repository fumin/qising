package math

import (
	"fmt"
	"testing"
)

func TestTranspose128(t *testing.T) {
	t.Parallel()
	tests := []struct {
		a [][]complex128
		b [][]complex128
	}{
		{
			a: [][]complex128{
				{0, 1, 2},
				{3i, 4i, 5i},
			},
			b: [][]complex128{
				{0, 3i},
				{1, 4i},
				{2, 5i},
			},
		},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("%#v", test.a), func(t *testing.T) {
			t.Parallel()
			a := New128(test.a)

			a = Transpose128(a)
			if err := Equal128(a, test.b, 0); err != nil {
				t.Fatalf("%+v", err)
			}

			a = Transpose128(a)
			if err := Equal128(a, test.a, 0); err != nil {
				t.Fatalf("%+v", err)
			}
		})
	}
}

func TestMatMul128(t *testing.T) {
	t.Parallel()
	tests := []struct {
		a [][]complex128
		b [][]complex128
		c [][]complex128
	}{
		{
			a: [][]complex128{
				{1, 2, 3},
				{3, 2, 1},
			},
			b: [][]complex128{
				{4, 5},
				{6, 5},
				{4, 6},
			},
			c: [][]complex128{
				{28, 33},
				{28, 31},
			},
		},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("%#v", test.a), func(t *testing.T) {
			t.Parallel()
			c := MatMul128(test.a, test.b)
			if err := Equal128(c, test.c, 0); err != nil {
				t.Fatalf("%+v", err)
			}
		})
	}
}
