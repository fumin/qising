package mat

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestDiskAdd(t *testing.T) {
	t.Parallel()
	tests := []struct {
		a          [][]complex64
		c          complex64
		b          [][]complex64
		z          *COO
		numNonZero int
	}{
		{
			a: [][]complex64{
				{1, 0},
				{0, 2i},
			},
			c: 1i,
			b: [][]complex64{
				{1i, 0},
				{2, -5},
			},
			z: M([][]complex64{
				{0, 0},
				{2i, -3i},
			}),
			numNonZero: 2,
		},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("%v", test.a), func(t *testing.T) {
			t.Parallel()
			dir, err := os.MkdirTemp("", "")
			if err != nil {
				t.Fatalf("%+v", err)
			}
			defer os.RemoveAll(dir)

			a := DiskM(filepath.Join(dir, "a.db"), test.a)
			b := DiskM(filepath.Join(dir, "b.db"), test.b)

			a.Add(test.c, b)
			if !a.COO().Equal(test.z) {
				t.Fatalf("%s, expected %s", a.COO(), test.z)
			}
			if a.NumNonZero() != test.numNonZero {
				t.Fatalf("%d, expected %d", a.NumNonZero(), test.numNonZero)
			}
		})
	}
}

func TestDiskKron(t *testing.T) {
	t.Parallel()
	tests := []struct {
		a [][]complex64
		b *COO
		c *COO
	}{
		{
			a: [][]complex64{
				{1, -4, 7},
				{-2, 0, 3},
			},
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
		{
			a: [][]complex64{{1}},
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
		t.Run(fmt.Sprintf("%v", test.a), func(t *testing.T) {
			t.Parallel()
			dir, err := os.MkdirTemp("", "")
			if err != nil {
				t.Fatalf("%+v", err)
			}
			defer os.RemoveAll(dir)

			a := DiskM(filepath.Join(dir, "a.db"), test.a)
			a.Kron(test.b)
			if !a.COO().Equal(test.c) {
				t.Fatalf("%s, expected %s", a.COO(), test.c)
			}
		})
	}
}
