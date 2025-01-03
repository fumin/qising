package tensor

import (
	"bytes"
	_ "embed"
	"fmt"
	"testing"
)

//go:embed testdata/west0479.mtx
var west0479Mtx []byte

func TestReadMatrixMarket(t *testing.T) {
	t.Parallel()

	type triplet struct {
		x int
		y int
		v complex64
	}
	tests := []struct {
		b    []byte
		data []triplet
	}{
		{
			b: west0479Mtx,
			data: []triplet{
				{x: 0, y: 82, v: 1.0},
				{x: 1, y: 17, v: 48.17647},
				{x: 2, y: 18, v: 83.5},
				{x: 3, y: 19, v: 171.9412},
				{x: 4, y: 20, v: 96.65138},
				{x: 5, y: 21, v: 168.2706},
				{x: 6, y: 22, v: 347.5872},
				{x: 7, y: 17, v: -1.0},
				{x: 7, y: 20, v: 2.5},
				{x: 7, y: 23, v: 1.0},
				{x: 8, y: 18, v: -1.0},
				{x: 8, y: 21, v: 2.5},
				{x: 8, y: 24, v: 1.0},
				{x: 9, y: 19, v: -1.0},
				{x: 9, y: 22, v: 2.5},
				{x: 9, y: 25, v: 1.0},
				{x: 10, y: 17, v: -3.347484e-05},
				{x: 10, y: 20, v: 3.347484e-05},
				{x: 10, y: 26, v: 1.010455},
				{x: 11, y: 18, v: -4.136539e-05},
				{x: 11, y: 21, v: 4.136539e-05},
			},
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			a, err := ReadMatrixMarket(bytes.NewBuffer(test.b))
			if err != nil {
				t.Fatalf("%+v", err)
			}
			nonzeros := make([]triplet, 0)
			for i := range a.Shape()[0] {
				for j := range a.Shape()[1] {
					if a.At(i, j) != 0 {
						nonzeros = append(nonzeros, triplet{x: i, y: j, v: a.At(i, j)})
					}
				}
			}
			for i := range test.data {
				if nonzeros[i] != test.data[i] {
					t.Fatalf("%d %#v %#v", i, nonzeros[i], test.data[i])
				}
			}
		})
	}
}
