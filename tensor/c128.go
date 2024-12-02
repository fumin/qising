package tensor

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/cblas128"
)

func T128(t *Dense) cblas128.General {
	g := cblas128.General{Rows: t.Shape()[0], Cols: t.Shape()[1]}
	g.Stride = g.Cols
	g.Data = make([]complex128, 0, g.Rows*g.Cols)
	for i := range g.Rows {
		for j := range g.Cols {
			g.Data = append(g.Data, complex128(t.At(i, j)))
		}
	}
	return g
}

func Gemm(ts ...*Dense) *Dense {
	t128 := T128(ts[0])
	for i := 1; i < len(ts); i++ {
		t1 := T128(Zeros(t128.Rows, ts[i].Shape()[1]))
		cblas128.Gemm(blas.NoTrans, blas.NoTrans, 1, t128, T128(ts[i]), 0, t1)
		t128 = t1
	}

	t := Zeros(t128.Rows, t128.Cols)
	for i := range t128.Rows {
		for j := range t128.Cols {
			t.SetAt([]int{i, j}, complex64(t128.Data[i*t128.Cols+j]))
		}
	}
	return t
}
