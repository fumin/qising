package math

import (
	"fmt"
	"testing"
)

func TestSVD128(t *testing.T) {
	t.Parallel()
	type testcase struct {
		a [][]complex128
		u [][]complex128
		s [][]complex128
		v [][]complex128
	}
	tests := []testcase{
		{
			a: [][]complex128{
				{1, 0, -1, -1i},
				{-2, 1 + 1i, 4, 3},
			},
			u: [][]complex128{
				{-0.198172 - 0.0990862i, 0.872197 + 0.436099i},
				{0.975146, 0.221563},
			},
			s: [][]complex128{
				{5.703, 0, 0, 0},
				{0, 1.21484, 0, 0},
			},
			v: [][]complex128{
				{-0.376725 - 0.0173744i, 0.353192 + 0.358977i, 0.342997i, 0.665133 - 0.210042i},
				{0.170988 - 0.170988i, 0.182381 - 0.182381i, -0.171499 + 0.857493i, -0.315063 + 0.105021i},
				{0.718702 + 0.0173744i, 0.0115709 - 0.359877i, 0, 0.595119},
				{0.530339 - 0.0347488i, 0.188167 + 0.717955i, 0.342997, -0.210042 - 0.070014i},
			},
		},
	}

	nTests := len(tests)
	for i := range nTests {
		tt := testcase{a: Transpose128(New128(tests[i].a)), u: Conjugate128(New128(tests[i].v)), s: Transpose128(New128(tests[i].s)), v: Conjugate128(New128(tests[i].u))}
		tests = append(tests, tt)
	}

	for _, test := range tests {
		t.Run(fmt.Sprintf("%#v", test.a), func(t *testing.T) {
			t.Parallel()

			u, s, v := SVD128(New128(test.a))

			// Change to ground truth basis.
			phase := MatMul128(Adjoint128(New128(u)), test.u)
			if len(u) < len(v) {
				phase = MatMul128(Adjoint128(New128(v)), test.v)
			}
			u, v = matmulTrun128(u, phase), matmulTrun128(v, phase)

			if err := Equal128(u, test.u, 1e-3); err != nil {
				t.Fatalf("%+v", err)
			}
			if err := Equal128(s, test.s, 1e-5); err != nil {
				t.Fatalf("%+v", err)
			}
			if err := Equal128(s, test.s, 1e-3); err != nil {
				t.Fatalf("%+v", err)
			}
		})
	}
}

func matmulTrun128(a, b [][]complex128) [][]complex128 {
	aDim, bDim := len(a), len(b)
	switch {
	case aDim > bDim:
		aTrun := make([][]complex128, bDim)
		for i := range aTrun {
			aTrun[i] = make([]complex128, bDim)
			copy(aTrun[i], a[i])
		}
		a = aTrun
	case aDim < bDim:
		bTrun := make([][]complex128, aDim)
		for i := range bTrun {
			bTrun[i] = make([]complex128, aDim)
			copy(bTrun[i], b[i])
		}
		b = bTrun
	}
	return MatMul128(a, b)
}
