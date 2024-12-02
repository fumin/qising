package tensor

import (
	"fmt"
	"testing"
)

func TestSVD(t *testing.T) {
	t.Parallel()
	type testcase struct {
		a *Tensor
		u *Tensor
		s *Tensor
		v *Tensor
	}
	tests := []testcase{
		{
			a: T().T2([][]complex64{
				{1, 0, -1, -1i},
				{-2, 1 + 1i, 4, 3},
			}),
			u: T().T2([][]complex64{
				{-0.198172 - 0.0990862i, 0.872197 + 0.436099i},
				{0.975146, 0.221563},
			}),
			s: T().T2([][]complex64{
				{5.703, 0, 0, 0},
				{0, 1.21484, 0, 0},
			}),
			v: T().T2([][]complex64{
				{-0.376725 - 0.0173744i, 0.353192 + 0.358977i, 0.342997i, 0.665133 - 0.210042i},
				{0.170988 - 0.170988i, 0.182381 - 0.182381i, -0.171499 + 0.857493i, -0.315063 + 0.105021i},
				{0.718702 + 0.0173744i, 0.0115709 - 0.359877i, 0, 0.595119},
				{0.530339 - 0.0347488i, 0.188167 + 0.717955i, 0.342997, -0.210042 - 0.070014i},
			}),
		},
	}

	nTests := len(tests)
	for i := range nTests {
		tt := testcase{
			a: T().Set(tests[i].a).Transpose([]int{1, 0}),
			u: T().Set(tests[i].v).Conj(),
			s: T().Set(tests[i].s).Transpose([]int{1, 0}),
			v: T().Set(tests[i].u).Conj()}
		tests = append(tests, tt)
	}

	svd := NewSVD()
	for _, test := range tests {
		t.Run(fmt.Sprintf("%#v", test.a), func(t *testing.T) {
			u, s, v := svd.Solve(T().Set(test.a))

			// Change to ground truth basis.
			phase := T().Set(u).Conj().Transpose([]int{1, 0}).Contract(test.u, 1, 0)
			if u.Shape[0] < v.Shape[0] {
				phase = T().Set(v).Conj().Transpose([]int{1, 0}).Contract(test.v, 1, 0)
			}
			u, v = matmulTrun(u, phase), matmulTrun(v, phase)

			// Check tensors are as expected.
			if err := equal(u, test.u, 1e-3); err != nil {
				t.Fatalf("%+v", err)
			}
			if err := equal(s, test.s, 1e-5); err != nil {
				t.Fatalf("%+v", err)
			}
			if err := equal(s, test.s, 1e-3); err != nil {
				t.Fatalf("%+v", err)
			}
		})
	}
}

func matmulTrun(a, b *Tensor) *Tensor {
	a, b = T().Set(a), T().Set(b)
	switch {
	case a.Shape[0] > b.Shape[0]:
		a = a.Slice([]int{0, 0}, []int{b.Shape[0], b.Shape[1]})
	case a.Shape[0] < b.Shape[0]:
		b = b.Slice([]int{0, 0}, []int{a.Shape[0], a.Shape[1]})
	}
	return a.Contract(b, 1, 0)
}
