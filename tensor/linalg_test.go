package tensor

import (
	"fmt"
	"testing"
)

func TestEig(t *testing.T) {
	t.Parallel()
	tests := []struct {
		a      *Dense
		lambda *Dense
		v      *Dense
	}{
		//{
		//	a:      T2([][]complex64{{1, 0}, {1, 3}}),
		//	lambda: []complex64{1, 3},
		//	v:      [][]complex64{{-2, 0}, {1, 1}},
		//},
		{
			a:      T2([][]complex64{{2, -1, -1, 0}, {-1, 3, -1, -1}, {-1, -1, 3, -1}, {0, -1, -1, 2}}),
			lambda: T1([]complex64{0, 2, 4, 4}),
			v:      T2([][]complex64{{0.5, 0.707106781, 0.5, -0.307596791}, {0.5, 0, 0.5, -0.249869288}, {0.5, 0, 0.5, 0.865062869}, {0.5, -0.707106781, 0.5, -0.307596791}}),
		},
	}
	eig := NewEig()
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			return
			eigvals, eigvecs := eig.Solve(test.a)
			t.Logf("eigvals %#v", eigvals.ToSlice2())
			t.Logf("eigvecs %#v", eigvecs.ToSlice2())
		})
	}
}

func TestHessenberg(t *testing.T) {
	t.Parallel()
	tests := []struct {
		a *Dense
		h [][]complex64
		q [][]complex64
	}{
		{
			a: T2([][]complex64{{2, -1, -1, 0}, {-1, 3, -1, -1}, {-1, -1, 3, -1}, {0, -1, -1, 2}}),
			h: [][]complex64{
				{2, 1.41421356, 0, 0},
				{1.41421356, 2, 1.41421356, 0},
				{0, 1.41421356, 2, 0},
				{0, 0, 0, 4},
			},
			q: [][]complex64{
				{1, 0, 0, 0},
				{0, -0.707106781, 0, -0.707106781},
				{0, -0.707106781, 0, 0.707106781},
				{0, 0, 1, 0},
			},
		},
	}
	hess := NewHessenberg()
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			h, q := hess.Solve(test.a)

			// Match ground truth basis.
			phase, buf := Zeros(1), Zeros(1)
			MatMul(phase, q.H(), T2(test.q))
			q.Copy(MatMul(buf, q, phase))
			MatMul(h, phase.H(), MatMul(buf, h, phase))

			// Check q @ h @ q.H == a
			a := Zeros(1)
			MatMul(a, q, MatMul(buf, h, q.H()))
			if err := equal2(a, test.a.ToSlice2(), 1e-5); err != nil {
				t.Fatalf("%+v", err)
			}

			// Check results.
			if err := equal2(h, test.h, 1e-5); err != nil {
				t.Fatalf("%+v", err)
			}
			if err := equal2(q, test.q, 1e-5); err != nil {
				t.Fatalf("%+v", err)
			}
		})
	}
}

func TestQR(t *testing.T) {
	t.Parallel()
	tests := []struct {
		a *Dense
		q [][]complex64
		r [][]complex64
	}{
		{
			a: T2([][]complex64{{12, -51, 4}, {6, 167, -68}, {-4, 24, -41}}),
			q: [][]complex64{{0.857143, -0.394286, -0.331429}, {0.428571, 0.902857, 0.0342857}, {-0.285714, 0.171429, -0.942857}},
			r: [][]complex64{{14, 21, -14}, {0, 175, -70}, {0, 0, 35}},
		},
		{
			a: T2([][]complex64{{99, 1, 3, 5}, {99, 1 + 2i, 4, 6}}).Conj().Transpose(1, 0).Slice([][2]int{{1, 4}, {0, 2}}),
			q: [][]complex64{{0.169031, -0.113478 - 0.964563i}, {0.507093, 0.156032 + 0.0851085i}, {0.845154, -0.0709238 + 0.141848i}},
			r: [][]complex64{{5.91608, 7.26833 - 0.338062i}, {0, 2.01424}, {0, 0}},
		},
		{
			a: T2([][]complex64{{1, 1 - 2i}, {3, 4}, {5, 6}, {1 + 3i, 4 + 1i}}),
			q: [][]complex64{{0.149071, -0.028991 - 0.446461i}, {0.447214, 0.173946 + 0.22613i}, {0.745356, 0.115964 + 0.376883i}, {0.149071 + 0.447214i, 0.527636 - 0.533434i}},
			r: [][]complex64{{6.7082, 7.45356 - 1.93793i}, {0, 3.83261}, {0, 0}, {0, 0}},
		},
	}
	qr := NewQR()
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			q, r := qr.Solve(test.a)

			// Check that a = q @ r.
			a := Zeros(1)
			MatMul(a, q, r)
			if err := equal2(a, test.a.ToSlice2(), 1e-4); err != nil {
				t.Fatalf("%+v %#v", err, a.ToSlice2())
			}

			// Check unitary.
			qqt := MatMul(Zeros(1), q, q.H())
			identity := Zeros(1).Eye(qqt.Shape()[0], 0).ToSlice2()
			if err := equal2(qqt, identity, 1e-6); err != nil {
				t.Fatalf("%#v", qqt.ToSlice2())
			}
			qReduced := q.Slice([][2]int{{0, q.Shape()[0]}, {0, r.Shape()[1]}})

			if err := equal2(qReduced, test.q, 1e-5); err != nil {
				t.Fatalf("%+v", err)
			}
			if err := equal2(r, test.r, 1e-5); err != nil {
				t.Fatalf("%+v", err)
			}
		})
	}
}

func TestSVD(t *testing.T) {
	t.Parallel()
	tests := []struct {
		a *Dense
		u [][]complex64
		s [][]complex64
		v [][]complex64
	}{
		{
			a: T2([][]complex64{
				{1, 0, -1, -1i},
				{-2, 1 + 1i, 4, 3},
			}),
			u: [][]complex64{
				{-0.198172 - 0.0990862i, 0.872197 + 0.436099i},
				{0.975146, 0.221563},
			},
			s: [][]complex64{
				{5.703, 0, 0, 0},
				{0, 1.21484, 0, 0},
			},
			v: [][]complex64{
				{-0.376725 - 0.0173744i, 0.353192 + 0.358977i, 0.342997i, 0.665133 - 0.210042i},
				{0.170988 - 0.170988i, 0.182381 - 0.182381i, -0.171499 + 0.857493i, -0.315063 + 0.105021i},
				{0.718702 + 0.0173744i, 0.0115709 - 0.359877i, 0, 0.595119},
				{0.530339 - 0.0347488i, 0.188167 + 0.717955i, 0.342997, -0.210042 - 0.070014i},
			},
		},
		{
			a: T2([][]complex64{{1, -2}, {0, 1 + 1i}, {-1, 4}, {-1i, 3}}),
			u: [][]complex64{
				{-0.376725 + 0.0173744i, 0.353192 - 0.358977i, -0.342997i, 0.665133 + 0.210042i},
				{0.170988 + 0.170988i, 0.182381 + 0.182381i, -0.171499 - 0.857493i, -0.315063 - 0.105021i},
				{0.718702 - 0.0173744i, 0.0115709 + 0.359877i, 0, 0.595119},
				{0.530339 + 0.0347488i, 0.188167 - 0.717955i, 0.342997, -0.210042 + 0.070014i},
			},
			s: [][]complex64{
				{5.703, 0},
				{0, 1.21484},
				{0, 0},
				{0, 0},
			},
			v: [][]complex64{
				{-0.198172 + 0.0990862i, 0.872197 - 0.436099i},
				{0.975146, 0.221563},
			},
		},
	}
	svd := NewSVD()
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			u, s, v := svd.Solve(test.a)

			// Change to ground truth basis.
			phase := Zeros(1)
			minD := min(s.shape[0], s.shape[1])
			trun := [][2]int{{0, minD}, {0, minD}}
			buf := Zeros(1)
			if u.Shape()[0] < v.Shape()[0] {
				MatMul(phase, v.H(), T2(test.v))
				u.Copy(MatMul(buf, u, phase.Slice(trun)))
				MatMul(s, phase.Slice(trun), MatMul(buf, s, phase.H()))
				v.Copy(MatMul(buf, v, phase))
			} else {
				MatMul(phase, u.H(), T2(test.u))
				u.Copy(MatMul(buf, u, phase))
				MatMul(s, phase.H(), MatMul(buf, s, phase.Slice(trun)))
				v.Copy(MatMul(buf, v, phase.Slice(trun)))
			}

			if err := equal2(u, test.u, 1e-3); err != nil {
				t.Fatalf("%+v", err)
			}
			if err := equal2(s, test.s, 5e-3); err != nil {
				t.Fatalf("%+v", err)
			}
			if err := equal2(v, test.v, 1e-3); err != nil {
				t.Fatalf("%+v", err)
			}
		})
	}
}
