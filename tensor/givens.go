package tensor

type givens struct {
	i int
	k int
	c complex64
	s complex64
}

// newGivens creates a Givens rotation matrix that sets t[k, col] to zero.
// The implementation here is based on:
// https://github.com/JuliaLang/LinearAlgebra.jl/blob/master/src/givens.jl
func newGivens(t *Dense, i, k, col int) givens {
	gvn := givens{i: i, k: k}

	f, g := t.At(i, col), t.At(k, col)

	// Scaling.
	fs, gs := f, g
	scale := max(abs(f), abs(g))
	count := 0
	switch {
	case scale >= 1/safmn2:
		for scale >= 1/safmn2 {
			count++
			fs *= safmn2
			gs *= safmn2
			scale *= safmn2
		}
	case scale <= safmn2:
		if g == 0 {
			gvn.c = 1
			gvn.s = 0
			return gvn
		}
		for scale <= safmn2 {
			count--
			fs /= safmn2
			gs /= safmn2
			scale /= safmn2
		}
	}

	// Compute sin and cos.
	f2 := real(fs)*real(fs) + imag(fs)*imag(fs)
	g2 := real(gs)*real(gs) + imag(gs)*imag(gs)
	if f2 <= max(g2, 1)*safmin {
		if f == 0 {
			gvn.c = 0
			d := abs(gs)
			gvn.s = complex(real(gs)/d, -imag(gs)/d)
			return gvn
		}
		f2s := abs(fs)
		g2s := sqrtf(g2)
		gvn.c = complex(f2s/g2s, 0)
		var ff complex64
		if abs(f) > 1 {
			d := abs(f)
			ff = complex(real(f)/d, imag(f)/d)
		} else {
			var safmx2 float32 = 1 / safmn2
			dr := safmx2 * real(f)
			di := safmx2 * imag(f)
			d := sqrtf(dr*dr + di*di)
			ff = complex(dr/d, di/d)
		}
		gvn.s = ff * complex(real(gs)/g2s, -imag(gs)/g2s)
		return gvn
	} else {
		f2s := sqrtf(1 + g2/f2)
		gvn.c = complex(1/f2s, 0)
		r := complex(f2s*real(fs), f2s*imag(fs))
		d := f2 + g2
		gvn.s = complex(real(r)/d, imag(r)/d)
		gvn.s *= conj(gs)
		return gvn
	}
}

func (gvn givens) applyLeft(a *Dense) {
	for j := range a.Shape()[1] {
		f := a.At(gvn.i, j)
		g := a.At(gvn.k, j)
		a.SetAt([]int{gvn.i, j}, gvn.c*f+gvn.s*g)
		a.SetAt([]int{gvn.k, j}, -conj(gvn.s)*f+gvn.c*g)
	}
}

func (gvn givens) applyRight(a *Dense) {
	for j := range a.Shape()[0] {
		f := a.At(j, gvn.i)
		g := a.At(j, gvn.k)
		a.SetAt([]int{j, gvn.i}, conj(gvn.c)*f+conj(gvn.s)*g)
		a.SetAt([]int{j, gvn.k}, -gvn.s*f+conj(gvn.c)*g)
	}
}

func chaseBulgeHessenberg(a, q, r *Dense) {
	q.Eye(a.Shape()[0], 0)
	r.Zeros(a.Shape()...).Set([]int{0, 0}, a)
	gs := make([]givens, 0, r.Shape()[1]-1)
	for j := range r.Shape()[1] - 1 {
		g := newGivens(r, j, j+1, j)
		g.applyLeft(r)
		g.applyRight(q)
		gs = append(gs, g)
	}

	a.Set([]int{0, 0}, r)
	for _, g := range gs {
		g.applyRight(a)
	}
}
