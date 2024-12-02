package tensor

import (
	"math/cmplx"
)

func (t *Dense) Eye(n, k int) *Dense {
	t.Zeros(n, n)
	for i := range n {
		j := i + k
		if !(j >= 0 && j < n) {
			continue
		}

		ptr := i*t.shape[1] + j
		t.data[ptr] = 1
	}
	return t
}

func (a *Dense) InfNorm() float32 {
	var norm float32 = -1
	for i := 0; i < a.Shape()[0]; i++ {
		var ni float32
		for j := 0; j < a.Shape()[1]; j++ {
			ni += abs(a.At(i, j))
		}
		norm = max(ni, norm)
	}
	return norm
}

type Diagonal struct {
	diag *Dense

	shape  [2]int
	digits [2]int
}

func (t *Diagonal) Shape() []int {
	return t.shape[:]
}

func (t *Diagonal) SetAt(digits []int, c complex64) {
	panic("not supported")
}

func (t *Diagonal) At(digits ...int) complex64 {
	if digits[0] != digits[1] {
		return 0
	}
	return t.diag.At(digits[0])
}

func (t *Diagonal) Digits() []int {
	return t.digits[:]
}

func (t *Diagonal) Data() []complex64 {
	return t.diag.Data()
}

func eig22(t Tensor) (complex64, complex64) {
	a, b := t.At(0, 0), t.At(0, 1)
	c, d := t.At(1, 0), t.At(1, 1)
	iSqrt := complex64(cmplx.Sqrt(complex128(a*a - 2*a*d + 4*b*c + d*d)))
	return 0.5 * (-iSqrt + a + d), 0.5 * (iSqrt + a + d)
}

func (t *Dense) Triu(k int) *Dense {
	t.initDigits()
	for t.incrDigits() {
		d := t.Digits()
		i, j := d[len(d)-2], d[len(d)-1]
		if j < i+k {
			t.SetAt(d, 0)
		}
	}
	return t
}

func (t *Dense) Tril(k int) *Dense {
	t.initDigits()
	for t.incrDigits() {
		d := t.Digits()
		i, j := d[len(d)-2], d[len(d)-1]
		if j > i+k {
			t.SetAt(d, 0)
		}
	}
	return t
}

func matmul(c, a, b *Dense) *Dense {
	m, an, n := a.Shape()[0], a.Shape()[1], b.Shape()[1]
	c.Zeros(m, n)
	adata, bdata := a.data[a.axis[1].start:], b.data[b.axis[1].start:]
	aStride, bStride := a.axis[1].size, b.axis[1].size
	if a.axisToView[0] == 0 {
		if b.axisToView[0] == 0 {
			if a.conj {
				if b.conj {
					for i := range m {
						for j := range n {
							ap := i * aStride
							bp := j
							var v complex64
							for _ = range an {
								av := conj(adata[ap])
								bv := conj(bdata[bp])
								v += av * bv

								ap++
								bp += bStride
							}
							c.data[i*n+j] = v
						}
					}
				} else {
					for i := range m {
						for j := range n {
							ap := i * aStride
							bp := j
							var v complex64
							for _ = range an {
								av := conj(adata[ap])
								bv := bdata[bp]
								v += av * bv

								ap++
								bp += bStride
							}
							c.data[i*n+j] = v
						}
					}
				}
			} else {
				if b.conj {
					for i := range m {
						for j := range n {
							ap := i * aStride
							bp := j
							var v complex64
							for _ = range an {
								av := adata[ap]
								bv := conj(bdata[bp])
								v += av * bv

								ap++
								bp += bStride
							}
							c.data[i*n+j] = v
						}
					}
				} else {
					for i := range m {
						for j := range n {
							ap := i * aStride
							bp := j
							var v complex64
							for _ = range an {
								av := adata[ap]
								bv := bdata[bp]
								v += av * bv

								ap++
								bp += bStride
							}
							c.data[i*n+j] = v
						}
					}
				}
			}
		} else {
			if a.conj {
				if b.conj {
					for i := range m {
						for j := range n {
							ap := i * aStride
							bp := j * bStride
							var v complex64
							for _ = range an {
								av := conj(adata[ap])
								bv := conj(bdata[bp])
								v += av * bv

								ap++
								bp++
							}
							c.data[i*n+j] = v
						}
					}

				} else {
					for i := range m {
						for j := range n {
							ap := i * aStride
							bp := j * bStride
							var v complex64
							for _ = range an {
								av := conj(adata[ap])
								bv := bdata[bp]
								v += av * bv

								ap++
								bp++
							}
							c.data[i*n+j] = v
						}
					}
				}
			} else {
				if b.conj {
					for i := range m {
						for j := range n {
							ap := i * aStride
							bp := j * bStride
							var v complex64
							for _ = range an {
								av := adata[ap]
								bv := conj(bdata[bp])
								v += av * bv

								ap++
								bp++
							}
							c.data[i*n+j] = v
						}
					}
				} else {
					for i := range m {
						for j := range n {
							ap := i * aStride
							bp := j * bStride
							var v complex64
							for _ = range an {
								av := adata[ap]
								bv := bdata[bp]
								v += av * bv

								ap++
								bp++
							}
							c.data[i*n+j] = v
						}
					}
				}
			}
		}
	} else {
		if b.axisToView[0] == 0 {
			if a.conj {
				if b.conj {
					for i := range m {
						for j := range n {
							ap := i
							bp := j
							var v complex64
							for _ = range an {
								av := conj(adata[ap])
								bv := conj(bdata[bp])
								v += av * bv

								ap += aStride
								bp += bStride
							}
							c.data[i*n+j] = v
						}
					}
				} else {
					for i := range m {
						for j := range n {
							ap := i
							bp := j
							var v complex64
							for _ = range an {
								av := conj(adata[ap])
								bv := bdata[bp]
								v += av * bv

								ap += aStride
								bp += bStride
							}
							c.data[i*n+j] = v
						}
					}
				}
			} else {
				if b.conj {
					for i := range m {
						for j := range n {
							ap := i
							bp := j
							var v complex64
							for _ = range an {
								av := adata[ap]
								bv := conj(bdata[bp])
								v += av * bv

								ap += aStride
								bp += bStride
							}
							c.data[i*n+j] = v
						}
					}
				} else {
					for i := range m {
						for j := range n {
							ap := i
							bp := j
							var v complex64
							for _ = range an {
								av := adata[ap]
								bv := bdata[bp]
								v += av * bv

								ap += aStride
								bp += bStride
							}
							c.data[i*n+j] = v
						}
					}
				}
			}
		} else {
			if a.conj {
				if b.conj {
					for i := range m {
						for j := range n {
							ap := i
							bp := j * bStride
							var v complex64
							for _ = range an {
								av := conj(adata[ap])
								bv := conj(bdata[bp])
								v += av * bv

								ap += aStride
								bp++
							}
							c.data[i*n+j] = v
						}
					}
				} else {
					for i := range m {
						for j := range n {
							ap := i
							bp := j * bStride
							var v complex64
							for _ = range an {
								av := conj(adata[ap])
								bv := bdata[bp]
								v += av * bv

								ap += aStride
								bp++
							}
							c.data[i*n+j] = v
						}
					}
				}
			} else {
				if b.conj {
					for i := range m {
						for j := range n {
							ap := i
							bp := j * bStride
							var v complex64
							for _ = range an {
								av := adata[ap]
								bv := conj(bdata[bp])
								v += av * bv

								ap += aStride
								bp++
							}
							c.data[i*n+j] = v
						}
					}
				} else {
					for i := range m {
						for j := range n {
							ap := i
							bp := j * bStride
							var v complex64
							for _ = range an {
								av := adata[ap]
								bv := bdata[bp]
								v += av * bv

								ap += aStride
								bp++
							}
							c.data[i*n+j] = v
						}
					}
				}
			}
		}
	}

	// for i := range m {
	// 	for j := range n {
	// 		var ap int
	// 		if a.axisToView[0] == 0 {
	// 			ap = i * a.axis[1].size
	// 			ap--
	// 		} else { // Transposed.
	// 			ap = i
	// 			ap -= a.axis[1].size
	// 		}
	// 		var bp int
	// 		if b.axisToView[0] == 0 {
	// 			bp = j
	// 			bp -= b.axis[1].size
	// 		} else { // Transposed.
	// 			bp = j * b.axis[1].size
	// 			bp--
	// 		}

	// 		var v complex64
	// 		for _ = range an {
	// 			if a.axisToView[0] == 0 {
	// 				ap++
	// 			} else { // Transposed.
	// 				ap += a.axis[1].size
	// 			}
	// 			av := adata[ap]
	// 			if a.conj {
	// 				av = conj(av)
	// 			}

	// 			if b.axisToView[0] == 0 {
	// 				bp += b.axis[1].size
	// 			} else { // Transposed.
	// 				bp++
	// 			}
	// 			bv := bdata[bp]
	// 			if b.conj {
	// 				bv = conj(bv)
	// 			}

	// 			v += av * bv
	// 		}
	// 		c.data[i*n+j] = v
	// 	}
	// }
	return c
}
