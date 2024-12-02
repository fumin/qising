package mps

import (
	"qising/mat"
)

var (
	PauliX = M2(mat.PauliX)
	PauliY = M2(mat.PauliY)
	PauliZ = M2(mat.PauliZ)
)

type Tensor struct {
	Shape  []int
	Data   []complex64
	digits []int
}

func M2(slice [][]complex64) *Tensor {
	t := &Tensor{Shape: []int{len(slice), len(slice[0])}}
	for i := range len(slice) {
		for j := range len(slice[0]) {
			t.Data = append(t.Data, slice[i][j])
		}
	}
	return t
}

func M3(slice [][][]complex64) *Tensor {
	t := &Tensor{Shape: []int{len(slice), len(slice[0]), len(slice[0][0])}}
	for i := range len(slice) {
		for j := range len(slice[0]) {
			for k := range len(slice[0][0]) {
				t.Data = append(t.Data, slice[i][j][k])
			}
		}
	}
	return t
}

func M4(slice [][][][]complex64) *Tensor {
	t := &Tensor{Shape: []int{len(slice), len(slice[0]), len(slice[0][0]), len(slice[0][0][0])}}
	for i := range len(slice) {
		for j := range len(slice[0]) {
			for k := range len(slice[0][0]) {
				for l := range len(slice[0][0][0]) {
					t.Data = append(t.Data, slice[i][j][k][l])
				}
			}
		}
	}
	return t
}

func (a *Tensor) Equal(b *Tensor) bool {
	if len(a.Shape) != len(b.Shape) {
		return false
	}
	for i := range a.Shape {
		if a.Shape[i] != b.Shape[i] {
			return false
		}
	}
	if len(a.Data) != len(b.Data) {
		return false
	}
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			return false
		}
	}
	return true
}

func (c *Tensor) Einsum(b *Tensor, aSumDim, bSumDim int) {
	// Let c = a*b, prepare a.Shape, a.Data, c.Shape, and c.Data.
	aShapeLen := len(c.Shape)
	// cShapeLen is len(c.Shape).
	cShapeLen := aShapeLen + len(b.Shape) - 2
	c.Shape = append(c.Shape, make([]int, cShapeLen)...)
	aShape := c.Shape[cShapeLen:]
	copy(aShape, c.Shape[:aShapeLen])
	c.Shape = c.Shape[:0]
	var cLen, aLen int = 1, 1
	for i, d := range aShape {
		aLen *= d
		if i != aSumDim {
			c.Shape = append(c.Shape, d)
			cLen *= d
		}
	}
	for i, d := range b.Shape {
		if i != bSumDim {
			c.Shape = append(c.Shape, d)
			cLen *= d
		}
	}
	c.Data = append(c.Data, make([]complex64, cLen)...)
	aData := c.Data[cLen:]
	copy(aData, c.Data[:aLen])
	c.Data = c.Data[:cLen]

	c.digits = c.digits[:0]
	c.digits = append(c.digits, make([]int, len(c.Shape))...)
	c.digits[len(c.digits)-1] = -1
	for incr(c.digits, c.Shape) {
		var v complex64
		for j := range aShape[aSumDim] {
			aDigits := b.digits[:0]
			for i := range aShape {
				switch {
				case i < aSumDim:
					aDigits = append(aDigits, c.digits[i])
				case i == aSumDim:
					aDigits = append(aDigits, j)
				default:
					aDigits = append(aDigits, c.digits[i-1])
				}
			}
			av := aData[at(aDigits, aShape)]

			b.digits = b.digits[:0]
			for k := range b.Shape {
				switch {
				case k < bSumDim:
					b.digits = append(b.digits, c.digits[len(aShape)-1+k])
				case k == bSumDim:
					b.digits = append(b.digits, j)
				default:
					b.digits = append(b.digits, c.digits[len(aShape)-1+k-1])
				}
			}
			bv := b.Data[at(b.digits, b.Shape)]

			v += av * bv
		}

		c.Data[at(c.digits, c.Shape)] = v
	}
}

func at(digits, shape []int) int {
	var idx int = 0
	var power int = 1
	for i := len(shape) - 1; i >= 0; i-- {
		idx += digits[i] * power
		power *= shape[i]
	}
	return idx
}

func incr(digits, shape []int) bool {
	digits[len(digits)-1]++

	for i := len(digits) - 1; i >= 1; i-- {
		if digits[i] < shape[i] {
			break
		}
		digits[i] = 0
		digits[i-1]++
	}

	return digits[0] < shape[0]
}

type MPO struct {
	P []Tensor
}
