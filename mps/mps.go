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

func (c *Tensor) Einsum(b *Tensor, aAxis, bAxis int) {
	// Let c = a*b, prepare aShape, aData, c.Shape, and c.Data.
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
		if i != aAxis {
			c.Shape = append(c.Shape, d)
			cLen *= d
		}
	}
	for i, d := range b.Shape {
		if i != bAxis {
			c.Shape = append(c.Shape, d)
			cLen *= d
		}
	}
	c.Data = append(c.Data, make([]complex64, cLen)...)
	aData := c.Data[cLen:]
	copy(aData, c.Data[:aLen])
	c.Data = c.Data[:cLen]

	var aGap int = 1
	for i := aAxis + 1; i < len(aShape); i++ {
		aGap *= aShape[i]
	}
	var bGap int = 1
	for i := bAxis + 1; i < len(b.Shape); i++ {
		bGap *= b.Shape[i]
	}

	c.digits = c.digits[:0]
	c.digits = append(c.digits, make([]int, len(c.Shape))...)
	c.digits[len(c.digits)-1] = -1
	var cIdx int = -1
	for incr(c.digits, c.Shape) {
		cIdx++

		// Find the start of a.Data.
		aDigits := b.digits[:0]
		aDigits = append(aDigits, make([]int, len(aShape))...)
		copy(aDigits[:aAxis], c.digits[:aAxis])
		aDigits[aAxis] = 0
		copy(aDigits[aAxis+1:], c.digits[aAxis:len(aShape)-1])
		aIdx := at(aDigits, aShape)

		// Find the start of b.Data.
		b.digits = b.digits[:0]
		b.digits = append(b.digits, make([]int, len(b.Shape))...)
		copy(b.digits[:bAxis], c.digits[len(aShape)-1:len(aShape)-1+bAxis])
		b.digits[bAxis] = 0
		copy(b.digits[bAxis+1:], c.digits[len(aShape)-1+bAxis:])
		bIdx := at(b.digits, b.Shape)

		var v complex64
		for j := range aShape[aAxis] {
			av := aData[aIdx+j*aGap]
			bv := b.Data[bIdx+j*bGap]
			v += av * bv
		}

		c.Data[cIdx] = v
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
