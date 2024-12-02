package mps

import (
	"fmt"
	"qising/mps/tensor"
	"testing"
)

func TestIsing(t *testing.T) {
	t.Parallel()
	tests := []struct {
		mpo []*tensor.Tensor
	}{
		{mpo: Ising([2]int{4, 1}, -100)},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("%#v", test.mpo), func(t *testing.T) {
			t.Parallel()
			mps := NewMPS(test.mpo, 2)
			t.Logf("%#v", mps)
		})
	}
}
