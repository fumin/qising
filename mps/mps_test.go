package mps

import (
	"fmt"
	"testing"

	"qising/tensor"
)

func TestIsing(t *testing.T) {
	t.Parallel()
	tests := []struct {
		mpo []*tensor.Dense
	}{
		{mpo: Ising([2]int{4, 1}, -100)},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("%#v", test.mpo), func(t *testing.T) {
			t.Parallel()
			bondDim := 3
			mps := NewMPS(test.mpo, bondDim)
			RightNormalize(mps)
			t.Logf("%#v", mps)
		})
	}
}
