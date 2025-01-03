package tensor

import (
	"fmt"
	"testing"
)

func TestGemm(t *testing.T) {
	t.Parallel()
	tests := []struct {
		ts []*Dense
		p  [][]complex64
	}{
		{
			ts: []*Dense{
				T2([][]complex64{{1 - 1i, 2}, {3, 4i}}),
				T2([][]complex64{{-1i, -2i}, {-3i, -4i}}),
				T2([][]complex64{{2 + 3i, -2}, {-3 + 1i, -4}}),
			},
			p: [][]complex64{{35 + 11i, 10 + 54i}, {-9 + 64i, -88 + 30i}},
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			p := Gemm(test.ts...)
			if err := equal2(p, test.p, epsilon); err != nil {
				t.Fatalf("%+v", err)
			}
		})
	}
}
