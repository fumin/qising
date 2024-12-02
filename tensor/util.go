package tensor

// SameArray returns true if x and y share the same underlying array. Sharing
// the same underlying array does not imply overlap, but rather that it is
// possible to reslice one or the other such that both point to the same memory
// region.
// For more details, please see https://groups.google.com/g/golang-nuts/c/ks1jvoyMYuc.
func SameArray(x, y []complex64) bool {
	return cap(x) > 0 && cap(y) > 0 && &(x[:cap(x)][cap(x)-1]) == &(y[:cap(y)][cap(y)-1])
}

// Overlap returns a slice pointing to the overlapping memory between x and y.
// Nil is returned if x and y do not share the same underlying array or if they
// do not overlap.
func Overlap(x, y []complex64) (z []complex64) {
	if len(x) == 0 || len(y) == 0 || !SameArray(x, y) {
		return
	} else if cap(x) < cap(y) {
		x, y = y, x
	}
	if cap(x)-len(x) < cap(y) {
		z = y
		if lxy := len(x) - (cap(x) - cap(y)); lxy < len(y) {
			z = z[:lxy]
		}
	}
	return
}
