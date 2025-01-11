package util

import "time"

type SkipThrottler struct {
	d    time.Duration
	last time.Time
}

func NewSkipThrottler(d time.Duration) *SkipThrottler {
	tt := &SkipThrottler{d: d, last: time.Date(0, 0, 0, 0, 0, 0, 0, time.UTC)}
	return tt
}

func (tt *SkipThrottler) Ok() bool {
	now := time.Now()
	if now.Before(tt.last.Add(tt.d)) {
		return false
	}

	tt.last = time.Now()
	return true
}
