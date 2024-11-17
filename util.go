package qising

import "time"

type skipThrottler struct {
	d    time.Duration
	last time.Time
}

func newSkipThrottler(d time.Duration) *skipThrottler {
	tt := &skipThrottler{d: d, last: time.Date(0, 0, 0, 0, 0, 0, 0, time.UTC)}
	return tt
}

func (tt *skipThrottler) Ok() bool {
	now := time.Now()
	if now.Before(tt.last.Add(tt.d)) {
		return false
	}

	tt.last = time.Now()
	return true
}
