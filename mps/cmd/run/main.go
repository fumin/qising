package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/cmplx"
	"os"
	"path/filepath"
	"slices"

	"github.com/fumin/qising/mps"
	"github.com/fumin/tensor"
	"github.com/pkg/errors"
)

var (
	runDir = flag.String("d", filepath.Join("runs", "qising"), "run directory")
)

type Config struct {
	l       int
	h       complex64
	bondDim int
	tol     float32
}

func newConfigs() []Config {
	hLogs := []float64{0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 1.5, 2}
	// Add negative logs, so that hLogs becomes {-2, -1, tcLog, 1, 2...}.
	hLogsLen := len(hLogs)
	for i := range hLogsLen {
		hLogs = append(hLogs, -hLogs[i])
	}
	// tcGuess is a guess of the critical temperature.
	const tcGuess float64 = 1
	tcLog := math.Log10(tcGuess)
	for i := range hLogs {
		hLogs[i] += tcLog
	}
	slices.Sort(hLogs)

	configs := make([]Config, 0)
	for _, l := range []int{25} {
		for _, hl := range hLogs {
			for _, bondDim := range []int{2, 4, 8} {
				h := math.Pow(10, hl)
				cfg := Config{l: l, h: complex(float32(h), 0)}
				cfg.bondDim = bondDim

				switch bondDim {
				case 2:
					cfg.tol = 1e-4
				case 4:
					cfg.tol = 1e-5
				default:
					cfg.tol = 1e-6
				}

				configs = append(configs, cfg)
			}
		}
	}
	return configs
}

func sqrt(x complex64) complex64 {
	return complex64(cmplx.Sqrt(complex128(x)))
}

type Statistics struct {
	cfg Config
	e0  float32
	m   float32
}

func solve(cfg Config) (Statistics, error) {
	n := [2]int{cfg.l, 1}
	h := mps.Ising(n, cfg.h)
	mz := mps.MagnetizationZ(n)

	// Buffers.
	fs := make([]*tensor.Dense, 0, len(h))
	for _ = range h {
		fs = append(fs, tensor.Zeros(1))
	}
	bufs := make([]*tensor.Dense, 0)
	for _ = range 10 {
		bufs = append(bufs, tensor.Zeros(1))
	}

	// Search for ground state.
	state := mps.RandMPS(h, cfg.bondDim)
	opt := mps.NewSearchGroundStateOptions().Tol(cfg.tol)
	if err := mps.SearchGroundState(fs, h, state, [10]*tensor.Dense(bufs), opt); err != nil {
		return Statistics{}, errors.Wrap(err, "")
	}

	// Calculate statistics.
	psiIP := mps.InnerProduct(state, state, [2]*tensor.Dense(bufs))
	e0 := mps.LExpressions(fs, h, state, [2]*tensor.Dense(bufs)) / psiIP
	// Calculate magnetization.
	m2 := mps.H2(mz, state, [2]*tensor.Dense(bufs)) / psiIP
	m := sqrt(m2) / complex(float32(len(state)), 0) // per spin

	return Statistics{cfg: cfg, e0: real(e0), m: real(m)}, nil
}

func main() {
	flag.Parse()
	log.SetFlags(log.Lmicroseconds | log.Llongfile | log.LstdFlags)

	if err := mainWithErr(); err != nil {
		log.Fatalf("%+v", err)
	}
}

func mainWithErr() error {
	if err := os.MkdirAll(*runDir, os.ModePerm); err != nil {
		return errors.Wrap(err, "")
	}

	configs := newConfigs()
	statistics := make([]Statistics, 0, len(configs))
	for _, cfg := range configs {
		stat, err := solve(cfg)
		if err != nil {
			return errors.Wrap(err, fmt.Sprintf("%#v", cfg))
		}
		statistics = append(statistics, stat)
		log.Printf("%#v", stat)
	}

	fmt.Printf("l,h,b,e0,m\n")
	for _, s := range statistics {
		fmt.Printf("%d,%f,%d,%f,%f\n", s.cfg.l, real(s.cfg.h), s.cfg.bondDim, s.e0, s.m)
	}

	return nil
}
