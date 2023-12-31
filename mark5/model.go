// Copyright 2023 The Lucid Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mark5

import (
	"math"
	"math/rand"

	. "github.com/pointlander/lucid/matrix"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

const (
	// Width is the width of the network
	Width = 8
	// Size is the size of the networm
	Size = Width * Width
)

// Random is a random variable
type Random struct {
	Mean   float32
	StdDev float32
}

// Mark5 is the mark5 model
func Mark5() {
	rng := rand.New(rand.NewSource(1))

	values := make(plotter.Values, 0, 1024)
	for e := 0; e < 1; e++ {
		dist := make([]Random, 0, Size)
		factor := float32(math.Sqrt(2.0 / float64(Width)))
		for i := 0; i < Size; i++ {
			dist = append(dist, Random{
				Mean:   float32(rng.NormFloat64()),
				StdDev: float32(rng.NormFloat64()) * factor,
			})
		}
		model := NewMatrix(0, Width, Width)
		for i := range dist[:Size] {
			model.Data = append(model.Data, dist[i].StdDev*float32(rng.NormFloat64())+dist[i].Mean)
		}
		input := NewMatrix(0, Width, 1)
		for i := 0; i < Width; i++ {
			input.Data = append(input.Data, factor*float32(rng.NormFloat64()))
		}
		for i := 0; i < 1024; i++ {
			output := Sigmoid(MulT(model, Normalize(input)))
			for _, value := range output.Data {
				values = append(values, float64(value))
			}
			copy(input.Data, output.Data)
			input.Data[0] = factor * float32(rng.NormFloat64())
		}
	}

	p := plot.New()
	p.Title.Text = "rnn distribution"
	histogram, err := plotter.NewHist(values, 256)
	if err != nil {
		panic(err)
	}
	p.Add(histogram)
	err = p.Save(8*vg.Inch, 8*vg.Inch, "rnn.png")
	if err != nil {
		panic(err)
	}
}
