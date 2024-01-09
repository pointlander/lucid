// Copyright 2023 The Lucid Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mark6

import (
	"fmt"
	"image/color"
	"math"
	"math/rand"
	"sort"

	"github.com/pointlander/datum/iris"
	. "github.com/pointlander/matrix"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

const (
	// ModelWindow is the window size
	ModelWindow = 32
	// GaussianWindow is the gaussian window
	GaussianWindow = 8
	// ModelSamples is the number of samples
	ModelSamples = 256
	// Inputs is the number of inputs
	Inputs = 4
	// Outputs is the number of outputs
	Outputs = 4
	// Embedding is the embedding size
	Embedding = 3 * 4
	// Clusters is the number of clusters
	Clusters = 3
)

var colors = [...]color.RGBA{
	{R: 0xff, G: 0x00, B: 0x00, A: 255},
	{R: 0x00, G: 0xff, B: 0x00, A: 255},
	{R: 0x00, G: 0x00, B: 0xff, A: 255},
}

// Random is a random variable
type Rand struct {
	Mean   float32
	StdDev float32
	Count  float32
}

// Set is a set of statistics
type Set [][]Random

// NewStatistics generates a new statistics model
func NewStatistics(inputs, outputs int) Set {
	statistics := make(Set, outputs)
	for i := range statistics {
		for j := 0; j < inputs; j++ {
			statistics[i] = append(statistics[i], Random{
				Mean:   0,
				StdDev: 1,
			})
		}
	}
	return statistics
}

// Sample samples from the statistics
func (s Set) Sample(rng *rand.Rand, inputs, outputs int) []Matrix {
	neurons := make([]Matrix, outputs)
	for j := range neurons {
		neurons[j] = NewMatrix(0, inputs, 1)
		for k := 0; k < inputs; k++ {
			v := float32(rng.NormFloat64())*s[j][k].StdDev + s[j][k].Mean
			if v > 0 {
				v = 1
			} else {
				v = -1
			}
			neurons[j].Data = append(neurons[j].Data, v)
		}
	}
	return neurons
}

// Net is a net
type Net struct {
	Inputs  int
	Outputs int
	Rng     *rand.Rand
	Q       Set
	K       Set
	V       Set
}

// NewNet makes a new network
func NewNet(seed int64, inputs, outputs int) Net {
	rng := rand.New(rand.NewSource(seed))
	return Net{
		Inputs:  inputs,
		Outputs: outputs,
		Rng:     rng,
		Q:       NewStatistics(inputs, outputs),
		K:       NewStatistics(inputs, outputs),
		V:       NewStatistics(inputs, outputs),
	}
}

// Sample is a sample of a random neural network
type Sample struct {
	Entropy float32
	Neurons []Matrix
	Outputs Matrix
	Out     Matrix
}

// CalculateStatistics calculates the statistics of systems
func (n Net) CalculateStatistics(systems []Sample) Set {
	statistics := make(Set, n.Outputs)
	for i := range statistics {
		for j := 0; j < n.Inputs; j++ {
			statistics[i] = append(statistics[i], Random{
				Mean:   0,
				StdDev: 0,
			})
		}
	}
	weights, sum := make([]float32, ModelWindow), float32(0)
	for i := range weights {
		sum += 1 / systems[i].Entropy
		weights[i] = 1 / systems[i].Entropy
	}
	for i := range weights {
		weights[i] /= sum
	}

	for i := range systems[:ModelWindow] {
		for j := range systems[i].Neurons {
			for k, value := range systems[i].Neurons[j].Data {
				statistics[j][k].Mean += weights[i] * value
			}
		}
	}
	for i := range systems[:ModelWindow] {
		for j := range systems[i].Neurons {
			for k, value := range systems[i].Neurons[j].Data {
				diff := statistics[j][k].Mean - value
				statistics[j][k].StdDev += weights[i] * diff * diff
			}
		}
	}
	for i := range statistics {
		for j := range statistics[i] {
			statistics[i][j].StdDev /= (ModelWindow - 1.0) / ModelWindow
			statistics[i][j].StdDev = float32(math.Sqrt(float64(statistics[i][j].StdDev)))
		}
	}
	return statistics
}

// Fire runs the network
func (n *Net) Fire(query, key, value Matrix) (float32, Matrix, Matrix, Matrix) {
	q := NewMatrix(0, n.Outputs, ModelSamples)
	k := NewMatrix(0, n.Outputs, ModelSamples)
	v := NewMatrix(0, n.Outputs, ModelSamples)
	systemsQ := make([]Sample, 0, 8)
	systemsK := make([]Sample, 0, 8)
	systemsV := make([]Sample, 0, 8)
	for i := 0; i < ModelSamples; i++ {
		neurons := n.Q.Sample(n.Rng, n.Inputs, n.Outputs)
		outputs := NewMatrix(0, n.Outputs, 1)
		for j := range neurons {
			out := MulT(neurons[j], query)
			q.Data = append(q.Data, out.Data[0])
			outputs.Data = append(outputs.Data, out.Data[0])
		}
		systemsQ = append(systemsQ, Sample{
			Neurons: neurons,
			Outputs: outputs,
		})
	}
	for i := 0; i < ModelSamples; i++ {
		neurons := n.K.Sample(n.Rng, n.Inputs, n.Outputs)
		outputs := NewMatrix(0, n.Outputs, 1)
		for j := range neurons {
			out := MulT(neurons[j], key)
			k.Data = append(k.Data, out.Data[0])
			outputs.Data = append(outputs.Data, out.Data[0])
		}
		systemsK = append(systemsK, Sample{
			Neurons: neurons,
			Outputs: outputs,
		})
	}
	for i := 0; i < ModelSamples; i++ {
		neurons := n.V.Sample(n.Rng, n.Inputs, n.Outputs)
		outputs := NewMatrix(0, n.Outputs, 1)
		for j := range neurons {
			out := MulT(neurons[j], value)
			v.Data = append(v.Data, out.Data[0])
			outputs.Data = append(outputs.Data, out.Data[0])
		}
		systemsV = append(systemsV, Sample{
			Neurons: neurons,
			Outputs: outputs,
		})
	}
	outputs, entropies := SelfEntropy(q, k, v)
	for i, entropy := range entropies {
		systemsQ[i].Entropy = entropy
		systemsQ[i].Out = outputs[i]
		systemsK[i].Entropy = entropy
		systemsK[i].Out = outputs[i]
		systemsV[i].Entropy = entropy
		systemsV[i].Out = outputs[i]
	}
	sort.Slice(systemsQ, func(i, j int) bool {
		return systemsQ[i].Entropy < systemsQ[j].Entropy
	})
	sort.Slice(systemsK, func(i, j int) bool {
		return systemsK[i].Entropy < systemsK[j].Entropy
	})
	sort.Slice(systemsV, func(i, j int) bool {
		return systemsV[i].Entropy < systemsV[j].Entropy
	})

	n.Q = n.CalculateStatistics(systemsQ)
	n.K = n.CalculateStatistics(systemsK)
	n.V = n.CalculateStatistics(systemsV)

	return systemsV[0].Entropy, systemsQ[0].Outputs, systemsK[0].Outputs, systemsV[0].Outputs
}

// Iris is a iris data point
type Iris struct {
	iris.Iris
	I         int
	Embedding []float32
	Cluster   int
}

// Mark6 is the mark6 model
func Mark6() {
	rng := rand.New(rand.NewSource(1))
	data, err := iris.Load()
	if err != nil {
		panic(err)
	}

	flowers := make([]Iris, len(data.Fisher))

	for i, value := range data.Fisher {
		sum := 0.0
		for _, v := range value.Measures {
			sum += v * v
		}
		length := math.Sqrt(sum)
		for i := range value.Measures {
			value.Measures[i] /= length
		}
		flowers[i].Iris = value
		flowers[i].I = i
	}
	net := NewNet(1, Inputs+1, Outputs)
	projection := NewNet(2, Outputs, 2)
	length := len(data.Fisher)
	const epochs = 5
	points := make(plotter.XYs, len(flowers))
	for i := 0; i < epochs; i++ {
		perm := rng.Perm(len(flowers))
		for epoch := 0; epoch < length; epoch++ {
			index := perm[epoch]
			query := NewMatrix(0, Inputs+1, 1)
			for _, value := range flowers[index].Measures {
				query.Data = append(query.Data, float32(value))
			}
			query.Data = append(query.Data, 0)
			key := NewMatrix(0, Inputs+1, 1)
			for _, value := range flowers[index].Measures {
				key.Data = append(key.Data, float32(value))
			}
			key.Data = append(key.Data, 0)
			value := NewMatrix(0, Inputs+1, 1)
			for _, v := range flowers[index].Measures {
				value.Data = append(value.Data, float32(v))
			}
			value.Data = append(value.Data, 0)
			label := flowers[index].Label
			entropy, q, k, v := net.Fire(query, key, value)
			fmt.Println(label, entropy, v.Data)
			copy(query.Data, q.Data)
			query.Data[4] = 1
			copy(key.Data, k.Data)
			key.Data[4] = 1
			copy(value.Data, v.Data)
			value.Data[4] = 1
			if i == epochs-1 {
				entropy, q, k, v = net.Fire(query, key, value)
				flowers[index].Embedding = append(flowers[index].Embedding, q.Data...)
				flowers[index].Embedding = append(flowers[index].Embedding, k.Data...)
				flowers[index].Embedding = append(flowers[index].Embedding, v.Data...)
				_, _, _, point := projection.Fire(q, k, v)
				points[index] = plotter.XY{X: float64(point.Data[0]), Y: float64(point.Data[1])}
				fmt.Println(label, entropy, v.Data)
			} else {
				entropy, q, k, v = net.Fire(query, key, value)
				projection.Fire(q, k, v)
				fmt.Println(label, entropy, v.Data)
			}
		}
	}

	p := plot.New()
	if err != nil {
		panic(err)
	}

	p.Title.Text = "symbols"
	p.X.Label.Text = "x"
	p.Y.Label.Text = "y"
	p.Legend.Top = true

	c := 0
	sets := make([]plotter.XYs, Clusters, Clusters)
	names := make([]string, Clusters)
	for j := range sets {
		sets[j] = make(plotter.XYs, 0, 50)
	}
	for key, value := range flowers {
		if names[value.Cluster] == "" {
			names[value.Cluster] = value.Label
		}
		sets[value.Cluster] = append(sets[value.Cluster], points[key])
	}
	for j, set := range sets {
		scatter, err := plotter.NewScatter(set)
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(1)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		scatter.GlyphStyle.Color = colors[c]
		p.Add(scatter)
		p.Legend.Add(names[j], scatter)
		c++
	}

	err = p.Save(8*vg.Inch, 8*vg.Inch, "symbols_projection.png")
	if err != nil {
		panic(err)
	}
}
