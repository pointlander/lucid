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

	. "github.com/pointlander/matrix"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

const (
	// FFWindow is the feedfoward window
	FFWindow = 64
	// ModelWindow is the window size
	ModelWindow = 128
	// ModelSamples is the number of samples
	ModelSamples = 256
	// Inputs is the number of inputs
	Inputs = 256
	// Outputs is the number of outputs
	Outputs = 32
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
	E       Set
	Q       Set
	K       Set
	V       Set
}

// NewNet makes a new network
func NewEmbeddingNet(seed int64, inputs, outputs int) Net {
	rng := rand.New(rand.NewSource(seed))
	return Net{
		Inputs:  inputs,
		Outputs: outputs,
		Rng:     rng,
		E:       NewStatistics(inputs, outputs),
		Q:       NewStatistics(outputs, outputs),
		K:       NewStatistics(outputs, outputs),
		V:       NewStatistics(outputs, outputs),
	}
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

// Fire with embedding runs the network
func (n *Net) FireEmbedding(value Matrix) (float32, Matrix, Matrix, Matrix) {
	q := NewMatrix(0, n.Outputs, ModelSamples)
	k := NewMatrix(0, n.Outputs, ModelSamples)
	v := NewMatrix(0, n.Outputs, ModelSamples)
	systemsE := make([]Sample, 0, 8)
	systemsQ := make([]Sample, 0, 8)
	systemsK := make([]Sample, 0, 8)
	systemsV := make([]Sample, 0, 8)
	for i := 0; i < ModelSamples; i++ {
		neurons := n.E.Sample(n.Rng, n.Inputs, n.Outputs)
		outputs := NewMatrix(0, n.Outputs, 1)
		for j := range neurons {
			out := MulT(neurons[j], value)
			outputs.Data = append(outputs.Data, out.Data[0])
		}
		systemsE = append(systemsE, Sample{
			Neurons: neurons,
			Outputs: Sigmoid(outputs),
		})
	}
	for i := 0; i < ModelSamples; i++ {
		neurons := n.Q.Sample(n.Rng, n.Outputs, n.Outputs)
		outputs := NewMatrix(0, n.Outputs, 1)
		for j := range neurons {
			out := MulT(neurons[j], systemsE[i].Outputs)
			q.Data = append(q.Data, out.Data[0])
			outputs.Data = append(outputs.Data, out.Data[0])
		}
		systemsQ = append(systemsQ, Sample{
			Neurons: neurons,
			Outputs: outputs,
		})
	}
	for i := 0; i < ModelSamples; i++ {
		neurons := n.K.Sample(n.Rng, n.Outputs, n.Outputs)
		outputs := NewMatrix(0, n.Outputs, 1)
		for j := range neurons {
			out := MulT(neurons[j], systemsE[i].Outputs)
			k.Data = append(k.Data, out.Data[0])
			outputs.Data = append(outputs.Data, out.Data[0])
		}
		systemsK = append(systemsK, Sample{
			Neurons: neurons,
			Outputs: outputs,
		})
	}
	for i := 0; i < ModelSamples; i++ {
		neurons := n.V.Sample(n.Rng, n.Outputs, n.Outputs)
		outputs := NewMatrix(0, n.Outputs, 1)
		for j := range neurons {
			out := MulT(neurons[j], systemsE[i].Outputs)
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
		systemsE[i].Entropy = entropy
		systemsE[i].Out = outputs[i]
		systemsQ[i].Entropy = entropy
		systemsQ[i].Out = outputs[i]
		systemsK[i].Entropy = entropy
		systemsK[i].Out = outputs[i]
		systemsV[i].Entropy = entropy
		systemsV[i].Out = outputs[i]
	}
	sort.Slice(systemsE, func(i, j int) bool {
		return systemsE[i].Entropy < systemsE[j].Entropy
	})
	sort.Slice(systemsQ, func(i, j int) bool {
		return systemsQ[i].Entropy < systemsQ[j].Entropy
	})
	sort.Slice(systemsK, func(i, j int) bool {
		return systemsK[i].Entropy < systemsK[j].Entropy
	})
	sort.Slice(systemsV, func(i, j int) bool {
		return systemsV[i].Entropy < systemsV[j].Entropy
	})

	n.E = n.CalculateStatistics(systemsE)
	n.Q = n.CalculateStatistics(systemsQ)
	n.K = n.CalculateStatistics(systemsK)
	n.V = n.CalculateStatistics(systemsV)

	return systemsV[0].Entropy, systemsQ[0].Outputs, systemsK[0].Outputs, systemsV[0].Outputs
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

// Mark6 is the mark6 model
func Mark6() {
	input := []byte("abcdefghijklmnopqrstuvwxyz")
	in := NewMatrix(0, 256, 1)
	in.Data = in.Data[:256]
	net := NewEmbeddingNet(1, Inputs, Outputs)
	projection := NewNet(2, Outputs, 2)
	length := len(input)
	const epochs = 8
	points := make(plotter.XYs, len(input))
	data := make([]Matrix, len(input))
	for e := 0; e < epochs; e++ {
		for i := 0; i < length; i++ {
			for j := range in.Data {
				in.Data[j] = 0
			}
			in.Data[input[i]] = 1
			entropy, q, k, v := net.FireEmbedding(in)
			data[i] = v
			fmt.Println(input[i], entropy)
			if e == epochs-1 {
				_, _, _, point := projection.Fire(Normalize(q), Normalize(k), Normalize(v))
				points[i] = plotter.XY{X: float64(point.Data[0]), Y: float64(point.Data[1])}
			} else {
				projection.Fire(q, k, v)
			}
		}
	}

	p := plot.New()

	p.Title.Text = "symbols"
	p.X.Label.Text = "x"
	p.Y.Label.Text = "y"
	p.Legend.Top = true

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	scatter.GlyphStyle.Color = colors[0]
	p.Add(scatter)
	p.Legend.Add("symbols", scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "symbols_projection.png")
	if err != nil {
		panic(err)
	}

	rng := rand.New(rand.NewSource(1))
	l1 := NewRandomMatrix(Outputs, Outputs)
	b1 := NewRandomMatrix(Outputs, 1)
	l2 := NewRandomMatrix(Outputs, Inputs)
	b2 := NewRandomMatrix(Inputs, 1)
	m := []RandomMatrix{l1, b1, l2, b2}
	type Sample struct {
		X    []Matrix
		Cost float32
	}
	var samples []Sample
	for e := 0; e < 256; e++ {
		samples = make([]Sample, 512, 512)
		for i := range samples {
			for _, r := range m {
				samples[i].X = append(samples[i].X, r.Sample(rng))
			}
			for j := range input {
				target := NewMatrix(0, 256, 1)
				target.Data = target.Data[:256]
				if j+1 == len(input) {
					target.Data[0] = 1
				} else {
					target.Data[input[j+1]] = 1
				}
				layer1 := Sigmoid(Add(MulT(samples[i].X[0], data[j]), samples[i].X[1]))
				layer2 := Add(MulT(samples[i].X[2], layer1), samples[i].X[3])
				cost := Quadratic(layer2, target)
				samples[i].Cost += cost.Data[0]
			}
		}
		sort.Slice(samples, func(i, j int) bool {
			return samples[i].Cost < samples[j].Cost
		})
		fmt.Println(e, samples[0].Cost)
		weights, sum := make([]float32, FFWindow), float32(0)
		for i := range weights {
			sum += 1 / samples[i].Cost
			weights[i] = 1 / samples[i].Cost
		}
		for i := range weights {
			weights[i] /= sum
		}
		for k := range m {
			n := NewRandomMatrix(m[k].Cols, m[k].Rows)
			for i := range n.Data {
				n.Data[i].StdDev = 0
			}
			for i := range samples[:FFWindow] {
				for j, value := range samples[i].X[k].Data {
					n.Data[j].Mean += weights[i] * value
				}
			}
			for i := range samples[:FFWindow] {
				for j, value := range samples[i].X[k].Data {
					diff := n.Data[j].Mean - value
					n.Data[j].StdDev += weights[i] * diff * diff
				}
			}
			for i := range n.Data {
				n.Data[i].StdDev /= (FFWindow - 1.0) / FFWindow
				n.Data[i].StdDev = float32(math.Sqrt(float64(n.Data[i].StdDev)))
			}
			m[k] = n
		}
	}

	for i := 0; i < length; i++ {
		for j := range in.Data {
			in.Data[j] = 0
		}
		in.Data[input[i]] = 1
		_, _, _, v := net.FireEmbedding(in)
		layer1 := Sigmoid(Add(MulT(samples[0].X[0], v), samples[0].X[1]))
		layer2 := Add(MulT(samples[0].X[2], layer1), samples[0].X[3])
		max, index := float32(0.0), 0
		for key, value := range layer2.Data {
			if value > max {
				max, index = value, key
			}
		}
		fmt.Println(string(input[i]), string(byte(index)))
	}
}