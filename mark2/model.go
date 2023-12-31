// Copyright 2023 The Lucid Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mark2

import (
	"fmt"
	"image/color"
	"math"
	"math/rand"
	"sort"

	"github.com/pointlander/datum/iris"
	. "github.com/pointlander/lucid/matrix"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

const (
	// Window is the window size
	Window = 32
	// Rate is the learning rate
	Rate = .3
	// Samples is the number of samples
	Samples = 256
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
type Random struct {
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
	weights, sum := make([]float32, Window), float32(0)
	for i := range weights {
		sum += 1 / systems[i].Entropy
		weights[i] = 1 / systems[i].Entropy
	}
	for i := range weights {
		weights[i] /= sum
	}

	for i := range systems[:Window] {
		for j := range systems[i].Neurons {
			for k, value := range systems[i].Neurons[j].Data {
				statistics[j][k].Mean += weights[i] * value
			}
		}
	}
	for i := range systems[:Window] {
		for j := range systems[i].Neurons {
			for k, value := range systems[i].Neurons[j].Data {
				diff := statistics[j][k].Mean - value
				statistics[j][k].StdDev += weights[i] * diff * diff
			}
		}
	}
	for i := range statistics {
		for j := range statistics[i] {
			statistics[i][j].StdDev /= (Window - 1.0) / Window
			statistics[i][j].StdDev = float32(math.Sqrt(float64(statistics[i][j].StdDev)))
		}
	}
	return statistics
}

// Fire runs the network
func (n *Net) Fire(query, key, value Matrix) (float32, Matrix, Matrix, Matrix) {
	q := NewMatrix(0, n.Outputs, Samples)
	k := NewMatrix(0, n.Outputs, Samples)
	v := NewMatrix(0, n.Outputs, Samples)
	systemsQ := make([]Sample, 0, 8)
	systemsK := make([]Sample, 0, 8)
	systemsV := make([]Sample, 0, 8)
	for i := 0; i < Samples; i++ {
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
	for i := 0; i < Samples; i++ {
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
	for i := 0; i < Samples; i++ {
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
}

// GaussianCluster is a gaussian clustering algorithm
func GaussianCluster(flowers []Iris) {
	rng := rand.New(rand.NewSource(1))
	a := make([]Random, Embedding*Clusters*len(flowers))
	for i := range a {
		a[i].StdDev = 1
	}
	type Sample struct {
		S []float32
		V float32
	}
	samples := make([]Sample, 256)
	for i := range samples {
		samples[i].S = make([]float32, Embedding*Clusters*len(flowers))
	}
	for i := 0; i < 256; i++ {
		for j := range samples {
			for k := range samples[j].S {
				d := a[k]
				samples[j].S[k] = d.StdDev*float32(rng.NormFloat64()) + d.Mean
				samples[j].V = 0
			}
			var clusters [Embedding * Clusters]Random
			var out [Embedding]Random
			for k := 0; k < Embedding*Clusters*len(flowers); k += Embedding * Clusters {
				for o := 0; o < Embedding*Clusters; o += Clusters {
					outsider := true
					for l := 0; l < Clusters; l++ {
						if samples[j].S[k+o+l] > 0 {
							clusters[o+l].Mean += flowers[k/(Embedding*Clusters)].Embedding[o/Clusters]
							clusters[o+l].Count++
							outsider = false
						}
					}
					if outsider {
						out[o/Clusters].Mean += flowers[k/(Embedding*3)].Embedding[o/3]
						out[o/Clusters].Count++
					}
				}
			}

			for k := 0; k < Embedding*Clusters*len(flowers); k += Embedding * Clusters {
				for o := 0; o < Embedding*Clusters; o += Clusters {
					outsider := true
					for l := 0; l < Clusters; l++ {
						if samples[j].S[k+o+l] > 0 {
							clusters[o+l].Mean /= clusters[o+l].Count
							outsider = false
						}
					}
					if outsider {
						out[o/Clusters].Mean /= out[o/Clusters].Count
					}
				}
			}

			for k := 0; k < Embedding*Clusters*len(flowers); k += Embedding * Clusters {
				for o := 0; o < Embedding*Clusters; o += Clusters {
					outsider := true
					for l := 0; l < Clusters; l++ {
						if samples[j].S[k+o+l] > 0 {
							diff := flowers[k/(Embedding*Clusters)].Embedding[o/Clusters] - clusters[o+l].Mean
							clusters[o+l].StdDev += diff * diff
							outsider = false
						}
					}
					if outsider {
						diff := flowers[k/(Embedding*Clusters)].Embedding[o/Clusters] - out[o/Clusters].Mean
						out[o/Clusters].StdDev += diff * diff
					}
				}
			}

			for k := 0; k < Embedding*Clusters*len(flowers); k += Embedding * Clusters {
				for o := 0; o < Embedding*Clusters; o += Clusters {
					outsider := true
					for l := 0; l < Clusters; l++ {
						if samples[j].S[k+o+l] > 0 {
							clusters[o+l].StdDev /= clusters[o+l].Count
							clusters[o+l].StdDev = float32(math.Sqrt(float64(clusters[o+l].StdDev)))
							outsider = false
						}
					}
					if outsider {
						out[o/Clusters].StdDev /= out[o/Clusters].Count
						out[o/Clusters].StdDev = float32(math.Sqrt(float64(out[o/Clusters].StdDev)))
					}
				}
			}

			v := float32(0.0)
			for k := range clusters {
				v += clusters[k].StdDev
			}
			for k := range out {
				v += out[k].StdDev
			}
			samples[j].V = v
		}

		sort.Slice(samples, func(i, j int) bool {
			return samples[i].V < samples[j].V
		})
		fmt.Println(samples[0].V)

		aa := make([]Random, Embedding*Clusters*len(flowers))
		weights, sum := make([]float32, Window), float32(0)
		for i := range weights {
			sum += 1 / samples[i].V
			weights[i] = 1 / samples[i].V
		}
		for i := range weights {
			weights[i] /= sum
		}

		for i := range samples[:Window] {
			for j, value := range samples[i].S {
				aa[j].Mean += weights[i] * value
			}
		}
		for i := range samples[:Window] {
			for j, value := range samples[i].S {
				diff := aa[j].Mean - value
				aa[j].StdDev += weights[i] * diff * diff
			}
		}
		for i := range aa {
			aa[i].StdDev /= (float32(Window) - 1) / float32(Window)
			aa[i].StdDev = float32(math.Sqrt(float64(aa[i].StdDev)))
		}

		a = aa
	}

	for k := 0; k < Embedding*Clusters*len(flowers); k += Embedding * Clusters {
		for j, value := range a[k : k+Embedding*Clusters] {
			if value.Mean > 0 {
				fmt.Printf("1 ")
			} else {
				fmt.Printf("0 ")
			}
			if j%Clusters == 2 {
				fmt.Printf("  ")
			}
		}
		fmt.Println()
	}
}

// Mark2 is the mark2 model
func Mark2() {
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
	//layer := NewNet(2, Inputs, 2*Inputs)
	//net := NewNet(1, 2*Inputs, Outputs)
	perm := rng.Perm(len(flowers))
	net := NewNet(1, Inputs+1, Outputs)
	projection := NewNet(2, Outputs, 2)
	length := len(data.Fisher)
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
		//_, output := layer.Fire(input)
		entropy, q, k, v := net.Fire(query, key, value)
		projection.Fire(q, k, v)
		fmt.Println(label, entropy, v.Data)
		copy(query.Data, q.Data)
		query.Data[4] = 1
		copy(key.Data, k.Data)
		key.Data[4] = 1
		copy(value.Data, v.Data)
		value.Data[4] = 1
		entropy, q, k, v = net.Fire(query, key, value)
		projection.Fire(q, k, v)
		fmt.Println(label, entropy, v.Data)
	}
	nn := map[string][6][]float32{
		"Iris-setosa":     [6][]float32{},
		"Iris-versicolor": [6][]float32{},
		"Iris-virginica":  [6][]float32{},
	}
	perm = rng.Perm(len(flowers))
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
		//e, output := layer.Fire(input)
		entropy, q, k, v := net.Fire(query, key, value)
		projection.Fire(q, k, v)
		if value := nn[label]; value[0] == nil {
			value[0] = q.Data
			value[1] = k.Data
			value[2] = v.Data
			nn[label] = value
		}
		fmt.Println(label, entropy, v.Data)
		copy(query.Data, q.Data)
		query.Data[4] = 1
		copy(key.Data, k.Data)
		key.Data[4] = 1
		copy(value.Data, v.Data)
		value.Data[4] = 1
		entropy, q, k, v = net.Fire(query, key, value)
		projection.Fire(q, k, v)
		fmt.Println(label, entropy, v.Data)
		if value := nn[label]; value[3] == nil {
			value[3] = q.Data
			value[4] = k.Data
			value[5] = v.Data
			nn[label] = value
		}
	}
	clusters := map[string][150]int{
		"Iris-setosa":     [150]int{},
		"Iris-versicolor": [150]int{},
		"Iris-virginica":  [150]int{},
	}
	//vectors := []Matrix{}
	search := func(query Matrix, epoch, a int) {
		min, index := math.MaxFloat32, ""
		for name, set := range nn {
			distance := 0.0
			for j, value := range set[a] {
				diff := float64(query.Data[j] - value)
				distance += diff * diff
			}
			distance = math.Sqrt(distance)
			if distance < min {
				min, index = distance, name
			}
		}
		list := clusters[index]
		list[epoch]++
		clusters[index] = list
	}
	perm = rng.Perm(len(flowers))
	points := make(plotter.XYs, len(flowers))
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
		//e, output := layer.Fire(input)
		entropy, q, k, v := net.Fire(query, key, value)
		projection.Fire(q, k, v)
		search(q, index, 0)
		search(k, index, 1)
		search(v, index, 2)
		copy(query.Data, q.Data)
		query.Data[4] = 1
		copy(key.Data, k.Data)
		key.Data[4] = 1
		copy(value.Data, v.Data)
		value.Data[4] = 1
		entropy, q, k, v = net.Fire(query, key, value)
		flowers[index].Embedding = append(flowers[index].Embedding, q.Data...)
		flowers[index].Embedding = append(flowers[index].Embedding, k.Data...)
		flowers[index].Embedding = append(flowers[index].Embedding, v.Data...)
		_, _, _, point := projection.Fire(q, k, v)
		points[index] = plotter.XY{X: float64(point.Data[0]), Y: float64(point.Data[1])}
		//vectors = append(vectors, output)
		fmt.Println(label, entropy, v.Data)
		search(q, index, 3)
		search(k, index, 4)
		search(v, index, 5)
	}
	results := map[string][]int{
		"Iris-setosa":     []int{},
		"Iris-versicolor": []int{},
		"Iris-virginica":  []int{},
	}
	for i := 0; i < 150; i++ {
		max, name := 0, ""
		for k, v := range clusters {
			if v[i] > max {
				max, name = v[i], k
			}
		}
		list := results[name]
		list = append(list, i)
		results[name] = list
	}
	fmt.Println(clusters)
	fmt.Println(results)

	GaussianCluster(flowers)

	p := plot.New()
	if err != nil {
		panic(err)
	}

	p.Title.Text = "iris"
	p.X.Label.Text = "x"
	p.Y.Label.Text = "y"
	p.Legend.Top = true

	c := 0
	for key, value := range results {
		set := make(plotter.XYs, 0, 50)
		for _, point := range value {
			set = append(set, points[point])
		}
		scatter, err := plotter.NewScatter(set)
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(1)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		scatter.GlyphStyle.Color = colors[c]
		p.Add(scatter)
		p.Legend.Add(key, scatter)
		c++
	}

	err = p.Save(8*vg.Inch, 8*vg.Inch, "projection.png")
	if err != nil {
		panic(err)
	}

	/*graph := pagerank.NewGraph64()
	for i := 0; i < len(vectors); i++ {
		for j := 0; j < len(vectors); j++ {
			a, b := vectors[i], vectors[j]
			distance := 0.0
			for k, value := range a.Data {
				diff := float64(b.Data[k] - value)
				distance += diff * diff
			}
			distance = math.Sqrt(distance)
			graph.Link(uint64(i), uint64(j), distance)
		}
	}
	type Rank struct {
		Rank  float64
		Index uint64
	}
	ranks := make([]Rank, 150)
	graph.Rank(0.85, 0.000001, func(node uint64, rank float64) {
		ranks[node] = Rank{
			Rank:  rank,
			Index: node,
		}
	})
	sort.Slice(ranks, func(i, j int) bool {
		return ranks[i].Rank < ranks[j].Rank
	})
	for _, v := range ranks {
		fmt.Println(v.Index, v.Rank)
	}*/
}
