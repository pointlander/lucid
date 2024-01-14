// Copyright 2023 The Lucid Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mark7

import (
	"fmt"
	"image/color"
	"math"
	"math/rand"

	"github.com/pointlander/datum/iris"
	. "github.com/pointlander/matrix"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

const (
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

// Iris is a iris data point
type Iris struct {
	iris.Iris
	I         int
	Embedding []float32
	Cluster   int
}

// Mark7 is the mark7 model
func Mark7() {
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
	in := NewMatrix(0, Embedding, len(flowers))
	for i := range flowers {
		in.Data = append(in.Data, flowers[i].Embedding...)
	}
	gmm := NewGMM()
	gmm.Clusters = Clusters
	//gmm.Window = 8
	//gmm.Epochs = 256
	out := gmm.GMM(in)
	for i, value := range out {
		flowers[i].Cluster = value
	}
	for i := range flowers {
		fmt.Println(flowers[i].Cluster, flowers[i].Label)
	}

	p := plot.New()
	if err != nil {
		panic(err)
	}

	p.Title.Text = "iris"
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
