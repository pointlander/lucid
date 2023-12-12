// Copyright 2023 The Lucid Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mark4

import (
	"fmt"
	"image"
	"image/color"
	"image/color/palette"
	"image/draw"
	"image/gif"
	"image/jpeg"
	"math"
	"math/rand"
	"os"
	"sort"

	. "github.com/pointlander/lucid/matrix"
)

const (
	// Window is the window size
	Window = 8
	// Samples is the number of samples
	Samples = 256
	// Size is the size of the filter
	Size = 64
	// Inputs is the number of inputs
	Inputs = Size
	// Outputs is the number of outputs
	Outputs = 8
)

// Random is a random variable
type Random struct {
	Mean   float32
	StdDev float32
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
	for i := range systems[:Window] {
		for j := range systems[i].Neurons {
			for k, value := range systems[i].Neurons[j].Data {
				statistics[j][k].Mean += value
			}
		}
	}
	for i := range statistics {
		for j := range statistics[i] {
			statistics[i][j].Mean /= Window
		}
	}
	for i := range systems[:Window] {
		for j := range systems[i].Neurons {
			for k, value := range systems[i].Neurons[j].Data {
				diff := statistics[j][k].Mean - value
				statistics[j][k].StdDev += diff * diff
			}
		}
	}
	for i := range statistics {
		for j := range statistics[i] {
			statistics[i][j].StdDev /= Window
			statistics[i][j].StdDev = float32(math.Sqrt(float64(statistics[i][j].StdDev)))
		}
	}
	return statistics
}

// Fire runs the network
func (n *Net) Fire(input Matrix) (float32, Matrix) {
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
			out := MulT(neurons[j], input)
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
			out := MulT(neurons[j], input)
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
			out := MulT(neurons[j], input)
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
	/*vector := NewMatrix(0, 3*n.Outputs, 1)
	vector.Data = append(vector.Data, systemsQ[0].Outputs.Data...)
	vector.Data = append(vector.Data, systemsK[0].Outputs.Data...)
	vector.Data = append(vector.Data, systemsV[0].Outputs.Data...)*/
	return systemsV[0].Entropy, systemsV[0].Outputs
}

// Mark4 is the mark4 model
func Mark4() {
	rng := rand.New(rand.NewSource(1))
	input, err := os.Open("test.jpg")
	if err != nil {
		panic(err)
	}
	defer input.Close()
	img, err := jpeg.Decode(input)
	if err != nil {
		panic(err)
	}
	size := img.Bounds().Size()
	out := image.NewRGBA(image.Rect(0, 0, size.X/Size, size.Y/Size))
	red := NewNet(1, Inputs, Outputs)
	green := NewNet(1, Inputs, Outputs)
	blue := NewNet(1, Inputs, Outputs)
	fmt.Println(size)
	type Coord struct {
		X int
		Y int
	}
	samples := make([]Coord, Size)
	for i := range samples {
		samples[i].X = rng.Intn(Size)
		samples[i].Y = rng.Intn(Size)
	}
	for i := 0; i < size.X; i += Size {
		for j := 0; j < size.Y; j += Size {
			rinput := NewMatrix(0, Inputs, 1)
			ginput := NewMatrix(0, Inputs, 1)
			binput := NewMatrix(0, Inputs, 1)
			for _, coord := range samples {
				original := img.At(i+coord.X, j+coord.Y)
				r, g, b, _ := original.RGBA()
				rinput.Data = append(rinput.Data, float32(r)/65535)
				ginput.Data = append(ginput.Data, float32(g)/65535)
				binput.Data = append(binput.Data, float32(b)/65535)
			}
			pixel := color.RGBA{
				A: 255,
			}
			_, value := red.Fire(rinput)
			for i, v := range value.Data {
				if v > 0 {
					pixel.R |= 1 << i
				}
			}
			_, value = green.Fire(ginput)
			for i, v := range value.Data {
				if v > 0 {
					pixel.G |= 1 << i
				}
			}
			_, value = blue.Fire(binput)
			for i, v := range value.Data {
				if v > 0 {
					pixel.B |= 1 << i
				}
			}
			out.Set(i/Size, j/Size, pixel)
			fmt.Println(i, j)
		}
	}
	output, err := os.Create("output.jpg")
	if err != nil {
		panic(err)
	}
	defer output.Close()
	err = jpeg.Encode(output, out, nil)
	if err != nil {
		panic(err)
	}
}

// Mark4a is the mark4a model
func Mark4a() {
	input, err := os.Open("test.jpg")
	if err != nil {
		panic(err)
	}
	defer input.Close()
	img, err := jpeg.Decode(input)
	if err != nil {
		panic(err)
	}
	size := img.Bounds().Size()

	net := NewNet(1, 3*8*8, 32)

	var images []*image.Paletted
	opts := gif.Options{
		NumColors: 256,
		Drawer:    draw.FloydSteinberg,
	}
	bounds := img.Bounds()
	x, y := 0, 0
	paletted := image.NewPaletted(bounds, palette.Plan9[:opts.NumColors])
	for x := 0; x < size.X; x++ {
		for y := 0; y < size.Y; y++ {
			paletted.SetColorIndex(x, y, 255)
		}
	}
	for i := 0; i < 256; i++ {
		fmt.Println(i)
		input := NewMatrix(0, 3*8*8, 1)
		for a := 0; a < 8; a++ {
			for b := 0; b < 8; b++ {
				pixel := img.At((x+a)%size.X, (y+b)%size.Y)
				r, g, b, _ := pixel.RGBA()
				y, cb, cr := color.RGBToYCbCr(uint8(r>>8), uint8(g>>8), uint8(b>>8))
				input.Data = append(input.Data, float32(y)/255)
				input.Data = append(input.Data, float32(cb)/255)
				input.Data = append(input.Data, float32(cr)/255)
			}
		}
		_, output := net.Fire(input)
		x, y = 0, 0
		for j := 0; j < 16; j++ {
			if output.Data[j] > 0 {
				x |= 1 << j
			}
		}
		for j := 0; j < 16; j++ {
			if output.Data[j+16] > 0 {
				y |= 1 << j
			}
		}
		for a := 0; a < 16; a++ {
			for b := 0; b < 16; b++ {
				paletted.SetColorIndex((a+x)%size.X, (b+y)%size.Y, 8)
			}
		}
		cp := image.NewPaletted(bounds, palette.Plan9[:opts.NumColors])
		copy(cp.Pix, paletted.Pix)
		images = append(images, cp)
	}
	animation := &gif.GIF{}
	for _, paletted := range images {
		animation.Image = append(animation.Image, paletted)
		animation.Delay = append(animation.Delay, 50)
	}

	f, _ := os.OpenFile("test.gif", os.O_WRONLY|os.O_CREATE, 0600)
	defer f.Close()
	gif.EncodeAll(f, animation)
}
