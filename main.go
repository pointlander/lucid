// Copyright RNN The Lucid Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
)

const (
	// Width is the width of the model
	Width = 4
)

// Inputs is the input to the first layer
type Inputs struct {
	Inputs [4]float32
	Epoch  int64
}

func neuron1(seed int64, id int, in <-chan Inputs, out [3]chan<- Input) {
	rng := rand.New(rand.NewSource(seed))
	weights := NewMatrix(0, 4, 1)
	bias := NewMatrix(0, 1, 1)
	factor := math.Sqrt(2.0 / float64(4))
	for i := 0; i < 4; i++ {
		weights.Data = append(weights.Data, float32(rng.NormFloat64()*factor))
	}
	bias.Data = append(bias.Data, 0)
	for input := range in {
		i := NewMatrix(0, 4, 1)
		i.Data = append(i.Data, input.Inputs[:]...)
		output := Step(Add(MulT(weights, i), bias))
		for j := range out {
			out[j] <- Input{
				Input: output.Data[0],
				Epoch: input.Epoch,
			}
		}
	}
}

// Input is the input into the second layer
type Input struct {
	Input float32
	Epoch int64
}

func neuron2(seed int64, id int, in [Width]<-chan Input, out chan<- float32) {
	rng := rand.New(rand.NewSource(seed))
	weights := NewMatrix(0, Width, 1)
	bias := NewMatrix(0, 1, 1)
	factor := math.Sqrt(2.0 / float64(Width))
	for i := 0; i < Width; i++ {
		weights.Data = append(weights.Data, float32(rng.NormFloat64()*factor))
	}
	bias.Data = append(bias.Data, 0)
	for {
		i := NewMatrix(0, Width, 1)
		for _, in := range in {
			i.Data = append(i.Data, (<-in).Input)
		}
		output := Add(MulT(weights, i), bias)
		out <- output.Data[0]
	}
}

func main() {
	rng := rand.New(rand.NewSource(1))

	input := make([]chan Inputs, Width)
	output := make([]chan Input, Width*3)
	for i := range input {
		input[i] = make(chan Inputs, 8)
	}
	for i := range output {
		output[i] = make(chan Input, 8)
	}
	top := make([]chan float32, 3)
	for i := range top {
		top[i] = make(chan float32, 8)
	}

	id := 0
	for i := 0; i < Width; i++ {
		go neuron1(rng.Int63(), id, input[i], [3]chan<- Input{output[i], output[i+Width], output[i+2*Width]})
		id++
	}
	for i := 0; i < 3; i++ {
		input := [Width]<-chan Input{}
		for j := range input {
			input[j] = output[Width*i+j]
		}
		go neuron2(rng.Int63(), id, input, top[i])
		id++
	}

	for epoch := 1; epoch < 256; epoch++ {
		in := Inputs{
			Epoch: int64(epoch),
		}
		for i := range in.Inputs {
			in.Inputs[i] = rng.Float32()
		}
		for i := range input {
			input[i] <- in
		}
		fmt.Println(epoch)
		for _, t := range top {
			fmt.Println(<-t)
		}
	}
}
