// Copyright RNN The Lucid Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/signal"
	"sync"
	"syscall"

	"github.com/pointlander/datum/iris"
)

const (
	// Width is the width of the model
	Width = 16
	// Window is the size of the window
	Window = 16
)

// Inputs is the input to the first layer
type Inputs struct {
	Inputs [][4]float32
	Labels []int
	Epoch  int64
}

func neuron1(seed int64, id int, in <-chan Inputs, out [3]chan<- Input, done chan []Loss,
	fini chan bool, dump chan Multi) {
	//rng := rand.New(rand.NewSource(seed))
	lock := sync.Mutex{}
	lock.Lock()
	multi := NewMulti(5)
	lock.Unlock()
	go func() {
		<-fini
		lock.Lock()
		dump <- multi
		lock.Unlock()
	}()
	for input := range in {
		seeds := rand.New(rand.NewSource(input.Epoch))
	sample:
		for {
			select {
			case losses := <-done:
				vars := make([][]float32, 4+1)
				for i := range vars {
					vars[i] = make([]float32, Window)
				}
				for j := 0; j < Window; j++ {
					samples := multi.Sample(rand.New(rand.NewSource(losses[j].Epoch + int64(id))))
					for k := 0; k < 4+1; k++ {
						vars[k][j] = samples[k]
					}
				}
				lock.Lock()
				multi = Factor(vars, false)
				lock.Unlock()
				break sample
			default:
				epoch := seeds.Int63()
				weights := NewMatrix(0, 4, 1)
				bias := NewMatrix(0, 1, 1)
				samples := multi.Sample(rand.New(rand.NewSource(epoch + int64(id))))
				index := 0
				for i := 0; i < 4; i++ {
					weights.Data = append(weights.Data, samples[index])
					index++
				}
				bias.Data = append(bias.Data, samples[index])
				o := Input{
					Input:  make([]float32, 3),
					Labels: input.Labels,
					Epoch:  epoch,
				}
				for j := range input.Inputs {
					i := NewMatrix(0, 4, 1)
					i.Data = append(i.Data, input.Inputs[j][:]...)
					output := Step(Add(MulT(weights, i), bias))
					o.Input[j] = output.Data[0]
				}
				for j := range out {
					out[j] <- o
				}
			}
		}
	}
	dump <- multi
}

// Input is the input into the second layer
type Input struct {
	Input  []float32
	Labels []int
	Epoch  int64
}

func neuron2(seed int64, id int, in [Width]<-chan Input, out chan<- Input, done chan []Loss,
	fini chan bool, dump chan Multi) {
	//rng := rand.New(rand.NewSource(seed))
	lock := sync.Mutex{}
	lock.Lock()
	multi := NewMulti(Width + 1)
	lock.Unlock()
	go func() {
		<-fini
		lock.Lock()
		dump <- multi
		lock.Unlock()
	}()
	for {
		select {
		case losses := <-done:
			vars := make([][]float32, Width+1)
			for i := range vars {
				vars[i] = make([]float32, Window)
			}
			for j := 0; j < Window; j++ {
				samples := multi.Sample(rand.New(rand.NewSource(losses[j].Epoch + int64(id))))
				for k := 0; k < Width+1; k++ {
					vars[k][j] = samples[k]
				}
			}
			lock.Lock()
			multi = Factor(vars, false)
			lock.Unlock()
		default:
			inputs := make([]Matrix, 3)
			for i := range inputs {
				inputs[i] = NewMatrix(0, Width, 1)
			}
			labels := []int{}
			var epoch = int64(0)
			for _, in := range in {
				select {
				case value := <-in:
					labels = value.Labels
					epoch = value.Epoch
					for k, v := range value.Input {
						inputs[k].Data = append(inputs[k].Data, v)
					}
				}
			}
			weights := NewMatrix(0, Width, 1)
			bias := NewMatrix(0, 1, 1)
			samples := multi.Sample(rand.New(rand.NewSource(epoch + int64(id))))
			index := 0
			for i := 0; i < Width; i++ {
				weights.Data = append(weights.Data, samples[index])
				index++
			}
			bias.Data = append(bias.Data, samples[index])
			o := Input{
				Input:  make([]float32, 3),
				Labels: labels,
				Epoch:  epoch,
			}
			for i := range inputs {
				output := Add(MulT(weights, inputs[i]), bias)
				o.Input[i] = output.Data[0]
			}
			out <- o
		}
	}
}

// Loss is a neural network output loss
type Loss struct {
	Loss  float32
	Epoch int64
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
	top := make([]chan Input, 3)
	for i := range top {
		top[i] = make(chan Input, 8)
	}
	done := make([]chan []Loss, Width)
	for i := range done {
		done[i] = make(chan []Loss, 8)
	}
	done1 := make([]chan []Loss, 3)
	for i := range done1 {
		done1[i] = make(chan []Loss, 8)
	}
	fini := make([]chan bool, Width)
	for i := range fini {
		fini[i] = make(chan bool, 8)
	}
	fini1 := make([]chan bool, 3)
	for i := range fini1 {
		fini1[i] = make(chan bool, 8)
	}
	dump := make([]chan Multi, Width)
	for i := range dump {
		dump[i] = make(chan Multi, 8)
	}
	dump1 := make([]chan Multi, 3)
	for i := range dump1 {
		dump1[i] = make(chan Multi, 8)
	}

	id := 0
	for i := 0; i < Width; i++ {
		go neuron1(rng.Int63(), id, input[i], [3]chan<- Input{output[i], output[i+Width],
			output[i+2*Width]}, done[i], fini[i], dump[i])
		id++
	}
	for i := 0; i < 3; i++ {
		input := [Width]<-chan Input{}
		for j := range input {
			input[j] = output[Width*i+j]
		}
		go neuron2(rng.Int63(), id, input, top[i], done1[i], fini1[i], dump1[i])
		id++
	}

	data, err := iris.Load()
	if err != nil {
		panic(err)
	}

	for _, value := range data.Fisher {
		sum := 0.0
		for _, v := range value.Measures {
			sum += v * v
		}
		length := math.Sqrt(sum)
		for i := range value.Measures {
			value.Measures[i] /= length
		}
	}

	losses := make([]Loss, Window)
	for i := range losses {
		losses[i].Loss = math.MaxFloat32
	}
	count := 0
	c, work := make(chan os.Signal), true
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-c
		work = false
	}()
	for epoch := 1; epoch < 256 && work; epoch++ {
		in := Inputs{
			Epoch:  int64(rng.Int31()),
			Labels: make([]int, 3),
			Inputs: make([][4]float32, 3),
		}
		indexes := [3]int{rng.Intn(50), 50 + rng.Intn(50), 100 + rng.Intn(50)}
		for i := range in.Inputs {
			for j := range in.Inputs[i] {
				in.Inputs[i][j] = float32(data.Fisher[indexes[i]].Measures[j])
			}
			in.Labels[i] = iris.Labels[data.Fisher[indexes[i]].Label]
		}
		for i := range input {
			input[i] <- in
		}
		fmt.Println(epoch)

	search:
		for work {
			outputs := make([]Matrix, 3)
			for i := range outputs {
				outputs[i] = NewMatrix(0, 3, 1)
			}
			labels := []int{}
			loss := Loss{}
			for _, in := range top {
				value := <-in
				labels = value.Labels
				loss.Epoch = value.Epoch
				for k, v := range value.Input {
					outputs[k].Data = append(outputs[k].Data, v)
				}
			}
			for i := range outputs {
				output := TaylorSoftmax(outputs[i])
				expected := make([]float32, 3)
				expected[labels[i]] = 1

				for i, v := range output.Data {
					diff := float32(v) - expected[i]
					loss.Loss += diff * diff
				}
			}
			index := 0
			for index < len(losses) {
				if loss.Loss < losses[index].Loss {
					count++
					loss, losses[index] = losses[index], loss
					if index == 0 {
						fmt.Println(losses[0])
						if count > Window {
							for _, done := range done {
								done <- losses
							}
							for _, done := range done1 {
								done <- losses
							}
							break search
						}
					}
					index++
					for index < len(losses) {
						loss, losses[index] = losses[index], loss
						index++
					}
				}
				index++
			}
		}
	}

	id = 0
	weights := NewMatrix(0, 4, Width)
	bias := NewMatrix(0, 1, Width)
	for i, value := range dump {
		fini[i] <- true
		multi := <-value
		samples := multi.U //multi.Sample(rand.New(rand.NewSource(losses[0].Epoch + int64(id))))
		index := 0
		for i := 0; i < 4; i++ {
			weights.Data = append(weights.Data, samples[index])
			index++
		}
		bias.Data = append(bias.Data, samples[index])
		id++
	}
	weights1 := NewMatrix(0, Width, 3)
	bias1 := NewMatrix(0, 1, 3)
	for i, value := range dump1 {
		fini1[i] <- true
		multi := <-value
		samples := multi.U //multi.Sample(rand.New(rand.NewSource(losses[0].Epoch + int64(id))))
		index := 0
		for i := 0; i < Width; i++ {
			weights1.Data = append(weights1.Data, samples[index])
			index++
		}
		bias1.Data = append(bias1.Data, samples[index])
		id++
	}

	correct := 0
	loss := 0.0
	for _, fisher := range data.Fisher {
		input := NewMatrix(0, 4, 1)
		for _, v := range fisher.Measures {
			input.Data = append(input.Data, float32(v))
		}

		output := Step(Add(MulT(weights, input), bias))
		output = TaylorSoftmax(Add(MulT(weights1, output), bias1))
		max, index := float32(0.0), 0
		for i, value := range output.Data {
			v := float32(value)
			if v > max {
				max, index = v, i
			}
		}
		fmt.Println(index, max)
		if index == iris.Labels[fisher.Label] {
			correct++
		}

		expected := make([]float32, 3)
		expected[iris.Labels[fisher.Label]] = 1

		for i, v := range output.Data {
			diff := float64(float32(v) - expected[i])
			loss += diff * diff
		}
	}
	fmt.Println("correct", correct, float64(correct)/150)
	fmt.Println("loss", loss)
}
