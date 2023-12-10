// Copyright 2023 The Lucid Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mark3

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/pointlander/datum/iris"
	. "github.com/pointlander/lucid/matrix"
)

func Mark3() {
	rng := rand.New(rand.NewSource(1))

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

	a := NewMatrix(0, 150, 150)
	for _, element := range data.Fisher {
		for _, value := range element.Measures {
			a.Data = append(a.Data, float32(value))
		}
		for i := 0; i < 150-4; i++ {
			a.Data = append(a.Data, 0 /*float32(rng.NormFloat64())*/)
		}
	}
	min := float32(math.MaxFloat32)
	for i := 0; i < 8*1024; i++ {
		b := NewMatrix(0, 150, 150)
		for j := 0; j < 150; j++ {
			for i := 0; i < 4; i++ {
				b.Data = append(b.Data, float32(rng.NormFloat64()))
			}
			for i := 0; i < 150-4; i++ {
				b.Data = append(b.Data, 0 /*float32(rng.NormFloat64())*/)
			}
		}
		ab := Normalize(MulT(a, b))
		abab := Sub(a, MulT(ab, T(ab)))
		sum := float32(0.0)
		for _, value := range abab.Data {
			sum += value * value
		}
		loss := float32(math.Sqrt(float64(sum)))
		if loss < min {
			min, a = loss, ab
			fmt.Println(loss)
		}
	}
	for i := 0; i < 150; i++ {
		fmt.Printf("%s ", data.Fisher[i].Label)
		for _, v := range a.Data[i*150 : i*150+64] {
			if v > 0 {
				fmt.Printf("1 ")
			} else {
				fmt.Printf("-1 ")
			}
		}
		fmt.Println()
	}
}
