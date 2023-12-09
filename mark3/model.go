// Copyright 2023 The Lucid Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mark3

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	. "github.com/pointlander/lucid/matrix"
)

func Mark3() {
	rng := rand.New(rand.NewSource(1))
	a := NewMatrix(0, 8, 8)
	for i := 0; i < 8*8; i++ {
		a.Data = append(a.Data, float32(rng.NormFloat64()))
	}
	type Sample struct {
		A    Matrix
		AA   Matrix
		Loss float32
	}
	min := float32(math.MaxFloat32)
	for {
		samples := []Sample{}
		for i := 0; i < 128; i++ {
			b := NewMatrix(0, 8, 8)
			for i := 0; i < 8*8; i++ {
				b.Data = append(b.Data, float32(rng.NormFloat64()))
			}
			ab := MulT(a, b)
			ab = Normalize(ab)
			abab := Sub(a, MulT(ab, T(ab)))
			sum := float32(0.0)
			for _, value := range abab.Data {
				sum += value * value
			}
			samples = append(samples, Sample{
				A:    ab,
				AA:   abab,
				Loss: float32(math.Sqrt(float64(sum))),
			})
		}
		sort.Slice(samples, func(i, j int) bool {
			return samples[i].Loss < samples[j].Loss
		})
		if samples[0].Loss < min {
			min = samples[0].Loss
			a = samples[0].A
			fmt.Println(samples[0].Loss)
		}
	}
}
