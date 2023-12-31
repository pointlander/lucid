// Copyright 2023 The Lucid Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"math/rand"

	"github.com/pointlander/datum/iris"
	"github.com/pointlander/lucid/mark1"
	"github.com/pointlander/lucid/mark2"
	"github.com/pointlander/lucid/mark3"
	"github.com/pointlander/lucid/mark4"
	"github.com/pointlander/lucid/mark5"
	. "github.com/pointlander/lucid/matrix"
)

var (
	// FlagMark1 is the mark1 model mode
	FlagMark1 = flag.Bool("mark1", false, "mark1 model")
	// FlagMark2 is the mark2 model
	FlagMark2 = flag.Bool("mark2", false, "mark2 model")
	// FlagMark3 is the mark3 model
	FlagMark3 = flag.Bool("mark3", false, "mark3 model")
	// FlagMark4 is the mark4 model
	FlagMark4 = flag.Bool("mark4", false, "mark4 model")
	// FlagMark4a is the mark4a model
	FlagMark4a = flag.Bool("mark4a", false, "mark4a model")
	// FlagMark5 is the mark5 model
	FlagMark5 = flag.Bool("mark5", false, "mark5 model")
)

func main() {
	flag.Parse()

	if *FlagMark1 {
		mark1.Mark1()
		return
	} else if *FlagMark2 {
		mark2.Mark2()
		return
	} else if *FlagMark3 {
		mark3.Mark3()
		return
	} else if *FlagMark4 {
		mark4.Mark4()
		return
	} else if *FlagMark4a {
		mark4.Mark4a()
		return
	} else if *FlagMark5 {
		mark5.Mark5()
		return
	}

	rng := rand.New(rand.NewSource(1))

	data, err := iris.Load()
	if err != nil {
		panic(err)
	}

	vars := make([][]float32, 4)
	for i := range vars {
		for j := range data.Fisher {
			vars[i] = append(vars[i], float32(data.Fisher[j].Measures[i]))
		}
	}
	multi := Factor(vars, true)

	for i := 0; i < 8; i++ {
		sample := multi.Sample(rng)
		fmt.Println(sample)
	}
	fmt.Println(multi.A.Data)
}
