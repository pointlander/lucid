// Copyright 2023 The Lucid Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"math/cmplx"
	"math/rand"

	"github.com/pointlander/datum/iris"
	"github.com/pointlander/lucid/mark1"
	"github.com/pointlander/lucid/mark2"
	"github.com/pointlander/lucid/mark3"
	"github.com/pointlander/lucid/mark4"
	"github.com/pointlander/lucid/mark5"
	"github.com/pointlander/lucid/mark6"
	"github.com/pointlander/lucid/mark7"

	. "github.com/pointlander/lucid/matrix"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
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
	// FlagMark6 is the mark6 model
	FlagMark6 = flag.Bool("mark6", false, "mark6 model")
	// FlagMark7 is the mark7 model
	FlagMark7 = flag.Bool("mark7", false, "mark7 model")
	// FlagMulti multi mode
	FlagMulti = flag.Bool("multi", false, "multi mode")
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
	} else if *FlagMark6 {
		mark6.Mark6()
		return
	} else if *FlagMark7 {
		mark7.Mark7()
		return
	}

	rng := rand.New(rand.NewSource(1))

	if *FlagMulti {
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

	x, y, z := [8]float64{}, [8]float64{}, [8]float64{}
	values := make(plotter.Values, 0, 1024)
	for i := 0; i < 8; i++ {
		x[i] = rng.NormFloat64()
		y[i] = rng.NormFloat64()
		z[i] = rng.NormFloat64()
	}
	for _, x := range x {
		for _, y := range y {
			for _, z := range z {
				values = append(values, x+y*z)
			}
		}
	}
	p := plot.New()
	p.Title.Text = "distribution"
	histogram, err := plotter.NewHist(values, 256)
	if err != nil {
		panic(err)
	}
	p.Add(histogram)
	err = p.Save(8*vg.Inch, 8*vg.Inch, "distribution.png")
	if err != nil {
		panic(err)
	}

	d := []complex128{
		cmplx.Exp(.1i),
		cmplx.Exp(.4i),
		cmplx.Exp(.5i),
	}
	sum := 0i
	for _, value := range d {
		fmt.Println(value, cmplx.Abs(value), cmplx.Phase(value))
		sum += value
	}
	fmt.Println(sum, cmplx.Abs(sum), cmplx.Phase(sum))
	for _, value := range d {
		value /= sum
		fmt.Println(value, cmplx.Abs(value), cmplx.Phase(value))
	}
}
