package dnn

import (
	"math/rand"
	"time"
)

type Conn struct {
	Weight, Delta float64
}

var crnd *rand.Rand

func init() {
	crnd = rand.New(rand.NewSource(time.Now().Unix()))
}

func (c* Conn) Init() {
	c.Weight = crnd.Float64() * 40 - 20
	c.Delta = 0
}
