package dnn

import (
	"math"
)

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func SigmoidDerive(x float64) float64 {
	e := math.Exp(-x)
	return e / math.Pow(1 + e, 2)
}

type Neuron struct {
	Index uint
	Conns []Conn
	Output float64
	gradient float64
}

func (neuron *Neuron) Init(connCount, index uint) {
	conns := make([]Conn, connCount)
	for c := range conns {
		conns[c].Init()
	}
	neuron.Index = index
	neuron.Conns = conns
	neuron.Output = 0
	neuron.gradient = 0
}

func (neuron *Neuron) Feed(prevLayer *Layer) {
	sum := 0.0
	idx := neuron.Index
	for n := range prevLayer.Neurons {
		prevNeuron := &prevLayer.Neurons[n]
		sum += prevNeuron.Output * prevNeuron.Conns[idx].Weight
	}
	neuron.Output = Sigmoid(sum)
}

func (neuron *Neuron) updateGradient(target float64) {
	delta := target - neuron.Output
	neuron.gradient = delta * SigmoidDerive(neuron.Output)
}

func (neuron *Neuron) deriveGradients(nextLayer *Layer) {
	sum := 0.0
	for n := 0; n < len(nextLayer.Neurons) - 1; n += 1 {
		tmp := &nextLayer.Neurons[n]
		sum += neuron.Conns[n].Weight * tmp.gradient
	}

	neuron.gradient = sum * SigmoidDerive(neuron.Output)
}

func (neuron *Neuron) updateWeight(prevLayer *Layer, rate float64) {
	for n := range prevLayer.Neurons {
		tmp := &prevLayer.Neurons[n]
		conn := &tmp.Conns[neuron.Index]

		// conn.Delta is too big ... have to read more on this ...
		Delta := rate * tmp.Output * neuron.gradient + conn.Delta * .2
		conn.Delta = Delta
		conn.Weight += Delta
	}
}
