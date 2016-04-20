package dnn

import (
	"fmt"
)

type Layer struct {
	Neurons []Neuron
}

type Net struct {
	Layers []Layer
}

func Create(topology []uint) (*Net, error) {
	layerCount := len(topology)
	if layerCount < 3 {
		return nil, fmt.Errorf("You need more than 2 layers")
	}
	
	layers := make([]Layer, layerCount)

	for l := range layers {
		neurons := make([]Neuron, topology[l] + 1)
		connCount := uint(0)
		if l < layerCount - 1 {
			connCount = topology[l + 1]
		}
		for n := range neurons {
			neurons[n].Init(connCount, uint(n))
		}
		layers[l].Neurons = neurons
	}
	
	return &Net{
		Layers: layers,
	}, nil
}

func (net *Net) feed(input []float64) (*Layer, error) {
	layerCount := len(net.Layers)
	if layerCount < 3 {
		return nil, fmt.Errorf("Unexpected layer count")
	}

	first := &net.Layers[0]
	if len(input) != len(first.Neurons) - 1 {
		return nil, fmt.Errorf("input size different than 1st layer size")
	}

	for i := range input {
		first.Neurons[i].Output = input[i]
	}

	for l := 1; l < len(net.Layers); l += 1 {
		prev := &net.Layers[l - 1]
		curr := &net.Layers[l]
		
		for n := 0; n < len(curr.Neurons) - 1; n += 1 {
			neuron := &curr.Neurons[n]
			neuron.Feed(prev)
		}
	}

	return &net.Layers[layerCount - 1], nil
}

func (net *Net) Train(input, target []float64, rate float64) (float64, error) {
	last, err := net.feed(input)
	if err != nil {
		return -1, err
	}

	if len(target) != len(last.Neurons) - 1 {
		return -1, fmt.Errorf("target size different than last layer size")
	}
	
	dist := 0.0
	for n := 0; n < len(last.Neurons) - 1; n += 1 {
		neuron := &last.Neurons[n]
		delta := target[n] - neuron.Output
		dist += delta * delta
		neuron.updateGradient(target[n])
	}

	// back propagation
	for l := len(net.Layers) - 2; l > 0; l -= 1 {
		layer := &net.Layers[l]
		next := &net.Layers[l + 1]

		for n := range layer.Neurons {
			neuron := &layer.Neurons[n]
			neuron.deriveGradients(next)
		}
	}

	// update weights
	for l := len(net.Layers) - 2; l > 0; l -= 1 {
		layer := &net.Layers[l]
		prev := &net.Layers[l - 1]

		for n := 0; n < len(layer.Neurons) - 1; n += 1 {
			neuron := &layer.Neurons[n]
			neuron.updateWeight(prev, rate)
		}
  	}
	
	return dist, nil
}

func (net *Net) Predict(input []float64) ([]float64, error) {
	last, err := net.feed(input)
	if err != nil {
		return nil, err
	}

	sz := len(last.Neurons) - 1
	result := make([]float64, sz)
	for n := 0; n < sz; n += 1 {
		neuron := &last.Neurons[n]
		result[n] = neuron.Output
	}
	
	return result, nil
}
