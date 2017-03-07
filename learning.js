class Synapse {
	constructor(parentNeuron, childNeuron) {
		this.w = Math.random() * 2 - 1;
		this.wDelta = 0;
		this.parent = parentNeuron;
		this.child = childNeuron;
		log(this.parent.id, this.child.id, 'weight', this.w);
		this.gradient = 0;
	}

	impulse(value) {
		this.child.inputImpulse(value * this.w);
	}

	backpropagateError(error) {
		this.parent.updateError(error * this.w);
	}

	gradientDescent(learnRate, momentum) {
		this.gradient = this.parent.output * this.child.error;
		this.wDelta = -learnRate * this.gradient + momentum * this.wDelta;
		this.w += this.wDelta;
		log(this.parent.id, this.child.id, 'gradient', this.gradient, 'new weight', this.w);
		this.child.gradientDescent(learnRate, momentum);
	}
}

class Neuron {
	constructor(activationFn, activationPrime, id) {
		this.id = id || 'N0';
		this.parentSynapses = [];
		// Output synapses
		this.synapses = [];
		this.sum = 0;
		this.error = 0;
		this.output = 1;
		this._activation = activationFn;
		this._activationP = activationPrime;
		this._errorSum = 0;
	}

	connect(neuron) {
		const syn = new Synapse(this, neuron);
		this.synapses.push(syn);
		neuron._connectParent(syn);
	}

	reset() {
		this.sum = 0;
		this.output = 1;
	}

	inputImpulse(value, activate=true) {
		this.sum += value;
		this.output = activate ? this._activation(this.sum) : this.sum;
	}

	// Update sum of synapses
	outputImpulse() {
		log(this.id, 'sum', this.sum, this.output);
		this.synapses.forEach(s => s.impulse(this.output));
		return this.output;
	}

	updateError(error) {
		this._errorSum += error;
		this.error = this._activationP(this.sum) * this._errorSum;
	}

	backpropagateError() {
		log(this.id, 'error', this._errorSum, this.error)
		this.parentSynapses.forEach(syn => syn.backpropagateError(this.error));
	}

	gradientDescent(learnRate, momentum) {
		this.synapses.forEach(syn => syn.gradientDescent(learnRate, momentum));
	}

	_connectParent(synapse) {
		this.parentSynapses.push(synapse);
	}
}

class Network {
	constructor(layers, activationFn, activationFnDerivative, bias=true) {
		this.bias = bias;
		this.activationFn = activationFn;
		this.activationFnDerivative = activationFnDerivative;
		this.network = this._constructNetwork(layers);
		this.inputNeurons = bias ? this.network[0].slice(0, -1)
			: this.nework[0];
		this.outputNeurons = this.network[this.network.length - 1];
		this.biasNeurons = bias ? this.network.slice(0, -1).map(layer => layer[layer.length - 1]) : [];
	}

	reset() {
		this._networkAction(n => n.reset());
	}

	sendInput(inputs) {
		var output = null;
		if(inputs.length === this.inputNeurons.length) {
			this._networkAction((neuron, layer, index) => {
				if(layer === 0 && index < inputs.length) {
					neuron.inputImpulse(inputs[index], false);
				}
				output = neuron.outputImpulse();
			});
		} else {
			console.error('Invalid number of inputs');
		}
		return output;
	}

	train(trainingData) {
		const error = this._backpropagate(trainingData);
		return error;
	}

	_backpropagate(trainingData) {
		const { learnRate, momentum, set } = trainingData;
		const errorRates = [];
		console.log(this.biasNeurons)
		set.forEach(item => {
			let actual = this.sendInput(item.inputs)
			let error = actual - item.ideal;
			this._networkAction((neuron, layer, index) => {
				if(this.outputNeurons.indexOf(neuron) === index && layer === this.network.length - 1) {
					neuron.updateError(error);
				}
				neuron.backpropagateError();
			}, true);
			this.inputNeurons.forEach(n => n.gradientDescent(learnRate, momentum));
			this.biasNeurons.forEach(n => n.gradientDescent(learnRate, momentum))
			errorRates.push(error);
			console.log((error.toFixed(2) * 100) + '%')
			this.reset();
		});
		return this._meanSquaredError(errorRates)
	}

	_averageError(errorRates) {
		let totalError = errorRates.reduce((acc, err) => acc + err, 0) / errorRates.length;
		return totalError;
	}

	_meanSquaredError(errorRates) {
		let totalError = errorRates.reduce((acc, err) => acc + Math.pow(err, 2), 0) / errorRates.length;
		return totalError;
	}

	_networkAction(actionFn, reverse=false) {
		if(reverse)
			for(let i = this.network.length - 1; i >= 0; i--) {
				for(let j = this.network[i].length - 1; j >= 0; j--) {
					actionFn(this.network[i][j], i, j);
				}
			}
		else
			this.network.forEach((layer, i) => layer.forEach((n, j) => actionFn(n, i, j)));
	}

	_constructNetwork(layers) {
		const network = layers.map((l, i) => {
			const group = Array.from(Array(l).keys(), (j) => new Neuron(this.activationFn, this.activationFnDerivative, this._id(i, j)));
			this.bias && i !== layers.length - 1 && group.push(new Neuron(this.activationFn, this.activationFnDerivative, this._id(i, group.length)));
			return group;
		});

		network.forEach((group, i) => {
			if(i < network.length - 1) {
				let nextGroup;
				if(i < network.length - 2 && this.bias)
					nextGroup = network[i + 1].slice(0, -1);
				else
					nextGroup = network[i + 1];
				group.forEach(n1 => {
					nextGroup.forEach(n2 => n1.connect(n2));
				});
			}
		});
		return network;
	}

	_id(i, j) {
		return `N${i}${j}`;
	}
}

const layers = [2, 3, 2, 1];
const activate = x => {
	return 1 / (1 + Math.exp(-x));
};
const activatePrime = x => {
	return activate(x) * (1 - activate(x));
};
const makeTrainingSet = size => {
	const samples = (function* (i) {
		while(1) {
			for(let i = 0; i < 2; i++) {
				for(let j = 0; j < 2; j++) {
					yield [i, j];
				}
			}
		}
	})();
	return Array.from(Array(size).keys(), i => {
		const inputs = samples.next().value;
		const ideal = inputs[0] ^ inputs[1];
		return { ideal, inputs };
	}); 
};
var debug = false;
const nets = Array.from(Array(1).keys(), () => new Network(layers, activate, activatePrime));
const set = makeTrainingSet(10000);
// set.forEach(i => console.log(i))
var best = {error: 14};
nets.forEach((net, i) => { 
	let error = net.train({
		learnRate: 0.01,
		momentum: 0.3,
		set
	});
	const out = net.sendInput([0, 0]).toFixed(2);
	console.log('#' + (Number(i) + 1) + '/' + nets.length, 'in: 0, 0', 'out:', out, (error * 100).toFixed(2) + '%')
	best = error < best.error ? {
		'in': '00',
		error: 0,
		out,
		net
	} : best;
});	
	console.log(`
	Error: ${best.error.toFixed(2) * 100}%
	Results:
		in: 0, 0
		out: ${best.net.sendInput([0, 0])}

		in: 0, 1
		out: ${best.net.sendInput([0, 1])}

		in: 1, 0
		out: ${best.net.sendInput([1, 0])}

		in: 1, 1
		out: ${best.net.sendInput([1, 1])}

	`)
function log() {
	if(debug)
		console.log.apply(console, arguments)
}