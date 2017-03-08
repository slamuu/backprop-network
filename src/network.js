'use strict';

const _ = require('lodash');
const math = require('mathjs');
const randgen = require('randgen');
const rnorm = randgen.rnorm;

const activation = require('./activation');
const sigmoid = activation.sigmoid;
const psigmoid = activation.psigmoid;

class Network {
  constructor(sizes, learning = 0.7, iteration = 10000) {
    this.sizes = sizes;
    this.learning = learning;
    this.iteration = Math.max(iteration, 1);

    this.w1 = math.matrix(_.range(sizes[1]).map(() => _.range(sizes[0]).map(() => rnorm(0, 1))));
    this.w2 = math.matrix(_.range(sizes[2]).map(() => _.range(sizes[1]).map(() => rnorm(0, 1))));

    this.b1 = math.matrix(_.range(sizes[1]).map(() => [rnorm(0, 1)]));
    this.b2 = math.matrix(_.range(sizes[2]).map(() => [rnorm(0, 1)]));
  }

  /**
   * Train network
   *
   * @param {Array<Object>} training List of training set
   */
  train(training) {
    for (let i = 1; i <= this.iteration; i++) {
      const results = _.map(training, (set) => this.forward(set.input, set.expect));

      _.forEach(results, (result) => {
        const input = result.input;
        const expect = result.expect;
        const hidden = result.hidden;
        const output = result.output;

        this.backward(input, expect, hidden, output);
      });

      _.forEach(results, (result) => {
        const hidden = result.hidden;
        const output = result.output;

        this.w1 = math.add(this.w1, math.multiply(hidden.delta.w, this.learning));
        this.w2 = math.add(this.w2, math.multiply(output.delta.w, this.learning));

        this.b1 = math.add(this.b1, math.multiply(hidden.delta.b, this.learning));
        this.b2 = math.add(this.b2, math.multiply(output.delta.b, this.learning));
      });
    }
  }

  /**
   * Predict the results from input
   *
   * @param {Array>} input Input matrix
   * @return {Array>} output Output matrix
   */
  predict(input) {
    const result = this.forward(input);

    return result.output.result;
  }

  /**
   * Feed forward input through network to get output
   *
   *   F(i * w + b)
   *
   *   Where:
   *     F is Activation function (In our case sigmoid)
   *     i is Input matrix from previous node
   *     w is weight matrix for current node
   *     b is Bias matrix for current node
   *
   *   Layers:
   *     (input) -> (hidden) -> (output)
   *
   * @param {Array} input Input matrix
   * @param {Array} expect Expected matrix output
   * @return {Object} Results from each layer
   */
  forward(input, expect) {
    const hidden = {};
    const output = {};

    hidden.output = math.add(math.multiply(this.w1, math.matrix(input)), this.b1);
    hidden.result = sigmoid(hidden.output);

    output.output = math.add(math.multiply(this.w2, hidden.result), this.b2);
    output.result = sigmoid(output.output);

    return { input, expect, hidden, output };
  }

  /**
   * Feed backward output with regards to input and expected result
   *
   * @param {Array} input Input matrix
   * @param {Array} expect Expected matrix output
   * @param {Object} hidden Hidden layer results
   * @param {Object} output Output layer results
   */
  backward(input, expect, hidden, output) {
    const error2 = math.subtract(math.matrix(expect), output.result);

    output.delta = {};
    output.delta.b = math.dotMultiply(error2, psigmoid(output.output));
    output.delta.w = math.multiply(output.delta.b, math.transpose(hidden.result));

    const error1 = math.multiply(math.transpose(this.w2), output.delta.b);

    hidden.delta = {};
    hidden.delta.b = math.dotMultiply(error1, psigmoid(hidden.output));
    hidden.delta.w = math.multiply(hidden.delta.b, math.transpose(math.matrix(input)));
  }
}

module.exports = Network;
