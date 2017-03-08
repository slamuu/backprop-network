#!/usr/bin/env node --harmony
'use strict';

const _ = require('lodash');
const math = require('mathjs');
const Network = require('./src/network');
const opts = { precision: 3 };

/**
 * Formats a number accounting for exponential
 *
 * @param {Number} value Number value
 * @return {String} Formatted number
 */
function format(value) {
  let num = _.toString(value.toFixed(2));

  if (Math.abs(num) < 1) {
    const exp = parseInt(num.split('e-')[1]);

    if (exp) {
      num *= Math.pow(10, exp - 1);
      num = `0.${_.repeat('0', exp - 1)}${num.substring(2)}`;
    }
  }
  else {
    const exp = parseInt(num.split('+')[1]);

    if (exp > 20) {
      exp -= 20;
      num /= Math.pow(10, exp);
      num += _.repeat('0', exp);
    }
  }

  return num;
}

/**
 * Formats a matrix accounting for exponential
 *
 * @param {Array} value Matrix value
 * @return {String} Formatted matrix
 */
function formatMatrix(value) {
  const matrix = value.valueOf();

  if (matrix[0].length === 1 && matrix[0].length === 1) {
    return format(matrix[0][0]);
  }
  else {
    const casted = _.map(matrix, (row) => {
      const content = _.map(row, format).join(', ');

      return `  [ ${content} ],`;
    });

    casted.unshift('[');
    casted.push(']');

    return casted.join('\n');
  }
}

/**
 * Displays the weights and bias
 *
 * @param {Network} network Network instance
 */
function display(network) {
  const { w1, w2, b1, b2 } = network;

  console.log(`Hidden weights ${formatMatrix(w1)}`);
  console.log(`Output weights ${formatMatrix(w2)}`);

  console.log(`Hidden bias ${formatMatrix(b1)}`);
  console.log(`Output bias ${formatMatrix(b2)}`);
}

const XOR = new Network([2, 5, 1]);

const examples = [
  { input: [[0], [0]], expect: [[0]] },
  { input: [[0], [1]], expect: [[1]] },
  { input: [[1], [0]], expect: [[1]] },
  { input: [[1], [1]], expect: [[0]] },
];

display(XOR)
console.log('----------');

XOR.train(examples);

console.log('Input 0, 0 -> 0 got', formatMatrix(XOR.predict([[0], [0]])));
console.log('Input 0, 1 -> 1 got', formatMatrix(XOR.predict([[0], [1]])));
console.log('Input 1, 0 -> 1 got', formatMatrix(XOR.predict([[1], [0]])));
console.log('Input 1, 1 -> 0 got', formatMatrix(XOR.predict([[1], [1]])));

console.log('----------');
display(XOR)