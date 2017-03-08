'use strict';

const _ = require('lodash');
const math = require('mathjs');
const randgen = require('randgen');
const rnorm = randgen.rnorm;

/**
 * Sigmoid function
 *
 * @param {Number} x Number to sigmoid
 * @return {Number} Value from sigmoid
 */
function _sigmoid(x) {
  return 1 / (1 + Math.pow(Math.E, -x));
}

/**
 * Sigmoid prime function
 *
 * @param {Number} x Number to sigmoid prime
 * @return {Number} Value from sigmoid prime
 */
function _psigmoid(x) {
  return sigmoid(x) * (1 - sigmoid(x));
}

/**
 * Memoized scalar sigmoid and sigmoid prime function
 *
 * @type {Object}
 */
const scalar = {
  sigmoid: _.memoize(_sigmoid),
  psigmoid: _.memoize(_sigmoid),
};

/**
 * Transform matrix through sigmoid
 *
 * @param {Array} input Input matrix
 * @return {Array} Transformed matrix
 */
function sigmoid(input) {
  return math.matrix(input).map((val) => scalar.sigmoid(val));
}

/**
 * Transform matrix through sigmoid prime
 *
 * @param {Array} input Input matrix
 * @return {Array} Transformed matrix
 */
function psigmoid(input) {
  return math.matrix(input).map((val) => scalar.psigmoid(val));
}

module.exports = {
  sigmoid,
  psigmoid,
};
