package cs224n.deep;

import java.lang.*;
import java.util.*;
import java.text.*;

import org.ejml.data.*;
import org.ejml.simple.*;


public class WindowModel {

	/* Weights except for the final layer */
	protected SimpleMatrix [] W;
	
	/* Bias */
	protected SimpleMatrix [] b;
	
	/* Final layer of MAXENT weights */
	protected SimpleMatrix U;
	
	
	/* Context window size */
	protected int windowSize;
	
	/* Word vector dimensions */
	protected int wordSize;
	
	/* Number of hidden layers */
	protected int numOfHiddenLayer;
	
	/* Size of each hidden layers */
	protected int[] hiddenSize;
	
	/* Learning rate */
	protected double alpha;
	
	
	/**
	 * Shallow architecture constructor
	 * 
	 * @param windowSize
	 * @param hiddenSize
	 * @param lr
	 */
	public WindowModel(int windowSize, int hiddenSize, double lr) {
		this.windowSize = windowSize;
		this.wordSize = 50;
		this.numOfHiddenLayer = 1;
		this.hiddenSize = new int[numOfHiddenLayer];
		this.alpha = lr;
	}
	
	
	/**
	 * Deep architecture constructor
	 * 
	 * @param windowSize
	 * @param hiddenSize
	 * @param lr
	 */
	public WindowModel(int windowSize, int [] hiddenSize, double lr) {
		this.windowSize = windowSize;
		this.wordSize = 50;
		this.numOfHiddenLayer = hiddenSize.length;
		this.hiddenSize = hiddenSize;
		this.alpha = lr;
	}
	
	
	private static SimpleMatrix sigmoid(SimpleMatrix M) {
		int R = M.numRows(), C = M.numCols();
		SimpleMatrix ret = new SimpleMatrix(R, C);
		for (int i = 0; i < R; ++i) {
			for (int j = 0; j < C; ++j) {
				ret.set(i, j, 1.0 / (1.0 + Math.exp(-M.get(i, j))));
			}
		}
		return ret;
	}
	
	private static SimpleMatrix tanh(SimpleMatrix M) {
		int R = M.numRows(), C = M.numCols();
		SimpleMatrix ret = new SimpleMatrix(R, C);
		for (int i = 0; i < R; ++i) {
			for (int j = 0; j < C; ++j) {
				ret.set(i, j, Math.tanh(M.get(i, j)));
			}
		}
		return ret;
	}
	
	
	private static SimpleMatrix sigmoidDerivative(SimpleMatrix M) {
		int R = M.numRows(), C = M.numCols();
		SimpleMatrix ret = new SimpleMatrix(R, C);
		for (int i = 0; i < R; ++i) {
			for (int j = 0; j < C; ++j) {
				double val = Math.exp(-M.get(i, j));
				ret.set(i, j, val / ((1+val)*(1+val)));
			}
		}
		return ret;
	}
	
	
	private static SimpleMatrix tanhDerivative(SimpleMatrix M) {
		int R = M.numRows(), C = M.numCols();
		SimpleMatrix ret = new SimpleMatrix(R, C);
		for (int i = 0; i < R; ++i) {
			for (int j = 0; j < C; ++j) {
				ret.set(i, j, 1.0 - Math.tanh(M.get(i, j)) * Math.tanh(M.get(i, j)));
			}
		}
		return ret;
	}
	
	
	public double sgdFeedforward(SimpleMatrix window) {
		SimpleMatrix temp = window;
		for (int i = 0; i < numOfHiddenLayer; ++i) {
			temp = tanh(W[i].mult(temp).plus(b[i]));
		}
		temp = sigmoid(U.transpose().mult(temp).plus(b[numOfHiddenLayer+1]));
		return temp.get(0);
	}
	
	
	public SimpleMatrix batchFeedforward(SimpleMatrix windows) {
		SimpleMatrix temp = windows;
		return temp;
	}
	
	

	/**
	 * Initializes the weights randomly.
	 */
	public void initWeights() {
		// TODO
		// initialize with bias inside as the last column
		// W = SimpleMatrix...
		// U for the score
		// U = SimpleMatrix...
	}

	/**
	 * Simplest SGD training
	 */
	public void train(List<Datum> _trainData) {
		// TODO
	}

	public void test(List<Datum> testData) {
		// TODO
	}

}
