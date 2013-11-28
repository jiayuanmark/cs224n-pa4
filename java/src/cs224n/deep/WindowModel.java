package cs224n.deep;

import java.lang.*;
import java.util.*;
import java.text.*;

import org.ejml.data.*;
import org.ejml.simple.*;


public class WindowModel {

	/* Word vectors */
	protected SimpleMatrix L;
	
	/* Weights except for the final layer */
	protected SimpleMatrix [] W;
		
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
		this.W = new SimpleMatrix[numOfHiddenLayer];
		this.hiddenSize = new int[numOfHiddenLayer];
		this.hiddenSize[0] = hiddenSize;
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
		this.W = new SimpleMatrix[numOfHiddenLayer];
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
	
	
	/**
	 * Batch feed-forward function
	 * 
	 * @param windows
	 * @return multiple feed forward function value
	 */
	public SimpleMatrix batchFeedforward(SimpleMatrix windows) {
		SimpleMatrix temp = windows;
		for (int i = 0; i < numOfHiddenLayer; ++i) {
			temp = tanh(W[i].mult(MatlabAPI.horzconcat(temp)));
		}
		temp = sigmoid(U.transpose().mult(MatlabAPI.horzconcat(temp)));
		return temp;
	}
	
	
	/**
	 * Initializes the weights randomly.
	 */
	public void initWeights(SimpleMatrix L) {
		this.L = L;
		for (int i = 0; i < numOfHiddenLayer; ++i) {
			
		}
	}

	/**
	 * Simplest SGD training
	 */
	public void train(List<Datum> trainData) {
		// TODO
	}

	public void test(List<Datum> testData) {
		// TODO
	}

}
