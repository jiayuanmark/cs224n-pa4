package cs224n.deep;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.HashMap;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;


public class WindowModel {
	
	/* Unseen word placeholder */
	public static final String UNKNOWN = "UUUNKKK";
	
	/* Cut-off point */
	public static final double cutoff = 0.40;
	
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
	
	/* Regularization constant */
	protected double C;
	
	/**
	 * Shallow architecture constructor
	 * 
	 * @param windowSize: context window size
	 * @param hiddenSize: hidden layer size
	 * @param lr: learning rate
	 * @param reg: regularization parameter
	 */
	public WindowModel(int windowSize, int hiddenSize, double lr, double reg) {
		this.windowSize = windowSize;
		this.wordSize = 50;
		this.numOfHiddenLayer = 1;
		this.W = new SimpleMatrix[numOfHiddenLayer];
		this.hiddenSize = new int[numOfHiddenLayer];
		this.hiddenSize[0] = hiddenSize;
		this.alpha = lr;
		this.C = reg;
	}
	
	
	/**
	 * Deep architecture constructor
	 * 
	 * @param windowSize: context window size
	 * @param hiddenSize: hidden layer size
	 * @param lr: learning rate
	 * @param reg: regularization parameter
	 */
	public WindowModel(int windowSize, int [] hiddenSize, double lr, double reg) {
		this.windowSize = windowSize;
		this.wordSize = 50;
		this.numOfHiddenLayer = hiddenSize.length;
		this.W = new SimpleMatrix[numOfHiddenLayer];
		this.hiddenSize = hiddenSize;
		this.alpha = lr;
		this.C = reg;
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
	
	
	
	protected List<List<Integer>> makeInputWindows(List<Datum> data) {
		int radius = windowSize / 2;
		HashMap<String, Integer> dict = FeatureFactory.getDictionary();
		
		List<List<Integer>> ret = new ArrayList<List<Integer>>();
		
		for (int i = 0; i < data.size(); ++i) {
			Datum instance = data.get(i);
			LinkedList<String> window = new LinkedList<String>();
			window.add(instance.word);
			
			// Expand window
			boolean left = instance.word.equals("<s>");
			boolean right = instance.word.equals("</s>");
			
			// Expand left
			int ll = i-1;
			for (int r = 0; r < radius; ++r) {
				if (left) window.addFirst("<s>");
				else if (ll < 0) window.addFirst("<s>");
				else {
					window.addFirst(data.get(ll).word);
					left = data.get(ll).word.equals("<s>");
				}
				--ll;
			}
			
			// Expand right
			int rr = i+1;
			for (int r = 0; r < radius; ++r) {
				if (right) window.addLast("</s>");
				else if (rr >= data.size()) window.addLast("</s>");
				else {
					window.addLast(data.get(rr).word);
					right = data.get(rr).word.equals("</s>");
				}
				++rr;
			}
			
			// String to index
			ArrayList<Integer> windowIndices = new ArrayList<Integer>();
			for (String word : window) {
				if (dict.containsKey(word)) windowIndices.add(dict.get(word));
				else windowIndices.add(dict.get(UNKNOWN));
			}
			ret.add(windowIndices);
		}
		return ret;
	}
	
	
	protected List<Double> makeLabels(List<Datum> data) {
		List<Double> labels = new ArrayList<Double>();
		for (int i = 0; i < data.size(); ++i) {
			if (data.get(i).label.equals("PERSON")) {
				labels.add(1.0);
			}
			else labels.add(0.0);
		}
		return labels;
	}
	
	
	protected SimpleMatrix makeInputVector(List<Integer> win) {
		SimpleMatrix vec = new SimpleMatrix(wordSize * windowSize, 1);
		for (int w = 0; w < win.size(); ++w) {
			vec.insertIntoThis(w * wordSize, 0, L.extractVector(false, win.get(w)));
		}
		return vec;
	}
	
	
	protected void evaluateStatistics(List<List<Integer>> Data, List<Double> Label) {
		int numData = Data.size();
		int truePositive = 0, falsePositive = 0, falseNegative = 0, trueNegative = 0;
		for (int i = 0; i < numData; ++i) {
			SimpleMatrix input = makeInputVector(Data.get(i));
			SimpleMatrix response = batchFeedforward(input);
			
			int result = 0;
			if (response.get(0) > cutoff) result = 1;
			int answer = Label.get(i).compareTo(1.0) == 0 ? 1 : 0;
			
			if (result == answer && answer == 1) {
				++truePositive;
			} else if (result == answer && result == 0) {
				++trueNegative;
			} else if (result != answer && result == 1) {
				++falsePositive;
			} else if (result != answer && answer == 1) {
				++falseNegative;
			}
		}
		
		System.out.println("-------------------------------");
		System.out.println("Dataset size: " + numData);
		System.out.println("PERSON Precision: " + (double)truePositive / ((double)(truePositive+falsePositive)));
		System.out.println("PERSON Recall: " + (double)truePositive / ((double)(truePositive+falseNegative)));
		System.out.println("NON-PERSON Precision: " + (double)trueNegative / ((double)(trueNegative+falseNegative)));
		System.out.println("NON-PERSON Recall: " + (double)trueNegative / ((double)(trueNegative+falsePositive)));
		System.out.println("-------------------------------");
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
		temp = sigmoid(U.mult(MatlabAPI.horzconcat(temp)));
		return temp;
	}
	
	/**
	 * Regularized cost function
	 * 
	 * @param X
	 * @param L 
	 * @return cost function value
	 */
	public double costFunction(SimpleMatrix X, SimpleMatrix L) {
		SimpleMatrix h = batchFeedforward(X);
		int M = X.numCols();
		
		double val = 0.0;
		// Binary cross entropy
		for (int i = 0; i < M; ++i) {
			val -= ( L.get(0, i) * Math.log(h.get(0, i)) + (1 - L.get(0, i)) * Math.log(1 - h.get(0, i)));
		}
		
		val /= M;
		
		// Regularization without bias
		for (int i = 0; i < W.length; ++i) {
			SimpleMatrix nobiasW = W[i].extractMatrix(0, W[i].numRows(), 1, W[i].numCols());
			val += C / (2 * M) * nobiasW.elementMult(nobiasW).elementSum();
		}
		SimpleMatrix nobiasU = U.extractMatrix(0, U.numRows(), 1, U.numCols());
		val += C / (2 * M) * nobiasU.elementMult(nobiasU).elementSum();
		
		return val;
	}
	
	
	
	
	/**
	 * Initializes the weights randomly.
	 */
	public void initWeights() {
		Random rand = new Random();
		L = FeatureFactory.getWordVectors();
		int fanIn = windowSize * wordSize;
		double epsilon;
		
		for (int i = 0; i < numOfHiddenLayer; ++i) {
			// Initialize hidden weights to random numbers
			epsilon = Math.sqrt(6.0) / Math.sqrt(hiddenSize[i] + fanIn); 
			W[i] = SimpleMatrix.random(hiddenSize[i], fanIn+1, -epsilon, epsilon, rand);
			
			// Initialize bias terms to zeros
			double [] zeros = new double[hiddenSize[i]];
			Arrays.fill(zeros, 0.0);
			W[i].setColumn(0, 0, zeros);
			
			// Record next fan-out
			fanIn = hiddenSize[i];
		}
		
		// Final layer
		epsilon = Math.sqrt(6.0) / Math.sqrt(1+fanIn);
		U = SimpleMatrix.random(1, fanIn+1, -epsilon, epsilon, rand);
		U.set(0, 0.0);
	}
	
	
	/**
	 * Backpropagation gradient
	 * 
	 * 
	 * 
	 */
	protected SimpleMatrix[] backpropGrad(SimpleMatrix batch, SimpleMatrix label) {
		SimpleMatrix [] a = new SimpleMatrix[numOfHiddenLayer+2];
		SimpleMatrix [] z = new SimpleMatrix[numOfHiddenLayer+2];
		
		// Forward propagation
		for (int i = 0; i < numOfHiddenLayer+2; ++i) {
			// Input layer
			if (i == 0) {
				z[i] = batch;
				a[i] = MatlabAPI.horzconcat(z[i]);
			}
			// Output layer
			else if (i == numOfHiddenLayer+1) {
				z[i] = U.mult(a[i-1]);
				a[i] = sigmoid(z[i]);
			}
			// Middle layer
			else {
				z[i] = W[i-1].mult(a[i-1]);
				a[i] = MatlabAPI.horzconcat(tanh(z[i]));
			}
		}
		
		// Initialize a list of gradient matrices
		SimpleMatrix[] grad = new SimpleMatrix[numOfHiddenLayer+2];
		for (int i = 0; i < grad.length; ++i) {
			// Gradient for L
			if (i == 0) {
				grad[i] = new SimpleMatrix(batch.numRows(), 1);
			}
			// Gradient for U
			else if (i == numOfHiddenLayer+1) {
				grad[i] = new SimpleMatrix(U.numRows(), U.numCols());
			}
			// Gradient for W
			else {
				grad[i] = new SimpleMatrix(W[i-1].numRows(), W[i-1].numCols());
			}
		}
		
		int M = batch.numCols();
		SimpleMatrix Error = a[numOfHiddenLayer+1].minus(label);
		
		// For each instance in this batch
		for (int m = 0; m < M; ++m) {
			// Backward propagation
			SimpleMatrix Delta = Error.extractVector(false, m);
			
			for (int i = numOfHiddenLayer+1; i >= 1; --i) {
				grad[i] = grad[i].plus(Delta.mult(a[i-1].extractVector(false, m).transpose()));
				
				// Output layer
				if (i == numOfHiddenLayer+1) {
					Delta = U.transpose().mult(Delta);
					Delta = Delta.extractMatrix(1, Delta.numRows(), 0, Delta.numCols());
					Delta = Delta.elementMult(tanhDerivative(z[i-1].extractVector(false, m)));
				}
				else if (i == 1) {
					Delta = W[i-1].transpose().mult(Delta);
					Delta = Delta.extractMatrix(1, Delta.numRows(), 0, Delta.numCols());
				}
				// Hidden layer
				else {
					Delta = W[i-1].transpose().mult(Delta);
					Delta = Delta.extractMatrix(1, Delta.numRows(), 0, Delta.numCols());
					Delta = Delta.elementMult(tanhDerivative(z[i-1].extractVector(false, m)));
				}
			}
			// Input layer
			grad[0] = grad[0].plus(Delta);
		}
		
		// Average and add regularization term
		for (int i = 0; i < grad.length; ++i) {
			// Gradient for L
			if (i == 0) {
				grad[i] = grad[i].divide(M);
			}
			// Gradient for U
			else if (i == numOfHiddenLayer+1) {
				SimpleMatrix nobiasU = U.copy();
				double [] arr = new double[nobiasU.numRows()];
				Arrays.fill(arr, 0.0);
				nobiasU.setColumn(0, 0, arr);
				grad[i] = grad[i].divide(M).plus(nobiasU.scale(C / M));
			}
			// Gradient for W
			else {
				SimpleMatrix nobiasW = W[i-1].copy();
				double [] arr = new double[nobiasW.numRows()];
				Arrays.fill(arr, 0.0);
				nobiasW.setColumn(0, 0, arr);
				grad[i] = grad[i].divide(M).plus(nobiasW.scale(C / M));
			}
		}
		
		return grad;
	}
	
	
	/**
	 * Numerical gradient
	 * 
	 * 
	 */
	protected SimpleMatrix[] numericalGrad(SimpleMatrix batch, SimpleMatrix label) {
		double EPS = 1e-4;
		
		// Compute numerical gradient
		SimpleMatrix[] grad = new SimpleMatrix[numOfHiddenLayer+2];
		for (int i = 0; i < grad.length; ++i) {
			SimpleMatrix M = null;
			
			// Gradient for L
			if (i == 0) {
				grad[i] = new SimpleMatrix(batch.numRows(), 1);
				for (int r = 0; r < batch.numRows(); ++r) {
					SimpleMatrix perturb = new SimpleMatrix(batch.numRows(), 1);
					perturb.set(r, 0, EPS);
					SimpleMatrix right = batch.plus(MatlabAPI.repmat(perturb, 1, batch.numCols()));
					SimpleMatrix left = batch.minus(MatlabAPI.repmat(perturb, 1, batch.numCols()));
					double rr = costFunction(right, label), ll = costFunction(left, label);
					grad[i].set(r, 0, (rr-ll)/(2*EPS));
				}
				continue;
			}
			// Gradient for U
			else if (i == numOfHiddenLayer+1) {
				M = U;
			}
			// Gradient for W
			else {
				M = W[i-1];
			}
			
			grad[i] = new SimpleMatrix(M.numRows(), M.numCols());
			for (int r = 0; r < M.numRows(); ++r) {
				for (int c = 0; c < M.numCols(); ++c) {
					double mid = M.get(r, c);
					M.set(r, c, mid+EPS);
					double right = costFunction(batch, label);
					M.set(r, c, mid-EPS);
					double left = costFunction(batch, label);
					M.set(r, c, mid);
					grad[i].set(r, c, (right - left) / (2 * EPS)); 
				}
			}
		}
		return grad;
	}
	
	/**
	 * Check if two gradients are the same
	 * 
	 * 
	 * @param grad1
	 * @param grad2
	 */
	public void checkGradient(SimpleMatrix [] grad1, SimpleMatrix [] grad2) {
		if (grad1.length != grad2.length) {
			System.err.println("Gradient length not matched");
			return;
		}
		
		boolean flag = true;
		for (int i = 0; i < grad1.length; ++i) {
			double diff = grad1[i].minus(grad2[i]).elementMaxAbs();
			if (diff >= 1e-8) {
				System.err.println("Check gradient failed at Level: " + i + "\tDiff: " + diff);
				flag = false;
			}
		}
		
		if (flag) System.err.println("Success!");
	}
	

	/**
	 * Stochastic gradient descent training
	 */
	public void train(List<Datum> trainData, int Epoch, boolean Verbose) {
		
		List<List<Integer>> TrainX = makeInputWindows(trainData);
		List<Double> TrainY = makeLabels(trainData);
		int numTrain = trainData.size();
		
		// Check gradient
		/*for (int i = 0; i < 10; ++i) {
			SimpleMatrix input = makeInputVector(TrainX.get(i));
			SimpleMatrix label = new SimpleMatrix(1, 1);
			label.set(TrainY.get(i));	
			SimpleMatrix [] G = backpropGrad(input, label);
			SimpleMatrix [] NG = numericalGrad(input, label);
			checkGradient(G, NG);
		}*/
		
		// All train data and label in the matrix format
		SimpleMatrix trainDataAll = null, trainLabelAll = null;
		if (Verbose) {
			trainDataAll = new SimpleMatrix(wordSize * windowSize, numTrain);
			trainLabelAll = new SimpleMatrix(1, numTrain);
			for (int i = 0; i < numTrain; ++i) {
				trainDataAll.insertIntoThis(0, i, makeInputVector(TrainX.get(i)));
				trainLabelAll.set(0, i, TrainY.get(i));
			}
		}
		
		// SGD
		for (int epoch = 0; epoch < Epoch; ++epoch) {
			System.out.println("Epoch " + epoch);
			
			// Randomly shuffle examples
			long seed = System.nanoTime();
			Collections.shuffle(TrainX, new Random(seed));
			Collections.shuffle(TrainY, new Random(seed));
			
			// For each training example
			for (int i = 0; i < numTrain; ++i) {
					
				// Make input
				SimpleMatrix input = makeInputVector(TrainX.get(i));
				SimpleMatrix label = new SimpleMatrix(1, 1);
				label.set(TrainY.get(i));
				
				if (i % 30000 == 0) {
					System.out.println("\tProcessing " + i + "/" + numTrain + " examples.");
					if (Verbose) {
						System.out.println("\tObjective function value: " 
											+ costFunction(trainDataAll, trainLabelAll));
					}
				}
				
				// Compute Gradient
				SimpleMatrix [] G = backpropGrad(input, label);
				
				// Update W
				for (int ll = 1; ll <= numOfHiddenLayer; ++ll) {
					W[ll-1] = W[ll-1].minus(G[ll].scale(alpha));
				}
				
				// Update U
				U = U.minus(G[numOfHiddenLayer+1].scale(alpha));
				
				// Update L
				input = input.minus(G[0].scale(alpha));
				List<Integer> wordIdx = TrainX.get(i);
				for (int idx = 0; idx < wordIdx.size(); ++idx) {
					L.insertIntoThis(0, wordIdx.get(idx), input.extractMatrix(idx * wordSize, (idx+1) * wordSize, 0, 1));
				}
			}
		}
		
		// Evaluate training statistics
		System.out.println("Training statistics");
		evaluateStatistics(TrainX, TrainY);
		System.out.println();
	}

	
	/**
	 * Run test on dev/test set
	 * 
	 * @param testData
	 */
	public void test(List<Datum> testData) {
		List<List<Integer>> TestX = makeInputWindows(testData);
		List<Double> TestY = makeLabels(testData);
		
		// Evaluate test statistics
		System.out.println("Test statistics");
		evaluateStatistics(TestX, TestY);
		System.out.println();
	}

	
	/**
	 * Dump the word vectors matrix L into a file.
	 * Each row is a word vector.
	 * 
	 * @param filename
	 * @throws IOException 
	 */
	public void dumpWordVectors(String filename) throws IOException {
		BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
		int numOfWord = L.numCols();
		for (int i = 0; i < numOfWord; ++i) {
			for (int j = 0; j < wordSize; ++j) {
				writer.write(Double.toString(L.get(j, i)));
				if (j != wordSize - 1) writer.write(" ");
			}
			writer.write("\n");
		}
		writer.close();
	}	
}
