package cs224n.deep;

import java.util.Collections;
import java.util.List;
import java.util.ArrayList;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

public class MinibatchModel extends WindowModel {

	/**
	 * Shallow architecture constructor
	 * 
	 * @param windowSize
	 * @param hiddenSize
	 * @param lr
	 * @param reg
	 */
	public MinibatchModel(int windowSize, int hiddenSize, double lr, double reg) {
		super(windowSize, hiddenSize, lr, reg);
	}
	
	/**
	 * Deep architecture constructor
	 * 
	 * @param windowSize
	 * @param hiddenSize
	 * @param lr
	 * @param reg
	 */
	public MinibatchModel(int windowSize, int[] hiddenSize, double lr, double reg) {
		super(windowSize, hiddenSize, lr, reg);
	}
	
	
	private List<SimpleMatrix> makeBatchData(List<Datum> data, int batchSize) {
		List<List<Integer> > windows = makeInputWindows(data);
		List<SimpleMatrix> ret = new ArrayList<SimpleMatrix>();
		for (int i = 0; i < data.size(); i += batchSize) {
			int numOfCols = i + batchSize < data.size() ? batchSize : data.size() - i;
			SimpleMatrix inputMatrix = new SimpleMatrix(wordSize * windowSize, numOfCols);
			for (int s = 0; s < numOfCols; ++s) {
				inputMatrix.insertIntoThis(0, s, makeInputVector(windows.get(i+s)));
			}
			ret.add(inputMatrix);
		}
		return ret;
	}
	
	
	private List<SimpleMatrix> makeBatchLabel(List<Datum> data, int batchSize) {
		List<Double> labels = makeLabels(data);
		List<SimpleMatrix> ret = new ArrayList<SimpleMatrix>();
		for (int i = 0; i < data.size(); i += batchSize) {
			int numOfCols = i + batchSize < data.size() ? batchSize : data.size() - i;
			SimpleMatrix labelMatrix = new SimpleMatrix(1, numOfCols);
			for (int s = 0; s < numOfCols; ++s) {
				labelMatrix.set(0, s, labels.get(i+s));
			}
			ret.add(labelMatrix);
		}
		return ret;
	}
	
	
	private List<List<List<Integer>>> partitionWindow(List<Datum> data, int batchSize) {
		List<List<Integer>> windows = makeInputWindows(data);
		List<List<List<Integer>>> partition = new ArrayList<List<List<Integer>>>();
		for (int i = 0; i < data.size(); i += batchSize) {
			int numOfCols = i + batchSize < data.size() ? batchSize : data.size() - i;
			List<List<Integer>> temp = new ArrayList<List<Integer>>();
			for (int s = 0; s < numOfCols; ++s) {
				temp.add(windows.get(i+s));
			}
			partition.add(temp);
		}
		return partition;
	}
	
	
	
	@Override
	public void train(List<Datum> trainData, int Epoch, boolean Verbose) {
				
		// Mini-batch gradient descent
		for (int epoch = 0; epoch < Epoch; ++epoch) {
			System.out.println("Epoch " + epoch);
			
			// Randomly shuffle examples
			long seed = System.nanoTime();
			Collections.shuffle(trainData, new Random(seed));
			
			// Make mini-batch data
			List<SimpleMatrix> TrainX = makeBatchData(trainData, 10);
			List<SimpleMatrix> TrainY = makeBatchLabel(trainData, 10);
			List<List<List<Integer>>> Partite = partitionWindow(trainData, 10);
			
			// For each batch
			int numBatch = TrainX.size();
			for (int i = 0; i < numBatch; ++i) {
				
				if (i % 2000 == 0) {
					System.out.println("\tProcessing " + i + "/" + numBatch + " mini-batches.");
					if (Verbose) {
						System.out.println("\tObjective function value: " + costFunction(TrainX.get(i), TrainY.get(i)));
					}
				}
				
				// Compute Gradient
				SimpleMatrix [] G = backpropGrad(TrainX.get(i), TrainY.get(i));
				
				// Update W
				for (int ll = 1; ll <= numOfHiddenLayer; ++ll) {
					W[ll-1] = W[ll-1].minus(G[ll].scale(alpha));
				}
				
				// Update U
				U = U.minus(G[numOfHiddenLayer+1].scale(alpha));
				
				// Update L
				SimpleMatrix batchInput = TrainX.get(i);
				SimpleMatrix batchOutput = TrainY.get(i);
				int numOfSample = batchInput.numCols();
				List<List<Integer>> Index = Partite.get(i);
				for (int sp = 0; sp < numOfSample; ++sp) {
					SimpleMatrix Lupdate = batchInput.extractMatrix(0, batchInput.numRows(), sp, sp+1);
					SimpleMatrix [] Gprime = backpropGrad(	Lupdate,
															batchOutput.extractMatrix(0, 1, sp, sp+1));
					Lupdate = Lupdate.minus(Gprime[0].scale(alpha));
					List<Integer> wordIdx = Index.get(sp);
					for (int w = 0; w < windowSize; ++w) {
						L.insertIntoThis(	0, wordIdx.get(w), 
											Lupdate.extractMatrix(w * wordSize, (w+1) * wordSize, 0, 1));
					}
				}
			}
		}
		
		// Evaluate training statistics
		System.out.println("Training statistics");
		evaluateStatistics(makeInputWindows(trainData), makeLabels(trainData));
		System.out.println();
	}
}
