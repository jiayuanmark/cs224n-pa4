package cs224n.deep;

import java.util.*;
import java.io.*;

public class NER {

	public static void main(String[] args) throws IOException {
		
		// Default parameters
		Map<String, String> options = new HashMap<String, String>();
		options.put("-window", "13");
		options.put("-layers", "300");
		options.put("-data", "../data/");
		options.put("-train", "train2");
		options.put("-test", "dev2");
		options.put("-alpha", "0.001");
		options.put("-regularize", "0.0001");
		options.put("-epoch", "10");

		// Command-line options supersede defaults
		options.putAll(CommandLineUtils.simpleCommandLineParser(args));
		
		String dataPath = options.get("-data");
		String train = options.get("-train");
		String test = options.get("-test");
		int windowSize = Integer.valueOf(options.get("-window")).intValue();
		double alpha = Double.valueOf(options.get("-alpha")).doubleValue();
		double C = Double.valueOf(options.get("-regularize")).doubleValue();
		int numOfEpoch = Integer.valueOf(options.get("-epoch")).intValue();
		boolean verbose = options.containsKey("-v");
		
		// Read in the train and test data sets
		List<Datum> trainData = FeatureFactory.readTrainData(dataPath + train);
		List<Datum> testData = FeatureFactory.readTestData(dataPath + test);

		// Read in dictionary and word vectors
		FeatureFactory.initializeVocab(dataPath + "vocab.txt");
		FeatureFactory.readWordVectors(dataPath + "wordVectors.txt");

		// Initialize model
		String[] layerStr = options.get("-layers").split(",");
		int [] layer = new int[layerStr.length];
		for (int i = 0; i < layerStr.length; ++i) {
			layer[i] = Integer.valueOf(layerStr[i]).intValue();
		}
		
		WindowModel model = new MinibatchModel(windowSize, layer, alpha, C);
		model.initWeights();

		// Check point
		System.out.println(FeatureFactory.getWordVectors().numRows() + " " + FeatureFactory.getWordVectors().numCols());
		System.out.println("Train examples: " + trainData.size());
		System.out.println("Test examples: " + testData.size());
		
		// Train and Test
		System.out.println("Start training...");
		model.train(trainData, numOfEpoch, verbose);
		System.out.println("Finish training...");
		model.test(testData);
		
		// Dump word vectors
		if (options.containsKey("-dump")) {
			model.dumpWordVectors(options.get("-dump"));
		}
	}
}
