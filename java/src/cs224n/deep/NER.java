package cs224n.deep;

import java.util.*;
import java.io.*;

import cs224n.deep.CommandLineUtils;


public class NER {

	public static void main(String[] args) throws IOException {
		
		
		Map<String, String> options = new HashMap<String, String>();
		options.put("-window-size",      "/afs/ir/class/cs224n/pa2/data/");
		options.put("-layers",      "100,80");
		options.put("-data",      "/Users/jiayuanm/Documents/cs224n/cs224n-pa4/data");
		options.put("-train",     "/train");
		options.put("-test",      "/dev");
		options.put("-alpha",    "0.0005");
		options.put("-regularize", "0.0001");

		// let command-line options supersede defaults .........................
		options.putAll(CommandLineUtils.simpleCommandLineParser(args));
		String dataPath = options.get("-data");
		String train = options.get("-train");
		String test = options.get("-test");
		
		int windowSize = Integer.valueOf(options.get("-window-size")).intValue();
		double alpha = Double.valueOf(options.get("-alpha")).doubleValue();
		double C = Double.valueOf(options.get("-regularize")).doubleValue();
		// Read in the train and test data sets
		List<Datum> trainData = FeatureFactory.readTrainData(dataPath + train);
		List<Datum> testData = FeatureFactory.readTestData(dataPath + test);

		// Read in dictionary and word vectors
		FeatureFactory.initializeVocab(dataPath + "/vocab.txt");
		FeatureFactory.readWordVectors(dataPath + "/wordVectors.txt");

		// Initialize model
		String[] layer_str = options.get("-layers").split(",");
		int [] layer = new int[layer_str.length];
		int i = 0;
		for (String str : layer_str) {
			layer[i] = Integer.valueOf(str).intValue();
			i++;
		}
		WindowModel model = new WindowModel(windowSize, layer, alpha, C);
		model.initWeights();

		// Check point
		System.out.println(FeatureFactory.getWordVectors().numRows() + " " + FeatureFactory.getWordVectors().numCols());
		System.out.println("Train examples: " + trainData.size());
		System.out.println("Test examples: " + testData.size());
		
		// Train and Test
		System.out.println("Start training...");
		model.train(trainData);
		System.out.println("Finish training...");
		model.test(testData);
		
		// Dump word vectors
		//model.dumpWordVectors("/Users/jiayuanm/Documents/cs224n/cs224n-pa4/tSNE/trainedVectors.txt");
	}
}