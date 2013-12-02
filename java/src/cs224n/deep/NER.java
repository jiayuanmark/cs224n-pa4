package cs224n.deep;

import java.util.*;
import java.io.*;


public class NER {

	public static void main(String[] args) throws IOException {
		if (args.length < 2) {
			System.out.println("USAGE: java -cp classes NER ../data/train ../data/dev");
			return;
		}

		// Read in the train and test data sets
		List<Datum> trainData = FeatureFactory.readTrainData(args[0]);
		List<Datum> testData = FeatureFactory.readTestData(args[1]);

		// Read in dictionary and word vectors
		FeatureFactory.initializeVocab("/Users/jiayuanm/Documents/cs224n/cs224n-pa4/data/vocab.txt");
		FeatureFactory.readWordVectors("/Users/jiayuanm/Documents/cs224n/cs224n-pa4/data/wordVectors.txt");

		// Initialize model
		int [] layer = {100, 80};
		WindowModel model = new WindowModel(7, layer, 0.0005, 0.0001);
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