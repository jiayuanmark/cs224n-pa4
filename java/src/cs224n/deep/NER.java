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
		WindowModel model = new WindowModel(5, 100, 0.001);
		model.initWeights();

		//
		System.out.println(FeatureFactory.getWordVectors().numRows() + " " + FeatureFactory.getWordVectors().numCols());
		
		// Train and Test
		System.out.println("Start training...");
		model.train(trainData);
		System.out.println("Finish training...");
		model.test(testData);
		
		// Dump word vectors
		model.dumpWordVectors("/Users/jiayuanm/Documents/cs224n/cs224n-pa4/tSNE/trainedVectors.txt");
	}
}