package cs224n.deep;

import java.util.*;
import java.io.*;

import org.ejml.simple.SimpleMatrix;

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
		FeatureFactory.initializeVocab("../data/vocab.txt");
		SimpleMatrix allVecs = FeatureFactory.readWordVectors("../data/wordVectors.txt");

		// Initialize model
		WindowModel model = new WindowModel(5, 100, 0.001);
		model.initWeights(allVecs);

		// Train and Test
		model.train(trainData);
		model.test(testData);
	}
}