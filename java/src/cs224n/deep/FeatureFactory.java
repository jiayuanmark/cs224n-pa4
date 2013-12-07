package cs224n.deep;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import org.ejml.simple.*;

public class FeatureFactory {

	private FeatureFactory() {

	}
	
	
	/** Public singleton pattern interfaces **/
	
	public static List<Datum> getTrainData() {
		return trainData;
	}
	
	public static List<Datum> getTestData() {
		return testData;
	}
	
	public static SimpleMatrix getWordVectors() {
		return allVecs;
	}
	
	public static HashMap<String, Integer> getDictionary() {
		return wordToNum;
	}
	
	private static List<Datum> trainData;

	/** Do not modify this method **/
	public static List<Datum> readTrainData(String filename) throws IOException {
		if (trainData == null) {
			trainData = read(filename);
		}
		return trainData;
	}

	private static List<Datum> testData;

	/** Do not modify this method **/
	public static List<Datum> readTestData(String filename) throws IOException {
		if (testData == null) {
			testData = read(filename);
		}
		return testData;
	}

	private static List<Datum> read(String filename)
			throws FileNotFoundException, IOException {

		List<Datum> data = new ArrayList<Datum>();
		BufferedReader in = new BufferedReader(new FileReader(filename));
		for (String line = in.readLine(); line != null; line = in.readLine()) {
			if (line.trim().length() == 0) {
				continue;
			}
			String[] bits = line.split("\\s+");
			String word = bits[0];
			String label = bits[1];

			Datum datum = new Datum(word, label);
			data.add(datum);
		}
		in.close();
		return data;
	}

	/**
	 * Look up table matrix with all word vectors with dimensionality n x |V|
	 * Access it directly in WindowModel
	 * 
	 */
	private static SimpleMatrix allVecs;

	public static SimpleMatrix readWordVectors(String vecFilename)
			throws IOException {

		if (allVecs != null)
			return allVecs;

		int numOfWords = wordToNum.size();
		int numOfDim = -1;

		double[][] data = null;

		BufferedReader in = new BufferedReader(new FileReader(vecFilename));
		int col = 0;
		for (String line = in.readLine(); line != null; line = in.readLine()) {
			if (line.trim().length() == 0) {
				continue;
			}
			String[] words = line.split("\\s+");
			if (data == null) {
				numOfDim = words.length;
				data = new double[numOfDim][numOfWords];
			}

			if (numOfDim != words.length) {
				System.err.println("Word vectors dimension unmatched!");
			}

			for (int row = 0; row < numOfDim; ++row) {
				data[row][col] = Double.parseDouble(words[row]);
			}
			++col;
		}
		in.close();

		allVecs = new SimpleMatrix(data);
		return allVecs;
	}

	/**
	 * Word to number lookups, just access them directly in WindowModel.
	 * 
	 */
	private static HashMap<String, Integer> wordToNum = new HashMap<String, Integer>();
	private static HashMap<Integer, String> numToWord = new HashMap<Integer, String>();

	public static HashMap<String, Integer> initializeVocab(String vocabFilename)
			throws IOException {

		if (wordToNum.size() != 0)
			return wordToNum;
		BufferedReader br = new BufferedReader(new FileReader(vocabFilename));
		int idx = 0;
		for (String line = br.readLine(); line != null; line = br.readLine()) {
			if (line.trim().length() == 0) {
				continue;
			}
			String[] words = line.split("\\s+");
			wordToNum.put(words[0], idx);
			numToWord.put(idx, words[0]);
			++idx;
		}
		br.close();
		return wordToNum;
	}

}
