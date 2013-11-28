package cs224n.deep;

import org.ejml.simple.*;

public class MatlabAPI {
	
	
	/**
	 * Matlab's repmat function
	 * 	B = repmat(A, X, Y)
	 * 
	 * @param A
	 * @param X
	 * @param Y
	 * @return B
	 */
	public static SimpleMatrix repmat(SimpleMatrix A, int X, int Y) {
		int row = A.numRows(), col = A.numCols();
		SimpleMatrix B = new SimpleMatrix(row * X, col * Y);		
		for (int xx = 0; xx < X; ++xx) {
			for (int yy = 0; yy < Y; ++yy) {
				B.insertIntoThis(xx * row, yy * col, A);
			}
		}
		return B;
	}
	
	/**
	 * Adding one 1's row on A
	 * 	B = [ones(1, size(A,2)); A]
	 * @param A
	 * @return B
	 */
	public static SimpleMatrix horzconcat(SimpleMatrix A) {
		SimpleMatrix B = new SimpleMatrix(1, A.numCols());
		for (int i = 0; i < A.numCols(); ++i) {
			B.set(0, i, 1.0);
		}
		return B.combine(B.numRows(), 0, A);
	}

}
