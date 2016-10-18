// Generalized Linear Models (GLMs)
// 1. Sometimes the relationship between the feature vector and target variable is nonlinear
// 2. LR can still model a nonlinear relationship if nonlinear basis functions are used
// 3. A basis function is a nonlinear function that is applied to feature vectors

// Training a GLM
// 1. A GLM prediction is similar to a LR prediction except with the presence of a basis function
// 2. Linear in w but nonlinear in x

import breeze.linalg.{*, DenseMatrix, DenseVector, inv}
import breeze.stats.mean

class LinearRegression(inputs: DenseMatrix[Double], 
					   outputs: DenseMatrix[Double],
					   basisFn: Option[DenseVector[Double] => DenseVector[Double]] = None){


/**
 * Class for a linear regression supervised learning model.
 * @param inputs A matrix whose rows are the input vectors corresponding to each training example.
 * @param outputs A matrix whose rows are the outputs corresponding to each training example
 * @param basisFn An optional basis function to be applied to the inputs (for generalized linear models).
 */

 	// If a basis function has been provided, apply it to each input example
 	val x = basisFn match {
 		case some(bf) => inputs(*,::).map(dv => bf(dv))
 		case None => inputs
 	}

 	def predict(weights: DenseMatrix[Double], 
 				input: DenseMatrix[Double]): DenseMatrix[Double] = {
 		/**
   			* Given an input example vector and a weight vector, predict the output
   			* @param weights Learning LR weight vector.
   			* @param input Input example vector.
  			 * @return Prediction.
   		*/
 		input * weights
 	}

 	def train(inputs: DenseMatrix[Double] = x, 
 			  outputs: DenseMatrix[Double] = outputs, 
 			  regularizationParam: Double = 0.0): DenseMatrix[Double] = {

 		/**
   			* Train a weight vector for a LR model.
   			* @param inputs The input training examples, by default they are the ones provided in the constructor
   			* @param outputs The output training examples, by default they are the ones provided in the constructor.
   			* @param regularizationParam The regularization penalty weight, by default it is zero (no regularization).
   			* @return A weight vector.
   		*/

   		val l = inputs.cols
   		val identMat = DenseMatrix.eye[Double](l)
   		val regPenalty = regularizationParam * l

   		// The normal equation for LR( with regularization)
   		inv(inputs.t * inputs + regPenalty * identMat) * (inputs.t * outputs)
 	}

 	def evaluate(weights: DenseMatrix[Double],
 				 inputs: DenseMatrix[Double],
 				 targets: DenseMatrix[Double],
 				 evaluator: (DenseMatrix[Double], DenseMatrix[Double] => Double)): Double = {

 		/**
   			* Compute the MSE for a LR model on test data.
   			* @param weights Weight vector for a learning LR model.
   			* @param inputs Inputs for test data.
   			* @param targets Outputs for test data.
   			* @return MSE.
   		*/

   		// Compute predictions
   		val preds = predict(weights, inputs)
   		// Compute predictions to targets with MSE
   		evaluator(preds, targets)
 	}

 	def crossValidation(folds:Int,
 						regularizationParam: Double,
 						evaluator: (DenseMatrix[Double], DenseMatrix[Double]) => Double): Double ={

 		/**
   			* Perform k-fold cross-validation using the entire dataset provided in constructor.
   			* @param folds The number of cross-validation folds to use.
   			* @param regularizationParam The regularization parameter to use.
   			* @return The average cross-validation error over all folds.
   		*/

   		val foldSize = x.rows / folds.toDouble

   		// segment dataset
   		val partitions = (0 to x.rows-1).grouped(math.ceil(foldSize).toInt)
   		val ptSet = (0 to x.rows-1).toSet

   		// compute test error for each fold
   		val xValError = partitions.foldRight(Vector.empty[Double]){(c, acc) => 

   			// training data points are all data points not in validation set.
   			val trainIdx = ptSet.diff(c.toSet)
   			val testIdx = c

   			// training data 
   			val trainX = x(trainIdx.toIndexedSeq, ::).toDenseMatrix
   			val trainY = outputs(trainIdx.toIndexedSeq, ::).toDenseMatrix

   			// test data
   			val testX = x(testIdx.toIndexedSeq, ::).toDenseMatrix
   			val testY = outputs(testIdx.toIndexedSeq, ::).toDenseMatrix

   			// train a weight vector with the above training data
   			val weights = train(trainX, trainY, regularizationParam)

   			// compute the error on the held-out test data
   			val error = evaluate(weights, testX, testY, evaluator)

   			// append error to the accumulator so it can be average later
   			acc :+ error
   		}

   		mean(xValError)

	}
}

// Input data
val data = Source.fromFile("boston_housing.data")
	.getLines()
	.map(x => line2Data(x))
	.toArray

// Convert to breeze matrix
val dm = DenseMatrix(data: _*)

// the inputs are all but the last column, outputs are the last column
val x = dm(::, 0 to 12)
val y = dm(::, -1).toDenseMatrix.t

// Create LR object with our dataset
val myLr = new LinearRegression(inputs=x, outputs=y)

// Train LR weights
val weights = myLr.train()
val testX = x(0 to 30, ::)
val testY = y(0 to 30, ::)

val pred = myLr.predict(weights,testX)

val mseEvaluator = (pred: DenseMatrix[Double], target: DenseMatrix[Double]) => mean((pred-target).map(x=> pow(x,2)))
val mse = myLr.evaluate(weights = weights, inputs = testX, targets = testY, evaluator = mseEvaluator)
print(mse)
