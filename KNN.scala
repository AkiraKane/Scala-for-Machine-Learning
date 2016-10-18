// The K-Nearest-Neighbour Classification Algorithm
// 1. A conceptually simple classification algorithm 
// 2. Classify a new data point x by taking a majority vote between the k closest data points to x in the traning data set
// 3. Requires a distance function in order to caculate distances between data points
// 4. The optimal k (number of neighbours) can be determined via cross-validation

import breeze.linalg.{*,DenseMatrix, DenseVector}

class NearestNeighhbours(k:Int,
						 dataX: DenseMatrix[Double],
						 dataY: Seq[String],
						 distanceFn: (DenseVector[Double], DenseVector[Double]) => Double) {

	/**
 		* An implementation of the k-nearest neighbours classification algorithm.
 		* @param k The number of neighbours to use for prediction.
 		* @param dataX Matrix of input examples.
 		* @param dataY Corresponding output classes.
 		* @param distanceFn Function used to compute 'near-ness'.
 	*/

 	def predict(x:DenseVector[Double]): String = {
 		/**
   			* Predict the output class corresponding to a given input example
   			* @param x input example
   			* @return predicted class
   		*/

   		// Compute the similarity to each example
   		val distances = dataX(*,::)
   							.map(r => distanceFn(r,x))

   		// Get the top K most similar classes
   		val topKClasses = distances
   							.toArray
   							.zipWithIndex
   							.sortBy(_._1)
   							.take(k)
   							.map{case (dist, idx) => dataY(idx)}

   		// Most frequent class in top K
   		topKClasses
   			.groupBy(k)
   			.mapValues(_.size)
   			.maxBy(_._2)
   			._1
 	}
}

def line2Data(line: String): (List[Double], String) = {
	val elements = line.split(",")
	val y = elements.last
	val x = elements
				.dropRight(1)
				.map(_.toDouble)
				.toList
	(x,y)
}

val data = Source
			.fromFile("ionosphere.data")
			.getLines()
			.map(x => line2Data(x))
			.toList

val outputs = data.map(_._2).toSeq
val inputs = DenseMatrix(data.map(_._1).toArray: _*)
val euclideanDist = (v1:DenseVector[Double], v2:DenseVector[Double]) => v1
																		.toArray
																		.zip(v2.toArray)
																		.map(x => pow((x._1 - x._2), 2))
																		.sum

val trainInputs = inputs(0 to 299, ::)
val trainOutputs = outputs.take(300)

val myNN = new NearestNeighbours(k=4, dataX=trainInputs, dataY=trainOutputs, euclideanDist)

val correct = 0
(300 to 350).foreach{ exampleId => val pred = myNN.predict(inputs(exapleId,::).t)
								   val target = outputs(exampleId)
								   if(pred == target) correct += 1}

println(correct.toDouble / (300 to 350).length)
