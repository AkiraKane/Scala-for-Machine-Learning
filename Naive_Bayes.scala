import breeze.linalg._
import breeze.numerics.log
import breeze.stats._
import breeze.stats.distributions.Gaussian

class NaiveBayes(dataX: DenseMatrix[Double], dataY: Seq[String]){

	val classes = dataY.distinct

	val likelihoods = {

		// map classes to row indexes of their correponding examples
		val classIdx = dataY.zipwithIndex.groupBy(_._1).mapValues(_.map(_._2)) 

		// for each class, create Normal distribution for each of its input features
		classIdx.mapValues{ idx => 

			val classData = dataX(idx,::).toDenseMatrix 
			
			classData(::, *).map{ col => 

				val empMean = mean(col)
			   
			    val empStddeev = stddev(col)

			    // std cannot be zero. If it is, set it to small value
			    val trueV = empStddev match {
			    	case 0.0 => 0.001
			    	case _   => empStddev
			    	}
			    
			    new Gaussian(mu= empMean, sigma = trueV)

			    }.toArray
			}
	}

	val priors = {

		val numExamples = dataY.length

		dataY
			.groupBy(identity)
			.mapValues(x => x.size / numExamples.toDouble)
	}

	def predict(x: DenseVector[Double]): String = {

		// Compute posteriors for each class
		val posteriors = classes.map{ cl => 

			val prior = priors(cl)
			val likelihoodDists = likelihoods(cl)
			val logLikelihoods = likelihoodDists
					.zip(x.toArray)
					.map{ case (dist, value) => log(dist.pdf(value))}

			val posterior = logLikelihoods.sum + log(prior)
			(cl, posterior)
		}

		posteriors
			.sortBy(-_._2)
			.head._1
	}
}

def row2Data(row: Seq[String]): (Seq[Double], String) = {

	val x = row.dropRight(1).map(_.toDouble)
	val y = row.last
	(x,y)
}

val dataset = Source.fromFile("ionosphere.data")
	.getLines()
	.map(r => row2Data(r.split(",")))
	.toList

val inputs = DenseMatrix(dataset.map(_._1): _*)
val outputs = dataset.map(_._2)

val trainInputs = inputs(0 to 299, ::)
val trainOutputs = outputs.take(300)

val myNB = new NaiveBayes(dataX = trainInputs, dataY = trainOutputs)

val correctCounter = 0
(300 to 350).foreach{ exampleId => val prediction = myNB.predict(inputs(exampleId,::).t)
								   val actual = outputs(exampleId)
								   if (prediction == actual) correctCounter += 1
								}
println(correctCounter.toDouble / (300 to 350).length)


