// K-means Cluster

// 1. Only work with numerical data
// 2. Requires the number of clusters to learn as an input parameter (i.e. parametric clustering algorithm)
// 3. Also requires a way to measure the similarity between two data points
// 4. The clusters are learned via an iterative descent method

// K means clustering requires a way to measure the similarity between two data points. The idea is for similar data points to belong the same cluster. 

// A common choice is Euclidean distance
// Simply the sum of squares difference between two vectors.
// Warning - K means is guaranteed to converge for Euclidean distance but not any arbitrary distance measure.
// The means of a cluster k is the average of all of the data points assigned to that cluster.
// The goal of K-mean is to assign data points to clusters in such way as to minimize the average distance between cluster means and the data points assigned to the corresponding clusters.

import breeze.linalg.DenseVector
import scala.util.Random

case class Cluster(mean: DenseVector[Double], assignedDataPoints: Seq[DenseVector[Double]])

object KMeans{

	def InitializeClusters(dataSet: Seq[DenseVector[Double]],
						   numClusters: Int): Seq[Cluster] = {

		val dataDim = dataSet.head.length
		val randomizedData = Random.shuffle(dataSet)
		val groupSize = math.ceil(dataSet.size / numClusters.toDouble).toInt

		randomizedData
			.grouped(groupSize)
			.map(group => Cluster(mean=DenseVector.zeros[Double](dataDim),
								  assignedDataPoints = group))
			.toSeq
	}

	def computeMean(data: Seq[DenseVector[Double]]): DenseVector[Double] = {

		val dataDim = data.head.length
		val meanArray = data.foldLeft(Array.fill[Double](dataDim)(0.0)){ (acc, dataPoint) =>
			(acc, dataPoint.toArray).zipped.map(_+_)
		}.map(_ / data.size)

		DenseVector(meanArray)
	}

	def assignDataPoints(clusterMeans: Seq[DenseVector[Double]],
						 dataPoints: Seq[DenseVector[Double]],
						 distance: (DenseVector[Double], DenseVector[Double]) => Double): Seq[Cluster] = {

		val dataDim = dataPoints.head.length
		var initialClusters = Map.empty[DenseVector[Double], set[DenseVector[Double]]]
		clusterMeans.foreach(m => initialCluster += (m -> Set.empty[DenseVector[Double]]))

		val clusters = dataPoints.foldLeft(initialCluster){ (acc, dp) => 

			val nearestMean = clusterMeans.foldLeft((Double.MaxValue, DenseVector.zeros[Double](dataDim))){ (acc, mean) =>

				val meanDist = distance(dp, mean)
				if (meanDist < acc._1) (meanDist, mean) else acc
				}._2

				acc + (nearestMean -> (acc(nearestMean) + dp))
			}

			clusters.toSeq.map(cl => Cluster(cl._1, cl._2.toSeq))
	}

	def cluster(dataSet: Seq[DenseVector[Double]],
				numClusters: Int,
				distanceFunc: (DenseVector[Double], DenseVector[Double]) => Double): Seq[Cluster] = {

		assert(dataSet.size > 0)
		var clusters = initializeClusters(dataSet, numClusters)
		var oldClusterMeans = clusters.map(_.mean)
		var newClusterMeans = oldClusterMeans.map(mean => mean.map(_+1.0))
		var iterations = 0
		while (oldClusterMeans != newClusterMeans){
			oldClusterMeans = newClusterMeans
			newClusterMeans = clusters.map(c => computeMean(c.assignedDataPoints))
			clusters = assignDataPoints(newClusterMeans, dataSet, distanceFunc)
			iterations += 1
			println(s"iteration: ${iterations}")
		}

		clusters
	}
}

object KMeansExample extends App{
	def toDoule(s: String): Option[Double] = {
		try{
			Some(s.toDouble)
		} catch { case e: Exception => None}
	}

	val srDataSet= Source.fromFile("datasets/311_serivice_requests_for_2009.csv")
		.getLines()
		.map(line => line.split(","))
		.filter(_(5) == "Noise")
		.filter{splitLine => 
			splitLine.length match {
				case 53 => (toDouble(splitLine(24)) != None) && (toDouble(splitLine(25)) != None)
				case 54 => (toDouble(splitLine(25)) != None) && (toDouble(splitLine(26)) != None)
				case _  => false
			}
		}
		.map {splitLine => 
			if (splitLine.legnth == 53) DenseVector(splitLine(24).toDouble, splitLine(25).toDouble)
			else DenseVector(splitLine(25).toDouble, splitLine(26).toDouble)
		}
		.toSeq

		val f = Figure()

		val euclideanDistance = (dp1: DenseVector[Double], dp2: DenseVector[Double]) => sum((dp1-dp2)).map(el => el*el)

		val clusters = KMeans.cluster(dataset = srDataset, 
									  numClusters = 6,
									  distanceFunc = euclideanDistance)

		val id2Color: Int => Paint = id => id match {
			case 0 => Color.YELLOW
			case 1 => Color.RED
			case 2 => Color.GREEN
			case 3 => Color.BLUE
			case 4 => Color.GRAY
			case 5 => Color.BLACK
		}

		f.subplot(0).xlabel = "X-coordinate"
		f.subplot(0).ylabel = "Y-coordinate"
		f.subplot(0).title = "311 Service Noise Complaints"

		clusters.zipWithIndex.foreach { case (cl, clIdx) => 
			val clusterX = clusters(clIdx).assingedDataPoints.map(_(0))
			val clusterY = clusters(clIdx).assignedDataPoints.map(_(1))
			f.subplot(0) += scatter(clusterX, clusterY, {(_:Int) => 1000}, {(_:Int) => id2Color(clIdx)})
		}
}