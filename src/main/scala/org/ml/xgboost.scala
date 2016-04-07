package org.ml

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest


/**
  * Created by obs on 3/14/16.
  */
object xgboost {

  def main(args: Array[String]) = {
    val conf = new SparkConf()
        .setAppName("xgboost  ")
        .setMaster("local")

    val sc = new SparkContext(conf)

    //test
    val rawData = sc.textFile("/Users/obs/git/testSparkScala/src/test/resources/test_data/breast-cancer-wisconsin.txt")
    println("raw data::::::::::::::::")
    rawData.take(10).foreach(println)

    val data = rawData.map(_.split(","))
                      .filter(_(6) != "?")
                      .map(_.drop(1))
                      .map(_.map(_.toDouble))

    val labeledPoints = data.map(x => LabeledPoint( if (x.last == 4) 1 else 0, Vectors.dense(x.init)))
    println("labeledPoints data::::::::::::::::")
    labeledPoints.take(10).foreach(println)

    val splits = labeledPoints.randomSplit(Array(0.7,0.3), seed = 50431)

    val (trainingData, testData) = (splits(0), splits(1))

    // Train a RandomForest model.
    // Empty categoricalFeaturesInfo indicates all features are continuous.

    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 20 // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "gini"
    val maxDepth = 4
    val maxBins = 32
    val seed = 5043

    val model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed)

    // Clear the prediction threshold so the model will return probabilities
    //model.clearThreshold


    // Evaluating model
    val predictionAndLabels = testData.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }

    // Instantiate metrics object
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)

    // Precision by threshold
    val precision = metrics.precisionByThreshold
    precision.foreach { case (t, p) =>
      println(s"Threshold: $t, Precision: $p")
    }

    // Recall by threshold
    val recall = metrics.recallByThreshold
    recall.foreach { case (t, r) =>
      println(s"Threshold: $t, Recall: $r")
    }

    // Precision-Recall Curve
    val PRC = metrics.pr

    // F-measure
    val f1Score = metrics.fMeasureByThreshold
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 1")
    }

    val beta = 0.5
    val fScore = metrics.fMeasureByThreshold(beta)
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 0.5")
    }

    // AUPRC
    val auPRC = metrics.areaUnderPR
    println("Area under precision-recall curve = " + auPRC)

    // Compute thresholds used in ROC and PR curves
    val thresholds = precision.map(_._1)

    // ROC Curve
    val roc = metrics.roc

    // AUROC
    val auROC = metrics.areaUnderROC
    println("Area under ROC = " + auROC)


  }

}
