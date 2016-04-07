package org.ml

import org.apache.spark.{SparkConf, SparkContext}


/**
  * Created by obs on 3/14/16.
  */
object digitalDataClean {

  def main(args: Array[String]) = {
    val conf = new SparkConf()
        .setAppName("digitalDataClean")
        .setMaster("local")

    val sc = new SparkContext(conf)

    //test
    val rawData = sc.textFile("/Users/obs/git/testSparkScala/src/test/resources/test_data/dp_sample.csv")
    println("raw data::::::::::::::::")
    rawData.take(10).foreach(println)

    val data = rawData.map(_.split(","))
                      .map(_.drop(1))





  }

}
