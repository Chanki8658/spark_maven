package org.test

import org.apache.spark.{SparkContext, SparkConf}

/**
  * Created by obs on 3/14/16.
  */
object wordCount {

  def main(args: Array[String]) = {
    val conf = new SparkConf()
        .setAppName("wordCount")
        .setMaster("local")

    val sc = new SparkContext(conf)

    val test = sc.textFile("/Users/obs/git/testSparkScala/src/test/resources/test_data/sample.txt")
    test.flatMap(line => line.split(" "))
        .map(word => (word,1))
        .reduceByKey(_+_)
        .saveAsTextFile("/Users/obs/git/testSparkScala/src/test/resources/test_data/sample.count.txt")
  }

}
