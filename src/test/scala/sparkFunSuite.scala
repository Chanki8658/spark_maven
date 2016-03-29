
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.scalatest.FunSuite

/**
  * Created by G01042336 on 28/10/2015.
  */
class SparkFunSuite extends FunSuite {

  def localTest(name : String)(f : SparkContext => Unit) : Unit = {
    this.test(name) {
      val conf = new SparkConf()
        .setAppName(name)
        .setMaster("local")
        .set("spark.default.parallelism", "1")
      val sc = new SparkContext(conf)
      try {
        f(sc)
      } finally {
        sc.stop()
      }
    }
  }
}
