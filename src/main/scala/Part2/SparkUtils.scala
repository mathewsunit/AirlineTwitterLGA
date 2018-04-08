package Part2

import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.rdd.RDD


object SparkUtils {

  def evaluateModel(predictionAndLabels: RDD[(Double, Double)], outputFile: String, title:String):String = {
    val pw = new StringBuffer()
    pw.append("\n")
    pw.append("\n")
    pw.append("=================== "+title+" =======================")
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val cfMatrix = metrics.confusionMatrix

    val stringOut =
      s"""
         |=================== Confusion matrix ==========================
         |          | %-15s                     %-15s
         |----------+----------------------------------------------------
         |Actual = 0| %-15f                     %-15f
         |Actual = 1| %-15f                     %-15f
         |===============================================================
       """.stripMargin.format("Predicted = 0", "Predicted = 1",
        cfMatrix.apply(0, 0), cfMatrix.apply(0, 1), cfMatrix.apply(1, 0), cfMatrix.apply(1, 1))

    pw.append(stringOut)
    pw.append("\nACCURACY " + ((cfMatrix(0,0) + cfMatrix(1,1))/(cfMatrix(0,0) + cfMatrix(0,1) + cfMatrix(1,0) + cfMatrix(1,1))))


    //cfMatrix.toArray

    val fpr = metrics.falsePositiveRate(0)
    val tpr = metrics.truePositiveRate(0)

    val analysis =
      s"""
         |False positive rate = $fpr
         |True positive rate = $tpr
     """.stripMargin

    pw.append(analysis)

    val m = new BinaryClassificationMetrics(predictionAndLabels)
    pw.append("PR " + m.areaUnderPR())
    pw.append("AUC " + m.areaUnderROC())
    pw.toString
  }
}