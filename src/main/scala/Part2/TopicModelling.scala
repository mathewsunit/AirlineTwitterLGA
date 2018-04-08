package Part2

import java.io.{BufferedWriter, OutputStreamWriter}
import java.util.regex.Pattern

import org.apache.hadoop.fs.{FSDataOutputStream, FileSystem, Path}
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions.{udf, _}
import org.apache.spark.sql.{Row, SparkSession}

object TopicModelling {
  val regex = "[\\.\\,\\:\\-\\!\\?\\n\\t,\\%\\#\\*\\|\\=\\(\\)\\\"\\>\\<\\/]"
  val pattern = Pattern.compile(regex)

  def clean: String => String = pattern.matcher(_).replaceAll(" ").split("[ ]+").mkString(" ")

  def main(args: Array[String]) = {
    val conf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("TfIdfSpark")
      .set("spark.driver.memory", "3g")
      .set("spark.executor.memory", "2g")

    val sc = SparkSession.builder.config(conf).getOrCreate()
    import sc.implicits._

    val df = sc.read.format("csv").option("header", "true").load(args(0))
    val df2 = df.filter(df("text").isNotNull)
    val cleanUdf = udf(clean)
    val tweets = df2.withColumn("text",  cleanUdf($"text"))
    val query = tweets.select("airline","airline_sentiment").rdd.map{ case Row(k: String, v: String) => (k,
      v match {
        case "positive"  => 5.0
        case "neutral"  => 2.5
        case "negative"  => 1.0
      })}.toDF("airline","sentiment")
    val inter1 = query.groupBy("airline").agg(mean("sentiment").alias("avg"))
    val inter2 = inter1.orderBy($"avg".desc).limit(2)
    val inter3 = tweets.join(inter2, inter2.col("airline") === tweets.col("airline")).withColumn("text", lower(col("text"))).withColumn("id",monotonically_increasing_id())
    val splits = inter3.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)
    val stopwords = Array("united","flight","usairways","americanair","southwestair","http","jetblue","thanks","service","hours","time","customer","hold","thank","please","plane","virginamerica")

    val tokenizer = new RegexTokenizer()
      .setPattern("[\\W_]+")
      .setMinTokenLength(4)
      .setInputCol("text")
      .setOutputCol("tokens")

    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setStopWords(stopwords)
      .setOutputCol("filtered")

    val vectorizer = new CountVectorizer()
      .setInputCol(remover.getOutputCol)
      .setOutputCol("features")
      .setVocabSize(10000)
      .setMinDF(5)

    import org.apache.spark.ml.clustering.LDA

    val lda = new LDA().setK(20).setMaxIter(50).setOptimizer("em")

    val pipeline = new Pipeline().setStages(Array(tokenizer, remover, vectorizer, lda))
    val model = pipeline.fit(training)

    import org.apache.spark.ml.clustering.DistributedLDAModel
    import org.apache.spark.ml.feature._

    val pipelineModel = pipeline.fit(tweets)
    val vectorizerModel = pipelineModel.stages(2).asInstanceOf[CountVectorizerModel]
    val ldaModel = pipelineModel.stages(3).asInstanceOf[DistributedLDAModel]
    val vocabList = vectorizerModel.vocabulary
    val termsIdx2Str = udf { (termIndices: Seq[Int]) => termIndices.map(idx => vocabList(idx)) }

    val topics = ldaModel.describeTopics(maxTermsPerTopic = 8)
      .withColumn("terms", termsIdx2Str(col("termIndices")))

    val stringify = udf((vs: Seq[String]) => s"""[${vs.mkString(",")}]""")

    val output = topics.select("terms").withColumn("terms", stringify($"terms")).collect().map(_.getString(0)).mkString("\n")

    val fs = FileSystem.get(sc.sparkContext.hadoopConfiguration)
    val path: Path = new Path(args(1))
    if (fs.exists(path)) {
      fs.delete(path, true)
    }
    val dataOutputStream: FSDataOutputStream = fs.create(path)
    val bw: BufferedWriter = new BufferedWriter(new OutputStreamWriter(dataOutputStream, "UTF-8"))
    bw.write(output)
    bw.close
  }
}