name := "Assignment3Part2"

version := "0.1"

scalaVersion := "2.11.8"

libraryDependencies ++= {
  val sparkVer = "2.3.0"
  val openCsvVer = "4.1"
  Seq(
    "org.apache.spark" %% "spark-core" % sparkVer,
    "org.apache.spark" %% "spark-mllib" % sparkVer,
    "org.apache.spark" %% "spark-sql" % sparkVer,
    "com.opencsv" % "opencsv" % openCsvVer
  )
}