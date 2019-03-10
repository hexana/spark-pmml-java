package main.java.com.exm.spark.funplay

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.Imputer
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.RFormula

class pmmlModel (session:SparkSession,inputPath:String,outputPath:String) extends java.io.Serializable{

  def execute(): Unit = {
    val testDatadf = session.read.format("csv").option("delimiter", ",").option("inferSchema", "true").option("header", "true").load(inputPath)
    testDatadf.describe("age").show()

    testDatadf.stat.crosstab("gender","smoking_status").show()
    val quantiles = testDatadf.stat.approxQuantile("bmi",Array(0.25,0.5,0.95),0.0)
    println(quantiles.toList)
    testDatadf.groupBy("smoking_status").count().show()
    val count=testDatadf.filter(testDatadf.col("stroke")===1 && testDatadf.col("gender") ==="Female").count()
    println(count.toInt)

    val missingInsertDF = testDatadf.na.fill("missing", Seq("smoking_status"))


    val imputer = new Imputer().setInputCols(Array("bmi"))
      .setOutputCols(Array("bmi_full"))
      .setStrategy("mean")

    val imputedDF=imputer.fit(missingInsertDF).transform(missingInsertDF)
    // imputedDF.show()

    val finalDF = imputedDF.drop(imputedDF.col("bmi"))
    // finalDF.show()

    val formula = new RFormula().setFormula("stroke ~ age + hypertension + heart_disease + gender + ever_married + work_type + bmi_full").setFeaturesCol("features").setLabelCol("label")
    val dtc  = new DecisionTreeClassifier()
    val pipeLine = new Pipeline().setStages(Array(formula,dtc))
    val Array(trainig,test)=finalDF.randomSplit(Array[Double](0.7,0.3))

    val pipelineModel = pipeLine.fit(trainig)
    val predictions =pipelineModel.transform(test)
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("stroke").setPredictionCol("prediction").setMetricName("accuracy")
    val values=   evaluator.evaluate(predictions)

    println(values)
    val schema = finalDF.schema
    PmmlCreator.getPmmlFile(pipelineModel,schema,outputPath)
  }

  }
