package com.exm.spark.funplay

import main.java.com.exm.spark.funplay.{DataCheck, pmmlModel}
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

object Generator {
  def main(args: Array[String]){
      val inputPath =args(0)
      val outputPath = args(1)
      val conf =new SparkConf()
      val spark = SparkSession.builder().appName("TestApp").config(conf).getOrCreate()
     // val job = new DataCheck(spark,inputPath,outputPath)

      val job = new pmmlModel(spark,inputPath,outputPath)
      job.execute()





  }

}
