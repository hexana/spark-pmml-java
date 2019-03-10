# spark-pmml-java
creating a pmml file of the model generated by spark

This example shows a sample code:

•	To evaluate given data set

•	How to impute values for missing fields for continuous and categorical fields

•	How to use a machine learning model and save it as a pmml file and 

•	How to use the pmml file for prediction.

Usage
Intention for this example is to explain how one can evaluate a given dataset using spark and can save model file as pmml file using pmml libraries.
Example uses a data file in csv format (https://www.kaggle.com/asaumya/healthcare-dataset-stroke-data) and provides code sample to evaluate different columns. 
For example describe a continuous column field ‘age’ provides below stats:
+-------+------------------+
|summary|               age|
+-------+------------------+
|  count|             43400|
|   mean| 42.21789400921646|
| stddev|22.519648680503554|
|    min|              0.08|
|    max|              82.0|
+-------+------------------+

crosstab stats between two columns ('gender' and 'smoking_status') provides data as below:

+---------------------+---------------+------------+----+------+
|gender_smoking_status|formerly smoked|never smoked|null|smokes|
+---------------------+---------------+------------+----+------+
|                Other|              6|           2|   2|     1|
|                 Male|           3370|        5483|5991|  2880|
|               Female|           4117|       10568|7299|  3681|
+---------------------+---------------+------------+----+------+

spark also provides quantile function which helps to provide percentile data about a column ('bmi' - 25, 50 and 95 percentile):
(23.2, 27.7, 42.6)

groupby feature can provide counts corresponds to different values in a column (different count of 'smoking_status' column):
+---------------+-----+
| smoking_status|count|
+---------------+-----+
|           null|13292|
|         smokes| 6562|
|   never smoked|16053|
|formerly smoked| 7493|
+---------------+-----+

code shows example of imputing a value 'missing' in categorical column 'smoking_status' and mean value in continuous column 'bmi' and creates new column 'bmi_full'.

To Save a model as a pmml file using spark needs to use RFormula() function. it takes all the columns which is required to be taken as features as well as the label column also.
(In this case 'stroke' is the lable column)

A decision tree model has been used for the example.

Usages (converting model to pmml):
Download the project:
git clone https://github.com/hexana/spark-pmml-java.git

build the project
cd spark-pmml-java; mvn clean install

Example:
spark-submit --packages org.jpmml:jpmml-sparkml:1.4.8 --class com.exm.spark.funplay.Generator --master local[*] <folder location of project jar>/spark-exm-1.0-SNAPSHOT.jar <csv file location>/train_2v.csv <outputlocation>

Usages (using jpmml library to read pmml file and making prediction)
Download the project:
git clone https://github.com/hexana/jpmml-java.git

build the project
cd jpmml-java; mvn clean install

Example:
java -jar spark-pmml-0.0.1-SNAPSHOT-jar-with-dependencies.jar <pmml file path>

stroke {result=0, probability_entries=[0=0.9739583333333334, 1=0.026041666666666668], entityId=16, confidence_entries=[]}












