from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr
#spark-submit --packages com.databricks:spark-xml_2.12:0.14.0 dataset_generation.py
# Initialize Spark Session
spark = SparkSession.builder \
    .appName("SCARA Data Processing") \
    .config("spark.jars.packages", "com.databricks:spark-xml_2.12:0.14.0") \
    .getOrCreate()

# Define path to the XML files
path_to_data = "C:/Users/vigne/OneDrive/Documents/ML/Predictive-Maintenance-master/Predictive-Maintenance-master/dataset/demo_data/*.dat"

# Load XML files
df = spark.read.format("xml") \
    .option("rowTag", "HistoricalTextData") \
    .load(path_to_data)

# Show initial structure
df.show()

# Extract and pivot data
pivoted_df = df.groupBy("TimeStamp").pivot("TagName").agg(expr("first(TagValue)"))

# Show the transformed DataFrame
pivoted_df.show()

# Optionally, save the DataFrame to a CSV file
# pivoted_df.write.format("csv").mode("overwrite").option("header", True).save("C:/Users/vigne/OneDrive/Documents/ML/Predictive-Maintenance-master/Predictive-Maintenance-master/dataset/raw_data")
pivoted_df.coalesce(1).write.format("csv").mode("overwrite").option("header", True).save("C:/Users/vigne/OneDrive/Documents/ML/Predictive-Maintenance-master/Predictive-Maintenance-master/dataset/raw_data")

# Stop Spark Session
spark.stop()
