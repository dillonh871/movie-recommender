# Import required libraries and create a Spark session
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col, explode
spark = SparkSession.builder.appName("Collaborative filtering").getOrCreate()

# Load movies and ratings data from CSV files
moviesDF = spark.read.options(header="True", inferSchema="True").csv("/FileStore/tables/movies.csv")
ratingsDF = spark.read.options(header="True", inferSchema="True").csv("/FileStore/tables/ratings.csv")

# Display the movies data
display(moviesDF)

# Display the ratings data
display(ratingsDF)

# Join the ratings and movies data on the 'movieId' column
ratings = ratingsDF.join(moviesDF, 'movieId', 'left')

# Split the data into training and testing sets
(train, test) = ratings.randomSplit([0.8, 0.2])

# Get the total count of ratings
ratings.count()

# Print the count of data in the training set
print(train.count())

# Show a sample of the training data
train.show()

# Print the count of data in the testing set
print(test.count())

# Show a sample of the testing data
test.show()

# Create an ALS (Alternating Least Squares) recommendation model
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", nonnegative=True, implicitPrefs=False, coldStartStrategy="drop")

# Define a parameter grid for hyperparameter tuning
param_grid = ParamGridBuilder() \
            .addGrid(als.rank, [10, 50, 100, 150]) \
            .addGrid(als.regParam, [.01, .05, .1, .15]) \
            .build()

# Create an evaluator to measure the RMSE (Root Mean Squared Error)
evaluator = RegressionEvaluator(
           metricName="rmse", 
           labelCol="rating", 
           predictionCol="prediction")

# Create a cross-validator for hyperparameter tuning
cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)

# Fit the model to the training data and find the best model
model = cv.fit(train)
best_model = model.bestModel

# Make predictions on the test data using the best model
test_predictions = best_model.transform(test)

# Calculate and print the RMSE for the test predictions
RMSE = evaluator.evaluate(test_predictions)
print(RMSE)

# Create recommendations for all users
recommendations = best_model.recommendForAllUsers(5)

# Store the recommendations in a DataFrame
df = recommendations

# Display the recommendations
display(df)

# Explode the recommendations column to get individual movie recommendations
df2 = df.withColumn("movieid_rating", explode("recommendations"))

# Display the exploded recommendations
display(df2)

# Select and display the userId, recommended movieId, and rating
display(df2.select("userId", col("movieid_rating.movieId"), col("movieid_rating.rating")))
