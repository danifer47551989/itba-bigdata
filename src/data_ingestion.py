import mlflow 
import logging
import datetime
import json
import pyspark.sql.functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DateType,
    FloatType,
    TimestampType,
    BooleanType,
)
from pyspark.sql import DataFrame
from pyspark.sql.window import Window

# If running outside Databricks, create a SparkSession only if `spark` is not already defined.
try:
    spark
except NameError:
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("itba-bigdata").getOrCreate()

SOURCE_PATH = "/Volumes/workspace/sentiment_analysis/raw"
BRONZE_PATH = "/Volumes/workspace/sentiment_analysis/bronze"
SILVER_PATH = "/Volumes/workspace/sentiment_analysis/silver"
GOLD_OUT_PATH = "/Volumes/workspace/sentiment_analysis/gold"
REQUIRED_FIELDS = [
    "review_id", "product_id", "customer_id",
    "star_rating", "review_date", "review_body"
]

SENTIMENT_MAP = {
    "negative": [1, 2],
    "neutral": [3],
    "positive": [4, 5]
}

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

bronze_schema = StructType([
    StructField("marketplace", StringType(), True),
    StructField("customer_id", StringType(), True),
    StructField("review_id", StringType(), True),
    StructField("product_id", StringType(), True),
    StructField("product_parent", StringType(), True),
    StructField("product_title", StringType(), True),
    StructField("product_category", StringType(), True),
    StructField("star_rating", IntegerType(), True),
    StructField("helpful_votes", IntegerType(), True),
    StructField("total_votes", IntegerType(), True),
    StructField("vine", StringType(), True),
    StructField("verified_purchase", StringType(), True),
    StructField("review_headline", StringType(), True),
    StructField("review_body", StringType(), True),
    StructField("review_date", DateType(), True)
])

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


def read_source(spark) -> DataFrame:
    logging.info(f"Reading source data from {SOURCE_PATH}")
    df = (
        spark.read
        .option("header", "true")
        .option("sep", "\t")
        .schema(bronze_schema)
        .csv(SOURCE_PATH)
    )
    return df


def add_metadata(df: DataFrame) -> DataFrame:
    return df.withColumn("ingestion_timestamp", F.current_timestamp()) \
             .withColumn("source_file", F.lit(SOURCE_PATH))


def write_bronze(df: DataFrame):
    df.write.mode("overwrite").parquet(BRONZE_PATH)
    logging.info(f"Bronze data written to {BRONZE_PATH}")


def bronze_ingestion():
    with mlflow.start_run() as run:
        df = read_source(spark)
        row_count = df.count()
        mlflow.log_metric("rows_read", row_count)

        df = add_metadata(df)
        df.show(5)
        write_bronze(df)

        mlflow.log_param("source", SOURCE_PATH)
        mlflow.log_param("output", BRONZE_PATH)
        mlflow.log_metric("columns", len(df.columns))


def bronze_validation():
    EXPERIMENT_NAME = "bronze_validation"
    mlflow.set_experiment(f'/Users/lhermannsauer@itba.edu.ar/{EXPERIMENT_NAME}')

    print(f"Reading Bronze data from: {BRONZE_PATH}")
    df = spark.read.parquet(BRONZE_PATH)
    print(f"Loaded {df.count():,} rows, {len(df.columns)} columns")

    print("Schema:")
    df.printSchema()

    expected_columns = ['marketplace',
                        'customer_id',
                        'review_id',
                        'product_id',
                        'product_parent',
                        'product_title',
                        'product_category',
                        'star_rating',
                        'helpful_votes',
                        'total_votes',
                        'vine',
                        'verified_purchase',
                        'review_headline',
                        'review_body',
                        'review_date',
                        'ingestion_timestamp',
                        'source_file']

    missing_cols = [c for c in expected_columns if c not in df.columns]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
    else:
        print("All expected columns present.")

    metrics = {}

    metrics["row_count"] = df.count()
    metrics["column_count"] = len(df.columns)
    metrics["distinct_reviews"] = df.select(
        F.countDistinct("review_id")).first()[0]
    metrics["distinct_products"] = df.select(
        F.countDistinct("product_id")).first()[0]

    if "review_date" in df.columns:
        date_summary = df.select(
            F.min("review_date").alias("min_date"),
            F.max("review_date").alias("max_date")
        ).first()
        metrics["min_review_date"] = date_summary["min_date"]
        metrics["max_review_date"] = date_summary["max_date"]

    # === Null analysis ===
    null_counts = (
        df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c)
                for c in df.columns])
    )
    print("Null counts:")
    null_counts.show(5, truncate=False)

    # === Duplicate detection ===
    if "review_id" in df.columns:
        dup_count = df.groupBy("review_id").count().filter("count > 1").count()
        metrics["duplicate_reviews"] = dup_count
        print(f"Duplicated review_id count: {dup_count}")

    # === Log results to MLflow ===
    with mlflow.start_run(run_name="bronze_validation_run"):
        mlflow.log_param("min_review_date", str(
            metrics.pop("min_review_date", None)))
        mlflow.log_param("max_review_date", str(
            metrics.pop("max_review_date", None)))
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)
        mlflow.log_param("bronze_path", BRONZE_PATH)
        mlflow.log_param("columns", ",".join(df.columns))
        # Save schema as artifact
        schema_path = "../data/bronze/schema.json"
        mlflow.log_artifact(schema_path)

    print("Metrics and schema logged to MLflow")

    # === Summary output ===
    print("=== Summary ===")
    for k, v in metrics.items():
        print(f"{k:25s}: {v}")

# SILVER DATA ingestion
def silver_ingestion():
    # Load Bronze data
    bronze_df = spark.read.parquet(BRONZE_PATH)
    bronze_count = bronze_df.count()

    # Type conversions
    silver_df = (
        bronze_df
        .withColumn("star_rating", F.col("star_rating").cast(FloatType()))
        .withColumn("helpful_votes", F.col("helpful_votes").cast(IntegerType()))
        .withColumn("total_votes", F.col("total_votes").cast(IntegerType()))
        .withColumn("vine", F.when(F.col("vine") == "Y", F.lit(True)).otherwise(F.lit(False)))
        .withColumn("verified_purchase", F.when(F.col("verified_purchase") == "Y", F.lit(True)).otherwise(F.lit(False)))
        .withColumn("review_date", F.to_date("review_date", "yyyy-MM-dd"))
        .withColumn("ingestion_timestamp", F.to_timestamp("ingestion_timestamp"))
    )

    # Standardize column names
    for c in silver_df.columns:
        silver_df = silver_df.withColumnRenamed(c, c.lower())

    # Drop nulls in essential fields
    required_fields = ["review_id", "product_id",
                    "customer_id", "star_rating", "review_date", "review_body"]
    silver_df = silver_df.dropna(subset=required_fields)

    # Keep ratings within valid range
    silver_df = silver_df.filter(
        (F.col("star_rating") >= 1) & (F.col("star_rating") <= 5.0))

    # Deduplicate
    silver_df = silver_df.dropDuplicates(["review_id"])

    # Log metrics
    silver_count = silver_df.count()
    invalid_rows = bronze_count - silver_count
    retention_ratio = round(silver_count / bronze_count, 4)

    with mlflow.start_run():

        mlflow.log_param("bronze_rows", bronze_count)
        mlflow.log_param("silver_rows", silver_count)
        mlflow.log_metric("invalid_rows", invalid_rows)
        mlflow.log_metric("retention_ratio", retention_ratio)
        mlflow.log_param(
            "process_date", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Save Silver Data
        silver_df.write.mode("overwrite").parquet(SILVER_PATH)


        # Sanity Check
        print(f"Bronze count: {bronze_count}")
        print(f"Silver count: {silver_count}")
        print(f"Retention ratio: {retention_ratio}")


# GOLD DATA INGESTION
def basic_text_cleaning(df, text_col="review_body", out_col="clean_text"):
    expr = F.col(text_col)
    expr = F.lower(F.regexp_replace(expr, r"\s+", " "))
    expr = F.regexp_replace(expr, r"<[^>]+>", '')
    expr = F.regexp_replace(expr, r"[“”«»„‟]", '"')
    df = df.withColumn(out_col, expr)
    df = df.withColumn("n_chars", F.length(F.col(out_col)))
    df = df.withColumn("n_words", F.size(F.split(F.col(out_col), " ")))
    return df

def map_ratings_to_labels(df, rating_col="star_rating", out_col="sentiment_label"):
    mapping = SENTIMENT_MAP
    expr = F.when(F.col(rating_col).isin(mapping["negative"]), F.lit("negative")) \
        .when(F.col(rating_col).isin(mapping["neutral"]), F.lit("neutral")) \
        .when(F.col(rating_col).isin(mapping["positive"]), F.lit("positive")) \
        .otherwise(F.lit("neutral"))
    return df.withColumn(out_col, expr)


def gold_ingestion():

    df = spark.read.parquet(SILVER_PATH)
    print(
        f"Loaded Silver dataset with {df.count()} rows and {len(df.columns)} columns.")
    missing = [c for c in REQUIRED_FIELDS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # Filter invalid or incomplete rows
    df = df.filter(
        F.col("review_body").isNotNull() &
        (F.length(F.trim(F.col("review_body"))) > 5) &
        (F.col("star_rating").isNotNull())
    )
    df = df.withColumn("star_rating", F.col("star_rating").cast(IntegerType()))
    df = df.filter(F.col("star_rating").between(1, 5))
    
    df = basic_text_cleaning(df)
    df = map_ratings_to_labels(df)


    product_window = Window.partitionBy(
        "product_id").orderBy(F.col("review_date").desc())
    user_window = Window.partitionBy("user_id").orderBy("review_date")

    df = df.withColumn("review_rank", F.rank().over(product_window))
    df = df.withColumn("avg_sentiment_product", F.avg(
        "sentiment_label").over(product_window.rowsBetween(-5, 0)))
    
    selected_cols = [
        "review_id", "product_id", "customer_id",
        "clean_text", "sentiment_label", "star_rating",
        "review_date", "helpful_votes", "total_votes", "verified_purchase"
        ]   
    selected_cols = [c for c in selected_cols if c in df.columns]

    df_gold = df.select(*selected_cols)

    df_gold.write.mode("overwrite").parquet(GOLD_OUT_PATH)

    created_at = df_gold.select(F.current_timestamp().alias(
        "created_at")).first()["created_at"]

    meta = {
        # convert to string if you plan to JSON dump
        "created_at": str(created_at),
        "total_rows": df_gold.count(),
        "label_distribution": {
            r["sentiment_label"]: r["count"]
            for r in df_gold.groupBy("sentiment_label").count().collect()
        }
    }

    with mlflow.start_run():
        mlflow.log_param("total_rows", df_gold.count())
        mlflow.log_param("n_columns", len(df_gold.columns))
        mlflow.log_param("columns", ", ".join(df_gold.columns))
        # Label distribution
        label_dist = df_gold.groupBy("sentiment_label").count().collect()
        total = sum([r["count"] for r in label_dist])

        for r in label_dist:
            label = r["sentiment_label"]
            ratio = r["count"] / total
            mlflow.log_metric(f"label_ratio_{label}", ratio)

        # Average review length
        # Use top-level `F` imported at module scope instead of re-importing here
        avg_length = df_gold.select(F.avg(F.length("clean_text"))).first()[0]
        mlflow.log_metric("avg_review_length", float(avg_length))

        # Missing data ratio (quick data health check)
        missing_ratios = {
            col: df_gold.filter(F.col(col).isNull()).count() / total
            for col in df_gold.columns
        }
        for col, ratio in missing_ratios.items():
            mlflow.log_metric(f"missing_ratio_{col}", ratio)

        completeness_score = 1 - \
            sum(missing_ratios.values()) / len(df_gold.columns)
        balance_score = 1 - \
            max(abs(r["count"] - total / len(label_dist)) /
                total for r in label_dist)

        mlflow.log_dict("metadata", json.dumps(meta))
        mlflow.log_metric("completeness_score", completeness_score)
        mlflow.log_metric("balance_score", balance_score)

def main():
    bronze_ingestion()
    bronze_validation()
    silver_ingestion()
    gold_ingestion()

if __name__ == "__main__":
    main()

