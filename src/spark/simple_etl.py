#!/usr/bin/env python3
"""
Simplified Spark ETL Pipeline for Job Market Analysis
Works with basic Spark setup (no Delta Lake for now)
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

class SimpleJobMarketETL:
    """Simplified ETL pipeline using Apache Spark"""
    
    def __init__(self):
        self.spark = self._create_spark_session()
    
    def _create_spark_session(self):
        """Create Spark session"""
        return SparkSession.builder \
            .appName("JobMarketAnalysis") \
            .master("local[*]") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.hadoop.fs.defaultFS", "file:///") \
            .getOrCreate()
    
    def process_github_data(self):
        """Process GitHub Archive data with Spark"""
        print("🔄 Processing GitHub Archive data with Spark...")
        
        try:
            # Read JSON files in parallel
            github_df = self.spark.read \
                .option("multiline", "true") \
                .option("recursiveFileLookup", "true") \
                .json("data/raw/github/*.json.gz")
            
            # Extract key fields and add metadata
            processed_df = github_df.select(
                col("id").alias("event_id"),
                col("type").alias("event_type"),
                col("actor.login").alias("user_login"),
                col("repo.name").alias("repo_name"),
                col("created_at").alias("timestamp"),
                current_timestamp().alias("processing_time")
            ).withColumn("date", to_date(col("timestamp")))
            
            # Write to Parquet format
            output_path = "data/bronze/github_events"
            processed_df.write \
                .mode("overwrite") \
                .parquet(output_path)
            
            print(f"✅ Processed {processed_df.count()} GitHub events")
            return processed_df
            
        except Exception as e:
            print(f"❌ Error processing GitHub data: {e}")
            return None
    
    def process_stackoverflow_data(self):
        """Process StackOverflow survey data with Spark"""
        print("🔄 Processing StackOverflow survey data with Spark...")
        
        try:
            # Read all survey files
            survey_files = [
                "data/raw/stackoverflow/survey_2019.csv",
                "data/raw/stackoverflow/survey_2020.csv",
                "data/raw/stackoverflow/survey_2021.csv",
                "data/raw/stackoverflow/survey_2022.csv",
                "data/raw/stackoverflow/survey_2023.csv",
                "data/raw/stackoverflow/survey_2024.csv",
                "data/raw/stackoverflow/survey_2025.csv"
            ]
            
            # Load and write each survey separately to avoid schema mismatches
            wrote_any = False
            for file_path in survey_files:
                if Path(file_path).exists():
                    df = self.spark.read.option("header", "true").csv(file_path)
                    year = file_path.split("_")[-1].split(".")[0]
                    df = df.withColumn("survey_year", lit(year))
                    out = f"data/bronze/stackoverflow_surveys_{year}"
                    df.write.mode("overwrite").parquet(out)
                    wrote_any = True
            
            if wrote_any:
                print("✅ Wrote StackOverflow surveys by year to bronze")
                return None
            else:
                print("⚠️ No StackOverflow survey files found")
                return None
                
        except Exception as e:
            print(f"❌ Error processing StackOverflow data: {e}")
            return None
    
    def process_kaggle_jobs_data(self):
        """Process Kaggle job data with Spark"""
        print("🔄 Processing Kaggle job data with Spark...")
        
        try:
            # Read job postings
            jobs_df = self.spark.read \
                .option("header", "true") \
                .csv("data/raw/kaggle/archive-2/postings.csv")
            
            # Read salaries
            salaries_df = self.spark.read \
                .option("header", "true") \
                .csv("data/raw/kaggle/archive-2/jobs/salaries.csv")
            
            # Read companies
            companies_df = self.spark.read \
                .option("header", "true") \
                .csv("data/raw/kaggle/archive-2/companies/companies.csv")
            
            # Write to Parquet
            jobs_df.write \
                .mode("overwrite") \
                .parquet("data/bronze/job_postings")
            
            salaries_df.write \
                .mode("overwrite") \
                .parquet("data/bronze/job_salaries")
            
            companies_df.write \
                .mode("overwrite") \
                .parquet("data/bronze/companies")
            
            print(f"✅ Processed {jobs_df.count()} job postings, {salaries_df.count()} salaries, {companies_df.count()} companies")
            return jobs_df, salaries_df, companies_df
            
        except Exception as e:
            print(f"❌ Error processing Kaggle data: {e}")
            return None, None, None
    
    def create_silver_layer(self):
        """Create unified Silver layer from Bronze data"""
        print("🔄 Creating Silver layer with unified datasets...")
        
        try:
            # Read from Bronze layer
            salaries_df = self.spark.read.parquet("data/bronze/job_salaries")

            # Map available columns: choose median if available, else max, else min
            salary_amount = (
                when(col("med_salary").isNotNull(), col("med_salary"))
                .otherwise(when(col("max_salary").isNotNull(), col("max_salary"))
                .otherwise(col("min_salary")))
            )

            unified_salaries = salaries_df.select(
                col("job_id"),
                col("currency").alias("currency"),
                salary_amount.alias("salary_amount"),
                col("pay_period").alias("period"),
                current_timestamp().alias("created_at")
            ).filter(col("salary_amount").isNotNull())
            
            # Write to Silver layer
            unified_salaries.write \
                .mode("overwrite") \
                .parquet("data/silver/unified_salaries")
            
            print(f"✅ Created unified salary dataset with {unified_salaries.count()} records")
            return unified_salaries
            
        except Exception as e:
            print(f"❌ Error creating Silver layer: {e}")
            return None
    
    def run_full_pipeline(self):
        """Run complete ETL pipeline"""
        print("🚀 Starting Spark ETL Pipeline for 126GB+ Job Market Data")
        print("=" * 60)
        
        try:
            # Process all data sources
            self.process_github_data()
            self.process_stackoverflow_data()
            self.process_kaggle_jobs_data()
            
            # Create data lake layers
            self.create_silver_layer()
            
            print("\n🎉 Spark ETL Pipeline Complete!")
            print("📊 Data processed and stored in Parquet format")
            print("🔗 Bronze Layer: Raw data with metadata")
            print("🔗 Silver Layer: Cleaned, unified datasets")
            
        except Exception as e:
            print(f"❌ Pipeline failed: {str(e)}")
            raise
        finally:
            self.spark.stop()

def main():
    """Main function to run Spark ETL pipeline"""
    etl = SimpleJobMarketETL()
    etl.run_full_pipeline()

if __name__ == "__main__":
    main()
