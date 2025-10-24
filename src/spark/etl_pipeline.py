#!/usr/bin/env python3
"""
Apache Spark ETL Pipeline for Job Market Analysis
Implements distributed processing for 126GB+ dataset
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from delta.tables import DeltaTable

class JobMarketSparkETL:
    """Distributed ETL pipeline using Apache Spark"""
    
    def __init__(self):
        self.spark = self._create_spark_session()
        self.setup_delta_lake()
    
    def _create_spark_session(self):
        """Create Spark session with Delta Lake support"""
        return SparkSession.builder \
            .appName("JobMarketAnalysis") \
            .master("local[*]") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.0.0") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .config("spark.hadoop.fs.defaultFS", "file:///") \
            .config("spark.sql.warehouse.dir", str(Path.cwd() / "spark-warehouse")) \
            .getOrCreate()
    
    def setup_delta_lake(self):
        """Initialize Delta Lake tables"""
        # Create Delta tables for each layer
        self.bronze_path = "data/delta/bronze"
        self.silver_path = "data/delta/silver"
        self.gold_path = "data/delta/gold"
        # Parquet mirror paths for compatibility with non-Delta consumers
        self.parquet_silver_path = "data/silver"
        self.parquet_gold_path = "data/gold"
        
        # Ensure directories exist
        Path(self.bronze_path).mkdir(parents=True, exist_ok=True)
        Path(self.silver_path).mkdir(parents=True, exist_ok=True)
        Path(self.gold_path).mkdir(parents=True, exist_ok=True)
        Path(self.parquet_silver_path).mkdir(parents=True, exist_ok=True)
        Path(self.parquet_gold_path).mkdir(parents=True, exist_ok=True)
    
    def process_github_data(self):
        """Process GitHub Archive data with Spark"""
        print("🔄 Processing GitHub Archive data with Spark...")
        
        # Read JSON files in parallel from local filesystem
        gh_path = str((Path.cwd() / "data" / "raw" / "github").resolve())
        github_df = self.spark.read \
            .option("multiline", "true") \
            .option("recursiveFileLookup", "true") \
            .json(gh_path)
        
        # Extract key fields and add metadata
        processed_df = github_df.select(
            col("id").alias("event_id"),
            col("type").alias("event_type"),
            col("actor.login").alias("user_login"),
            col("repo.name").alias("repo_name"),
            col("created_at").alias("timestamp"),
            col("payload").alias("payload"),
            current_timestamp().alias("processing_time")
        ).withColumn("date", to_date(col("timestamp")))
        
        # Write to Delta Lake Bronze layer
        processed_df.write \
            .format("delta") \
            .mode("overwrite") \
            .option("mergeSchema", "true") \
            .save(f"{self.bronze_path}/github_events")
        
        print(f"✅ Processed {processed_df.count()} GitHub events")
        return processed_df
    
    def process_stackoverflow_data(self):
        """Process StackOverflow survey data with Spark"""
        print("🔄 Processing StackOverflow survey data with Spark...")
        
        # Read all survey files
        base_so = Path.cwd() / "data" / "raw" / "stackoverflow"
        survey_files = [str((base_so / f"survey_{y}.csv").resolve()) for y in [2019,2020,2021,2022,2023,2024,2025]]
        
        # Union all surveys (sanitize column names)
        survey_dfs = []
        for file_path in survey_files:
            if Path(file_path).exists():
                import re as _re
                df = self.spark.read.option("header", "true").csv(file_path)
                safe_cols = [
                    _re.sub(r"[ ,;{}()=\n\t]", "_", c).strip("_") for c in df.columns
                ]
                df = df.toDF(*safe_cols)
                df = df.withColumn("survey_year", lit(file_path.split("_")[-1].split(".")[0]))
                survey_dfs.append(df)
        
        if survey_dfs:
            combined_df = survey_dfs[0]
            for df in survey_dfs[1:]:
                combined_df = combined_df.unionByName(df, allowMissingColumns=True)
            
            # Write to Delta Lake
            combined_df.write \
                .format("delta") \
                .mode("overwrite") \
                .option("mergeSchema", "true") \
                .save(f"{self.bronze_path}/stackoverflow_surveys")
            
            print(f"✅ Processed {combined_df.count()} survey responses")
            return combined_df
        else:
            print("⚠️ No StackOverflow survey files found")
            return None
    
    def process_kaggle_jobs_data(self):
        """Process Kaggle job data with Spark"""
        print("🔄 Processing Kaggle job data with Spark...")
        
        # Read job postings
        jobs_df = self.spark.read \
            .option("header", "true") \
            .csv(str((Path.cwd() / "data" / "raw" / "kaggle" / "archive-2" / "postings.csv").resolve()))
        
        # Read salaries
        salaries_df = self.spark.read \
            .option("header", "true") \
            .csv(str((Path.cwd() / "data" / "raw" / "kaggle" / "archive-2" / "jobs" / "salaries.csv").resolve()))
        
        # Read companies
        companies_df = self.spark.read \
            .option("header", "true") \
            .csv(str((Path.cwd() / "data" / "raw" / "kaggle" / "archive-2" / "companies" / "companies.csv").resolve()))
        
        # Write to Delta Lake
        jobs_df.write \
            .format("delta") \
            .mode("overwrite") \
            .save(f"{self.bronze_path}/job_postings")
        
        salaries_df.write \
            .format("delta") \
            .mode("overwrite") \
            .save(f"{self.bronze_path}/job_salaries")
        
        companies_df.write \
            .format("delta") \
            .mode("overwrite") \
            .save(f"{self.bronze_path}/companies")
        
        print(f"✅ Processed {jobs_df.count()} job postings, {salaries_df.count()} salaries, {companies_df.count()} companies")
        return jobs_df, salaries_df, companies_df

    def process_bls_data(self):
        """Process BLS employment data from Excel files to Delta Bronze"""
        print("🔄 Processing BLS data (Excel) with pandas → Spark...")
        from pathlib import Path as _Path
        import pandas as _pd
        bls_raw = _Path("data/raw/bls")
        if not bls_raw.exists():
            print("⚠️ No BLS folder found under data/raw/bls")
            return None
        frames = []
        for fp in sorted(bls_raw.glob("*.xlsx")):
            try:
                df = _pd.read_excel(fp)
                # Normalize common columns if present
                cols = {c.lower(): c for c in df.columns}
                # Keep flexible mapping
                series = df[cols.get('series_id')] if 'series_id' in cols else None
                year = df[cols.get('year')] if 'year' in cols else None
                period = df[cols.get('period')] if 'period' in cols else None
                value = df[cols.get('value')] if 'value' in cols else None
                tmp = _pd.DataFrame({
                    'series_id': series if series is not None else _pd.NA,
                    'year': year if year is not None else _pd.NA,
                    'period': period if period is not None else _pd.NA,
                    'value': value if value is not None else _pd.NA,
                    'source_file': fp.name
                })
                frames.append(tmp)
            except Exception as e:
                print(f"⚠️ Skipping BLS file {fp.name}: {e}")
        if not frames:
            print("⚠️ No BLS Excel files parsed")
            return None
        pdf = _pd.concat(frames, ignore_index=True)
        sdf = self.spark.createDataFrame(pdf)
        sdf.write \
            .format("delta") \
            .mode("overwrite") \
            .option("mergeSchema", "true") \
            .save(f"{self.bronze_path}/bls")
        print(f"✅ Processed BLS rows: {sdf.count()}")
        return sdf
    
    def create_silver_layer(self):
        """Create unified Silver layer from Bronze data"""
        print("🔄 Creating Silver layer with unified datasets...")
        
        # Read from Delta Lake Bronze layer
        github_df = None
        salaries_df = None
        bls_df = None
        so_df = None
        try:
            github_df = self.spark.read.format("delta").load(f"{self.bronze_path}/github_events")
        except Exception:
            pass
        try:
            salaries_df = self.spark.read.format("delta").load(f"{self.bronze_path}/job_salaries")
        except Exception:
            pass
        try:
            bls_df = self.spark.read.format("delta").load(f"{self.bronze_path}/bls")
        except Exception:
            pass
        try:
            so_df = self.spark.read.format("delta").load(f"{self.bronze_path}/stackoverflow_surveys")
        except Exception:
            pass
        
        # Create unified salary dataset with GitHub activity correlation
        unified_salaries = None
        if salaries_df is not None:
            # Flexible mapping: prefer med_salary, else max_salary, else min_salary, else salary
            cols = salaries_df.columns
            amount = None
            if "med_salary" in cols:
                amount = col("med_salary")
            elif "max_salary" in cols:
                amount = col("max_salary")
            elif "min_salary" in cols:
                amount = col("min_salary")
            elif "salary" in cols:
                amount = col("salary")
            else:
                amount = None
            currency_col = "currency" if "currency" in cols else ("salary_currency" if "salary_currency" in cols else None)
            period_col = "pay_period" if "pay_period" in cols else ("salary_period" if "salary_period" in cols else None)
            select_exprs = [col("job_id")]
            if currency_col:
                select_exprs.append(col(currency_col).alias("currency"))
            else:
                select_exprs.append(lit("USD").alias("currency"))
            if amount is not None:
                select_exprs.append(amount.alias("salary_amount"))
            else:
                select_exprs.append(lit(None).cast("double").alias("salary_amount"))
            if period_col:
                select_exprs.append(col(period_col).alias("period"))
            else:
                select_exprs.append(lit("YEARLY").alias("period"))
            select_exprs.append(current_timestamp().alias("created_at"))
            unified_salaries = salaries_df.select(*select_exprs).filter(col("salary_amount").isNotNull())
            # Write to Silver layer (Delta + Parquet mirror)
            unified_salaries.write \
                .format("delta") \
                .mode("overwrite") \
                .save(f"{self.silver_path}/unified_salaries")
            print(f"✅ Created unified salary dataset with {unified_salaries.count()} records")
            unified_salaries.write \
                .mode("overwrite") \
                .parquet(f"{self.parquet_silver_path}/unified_salaries")
        
        # Create normalized BLS employment silver
        bls_silver = None
        if bls_df is not None:
            bls_silver = bls_df.select(
                col("series_id"),
                col("year").cast("int").alias("year"),
                col("period").alias("period"),
                col("value").cast("double").alias("value"),
                current_timestamp().alias("created_at")
            )
            bls_silver.write \
                .format("delta") \
                .mode("overwrite") \
                .save(f"{self.silver_path}/bls_employment")
            print(f"✅ Created BLS silver with {bls_silver.count()} records")
            bls_silver.write \
                .mode("overwrite") \
                .parquet(f"{self.parquet_silver_path}/bls_employment")

        # Create normalized StackOverflow silver (basic schema)
        so_silver = None
        if so_df is not None:
            cols = so_df.columns
            def pick(cands):
                for c in cands:
                    if c in cols:
                        return c
                return None
            country_col = pick(["Country","country"]) or "Country"
            devtype_col = pick(["DevType","devtype"]) or "DevType"
            lang_col = pick(["LanguageHaveWorkedWith","languagehaveworkedwith"]) or "LanguageHaveWorkedWith"
            comp_col = pick(["CompTotal","comptotal","ConvertedCompYearly","convertedcompyearly"]) or "CompTotal"
            year_col = pick(["survey_year","Year","year"]) or "survey_year"
            # Select and normalize
            so_silver = so_df.select(
                col(country_col).alias("country"),
                col(devtype_col).alias("devtype"),
                col(lang_col).alias("languages"),
                col(comp_col).alias("compensation"),
                col(year_col).cast("int").alias("year"),
                current_timestamp().alias("created_at")
            )
            so_silver.write.format("delta").mode("overwrite").save(f"{self.silver_path}/stackoverflow_basic")
            so_silver.write.mode("overwrite").parquet(f"{self.parquet_silver_path}/stackoverflow_basic")
            print(f"✅ Created StackOverflow silver: stackoverflow_basic")

        return unified_salaries
    
    def create_gold_layer(self):
        """Create ML-ready Gold layer"""
        print("🔄 Creating Gold layer for ML models...")
        
        # Read from Silver layer
        salaries_df = self.spark.read.format("delta").load(f"{self.silver_path}/unified_salaries")
        
        # Feature engineering for ML
        ml_features = salaries_df.select(
            col("job_id"),
            col("salary_amount").cast("double").alias("salary"),
            col("currency"),
            col("period"),
            when(col("currency") == "USD", 1).otherwise(0).alias("is_usd"),
            when(col("period") == "YEARLY", 1).otherwise(0).alias("is_yearly"),
            current_timestamp().alias("feature_created_at")
        ).filter(col("salary").isNotNull() & (col("salary") > 0))
        
        # Write to Gold layer
        ml_features.write \
            .format("delta") \
            .mode("overwrite") \
            .save(f"{self.gold_path}/ml_features")
        
        print(f"✅ Created ML features with {ml_features.count()} records")
        # Parquet mirror for compatibility
        ml_features.write \
            .mode("overwrite") \
            .parquet(f"{self.parquet_gold_path}/ml_features")
        return ml_features

    def create_gold_postings_aggregates(self):
        """Create Gold aggregates from job postings: skills and locations"""
        print("🔄 Creating Gold aggregates from job postings...")
        try:
            postings = self.spark.read.format("delta").load(f"{self.bronze_path}/job_postings")
        except Exception:
            print("⚠️ job_postings Delta table not found; skipping postings gold")
            return None, None
        # Flexible column detection
        cols = postings.columns
        def pick(cands):
            for c in cands:
                if c in cols:
                    return c
            return None
        title_col = pick(["title","job_title","position","role","posting_title"]) or "title"
        skills_col = pick(["skills","key_skills","skills_mentioned","tags","tech_stack"]) 
        loc_col = pick(["location","city","state","country","job_location"]) 
        # Skills aggregate
        skills_agg = None
        if skills_col is not None:
            tokens = regexp_replace(lower(col(skills_col)), "[;/]", ",")
            exploded = split(tokens, ",")
            skills_agg = postings.select(explode(exploded).alias("skill")) \
                .withColumn("skill", trim(col("skill"))) \
                .filter(length(col("skill")) > 0) \
                .groupBy("skill").count() \
                .orderBy(col("count").desc())
            # Write
            skills_agg.write.format("delta").mode("overwrite").save(f"{self.gold_path}/skills_demand")
            skills_agg.write.mode("overwrite").parquet(f"{self.parquet_gold_path}/skills_demand")
            print("✅ Wrote skills_demand gold")
        else:
            print("⚠️ No skills column found; skipping skills gold")
        # Location aggregate
        locations_agg = None
        if loc_col is not None:
            locations_agg = postings.select(lower(trim(col(loc_col))).alias("location")) \
                .groupBy("location").count() \
                .orderBy(col("count").desc())
            locations_agg.write.format("delta").mode("overwrite").save(f"{self.gold_path}/location_hotspots")
            locations_agg.write.mode("overwrite").parquet(f"{self.parquet_gold_path}/location_hotspots")
            print("✅ Wrote location_hotspots gold")
        else:
            print("⚠️ No location column found; skipping location gold")
        return skills_agg, locations_agg

    def create_gold_skills_time_series(self):
        """Create monthly skill demand and skill co-occurrence gold tables."""
        print("🔄 Creating Gold monthly skill demand and co-occurrence...")
        try:
            postings = self.spark.read.format("delta").load(f"{self.bronze_path}/job_postings")
        except Exception:
            print("⚠️ job_postings Delta table not found; skipping skills time-series gold")
            return None, None
        cols = postings.columns
        def pick(cands):
            for c in cands:
                if c in cols:
                    return c
            return None
        skills_col = pick(["skills","key_skills","skills_mentioned","tags","tech_stack"]) 
        date_col = pick(["posted_date","date_posted","posting_date","created_at","created","post_date"]) 
        if skills_col is None or date_col is None:
            print("⚠️ Missing skills or date column; skipping skills time-series gold")
            return None, None
        # Normalize date -> year-month and skills tokens
        df = postings.withColumn("post_ts", to_timestamp(col(date_col))) \
            .withColumn("year_month", date_format(col("post_ts"), "yyyy-MM")) \
            .withColumn("_skills_raw", regexp_replace(lower(col(skills_col)), "[;/]", ","))
        tokens = split(col("_skills_raw"), ",")
        exploded = df.select(col("year_month"), explode(tokens).alias("skill")) \
            .withColumn("skill", trim(col("skill"))) \
            .filter(length(col("skill")) > 0)
        monthly_skill_demand = exploded.groupBy("year_month","skill").count().withColumnRenamed("count","mentions")
        monthly_skill_demand.write.format("delta").mode("overwrite").save(f"{self.gold_path}/monthly_skill_demand")
        monthly_skill_demand.write.mode("overwrite").parquet(f"{self.parquet_gold_path}/monthly_skill_demand")
        print("✅ Wrote monthly_skill_demand gold")
        # Skill co-occurrence within a posting (approx by splitting and exploding twice, then filtering pairs)
        from pyspark.sql.window import Window
        # Create an id per row to group skills by posting; if no id, use monotonically_increasing_id
        df_with_id = df.withColumn("row_id", monotonically_increasing_id())
        expl = df_with_id.select(col("row_id"), explode(tokens).alias("skill")) \
            .withColumn("skill", trim(lower(col("skill")))) \
            .filter(length(col("skill")) > 0)
        a = expl.select(col("row_id").alias("rid"), col("skill").alias("skill_a"))
        b = expl.select(col("row_id").alias("rid"), col("skill").alias("skill_b"))
        pairs = a.join(b, "rid").where(col("skill_a") < col("skill_b"))
        cooccurrence = pairs.groupBy("skill_a","skill_b").count().withColumnRenamed("count","cooccurs")
        cooccurrence.write.format("delta").mode("overwrite").save(f"{self.gold_path}/skill_cooccurrence")
        cooccurrence.write.mode("overwrite").parquet(f"{self.parquet_gold_path}/skill_cooccurrence")
        print("✅ Wrote skill_cooccurrence gold")
        return monthly_skill_demand, cooccurrence

    def create_gold_stackoverflow_aggregates(self):
        """Create StackOverflow gold aggregates (top languages, devtypes by country/year)"""
        print("🔄 Creating Gold aggregates from StackOverflow...")
        try:
            so = self.spark.read.format("delta").load(f"{self.silver_path}/stackoverflow_basic")
        except Exception:
            print("⚠️ stackoverflow_basic silver not found; skipping StackOverflow gold")
            return None, None
        # Languages: explode comma-separated
        langs = so.select(
            col("year"),
            col("country"),
            explode(split(lower(regexp_replace(col("languages"), r"[;/]", ",")), ",")).alias("language")
        ).withColumn("language", trim(col("language"))).filter(length(col("language")) > 0)
        top_langs = langs.groupBy("year","language").count().orderBy(col("count").desc())
        top_langs.write.format("delta").mode("overwrite").save(f"{self.gold_path}/so_top_languages")
        top_langs.write.mode("overwrite").parquet(f"{self.parquet_gold_path}/so_top_languages")
        print("✅ Wrote so_top_languages gold")
        # Devtypes by country/year
        devtypes = so.select(
            col("year"), col("country"),
            explode(split(lower(regexp_replace(col("devtype"), r"[;/]", ",")), ",")).alias("devtype")
        ).withColumn("devtype", trim(col("devtype"))).filter(length(col("devtype")) > 0)
        devtype_dist = devtypes.groupBy("year","country","devtype").count()
        devtype_dist.write.format("delta").mode("overwrite").save(f"{self.gold_path}/so_devtype_distribution")
        devtype_dist.write.mode("overwrite").parquet(f"{self.parquet_gold_path}/so_devtype_distribution")
        print("✅ Wrote so_devtype_distribution gold")
        return top_langs, devtype_dist

    def create_gold_company_aggregates(self):
        """Create company-level gold aggregates (hiring velocity, size distribution)"""
        print("🔄 Creating Gold aggregates for companies...")
        try:
            postings = self.spark.read.format("delta").load(f"{self.bronze_path}/job_postings")
        except Exception:
            print("⚠️ job_postings Delta table not found; skipping company gold")
            return None, None
        cols = postings.columns
        def pick(cands):
            for c in cands:
                if c in cols:
                    return c
            return None
        company_col = pick(["company","company_name","employer","org","organization"]) or "company"
        date_col = pick(["posted_date","date_posted","posting_date","created_at","created","post_date"]) or None
        size_col = pick(["company_size","employee_count","employees","size"]) or None
        # Hiring velocity: postings per company per month
        hiring_velocity = None
        if date_col is not None:
            hv = postings.withColumn("post_ts", to_timestamp(col(date_col))) \
                .withColumn("year_month", date_format(col("post_ts"), "yyyy-MM")) \
                .groupBy(company_col, "year_month").count().withColumnRenamed("count","postings")
            hiring_velocity = hv
            hv.write.format("delta").mode("overwrite").save(f"{self.gold_path}/company_hiring_velocity")
            hv.write.mode("overwrite").parquet(f"{self.parquet_gold_path}/company_hiring_velocity")
            print("✅ Wrote company_hiring_velocity gold")
        else:
            print("⚠️ No date column; skipping hiring velocity")
        # Size distribution snapshot
        size_distribution = None
        if size_col is not None:
            sd = postings.groupBy(company_col, size_col).count()
            size_distribution = sd
            sd.write.format("delta").mode("overwrite").save(f"{self.gold_path}/company_size_distribution")
            sd.write.mode("overwrite").parquet(f"{self.parquet_gold_path}/company_size_distribution")
            print("✅ Wrote company_size_distribution gold")
        else:
            print("⚠️ No size column; skipping size distribution")
        return hiring_velocity, size_distribution

    def create_gold_github_aggregates(self):
        """Create GitHub events gold aggregates (repo/type/hourly trends)"""
        print("🔄 Creating Gold aggregates for GitHub events...")
        try:
            gh = self.spark.read.format("delta").load(f"{self.bronze_path}/github_events")
        except Exception:
            print("⚠️ github_events Delta table not found; skipping GitHub gold")
            return None, None
        gh_ts = gh.withColumn("event_ts", to_timestamp(col("timestamp"))) \
                 .withColumn("hour", date_format(col("event_ts"), "yyyy-MM-dd HH:00"))
        by_repo_type = gh_ts.groupBy("repo_name","event_type").count().orderBy(col("count").desc())
        hourly_trends = gh_ts.groupBy("hour","event_type").count().orderBy("hour")
        by_repo_type.write.format("delta").mode("overwrite").save(f"{self.gold_path}/gh_repo_type_counts")
        by_repo_type.write.mode("overwrite").parquet(f"{self.parquet_gold_path}/gh_repo_type_counts")
        hourly_trends.write.format("delta").mode("overwrite").save(f"{self.gold_path}/gh_hourly_trends")
        hourly_trends.write.mode("overwrite").parquet(f"{self.parquet_gold_path}/gh_hourly_trends")
        print("✅ Wrote GitHub gold aggregates")
        return by_repo_type, hourly_trends

    def create_gold_github_language_monthly_and_linkage(self):
        """Create monthly GitHub language activity and link to job skills demand."""
        print("🔄 Creating GitHub language monthly and tech-job linkage...")
        try:
            gh = self.spark.read.format("delta").load(f"{self.bronze_path}/github_events")
        except Exception:
            print("⚠️ github_events not found; skipping language monthly")
            return None, None
        # Use language column if present; else infer from repo_name suffix (best-effort)
        cols = gh.columns
        lang_col = "language" if "language" in cols else None
        gh_lm = gh.withColumn("event_ts", to_timestamp(col("timestamp"))) \
                 .withColumn("year_month", date_format(col("event_ts"), "yyyy-MM"))
        if lang_col is not None:
            gh_lang = gh_lm.select(col("year_month"), lower(col(lang_col)).alias("language")).where(col("language").isNotNull())
        else:
            gh_lang = gh_lm.select(col("year_month"), lower(col("repo_name")).alias("language"))
        gh_lang_monthly = gh_lang.groupBy("year_month","language").count().withColumnRenamed("count","events")
        gh_lang_monthly.write.format("delta").mode("overwrite").save(f"{self.gold_path}/gh_language_monthly")
        gh_lang_monthly.write.mode("overwrite").parquet(f"{self.parquet_gold_path}/gh_language_monthly")
        print("✅ Wrote gh_language_monthly")
        # Linkage: join with monthly_skill_demand on language==skill and year_month
        try:
            monthly_skill = self.spark.read.parquet(f"{self.parquet_gold_path}/monthly_skill_demand")
        except Exception:
            monthly_skill = None
        if monthly_skill is not None:
            link = gh_lang_monthly.join(
                monthly_skill.withColumnRenamed("skill","language").withColumnRenamed("mentions","job_mentions"),
                on=["year_month","language"], how="inner"
            )
            link.write.format("delta").mode("overwrite").save(f"{self.gold_path}/tech_job_linkage")
            link.write.mode("overwrite").parquet(f"{self.parquet_gold_path}/tech_job_linkage")
            print("✅ Wrote tech_job_linkage")
            return gh_lang_monthly, link
        else:
            print("⚠️ monthly_skill_demand not found; skipping linkage")
            return gh_lang_monthly, None

    def create_gold_bls_health_index(self):
        """Create a simple yearly employment health index from BLS silver."""
        print("🔄 Creating BLS health index...")
        try:
            bls = self.spark.read.format("delta").load(f"{self.silver_path}/bls_employment")
        except Exception:
            print("⚠️ bls_employment silver not found; skipping BLS health")
            return None
        # Aggregate by year sum(value); normalize to index=100 at first year
        yearly = bls.groupBy("year").agg(_sum(col("value")).alias("employment"))
        # Collect min year and base value
        from pyspark.sql.window import Window
        base = yearly.orderBy("year").limit(1).collect()
        if not base:
            print("⚠️ No BLS yearly data")
            return None
        base_val = base[0]["employment"] if base[0]["employment"] else 1.0
        index = yearly.withColumn("health_index", (col("employment")/base_val)*100.0)
        index.write.format("delta").mode("overwrite").save(f"{self.gold_path}/bls_health_index")
        index.write.mode("overwrite").parquet(f"{self.parquet_gold_path}/bls_health_index")
        print("✅ Wrote bls_health_index")
        return index
    
    def run_full_pipeline(self):
        """Run complete ETL pipeline"""
        print("🚀 Starting Spark ETL Pipeline for 126GB+ Job Market Data")
        print("=" * 60)
        
        try:
            # Process all data sources
            self.process_github_data()
            self.process_stackoverflow_data()
            self.process_kaggle_jobs_data()
            self.process_bls_data()
            
            # Create data lake layers
            self.create_silver_layer()
            self.create_gold_layer()
            self.create_gold_postings_aggregates()
            self.create_gold_stackoverflow_aggregates()
            self.create_gold_company_aggregates()
            self.create_gold_github_aggregates()
            self.create_gold_skills_time_series()
            self.create_gold_github_language_monthly_and_linkage()
            self.create_gold_bls_health_index()
            
            print("\n🎉 Spark ETL Pipeline Complete!")
            print("📊 Data processed and stored in Delta Lake format")
            print("🔗 Bronze Layer: Raw data with metadata")
            print("🔗 Silver Layer: Cleaned, unified datasets")
            print("🔗 Gold Layer: ML-ready features")
            
        except Exception as e:
            print(f"❌ Pipeline failed: {str(e)}")
            raise
        finally:
            self.spark.stop()

def main():
    """Main function to run Spark ETL pipeline"""
    etl = JobMarketSparkETL()
    etl.run_full_pipeline()

if __name__ == "__main__":
    main()
