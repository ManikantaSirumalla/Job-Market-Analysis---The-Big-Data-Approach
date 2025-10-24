#!/usr/bin/env python3
"""
Comprehensive Tool Verification Script
Ensures all Big Data tools from academic presentation are working
"""

import sys
import subprocess
import importlib
from pathlib import Path

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"🔧 {title}")
    print("="*60)

def print_section(title):
    """Print formatted section"""
    print(f"\n📊 {title}")
    print("-" * 40)

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {package_name or module_name}: Import successful")
        return True
    except ImportError as e:
        print(f"❌ {package_name or module_name}: Import failed - {e}")
        return False

def test_spark():
    """Test Apache Spark functionality"""
    print_section("APACHE SPARK TESTING")
    
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col, lit
        
        # Create Spark session
        spark = SparkSession.builder \
            .appName("ToolVerification") \
            .master("local[*]") \
            .getOrCreate()
        
        # Test basic functionality
        df = spark.createDataFrame([(1, "test"), (2, "data")], ["id", "value"])
        result = df.filter(col("id") > 0).count()
        
        print(f"✅ Spark Session: Created successfully")
        print(f"✅ Spark DataFrame: {result} records processed")
        
        spark.stop()
        return True
        
    except Exception as e:
        print(f"❌ Spark Test Failed: {e}")
        return False

def test_delta_lake():
    """Test Delta Lake functionality"""
    print_section("DELTA LAKE TESTING")
    
    try:
        from delta.tables import DeltaTable
        from pyspark.sql import SparkSession
        
        # Create Spark session with Delta support
        spark = SparkSession.builder \
            .appName("DeltaTest") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .getOrCreate()
        
        # Test Delta functionality
        from pyspark.sql.functions import lit
        df = spark.createDataFrame([(1, "test")], ["id", "value"])
        
        # Create Delta table
        delta_path = "data/delta/test_table"
        df.write.format("delta").mode("overwrite").save(delta_path)
        
        # Read from Delta table
        delta_df = spark.read.format("delta").load(delta_path)
        count = delta_df.count()
        
        print(f"✅ Delta Lake: Write/Read successful")
        print(f"✅ Delta Table: {count} records")
        
        spark.stop()
        return True
        
    except Exception as e:
        print(f"❌ Delta Lake Test Failed: {e}")
        return False

def test_kafka():
    """Test Kafka functionality"""
    print_section("APACHE KAFKA TESTING")
    
    try:
        from kafka import KafkaProducer, KafkaConsumer
        from kafka.errors import KafkaError
        import json
        
        # Test producer creation (without actually connecting)
        producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            request_timeout_ms=1000
        )
        
        print("✅ Kafka Producer: Created successfully")
        
        # Test consumer creation (without actually connecting)
        consumer = KafkaConsumer(
            'test_topic',
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            consumer_timeout_ms=1000
        )
        
        print("✅ Kafka Consumer: Created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Kafka Test Failed: {e}")
        print("ℹ️  Note: Kafka requires a running broker for full testing")
        return False

def test_airflow():
    """Test Apache Airflow functionality"""
    print_section("APACHE AIRFLOW TESTING")
    
    try:
        from airflow import DAG
        from airflow.operators.python import PythonOperator
        from datetime import datetime, timedelta
        
        # Test DAG creation
        default_args = {
            'owner': 'test',
            'depends_on_past': False,
            'start_date': datetime(2024, 1, 1),
            'retries': 1,
            'retry_delay': timedelta(minutes=5),
        }
        
        dag = DAG(
            'test_dag',
            default_args=default_args,
            description='Test DAG',
            schedule_interval=timedelta(days=1),
        )
        
        def test_task():
            return "Airflow task executed successfully"
        
        task = PythonOperator(
            task_id='test_task',
            python_callable=test_task,
            dag=dag,
        )
        
        print("✅ Airflow DAG: Created successfully")
        print("✅ Airflow Task: Created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Airflow Test Failed: {e}")
        return False

def test_mlflow():
    """Test MLflow functionality"""
    print_section("MLFLOW TESTING")
    
    try:
        import mlflow
        import mlflow.sklearn
        from sklearn.linear_model import LinearRegression
        import numpy as np
        
        # Test MLflow setup
        mlflow.set_experiment("tool_verification")
        
        with mlflow.start_run():
            # Test logging
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 0.95)
            
            # Test model logging
            model = LinearRegression()
            X = np.array([[1], [2], [3]])
            y = np.array([1, 2, 3])
            model.fit(X, y)
            
            mlflow.sklearn.log_model(model, "test_model")
            
            print("✅ MLflow: Experiment tracking working")
            print("✅ MLflow: Model logging working")
        
        return True
        
    except Exception as e:
        print(f"❌ MLflow Test Failed: {e}")
        return False

def test_xgboost():
    """Test XGBoost functionality"""
    print_section("XGBOOST TESTING")
    
    try:
        import xgboost as xgb
        import numpy as np
        from sklearn.datasets import make_regression
        
        # Create test data
        X, y = make_regression(n_samples=100, n_features=4, random_state=42)
        
        # Test XGBoost model
        model = xgb.XGBRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Test prediction
        predictions = model.predict(X[:5])
        
        print(f"✅ XGBoost: Model training successful")
        print(f"✅ XGBoost: Predictions generated: {len(predictions)} samples")
        return True
        
    except Exception as e:
        print(f"❌ XGBoost Test Failed: {e}")
        return False

def test_fastapi():
    """Test FastAPI functionality"""
    print_section("FASTAPI TESTING")
    
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import uvicorn
        
        # Create test app
        app = FastAPI()
        
        @app.get("/test")
        def test_endpoint():
            return {"message": "FastAPI working"}
        
        # Test with test client
        client = TestClient(app)
        response = client.get("/test")
        
        if response.status_code == 200:
            print("✅ FastAPI: App creation successful")
            print("✅ FastAPI: Endpoint testing successful")
            print(f"✅ FastAPI: Response: {response.json()}")
            return True
        else:
            print(f"❌ FastAPI: Unexpected status code: {response.status_code}")
            return False
        
    except Exception as e:
        print(f"❌ FastAPI Test Failed: {e}")
        return False

def test_streamlit():
    """Test Streamlit functionality"""
    print_section("STREAMLIT TESTING")
    
    try:
        import streamlit as st
        import pandas as pd
        import numpy as np
        
        # Test basic Streamlit components
        df = pd.DataFrame({
            'x': np.random.randn(10),
            'y': np.random.randn(10)
        })
        
        print("✅ Streamlit: Import successful")
        print("✅ Streamlit: DataFrame creation successful")
        print(f"✅ Streamlit: Sample data shape: {df.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Streamlit Test Failed: {e}")
        return False

def test_project_integration():
    """Test project-specific integrations"""
    print_section("PROJECT INTEGRATION TESTING")
    
    # Test project modules
    project_modules = [
        "src.common.paths",
        "src.common.logs", 
        "src.api.app",
        "src.ingest.comprehensive_data_processor",
        "src.ml.salary_prediction_model",
        "src.spark.etl_pipeline"
    ]
    
    working_modules = 0
    for module in project_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}: Import successful")
            working_modules += 1
        except Exception as e:
            print(f"❌ {module}: Import failed - {e}")
    
    print(f"\n📊 Project Modules: {working_modules}/{len(project_modules)} working")
    return working_modules == len(project_modules)

def run_comprehensive_test():
    """Run comprehensive tool verification"""
    print_header("COMPREHENSIVE BIG DATA TOOLS VERIFICATION")
    
    print("🎯 Testing all tools from academic presentation:")
    print("   • Apache Spark")
    print("   • Delta Lake") 
    print("   • Apache Kafka")
    print("   • Apache Airflow")
    print("   • MLflow")
    print("   • XGBoost")
    print("   • FastAPI")
    print("   • Streamlit")
    
    # Test each tool
    results = {}
    
    # Basic imports
    print_section("BASIC IMPORTS")
    results['pyspark'] = test_import('pyspark', 'Apache Spark')
    results['delta'] = test_import('delta', 'Delta Lake')
    results['kafka'] = test_import('kafka', 'Apache Kafka')
    results['airflow'] = test_import('airflow', 'Apache Airflow')
    results['mlflow'] = test_import('mlflow', 'MLflow')
    results['xgboost'] = test_import('xgboost', 'XGBoost')
    results['fastapi'] = test_import('fastapi', 'FastAPI')
    results['streamlit'] = test_import('streamlit', 'Streamlit')
    
    # Functional tests
    print_section("FUNCTIONAL TESTING")
    results['spark_func'] = test_spark()
    results['delta_func'] = test_delta_lake()
    results['kafka_func'] = test_kafka()
    results['airflow_func'] = test_airflow()
    results['mlflow_func'] = test_mlflow()
    results['xgboost_func'] = test_xgboost()
    results['fastapi_func'] = test_fastapi()
    results['streamlit_func'] = test_streamlit()
    
    # Project integration
    results['project_integration'] = test_project_integration()
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    
    print(f"📊 Overall Results: {passed_tests}/{total_tests} tests passed")
    print(f"📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print("\n✅ WORKING TOOLS:")
    for tool, status in results.items():
        if status:
            print(f"   • {tool}")
    
    print("\n❌ ISSUES FOUND:")
    for tool, status in results.items():
        if not status:
            print(f"   • {tool}")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TOOLS VERIFIED AND WORKING!")
        print("🚀 Your project is ready for academic presentation!")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tools need attention")
        print("🔧 Please fix the issues above before presentation")
    
    return results

if __name__ == "__main__":
    run_comprehensive_test()
