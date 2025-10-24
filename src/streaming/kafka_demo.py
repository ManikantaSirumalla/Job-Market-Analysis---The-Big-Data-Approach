#!/usr/bin/env python3
"""
Kafka Demo for Job Market Analysis
Demonstrates streaming data ingestion capabilities
"""

import sys
import json
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError

class JobMarketKafkaDemo:
    """Kafka streaming demo for job market data"""
    
    def __init__(self, bootstrap_servers=['localhost:9092']):
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        self.consumer = None
    
    def create_producer(self):
        """Create Kafka producer"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                request_timeout_ms=1000,
                retries=1
            )
            print("✅ Kafka Producer created successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to create Kafka Producer: {e}")
            return False
    
    def create_consumer(self, topic='job_market_events'):
        """Create Kafka consumer"""
        try:
            self.consumer = KafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                consumer_timeout_ms=1000,
                auto_offset_reset='earliest'
            )
            print("✅ Kafka Consumer created successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to create Kafka Consumer: {e}")
            return False
    
    def send_job_event(self, event_data):
        """Send job market event to Kafka"""
        if not self.producer:
            print("❌ Producer not initialized")
            return False
        
        try:
            topic = 'job_market_events'
            future = self.producer.send(topic, event_data)
            record_metadata = future.get(timeout=1)
            print(f"✅ Event sent to {record_metadata.topic} partition {record_metadata.partition}")
            return True
        except Exception as e:
            print(f"❌ Failed to send event: {e}")
            return False
    
    def consume_events(self, max_events=5):
        """Consume job market events from Kafka"""
        if not self.consumer:
            print("❌ Consumer not initialized")
            return []
        
        events = []
        try:
            for message in self.consumer:
                events.append(message.value)
                print(f"📨 Received event: {message.value}")
                if len(events) >= max_events:
                    break
        except Exception as e:
            print(f"❌ Error consuming events: {e}")
        
        return events
    
    def demo_streaming_pipeline(self):
        """Demonstrate streaming data pipeline"""
        print("🚀 Starting Kafka Streaming Demo")
        print("=" * 40)
        
        # Create producer and consumer
        if not self.create_producer():
            print("⚠️  Producer creation failed - Kafka broker not running")
            print("ℹ️  To start Kafka: brew install kafka && brew services start kafka")
            return False
        
        if not self.create_consumer():
            print("⚠️  Consumer creation failed - Kafka broker not running")
            return False
        
        # Send sample events
        sample_events = [
            {
                "event_type": "job_posting",
                "job_id": "job_001",
                "title": "Data Scientist",
                "company": "Tech Corp",
                "salary": 120000,
                "location": "San Francisco",
                "timestamp": "2024-10-13T02:15:00Z"
            },
            {
                "event_type": "salary_update",
                "job_id": "job_002", 
                "title": "Software Engineer",
                "company": "Startup Inc",
                "salary": 95000,
                "location": "New York",
                "timestamp": "2024-10-13T02:16:00Z"
            },
            {
                "event_type": "job_posting",
                "job_id": "job_003",
                "title": "ML Engineer",
                "company": "AI Labs",
                "salary": 140000,
                "location": "Seattle",
                "timestamp": "2024-10-13T02:17:00Z"
            }
        ]
        
        print("\n📤 Sending sample events...")
        for event in sample_events:
            self.send_job_event(event)
            time.sleep(0.5)  # Small delay between events
        
        print("\n📥 Consuming events...")
        received_events = self.consume_events(max_events=3)
        
        print(f"\n✅ Demo complete! Processed {len(received_events)} events")
        return True
    
    def close(self):
        """Close producer and consumer"""
        if self.producer:
            self.producer.close()
        if self.consumer:
            self.consumer.close()

def main():
    """Main function to run Kafka demo"""
    demo = JobMarketKafkaDemo()
    
    try:
        success = demo.demo_streaming_pipeline()
        if success:
            print("\n🎉 Kafka streaming demo successful!")
        else:
            print("\n⚠️  Kafka demo completed with warnings")
    finally:
        demo.close()

if __name__ == "__main__":
    main()
