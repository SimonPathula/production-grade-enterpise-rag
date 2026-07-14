import logfire
from sqlalchemy import create_engine, Column, String, Text, DateTime, JSON, Float
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone
from google.cloud.sql.connector import Connector, IPTypes
from app.config import settings

Base = declarative_base()
class QueryLog(Base):
    __tablename__ = 'query_logs'
    id = Column(String, primary_key= True)
    query = Column(Text)
    response = Column(Text)
    latency = Column(Float)
    timestamp = Column(DateTime, default= lambda: datetime.now(timezone.utc))
    metadata_info = Column(JSON)

connector = Connector()
def getconn():
    conn = connector.connect(
        settings.DB_CONNECTION_NAME, 
        "pg8000",
        user = settings.DB_USER,
        password = settings.DB_PASS,
        db = settings.DB_NAME,
        ip_type = IPTypes.PUBLIC
    )
    return conn

try:
    if settings.DB_CONNECTION_NAME:
        engine = create_engine("postgresql+pg8000://", creator=getconn)
        SessionLocal = sessionmaker(autocommit = False, autoflush= False, bind=engine)
        Base.metadata.create_all(engine)
        logfire.info("Cloud SQL (POSTGRES) connected")
    else:
        logfire.warning("DB_CONNECTION_NAME not set. Audit logging disabled")
        SessionLocal = None
except Exception as e:
    logfire.error(f"Database init failed: {e}")
    raise e

def log_query_to_db(query_id: str, query: str, response: str, latency: float, metadata: dict):
    if not SessionLocal: return
    try:
        db = SessionLocal() 
        log_entry = QueryLog(
            id = query_id,
            query = query,
            response = response,
            latency = latency,
            metadata_info = metadata
        )
        db.add(log_entry)
        db.commit()
        db.close
    except Exception as e:
        logfire.info(f"DB logging failed: {e}")
        raise e
        
