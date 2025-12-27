from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
import os

Base = declarative_base()

class Game(Base):
    __tablename__ = 'games'
    __table_args__ = (UniqueConstraint('date', 'team_home', 'team_away', 'over_under_line', name='uq_game_unique'),)

    id = Column(Integer, primary_key=True)
    date = Column(DateTime, default=datetime.now)
    team_home = Column(String)
    team_away = Column(String)
    score_home = Column(Integer, nullable=True)
    score_away = Column(Integer, nullable=True)
    over_under_line = Column(Float) # The bookmaker's line
    
    # Results (computed after game)
    total_score = Column(Integer, nullable=True)
    result = Column(String, nullable=True) # 'Over', 'Under', 'Push'
    status = Column(String, default='scheduled') # 'scheduled', 'live', 'finished'

    predictions = relationship("Prediction", back_populates="game")

    def __repr__(self):
        return f"<Game(home='{self.team_home}', away='{self.team_away}', date='{self.date}')>"

class Prediction(Base):
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey('games.id'))
    predicted_total = Column(Float)
    predicted_class = Column(String) # 'Over', 'Under'
    confidence = Column(Float)
    explanation = Column(String, nullable=True)
    model_version = Column(String)
    created_at = Column(DateTime, default=datetime.now)

    game = relationship("Game", back_populates="predictions")

    def __repr__(self):
        return f"<Prediction(game_id={self.game_id}, pred='{self.predicted_class}')>"

class Metric(Base):
    __tablename__ = 'metrics'
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.now)
    mae = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)
    note = Column(String, nullable=True)

def init_db(db_path=None):
    import os
    use_connector = os.getenv('USE_CLOUD_SQL_CONNECTOR', 'false').lower() == 'true'
    engine = None
    if use_connector:
        from google.cloud.sql.connector import Connector, IPTypes
        import sqlalchemy
        connector = Connector()
        instance = os.getenv('CLOUDSQL_INSTANCE')
        db_user = os.getenv('DB_USER')
        db_pass = os.getenv('DB_PASS')
        db_name = os.getenv('DB_NAME')
        ip_pref = os.getenv('CLOUDSQL_IP_TYPE', 'PRIVATE').upper()
        ip_type = IPTypes.PRIVATE if ip_pref == 'PRIVATE' else IPTypes.PUBLIC
        def getconn():
            return connector.connect(
                instance,
                "pg8000",
                user=db_user,
                password=db_pass,
                db=db_name,
                ip_type=ip_type
            )
        engine = sqlalchemy.create_engine(
            "postgresql+pg8000://",
            creator=getconn,
            pool_pre_ping=True
        )
    else:
        url = db_path or os.getenv('DATABASE_URL') or 'sqlite:///ncaa_predictions.db'
        engine = create_engine(url, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    
    # Simple migration for explanation column
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE predictions ADD COLUMN explanation VARCHAR"))
            conn.commit()
    except Exception:
        pass # Column already exists
    
    Session = sessionmaker(bind=engine)
    return Session()
