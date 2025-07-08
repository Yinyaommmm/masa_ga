from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# 创建基类
Base = declarative_base()
class CropGif(Base):
    __tablename__ = 'crop_gif'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    gif_id = Column(Integer, nullable=False)
    video_id = Column(Integer, nullable=False)
    storage_path = Column(String(255), nullable=False)
    instance_id = Column(Integer, nullable=False)
    frame_idx = Column(Integer, nullable=False)
    label = Column(Integer, nullable=False)
    score = Column(Float, nullable=False)
    bbox_xmin = Column(Float, nullable=False)
    bbox_ymin = Column(Float, nullable=False)
    bbox_xmax = Column(Float, nullable=False)
    bbox_ymax = Column(Float, nullable=False)
    create_time = Column(DateTime, default=datetime.now)

class DBConnector:
    def __init__(self, db_url):
        """初始化数据库连接"""
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.session = None
    
    def create_tables(self):
        """创建所有表结构"""
        Base.metadata.create_all(self.engine)
    
    def get_session(self):
        """获取数据库会话"""
        if not self.session or self.session.is_active is False:
            self.session = self.Session()
        return self.session
    
    def add_crop(self, data):
        """添加裁剪图片记录"""
        session = self.get_session()
        try:
            crop = CropGif(**data)
            session.add(crop)
            session.commit()
            return crop.id
        except Exception as e:
            session.rollback()
            raise e
    
    def close(self):
        """关闭数据库会话"""
        if self.session:
            self.session.close()
            self.session = None