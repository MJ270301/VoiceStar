from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, Float, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base

class VoiceProfile(Base):
    """Voice profile model for managing user voices"""
    __tablename__ = "voice_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    profile_name = Column(String(255), nullable=False)
    voice_type = Column(String(100), nullable=False)  # cloned, celebrity, custom, standard
    gender = Column(String(20))  # male, female, neutral
    age_range = Column(String(50))  # young, adult, mature, senior
    accent = Column(String(100))
    pitch = Column(Float, default=1.0)
    speed = Column(Float, default=1.0)
    volume = Column(Float, default=1.0)
    emotion_range = Column(Text)  # JSON array of supported emotions
    language = Column(String(10), default="en-US")
    sample_audio_url = Column(String(500))
    voice_model_path = Column(String(500))
    is_celebrity = Column(Boolean, default=False)
    celebrity_name = Column(String(255))
    license_info = Column(Text)  # JSON object for licensing details
    usage_count = Column(Integer, default=0)
    quality_score = Column(Float, default=0.0)
    clone_progress = Column(Integer, default=0)  # 0-100 for cloning progress
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="voice_profiles")
    cloning_jobs = relationship("VoiceCloningJob", back_populates="voice_profile")

class VoiceCloningJob(Base):
    """Voice cloning job tracking model"""
    __tablename__ = "voice_cloning_jobs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    voice_profile_id = Column(Integer, ForeignKey("voice_profiles.id"), nullable=True)
    job_id = Column(String(100), unique=True, nullable=False)
    status = Column(String(50), default="queued")  # queued, processing, completed, failed
    progress = Column(Integer, default=0)  # 0-100
    sample_file_path = Column(String(500))
    model_output_path = Column(String(500))
    error_message = Column(Text)
    processing_time = Column(Integer)  # in seconds
    quality_metrics = Column(Text)  # JSON object of quality scores
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User")
    voice_profile = relationship("VoiceProfile", back_populates="cloning_jobs")

class EmotionMapping(Base):
    """Emotion to voice parameter mappings"""
    __tablename__ = "emotion_mappings"

    id = Column(Integer, primary_key=True, index=True)
    voice_profile_id = Column(Integer, ForeignKey("voice_profiles.id"), nullable=False)
    emotion = Column(String(50), nullable=False)  # happy, sad, angry, excited, calm, etc.
    pitch_adjustment = Column(Float, default=1.0)
    speed_adjustment = Column(Float, default=1.0)
    volume_adjustment = Column(Float, default=1.0)
    timbre_adjustment = Column(String(500))  # JSON object for spectral adjustments
    intensity = Column(Float, default=1.0)  # Emotion intensity 0-1
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    voice_profile = relationship("VoiceProfile")

class CelebrityVoice(Base):
    """Celebrity voice model catalog"""
    __tablename__ = "celebrity_voices"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    celebrity_type = Column(String(100))  # actor, musician, character, influencer
    category = Column(String(100))  # professional, entertainment, character
    gender = Column(String(20))
    language = Column(String(10), default="en-US")
    description = Column(Text)
    sample_audio_url = Column(String(500))
    model_path = Column(String(500))
    is_premium = Column(Boolean, default=False)
    licensing_required = Column(Boolean, default=True)
    license_cost = Column(Float)  # per usage cost
    usage_count = Column(Integer, default=0)
    rating = Column(Float, default=0.0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class VoiceUsage(Base):
    """Voice usage tracking for billing and analytics"""
    __tablename__ = "voice_usage"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    voice_profile_id = Column(Integer, ForeignKey("voice_profiles.id"), nullable=True)
    celebrity_voice_id = Column(Integer, ForeignKey("celebrity_voices.id"), nullable=True)
    usage_type = Column(String(50), nullable=False)  # synthesis, cloning, emotion_processing
    text_length = Column(Integer)  # characters processed
    audio_duration = Column(Float)  # seconds generated
    processing_time = Column(Integer)  # milliseconds
    cost = Column(Float, default=0.0)
    api_endpoint = Column(String(100))
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User")
    voice_profile = relationship("VoiceProfile")
    celebrity_voice = relationship("CelebrityVoice")

class VoiceQualityMetrics(Base):
    """Voice quality assessment metrics"""
    __tablename__ = "voice_quality_metrics"

    id = Column(Integer, primary_key=True, index=True)
    voice_profile_id = Column(Integer, ForeignKey("voice_profiles.id"), nullable=False)
    naturalness_score = Column(Float)  # MOS (Mean Opinion Score) for naturalness
    clarity_score = Column(Float)
    similarity_score = Column(Float)  # For cloned voices
    emotion_accuracy = Column(Float)  # Emotion expression accuracy
    consistency_score = Column(Float)  # Voice consistency across samples
    overall_score = Column(Float)
    assessment_date = Column(DateTime(timezone=True), server_default=func.now())
    assessed_by = Column(String(255))  # User or system
    feedback_notes = Column(Text)

    # Relationships
    voice_profile = relationship("VoiceProfile")

# Add User model (simplified for Voice AI Suite)
class User(Base):
    """User model for Voice AI Suite"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    voice_profiles = relationship("VoiceProfile", back_populates="user")
    cloning_jobs = relationship("VoiceCloningJob", back_populates="user")
    usage_logs = relationship("VoiceUsage", back_populates="user")