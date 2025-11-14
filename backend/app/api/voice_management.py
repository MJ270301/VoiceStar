import os
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.core.database import get_db
from app.models.voice_models import VoiceProfile, VoiceCloningJob, VoiceUsage
from app.services.voice_cloning import VoiceCloningService
from app.services.emotion_engine import EmotionEngine

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic models
class VoiceProfileCreate(BaseModel):
    profile_name: str
    voice_type: str = "cloned"
    gender: Optional[str] = None
    accent: Optional[str] = None
    pitch: Optional[float] = 1.0
    speed: Optional[float] = 1.0
    volume: Optional[float] = 1.0
    emotion_range: Optional[List[str]] = ["neutral", "happy", "professional"]
    language: Optional[str] = "en-US"

class VoiceProfileUpdate(BaseModel):
    profile_name: Optional[str] = None
    gender: Optional[str] = None
    accent: Optional[str] = None
    pitch: Optional[float] = None
    speed: Optional[float] = None
    volume: Optional[float] = None
    emotion_range: Optional[List[str]] = None
    language: Optional[str] = None

class EmotionMapping(BaseModel):
    emotion: str
    pitch_adjustment: Optional[float] = 1.0
    speed_adjustment: Optional[float] = 1.0
    volume_adjustment: Optional[float] = 1.0
    timbre_adjustment: Optional[Dict[str, Any]] = None
    intensity: Optional[float] = 0.5

class VoiceCloneRequest(BaseModel):
    profile_name: str
    voice_config: Optional[Dict[str, Any]] = None

class VoiceSynthesisRequest(BaseModel):
    text: str
    voice_profile_id: int
    emotion: Optional[str] = "neutral"
    speed: Optional[float] = None
    pitch: Optional[float] = None
    volume: Optional[float] = None

@router.post("/api/voice/clone", response_model=Dict[str, Any])
async def create_voice_clone(
    file: UploadFile = File(..., description="Audio sample for voice cloning (5-30 seconds)"),
    clone_request: VoiceCloneRequest = None,
    db: Session = Depends(get_db),
    current_user_id: int = 1  # TODO: Get from auth
):
    """Create voice clone from audio sample"""
    try:
        # Validate file
        if not file.filename.endswith(('.wav', '.mp3', '.m4a', '.flac')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported audio format. Please use WAV, MP3, M4A, or FLAC."
            )

        # Save uploaded file
        upload_dir = "./uploads"
        os.makedirs(upload_dir, exist_ok=True)

        file_path = os.path.join(upload_dir, f"clone_sample_{current_user_id}_{file.filename}")

        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Initialize voice cloning service
        cloning_service = VoiceCloningService(db)

        # Create voice clone
        result = await cloning_service.create_voice_clone(
            user_id=current_user_id,
            profile_name=clone_request.profile_name,
            sample_file_path=file_path,
            voice_config=clone_request.voice_config
        )

        if result.get("success"):
            return {
                "success": True,
                "voice_profile_id": result["voice_profile_id"],
                "job_id": result["job_id"],
                "estimated_time": result["estimated_time"],
                "status": result["status"],
                "message": "Voice cloning process started successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Voice cloning failed")
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating voice clone: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.get("/api/voice/clone/status/{job_id}", response_model=Dict[str, Any])
async def get_cloning_status(
    job_id: str,
    db: Session = Depends(get_db),
    current_user_id: int = 1  # TODO: Get from auth
):
    """Get voice cloning job status"""
    try:
        cloning_service = VoiceCloningService(db)
        result = await cloning_service.get_cloning_status(job_id)

        return result

    except Exception as e:
        logger.error(f"Error getting cloning status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.get("/api/voice/profiles", response_model=List[Dict[str, Any]])
async def get_user_voice_profiles(
    db: Session = Depends(get_db),
    current_user_id: int = 1  # TODO: Get from auth
):
    """Get all voice profiles for user"""
    try:
        cloning_service = VoiceCloningService(db)
        voices = await cloning_service.get_user_voices(current_user_id)

        return voices

    except Exception as e:
        logger.error(f"Error getting voice profiles: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/api/voice/custom", response_model=Dict[str, Any])
async def create_custom_voice_profile(
    profile_data: VoiceProfileCreate,
    db: Session = Depends(get_db),
    current_user_id: int = 1  # TODO: Get from auth
):
    """Create custom voice profile"""
    try:
        # Create voice profile
        voice_profile = VoiceProfile(
            user_id=current_user_id,
            profile_name=profile_data.profile_name,
            voice_type="custom",
            gender=profile_data.gender,
            accent=profile_data.accent,
            pitch=profile_data.pitch,
            speed=profile_data.speed,
            volume=profile_data.volume,
            emotion_range=profile_data.emotion_range,
            language=profile_data.language
        )

        db.add(voice_profile)
        db.commit()
        db.refresh(voice_profile)

        return {
            "success": True,
            "voice_profile_id": voice_profile.id,
            "message": "Custom voice profile created successfully"
        }

    except Exception as e:
        logger.error(f"Error creating custom voice profile: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.put("/api/voice/profiles/{voice_id}", response_model=Dict[str, Any])
async def update_voice_profile(
    voice_id: int,
    profile_data: VoiceProfileUpdate,
    db: Session = Depends(get_db),
    current_user_id: int = 1  # TODO: Get from auth
):
    """Update voice profile"""
    try:
        # Verify ownership
        voice_profile = db.query(VoiceProfile).filter(
            VoiceProfile.id == voice_id
        ).first()

        if not voice_profile or voice_profile.user_id != current_user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Voice profile not found"
            )

        cloning_service = VoiceCloningService(db)
        updates = profile_data.dict(exclude_unset=True)
        result = await cloning_service.update_voice_profile(voice_id, current_user_id, updates)

        if result.get("success"):
            return {
                "success": True,
                "message": result["message"]
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "Update failed")
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating voice profile: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/api/voice/synthesize", response_model=Dict[str, Any])
async def synthesize_speech(
    synthesis_request: VoiceSynthesisRequest,
    db: Session = Depends(get_db),
    current_user_id: int = 1  # TODO: Get from auth
):
    """Synthesize speech with voice profile and emotion"""
    try:
        # Get voice profile
        voice_profile = db.query(VoiceProfile).filter(
            VoiceProfile.id == synthesis_request.voice_profile_id
        ).first()

        if not voice_profile or voice_profile.user_id != current_user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Voice profile not found"
            )

        # Initialize emotion engine
        emotion_engine = EmotionEngine(db)

        # Analyze emotion in text (if provided)
        emotion_analysis = await emotion_engine.analyze_emotion(synthesis_request.text)

        # Map emotion to voice parameters
        voice_mapping = await emotion_engine.map_emotion_to_voice_parameters(
            voice_profile.id, emotion_analysis
        )

        if not voice_mapping.get("success"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to map emotion to voice parameters"
            )

        # Apply emotion-based adjustments
        voice_params = voice_mapping["voice_parameters"]

        # Override with explicit parameters if provided
        if synthesis_request.speed:
            voice_params["speed"] = synthesis_request.speed
        if synthesis_request.pitch:
            voice_params["pitch"] = synthesis_request.pitch
        if synthesis_request.volume:
            voice_params["volume"] = synthesis_request.volume

        # Synthesize speech (implementation would depend on TTS provider)
        synthesis_result = await self._synthesize_with_voicestar(
            synthesis_request.text, voice_profile, voice_params
        )

        # Update usage count
        voice_profile.usage_count += 1
        db.commit()

        # Log usage
        usage_log = VoiceUsage(
            user_id=current_user_id,
            voice_profile_id=voice_profile.id,
            usage_type="synthesis",
            text_length=len(synthesis_request.text),
            processing_time=synthesis_result.get("processing_time", 0),
            cost=0.01,  # Example cost calculation
            api_endpoint="tts",
            success=synthesis_result.get("success", False)
        )

        db.add(usage_log)
        db.commit()

        if synthesis_result.get("success"):
            return {
                "success": True,
                "audio_url": synthesis_result.get("audio_url"),
                "duration": synthesis_result.get("duration"),
                "emotion_applied": voice_params["emotion"],
                "voice_parameters": voice_params,
                "processing_time": synthesis_result.get("processing_time"),
                "emotion_analysis": emotion_analysis
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=synthesis_result.get("error", "Speech synthesis failed")
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error synthesizing speech: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/api/voice/configure", response_model=Dict[str, Any])
async def configure_voice_for_agent(
    voice_profile_id: int,
    agent_id: int,
    emotion_mappings: List[EmotionMapping],
    db: Session = Depends(get_db),
    current_user_id: int = 1  # TODO: Get from auth
):
    """Configure voice for AI agent with emotion mappings"""
    try:
        # Verify voice profile ownership
        voice_profile = db.query(VoiceProfile).filter(
            VoiceProfile.id == voice_profile_id
        ).first()

        if not voice_profile or voice_profile.user_id != current_user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Voice profile not found"
            )

        # Initialize emotion engine
        emotion_engine = EmotionEngine(db)

        # Create emotion mappings
        created_mappings = []
        for mapping in emotion_mappings:
            result = await emotion_engine.create_emotion_mapping(
                voice_profile_id, mapping.emotion, mapping.dict()
            )

            if result.get("success"):
                created_mappings.append(mapping.emotion)

        return {
            "success": True,
            "voice_profile_id": voice_profile_id,
            "agent_id": agent_id,
            "created_mappings": created_mappings,
            "total_mappings": len(emotion_mappings),
            "message": f"Voice configured for agent with {len(created_mappings)} emotion mappings"
        }

    except Exception as e:
        logger.error(f"Error configuring voice for agent: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

async def _synthesize_with_voicestar(text: str, voice_profile: VoiceProfile, voice_params: Dict[str, Any]) -> Dict[str, Any]:
    """Synthesize speech using VoiceStar model"""
    try:
        # This would integrate with the existing VoiceStar inference
        # For now, return a mock result

        import time
        start_time = time.time()

        # Calculate duration (rough estimate)
        words_per_minute = 150
        word_count = len(text.split())
        estimated_duration = (word_count / words_per_minute) * 60  # seconds

        processing_time = int((time.time() - start_time) * 1000)  # milliseconds

        return {
            "success": True,
            "audio_url": f"/api/audio/generated/{voice_profile.id}_{hash(text)}.wav",
            "duration": estimated_duration,
            "processing_time": processing_time,
            "sample_rate": 16000,
            "format": "wav"
        }

    except Exception as e:
        logger.error(f"Error in VoiceStar synthesis: {str(e)}")
        return {"success": False, "error": str(e)}

@router.delete("/api/voice/profiles/{voice_id}", response_model=Dict[str, Any])
async def delete_voice_profile(
    voice_id: int,
    db: Session = Depends(get_db),
    current_user_id: int = 1  # TODO: Get from auth
):
    """Delete voice profile"""
    try:
        # Verify ownership
        voice_profile = db.query(VoiceProfile).filter(
            VoiceProfile.id == voice_id
        ).first()

        if not voice_profile or voice_profile.user_id != current_user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Voice profile not found"
            )

        # Delete related mappings and usage logs
        # Related objects should be deleted by cascade if configured properly

        # Delete voice profile
        db.delete(voice_profile)
        db.commit()

        return {
            "success": True,
            "message": "Voice profile deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting voice profile: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )