import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.core.database import get_db
from app.services.emotion_engine import EmotionEngine

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic models
class EmotionAnalysisRequest(BaseModel):
    text: str
    audio_features: Dict[str, Any] = None

class EmotionMappingRequest(BaseModel):
    voice_profile_id: int
    emotion: str
    pitch_adjustment: float = 1.0
    speed_adjustment: float = 1.0
    volume_adjustment: float = 1.0
    timbre_adjustment: Dict[str, Any] = None
    intensity: float = 0.5

class VoiceEmotionTestRequest(BaseModel):
    voice_profile_id: int
    test_text: str
    target_emotions: List[str] = None

@router.post("/api/emotion/analyze", response_model=Dict[str, Any])
async def analyze_emotion(
    request: EmotionAnalysisRequest,
    db: Session = Depends(get_db),
    current_user_id: int = 1  # TODO: Get from auth
):
    """Analyze emotion from text and/or audio"""
    try:
        emotion_engine = EmotionEngine(db)
        result = await emotion_engine.analyze_emotion(
            request.text, request.audio_features
        )

        return {
            "success": True,
            "emotion_analysis": result,
            "analysis_metadata": {
                "text_length": len(request.text),
                "has_audio": bool(request.audio_features),
                "analysis_method": result.get("analysis_method", "hybrid")
            }
        }

    except Exception as e:
        logger.error(f"Error analyzing emotion: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Emotion analysis failed"
        )

@router.post("/api/emotion/mapping/create", response_model=Dict[str, Any])
async def create_emotion_mapping(
    request: EmotionMappingRequest,
    db: Session = Depends(get_db),
    current_user_id: int = 1  # TODO: Get from auth
):
    """Create emotion to voice parameter mapping"""
    try:
        emotion_engine = EmotionEngine(db)
        result = await emotion_engine.create_emotion_mapping(
            request.voice_profile_id, request.emotion, request.dict()
        )

        if result.get("success"):
            return {
                "success": True,
                "message": result["message"],
                "voice_profile_id": request.voice_profile_id,
                "emotion": request.emotion,
                "mapping_applied": {
                    "pitch_adjustment": request.pitch_adjustment,
                    "speed_adjustment": request.speed_adjustment,
                    "volume_adjustment": request.volume_adjustment,
                    "timbre_adjustment": request.timbre_adjustment,
                    "intensity": request.intensity
                }
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "Failed to create emotion mapping")
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating emotion mapping: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/api/emotion/voice-parameters", response_model=Dict[str, Any])
async def map_emotion_to_voice(
    voice_profile_id: int,
    emotion_request: EmotionAnalysisRequest,
    db: Session = Depends(get_db),
    current_user_id: int = 1  # TODO: Get from auth
):
    """Map emotion to voice synthesis parameters"""
    try:
        emotion_engine = EmotionEngine(db)
        emotion_result = await emotion_engine.analyze_emotion(
            emotion_request.text, emotion_request.audio_features
        )

        if "error" in emotion_result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Emotion analysis failed"
            )

        voice_mapping = await emotion_engine.map_emotion_to_voice_parameters(
            voice_profile_id, emotion_result
        )

        return {
            "success": True,
            "emotion_analysis": emotion_result,
            "voice_parameters": voice_mapping.get("voice_parameters"),
            "applied_mappings": voice_mapping.get("applied_mappings", {}),
            "message": "Emotion mapped to voice parameters successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error mapping emotion to voice: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/api/emotion/test", response_model=Dict[str, Any])
async def test_emotion_expression(
    request: VoiceEmotionTestRequest,
    db: Session = Depends(get_db),
    current_user_id: int = 1  # TODO: Get from auth
):
    """Test emotion expression with voice profile"""
    try:
        emotion_engine = EmotionEngine(db)

        # Analyze emotion in test text
        emotion_analysis = await emotion_engine.analyze_emotion(
            request.test_text, None
        )

        # Get voice parameters for each emotion
        results = []

        # Test primary emotion and up to 3 additional emotions
        test_emotions = [emotion_analysis.get("primary_emotion", "neutral")]
        if request.target_emotions:
            test_emotions.extend(request.target_emotions[:3])

        for emotion in test_emotions:
            if emotion:
                emotion_request = EmotionAnalysisRequest(
                    text=request.test_text
                )
                emotion_request.text = f"Expressing {emotion} emotion"

                voice_mapping = await emotion_engine.map_emotion_to_voice_parameters(
                    request.voice_profile_id, {"primary_emotion": emotion}
                )

                if voice_mapping.get("success"):
                    # Synthesize test audio (mock implementation)
                    synthesis_result = await _synthesize_emotion_test(
                        request.test_text, voice_mapping.get("voice_parameters", {})
                    )

                    results.append({
                        "emotion": emotion,
                        "voice_parameters": voice_mapping.get("voice_parameters"),
                        "audio_url": synthesis_result.get("audio_url"),
                        "success": synthesis_result.get("success", False)
                    })

        return {
            "success": True,
            "test_results": results,
            "test_text": request.test_text,
            "voice_profile_id": request.voice_profile_id
        }

    except Exception as e:
        logger.error(f"Error testing emotion expression: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.get("/api/emotion/mappings/{voice_profile_id}", response_model=Dict[str, Any])
async def get_emotion_mappings(
    voice_profile_id: int,
    db: Session = Depends(get_db),
    current_user_id: int = 1  # TODO: Get from auth
):
    """Get all emotion mappings for voice profile"""
    try:
        from app.models.voice_models import EmotionMapping

        mappings = db.query(EmotionMapping).filter(
            EmotionMapping.voice_profile_id == voice_profile_id
        ).all()

        mapping_list = []
        for mapping in mappings:
            mapping_list.append({
                "id": mapping.id,
                "emotion": mapping.emotion,
                "pitch_adjustment": mapping.pitch_adjustment,
                "speed_adjustment": mapping.speed_adjustment,
                "volume_adjustment": mapping.volume_adjustment,
                "timbre_adjustment": mapping.timbre_adjustment,
                "intensity": mapping.intensity,
                "created_at": mapping.created_at.isoformat()
            })

        return {
            "success": True,
            "voice_profile_id": voice_profile_id,
            "emotion_mappings": mapping_list,
            "total_mappings": len(mapping_list)
        }

    except Exception as e:
        logger.error(f"Error getting emotion mappings: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.get("/api/emotion/supported", response_model=Dict[str, Any])
async def get_supported_emotions():
    """Get list of supported emotions and their descriptions"""
    try:
        supported_emotions = {
            "happy": {
                "description": "Positive emotion expressing joy and contentment",
                "common_triggers": ["success", "achievement", "good news"],
                "voice_characteristics": {
                    "pitch_range": [1.0, 1.3],
                    "speed_range": [1.0, 1.1],
                    "volume_range": [1.0, 1.2]
                }
            },
            "sad": {
                "description": "Negative emotion expressing sorrow and disappointment",
                "common_triggers": ["loss", "failure", "bad news"],
                "voice_characteristics": {
                    "pitch_range": [0.8, 1.0],
                    "speed_range": [0.9, 1.0],
                    "volume_range": [0.8, 1.0]
                }
            },
            "angry": {
                "description": "Intense negative emotion expressing frustration and anger",
                "common_triggers": ["injustice", "obstacles", "conflict"],
                "voice_characteristics": {
                    "pitch_range": [1.1, 1.4],
                    "speed_range": [1.1, 1.3],
                    "volume_range": [1.1, 1.4]
                }
            },
            "excited": {
                "description": "High-energy positive emotion expressing enthusiasm",
                "common_triggers": ["opportunity", "anticipation", "new experiences"],
                "voice_characteristics": {
                    "pitch_range": [1.1, 1.3],
                    "speed_range": [1.1, 1.2],
                    "volume_range": [1.1, 1.3]
                }
            },
            "calm": {
                "description": "Neutral emotion expressing peacefulness and control",
                "common_triggers": ["routine", "meditation", "reflection"],
                "voice_characteristics": {
                    "pitch_range": [0.95, 1.05],
                    "speed_range": [0.95, 1.05],
                    "volume_range": [0.95, 1.05]
                }
            },
            "confused": {
                "description": "Uncertainty emotion expressing doubt and questioning",
                "common_triggers": ["complexity", "ambiguity", "new information"],
                "voice_characteristics": {
                    "pitch_range": [0.9, 1.1],
                    "speed_range": [0.9, 1.0],
                    "volume_range": [0.9, 1.0]
                }
            },
            "professional": {
                "description": "Controlled emotion suitable for business communication",
                "common_triggers": ["customer service", "presentations", "formal settings"],
                "voice_characteristics": {
                    "pitch_range": [0.98, 1.02],
                    "speed_range": [0.98, 1.02],
                    "volume_range": [0.98, 1.02]
                }
            }
        }

        return {
            "success": True,
            "supported_emotions": supported_emotions,
            "total_emotions": len(supported_emotions)
        }

    except Exception as e:
        logger.error(f"Error getting supported emotions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

async def _synthesize_emotion_test(text: str, voice_params: Dict[str, Any]) -> Dict[str, Any]:
    """Synthesize test audio for emotion testing"""
    try:
        # Mock synthesis for testing
        import time
        import hashlib

        start_time = time.time()

        # Simulate processing time
        await asyncio.sleep(0.5)  # 500ms processing time

        processing_time = int((time.time() - start_time) * 1000)

        # Generate mock audio URL
        audio_hash = hashlib.md5(f"{text}_{voice_params}".encode()).hexdigest()
        audio_url = f"/test_audio/{audio_hash}.wav"

        return {
            "success": True,
            "audio_url": audio_url,
            "duration": len(text.split()) * 0.4,  # Rough estimate
            "processing_time": processing_time,
            "voice_parameters": voice_params
        }

    except Exception as e:
        logger.error(f"Error synthesizing emotion test: {str(e)}")
        return {"success": False, "error": str(e)}