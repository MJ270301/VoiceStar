import os
import json
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from pydantic import BaseModel

from app.core.database import get_db
from app.models.voice_models import CelebrityVoice, VoiceUsage

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic models
class CelebrityVoiceResponse(BaseModel):
    id: int
    name: str
    celebrity_type: str
    category: str
    gender: str
    language: str
    description: str
    sample_audio_url: str
    is_premium: bool
    licensing_required: bool
    license_cost: Optional[float] = None
    rating: float
    usage_count: int

class CelebrityVoiceSearch(BaseModel):
    query: Optional[str] = None
    category: Optional[str] = None
    gender: Optional[str] = None
    language: Optional[str] = None
    celebrity_type: Optional[str] = None
    is_premium: Optional[bool] = None
    min_rating: Optional[float] = None
    sort_by: Optional[str] = "rating"  # rating, usage, name, created_at
    limit: Optional[int] = 20

class VoiceLicenseRequest(BaseModel):
    celebrity_voice_id: int
    usage_type: str = "commercial"  # commercial, personal, development
    duration_months: Optional[int] = 1

@router.get("/api/marketplace/voices", response_model=List[CelebrityVoiceResponse])
async def search_celebrity_voices(
    search_params: CelebrityVoiceSearch = Depends(),
    db: Session = Depends(get_db),
    current_user_id: int = 1  # TODO: Get from auth
):
    """Search celebrity voice catalog"""
    try:
        # Build query
        query = db.query(CelebrityVoice).filter(CelebrityVoice.is_active == True)

        # Apply filters
        if search_params.query:
            search_term = f"%{search_params.query.lower()}%"
            query = query.filter(
                or_(
                    CelebrityVoice.name.ilike(search_term),
                    CelebrityVoice.description.ilike(search_term),
                    CelebrityVoice.celebrity_type.ilike(search_term)
                )
            )

        if search_params.category:
            query = query.filter(CelebrityVoice.category == search_params.category)

        if search_params.gender:
            query = query.filter(CelebrityVoice.gender == search_params.gender)

        if search_params.language:
            query = query.filter(CelebrityVoice.language == search_params.language)

        if search_params.celebrity_type:
            query = query.filter(CelebrityVoice.celebrity_type == search_params.celebrity_type)

        if search_params.is_premium is not None:
            query = query.filter(CelebrityVoice.is_premium == search_params.is_premium)

        if search_params.min_rating:
            query = query.filter(CelebrityVoice.rating >= search_params.min_rating)

        # Apply sorting
        sort_mapping = {
            "rating": CelebrityVoice.rating.desc(),
            "usage": CelebrityVoice.usage_count.desc(),
            "name": CelebrityVoice.name.asc(),
            "created_at": CelebrityVoice.created_at.desc()
        }
        order_by = sort_mapping.get(search_params.sort_by, CelebrityVoice.rating.desc())
        query = query.order_by(order_by)

        # Apply limit
        if search_params.limit:
            query = query.limit(search_params.limit)

        # Execute query
        voices = query.all()

        # Convert to response format
        voice_list = []
        for voice in voices:
            voice_list.append(CelebrityVoiceResponse(
                id=voice.id,
                name=voice.name,
                celebrity_type=voice.celebrity_type,
                category=voice.category,
                gender=voice.gender,
                language=voice.language,
                description=voice.description,
                sample_audio_url=voice.sample_audio_url,
                is_premium=voice.is_premium,
                licensing_required=voice.licensing_required,
                license_cost=voice.license_cost,
                rating=voice.rating,
                usage_count=voice.usage_count
            ))

        return voice_list

    except Exception as e:
        logger.error(f"Error searching celebrity voices: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.get("/api/marketplace/voices/{voice_id}", response_model=Dict[str, Any])
async def get_celebrity_voice_details(
    voice_id: int,
    db: Session = Depends(get_db),
    current_user_id: int = 1  # TODO: Get from auth
):
    """Get detailed information about a celebrity voice"""
    try:
        voice = db.query(CelebrityVoice).filter(
            and_(CelebrityVoice.id == voice_id, CelebrityVoice.is_active == True)
        ).first()

        if not voice:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Celebrity voice not found"
            )

        # Get user's license status (if any)
        user_license = db.query(VoiceUsage).filter(
            and_(
                VoiceUsage.user_id == current_user_id,
                VoiceUsage.celebrity_voice_id == voice_id
            )
        ).order_by(VoiceUsage.created_at.desc()).first()

        return {
            "success": True,
            "voice": {
                "id": voice.id,
                "name": voice.name,
                "celebrity_type": voice.celebrity_type,
                "category": voice.category,
                "gender": voice.gender,
                "language": voice.language,
                "description": voice.description,
                "sample_audio_url": voice.sample_audio_url,
                "is_premium": voice.is_premium,
                "licensing_required": voice.licensing_required,
                "license_cost": voice.license_cost,
                "rating": voice.rating,
                "usage_count": voice.usage_count,
                "created_at": voice.created_at.isoformat()
            },
            "user_license": {
                "has_license": bool(user_license),
                "license_type": user_license.usage_type if user_license else None,
                "expires_at": user_license.created_at.isoformat() if user_license else None,
                "usage_count": db.query(VoiceUsage).filter(
                    and_(
                        VoiceUsage.user_id == current_user_id,
                        VoiceUsage.celebrity_voice_id == voice_id
                    )
                ).count()
            },
            "licensing_info": {
                "commercial_use_required": voice.licensing_required,
                "cost_per_use": voice.license_cost,
                "bulk_licensing_available": True,
                "educational_discount": 0.5  # 50% discount for educational use
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting celebrity voice details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/api/marketplace/license", response_model=Dict[str, Any])
async def license_celebrity_voice(
    license_request: VoiceLicenseRequest,
    db: Session = Depends(get_db),
    current_user_id: int = 1  # TODO: Get from auth
):
    """License a celebrity voice for use"""
    try:
        # Get celebrity voice
        voice = db.query(CelebrityVoice).filter(
            and_(CelebrityVoice.id == license_request.celebrity_voice_id, CelebrityVoice.is_active == True)
        ).first()

        if not voice:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Celebrity voice not found"
            )

        if not voice.licensing_required:
            return {
                "success": True,
                "message": "This voice does not require licensing",
                "voice_id": voice.id,
                "license_type": "free"
            }

        # Calculate licensing cost
        base_cost = voice.license_cost or 0.0
        duration_months = license_request.duration_months or 1

        # Apply discounts
        total_cost = base_cost * duration_months
        if license_request.usage_type == "educational":
            total_cost *= 0.5  # 50% educational discount
        elif license_request.usage_type == "bulk":
            total_cost *= 0.8  # 20% bulk discount

        # Create usage record
        usage_record = VoiceUsage(
            user_id=current_user_id,
            celebrity_voice_id=voice.id,
            usage_type=f"license_{license_request.usage_type}",
            text_length=0,  # N/A for licensing
            audio_duration=0,  # N/A for licensing
            processing_time=0,
            cost=total_cost,
            api_endpoint="marketplace_license"
        )

        db.add(usage_record)
        db.commit()

        return {
            "success": True,
            "voice_id": voice.id,
            "license_type": license_request.usage_type,
            "duration_months": duration_months,
            "total_cost": total_cost,
            "savings_applied": base_cost * duration_months - total_cost,
            "message": f"Successfully licensed {voice.name} for {duration_months} months"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error licensing celebrity voice: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.get("/api/marketplace/categories", response_model=Dict[str, Any])
async def get_voice_categories(
    db: Session = Depends(get_db),
    current_user_id: int = 1  # TODO: Get from auth
):
    """Get available celebrity voice categories"""
    try:
        # Get unique categories from database
        categories = db.query(CelebrityVoice.category).distinct().all()

        category_list = []
        for category in categories:
            category_list.append(category[0])

        # Get celebrity types
        celebrity_types = db.query(CelebrityVoice.celebrity_type).distinct().all()

        type_list = []
        for celeb_type in celebrity_types:
            type_list.append(celeb_type[0])

        return {
            "success": True,
            "categories": category_list,
            "celebrity_types": type_list,
            "genders": ["male", "female", "neutral"],
            "languages": ["en-US", "en-GB", "es-ES", "fr-FR", "de-DE", "it-IT", "pt-BR"],
            "usage_statistics": await _get_marketplace_statistics(db)
        }

    except Exception as e:
        logger.error(f"Error getting voice categories: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.get("/api/marketplace/popular", response_model=List[CelebrityVoiceResponse])
async def get_popular_voices(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
    current_user_id: int = 1  # TODO: Get from auth
):
    """Get most popular celebrity voices"""
    try:
        # Get voices by usage count and rating
        popular_voices = db.query(CelebrityVoice).filter(
            CelebrityVoice.is_active == True
        ).order_by(
            CelebrityVoice.usage_count.desc(),
            CelebrityVoice.rating.desc()
        ).limit(limit).all()

        voice_list = []
        for voice in popular_voices:
            voice_list.append(CelebrityVoiceResponse(
                id=voice.id,
                name=voice.name,
                celebrity_type=voice.celebrity_type,
                category=voice.category,
                gender=voice.gender,
                language=voice.language,
                description=voice.description,
                sample_audio_url=voice.sample_audio_url,
                is_premium=voice.is_premium,
                licensing_required=voice.licensing_required,
                license_cost=voice.license_cost,
                rating=voice.rating,
                usage_count=voice.usage_count
            ))

        return voice_list

    except Exception as e:
        logger.error(f"Error getting popular voices: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.get("/api/marketplace/premium", response_model=List[CelebrityVoiceResponse])
async def get_premium_voices(
    category: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user_id: int = 1  # TODO: Get from auth
):
    """Get premium celebrity voices"""
    try:
        # Get premium voices
        query = db.query(CelebrityVoice).filter(
            and_(CelebrityVoice.is_active == True, CelebrityVoice.is_premium == True)
        )

        if category:
            query = query.filter(CelebrityVoice.category == category)

        premium_voices = query.order_by(
            CelebrityVoice.rating.desc()
        ).all()

        voice_list = []
        for voice in premium_voices:
            voice_list.append(CelebrityVoiceResponse(
                id=voice.id,
                name=voice.name,
                celebrity_type=voice.celebrity_type,
                category=voice.category,
                gender=voice.gender,
                language=voice.language,
                description=voice.description,
                sample_audio_url=voice.sample_audio_url,
                is_premium=voice.is_premium,
                licensing_required=voice.licensing_required,
                license_cost=voice.license_cost,
                rating=voice.rating,
                usage_count=voice.usage_count
            ))

        return voice_list

    except Exception as e:
        logger.error(f"Error getting premium voices: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.get("/api/marketplace/my-licenses", response_model=List[Dict[str, Any]])
async def get_user_licenses(
    db: Session = Depends(get_db),
    current_user_id: int = 1  # TODO: Get from auth
):
    """Get user's licensed celebrity voices"""
    try:
        # Get user's licensed voices
        user_licenses = db.query(VoiceUsage).filter(
            and_(
                VoiceUsage.user_id == current_user_id,
                VoiceUsage.celebrity_voice_id.isnot(None),
                VoiceUsage.usage_type.like("license_%")
            )
        ).order_by(VoiceUsage.created_at.desc()).all()

        license_list = []
        for license_record in user_licenses:
            voice = license_record.celebrity_voice
            if voice:
                license_list.append({
                    "voice_id": voice.id,
                    "voice_name": voice.name,
                    "celebrity_type": voice.celebrity_type,
                    "license_type": license_record.usage_type,
                    "cost": license_record.cost,
                    "licensed_at": license_record.created_at.isoformat(),
                    "usage_count": db.query(VoiceUsage).filter(
                        and_(
                            VoiceUsage.user_id == current_user_id,
                            VoiceUsage.celebrity_voice_id == voice.id
                        )
                    ).count()
                })

        return license_list

    except Exception as e:
        logger.error(f"Error getting user licenses: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/api/marketplace/{voice_id}/sample", response_model=Dict[str, Any])
async def generate_voice_sample(
    voice_id: int,
    text: str = "Hello, this is a sample of my voice.",
    db: Session = Depends(get_db),
    current_user_id: int = 1  # TODO: Get from auth
):
    """Generate sample audio with celebrity voice"""
    try:
        # Get voice details
        voice = db.query(CelebrityVoice).filter(
            and_(CelebrityVoice.id == voice_id, CelebrityVoice.is_active == True)
        ).first()

        if not voice:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Celebrity voice not found"
            )

        # Check if user has license
        has_license = db.query(VoiceUsage).filter(
            and_(
                VoiceUsage.user_id == current_user_id,
                VoiceUsage.celebrity_voice_id == voice_id,
                VoiceUsage.usage_type.like("license_%")
            )
        ).first() is not None

        if not has_license and voice.licensing_required:
            return {
                "success": False,
                "error": "License required to use this voice",
                "license_cost": voice.license_cost,
                "message": "Please purchase a license to use this celebrity voice"
            }

        # Generate sample audio
        sample_result = await _generate_celebrity_sample(voice, text)

        # Log usage
        usage_record = VoiceUsage(
            user_id=current_user_id,
            celebrity_voice_id=voice_id,
            usage_type="sample_generation",
            text_length=len(text),
            audio_duration=sample_result.get("duration", 0),
            processing_time=sample_result.get("processing_time", 0),
            cost=0.01,  # Minimal cost for sample generation
            api_endpoint="marketplace_sample",
            success=sample_result.get("success", False)
        )

        db.add(usage_record)
        db.commit()

        return {
            "success": True,
            "sample_url": sample_result.get("audio_url"),
            "text": text,
            "duration": sample_result.get("duration"),
            "processing_time": sample_result.get("processing_time"),
            "voice_info": {
                "id": voice.id,
                "name": voice.name,
                "celebrity_type": voice.celebrity_type
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating voice sample: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

async def _get_marketplace_statistics(db: Session) -> Dict[str, Any]:
    """Get marketplace usage statistics"""
    try:
        from app.models.voice_models import User, VoiceProfile

        # Count total voices, users, licenses
        total_celebrity_voices = db.query(CelebrityVoice).filter(
            CelebrityVoice.is_active == True
        ).count()

        total_users = db.query(User).count()
        total_custom_voices = db.query(VoiceProfile).filter(
            VoiceProfile.voice_type == "custom"
        ).count()

        total_licenses = db.query(VoiceUsage).filter(
            VoiceUsage.usage_type.like("license_%")
        ).count()

        return {
            "total_celebrity_voices": total_celebrity_voices,
            "total_users": total_users,
            "total_custom_voices": total_custom_voices,
            "total_active_licenses": total_licenses,
            "most_popular_category": db.query(CelebrityVoice.category)
                                   .group_by(CelebrityVoice.category)
                                   .order_by(db.query(CelebrityVoice.category).count().desc())
                                   .first()[0] if total_celebrity_voices > 0 else None
        }

    except Exception as e:
        logger.error(f"Error getting marketplace statistics: {str(e)}")
        return {}

async def _generate_celebrity_sample(voice: CelebrityVoice, text: str) -> Dict[str, Any]:
    """Generate sample audio with celebrity voice"""
    try:
        import time
        import hashlib

        start_time = time.time()

        # Simulate celebrity voice synthesis
        # In production, this would use the actual voice model
        duration_estimate = len(text.split()) * 0.4  # Rough estimate

        processing_time = int((time.time() - start_time) * 1000)

        # Generate mock audio URL
        audio_hash = hashlib.md5(f"{voice.id}_{text}".encode()).hexdigest()
        audio_url = f"/samples/celebrity/{voice.id}_{audio_hash}.wav"

        return {
            "success": True,
            "audio_url": audio_url,
            "duration": duration_estimate,
            "processing_time": processing_time,
            "voice_model": voice.model_path
        }

    except Exception as e:
        logger.error(f"Error generating celebrity sample: {str(e)}")
        return {"success": False, "error": str(e)}