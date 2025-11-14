import os
import json
import logging
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session
from transformers import AutoTokenizer, AutoModel
import librosa

from app.models.voice_models import VoiceProfile, EmotionMapping

logger = logging.getLogger(__name__)

class EmotionEngine:
    """Advanced emotion detection and expressive response generation"""

    def __init__(self, db: Session):
        self.db = db
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emotion_model = None
        self.sentiment_model = None
        self.voice_emotion_mappings = {}
        self._load_emotion_models()

    def _load_emotion_models(self):
        """Load emotion detection and sentiment analysis models"""
        try:
            # Load emotion detection model
            self.emotion_model = AutoModel.from_pretrained(
                "facebook/wav2vec2-large-xlsr-53",
                trust_remote_code=True
            )
            self.emotion_model.to(self.device)

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "facebook/wav2vec2-large-xlsr-53"
            )

            logger.info("Emotion models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading emotion models: {str(e)}")
            # Fallback to rule-based emotion detection
            self.emotion_model = None

    async def analyze_emotion(self, text: str, audio_features: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze emotion from text and/or audio"""
        try:
            # Initialize result
            emotion_result = {
                "primary_emotion": "neutral",
                "secondary_emotions": [],
                "confidence": 0.5,
                "intensity": 0.5,
                "valence": 0.0,  # Positive-negative scale (-1 to 1)
                "arousal": 0.5,  # Calm-excited scale (0 to 1)
                "dominance": 0.5,  # Submissive-dominant scale (0 to 1)
                "analysis_method": "hybrid"
            }

            # Text-based emotion analysis
            text_emotion = await self._analyze_text_emotion(text)

            # Audio-based emotion analysis (if available)
            audio_emotion = None
            if audio_features:
                audio_emotion = await self._analyze_audio_emotion(audio_features)

            # Combine analyses
            if audio_emotion and text_emotion:
                # Weight audio higher if confidence is higher
                text_weight = 0.4
                audio_weight = 0.6

                combined_emotion = self._combine_emotion_analyses(
                    text_emotion, audio_emotion, text_weight, audio_weight
                )
                emotion_result.update(combined_emotion)
            elif text_emotion:
                emotion_result.update(text_emotion)
            elif audio_emotion:
                emotion_result.update(audio_emotion)

            return emotion_result

        except Exception as e:
            logger.error(f"Error analyzing emotion: {str(e)}")
            return {
                "primary_emotion": "neutral",
                "confidence": 0.3,
                "error": str(e)
            }

    async def _analyze_text_emotion(self, text: str) -> Dict[str, Any]:
        """Analyze emotion from text using NLP"""
        try:
            # Emotion keyword mapping
            emotion_keywords = {
                "happy": {
                    "words": ["happy", "joy", "excited", "wonderful", "amazing", "great", "fantastic", "love", "excellent", "smile", "laugh"],
                    "emojis": ["😊", "😄", "🎉", "💖", "✨"],
                    "intensifiers": ["very", "extremely", "really", "so", "absolutely"]
                },
                "sad": {
                    "words": ["sad", "unhappy", "depressed", "disappointed", "terrible", "awful", "hate", "cry", "tear"],
                    "emojis": ["😢", "😭", "💔", "😞"],
                    "intensifiers": ["very", "extremely", "deeply", "terribly"]
                },
                "angry": {
                    "words": ["angry", "furious", "mad", "enraged", "annoyed", "frustrated", "irritated"],
                    "emojis": ["😠", "😡", "💢", "👿"],
                    "intensifiers": ["very", "extremely", "really", "so"]
                },
                "excited": {
                    "words": ["excited", "thrilled", "enthusiastic", "eager", "cant wait", "looking forward"],
                    "emojis": ["🤩", "🎊", "🌟", "💫"],
                    "intensifiers": ["very", "extremely", "super", "totally"]
                },
                "calm": {
                    "words": ["calm", "relaxed", "peaceful", "serene", "tranquil", "comfortable"],
                    "emojis": ["😌", "🧘", "🕊"],
                    "intensifiers": ["very", "quite", "rather", "somewhat"]
                },
                "confused": {
                    "words": ["confused", "unclear", "dont understand", "what do you mean", "not sure", "puzzled"],
                    "emojis": ["😕", "🤔", "❓"],
                    "intensifiers": ["very", "quite", "rather", "somewhat"]
                }
            }

            text_lower = text.lower()
            emotion_scores = {}

            # Score each emotion
            for emotion, config in emotion_keywords.items():
                score = 0

                # Count emotion words
                for word in config["words"]:
                    if word in text_lower:
                        score += 1

                # Count emojis
                for emoji in config["emojis"]:
                    if emoji in text:
                        score += 2  # Emojis are stronger indicators

                # Check for intensifiers
                has_intensifier = any(intensifier in text_lower for intensifier in config["intensifiers"])
                if has_intensifier:
                    score *= 1.5

                emotion_scores[emotion] = score

            # Determine primary emotion and calculate properties
            if not emotion_scores or max(emotion_scores.values()) == 0:
                return {
                    "primary_emotion": "neutral",
                    "confidence": 0.8,
                    "valence": 0.0,
                    "arousal": 0.5
                }

            # Sort emotions by score
            sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
            primary_emotion = sorted_emotions[0][0]
            primary_score = sorted_emotions[0][1]

            # Get secondary emotions (those with significant scores)
            secondary_emotions = [
                emotion for emotion, score in sorted_emotions[1:4]
                if score > 0 and score >= primary_score * 0.3
            ]

            # Calculate emotional properties
            emotion_properties = self._calculate_emotional_properties(primary_emotion, primary_score, text_lower)

            return {
                "primary_emotion": primary_emotion,
                "secondary_emotions": secondary_emotions,
                "confidence": min(0.9, primary_score / 5.0),
                "intensity": primary_score / max(max(emotion_scores.values()), 1),
                **emotion_properties,
                "analysis_method": "text_nlp"
            }

        except Exception as e:
            logger.error(f"Error in text emotion analysis: {str(e)}")
            return {"primary_emotion": "neutral", "confidence": 0.3}

    async def _analyze_audio_emotion(self, audio_features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze emotion from audio features"""
        try:
            if not self.emotion_model:
                return self._fallback_audio_emotion_analysis(audio_features)

            # Extract audio features if not provided
            if "audio_data" not in audio_features:
                return {"primary_emotion": "neutral", "confidence": 0.3}

            # Use pre-trained model for emotion recognition
            audio_tensor = torch.tensor(audio_features["audio_data"]).unsqueeze(0)

            with torch.no_grad():
                if hasattr(self.emotion_model, 'predict'):
                    emotions = self.emotion_model.predict(audio_tensor)
                else:
                    # Fallback to feature extraction
                    features = self.emotion_model.extract_features(audio_tensor)
                    emotions = self._classify_emotions_from_features(features)

            return emotions

        except Exception as e:
            logger.error(f"Error in audio emotion analysis: {str(e)}")
            return self._fallback_audio_emotion_analysis(audio_features)

    def _fallback_audio_emotion_analysis(self, audio_features: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback rule-based audio emotion analysis"""
        try:
            if "audio_data" not in audio_features:
                return {"primary_emotion": "neutral", "confidence": 0.3}

            audio = audio_features["audio_data"]

            # Extract acoustic features
            # Pitch analysis
            pitches, magnitudes = librosa.piptrack(
                y=audio, sr=audio_features.get("sample_rate", 16000)
            )
            pitch_mean = np.nanmean(pitches)
            pitch_std = np.nanstd(pitches)

            # Energy analysis
            energy = librosa.feature.rms(y=audio)
            energy_mean = np.mean(energy)
            energy_std = np.std(energy)

            # Tempo analysis
            tempo, _ = librosa.beat.beat_track(y=audio, sr=audio_features.get("sample_rate", 16000))

            # Zero crossing rate (indicates speech activity)
            zcr = librosa.feature.zero_crossing_rate(audio)

            # Simple emotion classification based on features
            emotion_scores = {}

            # Happy/excited: higher pitch variation and energy
            emotion_scores["happy"] = min(1.0, (pitch_std / 50) * 0.5 + energy_mean * 0.5)

            # Sad: lower pitch, lower energy
            emotion_scores["sad"] = max(0, 1.0 - (pitch_mean / 200) * 0.5) * (1.0 - energy_mean) * 0.5

            # Angry: high energy, high pitch
            emotion_scores["angry"] = min(1.0, energy_mean * 0.7 + (pitch_mean / 300) * 0.3)

            # Calm: low energy variation, moderate pitch
            emotion_scores["calm"] = max(0, 1.0 - energy_std) * (1.0 - abs(pitch_std - 20) / 50)

            # Find dominant emotion
            if not emotion_scores:
                return {"primary_emotion": "neutral", "confidence": 0.3}

            sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
            primary_emotion = sorted_emotions[0][0]
            confidence = sorted_emotions[0][1]

            return {
                "primary_emotion": primary_emotion,
                "confidence": confidence,
                "analysis_method": "audio_acoustic",
                "features": {
                    "pitch_mean": float(pitch_mean) if not np.isnan(pitch_mean) else 150,
                    "pitch_std": float(pitch_std) if not np.isnan(pitch_std) else 20,
                    "energy_mean": float(energy_mean),
                    "energy_std": float(energy_std),
                    "tempo": tempo if tempo else 120
                }
            }

        except Exception as e:
            logger.error(f"Error in fallback audio emotion analysis: {str(e)}")
            return {"primary_emotion": "neutral", "confidence": 0.2}

    def _calculate_emotional_properties(self, emotion: str, score: int, text: str) -> Dict[str, float]:
        """Calculate valence, arousal, and dominance from emotion"""
        # Emotion property mapping
        emotion_properties = {
            "happy": {"valence": 0.8, "arousal": 0.7, "dominance": 0.6},
            "sad": {"valence": -0.7, "arousal": 0.3, "dominance": 0.2},
            "angry": {"valence": -0.6, "arousal": 0.9, "dominance": 0.8},
            "excited": {"valence": 0.9, "arousal": 0.9, "dominance": 0.7},
            "calm": {"valence": 0.2, "arousal": 0.2, "dominance": 0.5},
            "confused": {"valence": -0.2, "arousal": 0.6, "dominance": 0.3},
            "fear": {"valence": -0.8, "arousal": 0.8, "dominance": 0.1},
            "surprised": {"valence": 0.4, "arousal": 0.8, "dominance": 0.4}
        }

        base_properties = emotion_properties.get(emotion, {
            "valence": 0.0, "arousal": 0.5, "dominance": 0.5
        })

        # Adjust intensity based on score
        intensity_factor = min(1.0, score / 3.0)

        return {
            "valence": base_properties["valence"] * intensity_factor,
            "arousal": base_properties["arousal"] * intensity_factor,
            "dominance": base_properties["dominance"] * intensity_factor
        }

    def _combine_emotion_analyses(self, text_emotion: Dict[str, Any],
                                   audio_emotion: Dict[str, Any],
                                   text_weight: float, audio_weight: float) -> Dict[str, Any]:
        """Combine text and audio emotion analyses"""
        try:
            # Weighted averaging of primary emotions
            combined_emotion = {}

            # Combine valence, arousal, dominance
            text_valence = text_emotion.get("valence", 0)
            audio_valence = audio_emotion.get("valence", 0)
            combined_valence = text_valence * text_weight + audio_valence * audio_weight

            text_arousal = text_emotion.get("arousal", 0.5)
            audio_arousal = audio_emotion.get("arousal", 0.5)
            combined_arousal = text_arousal * text_weight + audio_arousal * audio_weight

            text_dominance = text_emotion.get("dominance", 0.5)
            audio_dominance = audio_emotion.get("dominance", 0.5)
            combined_dominance = text_dominance * text_weight + audio_dominance * audio_weight

            # Determine primary emotion from combined properties
            primary_emotion = self._determine_emotion_from_properties(
                combined_valence, combined_arousal, combined_dominance
            )

            # Combine confidence
            text_confidence = text_emotion.get("confidence", 0.5)
            audio_confidence = audio_emotion.get("confidence", 0.5)
            combined_confidence = text_confidence * text_weight + audio_confidence * audio_weight

            # Combine secondary emotions
            text_secondary = set(text_emotion.get("secondary_emotions", []))
            audio_secondary = set(audio_emotion.get("secondary_emotions", []))

            # Weight secondary emotions by their source confidence
            combined_secondary = []
            for emotion in text_secondary.union(audio_secondary):
                if emotion in text_secondary and emotion in audio_secondary:
                    combined_secondary.append(emotion)
                elif emotion in text_secondary:
                    if text_confidence > 0.6:
                        combined_secondary.append(emotion)
                elif emotion in audio_secondary:
                    if audio_confidence > 0.6:
                        combined_secondary.append(emotion)

            return {
                "primary_emotion": primary_emotion,
                "secondary_emotions": list(set(combined_secondary)),
                "confidence": combined_confidence,
                "intensity": (text_emotion.get("intensity", 0.5) * text_weight +
                             audio_emotion.get("intensity", 0.5) * audio_weight),
                "valence": combined_valence,
                "arousal": combined_arousal,
                "dominance": combined_dominance,
                "analysis_method": "hybrid_combined"
            }

        except Exception as e:
            logger.error(f"Error combining emotion analyses: {str(e)}")
            return {
                "primary_emotion": "neutral",
                "confidence": 0.3,
                "analysis_method": "hybrid_failed"
            }

    def _determine_emotion_from_properties(self, valence: float,
                                       arousal: float, dominance: float) -> str:
        """Determine emotion from valence, arousal, dominance properties"""
        # Simple rule-based mapping
        if valence > 0.6 and arousal > 0.6:
            return "excited"
        elif valence > 0.6 and arousal <= 0.6:
            return "happy"
        elif valence < -0.6 and arousal > 0.6:
            return "angry"
        elif valence < -0.6 and arousal <= 0.4:
            return "sad"
        elif arousal <= 0.4 and dominance <= 0.4:
            return "calm"
        elif valence < 0 and arousal > 0.6:
            return "confused"
        else:
            return "neutral"

    async def map_emotion_to_voice_parameters(self, voice_profile_id: int, emotion: Dict[str, Any]) -> Dict[str, Any]:
        """Map emotion to voice synthesis parameters"""
        try:
            # Get emotion mappings from database
            mappings = self.db.query(EmotionMapping).filter(
                EmotionMapping.voice_profile_id == voice_profile_id
            ).all()

            mapping_dict = {}
            for mapping in mappings:
                mapping_dict[mapping.emotion] = {
                    "pitch_adjustment": mapping.pitch_adjustment,
                    "speed_adjustment": mapping.speed_adjustment,
                    "volume_adjustment": mapping.volume_adjustment,
                    "timbre_adjustment": json.loads(mapping.timbre_adjustment or "{}"),
                    "intensity": mapping.intensity
                }

            # Get base voice profile
            voice_profile = self.db.query(VoiceProfile).filter(
                VoiceProfile.id == voice_profile_id
            ).first()

            if not voice_profile:
                return {"error": "Voice profile not found"}

            # Default emotion to voice parameter mapping
            emotion_name = emotion.get("primary_emotion", "neutral")
            intensity = emotion.get("intensity", 0.5)

            default_mappings = {
                "happy": {
                    "pitch_adjustment": 1.1,
                    "speed_adjustment": 1.05,
                    "volume_adjustment": 1.1,
                    "intensity": intensity
                },
                "sad": {
                    "pitch_adjustment": 0.9,
                    "speed_adjustment": 0.95,
                    "volume_adjustment": 0.9,
                    "intensity": intensity
                },
                "angry": {
                    "pitch_adjustment": 1.15,
                    "speed_adjustment": 1.1,
                    "volume_adjustment": 1.2,
                    "intensity": intensity
                },
                "excited": {
                    "pitch_adjustment": 1.2,
                    "speed_adjustment": 1.15,
                    "volume_adjustment": 1.15,
                    "intensity": intensity
                },
                "calm": {
                    "pitch_adjustment": 1.0,
                    "speed_adjustment": 0.95,
                    "volume_adjustment": 0.95,
                    "intensity": intensity
                },
                "confused": {
                    "pitch_adjustment": 0.95,
                    "speed_adjustment": 0.9,
                    "volume_adjustment": 0.9,
                    "intensity": intensity * 0.8
                },
                "neutral": {
                    "pitch_adjustment": 1.0,
                    "speed_adjustment": 1.0,
                    "volume_adjustment": 1.0,
                    "intensity": intensity
                }
            }

            # Use custom mappings if available, otherwise use defaults
            voice_params = mapping_dict.get(emotion_name, default_mappings.get(emotion_name, default_mappings["neutral"]))

            # Apply to base voice parameters
            base_pitch = voice_profile.pitch or 1.0
            base_speed = voice_profile.speed or 1.0
            base_volume = voice_profile.volume or 1.0

            adjusted_params = {
                "pitch": base_pitch * voice_params["pitch_adjustment"],
                "speed": base_speed * voice_params["speed_adjustment"],
                "volume": base_volume * voice_params["volume_adjustment"],
                "intensity": voice_params["intensity"],
                "emotion": emotion_name,
                "timbre_adjustments": voice_params.get("timbre_adjustment", {}),
                "applied_mappings": voice_params
            }

            return {
                "success": True,
                "voice_parameters": adjusted_params,
                "original_emotion": emotion,
                "voice_profile_id": voice_profile_id
            }

        except Exception as e:
            logger.error(f"Error mapping emotion to voice parameters: {str(e)}")
            return {"error": str(e)}

    async def create_emotion_mapping(self, voice_profile_id: int, emotion: str,
                                voice_params: Dict[str, Any]) -> Dict[str, Any]:
        """Create custom emotion mapping for voice profile"""
        try:
            # Check if mapping already exists
            existing_mapping = self.db.query(EmotionMapping).filter(
                and_(EmotionMapping.voice_profile_id == voice_profile_id,
                    EmotionMapping.emotion == emotion)
            ).first()

            if existing_mapping:
                # Update existing mapping
                existing_mapping.pitch_adjustment = voice_params.get("pitch_adjustment", 1.0)
                existing_mapping.speed_adjustment = voice_params.get("speed_adjustment", 1.0)
                existing_mapping.volume_adjustment = voice_params.get("volume_adjustment", 1.0)
                existing_mapping.timbre_adjustment = json.dumps(
                    voice_params.get("timbre_adjustment", {})
                )
                existing_mapping.intensity = voice_params.get("intensity", 0.5)
                self.db.commit()

                return {"success": True, "message": "Emotion mapping updated"}
            else:
                # Create new mapping
                new_mapping = EmotionMapping(
                    voice_profile_id=voice_profile_id,
                    emotion=emotion,
                    pitch_adjustment=voice_params.get("pitch_adjustment", 1.0),
                    speed_adjustment=voice_params.get("speed_adjustment", 1.0),
                    volume_adjustment=voice_params.get("volume_adjustment", 1.0),
                    timbre_adjustment=json.dumps(
                        voice_params.get("timbre_adjustment", {})
                    ),
                    intensity=voice_params.get("intensity", 0.5)
                )

                self.db.add(new_mapping)
                self.db.commit()

                return {"success": True, "message": "Emotion mapping created"}

        except Exception as e:
            logger.error(f"Error creating emotion mapping: {str(e)}")
            return {"success": False, "error": str(e)}