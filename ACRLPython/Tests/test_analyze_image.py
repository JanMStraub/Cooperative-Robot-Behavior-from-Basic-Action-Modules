#!/usr/bin/env python3
"""
Unit tests for AnalyzeImage.py

Tests the LM Studio vision processor and image analysis
"""

import pytest
import numpy as np
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add LLMCommunication directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "LLMCommunication"))

from LLMCommunication.vision.AnalyzeImage import LMStudioVisionProcessor, get_images_from_server, save_response
import LLMCommunication.config as cfg


class TestLMStudioVisionProcessorInitialization:
    """Test LMStudioVisionProcessor initialization"""

    @patch('LLMCommunication.vision.AnalyzeImage.OpenAI')
    def test_initialization_default_model(self, mock_openai_class):
        """Test initialization with default model"""
        mock_client = MagicMock()
        mock_client.models.list.return_value = []
        mock_openai_class.return_value = mock_client

        processor = LMStudioVisionProcessor()

        assert processor._model == cfg.DEFAULT_LMSTUDIO_MODEL
        assert processor._base_url == cfg.LMSTUDIO_BASE_URL
        mock_client.models.list.assert_called_once()

    @patch('LLMCommunication.vision.AnalyzeImage.OpenAI')
    def test_initialization_custom_model(self, mock_openai_class):
        """Test initialization with custom model"""
        mock_client = MagicMock()
        mock_client.models.list.return_value = []
        mock_openai_class.return_value = mock_client

        processor = LMStudioVisionProcessor(model="llava")

        assert processor._model == "llava"

    @patch('LLMCommunication.vision.AnalyzeImage.OpenAI')
    def test_initialization_custom_base_url(self, mock_openai_class):
        """Test initialization with custom base URL"""
        mock_client = MagicMock()
        mock_client.models.list.return_value = []
        mock_openai_class.return_value = mock_client

        processor = LMStudioVisionProcessor(base_url="http://custom-host:1234/v1")

        assert processor._base_url == "http://custom-host:1234/v1"
        mock_openai_class.assert_called_once_with(base_url="http://custom-host:1234/v1", api_key="not-needed")

    @patch('LLMCommunication.vision.AnalyzeImage.OpenAI')
    def test_initialization_connection_error(self, mock_openai_class):
        """Test initialization handles connection errors"""
        mock_client = MagicMock()
        mock_client.models.list.side_effect = Exception("Connection refused")
        mock_openai_class.return_value = mock_client

        with pytest.raises(ConnectionError, match="Cannot connect to LM Studio"):
            LMStudioVisionProcessor()


class TestLMStudioVisionProcessorEncoding:
    """Test image encoding functionality"""

    @patch('LLMCommunication.vision.AnalyzeImage.OpenAI')
    def test_encode_image_to_bytes(self, mock_openai_class, sample_image):
        """Test encoding numpy image to PNG bytes"""
        mock_client = MagicMock()
        mock_client.models.list.return_value = []
        mock_openai_class.return_value = mock_client

        processor = LMStudioVisionProcessor()

        encoded = processor.encode_image_to_bytes(sample_image)

        assert isinstance(encoded, bytes)
        assert len(encoded) > 0

    @patch('LLMCommunication.vision.AnalyzeImage.cv2.imencode')
    @patch('LLMCommunication.vision.AnalyzeImage.OpenAI')
    def test_encode_invalid_image_raises(self, mock_openai_class, mock_imencode):
        """Test encoding invalid image raises ValueError"""
        mock_client = MagicMock()
        mock_client.models.list.return_value = []
        mock_openai_class.return_value = mock_client

        # Mock cv2.imencode to return failure
        mock_imencode.return_value = (False, None)

        processor = LMStudioVisionProcessor()

        # Any image will fail due to mock
        invalid_image = np.array([[1, 2], [3, 4]])  # Wrong dimensions

        with pytest.raises(ValueError, match="Failed to encode image"):
            processor.encode_image_to_bytes(invalid_image)


class TestLMStudioVisionProcessorSendImages:
    """Test sending images to LM Studio"""

    @patch('LLMCommunication.vision.AnalyzeImage.OpenAI')
    def test_send_single_image(self, mock_openai_class, sample_image):
        """Test sending a single image for analysis"""
        mock_client = MagicMock()
        mock_client.models.list.return_value = []

        # Mock the chat completion response
        mock_choice = MagicMock()
        mock_choice.message.content = "I see a gradient image with red and blue colors."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        mock_openai_class.return_value = mock_client

        processor = LMStudioVisionProcessor()

        result = processor.send_images(
            images=[sample_image],
            camera_ids=["camera1"],
            prompt="What do you see?"
        )

        assert result["success"] is True
        assert result["response"] == "I see a gradient image with red and blue colors."
        assert result["metadata"]["model"] == cfg.DEFAULT_LMSTUDIO_MODEL
        assert result["metadata"]["image_count"] == 1
        assert result["metadata"]["camera_ids"] == ["camera1"]
        assert result["metadata"]["prompt"] == "What do you see?"
        assert "duration_seconds" in result["metadata"]

    @patch('LLMCommunication.vision.AnalyzeImage.OpenAI')
    def test_send_multiple_images(self, mock_openai_class, sample_image):
        """Test sending multiple images for analysis"""
        mock_client = MagicMock()
        mock_client.models.list.return_value = []

        mock_choice = MagicMock()
        mock_choice.message.content = "I see multiple camera views."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        mock_openai_class.return_value = mock_client

        processor = LMStudioVisionProcessor()

        result = processor.send_images(
            images=[sample_image, sample_image],
            camera_ids=["camera1", "camera2"],
            prompt="Compare these views"
        )

        assert result["success"] is True
        assert result["metadata"]["image_count"] == 2
        assert len(result["metadata"]["camera_ids"]) == 2

    @patch('LLMCommunication.vision.AnalyzeImage.OpenAI')
    def test_send_images_with_custom_temperature(self, mock_openai_class, sample_image):
        """Test sending images with custom temperature"""
        mock_client = MagicMock()
        mock_client.models.list.return_value = []

        mock_choice = MagicMock()
        mock_choice.message.content = "Response"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        mock_openai_class.return_value = mock_client

        processor = LMStudioVisionProcessor()

        result = processor.send_images(
            images=[sample_image],
            camera_ids=["camera1"],
            prompt="Test",
            temperature=0.5
        )

        # Verify chat was called with correct temperature
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["temperature"] == 0.5

    @patch('LLMCommunication.vision.AnalyzeImage.OpenAI')
    def test_send_images_empty_list_raises(self, mock_openai_class):
        """Test that sending empty image list raises ValueError"""
        mock_client = MagicMock()
        mock_client.models.list.return_value = []
        mock_openai_class.return_value = mock_client

        processor = LMStudioVisionProcessor()

        with pytest.raises(ValueError, match="No images provided"):
            processor.send_images(
                images=[],
                camera_ids=[],
                prompt="Test"
            )

    @patch('LLMCommunication.vision.AnalyzeImage.OpenAI')
    def test_send_images_lmstudio_error(self, mock_openai_class, sample_image):
        """Test handling of LM Studio errors"""
        mock_client = MagicMock()
        mock_client.models.list.return_value = []
        mock_client.chat.completions.create.side_effect = Exception("LM Studio model not found")
        mock_openai_class.return_value = mock_client

        processor = LMStudioVisionProcessor()

        with pytest.raises(Exception, match="LM Studio"):
            processor.send_images(
                images=[sample_image],
                camera_ids=["camera1"],
                prompt="Test"
            )

    @patch('LLMCommunication.vision.AnalyzeImage.OpenAI')
    def test_send_images_multiple_cameras_adds_context(self, mock_openai_class, sample_image):
        """Test that multiple camera prompt includes camera context"""
        mock_client = MagicMock()
        mock_client.models.list.return_value = []

        mock_choice = MagicMock()
        mock_choice.message.content = "Response"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        mock_openai_class.return_value = mock_client

        processor = LMStudioVisionProcessor()

        result = processor.send_images(
            images=[sample_image, sample_image],
            camera_ids=["camera1", "camera2"],
            prompt="What do you see?"
        )

        # Full prompt should include camera context
        full_prompt = result["metadata"]["full_prompt"]
        assert "camera1" in full_prompt
        assert "camera2" in full_prompt
        assert "What do you see?" in full_prompt


class TestGetImagesFromServer:
    """Test get_images_from_server function"""

    @patch('LLMCommunication.vision.AnalyzeImage.ImageStorage')
    def test_get_images_all_cameras(self, mock_storage_class, sample_image):
        """Test getting images from all cameras"""
        mock_storage = MagicMock()
        mock_storage.get_all_camera_ids.return_value = ["camera1", "camera2"]
        mock_storage.get_camera_image.return_value = sample_image
        mock_storage.get_camera_prompt.return_value = "test prompt"
        mock_storage.get_camera_age.return_value = 1.0
        mock_storage_class.get_instance.return_value = mock_storage

        images, camera_ids, prompts = get_images_from_server()

        assert len(images) == 2
        assert len(camera_ids) == 2
        assert len(prompts) == 2
        assert camera_ids == ["camera1", "camera2"]

    @patch('LLMCommunication.vision.AnalyzeImage.ImageStorage')
    def test_get_images_specific_cameras(self, mock_storage_class, sample_image):
        """Test getting images from specific cameras"""
        mock_storage = MagicMock()
        mock_storage.get_camera_image.return_value = sample_image
        mock_storage.get_camera_prompt.return_value = "prompt"
        mock_storage.get_camera_age.return_value = 1.0
        mock_storage_class.get_instance.return_value = mock_storage

        images, camera_ids, prompts = get_images_from_server(["camera1"])

        assert len(images) == 1
        assert camera_ids == ["camera1"]

    @patch('LLMCommunication.vision.AnalyzeImage.ImageStorage')
    def test_get_images_no_cameras_raises(self, mock_storage_class):
        """Test that no cameras raises ValueError"""
        mock_storage = MagicMock()
        mock_storage.get_all_camera_ids.return_value = []
        mock_storage_class.get_instance.return_value = mock_storage

        with pytest.raises(ValueError, match="No cameras available"):
            get_images_from_server()

    @patch('LLMCommunication.vision.AnalyzeImage.ImageStorage')
    def test_get_images_missing_camera(self, mock_storage_class):
        """Test handling of missing camera images"""
        mock_storage = MagicMock()
        mock_storage.get_camera_image.return_value = None
        mock_storage_class.get_instance.return_value = mock_storage

        with pytest.raises(ValueError, match="No images available"):
            get_images_from_server(["nonexistent"])


class TestSaveResponse:
    """Test save_response function"""

    def test_save_response_default_path(self, llm_result_dict, tmp_path, monkeypatch):
        """Test saving response with default path"""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        save_response(llm_result_dict)

        # Should create a JSON file
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 1

        # Verify content
        with open(json_files[0], 'r') as f:
            saved_data = json.load(f)
            assert saved_data["response"] == llm_result_dict["response"]

    def test_save_response_custom_path(self, llm_result_dict, tmp_path):
        """Test saving response with custom path"""
        output_path = tmp_path / "custom_output"

        save_response(llm_result_dict, str(output_path))

        # Should create custom named file
        json_file = tmp_path / "custom_output.json"
        assert json_file.exists()

        # Verify content
        with open(json_file, 'r') as f:
            saved_data = json.load(f)
            assert saved_data == llm_result_dict

    def test_save_response_unicode_content(self, tmp_path):
        """Test saving response with unicode characters"""
        result = {
            "response": "I see 日本語 characters 🤖",
            "camera_id": "Camera_日本語"
        }

        output_path = tmp_path / "unicode_test"
        save_response(result, str(output_path))

        json_file = tmp_path / "unicode_test.json"
        assert json_file.exists()

        # Verify unicode is preserved
        with open(json_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
            assert saved_data["response"] == result["response"]
            assert "日本語" in saved_data["response"]
            assert "🤖" in saved_data["response"]


class TestLMStudioVisionProcessorMetadata:
    """Test metadata in results"""

    @patch('LLMCommunication.vision.AnalyzeImage.OpenAI')
    def test_result_includes_timestamp(self, mock_openai_class, sample_image):
        """Test that result includes timestamp"""
        mock_client = MagicMock()
        mock_client.models.list.return_value = []

        mock_choice = MagicMock()
        mock_choice.message.content = "Response"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        mock_openai_class.return_value = mock_client

        processor = LMStudioVisionProcessor()

        result = processor.send_images(
            images=[sample_image],
            camera_ids=["camera1"],
            prompt="Test"
        )

        assert "timestamp" in result["metadata"]
        # Verify timestamp is valid ISO format
        timestamp = result["metadata"]["timestamp"]
        datetime.fromisoformat(timestamp)  # Should not raise

    @patch('LLMCommunication.vision.AnalyzeImage.OpenAI')
    def test_result_includes_duration(self, mock_openai_class, sample_image):
        """Test that result includes processing duration"""
        mock_client = MagicMock()
        mock_client.models.list.return_value = []

        mock_choice = MagicMock()
        mock_choice.message.content = "Response"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        mock_openai_class.return_value = mock_client

        processor = LMStudioVisionProcessor()

        result = processor.send_images(
            images=[sample_image],
            camera_ids=["camera1"],
            prompt="Test"
        )

        assert "duration_seconds" in result["metadata"]
        assert isinstance(result["metadata"]["duration_seconds"], float)
        assert result["metadata"]["duration_seconds"] >= 0
