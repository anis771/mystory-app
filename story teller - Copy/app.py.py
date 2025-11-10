import os
import traceback
import uuid
import random
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import re
import time
from typing import List, Dict, Optional
from werkzeug.utils import secure_filename, send_from_directory
from keybert import KeyBERT
import websocket
import threading
from urllib.parse import urlencode
from urllib.request import urlopen, Request
import yake
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import math
import csv
from pathlib import Path
from datetime import datetime
import numpy as np
from flask import send_file
import torch
import io
import urllib.request
import pyiqa

from torchvision.models.inception import inception_v3
from scipy.linalg import sqrtm
import torchvision.transforms as transforms
from bert_score import score
from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk


# from tensorflow.keras.applications import MobileNet
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.mobilenet import preprocess_input
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# import tensorflow_hub as hub
# from tensorflow.keras.models import load_model

from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Download NLTK data (run once)
nltk.download("punkt")
nltk.download('vader_lexicon')
 
# 3. Consistency (text-based, character persistence)
import re


# === Safe Import Block: Try to load dependencies ===
torch = None
T = None
open_clip = None
clip_pkg = None
_CLIP_BACKEND = None


try:
    import torch
    import torchvision.transforms as T
except ImportError:
    pass

if torch is not None:
    try:
        import open_clip

        _CLIP_BACKEND = "open_clip"
    except ImportError:
        try:
            import clip as clip_pkg

            _CLIP_BACKEND = "clip"
        except ImportError:
            pass

# If no backend is available, disable everything
if _CLIP_BACKEND is None:
    torch = None
    T = None
    print("CLIP backend not available: install 'open_clip_torch' or 'clip' package.")


# Simple stopwords for baseline extractor
try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    STOPWORDS = set(ENGLISH_STOP_WORDS)
except Exception:
    STOPWORDS = set(
        [
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "if",
            "while",
            "to",
            "of",
            "in",
            "on",
            "for",
            "with",
            "as",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "it",
            "this",
            "that",
            "these",
            "those",
            "at",
            "from",
            "up",
            "down",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "any",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
        ]
    )


app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
UPLOAD_FOLDER = "generated_images"
BACKEND_URL = "http://localhost:5000"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class ComfyUIClient:
    def __init__(self, server_address="127.0.0.1:8000"):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())

    def queue_prompt(self, prompt):
        """Submit a prompt to ComfyUI"""
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode("utf-8")
        req = Request(f"http://{self.server_address}/prompt", data=data)
        req.add_header("Content-Type", "application/json")

        try:
            response = urlopen(req)
            return json.loads(response.read())
        except Exception as e:
            print(f"Error queuing prompt: {e}")
            return None

    def get_image(self, filename, subfolder, folder_type):
        """Download generated image from ComfyUI"""
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urlencode(data)

        try:
            with urlopen(f"http://{self.server_address}/view?{url_values}") as response:
                return response.read()
        except Exception as e:
            print(f"Error getting image: {e}")
            return None

    def get_history(self, prompt_id):
        """Get execution history for a prompt"""
        try:
            with urlopen(
                f"http://{self.server_address}/history/{prompt_id}"
            ) as response:
                return json.loads(response.read())
        except Exception as e:
            print(f"Error getting history: {e}")
            return None

    def wait_for_completion(self, prompt_id, timeout=900):
        """Wait for prompt completion using WebSocket"""
        ws_url = f"ws://{self.server_address}/ws?clientId={self.client_id}"
        completed = False
        result_data = None

        def on_message(ws, message):
            nonlocal completed, result_data
            try:
                data = json.loads(message)
                if data["type"] == "executing":
                    executing_data = data["data"]
                    if (
                        executing_data["node"] is None
                        and executing_data["prompt_id"] == prompt_id
                    ):
                        completed = True
                        history = self.get_history(prompt_id)

                        if history and prompt_id in history:
                            result_data = history[prompt_id]
                        ws.close()
            except Exception as e:
                print(f"WebSocket message error: {e}")

        def on_error(ws, error):
            print(f"WebSocket error: {error}")

        def on_close(ws, close_status_code, close_msg):
            pass

        try:
            ws = websocket.WebSocketApp(
                ws_url, on_message=on_message, on_error=on_error, on_close=on_close
            )

            ws_thread = threading.Thread(target=ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()

            start_time = time.time()
            while not completed and (time.time() - start_time) < timeout:
                time.sleep(0.5)

            ws.close()

            if completed:
                return result_data
            else:
                print("Timeout waiting for completion")
                return None

        except Exception as e:
            print(f"WebSocket connection error: {e}")
            return None


class ImprovedStoryGenerator:
    def __init__(self):
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        comfyui_base = os.getenv("COMFYUI_URL", "http://localhost:8000")
        self.comfyui_url = f"{comfyui_base}/prompt"
        comfyui_host = comfyui_base.replace("http://", "").replace("https://", "")
        self.comfyui_client = ComfyUIClient(comfyui_host)
        self.model_warmed = False

        # Story context tracking
        self.current_story_context = {
            "characters": {},
            "settings": [],
            "art_style": None,
            "mood": None,
            "theme": None,
        }

        # Improved workflow configuration
        self.workflow_config = {
            "positive_prompt_prefix": "",  # Will be set dynamically based on story
            "negative_prompt": "low quality, blurry, distorted, bad anatomy, bad hands, extra limbs, extra fingers, multiple heads, disfigured, ugly, poorly drawn, text, watermark, signature, frame, border, glitch, inconsistent style, modern elements in fantasy setting",
            "steps": 25,
            "cfg_scale": 8,
            "width": 1024,
            "height": 1024,
            "sampler": "euler",
            "seed": -1,
            "base_steps": 20,
            "refiner_steps": 25,
            "base_end_step": 20,
            "refiner_start_step": 20,
        }
        self.initClip()

    def initClip(self):
        # ===== Metrics: CLIP encoder (optional) =====
        self.clip_device = None
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_ctx_len = 77

        if _CLIP_BACKEND and torch is not None:
            try:
                self.clip_device = "cuda" if torch.cuda.is_available() else "cpu"
                if _CLIP_BACKEND == "open_clip":
                    # ViT-B/32 is fast and good enough for comparisons
                    self.clip_model, _, self.clip_preprocess = (
                        open_clip.create_model_and_transforms(
                            "ViT-B-32", pretrained="openai"
                        )
                    )
                    self.clip_model = self.clip_model.to(self.clip_device).eval()
                    self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
                else:
                    # Fallback to OpenAI CLIP package
                    self.clip_model, self.clip_preprocess = clip_pkg.load(
                        "ViT-B/32", device=self.clip_device
                    )
                print("✅ CLIP loaded for metrics")
            except Exception as e:
                print(f"⚠️ CLIP could not be initialized: {e}")
                self.clip_device = None

    def warm_up_model(self) -> Dict:
        """Warm up the Ollama model and keep it alive"""
        try:
            print("🔥 Warming up Ollama model...")
            data = {
                "model": "llama3",
                "prompt": "Hello, I'm ready to generate stories!",
                "stream": False,
                "keep_alive": "30m",  # Keep model loaded for 30 minutes
                "options": {"num_predict": 10, "temperature": 0.7},
            }

            response = requests.post(
                self.ollama_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(data),
                timeout=60,
            )

            if response.status_code == 200:
                self.model_warmed = True
                print("✅ Model warmed up successfully! \n")
                return {"status": "success", "message": "Model is ready"}
            else:
                return {
                    "status": "error",
                    "message": f"Failed to warm up model: {response.status_code}",
                }

        except Exception as e:
            return {"status": "error", "message": f"Warmup failed: {str(e)}"}

    def generate_story(self, user_prompt: str, num_paragraphs: int = 5) -> Dict:
        """Enhanced story generation with better context preservation"""
        try:
            # First, extract story theme and establish consistent style
            theme_analysis = self._analyze_story_theme(user_prompt)
            self.current_story_context.update(theme_analysis)

            story_prompt = f"""
            Write a creative {num_paragraphs}-paragraph story based on the following idea:
            "{user_prompt}"

            Story Requirements:
            - The story should be imaginative, engaging, and rich in descriptive language.
            - Each paragraph must be between 10 to 15 sentences.
            - Focus on vivid **visual scenes** with strong **characters**, **settings**, and **actions** that are easy to illustrate.
            - IMPORTANT: Maintain consistent character names and descriptions throughout the story.
            - IMPORTANT: Keep settings and locations consistent and clearly described.
            - Include emotional tone, sensory details (sight, sound, feel), and dynamic interactions.
            - Ensure the story flows naturally across the paragraphs.
            - Use clear paragraph breaks with this exact marker: ---PARAGRAPH---
            
            Additional context for consistency:
            - Theme: {theme_analysis.get('theme', 'adventure')}
            - Setting type: {theme_analysis.get('setting_type', 'fantasy')}
            - Target mood: {theme_analysis.get('mood', 'whimsical')}

            Begin the story:
            """

            data = {
                "model": "llama3",
                "prompt": story_prompt,
                "stream": False,
                "keep_alive": "30m",
                "options": {"num_predict": 1500, "temperature": 0.8, "num_ctx": 8192},
            }

            print(f"--------------------------------------")
            print(f"📝 Generating story for: {user_prompt}")
            print(f"--------------------------------------")
            print(f"\n\n")

            response = requests.post(
                self.ollama_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(data),
                timeout=120,
            )

            if response.status_code == 200:
                result = response.json()
                story_text = result.get("response", "")

                # Extract and track story elements
                self._extract_story_elements(story_text)

                # Split story into paragraphs with context
                paragraphs = self._split_story_with_context(story_text)

                return {
                    "status": "success",
                    "story": {
                        "title": self._generate_title(user_prompt),
                        "paragraphs": paragraphs,
                        "total_paragraphs": len(paragraphs),
                        "context": self.current_story_context,
                    },
                }
            else:
                return {
                    "status": "error",
                    "message": f"Story generation failed: {response.status_code}",
                }

        except Exception as e:
            return {"status": "error", "message": f"Story generation error: {str(e)}"}

    def _analyze_story_theme(self, user_prompt: str) -> Dict:
        """Analyze the story prompt to establish consistent theme and style"""
        try:
            analysis_prompt = f"""
            Analyze this story idea and provide a JSON response with the following elements:
            Story idea: "{user_prompt}"
            
            Return ONLY a JSON object with:
            {{
                "theme": "main theme (e.g., adventure, mystery, fantasy, sci-fi)",
                "art_style": "best art style for this story (e.g., watercolor fantasy, digital painting, oil painting, anime style)",
                "mood": "overall mood (e.g., whimsical, dark, cheerful, mysterious)",
                "setting_type": "setting category (e.g., medieval, futuristic, contemporary, magical)",
                "color_palette": "suggested colors (e.g., warm earth tones, cool blues, vibrant rainbow)"
            }}
            """

            data = {
                "model": "llama3",
                "prompt": analysis_prompt,
                "stream": False,
                "options": {"num_predict": 150, "temperature": 0.3},
            }

            response = requests.post(self.ollama_url, json=data, timeout=30)
            if response.status_code == 200:
                result_text = response.json().get("response", "")
                # Extract JSON from response
                json_match = re.search(r"\{.*\}", result_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
        except:
            pass

        # Fallback theme analysis
        return {
            "theme": "fantasy adventure",
            "art_style": "storybook illustration, painterly",
            "mood": "whimsical and enchanting",
            "setting_type": "magical fantasy",
            "color_palette": "warm, vibrant colors",
        }

    def _extract_story_elements(self, story_text: str):
        """Extract and track characters, settings, and key elements from the story"""
        try:
            extraction_prompt = f"""
            Extract the following from this story and return as JSON:
            1. Main characters (names and brief descriptions)
            2. Key locations/settings
            3. Important objects or elements
            
            Story: {story_text[:1500]}
            
            Return ONLY a JSON object like:
            {{
                "characters": {{"name": "description"}},
                "locations": ["location1", "location2"],
                "key_elements": ["element1", "element2"]
            }}
            """

            data = {
                "model": "llama3",
                "prompt": extraction_prompt,
                "stream": False,
                "options": {"num_predict": 200, "temperature": 0.3},
            }

            response = requests.post(self.ollama_url, json=data, timeout=30)
            if response.status_code == 200:
                result_text = response.json().get("response", "")
                json_match = re.search(r"\{.*\}", result_text, re.DOTALL)
                if json_match:
                    elements = json.loads(json_match.group())
                    self.current_story_context["characters"] = elements.get(
                        "characters", {}
                    )
                    self.current_story_context["settings"] = elements.get(
                        "locations", []
                    )
                    self.current_story_context["key_elements"] = elements.get(
                        "key_elements", []
                    )
        except:
            # Fallback: simple extraction
            self._simple_element_extraction(story_text)

    def _simple_element_extraction(self, story_text: str):
        """Fallback method for extracting story elements"""
        # Look for capitalized words that might be names
        words = story_text.split()
        potential_names = [w for w in words if w[0].isupper() and len(w) > 2]
        # Store the most frequent ones as potential characters
        from collections import Counter

        name_counts = Counter(potential_names)
        top_names = [name for name, _ in name_counts.most_common(5)]
        for name in top_names:
            if name not in ["The", "And", "But", "When", "Then"]:
                self.current_story_context["characters"][
                    name
                ] = "character in the story"

    def _split_story_with_context(self, story_text: str) -> list[dict]:
        """
        Splits story into paragraphs while maintaining cumulative context.
        Each paragraph returned as dict:
        { "id": int, "text": str, "context": str }
        """
        if "---PARAGRAPH---" in story_text:
            raw = [p.strip() for p in story_text.split("---PARAGRAPH---")]
        else:
            raw = [p.strip() for p in story_text.split("\n\n")]

        paragraphs = []
        pid = 1
        rolling_context = ""  # keeps story-so-far

        for p in raw:
            if len(p) > 50:
                # update rolling context
                if rolling_context:
                    rolling_context += " " + p
                else:
                    rolling_context = p

                paragraphs.append(
                    {
                        "id": pid,
                        "text": p,
                        "context": rolling_context,  # cumulative so far
                    }
                )
                pid += 1
        return paragraphs

    def _summarize_for_context(self, previous: str, current: str) -> str:
        """Maintain a rolling summary of story context"""
        if not previous:
            return current[:200]
        return (previous + " " + current)[-400:]  # Keep last 400 chars

    def _generate_contextual_image_prompt(
        self,
        paragraph: str,
        paragraph_num: int,
        total_paragraphs: int,
        previous_context: str,
    ) -> str:
        """Generate image prompts that maintain story consistency"""
        try:
            # Get consistent art style from story context
            art_style = self.current_story_context.get(
                "art_style", "storybook illustration"
            )
            mood = self.current_story_context.get("mood", "whimsical")
            color_palette = self.current_story_context.get(
                "color_palette", "vibrant colors"
            )

            # Build character context
            character_desc = ""
            if self.current_story_context.get("characters"):
                chars = list(self.current_story_context["characters"].items())[:2]
                if chars:
                    character_desc = f"Characters: {', '.join([f'{name}: {desc}' for name, desc in chars])}. "

            prompt_request = f"""
            Create a detailed image generation prompt for this story paragraph.
            
            Story Context:
            - Overall theme: {self.current_story_context.get('theme', 'adventure')}
            - Art style MUST be: {art_style}
            - Mood MUST be: {mood}
            - Color palette: {color_palette}
            - {character_desc}
            - Previous events: {previous_context[:200] if previous_context else 'Story beginning'}
            
            Current paragraph (#{paragraph_num} of {total_paragraphs}):
            "{paragraph}"
            
            Requirements for the image prompt:
            1. Focus on the MAIN VISUAL SCENE in this paragraph
            2. Include specific character descriptions if they appear
            3. Describe the setting/environment clearly
            4. Maintain the exact art style: {art_style}
            5. Keep consistent with the story's mood and theme
            6. Be specific about actions, positions, and emotions
            7. Maximum 80 words
            
            Return ONLY the image prompt, nothing else.
            """

            data = {
                "model": "llama3",
                "prompt": prompt_request,
                "stream": False,
                "keep_alive": "30m",
                "options": {
                    "num_predict": 100,
                    "temperature": 0.5,  # Lower temperature for consistency
                },
            }

            response = requests.post(self.ollama_url, json=data, timeout=30)

            if response.status_code == 200:
                prompt = response.json().get("response", "").strip()
                # Ensure style consistency
                if art_style not in prompt.lower():
                    prompt = f"{prompt}, {art_style}"
                if mood not in prompt.lower():
                    prompt = f"{prompt}, {mood} atmosphere"
                return prompt

        except Exception as e:
            print(f"Contextual prompt generation failed: {e}")

        # Fallback with context preservation
        return self._fallback_contextual_prompt(paragraph, paragraph_num)

    def _fallback_contextual_prompt(self, paragraph: str, paragraph_num: int) -> str:
        """Fallback prompt generation with basic context"""
        # Extract key visual elements
        art_style = self.current_story_context.get(
            "art_style", "storybook illustration"
        )

        # Simple keyword extraction
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(
            paragraph, keyphrase_ngram_range=(1, 3), stop_words="english"
        )
        keywords = [kw[0] for kw in keywords[:5]]

        # Build prompt with consistent style
        prompt = f"{', '.join(keywords)}, {art_style}, scene {paragraph_num}"

        # Add any known characters
        if self.current_story_context.get("characters"):
            char_names = list(self.current_story_context["characters"].keys())[:2]
            if char_names:
                prompt = f"{', '.join(char_names)}, {prompt}"

        return prompt

    def _prepare_comfyui_workflow(self, prompt: str) -> Dict:
        """Prepare the SDXL ComfyUI workflow with dynamic style prefix"""
        try:
            with open(
                "comfyui_workflow/sdxl_story_image.json", "r", encoding="utf-8"
            ) as f:
                workflow_dict = json.load(f)

            # Build dynamic prefix based on story context
            style_elements = []
            if self.current_story_context.get("art_style"):
                style_elements.append(self.current_story_context["art_style"])
            if self.current_story_context.get("mood"):
                style_elements.append(f"{self.current_story_context['mood']} mood")
            if self.current_story_context.get("color_palette"):
                style_elements.append(self.current_story_context["color_palette"])

            # Add quality markers
            style_elements.extend(["high quality", "detailed", "consistent style"])

            dynamic_prefix = ", ".join(style_elements) + ", "
            enhanced_prompt = f"{dynamic_prefix}{prompt}"

            # Update nodes as before
            if "6" in workflow_dict:
                workflow_dict["6"]["inputs"]["text"] = enhanced_prompt
            if "7" in workflow_dict:
                workflow_dict["7"]["inputs"]["text"] = self.workflow_config[
                    "negative_prompt"
                ]
            if "15" in workflow_dict:
                workflow_dict["15"]["inputs"]["text"] = enhanced_prompt
            if "16" in workflow_dict:
                workflow_dict["16"]["inputs"]["text"] = self.workflow_config[
                    "negative_prompt"
                ]

            # Rest of the configuration remains the same
            if "5" in workflow_dict:
                workflow_dict["5"]["inputs"]["width"] = self.workflow_config["width"]
                workflow_dict["5"]["inputs"]["height"] = self.workflow_config["height"]

            if "10" in workflow_dict:
                workflow_dict["10"]["inputs"]["steps"] = self.workflow_config[
                    "refiner_steps"
                ]
                workflow_dict["10"]["inputs"]["cfg"] = self.workflow_config["cfg_scale"]
                workflow_dict["10"]["inputs"]["sampler_name"] = self.workflow_config[
                    "sampler"
                ]
                workflow_dict["10"]["inputs"]["end_at_step"] = self.workflow_config[
                    "base_end_step"
                ]
                if self.workflow_config["seed"] == -1:
                    workflow_dict["10"]["inputs"]["noise_seed"] = (
                        int(time.time() * 1000) % 1000000
                    )

            if "11" in workflow_dict:
                workflow_dict["11"]["inputs"]["steps"] = self.workflow_config[
                    "refiner_steps"
                ]
                workflow_dict["11"]["inputs"]["cfg"] = self.workflow_config["cfg_scale"]
                workflow_dict["11"]["inputs"]["sampler_name"] = self.workflow_config[
                    "sampler"
                ]
                workflow_dict["11"]["inputs"]["start_at_step"] = self.workflow_config[
                    "refiner_start_step"
                ]

            return workflow_dict

        except Exception as e:
            print(f"Error preparing workflow: {e}")
            raise

    def generate_image_with_comfyui(
        self, prompt: str, paragraph_id: int, story_context: Optional[Dict] = None
    ) -> Dict:
        """Generate image with optional story context"""
        try:
            # If story context is provided, update the current context
            if story_context:
                self.current_story_context.update(story_context)

            print(f"🎨 Generating image for paragraph {paragraph_id}")
            print(f"📝 Prompt: {prompt}")
            print(
                f"🎨 Style: {self.current_story_context.get('art_style', 'default')} \n"
            )

            start_time = time.time()

            # Prepare workflow
            workflow = self._prepare_comfyui_workflow(prompt)

            # Submit prompt to ComfyUI
            prompt_response = self.comfyui_client.queue_prompt(workflow)

            if not prompt_response:
                return {
                    "status": "error",
                    "message": "Failed to submit prompt to ComfyUI",
                }

            prompt_id = prompt_response.get("prompt_id")
            if not prompt_id:
                return {
                    "status": "error",
                    "message": "No prompt ID received from ComfyUI",
                }

            print(f"Prompt submitted with ID: {prompt_id} \n")

            history_response = self.comfyui_client.wait_for_completion(
                prompt_id, timeout=900
            )
            # if not history_response:
            #     return {
            #         "status": "error",
            #         "message": "Image generation timed out or failed"
            #     }
            if not history_response:
                # Add a retry loop to fetch the history
                # This is the key fix for the "prompt not found" issue
                retries = 5
                history = None
                while retries > 0:
                    history = self.comfyui_client.get_history(prompt_id)
                    if history and prompt_id in history:
                        history_response = history[prompt_id]
                        break  # Found it, exit the loop
                    print(
                        f"Prompt history not found. Retrying in 1 second... ({retries} attempts left)\n"
                    )
                    time.sleep(
                        1
                    )  # This sleep is in the main thread, not the WebSocket thread
                    retries -= 1

                if not history or prompt_id not in history:
                    return {
                        "status": "error",
                        "message": "Prompt not found in history after multiple attempts",
                    }

            # Extract image information
            outputs = history_response.get("outputs", {})
            image_info = None
            for node_id, node_output in outputs.items():
                if "images" in node_output:
                    image_info = node_output["images"][0]
                    break

            if not image_info:
                return {"status": "error", "message": "No image output found"}

            # Download the image
            image_data = self.comfyui_client.get_image(
                image_info["filename"],
                image_info.get("subfolder", ""),
                image_info.get("type", "output"),
            )

            if not image_data:
                return {
                    "status": "error",
                    "message": "Failed to download generated image",
                }

            # Save image locally
            image_url = self._save_image_from_comfyui(image_data, paragraph_id)

            if not image_url:
                return {"status": "error", "message": "Failed to save image locally"}

            generation_time = time.time() - start_time
            print(f"✅ Image generated in {generation_time:.2f}s \n")

            return {
                "status": "success",
                "image_url": image_url,
                "generation_time": generation_time,
                "message": "Image generated successfully",
            }

        except Exception as e:
            print(f"Image generation error: {e}")
            return {
                "status": "error",
                "message": f"Image generation failed: {str(e)}",
                "traceback": traceback.format_exc(),
            }

    def _save_image_from_comfyui(self, image_data: bytes, paragraph_id: int) -> str:
        """Save image data to local storage"""
        try:
            filename = f"paragraph_{paragraph_id}_{uuid.uuid4().hex}.png"
            filename = secure_filename(filename)

            destination = os.path.join(UPLOAD_FOLDER, filename)
            with open(destination, "wb") as f:
                f.write(image_data)

            return f"{BACKEND_URL}/generated_images/{filename}"
        except Exception as e:
            print(f"Error saving image: {e}")
            return None

    # Keep other utility methods from original code
    def _generate_title(self, user_prompt: str) -> str:
        """Generate a title for the story"""
        words = user_prompt.split()
        if len(words) <= 4:
            return f"The Tale of {user_prompt.title()}"
        else:
            return f"The Adventures of {' '.join(words[:3]).title()}"

        # --------- Keyword extraction methods ---------

    # ---------- KEYWORD EXTRACTORS (robust, never empty) ----------
    def _kw_simple(self, paragraph: str, k: int = 8) -> list[str]:
        words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", paragraph.lower())
        # crude de-dup and order by length desc as proxy for descriptiveness
        seen = set()
        uniq = []
        for w in words:
            if w not in seen:
                seen.add(w)
                uniq.append(w)
        cand = sorted(uniq, key=len, reverse=True)[:k]
        return (
            cand if cand else ["scene", "illustration", "character", "setting", "mood"]
        )

    def _kw_yake(self, paragraph: str, k: int = 8) -> list[str]:
        try:

            kw_extractor = yake.KeywordExtractor(lan="en", n=1, top=k)
            kws = [kw for kw, score in kw_extractor.extract_keywords(paragraph)]
            return kws or self._kw_simple(paragraph, k)
        except Exception:
            return self._kw_simple(paragraph, k)

    def _kw_keybert(self, paragraph: str, k: int = 8) -> list[str]:
        try:

            if not hasattr(self, "_kb"):
                # lazy init once
                self._kb = KeyBERT()
            pairs = self._kb.extract_keywords(
                paragraph, keyphrase_ngram_range=(1, 1), stop_words="english", top_n=k
            )
            kws = [p[0] for p in pairs]
            return kws or self._kw_simple(paragraph, k)
        except Exception:
            return self._kw_simple(paragraph, k)

    def _extract_keywords_keybert(self, text: str, top_k: int = 8):
        try:
            kw_model = KeyBERT()
            kws = kw_model.extract_keywords(
                text, keyphrase_ngram_range=(1, 3), stop_words="english", top_n=top_k
            )
            return [k for k, _ in kws]
        except Exception as e:
            print(f"KeyBERT error: {e}")
            return []

    def _extract_keywords_yake(self, text: str, top_k: int = 8):
        if yake is None:
            print("YAKE not available; install with `pip install yake`")
            return []
        try:
            extractor = yake.KeywordExtractor(lan="en", n=3, top=top_k, dedupLim=0.9)
            kws = extractor.extract_keywords(text)
            # YAKE returns [(kw, score)], lower score = better
            kws = [kw for kw, score in sorted(kws, key=lambda x: x[1])]
            return kws[:top_k]
        except Exception as e:
            print(f"YAKE error: {e}")
            return []

    def _extract_keywords_simple(self, text: str, top_k: int = 8):
        # Very lightweight frequency-based baseline over non-stopword tokens
        tokens = re.findall(r"[A-Za-z][A-Za-z\-']+", text.lower())
        freq = {}
        for t in tokens:
            if t in STOPWORDS or len(t) <= 2:
                continue
            freq[t] = freq.get(t, 0) + 1
        # return single words; for a bit of variety, let’s try to stitch some bigrams if adjacent words are frequent
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        single_words = [w for w, c in sorted_words[: top_k * 2]]
        # make naive bigrams from the paragraph to capture some phrases
        words_seq = [
            w
            for w in re.findall(r"[A-Za-z][A-Za-z\-']+", text.lower())
            if w not in STOPWORDS
        ]
        bigrams = {}
        for i in range(len(words_seq) - 1):
            bg = f"{words_seq[i]} {words_seq[i+1]}"
            if all(len(x) > 2 for x in words_seq[i : i + 2]):
                bigrams[bg] = bigrams.get(bg, 0) + 1
        top_bigrams = [
            bg
            for bg, c in sorted(bigrams.items(), key=lambda x: x[1], reverse=True)[
                : top_k // 2
            ]
        ]
        candidates = single_words[:top_k] + top_bigrams
        # unique while preserving order
        seen, uniq = set(), []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        return uniq[:top_k]

    def _extract_keywords(self, text, method="keybert"):
        if method == "yake":
            return self._extract_with_yake(text)
        elif method == "keybert":
            return self._extract_with_keybert(text)
        elif method == "simple":
            return self._extract_simple(text)
        else:
            return text  # fallback

    def _build_prompt_from_keywords(self, keywords, paragraph_num: int) -> str:
        art_style = self.current_story_context.get(
            "art_style", "storybook illustration"
        )
        mood = self.current_story_context.get("mood", None)
        base = ", ".join(keywords)
        prompt = f"{base}, {art_style}, scene {paragraph_num}"
        if mood:
            prompt += f", {mood} atmosphere"
        # Add known characters (first two)
        if self.current_story_context.get("characters"):
            char_names = list(self.current_story_context["characters"].keys())[:2]
            if char_names:
                prompt = f"{', '.join(char_names)}, {prompt}"
        return prompt

        # --------- CLIP scoring helpers ---------

    def _clip_encode_text(self, texts):
        if _CLIP_BACKEND is None or self.clip_model is None:
            return None

        # Ensure input is list
        if isinstance(texts, str):
            texts = [texts]

        with torch.no_grad():
            if _CLIP_BACKEND == "open_clip":
                toks = self.clip_tokenizer(texts).to(self.clip_device)
                txt = self.clip_model.encode_text(toks)
            else:
                toks = clip_pkg.tokenize(texts, context_length=self.clip_ctx_len).to(
                    self.clip_device
                )
                txt = self.clip_model.encode_text(toks)

            txt = txt / txt.norm(dim=-1, keepdim=True)
            return txt

    def _clip_encode_image(self, pil_image: Image.Image):
        if _CLIP_BACKEND is None or self.clip_model is None:
            return None

        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        with torch.no_grad():
            img = self.clip_preprocess(pil_image).unsqueeze(0).to(self.clip_device)
            img = self.clip_model.encode_image(img)
            img = img / img.norm(dim=-1, keepdim=True)
            return img

    # def _clip_cosine(self, a, b):
    #     if a is None or b is None:
    #         return None
    #     return float((a @ b.T).squeeze().item())
    def _clip_cosine(self, a, b):
        if a is None or b is None:
            return None

        # Debug: Check shapes
        if a.dim() != 2 or b.dim() != 2:
            print(f"CLIP cosine: expected 2D tensors, got a={a.shape}, b={b.shape}")
            return None

        if a.size(0) != 1 or b.size(0) != 1:
            print(f"CLIP cosine: expected batch size 1, got a={a.shape}, b={b.shape}")
            return None

        with torch.no_grad():
            similarity = (
                a @ b.T
            ).squeeze()  # Should be scalar or single-element tensor
            if similarity.numel() != 1:
                print(
                    f"CLIP cosine: too many elements after matmul: {similarity.shape}"
                )
                return None
            return float(similarity.item())

    def _compute_clip_scores(
        self, image_bytes: bytes, paragraph_text: str, prompt_text: str
    ):
        if self.clip_model is None:
            return {"clip_image_paragraph": None, "clip_image_prompt": None}

        try:
            img = Image.open(BytesIO(image_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")

            img_emb = self._clip_encode_image(img)
            txt_par = self._clip_encode_text(paragraph_text)  # Now supports str
            txt_prm = self._clip_encode_text(prompt_text)

            return {
                "clip_image_paragraph": self._clip_cosine(img_emb, txt_par),
                "clip_image_prompt": self._clip_cosine(img_emb, txt_prm),
            }
        except (OSError, UnidentifiedImageError) as e:
            print(f"Image decoding error in CLIP: {e}")
            return {"clip_image_paragraph": None, "clip_image_prompt": None}
        except Exception as e:
            print(f"Unexpected CLIP scoring error: {type(e).__name__}: {e}")
            return {"clip_image_paragraph": None, "clip_image_prompt": None}

    def compare_extractors_on_paragraph(
        self, paragraph_text: str, paragraph_id: int, top_k: int = 8
    ):
        methods = ["yake", "keybert", "simple"]
        results = []

        for method in methods:
            try:
                # 1) Keywords & prompt
                keywords = self._extract_keywords(paragraph_text, method)
                prompt = self._build_prompt_from_keywords(keywords, paragraph_id)

                # 2) Generate with ComfyUI (we also need the image bytes for CLIP)
                start = time.time()
                workflow = self._prepare_comfyui_workflow(prompt)
                prompt_response = self.comfyui_client.queue_prompt(workflow)
                if not prompt_response or not prompt_response.get("prompt_id"):
                    results.append(
                        {
                            "method": method,
                            "status": "error",
                            "message": "Failed to submit prompt to ComfyUI",
                            "keywords": keywords,
                            "prompt": prompt,
                        }
                    )
                    continue

                pid = prompt_response["prompt_id"]
                history = self.comfyui_client.wait_for_completion(pid, timeout=900)
                if not history:
                    results.append(
                        {
                            "method": method,
                            "status": "error",
                            "message": "ComfyUI timeout/failure",
                            "keywords": keywords,
                            "prompt": prompt,
                        }
                    )
                    continue

                # Find the image node output
                outputs = history.get("outputs", {})
                image_info = None
                for node_id, node_output in outputs.items():
                    if "images" in node_output:
                        image_info = node_output["images"][0]
                        break
                if not image_info:
                    results.append(
                        {
                            "method": method,
                            "status": "error",
                            "message": "No image output found",
                            "keywords": keywords,
                            "prompt": prompt,
                        }
                    )
                    continue

                # Download image bytes
                image_bytes = self.comfyui_client.get_image(
                    image_info["filename"],
                    image_info.get("subfolder", ""),
                    image_info.get("type", "output"),
                )
                if not image_bytes:
                    results.append(
                        {
                            "method": method,
                            "status": "error",
                            "message": "Failed to download generated image",
                            "keywords": keywords,
                            "prompt": prompt,
                        }
                    )
                    continue

                # Save image locally (to keep your existing behavior)
                image_url = self._save_image_from_comfyui(image_bytes, paragraph_id)
                gen_time = time.time() - start

                # 3) Metrics
                clip_scores = self._compute_clip_scores(
                    image_bytes=image_bytes,
                    paragraph_text=paragraph_text,
                    prompt_text=prompt,
                )

                results.append(
                    {
                        "method": method,
                        "status": "success",
                        "keywords": keywords,
                        "prompt": prompt,
                        "image_url": image_url,
                        "generation_time": gen_time,
                        **clip_scores,
                    }
                )

            except Exception as e:
                results.append(
                    {
                        "method": method,
                        "status": "error",
                        "message": f"Exception: {str(e)}",
                    }
                )

        return {"paragraph_id": paragraph_id, "results": results}

    def compare_extractors_for_story(self, paragraphs, top_k=8, story_context=None):
        """
        paragraphs: list[str]
        Returns:
          {
            run_id, created_at, per_paragraph: [...], aggregates: {...},
            files: {csv_path, json_path, manifest_path}
          }
        """
        if story_context:
            self.current_story_context.update(story_context)

        run_id = f"run_{uuid.uuid4().hex[:8]}"
        run_dir = EXPERIMENTS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        per_paragraph = []
        csv_rows = []
        csv_header = [
            "run_id",
            "paragraph_id",
            "method",
            "generation_time",
            "clip_image_paragraph",
            "clip_image_prompt",
            "image_url",
            "keywords",
            "prompt",
        ]

        # Do paragraph-by-paragraph comparisons
        for idx, ptext in enumerate(paragraphs, start=1):
            comp = self.compare_extractors_on_paragraph(
                paragraph_text=ptext, paragraph_id=idx, top_k=top_k
            )
            per_paragraph.append(
                {
                    "paragraph_id": idx,
                    "paragraph_text": ptext,
                    "results": comp["results"],
                }
            )

            # rows for CSV
            for r in comp["results"]:
                csv_rows.append(
                    [
                        run_id,
                        idx,
                        r.get("method"),
                        r.get("generation_time"),
                        _safe_float(r.get("clip_image_paragraph")),
                        _safe_float(r.get("clip_image_prompt")),
                        r.get("image_url"),
                        "; ".join(r.get("keywords") or []),
                        r.get("prompt"),
                    ]
                )

        # Compute aggregates per method
        methods = ["yake", "keybert", "simple"]
        aggregates = {}
        for m in methods:
            times, c_ip, c_ipp = [], [], []
            for pp in per_paragraph:
                for r in pp["results"]:
                    if r.get("method") == m and r.get("status") == "success":
                        if r.get("generation_time") is not None:
                            times.append(float(r["generation_time"]))
                        if r.get("clip_image_paragraph") is not None:
                            c_ip.append(float(r["clip_image_paragraph"]))
                        if r.get("clip_image_prompt") is not None:
                            c_ipp.append(float(r["clip_image_prompt"]))

            def _avg(lst):
                return sum(lst) / len(lst) if lst else None

            aggregates[m] = {
                "n_success": len(
                    [
                        1
                        for pp in per_paragraph
                        for r in pp["results"]
                        if r.get("method") == m and r.get("status") == "success"
                    ]
                ),
                "avg_generation_time": _avg(times),
                "avg_clip_image_paragraph": _avg(c_ip),
                "avg_clip_image_prompt": _avg(c_ipp),
            }

        # Save artifacts
        manifest = {
            "run_id": run_id,
            "created_at": _now_iso(),
            "top_k": top_k,
            "story_context": self.current_story_context,
            "num_paragraphs": len(paragraphs),
        }
        _write_json(run_dir / "manifest.json", manifest)
        _write_json(
            run_dir / "results.json",
            {"per_paragraph": per_paragraph, "aggregates": aggregates},
        )

        csv_path = run_dir / "results.csv"
        _append_csv(csv_path, csv_rows, csv_header)

        return {
            "run_id": run_id,
            "created_at": manifest["created_at"],
            "per_paragraph": per_paragraph,
            "aggregates": aggregates,
            "files": {
                "csv_path": str(csv_path),
                "json_path": str(run_dir / "results.json"),
                "manifest_path": str(run_dir / "manifest.json"),
            },
        }


# Initialize the story generator
story_gen = ImprovedStoryGenerator()
# Initialize once
sentiment_analyzer = SentimentIntensityAnalyzer()

EXPERIMENTS_DIR = Path("experiments")
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


def _now_iso():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _append_csv(path: Path, rows: list, header: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        for r in rows:
            w.writerow(r)


def save_experiment(story, results):

    print("Saving experiments data...\n")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"exp_{timestamp}"

    # Ensure experiments dir exists
    EXPERIMENTS_DIR = os.path.join(os.getcwd(), "experiments")
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    # Build JSON structure
    exp_data = {
        "id": exp_id,
        "title": story["title"],
        "context": story.get("context", {}),
        "consistency": story.get("consistency", ""),
        "paragraphs": [],
    }

    for p in story["paragraphs"]:
        entry = {"id": p["id"], "text": p["text"], "results": {}}
        para_results = results.get(p["id"], {}) or {}

        for method, res in para_results.items():
            safe_res = res or {}
            entry["results"][method] = {
                "prompt": safe_res.get("prompt", ""),
                "image_url": safe_res.get("image_url", ""),
                "clip_score": safe_res.get("clip_score", None),
                "bias": safe_res.get("bias", ""),
                "bert_score": safe_res.get("bert_score", ""),
                "rouge_score": safe_res.get("rouge_score", ""),
                "sentiment": safe_res.get("sentiment", {})   # <-- NEW
            }
        exp_data["paragraphs"].append(entry)

    # Save JSON
    with open(
        os.path.join(EXPERIMENTS_DIR, f"{exp_id}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(exp_data, f, indent=2, ensure_ascii=False)

    # Append to CSV
    csv_file = os.path.join(EXPERIMENTS_DIR, "summary.csv")
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writerow(
                [
                    "experiment_id",
                    "story_title",
                    "paragraph_id",
                    "method",
                    "prompt",
                    "image_url",
                    "clip_score",
                    "bias",
                    "bert_score",
                    "rouge_score",
                    "sentiment" 
                ]
            )
        for p in exp_data["paragraphs"]:
            for method, res in p["results"].items():
                writer.writerow(
                    [
                        exp_id,
                        exp_data["title"],
                        p["id"],
                        method,
                        res["prompt"],
                        res["image_url"],
                        res["clip_score"],
                        res["bias"],
                        str(res["bert_score"]),
                        res["rouge_score"],
                        str(res["sentiment"])   # <-- NEW
                    ]
                )
    print(f"Experiment saved with id: {exp_id}\n")
    return exp_id


def update_experiment(
    exp_id, paragraph_id, method, prompt, image_url, clip_score, brisque_score
):

    print(f"clip score going to update {str(clip_score)}")
    print(f"image url going to update {image_url}")

    exp_path = os.path.join(EXPERIMENTS_DIR, f"{exp_id}.json")

    if not os.path.isfile(exp_path):
        print(f"Experiment {exp_id} not found, skipping update")
        return

    # Load existing JSON
    with open(exp_path, "r", encoding="utf-8") as f:
        exp_data = json.load(f)

    # Find and update the paragraph
    for p in exp_data["paragraphs"]:
        if p["id"] == paragraph_id:
            # Initialize results if it doesn't exist
            if "results" not in p:
                p["results"] = {}

            # Initialize method entry if it doesn't exist
            if method not in p["results"]:
                p["results"][method] = {
                    "prompt": None,
                    "image_url": None,
                    "clip_score": None,
                    "brisque_score": None,
                }

            # Update only the provided fields
            if prompt is not None:
                p["results"][method]["prompt"] = prompt
            if image_url is not None:
                p["results"][method]["image_url"] = image_url
            if clip_score is not None:
                p["results"][method]["clip_score"] = clip_score
            if brisque_model is not None:
                p["results"][method]["brisque_score"] = brisque_score

    # Save the updated JSON
    with open(exp_path, "w", encoding="utf-8") as f:
        json.dump(exp_data, f, indent=2, ensure_ascii=False)

    # Also update CSV row
    csv_file = os.path.join(EXPERIMENTS_DIR, "summary.csv")

    # Check if file exists to determine if we need headers
    file_exists = os.path.isfile(csv_file)

    try:
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)

            # Write headers if file is new
            if not file_exists or os.stat(csv_file).st_size == 0:
                writer.writerow(
                    [
                        "exp_id",
                        "title",
                        "paragraph_id",
                        "method",
                        "prompt",
                        "image_url",
                        "clip_score",
                    ]
                )

            writer.writerow(
                [
                    exp_id,
                    exp_data["title"],
                    paragraph_id,
                    method,
                    prompt,
                    image_url,
                    clip_score,
                ]
            )
    except PermissionError:
        print(f"ERROR: Could not write to {csv_file}. Please check file permissions.")


def get_image_bytes_from_url(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an error for bad responses
        return response.content
    except Exception as e:
        print(f"Error fetching image from URL: {e}")
        return None


# ===================== NEW METRIC HELPERS ===================== #
def analyze_sentiment(text: str) -> dict:
    """
    Perform sentiment analysis on text using NLTK's VADER.
    Returns compound, pos, neu, neg scores.
    """
    if not text or not isinstance(text, str):
        return {"compound": 0.0, "pos": 0.0, "neu": 0.0, "neg": 0.0}
    
    scores = sentiment_analyzer.polarity_scores(text)
    return scores


def check_character_consistency(paragraphs: list[dict]):
    seen_chars = set()
    consistency_scores = []
    for p in paragraphs:
        print(f"\n\nchecking consistency {p['text']}")
        # Fixed regex pattern - removed extra backslashes
        chars = set(re.findall(r"\b[A-Z][a-z]{2,}\b", p["text"]))
        print(f"Found characters: {chars}")  # Debug print
        if seen_chars:
            overlap = len(seen_chars.intersection(chars)) / max(1, len(seen_chars))
            consistency_scores.append(overlap)
            print(f"Overlap: {overlap}")  # Debug print
        seen_chars.update(chars)
    return (
        sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0
    )


# 4. Bias (simple lexicon-based check)
BIAS_KEYWORDS = {# Gender
    "man", "woman", "boy", "girl", "male", "female", "husband", "wife",
    "father", "mother", "son", "daughter", "brother", "sister",

    # Race / Ethnicity (⚠️ use carefully, just for detection)
    "black", "white", "asian", "african", "european", "indian", "latino",
    "arab", "oriental", "native", "tribal",

    # Religion
    "muslim", "christian", "jew", "hindu", "buddhist", "islamic", "catholic",
    "orthodox", "sikh",

    # Social / Class
    "king", "queen", "prince", "princess", "slave", "servant", "master",
    "lord", "lady", "peasant", "noble", "warrior", "soldier", "knight",

    # Jobs / Roles
    "nurse", "doctor", "teacher", "engineer", "scientist", "maid", "cook",
    "cleaner", "farmer", "soldier", "politician", "leader",

    # Descriptors / Potential stereotypes
    "weak", "strong", "violent", "aggressive", "obedient", "exotic",
    "barbaric", "civilized", "primitive"}


def check_bias_in_keywords(keywords: list[str]):
    return any(k.lower() in BIAS_KEYWORDS for k in keywords)



def preprocess_for_bertScore(text):
    """
    Convert any input to a proper string for BERTScore.
    Handles strings, lists, and nested lists.
    """
    if isinstance(text, str):
        return text

    # If it's a list of strings, join them
    if isinstance(text, list) and all(isinstance(item, str) for item in text):
        return " ".join(
            text
        )  # Using space instead of comma for better BERT understanding

    # If it's a list of lists (nested), flatten and join
    if isinstance(text, list) and any(isinstance(item, list) for item in text):
        flattened = []
        for item in text:
            if isinstance(item, list):
                flattened.extend([str(i) for i in item if isinstance(i, str)])
            else:
                flattened.append(str(item))
        return " ".join(flattened)

    # Fallback: convert to string
    return str(text)


def calculate_bertscore(
    candidates, references, model_type="microsoft/deberta-large-mnli"
):
    """
    Safe BERTScore calculation with input preprocessing.
    """
    # Preprocess all inputs to ensure they are strings
    processed_candidates = [
        preprocess_for_bertScore(candidate) for candidate in candidates
    ]
    processed_references = [
        preprocess_for_bertScore(reference) for reference in references
    ]

    print(f"Sample candidate: {processed_candidates[0][:100]}...")
    print(f"Sample reference: {processed_references[0][:100]}...")

    with torch.no_grad():
        try:
            P, R, F1 = score(
                processed_candidates,
                processed_references,
                model_type=model_type,
                lang="en",
                verbose=False,  # Set to True for more details
                batch_size=4,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            return {"precision": P.tolist(), "recall": R.tolist(), "f1": F1.tolist()}
        except Exception as e:
            print(f"BERTScore Error: {e}")
            # Return dummy scores for error handling
            dummy = [0.0] * len(processed_candidates)
            return {"precision": dummy, "recall": dummy, "f1": dummy}


def preprocess_for_rouge(text):
    """
    Convert any input to a proper string for ROUGE scoring.
    More robust version specifically for ROUGE.
    """
    if isinstance(text, str):
        # Clean the string: remove extra spaces, special characters
        text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with single space
        text = text.strip()
        return text

    # If it's a list of strings, join them with spaces
    if isinstance(text, list):
        # Handle both flat lists and nested lists
        words = []
        for item in text:
            if isinstance(item, str):
                words.append(item)
            elif isinstance(item, list):
                words.extend([str(i) for i in item if isinstance(i, str)])
            else:
                words.append(str(item))
        return " ".join(words)

    # Fallback for any other type
    return str(text)


def calculate_rouge_for_paragraph(keywords, paragraph):
    """
    Calculate ROUGE scores for keywords extracted from a single paragraph.

    Args:
        keywords: Extracted keywords (string or list)
        paragraph: The original paragraph text

    Returns:
        dict: ROUGE scores for this single paragraph
    """
    # Initialize scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    # Convert keywords to string format
    if isinstance(keywords, list):
        keyword_text = " ".join(str(kw) for kw in keywords)
    else:
        keyword_text = str(keywords)

    # Clean up whitespace
    keyword_text = " ".join(keyword_text.split())
    paragraph_clean = " ".join(paragraph.split())

    print(f"Keywords: '{keyword_text}'")
    print(f"Paragraph: '{paragraph_clean[:100]}...'")
    print(
        f"Keyword length: {len(keyword_text)}, Paragraph length: {len(paragraph_clean)}"
    )

    # Calculate ROUGE scores
    scores = scorer.score(paragraph_clean, keyword_text)

    return {
        "rouge1": {
            "precision": scores["rouge1"].precision,
            "recall": scores["rouge1"].recall,
            "f1": scores["rouge1"].fmeasure,
        },
        "rouge2": {
            "precision": scores["rouge2"].precision,
            "recall": scores["rouge2"].recall,
            "f1": scores["rouge2"].fmeasure,
        },
        "rougeL": {
            "precision": scores["rougeL"].precision,
            "recall": scores["rougeL"].recall,
            "f1": scores["rougeL"].fmeasure,
        },
    }



# Load BRISQUE model once
brisque_model = pyiqa.create_metric('brisque', device='cuda')

def compute_brisque(image_path: str) -> float:
    """
    Compute BRISQUE score for a single image.
    Lower is better (high quality).
    """
    try:
        score = brisque_model(image_path).item()
        return score
    except Exception as e:
        print(f"BRISQUE failed for {image_path}: {e}")
        return None

# ===================== INTEGRATION INTO EXPERIMENTS ===================== #




@app.route("/api/warmup", methods=["POST"])
def warmup_server():
    """Warm up the Ollama model"""
    result = story_gen.warm_up_model()
    return jsonify(result)


@app.route("/api/generate-story", methods=["POST"])
def generate_story():
    """
    Body:
      {
        "prompt": str,
        "paragraphs": int,
        "mode": "single"|"compare",
        "extractor": "yake"|"keybert"|"simple"   # used only when mode="single"
      }
    Returns (always):
      {
        "status": "success",
        "story": {
          "title": str,
          "paragraphs": [
            # mode=single:
            {"id": int, "text": str, "image_prompt": str}
            # mode=compare:
            {"id": int, "text": str, "images": [
                {"method":"yake", "image_prompt": str, "image_url": null, "image_error": false},
                {"method":"keybert", "image_prompt": str, "image_url": null, "image_error": false},
                {"method":"simple", "image_prompt": str, "image_url": null, "image_error": false}
            ]}
          ],
          "total_paragraphs": int,
          "context": str
        }
      }
    """
    try:
        data = request.json or {}
        user_prompt = data.get("prompt", "").strip()
        num_paragraphs = int(data.get("paragraphs", 5))
        mode = (data.get("mode") or "single").lower()
        extractor = (data.get("extractor") or "simple").lower()

        if not user_prompt:
            return jsonify({"status": "error", "message": "Prompt is required"}), 400

        # Warm-up if needed
        if not story_gen.model_warmed:
            warmup_result = story_gen.warm_up_model()
            if warmup_result["status"] != "success":
                return jsonify(warmup_result), 500

        # 1) Ask LLM for story text
        result = story_gen.generate_story(user_prompt, num_paragraphs)

        if result.get("status") != "success":
            return jsonify(result), 500

        story_obj = result["story"]

        paragraphs = story_obj.get("paragraphs", [])

        print(f"\nStory generated with {len(story_obj)} paragraphs \n")

        # 'paragraphs' are objects: {"id","text"}

        # 2) Attach prompts depending on mode
        enriched = []
        for p in paragraphs:
            text = p["text"]
            paragraph_id = p["id"]

            if mode == "compare":
                # build 3 prompts deterministically
                methods = [
                    ("yake", story_gen._kw_yake(text)),
                    ("keybert", story_gen._kw_keybert(text)),
                    ("simple", story_gen._kw_simple(text)),
                ]
                images = []
                for m, kws in methods:
                    iprompt = story_gen._build_prompt_from_keywords(kws, paragraph_id)
                    images.append(
                        {
                            "method": m,
                            "image_prompt": iprompt,
                            "image_url": None,
                            "image_error": False,
                        }
                    )
                enriched.append({"id": p["id"], "text": text, "images": images})
            else:
                # single mode → honor requested extractor
                if extractor == "yake":
                    kws = story_gen._kw_yake(text)
                elif extractor == "keybert":
                    kws = story_gen._kw_keybert(text)
                else:
                    kws = story_gen._kw_simple(text)
                iprompt = story_gen._build_prompt_from_keywords(kws, paragraph_id)
                enriched.append({"id": p["id"], "text": text, "image_prompt": iprompt})

        # 3) Return normalized story
        final_story = {
            "title": story_obj["title"],
            "paragraphs": enriched,
            "total_paragraphs": len(enriched),
            "context": story_obj.get("context", ""),  # keep your context field
        }

        return jsonify({"status": "success", "story": final_story})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"}), 500


@app.route("/api/generate-image", methods=["POST"])
def generate_image():
    data = request.json
    paragraph_id = data.get("paragraph_id")
    prompt = data.get("prompt")
    story_context = data.get("story_context")
    method = data.get("method", "single")  # NEW: allow method tag

    try:
        response = story_gen.generate_image_with_comfyui(
            prompt, paragraph_id, story_context
        )
        if response.get("status") != "success":
            return jsonify(response), 500

        image_url = response.get("image_url")

        # 2) If image was generated, compute CLIP scores
        clip_scores = {"clip_image_paragraph": None, "clip_image_prompt": None}
        brisque_score=""
        if image_url:
            try:
                # Get the actual image bytes (you'll need to implement this part)
                # This depends on how your images are stored/accessed
                image_bytes = get_image_bytes_from_url(
                    image_url
                )  # You need to implement this

                if image_bytes:
                    print("computing clip score")
                    # Compute both CLIP scores (image vs paragraph and image vs prompt)
                    clip_scores = story_gen._compute_clip_scores(
                        image_bytes=image_bytes,
                        paragraph_text=story_context,  # or whatever text represents the paragraph
                        prompt_text=prompt,
                    )
                    print(f"clip score : {str(clip_scores)}")

            except Exception as e:
                print(f"Error computing CLIP scores: {e}")

            try:
                parts = image_url.split('/')

                # Get the last item in the list, which is the file name
                file_name = parts[-1]
                brisque_score = compute_brisque( os.path.join(UPLOAD_FOLDER, file_name))
            except Exception as e:
                print(f"Error computing brisque scores: {e}")

        # Add CLIP scores to response
        response.update(clip_scores)

        # For backward compatibility, also include the main clip_score (prompt)
        response["clip_score"] = clip_scores["clip_image_prompt"]

        # 3) Update experiment storage if experiment exists
        exp_id = data.get("experiment_id")  # frontend should pass exp_id if available
        if exp_id:
            update_experiment(
                exp_id,
                paragraph_id,
                method,
                prompt,
                image_url,
                clip_scores[
                    "clip_image_prompt"
                ],  # Using prompt score for backward compatibility
                brisque_score,
            )

        print(f"\nGenerate image with Comfy UI response \n {response} \n")
        return jsonify(response)

    except Exception as e:
        print(f"\nGenerate image with Comfy UI exception response \n {e} \n")
        return jsonify({"status": "error", "message": str(e)})


@app.route("/api/compare-keyword-methods", methods=["POST"])
def compare_keyword_methods():
    """
    Body:
    {
      "paragraph_text": "...",
      "paragraph_id": 1,
      "top_k": 8,
      "story_context": {...}  # optional, to enforce style/mood consistency
    }
    """
    try:
        data = request.json or {}
        paragraph_text = data.get("paragraph_text", "").strip()
        paragraph_id = int(data.get("paragraph_id", 1))
        top_k = int(data.get("top_k", 8))
        story_context = data.get("story_context")

        if not paragraph_text:
            return (
                jsonify({"status": "error", "message": "paragraph_text is required"}),
                400,
            )

        if story_context:
            story_gen.current_story_context.update(story_context)

        comp = story_gen.compare_extractors_on_paragraph(
            paragraph_text=paragraph_text, paragraph_id=paragraph_id, top_k=top_k
        )
        return jsonify({"status": "success", "comparison": comp})

    except Exception as e:
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"}), 500


@app.route("/api/compare-story", methods=["POST"])
def compare_story():
    try:
        data = request.json or {}
        user_prompt = data.get("prompt", "").strip()
        num_paragraphs = int(data.get("paragraphs", 5))

        if not user_prompt:
            return jsonify({"status": "error", "message": "Prompt is required"}), 400

        # Warmup if needed
        if not story_gen.model_warmed:
            warmup_result = story_gen.warm_up_model()
            if warmup_result["status"] != "success":
                return jsonify(warmup_result), 500

        # Generate story text first
        result = story_gen.generate_story(user_prompt, num_paragraphs)

        if result.get("status") != "success":
            return jsonify(result), 500

        story_obj = result["story"]
        paragraphs = story_obj.get("paragraphs", [])

        print("hello we are here")
        consistency = check_character_consistency(paragraphs)  # check for consistency a

        enriched = []

        for p in paragraphs:
            text = p["text"]
            pid = p["id"]

            methods = [
                ("yake", story_gen._kw_yake(text)),
                ("keybert", story_gen._kw_keybert(text)),
                ("simple", story_gen._kw_simple(text)),
            ]
            images = []
            for m, kws in methods:
                iprompt = story_gen._build_prompt_from_keywords(kws, pid)

                bias_flag = check_bias_in_keywords(kws)

                bert_scores = calculate_bertscore([kws], [text])
                print(f"bert score {bert_scores}")

                rouge_scores = calculate_rouge_for_paragraph(kws, text)
                print(f"rouge score {rouge_scores}")

                sentiment_scores = analyze_sentiment(text)
                print(f"sentiment score {sentiment_scores}")

                images.append(
                    {
                        "method": m,
                        "image_prompt": iprompt,
                        "image_url": None,
                        "image_error": False,
                        "bias": bias_flag,
                        "bert_score": bert_scores,
                        "rouge_score": rouge_scores,
                        "sentiment": sentiment_scores   # <-- NEW
                    }
                )
            enriched.append({"id": pid, "text": text, "images": images})

        # Normalized output (same as generate-story with mode=compare)
        final_story = {
            "title": story_obj["title"],
            "paragraphs": enriched,
            "total_paragraphs": len(enriched),
            "context": story_obj.get("context", ""),
            "consistency": consistency,
        }

        # after building final_story
        save_results = {}
        for p in final_story["paragraphs"]:
            para_res = {}

            for img in p.get("images", []):

                para_res[img["method"]] = {
                    "prompt": img.get("image_prompt", ""),
                    "image_url": img.get("image_url", ""),
                    "bias": img.get("bias", ""),
                    "clip_score": img.get("clip_score", None),
                    "bert_score": img.get("bert_score", ""),
                    "rouge_score": img.get("rouge_score", ""),
                    "sentiment":  img.get("sentiment","")   # <-- NEW
                }
            save_results[p["id"]] = para_res

        # Save experiment for later retrieval
        exp_id = save_experiment(
            final_story, save_results
        )  # pass empty results for now
        return jsonify(
            {"status": "success", "experiment_id": exp_id, "story": final_story}
        )

    except Exception as e:
        print(f"exception occur {str(e)}")
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"}), 500


@app.route("/api/experiments", methods=["GET"])
def list_experiments():
    exps = []
    for fname in sorted(os.listdir(EXPERIMENTS_DIR)):
        if fname.endswith(".json"):
            with open(os.path.join(EXPERIMENTS_DIR, fname), encoding="utf-8") as f:
                data = json.load(f)
                exps.append({"id": data["id"], "title": data["title"]})
    return jsonify({"experiments": exps})


@app.route("/api/experiments/<exp_id>", methods=["GET"])
def get_experiment(exp_id):
    path = os.path.join(EXPERIMENTS_DIR, f"{exp_id}.json")
    if not os.path.exists(path):
        return jsonify({"error": "Not found"}), 404
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/api/experiments/<exp_id>/csv", methods=["GET"])
def get_experiment_csv(exp_id):
    path = os.path.join(EXPERIMENTS_DIR, "summary.csv")
    if not os.path.exists(path):
        return jsonify({"error": "No summary yet"}), 404
    # Stream the CSV file
    return send_file(path, as_attachment=True, download_name="summary.csv")


# Add route to serve generated images
@app.route("/generated_images/<filename>")
def serve_image(filename):
    try:
        # Try both new and old versions of send_from_directory
        try:
            # New version (Flask 2.2+)
            return send_from_directory(
                directory=UPLOAD_FOLDER, path=filename, environ=request.environ
            )
        except TypeError:
            # Old version (pre Flask 2.2)
            return send_from_directory(UPLOAD_FOLDER, filename)
    except FileNotFoundError:
        abort(404)
    except Exception as e:
        app.logger.error(f"Error serving image {filename}: {str(e)}")
        abort(500)


@app.route("/api/status", methods=["GET"])
def get_status():
    """Get server status"""
    return jsonify(
        {
            "status": "running",
            "model_warmed": story_gen.model_warmed,
            "timestamp": int(time.time()),
        }
    )


if __name__ == "__main__":
    print("🚀 Starting Story Generator Backend...")
    print("📡 Ollama URL:", story_gen.ollama_url)
    print("🎨 ComfyUI URL:", story_gen.comfyui_url)
    print("🌐 Server starting on http://localhost:5000")
    print("\n\n")
    # test_image_generation()
    app.run(debug=True, host="0.0.0.0", port=5000)
