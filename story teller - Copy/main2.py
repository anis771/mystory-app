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
import timeout_decorator
from werkzeug.utils import secure_filename, send_from_directory
from keybert import KeyBERT
import websocket
import threading
from urllib.parse import urlencode
from urllib.request import urlopen, Request

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
UPLOAD_FOLDER = 'generated_images'
BACKEND_URL='http://localhost:5000'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class ComfyUIClient:
    def __init__(self, server_address="127.0.0.1:8000"):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())

    def queue_prompt(self, prompt):
        """Submit a prompt to ComfyUI"""
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = Request(f"http://{self.server_address}/prompt", data=data)
        req.add_header('Content-Type', 'application/json')

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
            with urlopen(f"http://{self.server_address}/history/{prompt_id}") as response:
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
                if data['type'] == 'executing':
                    executing_data = data['data']
                    if executing_data['node'] is None and executing_data['prompt_id'] == prompt_id:
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
                ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
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
        self.ollama_url = "http://localhost:11434/api/generate"
        self.comfyui_url = "http://localhost:8000/prompt"
        self.comfyui_client = ComfyUIClient("127.0.0.1:8000")
        self.model_warmed = False
        
        # Story context tracking
        self.current_story_context = {
            "characters": {},
            "settings": [],
            "art_style": None,
            "mood": None,
            "theme": None
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
            "refiner_start_step": 20
        }

    def warm_up_model(self) -> Dict:
        """Warm up the Ollama model and keep it alive"""
        try:
            print("🔥 Warming up Ollama model...")
            data = {
                "model": "llama3",
                "prompt": "Hello, I'm ready to generate stories!",
                "stream": False,
                "keep_alive": "30m",  # Keep model loaded for 30 minutes
                "options": {
                    "num_predict": 10,
                    "temperature": 0.7
                }
            }

            response = requests.post(
                self.ollama_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(data),
                timeout=60
            )

            if response.status_code == 200:
                self.model_warmed = True
                print("✅ Model warmed up successfully!")
                return {"status": "success", "message": "Model is ready"}
            else:
                return {"status": "error", "message": f"Failed to warm up model: {response.status_code}"}

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
                "options": {
                    "num_predict": 1200,
                    "temperature": 0.8,
                    "num_ctx": 8192 
                }
            }

            print(f"📝 Generating story for: {user_prompt}")
            response = requests.post(
                self.ollama_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(data),
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
                        "context": self.current_story_context
                    }
                }
            else:
                return {"status": "error", "message": f"Story generation failed: {response.status_code}"}

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
                "options": {
                    "num_predict": 150,
                    "temperature": 0.3
                }
            }

            response = requests.post(self.ollama_url, json=data, timeout=30)
            if response.status_code == 200:
                result_text = response.json().get("response", "")
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
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
            "color_palette": "warm, vibrant colors"
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
                "options": {
                    "num_predict": 200,
                    "temperature": 0.3
                }
            }

            response = requests.post(self.ollama_url, json=data, timeout=30)
            if response.status_code == 200:
                result_text = response.json().get("response", "")
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    elements = json.loads(json_match.group())
                    self.current_story_context["characters"] = elements.get("characters", {})
                    self.current_story_context["settings"] = elements.get("locations", [])
                    self.current_story_context["key_elements"] = elements.get("key_elements", [])
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
            if name not in ['The', 'And', 'But', 'When', 'Then']:
                self.current_story_context["characters"][name] = "character in the story"

    def _split_story_with_context(self, story_text: str) -> List[Dict]:
        """Split story into paragraphs while maintaining context"""
        if "---PARAGRAPH---" in story_text:
            raw_paragraphs = story_text.split("---PARAGRAPH---")
        else:
            raw_paragraphs = story_text.split("\n\n")

        paragraphs = []
        previous_context = ""
        
        for i, paragraph in enumerate(raw_paragraphs):
            paragraph = paragraph.strip()
            if paragraph and len(paragraph) > 50:
                # Generate image prompt with full context
                image_prompt = self._generate_contextual_image_prompt(
                    paragraph, 
                    i + 1, 
                    len(raw_paragraphs),
                    previous_context
                )
                
                paragraphs.append({
                    "id": i + 1,
                    "text": paragraph,
                    "image_prompt": image_prompt,
                    "image_url": None
                })
                
                # Keep a summary of previous paragraphs for context
                previous_context = self._summarize_for_context(previous_context, paragraph)

        return paragraphs

    def _summarize_for_context(self, previous: str, current: str) -> str:
        """Maintain a rolling summary of story context"""
        if not previous:
            return current[:200]
        return (previous + " " + current)[-400:]  # Keep last 400 chars

    def _generate_contextual_image_prompt(self, paragraph: str, paragraph_num: int, total_paragraphs: int, previous_context: str) -> str:
        """Generate image prompts that maintain story consistency"""
        try:
            # Get consistent art style from story context
            art_style = self.current_story_context.get("art_style", "storybook illustration")
            mood = self.current_story_context.get("mood", "whimsical")
            color_palette = self.current_story_context.get("color_palette", "vibrant colors")
            
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
                    "temperature": 0.5  # Lower temperature for consistency
                }
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
        art_style = self.current_story_context.get("art_style", "storybook illustration")
        
        # Simple keyword extraction
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(paragraph, keyphrase_ngram_range=(1, 3), stop_words='english')
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
            with open("comfyui_workflow/sdxl_story_image.json", "r", encoding="utf-8") as f:
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
                workflow_dict["7"]["inputs"]["text"] = self.workflow_config["negative_prompt"]
            if "15" in workflow_dict:
                workflow_dict["15"]["inputs"]["text"] = enhanced_prompt
            if "16" in workflow_dict:
                workflow_dict["16"]["inputs"]["text"] = self.workflow_config["negative_prompt"]
            
            # Rest of the configuration remains the same
            if "5" in workflow_dict:
                workflow_dict["5"]["inputs"]["width"] = self.workflow_config["width"]
                workflow_dict["5"]["inputs"]["height"] = self.workflow_config["height"]
            
            if "10" in workflow_dict:
                workflow_dict["10"]["inputs"]["steps"] = self.workflow_config["refiner_steps"]
                workflow_dict["10"]["inputs"]["cfg"] = self.workflow_config["cfg_scale"]
                workflow_dict["10"]["inputs"]["sampler_name"] = self.workflow_config["sampler"]
                workflow_dict["10"]["inputs"]["end_at_step"] = self.workflow_config["base_end_step"]
                if self.workflow_config["seed"] == -1:
                    workflow_dict["10"]["inputs"]["noise_seed"] = int(time.time() * 1000) % 1000000
            
            if "11" in workflow_dict:
                workflow_dict["11"]["inputs"]["steps"] = self.workflow_config["refiner_steps"]
                workflow_dict["11"]["inputs"]["cfg"] = self.workflow_config["cfg_scale"]
                workflow_dict["11"]["inputs"]["sampler_name"] = self.workflow_config["sampler"]
                workflow_dict["11"]["inputs"]["start_at_step"] = self.workflow_config["refiner_start_step"]

            return workflow_dict

        except Exception as e:
            print(f"Error preparing workflow: {e}")
            raise

    def generate_image_with_comfyui(self, prompt: str, paragraph_id: int, story_context: Optional[Dict] = None) -> Dict:
        """Generate image with optional story context"""
        try:
            # If story context is provided, update the current context
            if story_context:
                self.current_story_context.update(story_context)
            
            print(f"🎨 Generating image for paragraph {paragraph_id}")
            print(f"📝 Prompt: {prompt}")
            print(f"🎨 Style: {self.current_story_context.get('art_style', 'default')}")
            
            start_time = time.time()

            # Prepare workflow
            workflow = self._prepare_comfyui_workflow(prompt)

            # Submit prompt to ComfyUI
            prompt_response = self.comfyui_client.queue_prompt(workflow)

            if not prompt_response:
                return {
                    "status": "error",
                    "message": "Failed to submit prompt to ComfyUI"
                }

            prompt_id = prompt_response.get('prompt_id')
            if not prompt_id:
                return {
                    "status": "error",
                    "message": "No prompt ID received from ComfyUI"
                }

            print(f"Prompt submitted with ID: {prompt_id}")

            # Wait for completion
            history = self.comfyui_client.wait_for_completion(prompt_id, timeout=900)

            if not history:
                return {
                    "status": "error",
                    "message": "Image generation timed out or failed"
                }

            # Extract image information
            outputs = history.get('outputs', {})
            image_info = None
            for node_id, node_output in outputs.items():
                if 'images' in node_output:
                    image_info = node_output['images'][0]
                    break

            if not image_info:
                return {
                    "status": "error",
                    "message": "No image output found"
                }

            # Download the image
            image_data = self.comfyui_client.get_image(
                image_info['filename'],
                image_info.get('subfolder', ''),
                image_info.get('type', 'output')
            )

            if not image_data:
                return {
                    "status": "error",
                    "message": "Failed to download generated image"
                }

            # Save image locally
            image_url = self._save_image_from_comfyui(image_data, paragraph_id)

            if not image_url:
                return {
                    "status": "error",
                    "message": "Failed to save image locally"
                }

            generation_time = time.time() - start_time
            print(f"✅ Image generated in {generation_time:.2f}s")

            return {
                "status": "success",
                "image_url": image_url,
                "generation_time": generation_time,
                "message": "Image generated successfully"
            }

        except Exception as e:
            print(f"Image generation error: {e}")
            return {
                "status": "error",
                "message": f"Image generation failed: {str(e)}",
                "traceback": traceback.format_exc()
            }

    def _save_image_from_comfyui(self, image_data: bytes, paragraph_id: int) -> str:
        """Save image data to local storage"""
        try:
            filename = f"paragraph_{paragraph_id}_{uuid.uuid4().hex}.png"
            filename = secure_filename(filename)

            destination = os.path.join(UPLOAD_FOLDER, filename)
            with open(destination, 'wb') as f:
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


# Initialize the story generator
story_gen = ImprovedStoryGenerator()


@app.route('/api/warmup', methods=['POST'])
def warmup_server():
    """Warm up the Ollama model"""
    result = story_gen.warm_up_model()
    return jsonify(result)


@app.route('/api/generate-story', methods=['POST'])
def generate_story():
    """Generate a complete story with image prompts"""
    try:
        data = request.json
        user_prompt = data.get('prompt', '')
        num_paragraphs = data.get('paragraphs', 5)

        if not user_prompt:
            return jsonify({"status": "error", "message": "Prompt is required"}), 400

        if not story_gen.model_warmed:
            # Auto warm-up if not already done
            warmup_result = story_gen.warm_up_model()
            if warmup_result["status"] != "success":
                return jsonify(warmup_result), 500

        result = story_gen.generate_story(user_prompt, num_paragraphs)
        return jsonify(result)

    except Exception as e:
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"}), 500


@app.route('/api/generate-image', methods=['POST'])
def generate_image():
    """Generate image for a specific paragraph with optional context"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        paragraph_id = data.get('paragraph_id', 1)
        story_context = data.get('story_context', None)  # NEW: Accept story context

        if not prompt:
            return jsonify({"status": "error", "message": "Image prompt is required"}), 400

        result = story_gen.generate_image_with_comfyui(prompt, paragraph_id, story_context)
        return jsonify(result)

    except Exception as e:
        return jsonify({"status": "error", "message": f"Image generation error: {str(e)}"}), 500


# Add route to serve generated images
@app.route('/generated_images/<filename>')
def serve_image(filename):
    try:
        # Try both new and old versions of send_from_directory
        try:
            # New version (Flask 2.2+)
            return send_from_directory(
                directory=UPLOAD_FOLDER,
                path=filename,
                environ=request.environ
            )
        except TypeError:
            # Old version (pre Flask 2.2)
            return send_from_directory(UPLOAD_FOLDER, filename)
    except FileNotFoundError:
        abort(404)
    except Exception as e:
        app.logger.error(f"Error serving image {filename}: {str(e)}")
        abort(500)

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get server status"""
    return jsonify({
        "status": "running",
        "model_warmed": story_gen.model_warmed,
        "timestamp": int(time.time())
    })


def test_image_generation():
    # Initialize the story generator
    story_gen1 = ImprovedStoryGenerator()

    # Test parameters
    test_prompt = "A magical forest with glowing trees and fairies dancing in the moonlight"
    paragraph_id = 1

    print("=== Testing Image Generation ===")
    print(f"Using prompt: '{test_prompt}'")
    print(f"Paragraph ID: {paragraph_id}")

    # First warm up the model if needed
    # print("\n1. Warming up model...")
    # warmup_result = story_gen1.warm_up_model()
    # print("Warmup result:", warmup_result)
    #
    # if warmup_result["status"] != "success":
    #     print("❌ Model warmup failed, cannot proceed with image generation")
    #     return

    # Test the image generation
    print("\n2. Generating image...")
    result = story_gen1.generate_image_with_comfyui(test_prompt, paragraph_id)

    print("\n3. Generation result:")
    print(json.dumps(result, indent=2))

    if result["status"] == "success":
        print("\n✅ Image generated successfully!")
        print(f"Image URL: {result['image_url']}")
        print(f"Generation time: {result['generation_time']:.2f} seconds")

        # Verify the image file exists
        image_path = os.path.join('generated_images', os.path.basename(result['image_url']))
        if os.path.exists(image_path):
            print(f"✅ Image file found at: {image_path}")
            print(f"File size: {os.path.getsize(image_path) / 1024:.2f} KB")
        else:
            print(f"❌ Image file not found at: {image_path}")
    else:
        print("\n❌ Image generation failed")
        print("Error:", result["message"])


if __name__ == '__main__':
    print("🚀 Starting Story Generator Backend...")
    print("📡 Ollama URL:", story_gen.ollama_url)
    print("🎨 ComfyUI URL:", story_gen.comfyui_url)
    print("🌐 Server starting on http://localhost:5000")
    # test_image_generation()
    app.run(debug=True, host='0.0.0.0', port=5000)
