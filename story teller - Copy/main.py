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
from typing import List, Dict
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


class StoryGenerator:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"
        self.comfyui_url = "http://localhost:8000/prompt"  # Default ComfyUI port
        self.comfyui_client = ComfyUIClient("127.0.0.1:8000")  # Changed this line
        self.model_warmed = False

        # UPDATED: SDXL workflow configuration
        self.workflow_config = {
            "positive_prompt_prefix": "high quality, finely detailed, vibrant colors, soft lighting, storybook illustration, fantasy art, whimsical, painterly, dreamy atmosphere,",
            "negative_prompt": "low quality, blurry, distorted, bad anatomy, bad hands, extra limbs, extra fingers, multiple heads, disfigured, ugly, poorly drawn, text, watermark, signature, frame, border, glitch",
            "steps": 25,
            "cfg_scale": 8,
            "width": 1024,  # UPDATED: SDXL optimal resolution
            "height": 1024,  # UPDATED: SDXL optimal resolution
            "sampler": "euler",
            "seed": -1,
            "base_steps": 20,      # NEW: Steps for base model
            "refiner_steps": 25,   # NEW: Total steps including refiner
            "base_end_step": 20,   # NEW: When to switch to refiner
            "refiner_start_step": 20  # NEW: When refiner starts
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
        """Generate a story based on user prompt"""
        try:
            story_prompt = f"""
            Write a creative {num_paragraphs}-paragraph story based on the following idea:
            "{user_prompt}"

            Story Requirements:
            - The story should be imaginative, engaging, and rich in descriptive language.
            - Each paragraph must be between 10 to 15 sentences.
            - Focus on vivid **visual scenes** with strong **characters**, **settings**, and **actions** that are easy to illustrate.
            - Include emotional tone, sensory details (sight, sound, feel), and dynamic interactions.
            - Ensure the story flows naturally across the paragraphs.
            - Use clear paragraph breaks with this exact marker: ---PARAGRAPH---

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

                # Split story into paragraphs
                paragraphs = self._split_story_into_paragraphs(story_text)

                return {
                    "status": "success",
                    "story": {
                        "title": self._generate_title(user_prompt),
                        "paragraphs": paragraphs,
                        "total_paragraphs": len(paragraphs)
                    }
                }
            else:
                return {"status": "error", "message": f"Story generation failed: {response.status_code}"}

        except Exception as e:
            return {"status": "error", "message": f"Story generation error: {str(e)}"}

    def _split_story_into_paragraphs(self, story_text: str) -> List[Dict]:
        """Split story into paragraphs and generate image prompts"""
        # Split by paragraph markers or double newlines
        if "---PARAGRAPH---" in story_text:
            raw_paragraphs = story_text.split("---PARAGRAPH---")
        else:
            raw_paragraphs = story_text.split("\n\n")

        paragraphs = []
        for i, paragraph in enumerate(raw_paragraphs):
            paragraph = paragraph.strip()
            if paragraph and len(paragraph) > 50:  # Filter out very short paragraphs
                image_prompt = self._extract_image_prompt(paragraph)
                paragraphs.append({
                    "id": i + 1,
                    "text": paragraph,
                    "image_prompt": image_prompt,
                    "image_url": None  # Will be populated after image generation
                })

        return paragraphs
    

    def _extract_image_prompt(self, paragraph: str) -> str:
        """KeyBERT keyword extraction + LLM refinement for SDXL-optimized prompts"""
        try:
            # Step 1: Extract top keywords using KeyBERT
            kw_model = KeyBERT()
            keywords = kw_model.extract_keywords(paragraph, keyphrase_ngram_range=(1, 2))
            print(f"Extracted Total keywords : {keywords}")
            keywords = [kw[0] for kw in keywords[:5]]  # Keep top 5

            print(f"Selected keywords : {keywords}")
          
            ART_STYLE = random.choice(["digital painting", "watercolor illustration"])
            MOOD = random.choice(["glowing twilight", "golden hour", "misty morning"])
            PERSPECTIVE = random.choice(["wide shot", "close-up", "over-the-shoulder"])
            # Step 2: Refine with LLM into an image prompt
            refinement_prompt = f"""
            You are generating an image prompt for a powerful image generation model like Stable Diffusion XL.

            Your job is to turn these keywords into a concrete, vivid, and clear image generation prompt:

            Keywords: {', '.join(keywords)}

            The prompt must:
            - Focus on specific **visuals**: setting, characters, objects, actions
            - Be less than 50 words
            - Start with the **main subject**
            - Include **art style** {ART_STYLE}
            - Include **mood or lighting** {MOOD}
            - Optionally add **camera perspective** {PERSPECTIVE}

            Just return the final image prompt only.
            """

            # Use your existing LLM call
            data = {
                "model": "llama3",
                "prompt": refinement_prompt,
                "stream": False,
                "keep_alive": "30m",
                "options": {
                    "num_predict": 80,
                    "temperature": 0.6
                }
            }

            response = requests.post(
                self.ollama_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(data),
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                raise Exception(f"LLM responded with status code {response.status_code}")

        except Exception as e:
            print(f"⚠️ Enhanced image prompt generation failed: {e}")
            # Fallback: original paragraph-based LLM generation
            try:
                fallback_prompt = f"""
                Based on this story paragraph, create a concise image generation prompt (max 50 words) that captures the main visual scene:

                Paragraph: "{paragraph}"

                Image prompt (describe the scene, characters, setting, mood):
                """

                fallback_data = {
                    "model": "llama3",
                    "prompt": fallback_prompt,
                    "stream": False,
                    "keep_alive": "30m",
                    "options": {
                        "num_predict": 80,
                        "temperature": 0.6
                    }
                }

                response = requests.post(
                    self.ollama_url,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(fallback_data),
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "").strip()

            except Exception as inner_e:
                print(f"⚠️ Fallback prompt generation also failed: {inner_e}")

            # Final fallback
            return self._simple_prompt_extraction(paragraph)
    
    # def _extract_image_prompt(self, paragraph: str) -> str:
    #     """Extract/generate an image prompt from a paragraph"""
    #     try:
    #         prompt_generation = f"""
    #         Based on this story paragraph, create a concise image generation prompt (max 50 words) that captures the main visual scene:

    #         Paragraph: "{paragraph}"

    #         Image prompt (describe the scene, characters, setting, mood):
    #         """

    #         data = {
    #             "model": "llama3",
    #             "prompt": prompt_generation,
    #             "stream": False,
    #             "keep_alive": "30m",
    #             "options": {
    #                 "num_predict": 80,
    #                 "temperature": 0.6
    #             }
    #         }

    #         response = requests.post(
    #             self.ollama_url,
    #             headers={"Content-Type": "application/json"},
    #             data=json.dumps(data),
    #             timeout=30
    #         )

    #         if response.status_code == 200:
    #             result = response.json()
    #             return result.get("response", "").strip()
    #         else:
    #             # Fallback: simple keyword extraction
    #             return self._simple_prompt_extraction(paragraph)

    #     except Exception as e:
    #         print(f"⚠️ Image prompt generation failed: {e}")
    #         return self._simple_prompt_extraction(paragraph)

    def _simple_prompt_extraction(self, paragraph: str) -> str:
        """Fallback method for creating image prompts"""
        # Simple keyword-based prompt creation
        words = paragraph.lower().split()
        keywords = [word for word in words if len(word) > 4][:8]
        return f"Illustration of {' '.join(keywords[:5])}, storybook style"

    def _generate_title(self, user_prompt: str) -> str:
        """Generate a title for the story"""
        # Simple title generation based on user prompt
        words = user_prompt.split()
        if len(words) <= 4:
            return f"The Tale of {user_prompt.title()}"
        else:
            return f"The Adventures of {' '.join(words[:3]).title()}"
        

    def _prepare_comfyui_workflow(self, prompt: str) -> Dict:
        """UPDATED: Prepare the SDXL ComfyUI workflow JSON with the provided prompt."""
        try:
            # UPDATED: Load the SDXL workflow JSON file
            with open("comfyui_workflow/sdxl_story_image.json", "r", encoding="utf-8") as f:
                workflow_dict = json.load(f)

            # Add quality and style prefixes for better story illustrations
            enhanced_prompt = f"{self.workflow_config['positive_prompt_prefix']}{prompt}"

            # UPDATED: Update the positive prompt for BASE model (node 6)
            if "6" in workflow_dict:
                workflow_dict["6"]["inputs"]["text"] = enhanced_prompt

            # UPDATED: Update the negative prompt for BASE model (node 7)
            if "7" in workflow_dict:
                workflow_dict["7"]["inputs"]["text"] = self.workflow_config["negative_prompt"]

            # UPDATED: Update the positive prompt for REFINER model (node 15)
            if "15" in workflow_dict:
                workflow_dict["15"]["inputs"]["text"] = enhanced_prompt

            # UPDATED: Update the negative prompt for REFINER model (node 16)
            if "16" in workflow_dict:
                workflow_dict["16"]["inputs"]["text"] = self.workflow_config["negative_prompt"]

            # UPDATED: Update image dimensions to SDXL resolution (node 5)
            if "5" in workflow_dict:
                workflow_dict["5"]["inputs"]["width"] = self.workflow_config["width"]
                workflow_dict["5"]["inputs"]["height"] = self.workflow_config["height"]

            # UPDATED: Update BASE sampling parameters (node 10)
            if "10" in workflow_dict:
                workflow_dict["10"]["inputs"]["steps"] = self.workflow_config["refiner_steps"]
                workflow_dict["10"]["inputs"]["cfg"] = self.workflow_config["cfg_scale"]
                workflow_dict["10"]["inputs"]["sampler_name"] = self.workflow_config["sampler"]
                workflow_dict["10"]["inputs"]["end_at_step"] = self.workflow_config["base_end_step"]
                # Generate random seed if needed
                if self.workflow_config["seed"] == -1:
                    workflow_dict["10"]["inputs"]["noise_seed"] = int(time.time() * 1000) % 1000000

            # UPDATED: Update REFINER sampling parameters (node 11)
            if "11" in workflow_dict:
                workflow_dict["11"]["inputs"]["steps"] = self.workflow_config["refiner_steps"]
                workflow_dict["11"]["inputs"]["cfg"] = self.workflow_config["cfg_scale"]
                workflow_dict["11"]["inputs"]["sampler_name"] = self.workflow_config["sampler"]
                workflow_dict["11"]["inputs"]["start_at_step"] = self.workflow_config["refiner_start_step"]

            return workflow_dict

        except FileNotFoundError:
            print("❌ SDXL Workflow JSON file not found at comfyui_workflow/sdxl_story_image.json")
            raise
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in workflow file: {e}")
            raise
        except Exception as e:
            print(f"❌ Error preparing SDXL workflow: {e}")
            raise


    def _clean_image_prompt(self, prompt: str) -> str:
        """UPDATED: Enhanced prompt cleaning and enhancement for SDXL story illustrations"""
        # Remove unwanted characters but keep essential punctuation
        clean_prompt = re.sub(r'[^\w\s.,-]', '', prompt)

        # Remove redundant words and focus on visual elements
        stop_words = {'and', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        words = clean_prompt.lower().split()
        filtered_words = [word for word in words if word not in stop_words or len(word) > 4]

        # UPDATED: Prioritize visual descriptors - SDXL can handle longer prompts
        if len(filtered_words) > 20:  # Increased from 15 for SDXL
            # Keep the most descriptive words (longer words tend to be more descriptive)
            filtered_words = sorted(filtered_words, key=len, reverse=True)[:20]

        clean_prompt = ' '.join(filtered_words)

        # Add style enhancement for story illustrations
        # if not any(style in clean_prompt.lower() for style in ['illustration', 'art', 'painting', 'drawing']):
        #     clean_prompt += ", children's book illustration style"

        return clean_prompt.strip()[:300]  # UPDATED: Increased limit for SDXL


    # def _prepare_comfyui_workflow(self, prompt: str) -> Dict:
    #     """Prepare the ComfyUI workflow JSON with the provided prompt."""
    #     try:
    #         # Load the workflow JSON file
    #         with open("comfyui_workflow/story_image.json", "r", encoding="utf-8") as f:
    #             workflow_dict = json.load(f)

    #         # Add quality and style prefixes for better story illustrations
    #         enhanced_prompt = f"{self.workflow_config['positive_prompt_prefix']}{prompt}"

    #         # Update the positive prompt (node 6)
    #         if "6" in workflow_dict:
    #             workflow_dict["6"]["inputs"]["text"] = enhanced_prompt

    #         # Update the negative prompt (node 7)
    #         if "7" in workflow_dict:
    #             workflow_dict["7"]["inputs"]["text"] = self.workflow_config["negative_prompt"]

    #         # Update image dimensions to match SD 2.1 768 optimal resolution (node 5)
    #         if "5" in workflow_dict:
    #             workflow_dict["5"]["inputs"]["width"] = self.workflow_config["width"]
    #             workflow_dict["5"]["inputs"]["height"] = self.workflow_config["height"]

    #         # Update sampling parameters (node 3)
    #         if "3" in workflow_dict:
    #             workflow_dict["3"]["inputs"]["steps"] = self.workflow_config["steps"]
    #             workflow_dict["3"]["inputs"]["cfg"] = self.workflow_config["cfg_scale"]
    #             workflow_dict["3"]["inputs"]["sampler_name"] = self.workflow_config["sampler"]
    #             # Keep the existing seed or use a random one
    #             # workflow_dict["3"]["inputs"]["seed"] = self.workflow_config.get("seed",
    #             #                                                                 workflow_dict["3"]["inputs"]["seed"])

    #         return workflow_dict

    #     except FileNotFoundError:
    #         print("❌ Workflow JSON file not found at comfyui_workflow/story_image.json")
    #         raise
    #     except json.JSONDecodeError as e:
    #         print(f"❌ Invalid JSON in workflow file: {e}")
    #         raise
    #     except Exception as e:
    #         print(f"❌ Error preparing workflow: {e}")
    #         raise

    # def _clean_image_prompt(self, prompt: str) -> str:
    #     """Enhanced prompt cleaning and enhancement for story illustrations"""
    #     # Remove unwanted characters but keep essential punctuation
    #     clean_prompt = re.sub(r'[^\w\s.,-]', '', prompt)

    #     # Remove redundant words and focus on visual elements
    #     stop_words = {'and', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
    #     words = clean_prompt.lower().split()
    #     filtered_words = [word for word in words if word not in stop_words or len(word) > 4]

    #     # Prioritize visual descriptors and limit length for SD 2.1
    #     if len(filtered_words) > 15:
    #         # Keep the most descriptive words (longer words tend to be more descriptive)
    #         filtered_words = sorted(filtered_words, key=len, reverse=True)[:15]

    #     clean_prompt = ' '.join(filtered_words)

    #     # Add style enhancement for story illustrations
    #     if not any(style in clean_prompt.lower() for style in ['illustration', 'art', 'painting', 'drawing']):
    #         clean_prompt += ", children's book illustration style"

    #     return clean_prompt.strip()[:200]  # Limit for SD 2.1 optimal performance

    def _get_output_image_filename(self, comfyui_response: Dict) -> str:
        """Extract the output image filename from ComfyUI response"""
        try:
            # The response contains node outputs - find the SaveImage node
            for node_id, node_output in comfyui_response.items():
                if "images" in node_output:
                    for image_info in node_output["images"]:
                        if "filename" in image_info:
                            return image_info["filename"]
            return None
        except:
            return None

    def _wait_for_image(self, filename: str, timeout: int = 30) -> str:
        """Wait for the image file to appear in ComfyUI output directory"""
        comfyui_output = "ComfyUI/output"  # Default ComfyUI output directory
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                for root, _, files in os.walk(comfyui_output):
                    if filename in files:
                        return os.path.join(root, filename)
                time.sleep(1)
            except:
                time.sleep(1)

        return None


    def _save_image_from_comfyui(self, image_data: bytes, paragraph_id: int) -> str:
        """Save image data to local storage and return URL"""
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


    def generate_image_with_comfyui(self, prompt: str, paragraph_id: int) -> Dict:
        """UPDATED: Generate image using ComfyUI SDXL with proper WebSocket communication"""
        try:
            print(f"🎨 Generating SDXL image for paragraph {paragraph_id}")
            start_time = time.time()

            # Clean and prepare the prompt
            clean_prompt = self._clean_image_prompt(prompt)
            
            print(f"Promt for SDXL image generation:  {prompt}")
           

            # Prepare SDXL workflow
            workflow = self._prepare_comfyui_workflow(clean_prompt)

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

            print(f"SDXL Prompt submitted with ID: {prompt_id}")

            # Wait for completion (SDXL takes longer)
            history = self.comfyui_client.wait_for_completion(prompt_id, timeout=900)  # Increased timeout for SDXL

            if not history:
                return {
                    "status": "error",
                    "message": "SDXL image generation timed out or failed"
                }

            # Extract image information from history
            outputs = history.get('outputs', {})

            # UPDATED: Find the SaveImage node output (node "19" in SDXL workflow)
            image_info = None
            for node_id, node_output in outputs.items():
                if 'images' in node_output:
                    image_info = node_output['images'][0]
                    break

            if not image_info:
                return {
                    "status": "error",
                    "message": "No image output found in SDXL ComfyUI response"
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
                    "message": "Failed to download generated SDXL image"
                }

            # Save image locally
            image_url = self._save_image_from_comfyui(image_data, paragraph_id)

            if not image_url:
                return {
                    "status": "error",
                    "message": "Failed to save SDXL image locally"
                }

            generation_time = time.time() - start_time
            print(f"✅ SDXL Image generated in {generation_time:.2f}s: {image_url}")

            return {
                "status": "success",
                "image_url": image_url,
                "generation_time": generation_time,
                "message": "SDXL image generated successfully"
            }

        except Exception as e:
            print(f"SDXL ComfyUI generation error: {e}")
            return {
                "status": "error",
                "message": f"SDXL image generation failed: {str(e)}",
                "traceback": traceback.format_exc()
            }


# Initialize the story generator
story_gen = StoryGenerator()


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
    """Generate image for a specific paragraph"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        paragraph_id = data.get('paragraph_id', 1)

        if not prompt:
            return jsonify({"status": "error", "message": "Image prompt is required"}), 400

        result = story_gen.generate_image_with_comfyui(prompt, paragraph_id)
        print(result)
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
    story_gen1 = StoryGenerator()

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
