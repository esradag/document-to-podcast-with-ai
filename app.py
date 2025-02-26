import gradio as gr
import PyPDF2
import docx
import os
import tempfile
from gtts import gTTS
from pydub import AudioSegment
import io
import shutil
import openai
import requests
import json
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Text extraction functions
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text

# Retry decorator for API calls
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60)
)
def summarize_with_ai(text, model_name, api_key, podcast_style="informative", max_tokens=1500):
    """
    Summarize text using various AI models
    
    Args:
        text: Text to summarize
        model_name: Name of the model to use
        api_key: API key for the selected model
        podcast_style: Style of podcast (informative, conversational, storytelling)
        max_tokens: Maximum tokens in response
        
    Returns:
        Summarized text
    """
    # Select appropriate system message based on podcast style
    system_messages = {
        "informative": "You are an expert podcast creator that summarizes documents in an informative, factual style. Focus on key information and main points.",
        "conversational": "You are a friendly podcast host that summarizes documents in a conversational, engaging style. Use a casual tone, rhetorical questions, and make the content feel like a chat.",
        "storytelling": "You are a masterful storyteller that converts document content into narrative podcast form. Create a compelling narrative arc with a beginning, middle, and end."
    }
    
    user_message = f"Summarize the following document for a podcast script:\n\n{text}"
    system_message = system_messages.get(podcast_style, system_messages["informative"])
    
    # Check if text is too long - we may need to chunk it
    if len(text) > 15000:
        logger.info(f"Text is very long ({len(text)} chars), truncating for summarization")
        # Simple truncation approach - in production, use a more sophisticated chunking strategy
        text = text[:15000] + "...[content truncated for length]"
    
    try:
        # OpenAI models (GPT)
        if model_name in ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]:
            openai.api_key = api_key
            temperature = 0.7 if podcast_style == "conversational" else 0.4
            
            response = openai.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            return response.choices[0].message.content
            
        # Google Gemini
        elif model_name == "gemini-pro":
            gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
            
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": f"{system_message}\n\n{user_message}"}]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.7 if podcast_style == "conversational" else 0.4,
                    "maxOutputTokens": max_tokens,
                    "topP": 0.95
                }
            }
            
            response = requests.post(
                f"{gemini_url}?key={api_key}",
                headers=headers,
                json=data
            )
            
            response.raise_for_status()  # Raise exception for HTTP errors
            response_json = response.json()
            
            # Extract text from Gemini response
            if "candidates" in response_json and len(response_json["candidates"]) > 0:
                candidate = response_json["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    for part in candidate["content"]["parts"]:
                        if "text" in part:
                            return part["text"]
            
            raise ValueError("Could not extract text from Gemini response")
            
        # Anthropic Claude
        elif model_name == "claude-3-sonnet":
            claude_url = "https://api.anthropic.com/v1/messages"
            
            headers = {
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": "claude-3-sonnet-20240229",  # Fixed model for Claude
                "max_tokens": max_tokens,
                "system": system_message,
                "messages": [
                    {"role": "user", "content": user_message}
                ],
                "temperature": 0.7 if podcast_style == "conversational" else 0.4
            }
            
            response = requests.post(
                claude_url,
                headers=headers,
                json=data
            )
            
            response.raise_for_status()  # Raise exception for HTTP errors
            response_json = response.json()
            
            if "content" in response_json and len(response_json["content"]) > 0:
                for content_item in response_json["content"]:
                    if content_item["type"] == "text":
                        return content_item["text"]
            
            raise ValueError("Could not extract text from Claude response")
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
    except Exception as e:
        logger.error(f"Error in AI summarization with {model_name}: {str(e)}")
        # Return error message and fallback to simple truncation if API fails
        return f"Error using {model_name} API: {str(e)}\n\nFallback summary:\n\n" + (text[:1500] + "..." if len(text) > 2000 else text)

def convert_to_podcast_format(text, podcast_style="informative"):
    """
    Convert text to podcast format with different styles.
    
    With AI integration, this function is simplified as most formatting
    is handled by the API prompt, but we can still add intro/outro.
    """
    # Prepare intro and outro based on style
    if podcast_style == "informative":
        intro = "Hello and welcome to today's podcast. We'll be exploring an important topic in detail. "
        outro = "\n\nThank you for listening to this informative podcast. If you found this helpful, please subscribe for more content like this."
    elif podcast_style == "conversational":
        intro = "Hey there! Thanks for tuning in today. We've got a really interesting topic to chat about. "
        outro = "\n\nAnyway, that's all for now! Hope you enjoyed our conversation and we'll catch you next time. Don't forget to like and subscribe!"
    elif podcast_style == "storytelling":
        intro = "Gather around as we embark on a journey through an intriguing story. "
        outro = "\n\nAnd that concludes our tale for today. We hope it sparked your imagination and we look forward to sharing another story with you soon."
    else:
        # Default
        intro = "Welcome to this podcast episode. Today, we'll discuss an important topic. "
        outro = "\n\nThank you for listening, and we hope to see you in the next episode!"
    
    # Check if the text already has a good intro/conclusion from the AI
    # If so, we might not need to add our own
    if len(text.split('\n\n')) > 3:
        # AI probably already formatted it well
        return text
    else:
        # Add our formatting
        return intro + text + outro

# Create podcast audio file
def create_podcast_audio(text, voice_type="en", speed=1.0):
    # Create audio file
    tts = gTTS(text=text, lang=voice_type, slow=False)
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    temp_file.close()
    
    tts.save(temp_file.name)
    
    # Adjust audio speed (using pydub)
    if speed != 1.0:
        audio = AudioSegment.from_mp3(temp_file.name)
        # Process for speed change (speed > 1.0 for faster)
        audio_speed_adjusted = audio._spawn(audio.raw_data, overrides={
            "frame_rate": int(audio.frame_rate * speed)
        })
        audio_speed_adjusted.export(temp_file.name, format="mp3")
    
    return temp_file.name

# Main processing function
def process_document_to_podcast(document, model_name, api_key, voice_type, speed, podcast_style, custom_instructions):
    if not api_key or len(api_key.strip()) < 10:  # Basic validation
        return "Please enter a valid API key.", None
    
    if document is None:
        return "Please upload a document.", None
    
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Process the value returned by Gradio's File component
        if isinstance(document, str):
            # If returned as a file path
            file_path = document
            file_name = os.path.basename(file_path)
            temp_file_path = os.path.join(temp_dir, file_name)
            shutil.copy(file_path, temp_file_path)
        else:
            # If returned as a file object
            file_name = document.name
            temp_file_path = os.path.join(temp_dir, file_name)
            
            # Read and save the file content as bytes
            if hasattr(document, 'read'):
                with open(temp_file_path, "wb") as f:
                    # Reset file object position to the beginning
                    if hasattr(document, 'seek'):
                        document.seek(0)
                    content = document.read()
                    f.write(content)
            else:
                return "File reading error.", None
        
        # Extract text based on file extension
        file_ext = os.path.splitext(file_name)[1].lower()
        
        if file_ext == ".pdf":
            text = extract_text_from_pdf(temp_file_path)
        elif file_ext == ".docx":
            text = extract_text_from_docx(temp_file_path)
        elif file_ext == ".txt":
            text = extract_text_from_txt(temp_file_path)
        else:
            return f"Unsupported file format: {file_ext}! Please upload a PDF, DOCX, or TXT file.", None
        
        # Check if text is empty
        if not text or text.isspace():
            return "No text found in the document or the text is empty.", None
        
        logger.info(f"Processing document: {file_name}, Size: {len(text)} characters")
        logger.info(f"Using model: {model_name} with style: {podcast_style}")
        
        # Summarize the text with the selected AI
        summarized_text = summarize_with_ai(
            text=text,
            model_name=model_name,
            api_key=api_key,
            podcast_style=podcast_style,
            max_tokens=1500  # Adjust based on your needs
        )
        
        # Apply custom podcast formatting if needed
        podcast_text = convert_to_podcast_format(summarized_text, podcast_style)
            
        # Apply custom instructions if provided
        if custom_instructions:
            podcast_text = f"{custom_instructions}\n\n{podcast_text}"
        
        # Create audio file
        podcast_audio_path = create_podcast_audio(
            text=podcast_text,
            voice_type=voice_type,
            speed=float(speed)
        )
        
        # Clean up temporary files
        try:
            os.remove(temp_file_path)
            os.rmdir(temp_dir)
        except Exception as e:
            logger.warning(f"Error cleaning up temp files: {str(e)}")
        
        return podcast_text, podcast_audio_path
    
    except Exception as e:
        error_message = f"Error during processing: {str(e)}"
        logger.error(error_message)
        return error_message, None

# Gradio interface
def create_interface():
    with gr.Blocks(title="Document to Podcast") as app:
        gr.Markdown("# Document to Podcast with AI")
        
        with gr.Row():
            with gr.Column():
                document_input = gr.File(
                    label="Upload Document",
                    file_types=[".pdf", ".docx", ".txt"]
                )
                
                with gr.Group():
                    gr.Markdown("### Settings")
                    
                    # Combined model selection
                    model_name = gr.Dropdown(
                        choices=[
                            "gpt-3.5-turbo", 
                            "gpt-4", 
                            "gpt-4-turbo", 
                            "gemini-pro", 
                            "claude-3-sonnet"
                        ],
                        value="gpt-3.5-turbo",
                        label="AI Model"
                    )
                    
                    # Single API key input that changes based on model
                    api_key = gr.Textbox(
                        label="API Key",
                        placeholder="Enter your API key here...",
                        type="password"
                    )
                    
                    # Other podcast settings
                    podcast_style = gr.Dropdown(
                        choices=["informative", "conversational", "storytelling"], 
                        value="informative", 
                        label="Podcast Style"
                    )
                    voice_type = gr.Dropdown(
                        choices=["en", "fr", "de", "es", "it", "tr"], 
                        value="en", 
                        label="Voice Language"
                    )
                    speed = gr.Slider(
                        minimum=0.5, 
                        maximum=1.5, 
                        value=1.0, 
                        step=0.1, 
                        label="Speech Rate"
                    )
                    custom_instructions = gr.Textbox(
                        label="Custom Instructions",
                        placeholder="Add custom introduction..."
                    )
                
                process_btn = gr.Button("Generate Podcast", variant="primary")
            
            with gr.Column():
                with gr.Tab("Podcast Text"):
                    text_output = gr.Textbox(label="Generated Text", lines=12)
                
                with gr.Tab("Podcast Audio"):
                    audio_output = gr.Audio(label="Audio")
        
        # Function to update API key label based on selected model
        def update_api_key_label(model):
            if model.startswith("gpt"):
                return gr.update(label="OpenAI API Key")
            elif model.startswith("gemini"):
                return gr.update(label="Google Gemini API Key")
            elif model.startswith("claude"):
                return gr.update(label="Anthropic Claude API Key")
            return gr.update(label="API Key")
        
        # Connect the process button
        process_btn.click(
            fn=process_document_to_podcast,
            inputs=[
                document_input,
                model_name,
                api_key,
                voice_type, 
                speed, 
                podcast_style,
                custom_instructions
            ],
            outputs=[text_output, audio_output]
        )
        
        # Update API key label when model changes
        model_name.change(
            fn=update_api_key_label,
            inputs=[model_name],
            outputs=[api_key]
        )
        
        gr.Markdown("Supports PDF, DOCX, and TXT files. API keys are used only for processing and not stored.")
    
    return app

# Start the application
if __name__ == "__main__":
    app = create_interface()
    app.launch()