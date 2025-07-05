from flask import Flask, render_template, request, jsonify, Response
import json
import os
from google import genai
from google.genai import types
import argparse
import datetime
import markdown
import re

app = Flask(__name__)

# Configure the Gemini API key
# IMPORTANT: Set the GOOGLE_API_KEY environment variable before running the app.
# For example, in your terminal: export GOOGLE_API_KEY='your_api_key'
api_key = os.getenv("GOOGLE_API_KEY")
client = None
if not api_key:
    print("WARNING: No GOOGLE_API_KEY set. You'll need to set it before using the app.")
    print("Set it with: $env:GOOGLE_API_KEY='your_api_key' (PowerShell)")
    print("Or with: set GOOGLE_API_KEY=your_api_key (Command Prompt)")
else:
    client = genai.Client(api_key=api_key)
    print(f"Google AI client configured (API key length: {len(api_key)})")

# Path to the JSON file (will be set by command-line argument)

def load_chat_history():
    """Loads and transforms chat history from the JSON file."""
    file_path = app.config.get('JSON_FILE')
    print(f"Attempting to load chat history from: {file_path}")
    if not file_path:
        print("JSON_FILE not configured in the app.")
        return {"turns": []}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        source_turns = []
        
        # Check for Google AI Studio format with chunkedPrompt
        if isinstance(data, dict) and 'chunkedPrompt' in data and 'chunks' in data['chunkedPrompt']:
            print("Found Google AI Studio format with chunkedPrompt.chunks")
            source_turns = data['chunkedPrompt']['chunks']
        elif isinstance(data, list):
            source_turns = data
        elif isinstance(data, dict) and 'turns' in data:
            source_turns = data['turns']
        
        transformed_turns = []
        for turn in source_turns:
            # Skip thought entries
            if turn.get('isThought'):
                continue

            text_content = ""
            if 'text' in turn:
                text_content = turn['text']
            elif 'parts' in turn and isinstance(turn.get('parts'), list) and len(turn['parts']) > 0 and 'text' in turn['parts'][0]:
                text_content = turn['parts'][0]['text']

            # Only add turns that have both role and text content
            if 'role' in turn and text_content.strip():
                 transformed_turns.append({
                     'role': turn['role'],
                     'parts': [{'text': text_content}]
                 })

        print(f"Successfully loaded and transformed {len(transformed_turns)} turns.")
        return {"turns": transformed_turns}

    except FileNotFoundError:
        print(f"File not found at path: {file_path}")
        return {"turns": []}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        return {"turns": []}

def save_chat_history(data):
    """Saves the chat history to the JSON file."""
    with open(app.config['JSON_FILE'], 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

@app.route('/')
def index():
    """Renders the main chat page."""
    return render_template('index.html')

@app.route('/get_history')
def get_history():
    """Returns the chat history as JSON with formatted HTML."""
    chat_data = load_chat_history()
    formatted_turns = []
    
    for turn in chat_data.get("turns", []):
        formatted_turn = {
            'role': turn['role'],
            'parts': []
        }
        
        for part in turn.get('parts', []):
            formatted_part = {
                'text': part.get('text', ''),
                'html': format_text(part.get('text', ''))
            }
            formatted_turn['parts'].append(formatted_part)
        
        formatted_turns.append(formatted_turn)
    
    return jsonify(formatted_turns)

@app.route('/send_prompt', methods=['POST'])
def send_prompt():
    """Receives a prompt, sends it to Gemini, streams the response, and saves the interaction."""
    # Check if API key is available
    if not client:
        return jsonify({"error": "Google API client not configured. Please set the GOOGLE_API_KEY environment variable."}), 500
    
    user_prompt = request.json.get('prompt')
    system_instruction = request.json.get('systemInstruction', '')
    cache_name = request.json.get('cacheName', '')
    
    if not user_prompt:
        return jsonify({"error": "Prompt is required"}), 400

    def generate():
        chat_data = load_chat_history()
        try:
            print(f"Processing prompt: {user_prompt[:50]}...")
            print(f"Using cache: {cache_name}")
            print(f"Using system instruction: {bool(system_instruction.strip())}")
            
            model_name = "gemini-2.5-flash"
            
            # Prepare config based on whether we have cache or system instruction
            if cache_name.strip():
                print(f"Generating with cache: {cache_name}")
                config = types.GenerateContentConfig(
                    cached_content=cache_name
                )
            elif system_instruction.strip():
                print("Generating with system instruction")
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction
                )
            else:
                print("Generating with basic config")
                config = types.GenerateContentConfig()
            
            print("Generating content...")
            response = client.models.generate_content_stream(
                model=model_name,
                contents=user_prompt,
                config=config
            )
            
            full_response = ""
            chunk_count = 0
            usage_metadata = None
            
            for chunk in response:
                if chunk.text:
                    chunk_count += 1
                    print(f"Received chunk {chunk_count}: {len(chunk.text)} characters")
                    full_response += chunk.text
                    # Send just the raw text chunk for streaming
                    yield f"data: {json.dumps({'text': chunk.text})}\n\n"
                
                # Capture usage metadata from the final chunk
                if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                    usage_metadata = chunk.usage_metadata

            print(f"Generation complete. Total response length: {len(full_response)}")
            
            # Send token usage data
            if usage_metadata:
                token_data = {
                    'type': 'token_usage',
                    'prompt_token_count': usage_metadata.prompt_token_count,
                    'candidates_token_count': usage_metadata.candidates_token_count,
                    'total_token_count': usage_metadata.total_token_count
                }
                yield f"data: {json.dumps(token_data)}\n\n"
                print(f"Token usage - Prompt: {usage_metadata.prompt_token_count}, Response: {usage_metadata.candidates_token_count}, Total: {usage_metadata.total_token_count}")
            
            # Once streaming is complete, save the full interaction
            chat_data.get("turns", []).append({"role": "user", "parts": [{"text": user_prompt}]})
            chat_data.get("turns", []).append({"role": "model", "parts": [{"text": full_response}]})
            save_chat_history(chat_data)
            print("Chat history saved")
            
            # Auto-update cache if we have a cache name
            if cache_name.strip():
                try:
                    print(f"Updating cache: {cache_name}")
                    # Prepare updated content for cache
                    updated_contents = []
                    
                    # Add conversation history
                    for turn in chat_data.get("turns", []):
                        updated_contents.append(turn)
                    
                    # Update the cached content
                    cache_config = types.CreateCachedContentConfig(
                        system_instruction=system_instruction if system_instruction.strip() else None,
                        contents=updated_contents
                    )
                    
                    client.caches.update(
                        name=cache_name,
                        config=cache_config
                    )
                    print(f"Cache {cache_name} updated with new conversation turn")
                    
                except Exception as cache_error:
                    print(f"Failed to update cache: {cache_error}")
                    # Don't fail the main request if cache update fails

        except Exception as e:
            print(f"Error in generate(): {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/cache/send', methods=['POST'])
def send_to_cache():
    """Send current conversation context to Gemini cache."""
    if not client:
        return jsonify({"success": False, "error": "Google API client not configured"})
    
    try:
        system_instruction = request.json.get('systemInstruction', '')
        chat_data = load_chat_history()
        
        # Prepare content for caching
        contents = []
        
        # Add conversation history
        for turn in chat_data.get("turns", []):
            contents.append(turn)
        
        if not contents:
            return jsonify({"success": False, "error": "No content to cache"})
        
        # Create cached content
        cache_config = types.CreateCachedContentConfig(
            system_instruction=system_instruction if system_instruction.strip() else None,
            contents=contents,
            ttl="28800s"  # Cache for 8 hour (28800 seconds)
        )
        
        cache = client.caches.create(
            model="gemini-2.5-flash",
            config=cache_config
        )
        
        return jsonify({
            "success": True, 
            "cacheName": cache.name,
            "message": "Context successfully cached"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/cache/update', methods=['POST'])
def update_cache():
    """Update existing cache with current conversation context."""
    if not client:
        return jsonify({"success": False, "error": "Google API client not configured"})
        
    try:
        cache_name = request.json.get('cacheName')
        system_instruction = request.json.get('systemInstruction', '')
        
        if not cache_name:
            return jsonify({"success": False, "error": "Cache name is required"})
        
        chat_data = load_chat_history()
        
        # Prepare updated content
        contents = []
        
        # Add conversation history
        for turn in chat_data.get("turns", []):
            contents.append(turn)
        
        # Update the cached content
        cache_config = types.CreateCachedContentConfig(
            system_instruction=system_instruction if system_instruction.strip() else None,
            contents=contents
        )
        
        client.caches.update(
            name=cache_name,
            config=cache_config
        )
        
        return jsonify({
            "success": True,
            "message": "Cache successfully updated"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/cache/clear/<cache_name>', methods=['DELETE'])
def clear_cache(cache_name):
    """Clear/delete a cached context."""
    if not client:
        return jsonify({"success": False, "error": "Google API client not configured"})
        
    try:
        client.caches.delete(cache_name)
        
        return jsonify({
            "success": True,
            "message": "Cache successfully cleared"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/count_tokens', methods=['POST'])
def count_tokens():
    """Count tokens for given content using Gemini API."""
    if not client:
        return jsonify({"error": "Google API client not configured."}), 500
    
    try:
        data = request.json
        text = data.get('text', '')
        include_history = data.get('include_history', False)
        
        if include_history:
            # Load current chat history
            chat_data = load_chat_history()
            history = []
            
            # Convert to genai format
            for turn in chat_data.get("turns", []):
                if turn['role'] == 'user':
                    history.append(types.UserContent(
                        parts=[types.Part(text=turn['parts'][0]['text'])]
                    ))
                elif turn['role'] == 'model':
                    history.append(types.Content(
                        role="model",
                        parts=[types.Part(text=turn['parts'][0]['text'])]
                    ))
            
            # Add new text if provided
            if text:
                history.append(types.UserContent(
                    parts=[types.Part(text=text)]
                ))
            
            # Count tokens for the entire history
            response = client.models.count_tokens(
                model="gemini-2.5-flash",
                contents=history
            )
            
            return jsonify({
                "success": True,
                "total_tokens": response.total_tokens
            })
        else:
            # Count tokens for just the provided text
            if not text:
                return jsonify({"error": "No text provided"}), 400
                
            response = client.models.count_tokens(
                model="gemini-2.5-flash",
                contents=[types.UserContent(parts=[types.Part(text=text)])]
            )
            
            return jsonify({
                "success": True,
                "total_tokens": response.total_tokens
            })
            
    except Exception as e:
        print(f"Error counting tokens: {e}")
        return jsonify({"error": str(e)}), 500

def format_text(text):
    """Format text with proper Unicode handling and markdown conversion."""
    try:
        if not text:
            return ""
            
        # Handle escaped Unicode characters first
        text = text.replace('\\u201c', '"')  # Left double quotation mark
        text = text.replace('\\u201d', '"')  # Right double quotation mark  
        text = text.replace('\\u2019', "'")  # Right single quotation mark
        text = text.replace('\\u2013', '–')  # En dash
        text = text.replace('\\u2014', '—')  # Em dash
        text = text.replace('\\u2026', '…')  # Horizontal ellipsis
        
        # Handle newlines and convert to markdown-style
        text = re.sub(r'\\n\\n\\n+', '\n\n---\n\n', text)  # Triple+ newlines become horizontal rules
        text = text.replace('\\n\\n', '\n\n')  # Double newlines become paragraph breaks
        text = text.replace('\\n', '\n')  # Single newlines
        
        # Convert markdown to HTML
        md = markdown.Markdown(extensions=['extra', 'codehilite'])
        html = md.convert(text)
        
        return html
    except Exception as e:
        print(f"Error formatting text: {e}")
        # Return text with basic HTML escaping as fallback
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Gemini Reader web app.')
    parser.add_argument('--file', default='anna-kendrick.json', help='The Google AI Studio JSON file to read.')
    args = parser.parse_args()

    # Make the path absolute and store it in the app config
    file_path = os.path.abspath(args.file)
    app.config['JSON_FILE'] = file_path

    print(f"Set JSON file to: {app.config['JSON_FILE']}")

    app.run(host='0.0.0.0', port=5000, debug=True)
