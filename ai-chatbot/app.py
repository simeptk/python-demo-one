from flask import Flask, render_template, request, session, redirect, url_for, send_file
from markupsafe import Markup
import google.generativeai as genai
import os
import requests
import json
from dotenv import load_dotenv
import markdown
from markdown.extensions.tables import TableExtension
import datetime
import glob
import uuid
import logging
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Required for session management

# Token configuration (from PowerShell script)
CLIENT_ID = os.getenv("FORDLLM_CLIENT_ID")
CLIENT_SECRET = os.getenv("FORDLLM_CLIENT_SECRET")
TENANT_ID = "c990bb7a-51f4-439b-bd36-9c07fb1041c0"
RESOURCE_APP_ID_URI = "api://6af47983-2540-43ae-89ff-4b93bf4eeb33"
TOKEN_CACHE_FILE = os.path.join(os.path.expanduser("~"), ".fordllm-token-cache.json")

# Create output directories if they don't exist
OUTPUT_DIR = 'responses'
TEMP_DIR = 'temp_responses'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Note: The following token functions are for Azure AD authentication
# These are currently not used as the app is using Google's Gemini API
# If implementing Ford LLM API integration, use these functions to get authentication tokens

# def get_new_token():
#     """Get a new token from Azure AD"""
#     logger.debug("Requesting new token...")
#     url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
    
#     body = {
#         "client_id": CLIENT_ID,
#         "scope": f"{RESOURCE_APP_ID_URI}/.default",
#         "client_secret": CLIENT_SECRET,
#         "grant_type": "client_credentials"
#     }
    
#     try:
#         response = requests.post(url, data=body)
#         response.raise_for_status()  # Raise exception for HTTP errors
        
#         data = response.json()
        
#         # Calculate token expiration time (subtract 5 minutes for safety margin)
#         expires_in = data['expires_in'] - 300
#         expiration_time = (datetime.datetime.now() + datetime.timedelta(seconds=expires_in)).isoformat()
        
#         # Store token and expiration time
#         token_cache = {
#             "AccessToken": data['access_token'],
#             "ExpiresAt": expiration_time
#         }
        
#         # Save to cache file
#         with open(TOKEN_CACHE_FILE, 'w') as f:
#             json.dump(token_cache, f)
        
#         logger.debug("New token generated and cached")
#         return data['access_token']
    
#     except Exception as e:
#         logger.error(f"Error occurred while generating token: {str(e)}")
#         return None

# def get_cached_token():
#     """Get token from cache if it's still valid"""
#     # Check if cache file exists
#     if os.path.exists(TOKEN_CACHE_FILE):
#         try:
#             # Read cached token
#             with open(TOKEN_CACHE_FILE, 'r') as f:
#                 token_cache = json.load(f)
            
#             # Convert string to datetime and ensure both are timezone-naive
#             expires_at = datetime.datetime.fromisoformat(token_cache["ExpiresAt"])
#             # Remove timezone information if present
#             if hasattr(expires_at, 'tzinfo') and expires_at.tzinfo is not None:
#                 # Convert to naive datetime in local time
#                 expires_at = expires_at.replace(tzinfo=None)
            
#             current_time = datetime.datetime.now()
            
#             if current_time < expires_at:
#                 logger.debug("Using existing valid token")
#                 return token_cache["AccessToken"]
#             else:
#                 logger.debug("Cached token has expired")
#                 return None
        
#         except Exception as e:
#             logger.error(f"Error reading cached token: {str(e)}")
#             return None
#     else:
#         logger.debug("No cached token found")
#         return None

# def get_token():
#     """Get a valid token, either from cache or by requesting a new one"""
#     token = get_cached_token()
    
#     if not token:
#         token = get_new_token()
    
#     return token

def get_next_filename():
    """Generate a sequential filename like output-001.md, output-002.md, etc."""
    existing_files = glob.glob(os.path.join(OUTPUT_DIR, 'output-*.md'))
    if not existing_files:
        return 'output-001.md'
    
    # Extract numbers from existing filenames and find the highest one
    numbers = [int(os.path.basename(f).split('-')[1].split('.')[0]) for f in existing_files]
    next_number = max(numbers) + 1 if numbers else 1
    return f'output-{next_number:03d}.md'

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']
        response = generate_content(user_input)
        
        # Store response in temporary file instead of session
        temp_id = str(uuid.uuid4())
        temp_file = os.path.join(TEMP_DIR, f"{temp_id}.md")
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(f"QUERY: {user_input}\n\nRESPONSE: {response}")
        
        # Store only the ID in session
        session['temp_response_id'] = temp_id
        
        # Convert markdown to HTML with extended support for tables and other formatting
        formatted_response = Markup(markdown.markdown(
            response, 
            extensions=['tables', 'fenced_code', 'nl2br', 'sane_lists']
        ))
        return render_template('index.html', user_input=user_input, ai_response=formatted_response)
    return render_template('index.html')

@app.route("/save_response", methods=['POST'])
def save_response():
    """Save the last response as a markdown file"""
    try:
        if 'temp_response_id' not in session:
            logger.error("No temp_response_id in session")
            return redirect(url_for('index'))
        
        temp_id = session['temp_response_id']
        temp_file = os.path.join(TEMP_DIR, f"{temp_id}.md")
        
        if not os.path.exists(temp_file):
            logger.error(f"Temp file not found: {temp_file}")
            return redirect(url_for('index'))
        
        # Read the temporary file
        with open(temp_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract query and response
        parts = content.split("\n\nRESPONSE: ", 1)
        query = parts[0].replace("QUERY: ", "")
        response = parts[1] if len(parts) > 1 else "No response found"
        
        # Generate filename
        filename = get_next_filename()
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # Create markdown content with timestamp, query and response
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        md_content = f"""# AI Response - {timestamp}

## Query
{query}

## Response
{response}
"""
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.debug(f"Saved response to file: {filepath}")
        
        # Clean up temp file
        os.remove(temp_file)
        
        # Return the saved file for download
        return send_file(filepath, as_attachment=True, download_name=filename)
    
    except Exception as e:
        logger.exception(f"Error saving response: {str(e)}")
        return f"Error saving response: {str(e)}", 500

def generate_content(prompt):
    try:
        # Use the Google API key for Gemini, not the Ford token
        # The Ford token is for a different API and won't work with Gemini
        google_api_key = os.getenv("GEMINI_API_KEY")
        
        if not google_api_key:
            logger.error("GEMINI_API_KEY not found in environment variables")
            return "Error: Missing GEMINI_API_KEY in environment variables."
        
        # Configure Gemini with the proper API key
        genai.configure(api_key=google_api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Enhanced prompt with detailed formatting instructions
        structured_prompt = f"""
        {prompt}
        
        Please format your response with clear, structured Markdown:
        
        1. For any lists, use proper Markdown bullet points or numbered lists with spacing
        2. For any tabular data, use proper Markdown table format:
           | Header1 | Header2 | Header3 |
           |---------|---------|---------|
           | Value1  | Value2  | Value3  |
        3. Use headings (## or ###) for different sections
        4. Use **bold** and *italic* for emphasis where appropriate
        5. Ensure there's blank lines between sections for readability
        """
        response = model.generate_content(structured_prompt)
        return response.text
    except Exception as e:
        logger.exception(f"Error generating content: {str(e)}")
        return f"Error generating content: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)