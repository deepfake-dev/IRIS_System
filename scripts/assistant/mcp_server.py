import os
import sys
import time
import socket
import subprocess
import webbrowser
from google import genai
from google.genai import types
from mcp.server.fastmcp import FastMCP

# Initialize the MCP Server
mcp = FastMCP("Iris_Agent_Tools")

# [CLEANUP] Pull API key securely from environment variables
api_key = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
client = genai.Client(api_key=api_key)

def is_port_in_use(port: int) -> bool:
    """Helper function to check if the server is already running."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

@mcp.tool()
def open_deepfake_detector() -> str:
    """
    CRITICAL TOOL: Opens the university's deepfake detection and media provenance tool 
    directly on the user's screen. Call this when the user asks to check or verify a video.
    """
    port = int(os.environ.get("PROVENANCE_PORT", 4321))
    local_ip = str(os.environ.get("LOCAL_IP", "localhost"))

    env = os.environ.copy()
    env["PROVENANCE_TIMEOUT"] = "90"
    
    # Using localhost ensures the self-signed cert.pem works without severe browser warnings
    url = f"https://{local_ip}:{port}"
    
    if not is_port_in_use(port):
        print(f"\n[MCP] Port {port} is free. Spawning Provenance Server on-demand...")
        
        # 1. Get exact absolute paths to avoid working directory confusion
        app_dir = os.path.abspath(os.path.join(os.getcwd(), 'scripts', 'provenance_checker'))
        
        # [CLEANUP] Use sys.executable to run uvicorn. This is bulletproof 
        # on Windows, Mac, and Linux, regardless of virtual env folder names.
        cmd = [
            sys.executable, "-m", "uvicorn", "server:app",
            "--host", "0.0.0.0",
            "--port", str(port),
            "--app-dir", app_dir,
            "--ssl-keyfile", "key.pem",
            "--ssl-certfile", "cert.pem"
        ]
        
        try:
            # 3. Spawn the server in a new console
            # Note: creationflags=subprocess.CREATE_NEW_CONSOLE is Windows-only.
            subprocess.Popen(
                cmd, 
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                env=env
            )
            print("[MCP] Server spawned in a new window. Waiting for it to come online...")
            
            # 4. Smart Polling Loop (Waits up to 30 seconds for the server to bind)
            max_retries = 30
            server_ready = False
            for i in range(max_retries):
                if is_port_in_use(port):
                    print(f"[MCP] Server detected on port {port} after {i} seconds!")
                    server_ready = True
                    time.sleep(1) # Give Uvicorn 1 extra second to fully initialize HTTP routes
                    break
                time.sleep(1)
                
            if not server_ready:
                return "I tried to launch the deepfake detector, but it took longer than 30 seconds to start. Please check the console for errors."
                
        except Exception as e:
            print(f"[MCP] Failed to spawn server: {e}")
            return f"I encountered an error trying to launch the deepfake detector: {e}"
    else:
        print(f"\n[MCP] Provenance Server is already running on port {port}.")
    
    return (
        f"Deepfake Detector launched securely. "
        f"CRITICAL INSTRUCTION FOR IRIS: You MUST include the exact raw link '{url}' "
        f"in your final spoken response. Do not paraphrase or omit the 'https://' link. "
        f"If you do not include the exact URL in your text, the QR code UI will fail to trigger."
    )

# [CLEANUP] Removed the duplicate get_kiosk_location block
@mcp.tool()
def get_kiosk_location(current_location: str) -> str:
    """Use this to tell the user where they currently are on campus."""
    return f"The user is currently at the following kiosk location: {current_location}"

@mcp.tool()
def search_university_news() -> str:
    """Fetches the latest news, announcements, and events for Batangas State University."""
    # Replace this stub with an actual web scraper or RSS feed parser
    return (
        "Latest News: BatStateU Alangilan launches new AI-powered IRIS kiosks. "
        "The university is also hosting a robotics symposium next week."
    )

@mcp.tool()
def ask_gemini(query: str) -> str:
    """
    CRITICAL TOOL: Use this to search for general knowledge, world facts, 
    math, coding, or ANY question that cannot be answered by the campus database.
    """
    print(f"\n[MCP] Reaching out to Gemini for: {query}")
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash', # [NOTE] Updated model name to standard identifier
            contents=types.Part.from_text(text=query),
            config=types.GenerateContentConfig(
                temperature=0,
                top_p=0.95,
                top_k=20,
            ),
        )
        return response.text
    except Exception as e:
        return f"I'm sorry, I couldn't reach my cloud brain right now. Error: {str(e)}"

if __name__ == "__main__":
    # Runs the server using standard input/output for the MCP protocol
    mcp.run()