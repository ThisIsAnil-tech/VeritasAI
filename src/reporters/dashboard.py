import os
import http.server
import socketserver
import webbrowser
import threading
import socket
from pathlib import Path
from loguru import logger

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        # Redirect standard http logs to loguru
        logger.info(f"Dashboard server request: {format % args}")

def find_free_port(start_port: int = 8000) -> int:
    """Find an available port starting from start_port."""
    port = start_port
    while port < 9000:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
        port += 1
    return start_port

def serve_dashboard(directory: str = "./reports", port: int = 8000, open_browser: bool = True):
    """
    Start a local HTTP server serving files in directory and open latest report in browser.
    """
    reports_dir = Path(directory).resolve()
    if not reports_dir.exists():
        logger.warning(f"Reports directory {reports_dir} does not exist. Creating it.")
        reports_dir.mkdir(parents=True, exist_ok=True)

    # Change working directory of the server to the reports directory
    os.chdir(reports_dir)
    
    actual_port = find_free_port(port)
    server_address = ('', actual_port)

    # Locate latest HTML report
    html_files = sorted(reports_dir.glob("*.html"))
    latest_report_file = html_files[-1].name if html_files else ""

    url = f"http://localhost:{actual_port}/"
    if latest_report_file:
        url += latest_report_file

    logger.info(f"Starting local dashboard server at: {url}")
    
    # Configure socket server to allow port re-use
    socketserver.TCPServer.allow_reuse_address = True
    
    try:
        with socketserver.TCPServer(server_address, Handler) as httpd:
            if open_browser:
                # Open browser in a separate thread to prevent blocking startup
                def _open():
                    logger.info("Opening browser window...")
                    webbrowser.open(url)
                threading.Thread(target=_open, daemon=True).start()
                
            logger.info("Press Ctrl+C to terminate the dashboard server.")
            httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Dashboard server stopped by user.")
    except Exception as e:
        logger.error(f"Dashboard server failed to run: {str(e)}")
