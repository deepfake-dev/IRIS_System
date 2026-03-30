import http.server
import ssl
import sys
import os

port = int(sys.argv[1]) if len(sys.argv) > 1 else 5050
directory = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()

os.chdir(directory)
server_address = ('0.0.0.0', port)
httpd = http.server.HTTPServer(server_address, http.server.SimpleHTTPRequestHandler)

# Wrap the server in your new SSL certificates!
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(certfile="../../cert.pem", keyfile="../../key.pem")
httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

print(f"🔒 Secure KIOSK running on https://0.0.0.0:{port}")
httpd.serve_forever()