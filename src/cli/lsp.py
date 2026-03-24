import json
import logging
import sys

logger = logging.getLogger(__name__)

def read_message():
    """Read a JSON-RPC message from stdin."""
    line = sys.stdin.readline()
    if not line:
        return None
        
    if not line.startswith("Content-Length: "):
        return None
        
    content_length = int(line[16:].strip())
    
    # Read the empty line
    sys.stdin.readline()
    
    # Read the body
    body = sys.stdin.read(content_length)
    return json.loads(body)

def write_message(msg):
    """Write a JSON-RPC message to stdout."""
    body = json.dumps(msg)
    sys.stdout.write(f"Content-Length: {len(body)}\\r\\n\\r\\n{body}")
    sys.stdout.flush()

def handle_initialize(msg_id):
    """Handle the initialize request."""
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": {
            "capabilities": {
                "textDocumentSync": 1, # Full sync
                "completionProvider": {
                    "resolveProvider": False,
                    "triggerCharacters": ["."]
                },
                "hoverProvider": True
            },
            "serverInfo": {
                "name": "gnn-lsp",
                "version": "1.0.0"
            }
        }
    }

def handle_hover(msg_id, params):
    """Handle the textDocument/hover request."""
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": {
            "contents": {
                "kind": "markdown",
                "value": "**GNN Identifier**\\n\\nGeneralized Notation Notation construct."
            }
        }
    }

def publish_diagnostics(uri, text):
    """Run basic validation and publish diagnostics."""
    diagnostics = []
    
    # Simple syntax check: look for missing closing braces
    if "{" in text and "}" not in text:
        diagnostics.append({
            "range": {
                "start": {"line": 0, "character": 0},
                "end": {"line": 0, "character": 100}
            },
            "severity": 1, # Error
            "message": "Missing closing brace '}'"
        })
        
    write_message({
        "jsonrpc": "2.0",
        "method": "textDocument/publishDiagnostics",
        "params": {
            "uri": uri,
            "diagnostics": diagnostics
        }
    })

def start_lsp():
    """Start the Language Server Protocol loop on stdin/stdout."""
    # Setup simple logging to a file to avoid corrupting stdout
    logging.basicConfig(filename='gnn-lsp.log', level=logging.INFO)
    logger.info("Starting GNN LSP Server...")
    
    while True:
        try:
            msg = read_message()
            if not msg:
                break
                
            logger.info(f"Received: {msg.get('method')}")
            
            method = msg.get("method")
            msg_id = msg.get("id")
            
            if method == "initialize":
                write_message(handle_initialize(msg_id))
            elif method == "initialized":
                pass
            elif method == "textDocument/hover":
                write_message(handle_hover(msg_id, msg.get("params")))
            elif method == "textDocument/didOpen":
                params = msg.get("params", {})
                doc = params.get("textDocument", {})
                uri = doc.get("uri", "")
                text = doc.get("text", "")
                if uri and text:
                    publish_diagnostics(uri, text)
            elif method == "textDocument/didChange":
                params = msg.get("params", {})
                doc = params.get("textDocument", {})
                uri = doc.get("uri", "")
                changes = params.get("contentChanges", [])
                if uri and changes:
                    # Sync sends full text
                    text = changes[0].get("text", "")
                    publish_diagnostics(uri, text)
            elif method == "shutdown":
                write_message({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": None
                })
            elif method == "exit":
                break
            else:
                # Ignore unhandled notifications
                if msg_id is not None:
                    # Return method not found if it is a request
                    write_message({
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {
                            "code": -32601,
                            "message": "Method not found"
                        }
                    })
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            break
            
    logger.info("GNN LSP Server shutting down.")
