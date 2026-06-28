import os
import sys
import base64
from email.mime.text import MIMEText
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("Please install mcp SDK: pip install mcp", file=sys.stderr)
    sys.exit(1)

# Initialize FastMCP Server
mcp = FastMCP("Gmail Local Server")

# Scopes required to send emails
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def get_gmail_service():
    creds = None
    # Token file stores the user's access and refresh tokens
    token_path = os.environ.get("GMAIL_TOKEN_PATH", "token.json")
    creds_path = os.environ.get("GMAIL_CREDENTIALS_PATH", "credentials.json")

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(creds_path):
                raise FileNotFoundError(
                    f"Credentials file not found at '{creds_path}'. "
                    "Please download OAuth credentials JSON from Google Cloud Console "
                    "and place it at this path or configure GMAIL_CREDENTIALS_PATH."
                )
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save credentials for the next run
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
            
    return build('gmail', 'v1', credentials=creds)

@mcp.tool()
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email using Gmail API.
    
    Args:
        to (str): Recipient email address.
        subject (str): Email subject line.
        body (str): Email body text content.
    """
    try:
        service = get_gmail_service()
        message = MIMEText(body)
        message['to'] = to
        message['subject'] = subject
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        
        send_result = service.users().messages().send(userId="me", body={"raw": raw}).execute()
        return f"Email sent successfully. Message ID: {send_result.get('id')}"
    except Exception as e:
        return f"Failed to send email: {str(e)}"

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "auth":
        print("Starting Gmail authentication flow...")
        service = get_gmail_service()
        print("Authentication successful! token.json has been created.")
    else:
        mcp.run()
