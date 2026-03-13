from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
from app.inference_service import generate_image_caption

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "app", "static")

# Create folder to store images
os.makedirs("app/static", exist_ok=True)

# Serve static files (images)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Simple and clean HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Image Caption Generator</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: system-ui, -apple-system, sans-serif; }
        body { background: #f5f7fa; min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 20px; }
        .card { max-width: 500px; width: 100%; background: white; border-radius: 12px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); padding: 30px; }
        h2 { color: #1e293b; margin-bottom: 20px; text-align: center; font-weight: 500; }
        .upload-box { border: 2px dashed #cbd5e1; border-radius: 8px; padding: 30px; text-align: center; background: #f8fafc; }
        .upload-box:hover { border-color: #3b82f6; background: #eff6ff; }
        input[type=file] { margin: 20px 0; width: 100%; padding: 10px; border: 1px solid #e2e8f0; border-radius: 6px; }
        button { background: #1e293b; color: white; border: none; padding: 12px 30px; border-radius: 6px; font-size: 16px; cursor: pointer; width: 100%; }
        button:hover { background: #0f172a; }
        img { max-width: 100%; max-height: 300px; border-radius: 8px; margin: 20px 0; display: block; }
        .caption { background: #f8fafc; padding: 20px; border-radius: 8px; border-left: 4px solid #1e293b; margin: 20px 0; font-size: 18px; line-height: 1.6; }
        a { display: inline-block; color: #1e293b; text-decoration: none; padding: 10px 20px; border: 1px solid #e2e8f0; border-radius: 6px; margin-top: 15px; }
        a:hover { background: #f8fafc; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="card">
        <h2>Image Caption Generator</h2>
        
        <!-- Upload Form -->
        <div id="uploadForm">
            <div class="upload-box">
                <p style="font-size: 48px; margin-bottom: 10px;">📷</p>
                <p style="color: #64748b; margin-bottom: 15px;">JPG, PNG, GIF (max 10MB)</p>
                <form action="/upload/" method="post" enctype="multipart/form-data" id="form">
                    <input type="file" name="file" accept="image/*" required id="fileInput">
                    <button type="submit">Generate Caption</button>
                </form>
            </div>
        </div>
        
        <!-- Result (hidden initially) -->
        <div id="result" class="hidden">
            <img src="" alt="Uploaded" id="resultImage">
            <div class="caption" id="captionText"></div>
            <a href="/" onclick="window.location.reload()">← Upload Another</a>
        </div>
    </div>

    <script>
        document.getElementById('form').onsubmit = async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            
            // Show loading
            document.getElementById('uploadForm').style.display = 'none';
            document.querySelector('.card').innerHTML += '<p style="text-align:center;">Processing...</p>';
            
            try {
                const res = await fetch('/upload/', { method: 'POST', body: formData });
                const html = await res.text();
                const parser = new DOMParser();
                const doc = parser.parseFromString(html, 'text/html');
                
                document.querySelector('.card').innerHTML = `
                    <h2>Generated Caption</h2>
                    <img src="${doc.querySelector('img').src}" style="max-width:100%; border-radius:8px; margin:20px 0;">
                    <div class="caption">${doc.querySelector('p').textContent}</div>
                    <a href="/" onclick="window.location.reload()">← Upload Another</a>
                `;
            } catch (error) {
                alert('Error processing image');
                window.location.reload();
            }
        };
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def upload_form():
    return HTML_TEMPLATE

@app.post("/upload/", response_class=HTMLResponse)
async def upload_image(file: UploadFile = File(...)):
    file_path = os.path.join(STATIC_DIR, "temp.jpg")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    caption = generate_image_caption(file_path)
    print(file_path)
    
    # Simple result HTML
    return f"""
    <html>
        <body>
            <h2>Uploaded Image:</h2>
            <img src="/static/temp.jpg" width="300">
            <h3>Caption:</h3>
            <p>{caption}</p>
            <br><br>
            <a href="/">Upload another image</a>
        </body>
    </html>
    """