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


@app.get("/", response_class=HTMLResponse)
def upload_form():
    return """
    <html>
        <body>
            <h2>Upload Image</h2>
            <form action="/upload/" method="post" enctype="multipart/form-data">
                <input type="file" name="file">
                <input type="submit">
            </form>
        </body>
    </html>
    """


@app.post("/upload/", response_class=HTMLResponse)
async def upload_image(file: UploadFile = File(...)):
    # file_path = f"static/{file.filename}"
    file_path =  os.path.join(STATIC_DIR, "temp.jpg")                   

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    caption = generate_image_caption(file_path)
    print(file_path)    

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

