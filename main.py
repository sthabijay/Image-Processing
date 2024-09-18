from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.staticfiles import StaticFiles
import os
import shutil
from pydantic import BaseModel
from typing import Optional
import cv2 
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from jose import JWTError, jwt

def getMetadata(path: str):
    image_path = path
    image = Image.open("./uploaded_images/jpt.jpg")

    # Display basic image metadata
    print(f"Image Format: {image.format}")
    print(f"Image Size: {image.size}")
    print(f"Image Mode: {image.mode}")

    # Try to get EXIF metadata (if available)
    exif_data = image._getexif()
    if exif_data is not None:
        print("\nEXIF Metadata:")
        for tag_id, value in exif_data.items():
            # Get the tag name
            tag = TAGS.get(tag_id, tag_id)
            print(f"{tag}: {value}")
    else:
        print("No EXIF metadata found.")

app = FastAPI()

SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

fake_users_db = {"ram": "asdk"}

UPLOAD_FOLDER = "uploaded_images"

# Create the folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Mount the static files for the uploaded images
app.mount("/static/images", StaticFiles(directory=UPLOAD_FOLDER), name="images")

@app.get("/")
def root():
    return {"message": "image processing fastapi practice"}

class User(BaseModel):
    username: str
    disabled: Optional[bool] = None


class UserInDB(User):
    hashed_password: str


class UserOut(BaseModel):
    username: str
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


# Utility to hash passwords
def get_password_hash(password: str):
    return pwd_context.hash(password)


# Utility to verify password
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


# Utility to create JWT access tokens
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# Utility to retrieve user from database
def get_user(username: str):
    user = fake_users_db.get(username)
    if user:
        return UserInDB(**user)


# Utility to authenticate user
def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


# Dependency to get the current user from JWT token
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(username=username)
    if user is None:
        raise credentials_exception
    return user


@app.post("/register", response_model=UserOut)
async def register(username: str, password: str):
    if username in fake_users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )
    
    # Hash the password and store the user in the fake DB
    hashed_password = get_password_hash(password)
    fake_users_db[username] = {
        "username": username,
        "hashed_password": hashed_password,
        "disabled": False
    }

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": username}, expires_delta=access_token_expires)

    return {"username": username, "access_token": access_token, "token_type": "bearer"}

@app.post("/login", response_model=UserOut)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create JWT token for the authenticated user
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)

    return {"username": user.username, "access_token": access_token, "token_type": "bearer"}

@app.post("/images")
async def images(file: UploadFile = File(...)):
    
    file_location = f"{UPLOAD_FOLDER}/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Create the URL for the uploaded image
    file_url = f"http://127.0.0.1:8000/static/images/{file.filename}"
    
    return {
        "fileURL": file_url,
        "metadata": file.file
    }    

# Define the Resize model
class ResizeModel(BaseModel):
    width: int = 300
    height: int = 300

# Define the Crop model
class CropModel(BaseModel):
    width: int = 0
    height: int = 0
    x: int = 200
    y: int = 200

# Define the Filters model
class FiltersModel(BaseModel):
    grayscale: bool = False
    sepia: bool = True

# Define the main Transformations model
class TransformationsModel(BaseModel):
    resize: Optional[ResizeModel]  # Optional if the user may not provide it
    crop: Optional[CropModel]      # Optional as well
    rotate: Optional[int]        # Optional with rotation as a number (degrees)
    format: Optional[str]          # Optional image format (e.g., "png", "jpg")
    filters: Optional[FiltersModel]# Optional filters (grayscale and sepia)

@app.post("/image/{id}/transform")
def transform(transform: TransformationsModel, id: str):    

    image = cv2.imread(f"./uploaded_images/{id}")
    print(image.all)
    
    resize = transform.resize 
    crop = transform.crop
    rotate = transform.rotate
    image_format = transform.format
    filters = transform.filters  
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #resize
    transformedImage = cv2.resize(image_rgb, (resize.width, resize.height))
    
    #crop
    transformedImage = transformedImage[crop.width: crop.x, crop.height: crop.y]
    
    #rotate
    center = (image_rgb.shape[1] // 2, image_rgb.shape[0] // 2)
    angle = rotate
    scale = 1

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    transformedImage = cv2.warpAffine(image_rgb, rotation_matrix, (transformedImage.shape[1], transformedImage.shape[0]))
    
    #filter
    if(filters.grayscale):
        transformedImage = cv2.cvtColor(transformedImage, cv2.COLOR_RGB2GRAY)
    
    if(filters.sepia):
        transformedImage = np.array(transformedImage, dtype=np.float32) / 255.0

        # Define the sepia filter matrix
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                [0.349, 0.686, 0.168],
                                [0.393, 0.769, 0.189]])

        # Apply the sepia filter
        sepia_image = cv2.transform(transformedImage, sepia_filter)

        # Clip values to be between 0 and 1, then convert back to uint8
        sepia_image = np.clip(sepia_image, 0, 1)
        sepia_image = (sepia_image * 255).astype(np.uint8)
        transformedImage = sepia_image
    
    #uploaded
    cv2.imwrite(f"./uploaded_images/{id}", transformedImage)

    file_url = f"http://127.0.0.1:8000/static/images/{id}"   
    
    return{
        "fileURL": file_url,
    }
    
@app.get("/images/{id}")
def getImage(id: str):   
    file_url = f"http://127.0.0.1:8000/static/images/{id}"
    
    return{
        "fileURL": file_url,
    }

@app.get("/images")  
def getImages(pages: int=1, limit: int=10):
    dir = "./uploaded_images"
    files_scan = os.listdir(dir)

    files = ([f for f in files_scan if os.path.isfile(os.path.join(dir, f))])
    file_count = len(files) 
    
    if(file_count < limit):
        limit = file_count
        
    response = {}
    
    for file in files[pages-1:limit]:
         response.update({file: f"http://127.0.0.1:8000/static/images/{file}"})
    
    return response