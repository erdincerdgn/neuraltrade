# Video Thumbnail Generation

The reel feature includes **optional** automatic thumbnail generation from uploaded videos.

## Deployment Options

### Option 1: No FFmpeg (Recommended for Production)
Thumbnails are optional. The app works perfectly without FFmpeg:
- Video uploads succeed without thumbnails
- Frontend can use a default thumbnail or show video preview
- **No server dependencies needed** ✅

### Option 2: Docker Deployment (Recommended)
Use a Docker image with FFmpeg pre-installed:

```dockerfile
FROM node:20-alpine

# Install FFmpeg in Docker
RUN apk add --no-cache ffmpeg

WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

CMD ["npm", "run", "start:prod"]
```

### Option 3: Cloud Platform with FFmpeg
Some cloud platforms include FFmpeg:
- **Heroku**: Use buildpack `https://github.com/jonathanong/heroku-buildpack-ffmpeg-latest`
- **Railway/Render**: Use Docker deployment
- **AWS EC2/ECS**: Install FFmpeg in AMI or container

### Option 4: Local Development Only
For Windows development:
```powershell
# Already installed at C:\ffmpeg\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe
ffmpeg -version
```

## Linux Server Installation

If deploying to a Linux server without Docker:

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y ffmpeg

# CentOS/RHEL
sudo yum install -y ffmpeg

# Verify
ffmpeg -version
```

## How It Works

When a user uploads a reel video:
1. The video is saved temporarily
2. FFmpeg extracts a frame at 1 second
3. The frame is converted to WebP format (720x1280)
4. Both video and thumbnail are uploaded to S3
5. The thumbnail URL is automatically saved in the database

## API Response

When uploading a video to `/file-upload?type=reel-videos`, the response will include:
```json
{
  "url": "https://bucket.s3.region.amazonaws.com/reel-videos/video.mp4",
  "thumbnailUrl": "https://bucket.s3.region.amazonaws.com/reel-videos/thumbnail.webp"
}
```

Use both URLs when creating a reel:
```json
{
  "videoUrl": "...",
  "thumbnailUrl": "...",
  "caption": "My awesome reel!"
}
```

If thumbnail generation fails, the video will still be uploaded successfully (thumbnailUrl will be undefined).
