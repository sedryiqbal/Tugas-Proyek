import cv2
import base64
import asyncio
import websockets
import json

async def stream_video():
    cap = cv2.VideoCapture(0)
    
    async def send_frame(websocket):
        while True:
            ret, frame = cap.read()
            # Proses face recognition
            # Convert frame ke base64 untuk dikirim
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            data = {
                'frame': frame_base64,
                'recognition_data': {
                    'name': 'nama_terdeteksi',
                    'confidence': 0.95
                }
            }
            await websocket.send(json.dumps(data))
            await asyncio.sleep(0.033)  # ~30 FPS