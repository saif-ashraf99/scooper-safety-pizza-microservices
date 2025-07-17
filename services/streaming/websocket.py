import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from connection_manager import manager

router = APIRouter()

@router.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # keep-alive
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)