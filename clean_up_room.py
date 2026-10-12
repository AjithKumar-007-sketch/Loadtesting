import asyncio
from typing import AsyncContextManager
import aiohttp
from livekit import api
from livekit.api import TwirpError
from livekit.protocol.egress import (
    EgressInfo,
    EncodingOptionsPreset,
    ListEgressRequest,
    EncodedFileType,
    S3Upload,
    StopEgressRequest,
    GCPUpload,
)
from livekit.protocol.room import (
    ListRoomsRequest,
    ListRoomsResponse,
    DeleteRoomRequest,
    DeleteRoomResponse,
)
from livekit.protocol.sip import (
    CreateSIPParticipantRequest,
    SIPParticipantInfo,
    CreateSIPOutboundTrunkRequest,
    SIPOutboundTrunkInfo,
)


class LivekitClientManager:
    """Async context manager for Twilio HTTP client"""

    def __init__(self):
        self.livekit_api = api.LiveKitAPI(
            timeout=aiohttp.ClientTimeout(total=30),
        )

    async def __aenter__(self):
        return self.livekit_api

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.livekit_api:
            await self.livekit_api.aclose()


class LivekitService:
    @staticmethod
    def get_client() -> AsyncContextManager[api.LiveKitAPI]:
        """Get a Twilio client instance with automatic cleanup"""
        return LivekitClientManager()

    @staticmethod
    async def end_call(
        room_name: str,
    ):
        try:
            async with LivekitService.get_client() as livekit_api:
                response: DeleteRoomResponse = await livekit_api.room.delete_room(
                    DeleteRoomRequest(room=room_name)
                )

                return response
        except TwirpError as e:
            te: TwirpError = e
            if te.code == "not_found":
                return None
        except Exception as e:
            print(f"An error occurred while ending a call: {e}")

    @staticmethod
    async def list_rooms() -> ListRoomsResponse:
        async with LivekitService.get_client() as livekit_api:
            rooms = await livekit_api.room.list_rooms(ListRoomsRequest())
            return rooms


async def main():
    rooms = await LivekitService.list_rooms()
    print("Rooms: ", len(rooms.rooms))
    for room in rooms.rooms:
        print("Room: ", room.name)
        await LivekitService.end_call(room.name)


if __name__ == "__main__":
    asyncio.run(main())
