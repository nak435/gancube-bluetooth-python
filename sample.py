# Required dependencies
import asyncio

from gan_smart_cube import (
  connect_gan_cube,
  GanCubeEvent
)

def handle_cube_event(event: GanCubeEvent):
    """Handle all cube events"""
    print("GanCubeEvent", event)
    return

async def on_connect_cube():
    try:
        conn = await connect_gan_cube()
        conn.events.subscribe(handle_cube_event)
        await conn.sendCubeCommand({"type": "REQUEST_HARDWARE"})
        await conn.sendCubeCommand({"type": "REQUEST_FACELETS"})
        await conn.sendCubeCommand({"type": "REQUEST_BATTERY"})
        while True:
            await asyncio.sleep(1)

    except Exception as e:
        print(f"Connection failed: {e}")
        #import traceback
        #print(traceback.format_exc())

if __name__ == '__main__':
    asyncio.run(on_connect_cube())
