from typing import TypedDict, Union, Optional, List, Any, Protocol, Callable
from abc import ABC, abstractmethod
import asyncio
import struct
from dataclasses import dataclass
from gan_utils import now, toKociembaFacelets
from gan_cube_encrypter import GanCubeEncrypter

# Import for Observable-like functionality

class Subject:
  operators = []
  subs = []
  cachedItems = []
  cachedNum = 0
  completed = False

  '''
  __init__
  '''
  def __init__(self, *, cached = 0, init = []):
    if (cached > 0):
      self.cachedNum = cached
      if (len(init)):
        self.cachedItems = [*init[-cached:]]

  '''
  pipe
  '''
  def pipe(self, *ops):
    self.operators.extend(ops)
    return self

  '''
  next
  '''
  def next(self, value):
    if self.completed:
      return

    self.__putInCache(value)
    value = self.__applyOperators(value)

    for sub in self.subs:
      if isinstance(sub, type({})):
        if 'next' in sub:
          sub['next'](value)
      else:
        sub(value)

  '''
  complete
  '''
  def complete(self):
    if self.completed:
      return

    self.completed = True
    for sub in self.subs:
      if isinstance(sub, type({})):
        if 'complete' in sub:
          sub['complete']()

  '''
  error
  '''
  def error(self, message):
    if self.completed:
      return

    errorHandlers = 0

    for sub in self.subs:
      if isinstance(sub, type({})):
        if 'error' in sub:
          errorHandlers += 1
          sub['error'](message)

    if (errorHandlers == 0):
      raise Exception(message)

  '''
  subscribe
  '''
  def subscribe(self, fn):
    self.subs.append(fn)

    if (self.cachedNum > 0 and len(self.cachedItems)):
      for cachedVal in self.cachedItems:
        self.next(cachedVal)

  '''
  __applyOperators
  '''
  def __applyOperators(self, value):
    if (len(self.operators)):
      for op in self.operators:
        value = op(value)

    return value

  '''
  __putInCache
  '''
  def __putInCache(self, value):
    if (len(self.cachedItems) < self.cachedNum):
      self.cachedItems.append(value)
    else:
      self.cachedItems = self.cachedItems[1:] + [value]

# -----------------

from threading import Event
import weakref
from bleak import BleakClient
from bleak.backends.characteristic import BleakGATTCharacteristic


# Command for requesting information about GAN Smart Cube hardware
class GanCubeReqHardwareCommand(TypedDict):
    type: str  # "REQUEST_HARDWARE"

# Command for requesting information about current facelets state
class GanCubeReqFaceletsCommand(TypedDict):
    type: str  # "REQUEST_FACELETS"

# Command for requesting information about current battery level
class GanCubeReqBatteryCommand(TypedDict):
    type: str  # "REQUEST_BATTERY"

# Command for resetting GAN Smart Cube internal facelets state to solved state
class GanCubeReqResetCommand(TypedDict):
    type: str  # "REQUEST_RESET"

# Command message
GanCubeCommand = Union[GanCubeReqHardwareCommand, GanCubeReqFaceletsCommand, GanCubeReqBatteryCommand, GanCubeReqResetCommand]

# Representation of GAN Smart Cube move
class GanCubeMove(TypedDict):
    face: int  # Face: 0 - U, 1 - R, 2 - F, 3 - D, 4 - L, 5 - B
    direction: int  # Face direction: 0 - CW, 1 - CCW
    move: str  # Cube move in common string notation, like R' or U
    localTimestamp: Optional[int]  # Timestamp according to host device clock, null in case if bluetooth event was missed and recovered
    cubeTimestamp: Optional[int]  # Timestamp according to cube internal clock, for some cube models may be null in case if bluetooth event was missed and recovered

# Move event
class GanCubeMoveEvent(GanCubeMove):
    type: str  # "MOVE"
    serial: int  # Serial number, value range 0-255, increased in a circle on each facelets state change

# Representation of GAN Smart Cube facelets state
class GanCubeState(TypedDict):
    CP: List[int]  # Corner Permutation: 8 elements, values from 0 to 7
    CO: List[int]  # Corner Orientation: 8 elements, values from 0 to 2
    EP: List[int]  # Edge Permutation: 12 elements, values from 0 to 11
    EO: List[int]  # Edge Orientation: 12 elements, values from 0 to 1

# Facelets event
class GanCubeFaceletsEvent(TypedDict):
    type: str  # "FACELETS"
    serial: int  # Serial number, value range 0-255, increased in a circle on each facelets state change
    facelets: str  # Cube facelets state in the Kociemba notation like "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
    state: GanCubeState  # Cube state representing corners and edges orientation and permutation

# Quaternion to represent orientation
class GanCubeOrientationQuaternion(TypedDict):
    x: float
    y: float
    z: float
    w: float

# Representation of angular velocity by axes
class GanCubeAngularVelocity(TypedDict):
    x: float
    y: float
    z: float

# Gyroscope event
class GanCubeGyroEvent(TypedDict):
    type: str  # "GYRO"
    quaternion: GanCubeOrientationQuaternion  # Cube orientation quaternion, uses Right-Handed coordinate system, +X - Red, +Y - Blue, +Z - White
    velocity: Optional[GanCubeAngularVelocity]  # Cube angular velocity over current ODR time frame

# Battery event
class GanCubeBatteryEvent(TypedDict):
    type: str  # "BATTERY"
    batteryLevel: int  # Current battery level in percent

# Hardware event
class GanCubeHardwareEvent(TypedDict):
    type: str  # "HARDWARE"
    hardwareName: Optional[str]  # Internal cube hardware device model name
    softwareVersion: Optional[str]  # Software/Firmware version of the cube
    hardwareVersion: Optional[str]  # Hardware version of the cube
    productDate: Optional[str]  # Production Date of the cube
    gyroSupported: Optional[bool]  # Is gyroscope supported by this cube model

# Disconnect event
class GanCubeDisconnectEvent(TypedDict):
    type: str  # "DISCONNECT"

# All possible event message types
GanCubeEventMessage = Union[GanCubeMoveEvent, GanCubeFaceletsEvent, GanCubeGyroEvent, GanCubeBatteryEvent, GanCubeHardwareEvent, GanCubeDisconnectEvent]

# Cube event / response to command
#class GanCubeEvent(GanCubeEventMessage):
class GanCubeEvent(TypedDict):
    timestamp: int

# We need to merge the timestamp with the event message
def create_gan_cube_event(timestamp: int, event_message: GanCubeEventMessage) -> GanCubeEvent:
    result = dict(event_message)
    result['timestamp'] = timestamp
    return result

# Extension to the BluetoothDevice for storing and accessing device MAC address
class BluetoothDeviceWithMAC:
    def __init__(self, name: Optional[str] = None, mac: Optional[str] = None):
        self.name = name
        self.mac = mac
        self.gatt = None
        self._event_listeners = {}

    def addEventListener(self, event_type: str, callback: Callable):
        if event_type not in self._event_listeners:
            self._event_listeners[event_type] = []
        self._event_listeners[event_type].append(callback)

    def removeEventListener(self, event_type: str, callback: Callable):
        if event_type in self._event_listeners:
            if callback in self._event_listeners[event_type]:
                self._event_listeners[event_type].remove(callback)

# Mock Bluetooth characteristics for Python equivalent
class BluetoothRemoteGATTCharacteristic:
    def __init__(self):
        self.value = None
        self._event_listeners = {}
        self._notifications_started = False

    def addEventListener(self, event_type: str, callback: Callable):
        if event_type not in self._event_listeners:
            self._event_listeners[event_type] = []
        self._event_listeners[event_type].append(callback)

    def removeEventListener(self, event_type: str, callback: Callable):
        if event_type in self._event_listeners:
            if callback in self._event_listeners[event_type]:
                self._event_listeners[event_type].remove(callback)

    async def writeValue(self, value: bytes) -> None:
        # Implementation would depend on actual Bluetooth library
        pass

    async def startNotifications(self) -> None:
        self._notifications_started = True

    async def stopNotifications(self) -> None:
        self._notifications_started = False

# Connection object representing connection API and state
class GanCubeConnection(Protocol):
    @property
    def deviceName(self) -> str:
        """Connected Bluetooth cube device name"""
        ...

    @property
    def deviceMAC(self) -> str:
        """Connected Bluetooth cube device MAC address"""
        ...

    @property
    def events(self) -> Subject:
        """RxJS Subject to subscribe for cube event messages"""
        ...

    @events.setter
    def events(self, value) -> None:
        ...

    async def sendCubeCommand(self, command: GanCubeCommand) -> None:
        """Method to send command to the cube"""
        ...

    async def disconnect(self) -> None:
        """Close this connection"""
        ...

# Raw connection interface for internal use
class GanCubeRawConnection(Protocol):
    async def sendCommandMessage(self, message: bytes) -> None:
        ...

    async def disconnect(self) -> None:
        ...

# Protocol Driver interface
class GanProtocolDriver(Protocol):
    def createCommandMessage(self, command: GanCubeCommand) -> Optional[bytes]:
        """Create binary command message for cube device"""
        ...

    async def handleStateEvent(self, conn: GanCubeRawConnection, eventMessage: bytes) -> List[GanCubeEvent]:
        """Handle binary event messages from cube device"""
        ...

# Calculate sum of all numbers in array
def sum_array(arr: List[int]) -> int:
    return sum(arr)

# Implementation of classic command/response connection with GAN Smart Cube device
class GanCubeClassicConnection(GanCubeConnection, GanCubeRawConnection):

    def __init__(self,
                 device: BluetoothDeviceWithMAC,
                 commandCharacteristic: BluetoothRemoteGATTCharacteristic,
                 stateCharacteristic: BluetoothRemoteGATTCharacteristic,
                 encrypter: GanCubeEncrypter,
                 driver: GanProtocolDriver,
                 client: BleakClient):
        self.device = device
        self.commandCharacteristic = commandCharacteristic
        self.stateCharacteristic = stateCharacteristic
        self.encrypter = encrypter
        self.driver = driver
        self._events = Subject()
        self._client = client

    @classmethod
    async def create(cls,
                     device: BluetoothDeviceWithMAC,
                     commandCharacteristic: BluetoothRemoteGATTCharacteristic,
                     stateCharacteristic: BluetoothRemoteGATTCharacteristic,
                     encrypter: GanCubeEncrypter,
                     driver: GanProtocolDriver,
                     client: BleakClient) -> GanCubeConnection:
        conn = cls(device, commandCharacteristic, stateCharacteristic, encrypter, driver, client)
        #conn.device.addEventListener('gattserverdisconnected', conn.onDisconnect)
        client.disconnected_callback = conn.onDisconnect
        #conn.stateCharacteristic.addEventListener('characteristicvaluechanged', conn.onStateUpdate)
        #await conn.stateCharacteristic.startNotifications()
        await client.start_notify(stateCharacteristic, conn.onStateUpdate)
        return conn

    @property
    def deviceName(self) -> str:
        return self.device.name or "GAN-XXXX"

    @property
    def deviceMAC(self) -> str:
        return self.device.mac or "00:00:00:00:00:00"

    @property
    def events(self) -> Subject:
        return self._events

    @events.setter
    def events(self, value) -> None:
        self._events = value

    async def sendCommandMessage(self, message: bytes) -> None:
        encryptedMessage = self.encrypter.encrypt(message)
        #return await self.commandCharacteristic.writeValue(encryptedMessage)
        return await self._client.write_gatt_char(self.commandCharacteristic, encryptedMessage)


    #async def onStateUpdate(self, evt: Any) -> None:
    async def onStateUpdate(self, sender: BleakGATTCharacteristic, data: bytearray) -> None:
        #characteristic = evt.target
        #eventMessage = characteristic.value
        eventMessage = data
        if eventMessage and len(eventMessage) >= 16:
            decryptedMessage = self.encrypter.decrypt(bytes(eventMessage))
            cubeEvents = await self.driver.handleStateEvent(self, decryptedMessage)
            for e in cubeEvents:
                self.events.next(e)

    async def onDisconnect(self, client: BleakClient) -> Any:
        self.device.removeEventListener('gattserverdisconnected', self.onDisconnect)
        self.stateCharacteristic.removeEventListener('characteristicvaluechanged', self.onStateUpdate)
        self.events.next(create_gan_cube_event(now(), {"type": "DISCONNECT"}))
        self.events.unsubscribe()
        try:
            return await self.stateCharacteristic.stopNotifications()
        except:
            pass

    async def sendCubeCommand(self, command: GanCubeCommand) -> None:
        commandMessage = self.driver.createCommandMessage(command)
        if commandMessage:
            return await self.sendCommandMessage(commandMessage)

    async def disconnect(self) -> None:
        await self.onDisconnect()
        if self.device.gatt and hasattr(self.device.gatt, 'connected') and self.device.gatt.connected:
            if hasattr(self.device.gatt, 'disconnect'):
                self.device.gatt.disconnect()

# View for binary protocol messages allowing to retrieve from message arbitrary length bit words
class GanProtocolMessageView:

    def __init__(self, message: bytes):
        self.bits = ''.join(format(byte, '08b') for byte in message)

    def getBitWord(self, startBit: int, bitLength: int, littleEndian: bool = False) -> int:
        if bitLength <= 8:
            return int(self.bits[startBit:startBit + bitLength], 2)
        elif bitLength == 16 or bitLength == 32:
            buf = bytearray(bitLength // 8)
            for i in range(len(buf)):
                buf[i] = int(self.bits[8 * i + startBit:8 * i + startBit + 8], 2)

            if bitLength == 16:
                return struct.unpack('<H' if littleEndian else '>H', buf)[0]
            else:
                return struct.unpack('<I' if littleEndian else '>I', buf)[0]
        else:
            raise ValueError('Unsupported bit word length')

# Driver implementation for GAN Gen2 protocol, supported cubes:
#  - GAN Mini ui FreePlay
#  - GAN12 ui FreePlay
#  - GAN12 ui
#  - GAN356 i Carry S
#  - GAN356 i Carry
#  - GAN356 i 3
#  - Monster Go 3Ai
class GanGen2ProtocolDriver(GanProtocolDriver):

    def __init__(self):
        self.lastSerial: int = -1
        self.lastMoveTimestamp: int = 0
        self.cubeTimestamp: int = 0

    def createCommandMessage(self, command: GanCubeCommand) -> Optional[bytes]:
        msg = bytearray(20)

        if command['type'] == 'REQUEST_FACELETS':
            msg[0] = 0x04
        elif command['type'] == 'REQUEST_HARDWARE':
            msg[0] = 0x05
        elif command['type'] == 'REQUEST_BATTERY':
            msg[0] = 0x09
        elif command['type'] == 'REQUEST_RESET':
            msg[:] = [0x0A, 0x05, 0x39, 0x77, 0x00, 0x00, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        else:
            return None

        return bytes(msg)

    async def handleStateEvent(self, conn: GanCubeRawConnection, eventMessage: bytes) -> List[GanCubeEvent]:
        timestamp = now()

        cubeEvents: List[GanCubeEvent] = []
        msg = GanProtocolMessageView(eventMessage)
        eventType = msg.getBitWord(0, 4)

        if eventType == 0x01:  # GYRO
            # Orientation Quaternion
            qw = msg.getBitWord(4, 16)
            qx = msg.getBitWord(20, 16)
            qy = msg.getBitWord(36, 16)
            qz = msg.getBitWord(52, 16)

            # Angular Velocity
            vx = msg.getBitWord(68, 4)
            vy = msg.getBitWord(72, 4)
            vz = msg.getBitWord(76, 4)

            cubeEvents.append(create_gan_cube_event(timestamp, {
                "type": "GYRO",
                "quaternion": {
                    "x": (1 - (qx >> 15) * 2) * (qx & 0x7FFF) / 0x7FFF,
                    "y": (1 - (qy >> 15) * 2) * (qy & 0x7FFF) / 0x7FFF,
                    "z": (1 - (qz >> 15) * 2) * (qz & 0x7FFF) / 0x7FFF,
                    "w": (1 - (qw >> 15) * 2) * (qw & 0x7FFF) / 0x7FFF
                },
                "velocity": {
                    "x": (1 - (vx >> 3) * 2) * (vx & 0x7),
                    "y": (1 - (vy >> 3) * 2) * (vy & 0x7),
                    "z": (1 - (vz >> 3) * 2) * (vz & 0x7)
                }
            }))

        elif eventType == 0x02:  # MOVE
            if self.lastSerial != -1:  # Accept move events only after first facelets state event received
                serial = msg.getBitWord(4, 8)
                diff = min((serial - self.lastSerial) & 0xFF, 7)
                self.lastSerial = serial

                if diff > 0:
                    for i in range(diff - 1, -1, -1):
                        face = msg.getBitWord(12 + 5 * i, 4)
                        direction = msg.getBitWord(16 + 5 * i, 1)
                        move = "URFDLB"[face] + " '"[direction]
                        elapsed = msg.getBitWord(47 + 16 * i, 16)
                        if elapsed == 0:  # In case of 16-bit cube timestamp register overflow
                            elapsed = timestamp - self.lastMoveTimestamp
                        self.cubeTimestamp += elapsed
                        cubeEvents.append(create_gan_cube_event(timestamp, {
                            "type": "MOVE",
                            "serial": (serial - i) & 0xFF,
                            "localTimestamp": timestamp if i == 0 else None,  # Missed and recovered events has no meaningful local timestamps
                            "cubeTimestamp": self.cubeTimestamp,
                            "face": face,
                            "direction": direction,
                            "move": move.strip()
                        }))
                    self.lastMoveTimestamp = timestamp

        elif eventType == 0x04:  # FACELETS
            serial = msg.getBitWord(4, 8)

            if self.lastSerial == -1:
                self.lastSerial = serial

            # Corner/Edge Permutation/Orientation
            cp: List[int] = []
            co: List[int] = []
            ep: List[int] = []
            eo: List[int] = []

            # Corners
            for i in range(7):
                cp.append(msg.getBitWord(12 + i * 3, 3))
                co.append(msg.getBitWord(33 + i * 2, 2))
            cp.append(28 - sum_array(cp))
            co.append((3 - (sum_array(co) % 3)) % 3)

            # Edges
            for i in range(11):
                ep.append(msg.getBitWord(47 + i * 4, 4))
                eo.append(msg.getBitWord(91 + i, 1))
            ep.append(66 - sum_array(ep))
            eo.append((2 - (sum_array(eo) % 2)) % 2)

            cubeEvents.append(create_gan_cube_event(timestamp, {
                "type": "FACELETS",
                "serial": serial,
                "facelets": toKociembaFacelets(cp, co, ep, eo),
                "state": {
                    "CP": cp,
                    "CO": co,
                    "EP": ep,
                    "EO": eo
                }
            }))

        elif eventType == 0x05:  # HARDWARE
            hwMajor = msg.getBitWord(8, 8)
            hwMinor = msg.getBitWord(16, 8)
            swMajor = msg.getBitWord(24, 8)
            swMinor = msg.getBitWord(32, 8)
            gyroSupported = msg.getBitWord(104, 1)

            hardwareName = ''
            for i in range(8):
                hardwareName += chr(msg.getBitWord(i * 8 + 40, 8))

            cubeEvents.append(create_gan_cube_event(timestamp, {
                "type": "HARDWARE",
                "hardwareName": hardwareName,
                "hardwareVersion": f"{hwMajor}.{hwMinor}",
                "softwareVersion": f"{swMajor}.{swMinor}",
                "gyroSupported": bool(gyroSupported)
            }))

        elif eventType == 0x09:  # BATTERY
            batteryLevel = msg.getBitWord(8, 8)

            cubeEvents.append(create_gan_cube_event(timestamp, {
                "type": "BATTERY",
                "batteryLevel": min(batteryLevel, 100)
            }))

        elif eventType == 0x0D:  # DISCONNECT
            await conn.disconnect()

        return cubeEvents

# Driver implementation for GAN Gen3 protocol, supported cubes:
#  - GAN356 i Carry 2
class GanGen3ProtocolDriver(GanProtocolDriver):

    def __init__(self):
        self.serial: int = -1
        self.lastSerial: int = -1
        self.lastLocalTimestamp: Optional[int] = None
        self.moveBuffer: List[GanCubeEvent] = []

    def createCommandMessage(self, command: GanCubeCommand) -> Optional[bytes]:
        msg = bytearray(16)

        if command['type'] == 'REQUEST_FACELETS':
            msg[:2] = [0x68, 0x01]
        elif command['type'] == 'REQUEST_HARDWARE':
            msg[:2] = [0x68, 0x04]
        elif command['type'] == 'REQUEST_BATTERY':
            msg[:2] = [0x68, 0x07]
        elif command['type'] == 'REQUEST_RESET':
            msg[:] = [0x68, 0x05, 0x05, 0x39, 0x77, 0x00, 0x00, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0x00, 0x00, 0x00]
        else:
            return None

        return bytes(msg)

    # Private cube command for requesting move history
    async def requestMoveHistory(self, conn: GanCubeRawConnection, serial: int, count: int) -> None:
        msg = bytearray(16)
        # Move history response data is byte-aligned, and moves always starting with near-ceil odd serial number, regardless of requested.
        # Adjust serial and count to get odd serial aligned history window with even number of moves inside.
        if serial % 2 == 0:
            serial = (serial - 1) & 0xFF
        if count % 2 == 1:
            count += 1
        # Never overflow requested history window beyond the serial number cycle edge 255 -> 0.
        # Because due to iCarry2 firmware bug the moves beyond the edge will be spoofed with 'D' (just zero bytes).
        count = min(count, serial + 1)
        msg[:6] = [0x68, 0x03, serial, 0, count, 0]
        try:
            await conn.sendCommandMessage(bytes(msg))
        except:
            # We can safely suppress and ignore possible GATT write errors, requestMoveHistory command is automatically retried on next move event
            pass

    # Evict move events from FIFO buffer until missing move event detected
    # In case of missing move, and if connection is provided, submit request for move history to fill gap in buffer
    async def evictMoveBuffer(self, conn: Optional[GanCubeRawConnection] = None) -> List[GanCubeEvent]:
        evictedEvents: List[GanCubeEvent] = []
        while len(self.moveBuffer) > 0:
            bufferHead = self.moveBuffer[0]
            diff = 1 if self.lastSerial == -1 else (bufferHead['serial'] - self.lastSerial) & 0xFF
            if diff > 1:
                if conn:
                    await self.requestMoveHistory(conn, bufferHead['serial'], diff)
                break
            else:
                evictedEvents.append(self.moveBuffer.pop(0))
                self.lastSerial = bufferHead['serial']

        # Probably something went wrong and buffer is no longer evicted, so forcibly disconnect the cube
        if conn and len(self.moveBuffer) > 16:
            await conn.disconnect()
        return evictedEvents

    # Check if circular serial number (modulo 256) fits into (start,end) serial number range.
    # By default range is open, set closedStart / closedEnd to make it closed.
    def isSerialInRange(self, start: int, end: int, serial: int, closedStart: bool = False, closedEnd: bool = False) -> bool:
        return (((end - start) & 0xFF) >= ((serial - start) & 0xFF)
                and (closedStart or ((start - serial) & 0xFF) > 0)
                and (closedEnd or ((end - serial) & 0xFF) > 0))

    # Used to inject missed moves to FIFO buffer
    def injectMissedMoveToBuffer(self, move: GanCubeEvent) -> None:
        if move['type'] == "MOVE":
            if len(self.moveBuffer) > 0:
                bufferHead = self.moveBuffer[0]
                # Skip if move event with the same serial already in the buffer
                if any(e['type'] == "MOVE" and e['serial'] == move['serial'] for e in self.moveBuffer):
                    return
                # Skip if move serial does not fit in range between last evicted event and event on buffer head, i.e. event must be one of missed
                if not self.isSerialInRange(self.lastSerial, bufferHead['serial'], move['serial']):
                    return
                # Move history events should be injected in reverse order, so just put suitable event on buffer head
                if move['serial'] == ((bufferHead['serial'] - 1) & 0xFF):
                    self.moveBuffer.insert(0, move)
            else:
                # This case happens when lost move is recovered using periodic
                # facelets state event, and being inserted into the empty buffer.
                if self.isSerialInRange(self.lastSerial, self.serial, move['serial'], False, True):
                    self.moveBuffer.insert(0, move)

    # Used in response to periodic facelets event to check if any moves missed
    async def checkIfMoveMissed(self, conn: GanCubeRawConnection) -> None:
        diff = (self.serial - self.lastSerial) & 0xFF
        if diff > 0:
            if self.serial != 0:  # Constraint to avoid iCarry2 firmware bug with facelets state event at 255 move counter
                bufferHead = self.moveBuffer[0] if self.moveBuffer else None
                startSerial = bufferHead['serial'] if bufferHead else (self.serial + 1) & 0xFF
                await self.requestMoveHistory(conn, startSerial, diff + 1)

    async def handleStateEvent(self, conn: GanCubeRawConnection, eventMessage: bytes) -> List[GanCubeEvent]:
        timestamp = now()

        cubeEvents: List[GanCubeEvent] = []
        msg = GanProtocolMessageView(eventMessage)

        magic = msg.getBitWord(0, 8)
        eventType = msg.getBitWord(8, 8)
        dataLength = msg.getBitWord(16, 8)

        if magic == 0x55 and dataLength > 0:

            if eventType == 0x01:  # MOVE
                if self.lastSerial != -1:  # Accept move events only after first facelets state event received
                    self.lastLocalTimestamp = timestamp
                    cubeTimestamp = msg.getBitWord(24, 32, True)
                    serial = self.serial = msg.getBitWord(56, 16, True)

                    direction = msg.getBitWord(72, 2)
                    face = [2, 32, 8, 1, 16, 4].index(msg.getBitWord(74, 6)) if msg.getBitWord(74, 6) in [2, 32, 8, 1, 16, 4] else -1
                    move = "URFDLB"[face] + " '"[direction] if face >= 0 else ""

                    # put move event into FIFO buffer
                    if face >= 0:
                        self.moveBuffer.append(create_gan_cube_event(timestamp, {
                            "type": "MOVE",
                            "serial": serial,
                            "localTimestamp": timestamp,
                            "cubeTimestamp": cubeTimestamp,
                            "face": face,
                            "direction": direction,
                            "move": move.strip()
                        }))

                    # evict move events from FIFO buffer
                    cubeEvents = await self.evictMoveBuffer(conn)

            elif eventType == 0x06:  # MOVE_HISTORY
                startSerial = msg.getBitWord(24, 8)
                count = (dataLength - 1) * 2

                # inject missed moves into FIFO buffer
                for i in range(count):
                    face = [1, 5, 3, 0, 4, 2].index(msg.getBitWord(32 + 4 * i, 3)) if msg.getBitWord(32 + 4 * i, 3) in [1, 5, 3, 0, 4, 2] else -1
                    direction = msg.getBitWord(35 + 4 * i, 1)
                    if face >= 0:
                        move = "URFDLB"[face] + " '"[direction]
                        self.injectMissedMoveToBuffer(create_gan_cube_event(timestamp, {
                            "type": "MOVE",
                            "serial": (startSerial - i) & 0xFF,
                            "localTimestamp": None,  # Missed and recovered events has no meaningful local timestamps
                            "cubeTimestamp": None,  # Missed and recovered events has no meaningful cube timestamps
                            "face": face,
                            "direction": direction,
                            "move": move.strip()
                        }))

                # evict move events from FIFO buffer
                cubeEvents = await self.evictMoveBuffer(conn)

        return cubeEvents
