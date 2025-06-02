import asyncio
import struct
from typing import Optional, Callable, Dict, Any, List, Union, Protocol
from bleak import BleakClient, BleakScanner
from bleak.backends.device import BLEDevice
from bleak.backends.characteristic import BleakGATTCharacteristic
from bleak.backends.service import BleakGATTService

# Import the equivalent Python modules
import gan_cube_definitions as def_module
from gan_cube_encrypter import GanGen2CubeEncrypter, GanGen3CubeEncrypter, GanGen4CubeEncrypter
from gan_cube_protocol import (
    BluetoothDeviceWithMAC,
    GanCubeConnection,
    GanCubeCommand,
    GanCubeEvent,
    GanCubeMove,
    GanCubeClassicConnection,
    GanGen2ProtocolDriver
)

class BluetoothManufacturerData:
    """Equivalent to Web Bluetooth's BluetoothManufacturerData Map"""
    def __init__(self, data: Dict[int, bytes]):
        self._data = data

    def has(self, key: int) -> bool:
        return key in self._data

    def get(self, key: int) -> Optional[bytes]:
        return self._data.get(key)

class DataView:
    """Python equivalent of JavaScript's DataView"""
    def __init__(self, buffer: Union[bytes, bytearray], byte_offset: int = 0, byte_length: Optional[int] = None):
        self.buffer = buffer
        self.byte_offset = byte_offset
        if byte_length is None:
            self.byte_length = len(buffer) - byte_offset
        else:
            self.byte_length = byte_length
        self._data = buffer[byte_offset:byte_offset + self.byte_length]

    def get_uint8(self, byte_offset: int) -> int:
        return self._data[byte_offset]

class BluetoothAdvertisingEvent:
    """Mock event class for advertisement data"""
    def __init__(self, manufacturer_data: BluetoothManufacturerData):
        self.manufacturer_data = manufacturer_data

def get_manufacturer_data_bytes(manufacturer_data: Union[BluetoothManufacturerData, DataView]) -> Optional[DataView]:
    """Iterate over all known GAN cube CICs to find Manufacturer Specific Data"""
    # Workaround for Bluefy browser which may return raw DataView directly instead of Map
    if isinstance(manufacturer_data, DataView):
        return DataView(manufacturer_data.buffer[2:11])

    for id_val in def_module.GAN_CIC_LIST:
        if id_val in manufacturer_data:
            data = manufacturer_data[id_val]
            if data:
                return DataView(data[:9])
    return None

def extract_mac(manufacturer_data: Dict[int, bytes]) -> str:
    """Extract MAC from last 6 bytes of Manufacturer Specific Data"""
    mac: List[str] = []
    data_view = get_manufacturer_data_bytes(manufacturer_data)
    if data_view and data_view.byte_length >= 6:
        for i in range(1, 7):
            mac.append(format(data_view.get_uint8(data_view.byte_length - i), '02X'))
    return ":".join(mac)

async def auto_retrieve_mac_address(manufacturer_data) -> Optional[str]:
    """If browser supports Web Bluetooth watchAdvertisements() API, try to retrieve MAC address automatically"""
    # In Python with bleak, we can scan for advertisements
    mac = extract_mac(manufacturer_data)
    return mac

# Type representing function interface to implement custom MAC address provider
MacAddressProvider = Callable[[BLEDevice, Optional[bool]], Optional[str]]

class NavigatorBluetooth:
    """Mock navigator.bluetooth equivalent for Python"""
    @staticmethod
    async def request_device(options: Dict[str, Any]) -> Any:
        """Request user for the bluetooth device (popup selection dialog)"""
        scanner = BleakScanner()
        devices = await scanner.discover(return_adv=True)

        # Filter devices based on name prefixes
        filters = options.get('filters', [])
        matching_devices = []

        for device, adv_data in devices.values():
            if device.name:
                for filter_item in filters:
                    name_prefix = filter_item.get('namePrefix')
                    if name_prefix and device.name.startswith(name_prefix):
                        matching_devices.append((device, adv_data.manufacturer_data))
                        break

        if len(matching_devices) == 0:
            raise Exception("No matching devices found")

        # Return first matching device (in real implementation, this would show a selection dialog)
        return matching_devices[0]

# Mock navigator object
class Navigator:
    bluetooth = NavigatorBluetooth()

navigator = Navigator()

async def connect_gan_cube(custom_mac_address_provider: Optional[MacAddressProvider] = None) -> GanCubeConnection:
    """
    Initiate new connection with the GAN Smart Cube device
    @param custom_mac_address_provider Optional custom provider for cube MAC address
    @returns Object representing connection API and state
    """

    # Request user for the bluetooth device (popup selection dialog)
    device, adv_data = await navigator.bluetooth.request_device({
        'filters': [
            {'namePrefix': "GAN"},
            {'namePrefix': "MG"},
            {'namePrefix': "AiCube"}
        ],
        'optionalServices': [def_module.GAN_GEN2_SERVICE, def_module.GAN_GEN3_SERVICE, def_module.GAN_GEN4_SERVICE],
        'optionalManufacturerData': def_module.GAN_CIC_LIST
    })

    # Retrieve cube MAC address needed for key salting
    mac = None
    if custom_mac_address_provider:
        mac = await custom_mac_address_provider(device, adv_data, False)

    if not mac:
        mac = await auto_retrieve_mac_address(adv_data)

    if not mac and custom_mac_address_provider:
        mac = await custom_mac_address_provider(device, adv_data, True)

    if not mac:
        raise Exception('Unable to determine cube MAC address, connection is not possible!')


    # Create encryption salt from MAC address bytes placed in reverse order
    mac_parts = mac.replace('-', ':').replace(' ', ':').split(':')
    salt = bytes([int(c, 16) for c in reversed(mac_parts)])

    # Connect to GATT and get device primary services
    client = BleakClient(device.address)
    await client.connect()
    services = client.services

    conn: Optional[GanCubeConnection] = None

    # Resolve type of connected cube device and setup appropriate encryption / protocol driver
    for service in services:
        service_uuid = service.uuid.lower()
        if service_uuid == def_module.GAN_GEN2_SERVICE:
            command_characteristic = service.get_characteristic(def_module.GAN_GEN2_COMMAND_CHARACTERISTIC)
            state_characteristic = service.get_characteristic(def_module.GAN_GEN2_STATE_CHARACTERISTIC)
            key = def_module.GAN_ENCRYPTION_KEYS[1] if device.name and device.name.startswith('AiCube') else def_module.GAN_ENCRYPTION_KEYS[0]
            encrypter = GanGen2CubeEncrypter(bytes(key['key']), bytes(key['iv']), salt)
            driver = GanGen2ProtocolDriver()
            conn = await GanCubeClassicConnection.create(device, command_characteristic, state_characteristic, encrypter, driver, client)
            break
        elif service_uuid == def_module.GAN_GEN3_SERVICE:
            command_characteristic = service.get_characteristic(def_module.GAN_GEN3_COMMAND_CHARACTERISTIC)
            state_characteristic = service.get_characteristic(def_module.GAN_GEN3_STATE_CHARACTERISTIC)
            key = def_module.GAN_ENCRYPTION_KEYS[0]
            encrypter = GanGen3CubeEncrypter(bytes(key['key']), bytes(key['iv']), salt)
            driver = GanGen3ProtocolDriver()
            conn = await GanCubeClassicConnection.create(device, command_characteristic, state_characteristic, encrypter, driver, client)
            break
        elif service_uuid == def_module.GAN_GEN4_SERVICE:
            command_characteristic = service.get_characteristic(def_module.GAN_GEN4_COMMAND_CHARACTERISTIC)
            state_characteristic = service.get_characteristic(def_module.GAN_GEN4_STATE_CHARACTERISTIC)
            key = def_module.GAN_ENCRYPTION_KEYS[0]
            encrypter = GanGen4CubeEncrypter(bytes(key['key']), bytes(key['iv']), salt)
            driver = GanGen4ProtocolDriver()
            conn = await GanCubeClassicConnection.create(device, command_characteristic, state_characteristic, encrypter, driver, client)
            break

    if not conn:
        raise Exception("Can't find target BLE services - wrong or unsupported cube device model")

    return conn
