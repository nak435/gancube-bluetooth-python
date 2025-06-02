from Crypto.Cipher import AES #pip install pycryptodome
from abc import ABC, abstractmethod
from typing import Union

class GanCubeEncrypter(ABC):
    """
    Common cube encrypter interface
    """

    @abstractmethod
    def encrypt(self, data: bytes) -> bytes:
        """Encrypt binary message buffer represented as bytes"""
        pass

    @abstractmethod
    def decrypt(self, data: bytes) -> bytes:
        """Decrypt binary message buffer represented as bytes"""
        pass


class GanGen2CubeEncrypter(GanCubeEncrypter):
    """
    Implementation for encryption scheme used in the GAN Gen2 Smart Cubes
    """

    def __init__(self, key: bytes, iv: bytes, salt: bytes):
        if len(key) != 16:
            raise Exception("Key must be 16 bytes (128-bit) long")
        if len(iv) != 16:
            raise Exception("Iv must be 16 bytes (128-bit) long")
        if len(salt) != 6:
            raise Exception("Salt must be 6 bytes (48-bit) long")

        # Apply salt to key and iv
        self._key = bytearray(key)
        self._iv = bytearray(iv)
        for i in range(6):
            self._key[i] = (key[i] + salt[i]) % 0xFF
            self._iv[i] = (iv[i] + salt[i]) % 0xFF

        self._key = bytes(self._key)
        self._iv = bytes(self._iv)

    def _encrypt_chunk(self, buffer: bytearray, offset: int) -> None:
        """Encrypt 16-byte buffer chunk starting at offset using AES-128-CBC"""
        cipher = AES.new(self._key, AES.MODE_CBC, self._iv)
        chunk = cipher.encrypt(bytes(buffer[offset:offset + 16]))
        buffer[offset:offset + 16] = chunk

    def _decrypt_chunk(self, buffer: bytearray, offset: int) -> None:
        """Decrypt 16-byte buffer chunk starting at offset using AES-128-CBC"""
        cipher = AES.new(self._key, AES.MODE_CBC, self._iv)
        chunk = cipher.decrypt(bytes(buffer[offset:offset + 16]))
        buffer[offset:offset + 16] = chunk

    def encrypt(self, data: bytes) -> bytes:
        if len(data) < 16:
            raise Exception('Data must be at least 16 bytes long')
        res = bytearray(data)
        # encrypt 16-byte chunk aligned to message start
        self._encrypt_chunk(res, 0)
        # encrypt 16-byte chunk aligned to message end
        if len(res) > 16:
            self._encrypt_chunk(res, len(res) - 16)
        return bytes(res)

    def decrypt(self, data: bytes) -> bytes:
        if len(data) < 16:
            raise Exception('Data must be at least 16 bytes long')
        res = bytearray(data)
        # decrypt 16-byte chunk aligned to message end
        if len(res) > 16:
            self._decrypt_chunk(res, len(res) - 16)
        # decrypt 16-byte chunk aligned to message start
        self._decrypt_chunk(res, 0)
        return bytes(res)


class GanGen3CubeEncrypter(GanGen2CubeEncrypter):
    """
    Implementation for encryption scheme used in the GAN Gen3 cubes
    """
    """101 its just the same"""
    pass


class GanGen4CubeEncrypter(GanGen2CubeEncrypter):
    """
    Implementation for encryption scheme used in the GAN Gen3 cubes
    """
    """amazing, it's still the same"""
    pass
