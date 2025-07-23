"""
Encryption Manager - Data encryption utilities for CrewGraph AI

This module provides encryption capabilities for securing sensitive data including:
- Symmetric encryption for data at rest
- Key management and rotation
- Secure data serialization
- Field-level encryption
- Configuration encryption

Features:
- Multiple encryption algorithms
- Key derivation and management
- Secure random generation
- Performance optimized
- Easy integration

Created by: Vatsal216
Date: 2025-07-23
"""

import os
import base64
import json
import hashlib
import secrets
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import threading

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

from ..utils.logging import get_logger
from ..utils.exceptions import CrewGraphError

logger = get_logger(__name__)


class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms"""
    FERNET = "fernet"  # Symmetric encryption (recommended)
    AES_256_GCM = "aes_256_gcm"  # AES-256 in GCM mode
    CHACHA20_POLY1305 = "chacha20_poly1305"  # ChaCha20-Poly1305


class KeyDerivationFunction(Enum):
    """Key derivation functions"""
    PBKDF2 = "pbkdf2"
    SCRYPT = "scrypt"


@dataclass
class CryptoConfig:
    """Encryption configuration"""
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.FERNET
    key_derivation: KeyDerivationFunction = KeyDerivationFunction.PBKDF2
    key_size: int = 32  # bytes
    salt_size: int = 16  # bytes
    iterations: int = 100000  # for PBKDF2
    memory_cost: int = 64 * 1024 * 1024  # for Scrypt (64MB)
    parallelization: int = 1  # for Scrypt
    enable_compression: bool = True
    key_rotation_enabled: bool = True
    max_key_age_days: int = 90


@dataclass
class EncryptedData:
    """Encrypted data container"""
    ciphertext: str  # Base64 encoded
    algorithm: str
    salt: str  # Base64 encoded
    nonce: Optional[str] = None  # Base64 encoded (for some algorithms)
    key_version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "ciphertext": self.ciphertext,
            "algorithm": self.algorithm,
            "salt": self.salt,
            "nonce": self.nonce,
            "key_version": self.key_version,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EncryptedData":
        """Create from dictionary"""
        return cls(
            ciphertext=data["ciphertext"],
            algorithm=data["algorithm"],
            salt=data["salt"],
            nonce=data.get("nonce"),
            key_version=data.get("key_version", 1),
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "EncryptedData":
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))


class EncryptionManager:
    """
    Comprehensive encryption manager for CrewGraph AI.
    
    Provides secure encryption and decryption capabilities with
    key management, algorithm flexibility, and enterprise features.
    """
    
    def __init__(self, 
                 config: Optional[CryptoConfig] = None,
                 master_key: Optional[str] = None,
                 key_file: Optional[str] = None):
        """
        Initialize encryption manager.
        
        Args:
            config: Encryption configuration
            master_key: Master encryption key (base64 encoded)
            key_file: Path to key file
            
        Raises:
            CrewGraphError: If cryptography library not available
        """
        if not CRYPTO_AVAILABLE:
            raise CrewGraphError(
                "Cryptography library not available. "
                "Install with: pip install cryptography"
            )
        
        self.config = config or CryptoConfig()
        
        # Key management
        self._keys: Dict[int, bytes] = {}  # version -> key
        self._current_key_version = 1
        self._master_key: Optional[bytes] = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize keys
        self._initialize_keys(master_key, key_file)
        
        logger.info("EncryptionManager initialized")
        logger.info(f"Algorithm: {self.config.algorithm.value}")
        logger.info(f"Key derivation: {self.config.key_derivation.value}")
    
    def _initialize_keys(self, master_key: Optional[str], key_file: Optional[str]):
        """Initialize encryption keys"""
        if master_key:
            # Use provided master key
            self._master_key = base64.b64decode(master_key.encode())
        elif key_file and os.path.exists(key_file):
            # Load key from file
            try:
                with open(key_file, 'rb') as f:
                    self._master_key = f.read()
                logger.info(f"Master key loaded from file: {key_file}")
            except Exception as e:
                logger.error(f"Failed to load key file: {e}")
                self._master_key = None
        
        if not self._master_key:
            # Generate new master key
            self._master_key = self._generate_key()
            logger.info("New master key generated")
        
        # Derive current encryption key
        self._keys[self._current_key_version] = self._master_key[:self.config.key_size]
    
    def _generate_key(self) -> bytes:
        """Generate a secure random key"""
        return secrets.token_bytes(self.config.key_size)
    
    def _generate_salt(self) -> bytes:
        """Generate a secure random salt"""
        return secrets.token_bytes(self.config.salt_size)
    
    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password and salt"""
        password_bytes = password.encode('utf-8')
        
        if self.config.key_derivation == KeyDerivationFunction.PBKDF2:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=self.config.key_size,
                salt=salt,
                iterations=self.config.iterations,
                backend=default_backend()
            )
        elif self.config.key_derivation == KeyDerivationFunction.SCRYPT:
            kdf = Scrypt(
                algorithm=hashes.SHA256(),
                length=self.config.key_size,
                salt=salt,
                n=2**14,  # CPU/memory cost
                r=8,      # Block size
                p=self.config.parallelization,
                backend=default_backend()
            )
        else:
            raise CrewGraphError(f"Unsupported key derivation: {self.config.key_derivation}")
        
        return kdf.derive(password_bytes)
    
    def encrypt(self, data: Any, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Encrypt data and return as JSON string.
        
        Args:
            data: Data to encrypt (will be JSON serialized)
            context: Additional context for encryption
            
        Returns:
            JSON string containing encrypted data
            
        Raises:
            CrewGraphError: If encryption fails
        """
        try:
            # Serialize data
            if isinstance(data, (str, bytes)):
                plaintext = data if isinstance(data, str) else data.decode('utf-8')
            else:
                plaintext = json.dumps(data, default=str)
            
            # Compress if enabled
            if self.config.enable_compression:
                import zlib
                compressed = zlib.compress(plaintext.encode('utf-8'))
                plaintext_bytes = compressed
                compressed_flag = True
            else:
                plaintext_bytes = plaintext.encode('utf-8')
                compressed_flag = False
            
            # Generate salt and derive key
            salt = self._generate_salt()
            
            with self._lock:
                encryption_key = self._keys[self._current_key_version]
            
            # Encrypt based on algorithm
            if self.config.algorithm == EncryptionAlgorithm.FERNET:
                encrypted_data = self._encrypt_fernet(plaintext_bytes, encryption_key, salt)
            elif self.config.algorithm == EncryptionAlgorithm.AES_256_GCM:
                encrypted_data = self._encrypt_aes_gcm(plaintext_bytes, encryption_key, salt)
            elif self.config.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                encrypted_data = self._encrypt_chacha20(plaintext_bytes, encryption_key, salt)
            else:
                raise CrewGraphError(f"Unsupported algorithm: {self.config.algorithm}")
            
            # Add metadata
            encrypted_data.metadata.update({
                "compressed": compressed_flag,
                "original_type": type(data).__name__
            })
            
            if context:
                encrypted_data.metadata.update(context)
            
            return encrypted_data.to_json()
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise CrewGraphError(f"Encryption failed: {e}")
    
    def decrypt(self, encrypted_json: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Decrypt data from JSON string.
        
        Args:
            encrypted_json: JSON string containing encrypted data
            context: Additional context for decryption
            
        Returns:
            Decrypted and deserialized data
            
        Raises:
            CrewGraphError: If decryption fails
        """
        try:
            # Parse encrypted data
            encrypted_data = EncryptedData.from_json(encrypted_json)
            
            # Get encryption key
            with self._lock:
                if encrypted_data.key_version not in self._keys:
                    raise CrewGraphError(f"Unknown key version: {encrypted_data.key_version}")
                encryption_key = self._keys[encrypted_data.key_version]
            
            # Decrypt based on algorithm
            algorithm = EncryptionAlgorithm(encrypted_data.algorithm)
            salt = base64.b64decode(encrypted_data.salt)
            
            if algorithm == EncryptionAlgorithm.FERNET:
                plaintext_bytes = self._decrypt_fernet(encrypted_data, encryption_key, salt)
            elif algorithm == EncryptionAlgorithm.AES_256_GCM:
                plaintext_bytes = self._decrypt_aes_gcm(encrypted_data, encryption_key, salt)
            elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                plaintext_bytes = self._decrypt_chacha20(encrypted_data, encryption_key, salt)
            else:
                raise CrewGraphError(f"Unsupported algorithm: {algorithm}")
            
            # Decompress if needed
            if encrypted_data.metadata.get("compressed", False):
                import zlib
                plaintext_bytes = zlib.decompress(plaintext_bytes)
            
            plaintext = plaintext_bytes.decode('utf-8')
            
            # Deserialize based on original type
            original_type = encrypted_data.metadata.get("original_type", "str")
            if original_type in ("dict", "list", "int", "float", "bool"):
                return json.loads(plaintext)
            else:
                return plaintext
                
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise CrewGraphError(f"Decryption failed: {e}")
    
    def _encrypt_fernet(self, data: bytes, key: bytes, salt: bytes) -> EncryptedData:
        """Encrypt using Fernet algorithm"""
        # Derive key for Fernet (needs URL-safe base64 encoded key)
        derived_key = hashlib.sha256(key + salt).digest()
        fernet_key = base64.urlsafe_b64encode(derived_key)
        
        f = Fernet(fernet_key)
        ciphertext = f.encrypt(data)
        
        return EncryptedData(
            ciphertext=base64.b64encode(ciphertext).decode('ascii'),
            algorithm=EncryptionAlgorithm.FERNET.value,
            salt=base64.b64encode(salt).decode('ascii')
        )
    
    def _decrypt_fernet(self, encrypted_data: EncryptedData, key: bytes, salt: bytes) -> bytes:
        """Decrypt using Fernet algorithm"""
        # Derive key for Fernet
        derived_key = hashlib.sha256(key + salt).digest()
        fernet_key = base64.urlsafe_b64encode(derived_key)
        
        f = Fernet(fernet_key)
        ciphertext = base64.b64decode(encrypted_data.ciphertext)
        
        return f.decrypt(ciphertext)
    
    def _encrypt_aes_gcm(self, data: bytes, key: bytes, salt: bytes) -> EncryptedData:
        """Encrypt using AES-256-GCM"""
        # Derive key
        derived_key = hashlib.sha256(key + salt).digest()
        
        # Generate nonce
        nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM
        
        # Encrypt
        cipher = Cipher(
            algorithms.AES(derived_key),
            modes.GCM(nonce),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        # Combine ciphertext and tag
        ciphertext_with_tag = ciphertext + encryptor.tag
        
        return EncryptedData(
            ciphertext=base64.b64encode(ciphertext_with_tag).decode('ascii'),
            algorithm=EncryptionAlgorithm.AES_256_GCM.value,
            salt=base64.b64encode(salt).decode('ascii'),
            nonce=base64.b64encode(nonce).decode('ascii')
        )
    
    def _decrypt_aes_gcm(self, encrypted_data: EncryptedData, key: bytes, salt: bytes) -> bytes:
        """Decrypt using AES-256-GCM"""
        # Derive key
        derived_key = hashlib.sha256(key + salt).digest()
        
        # Extract nonce, ciphertext, and tag
        nonce = base64.b64decode(encrypted_data.nonce)
        ciphertext_with_tag = base64.b64decode(encrypted_data.ciphertext)
        ciphertext = ciphertext_with_tag[:-16]  # All but last 16 bytes
        tag = ciphertext_with_tag[-16:]  # Last 16 bytes
        
        # Decrypt
        cipher = Cipher(
            algorithms.AES(derived_key),
            modes.GCM(nonce, tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        
        return decryptor.update(ciphertext) + decryptor.finalize()
    
    def _encrypt_chacha20(self, data: bytes, key: bytes, salt: bytes) -> EncryptedData:
        """Encrypt using ChaCha20-Poly1305"""
        # Derive key
        derived_key = hashlib.sha256(key + salt).digest()
        
        # Generate nonce
        nonce = secrets.token_bytes(12)  # 96-bit nonce
        
        # Encrypt
        cipher = Cipher(
            algorithms.ChaCha20(derived_key, nonce),
            mode=None,
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        return EncryptedData(
            ciphertext=base64.b64encode(ciphertext).decode('ascii'),
            algorithm=EncryptionAlgorithm.CHACHA20_POLY1305.value,
            salt=base64.b64encode(salt).decode('ascii'),
            nonce=base64.b64encode(nonce).decode('ascii')
        )
    
    def _decrypt_chacha20(self, encrypted_data: EncryptedData, key: bytes, salt: bytes) -> bytes:
        """Decrypt using ChaCha20-Poly1305"""
        # Derive key
        derived_key = hashlib.sha256(key + salt).digest()
        
        # Extract nonce and ciphertext
        nonce = base64.b64decode(encrypted_data.nonce)
        ciphertext = base64.b64decode(encrypted_data.ciphertext)
        
        # Decrypt
        cipher = Cipher(
            algorithms.ChaCha20(derived_key, nonce),
            mode=None,
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        
        return decryptor.update(ciphertext) + decryptor.finalize()
    
    def encrypt_field(self, data: Dict[str, Any], field_name: str) -> Dict[str, Any]:
        """
        Encrypt a specific field in a dictionary.
        
        Args:
            data: Dictionary containing the field
            field_name: Name of field to encrypt
            
        Returns:
            Dictionary with encrypted field
        """
        if field_name not in data:
            return data
        
        result = data.copy()
        field_value = result[field_name]
        
        # Encrypt the field value
        encrypted_value = self.encrypt(field_value)
        result[field_name] = encrypted_value
        
        # Mark field as encrypted
        result[f"{field_name}_encrypted"] = True
        
        return result
    
    def decrypt_field(self, data: Dict[str, Any], field_name: str) -> Dict[str, Any]:
        """
        Decrypt a specific field in a dictionary.
        
        Args:
            data: Dictionary containing the encrypted field
            field_name: Name of field to decrypt
            
        Returns:
            Dictionary with decrypted field
        """
        if field_name not in data or not data.get(f"{field_name}_encrypted", False):
            return data
        
        result = data.copy()
        encrypted_value = result[field_name]
        
        # Decrypt the field value
        decrypted_value = self.decrypt(encrypted_value)
        result[field_name] = decrypted_value
        
        # Remove encryption marker
        result.pop(f"{field_name}_encrypted", None)
        
        return result
    
    def rotate_keys(self) -> bool:
        """
        Rotate encryption keys.
        
        Returns:
            True if rotation successful
        """
        try:
            with self._lock:
                # Generate new key
                new_version = self._current_key_version + 1
                new_key = self._generate_key()
                
                # Store new key
                self._keys[new_version] = new_key
                self._current_key_version = new_version
                
            logger.info(f"Keys rotated to version {new_version}")
            return True
            
        except Exception as e:
            logger.error(f"Key rotation failed: {e}")
            return False
    
    def export_master_key(self) -> str:
        """Export master key as base64 string (for backup/transfer)"""
        if not self._master_key:
            raise CrewGraphError("No master key available")
        
        return base64.b64encode(self._master_key).decode('ascii')
    
    def save_key_to_file(self, file_path: str) -> bool:
        """
        Save master key to file.
        
        Args:
            file_path: Path to save key file
            
        Returns:
            True if save successful
        """
        try:
            with open(file_path, 'wb') as f:
                f.write(self._master_key)
            
            # Set restrictive permissions
            os.chmod(file_path, 0o600)
            
            logger.info(f"Master key saved to: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save key file: {e}")
            return False
    
    def get_encryption_info(self) -> Dict[str, Any]:
        """Get encryption manager information"""
        with self._lock:
            return {
                "algorithm": self.config.algorithm.value,
                "key_derivation": self.config.key_derivation.value,
                "key_size": self.config.key_size,
                "current_key_version": self._current_key_version,
                "total_key_versions": len(self._keys),
                "compression_enabled": self.config.enable_compression,
                "key_rotation_enabled": self.config.key_rotation_enabled
            }
    
    def test_encryption(self) -> bool:
        """Test encryption/decryption functionality"""
        try:
            test_data = {"test": "data", "number": 42, "list": [1, 2, 3]}
            
            # Encrypt
            encrypted = self.encrypt(test_data)
            
            # Decrypt
            decrypted = self.decrypt(encrypted)
            
            # Verify
            return decrypted == test_data
            
        except Exception as e:
            logger.error(f"Encryption test failed: {e}")
            return False