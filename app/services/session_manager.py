from typing import Optional, Dict, Any
import asyncio
import json
import time
import uuid
import os
import pickle
from app.core.config import get_settings
from app.core.logger import get_logger
from app.services.parameter_extractor import ParameterExtractor
from app.services.prompt_manager import PromptManager

settings = get_settings()
logger = get_logger()

class SessionManagerService:
    """
    Service for managing user sessions using in-memory storage with disk persistence.
    
    This service handles:
    - Creating and retrieving PromptManager instances by session ID
    - Managing session expiration and cleanup
    - Persisting session state in memory and on disk
    """
    
    # Class-level variable to avoid multiple instances losing sessions
    _instance = None
    _initialized = False
    _sessions_dir = "sessions"
    
    def __new__(cls, parameter_extractor: ParameterExtractor):
        """Singleton pattern to ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super(SessionManagerService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, parameter_extractor: ParameterExtractor):
        """Initialize the session manager."""
        # Skip initialization if already done
        if SessionManagerService._initialized:
            return
            
        self.parameter_extractor = parameter_extractor
        
        # In-memory store for sessions
        self.memory_sessions = {}
        
        # Set session TTL (1 hour default)
        self.session_ttl = settings.CACHE_TTL
        
        # Ensure sessions directory exists
        os.makedirs(self._sessions_dir, exist_ok=True)
        
        # Load existing sessions from disk
        self._load_sessions_from_disk()
        
        SessionManagerService._initialized = True
        logger.info(f"SessionManagerService initialized with disk persistence (TTL: {self.session_ttl}s)")
    
    # Rate limiting for session creation
    _last_sessions = {}  # IP -> [timestamp, timestamp, ...]
    _max_sessions_per_minute = 10  # Maximum sessions per minute per IP
    
    async def create_session(self, ip_address="unknown") -> str:
        """
        Create a new session with a unique ID.
        
        Args:
            ip_address: The client's IP address for rate limiting
            
        Returns:
            str: The new session ID
            
        Raises:
            Exception: If rate limit is exceeded
        """
        # Apply rate limiting - keep only the last minute of sessions
        now = time.time()
        if ip_address in self._last_sessions:
            # Remove entries older than 60 seconds
            self._last_sessions[ip_address] = [
                ts for ts in self._last_sessions[ip_address] 
                if now - ts < 60
            ]
            
            # Check if rate limit exceeded
            if len(self._last_sessions[ip_address]) >= self._max_sessions_per_minute:
                logger.warning(f"Rate limit exceeded for IP {ip_address}: {len(self._last_sessions[ip_address])} sessions in the last minute")
                raise Exception("Rate limit exceeded. Please try again later.")
                
            # Add this session
            self._last_sessions[ip_address].append(now)
        else:
            # First session for this IP
            self._last_sessions[ip_address] = [now]
        
        # Create the session
        session_id = str(uuid.uuid4())
        
        # Create a new PromptManager for this session
        prompt_manager = PromptManager(self.parameter_extractor, session_id=session_id)
        
        # Store in memory cache
        self.memory_sessions[session_id] = prompt_manager
        
        logger.info(f"Created new session {session_id} for IP {ip_address}")
        return session_id
        
    async def get_prompt_manager(self, session_id: str) -> PromptManager:
        """
        Get the PromptManager for a session.
        
        Args:
            session_id: The session ID
            
        Returns:
            PromptManager: The prompt manager for this session
            
        If the session doesn't exist, creates a new one with the given ID.
        """
        # Check if session exists in memory
        if session_id in self.memory_sessions:
            prompt_manager = self.memory_sessions[session_id]
            logger.info(f"Retrieved session {session_id} from memory cache")
            prompt_manager.touch()
            await self._save_prompt_manager(session_id, prompt_manager)  # Update last_accessed on disk
            return prompt_manager
        else:
            # Check if session exists on disk
            file_path = self._get_session_path(session_id)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        prompt_manager = pickle.load(f)
                    
                    # Reattach the parameter_extractor
                    prompt_manager.parameter_extractor = self.parameter_extractor
                    
                    # Check if session is expired
                    now = time.time()
                    if now - prompt_manager.last_accessed > self.session_ttl:
                        # Session is expired, create a new one
                        logger.info(f"Session {session_id} found on disk but expired, creating new prompt manager")
                        prompt_manager = PromptManager(self.parameter_extractor, session_id=session_id)
                    else:
                        logger.info(f"Retrieved session {session_id} from disk")
                    
                    # Update timestamp and cache in memory
                    prompt_manager.touch()
                    self.memory_sessions[session_id] = prompt_manager
                    await self._save_prompt_manager(session_id, prompt_manager)
                    return prompt_manager
                except Exception as e:
                    logger.error(f"Error loading session {session_id} from disk: {str(e)}")
            
            # Create a new session if not in memory or on disk
            logger.info(f"Session {session_id} not found, creating new prompt manager")
            prompt_manager = PromptManager(self.parameter_extractor, session_id=session_id)
            
            # Cache in memory and persist to disk
            self.memory_sessions[session_id] = prompt_manager
            await self._save_prompt_manager(session_id, prompt_manager)
            
            return prompt_manager
    
    def _get_session_path(self, session_id: str) -> str:
        """Get the file path for a session."""
        return os.path.join(self._sessions_dir, f"{session_id}.pickle")
        
    def _load_sessions_from_disk(self):
        """Load existing sessions from disk."""
        try:
            session_files = [f for f in os.listdir(self._sessions_dir) if f.endswith('.pickle')]
            count = 0
            
            for filename in session_files:
                session_id = filename.replace('.pickle', '')
                file_path = os.path.join(self._sessions_dir, filename)
                
                try:
                    with open(file_path, 'rb') as f:
                        prompt_manager = pickle.load(f)
                    
                    # Reattach the parameter_extractor
                    prompt_manager.parameter_extractor = self.parameter_extractor
                        
                    # Only load if not expired
                    now = time.time()
                    if now - prompt_manager.last_accessed <= self.session_ttl:
                        self.memory_sessions[session_id] = prompt_manager
                        count += 1
                except Exception as e:
                    logger.error(f"Error loading session {session_id} from disk: {str(e)}")
                    
            if count > 0:
                logger.info(f"Loaded {count} active sessions from disk")
        except Exception as e:
            logger.error(f"Error loading sessions from disk: {str(e)}")
    
    async def _save_prompt_manager(self, session_id: str, prompt_manager: PromptManager) -> bool:
        """
        Save a prompt manager to memory and disk.
        
        Args:
            session_id: The session ID
            prompt_manager: The prompt manager to save
            
        Returns:
            bool: True if saved successfully
        """
        # Store in memory
        self.memory_sessions[session_id] = prompt_manager
        
        # Store on disk
        try:
            file_path = self._get_session_path(session_id)
            with open(file_path, 'wb') as f:
                pickle.dump(prompt_manager, f)
            logger.debug(f"Saved session {session_id} to memory and disk")
            return True
        except Exception as e:
            logger.error(f"Error saving session {session_id} to disk: {str(e)}")
            return False
            
    async def cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions from memory and disk.
        
        Returns:
            int: Number of sessions removed
        """
        removed_count = 0
        
        # Clean up memory sessions
        memory_removed = 0
        
        # If the memory cache gets too large, remove the oldest sessions from memory only
        # (they'll still be on disk if needed later)
        MAX_MEMORY_SESSIONS = 100  # Reasonable limit for a small deployment
        if len(self.memory_sessions) > MAX_MEMORY_SESSIONS:
            # Calculate which sessions to keep based on last_accessed
            sessions_with_time = [
                (sid, pm.last_accessed) 
                for sid, pm in self.memory_sessions.items()
            ]
            # Sort by last_accessed (most recent first)
            sessions_with_time.sort(key=lambda x: x[1], reverse=True)
            # Keep only the MAX_MEMORY_SESSIONS most recent
            sessions_to_keep = sessions_with_time[:MAX_MEMORY_SESSIONS]
            sessions_to_keep_ids = {sid for sid, _ in sessions_to_keep}
            
            # Remove older sessions from memory only
            for sid in list(self.memory_sessions.keys()):
                if sid not in sessions_to_keep_ids:
                    del self.memory_sessions[sid]
                    memory_removed += 1
        
        # Remove expired sessions from both memory and disk
        now = time.time()
        
        # Check memory sessions
        memory_expired = []
        for sid, pm in self.memory_sessions.items():
            if now - pm.last_accessed > self.session_ttl:
                memory_expired.append(sid)
                
        for sid in memory_expired:
            del self.memory_sessions[sid]
            # Also remove from disk
            file_path = self._get_session_path(sid)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.error(f"Error removing session file {file_path}: {str(e)}")
            removed_count += 1
        
        # Check disk sessions that might not be in memory
        try:
            for filename in os.listdir(self._sessions_dir):
                if not filename.endswith('.pickle'):
                    continue
                    
                session_id = filename.replace('.pickle', '')
                
                # Skip if already handled in memory cleanup
                if session_id in memory_expired:
                    continue
                    
                # Skip if active in memory
                if session_id in self.memory_sessions:
                    continue
                    
                file_path = os.path.join(self._sessions_dir, filename)
                
                try:
                    # Check if file is expired
                    with open(file_path, 'rb') as f:
                        prompt_manager = pickle.load(f)
                    
                    # We don't need to reattach parameter_extractor here since we're just checking expiration
                    if now - prompt_manager.last_accessed > self.session_ttl:
                        os.remove(file_path)
                        removed_count += 1
                except Exception as e:
                    # If file is corrupted, remove it
                    logger.error(f"Error checking session file {file_path}: {str(e)}")
                    try:
                        os.remove(file_path)
                        removed_count += 1
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"Error cleaning up session files: {str(e)}")
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} expired sessions")
        if memory_removed > 0:
            logger.info(f"Removed {memory_removed} sessions from memory cache (still on disk)")
            
        return removed_count + memory_removed
            
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from memory and disk.
        
        Args:
            session_id: The session ID to delete
            
        Returns:
            bool: True if session was deleted, False otherwise
        """
        deleted = False
        
        # Delete from memory if present
        if session_id in self.memory_sessions:
            del self.memory_sessions[session_id]
            deleted = True
        
        # Delete from disk if present
        file_path = self._get_session_path(session_id)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                deleted = True
            except Exception as e:
                logger.error(f"Error removing session file {file_path}: {str(e)}")
        
        if deleted:
            logger.info(f"Deleted session {session_id}")
            return True
        else:
            logger.info(f"Session {session_id} not found for deletion")
            return False
            
    async def list_active_sessions(self) -> Dict[str, float]:
        """
        List all active sessions with their last access time.
        Includes sessions from both memory and disk.
        
        Returns:
            Dict[str, float]: Dictionary mapping session IDs to last access timestamps
        """
        # Start with memory-cached sessions
        results = {}
        for sid, pm in self.memory_sessions.items():
            results[sid] = pm.last_accessed
        
        # Add sessions from disk that aren't in memory
        try:
            for filename in os.listdir(self._sessions_dir):
                if not filename.endswith('.pickle'):
                    continue
                    
                session_id = filename.replace('.pickle', '')
                
                # Skip if already in results
                if session_id in results:
                    continue
                    
                file_path = os.path.join(self._sessions_dir, filename)
                
                try:
                    with open(file_path, 'rb') as f:
                        prompt_manager = pickle.load(f)
                    
                    # Add to results
                    results[session_id] = prompt_manager.last_accessed
                except Exception as e:
                    logger.error(f"Error reading session file {file_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Error listing session files: {str(e)}")
            
        return results